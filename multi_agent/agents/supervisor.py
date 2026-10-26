from __future__ import annotations

import logging
import operator
import os
from pathlib import Path
from datetime import datetime
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, RunnableConfig

from langgraph_agent.multi_agent.agents.researcher import graph as research_agent
from langgraph_agent.multi_agent.agents.copywriter import graph as copywriter_agent

from langgraph_agent.config import Settings

load_dotenv()
logger = logging.getLogger(__name__)


def _extract_usage(response) -> dict:
    """Return token usage and whether the response exposed usable usage metadata."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict) or not any(
        key in usage for key in ("input_tokens", "output_tokens", "total_tokens")
    ):
        response_metadata = getattr(response, "response_metadata", None) or {}
        if isinstance(response_metadata, dict):
            usage = response_metadata.get("token_usage") or response_metadata.get("usage_metadata")

    if isinstance(usage, dict) and any(
        key in usage for key in ("input_tokens", "output_tokens", "total_tokens")
    ):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(total_tokens or 0),
            "available": True,
        }

    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "available": False,
    }

MULTI_AGENT_DIR = Path(__file__).resolve().parents[1]
supervisor_prompt = (MULTI_AGENT_DIR / "prompts" / "supervisor.md").read_text(encoding="utf-8")

settings = Settings()
api_key = settings.gemini_api_key.get_secret_value().strip()
if not api_key:
    raise RuntimeError("Missing Google gemini key from environmental variables")

model = settings.gemini_model.strip()
model = ChatGoogleGenerativeAI(model= model, api_key= api_key)

MAX_HANDOFFS = settings.MAX_AGENT_HANDOFFS
TOTAL_TOKEN_BUDGET = settings.TOTAL_TOKEN_BUDGET

class SupervisorState(BaseModel):
    """Shared state for supervisor orchestration."""
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    research_reports: Annotated[list, operator.add] = Field(default_factory=list)
    task_description: str | None = None
    handoff_count: int = 0
    last_agent_status: Literal["not_started", "running", "completed", "failed"] = "not_started"
    last_agent_error: str | None = None
    output_paths: Annotated[list[str], operator.add] = Field(default_factory=list)
    total_tokens_used: int = 0          # supervisor + sub-agent LLM calls
    token_usage_uncertain: bool = False
    budget_exceeded: bool = False   
    budget_exceeded_reason: str | None = None   

@tool
async def handoff_to_subagent(
    agent_name: Literal["researcher", "copywriter"],
    task_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[SupervisorState, InjectedState],
) -> Command:
    """
    Delegate a task to the researcher or copywriter, subject to a handoff budget.
    
    Args:
        agent_name: name of the agent to invoke
        task_description: task description which will be given to the agent
    """
    
    handoff_count = state.get("handoff_count", 0) if isinstance(state, dict) else state.handoff_count
    total_tokens_used = state.get("total_tokens_used", 0) if isinstance(state, dict) else state.total_tokens_used
    token_usage_uncertain = state.get("token_usage_uncertain", False) if isinstance(state, dict) else state.token_usage_uncertain

    if token_usage_uncertain:
        return Command(goto="supervisor", update={
            "budget_exceeded": True,
            "budget_exceeded_reason": (
                "Token usage could not be verified for a previous LLM call. "
                "Further delegation is stopped to avoid an uncontrolled total token budget."
            ),
            "messages": [ToolMessage(
                name="handoff_to_subagent",
                content=(
                    "Handoff rejected because token usage could not be verified. "
                    "This is a budget-safety stop, not an agent failure. Summarize the results already available."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if handoff_count >= MAX_HANDOFFS:
        return Command(goto="supervisor", update={
            "budget_exceeded": True,
            "budget_exceeded_reason": f"Handoff budget ({MAX_HANDOFFS}) reached.",
            # last_agent_status is intentionally left untouched — no agent ran this call
            "messages": [ToolMessage(
                name="handoff_to_subagent",
                content=(
                    f"Handoff rejected: maximum handoff budget ({MAX_HANDOFFS}) reached. "
                    "This is an orchestration limit, not a failure of the researcher or copywriter — "
                    "their prior results (if any) are still valid. Summarize what has been accomplished "
                    "so far and do not attempt further delegation."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if total_tokens_used >= TOTAL_TOKEN_BUDGET:
        return Command(goto="supervisor", update={
            "budget_exceeded": True,
            "budget_exceeded_reason": f"Total token budget ({TOTAL_TOKEN_BUDGET}) reached.",
            "messages": [ToolMessage(
                name="handoff_to_subagent",
                content=(
                    f"Handoff rejected: total token budget ({TOTAL_TOKEN_BUDGET}) reached "
                    f"({total_tokens_used} used so far). This is an orchestration limit, not an agent "
                    "failure. Summarize what has been accomplished so far and do not delegate again."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    return Command(
        goto=f"call_{agent_name}",
        update={
            "task_description": task_description,
            "handoff_count": handoff_count + 1,
            "last_agent_status": "running",
            "last_agent_error": None,
            "messages": [ToolMessage(
                name=f"handoff_to_{agent_name}",
                content=f"Successfully handed off to {agent_name}.",
                tool_call_id=tool_call_id,
            )],
        },
    )


async def call_researcher(state: SupervisorState, config: RunnableConfig | None = None):
    """Run the researcher and pass its structured result back to the supervisor."""
    try:
        result = await research_agent.ainvoke(
            {"messages": [HumanMessage(content=state.task_description or "Research the requested topic.")]},
            config={**(config or {}), "recursion_limit": 40},
        )
        reports = result.get("research_reports", [])
        status = result.get("status", "failed")
        error = result.get("error")
        run_tokens = result.get("tokens_used", 0)
        run_usage_uncertain = result.get("token_usage_uncertain", False)
        new_total = state.total_tokens_used + run_tokens

        if status != "completed" or not reports:
            status = "failed"
            error = error or "Researcher did not return a completed research report."
            content = f"Researcher status: failed. Error: {error}\nTokens used this run: {run_tokens} (total so far: {new_total}/{TOTAL_TOKEN_BUDGET})"
        else:
            content = ("Research completed. Reports:\n" + "\n\n".join(
            report.model_dump_json() if hasattr(report, "model_dump_json") else str(report)
            for report in reports
            )
            + f"\n\nTokens used this run: {run_tokens} (total so far: {new_total}/{TOTAL_TOKEN_BUDGET})"
            )
        return {
            "research_reports": reports if status == "completed" else [],
            "messages": [AIMessage(name="researcher", content=content)],
            "last_agent_status": status,
            "last_agent_error": error,
            "total_tokens_used": new_total,
            "token_usage_uncertain": state.token_usage_uncertain or run_usage_uncertain,
        }
    except Exception as exc:
        logger.exception("Researcher subagent failed")
        return {
            "messages": [AIMessage(name="researcher", content=f"Researcher failed: {type(exc).__name__}: {exc}")],
            "last_agent_status": "failed",
            "last_agent_error": f"{type(exc).__name__}: {exc}",
            "token_usage_uncertain": True,
        }


async def call_copywriter(state: SupervisorState, config: RunnableConfig | None = None):
    """Run the copywriter and propagate its explicit status and output paths."""
    try:
        result = await copywriter_agent.ainvoke(
            {
                "messages": [HumanMessage(content=state.task_description or "Create the requested content.")],
                "research_reports": state.research_reports,
            },
            config={**(config or {}), "recursion_limit": 40},
        )
        status = result.get("status", "failed")
        output_paths = result.get("output_paths", [])
        error = result.get("error")
        run_tokens = result.get("tokens_used", 0)
        run_usage_uncertain = result.get("token_usage_uncertain", False)
        new_total = state.total_tokens_used + run_tokens

        if status != "completed" or not output_paths:
            status = "failed"
            error = error or "Copywriter did not save an output file."
            content = f"Copywriter status: failed. Error: {error}\nTokens used this run: {run_tokens} (total so far: {new_total}/{TOTAL_TOKEN_BUDGET})"
        else:
            content = ("Copywriter completed successfully. Saved files:\n" 
            + "\n".join(output_paths) 
            + f"\n\nTokens used this run: {run_tokens} (total so far: {new_total}/{TOTAL_TOKEN_BUDGET})"
            )
        return {
            "messages": [AIMessage(name="copywriter", content=content)],
            "last_agent_status": status,
            "last_agent_error": error,
            "output_paths": output_paths if status == "completed" else [],
            "total_tokens_used": new_total,
            "token_usage_uncertain": state.token_usage_uncertain or run_usage_uncertain,
        }
    except Exception as exc:
        logger.exception("Copywriter subagent failed")
        return {
            "messages": [AIMessage(name="copywriter", content=f"Copywriter failed: {type(exc).__name__}: {exc}")],
            "last_agent_status": "failed",
            "last_agent_error": f"{type(exc).__name__}: {exc}",
            "token_usage_uncertain": True,
        }


tools = [handoff_to_subagent]
llm_with_tools = model.bind_tools(tools, parallel_tool_calls=False)


async def supervisor(state: SupervisorState):
    """Run supervisor reasoning and account for its own model usage."""
    handoff_count = state.get("handoff_count", 0) if isinstance(state, dict) else state.handoff_count
    total_tokens_used = state.get("total_tokens_used", 0) if isinstance(state, dict) else state.total_tokens_used
    budget_exceeded = state.get("budget_exceeded", False) if isinstance(state, dict) else state.budget_exceeded
    token_usage_uncertain = state.get("token_usage_uncertain", False) if isinstance(state, dict) else state.token_usage_uncertain

    hit_limit = (
        handoff_count >= MAX_HANDOFFS
        or total_tokens_used >= TOTAL_TOKEN_BUDGET
        or budget_exceeded
        or token_usage_uncertain
    )

    if hit_limit:
        if isinstance(state, dict):
            reason = state.get("budget_exceeded_reason")
        else:
            reason = state.budget_exceeded_reason
        if not reason and token_usage_uncertain:
            reason = (
                "Token usage could not be verified for one or more LLM calls. "
                "Further delegation is stopped to protect the total token budget."
            )
        reason = reason or "A delegation or total token budget limit was reached."
        wrap_up_prompt = SystemMessage(content=(
            supervisor_prompt.format(current_datetime=datetime.now())
            + f"\n\nIMPORTANT: {reason} You may NOT call any more tools. "
            "Using only the research reports and results already present in the conversation, "
            "write the best possible final answer for the user. Clearly state any part of their "
            "request you were unable to complete because of this limit."
        ))
        # The wrap-up call is intentionally allowed so the user receives a final status.
        response = await model.ainvoke([wrap_up_prompt] + state.messages)
    else:
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=supervisor_prompt.format(current_datetime=datetime.now()))
        ] + state.messages)

    usage = _extract_usage(response)
    logger.info("Supervisor response token usage: %s", usage)
    return {
        "messages": [response],
        "total_tokens_used": total_tokens_used + usage["total_tokens"],
        "token_usage_uncertain": token_usage_uncertain or not usage["available"],
    }


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("supervisor_toolnode", ToolNode(tools, handle_tool_errors=True))
builder.add_node("call_researcher", call_researcher)
builder.add_node("call_copywriter", call_copywriter)
builder.set_entry_point("supervisor")
builder.add_conditional_edges("supervisor", tools_condition, {"tools": "supervisor_toolnode", END: END})

builder.add_edge("call_researcher", "supervisor")
builder.add_edge("call_copywriter", "supervisor")
graph = builder.compile(checkpointer=MemorySaver())

print(graph.get_graph().edges)