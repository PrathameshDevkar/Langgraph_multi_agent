from __future__ import annotations

import asyncio
import operator
import re
import datetime
from pathlib import Path
from typing import Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.types import Command

from langgraph_agent.config import Settings

load_dotenv()

MULTI_AGENT_DIR = Path(__file__).resolve().parents[1]
copywriter_prompt = (MULTI_AGENT_DIR/ "prompts" / "copywriter.md").read_text(encoding= "utf-8")
linkedin_example = (MULTI_AGENT_DIR / "example_content" / "linkedin.md").read_text(encoding = "utf-8")
blog_example = (MULTI_AGENT_DIR / "example_content" / "blog.md").read_text(encoding = "utf-8")
OUTPUT_DIR = Path.cwd() / "ai_files"

settings = Settings()
api_key = settings.gemini_api_key.get_secret_value().strip()
if not api_key:
    raise RuntimeError("Missing Google gemini key from environmental variables")

model = settings.gemini_model.strip()
model = ChatGoogleGenerativeAI(model= model, api_key= api_key)

# -------Loop / budget controls----------
MAX_STEPS = settings.COPYWRITER_MAX_STEPS
MAX_TOOL_CALLS = settings.COPYWRITER_MAX_TOOL_CALLS
TOKEN_BUDGET = settings.COPYWRITER_TOKEN_BUDGET

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

class CopyWriterState(BaseModel):
    "State for one copywriter run"
    messages: Annotated[list, add_messages] = Field(default_factory = list)
    research_reports:Annotated[list,operator.add] = Field(default_factory = list)
    status: Literal["running", "completed", "failed"] = "running"
    output_paths: Annotated[list[str], operator.add] = Field(default_factory = list)
    error:str | None = None
    step_count: int = 0
    tool_call_count: int = 0
    tokens_used: int = 0
    token_usage_uncertain: bool = False

@tool
async def review_research_reports(state:Annotated[CopyWriterState, InjectedState]):
    """Returns available research reports for grounding the content."""
    reports = []
    research_reports = state.get("research_reports", []) if isinstance(state, dict) else state.research_reports
    for report in research_reports:
        if hasattr(report, "model_dump_json"):
            reports.append(report.model_dump_json())
        else:
            reports.append(str(report))

    return reports

def _write_markdown(title: str, content: str, content_type:str) -> str:
    """Write Markdown safely, avoiding path traversal and accidental overwrites."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title.strip()).strip("._-")
    if not safe_stem:
        safe_stem = content_type
    safe_stem = safe_stem[:100]
    candidate = OUTPUT_DIR / f"{safe_stem}.md"
    suffix = 2
    while candidate.exists():
        candidate = OUTPUT_DIR / f"{safe_stem}_{suffix}.md"
        suffix += 1
    candidate.write_text(content, encoding="utf-8")
    return str(candidate.resolve())

async def _save_post(title: str, content: str, content_type: str, tool_call_id: str) -> Command:
    path = await asyncio.to_thread(_write_markdown, title, content, content_type)
    return Command(update={
        "output_paths":[path],
        "status":"completed",
        "error":None,
        "messages":[ToolMessage(
            name = f"generate_{content_type}_post",
            content = f"Saved{content_type} content to {path}",
            tool_call_id = tool_call_id
        )]
    })

@tool
async def generate_linkedin_post(
    title: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Save linkedin post as markdown file in ai_files/.

    Args: 
        title: title of the post
        content: content to add in the post
    """
    return await _save_post(title, content, "linkedin", tool_call_id)

@tool
async def generate_blog_post(
    title: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Save blog post as Markdown file in ai_files./

    Args:   
        title: title for the blog post
        content: content to add in the blog post
    """
    return await _save_post(title, content, "blog", tool_call_id)

tools = [review_research_reports, generate_linkedin_post, generate_blog_post]
llm_with_tools = model.bind_tools(tools, parallel_tool_calls = False)

async def copywriter(state: CopyWriterState):
    "Run one copywriter reasoning step, subject to step/tool-call/token budgets."
    # A saved output means the task is already complete. Avoid another model
    # call and prevent a subsequent budget check from invalidating success.
    if state.status == "completed" and state.output_paths:
        return {"status": "completed", "error": None}

    step_count = state.step_count + 1

    if step_count > MAX_STEPS:
        return {
            "status": "failed",
            "error": f"Copywriter exceeded max reasoning steps ({MAX_STEPS}) without saving output.",
            "step_count": step_count,
        }
    if state.tokens_used >= TOKEN_BUDGET:
        return {
            "status": "failed",
            "error": f"Copywriter exceeded its token budget ({TOKEN_BUDGET} tokens).",
            "step_count": step_count,
        }
    if state.tool_call_count >= MAX_TOOL_CALLS:
        return {
            "status": "failed",
            "error": f"Copywriter exceeded max tool calls ({MAX_TOOL_CALLS}) without saving output.",
            "step_count": step_count,
        }

    system_prompt = SystemMessage(content=copywriter_prompt.format(
        current_datetime = datetime.now(),
        linkedin_example = linkedin_example,
        blog_example = blog_example
    ))
    response = await llm_with_tools.ainvoke([system_prompt] + state.messages)

    usage = _extract_usage(response)
    new_tool_calls = len(getattr(response, "tool_calls", None) or [])

    update= {
        "messages":[response],
        "step_count": step_count,
        "tool_call_count": state.tool_call_count + new_tool_calls,
        "tokens_used": state.tokens_used + usage["total_tokens"],
        "token_usage_uncertain": state.token_usage_uncertain or not usage["available"],
    }

    if not getattr(response, "tool_calls", None) and not state.output_paths:
        update["status"] = "failed"
        update["error"] = "Copywriter stopped without saving an output file"

    return update

builder = StateGraph(CopyWriterState)
builder.add_node("copywriter", copywriter)
builder.add_node("copywriter_toolnode", ToolNode(tools, handle_tool_errors = True))
builder.set_entry_point("copywriter")
builder.add_conditional_edges(
    "copywriter", tools_condition,{
        "tools": "copywriter_toolnode",
        END:END
    }
)
builder.add_edge("copywriter_toolnode", "copywriter")
graph = builder.compile()