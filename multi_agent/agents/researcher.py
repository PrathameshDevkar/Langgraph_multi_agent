from __future__ import annotations

import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch, TavilyExtract
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from langgraph_agent.config import Settings

load_dotenv()

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "researcher.md"
researcher_prompt= PROMPT_PATH.read_text(encoding= "utf-8")

settings = Settings()
api_key = settings.gemini_api_key.get_secret_value().strip()
if not api_key:
    raise RuntimeError("Missing Google gemini key from environmental variables")

model = settings.gemini_model.strip()
model = ChatGoogleGenerativeAI(model= model, api_key= api_key)

# Loop and budget control
MAX_STEPS = settings.RESEARCHER_MAX_STEPS
MAX_TOOL_CALLS = settings.RESEARCHER_MAX_TOOL_CALLS
TOKEN_BUDGET = settings.RESEARCHER_TOKEN_BUDGET

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

class ResearchReport(BaseModel):
    topic: str
    report: str

class ResearcherState(BaseModel):
    "State for one researcher run"
    messages: Annotated[list, add_messages]= Field(default_factory = list)
    research_reports: Annotated[list[ResearchReport], operator.add] = Field(default_factory = list)
    status: Literal["completed", "failed", "running"] = "running"
    error: str | None = None
    step_count : int = 0
    tool_call_count: int= 0
    tokens_used: int = 0
    token_usage_uncertain: bool = False

@tool 
async def search_web(query: str, num_results: int = 3) -> dict:
    """Search the web and return up to three results with title, url and content preview
    
    Args:
        query: query to search on web
        num_results: number of results to return, max= 3
    
    returns:
        dictionary of search result
    """
    bounded_num_results = max(1, min(num_results, 3))
    web_search = TavilySearch(max_results = bounded_num_results, topic= "general")
    search_results = await web_search.ainvoke({"query": query})

    processed_results = {"query":query, "results":[]}
    for result in search_results.get("results",[]):
        processed_results["results"].append({
            "title":result.get("title",""),
            "url":result.get("url",""),
            "content_preview":result.get("content","")
        })
    return processed_results

@tool
async def extract_content_from_webpage(urls:list[str]) -> list[dict]:
    """
    Extract content from the supplied webpage URLS

    Args:
        urls: list of urls from which the content will eb extracted

    Returns:
        list of extracted content from the urls
    """
    web_extract= TavilyExtract()
    result = await web_extract.ainvoke({"urls":urls})
    return result.get("results", result.get("results",[])) if isinstance(result,dict) else result

@tool
async def generate_research_report(
    topic:str,
    report:str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    validate and store the final research report for this run.
    
    Args:
        topic: topic of the resaerch report
        report: content of the research report
    
    """
    research_report = ResearchReport(
        topic = topic,
        report = report
    )

    return Command(update={
        "research_reports": [research_report],
        "status":"completed",
        "error": None,
        "messages": [ToolMessage(
            name="generate_research_report",
            content= research_report.model_dump_json(),
            tool_call_id = tool_call_id,
        )]
    })

tools=[search_web, extract_content_from_webpage, generate_research_report]

llm_with_tools = model.bind_tools(tools)

async def researcher(state: ResearcherState):
    "Run one researcher reasoning step, subject to step/tool-call/token budgets."
    # The report tool has already completed the task. Do not make another model
    # call or let a later budget check overwrite a successful result.
    if state.status == "completed" and state.research_reports:
        return {"status": "completed", "error": None}

    step_count = state.step_count + 1

    if step_count> MAX_STEPS:
        return {
            "status": "failed",
            "error": f"Researcher exceeded max reasoning steps {MAX_STEPS} without producing report",
            "step_count": step_count
        }

    if state.tokens_used >= TOKEN_BUDGET:
        return {
            "status":"failed",
            "error": f"Researcher exceeded its token budget ({TOKEN_BUDGET}) tokens.",
            "step_count": step_count
        }
    
    if state.tool_call_count >= MAX_TOOL_CALLS:
        return{
            "status":"failed",
            "error": f"Researcher exceeded max tool calls({MAX_TOOL_CALLS}) without producing a report.",
            "step_count": step_count
        }
    
    response = await llm_with_tools.ainvoke([
        SystemMessage(content = researcher_prompt.format(current_datetime = datetime.now()))
    ] + state.messages)

    usage = _extract_usage(response)
    new_tool_calls = len(getattr(response, "tool_calls", None) or [])

    update = {
        "messages":[response],
        "step_count": step_count,
        "tool_call_count": state.tool_call_count + new_tool_calls,
        "tokens_used": state.tokens_used + usage["total_tokens"],
        "token_usage_uncertain": state.token_usage_uncertain or not usage["available"],
    }

    if not getattr(response, "tool_calls", None) and state.status != "completed":
        update["status"] = "failed"
        update["error"] = "Researcher stopped without generating a valid research report"

    return update

builder = StateGraph(ResearcherState)
builder.add_node("researcher", researcher)
builder.add_node("researcher_toolnode", ToolNode(tools, handle_tool_errors = True))
builder.set_entry_point("researcher")
builder.add_conditional_edges(
    "researcher", tools_condition,{
        "tools":"researcher_toolnode",
        END: END
    }
)
builder.add_edge("researcher_toolnode","researcher")
graph = builder.compile()