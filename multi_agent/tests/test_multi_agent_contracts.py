"""Baseline contract tests for the LangGraph multi-agent workflow.

Place this file in the project's tests/ directory. Run from the repository root:
    pytest -q tests/test_multi_agent_contracts.py

These tests intentionally mock subagent calls; they do not call real LLMs or web APIs.
"""

import asyncio
import importlib
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# Import by module path to avoid ambiguity between the package and module names.
supervisor_module = importlib.import_module(
    "langgraph_agent.multi_agent.agents.supervisor"
)
researcher_module = importlib.import_module(
    "langgraph_agent.multi_agent.agents.researcher"
)
copywriter_module = importlib.import_module(
    "langgraph_agent.multi_agent.agents.copywriter"
)


def _edges(compiled_graph):
    """Return (source, target) edge pairs from a compiled LangGraph."""
    return {(edge.source, edge.target) for edge in compiled_graph.get_graph().edges}


def test_research_report_contract_has_expected_fields():
    report = researcher_module.ResearchReport(
        topic="RAG evaluation",
        report="Faithfulness measures whether an answer is supported by retrieved context.",
    )

    assert report.topic == "RAG evaluation"
    assert report.report
    assert set(report.model_dump().keys()) == {"topic", "report"}


def test_state_contracts_expose_shared_research_reports():
    supervisor_state = supervisor_module.SupervisorState()
    researcher_state = researcher_module.ResearcherState()
    copywriter_state = copywriter_module.CopyWriterState()

    assert supervisor_state.research_reports == []
    assert researcher_state.research_reports == []
    assert copywriter_state.research_reports == []


def test_researcher_subgraph_has_tool_use_loop():
    edges = _edges(researcher_module.graph)

    assert ("researcher_toolnode", "researcher") in edges
    # The conditional edge is represented by one or more generated branch edges.
    assert any(source == "researcher" for source, _ in edges)


def test_copywriter_subgraph_has_tool_use_loop():
    edges = _edges(copywriter_module.graph)

    assert ("copywriter_toolnode", "copywriter") in edges
    assert any(source == "copywriter" for source, _ in edges)


def test_supervisor_routes_subagent_results_back_to_supervisor():
    edges = _edges(supervisor_module.graph)

    assert ("call_researcher", "supervisor") in edges
    assert ("call_copywriter", "supervisor") in edges
    assert "call_researcher" in supervisor_module.graph.get_graph().nodes
    assert "call_copywriter" in supervisor_module.graph.get_graph().nodes


def test_call_researcher_propagates_reports_and_last_message(monkeypatch):
    expected_report = researcher_module.ResearchReport(
        topic="Agentic workflows", report="A bounded retry loop needs a maximum attempt count."
    )

    class FakeResearchAgent:
        async def ainvoke(self, input, config=None):
            assert isinstance(input["messages"][0], HumanMessage)
            assert input["messages"][0].content == "Research bounded retry loops"
            return {
                "messages": [AIMessage(content="Research completed")],
                "research_reports": [expected_report],
                "status":"completed",
                "error":None            
                }

    monkeypatch.setattr(supervisor_module, "research_agent", FakeResearchAgent())
    state = supervisor_module.SupervisorState(task_description="Research bounded retry loops")

    result = asyncio.run(supervisor_module.call_researcher(state, config={}))
    print(f"\n\n================result1 is:{result}=====================\n\n")
    assert result["research_reports"] == [expected_report]
    assert result["messages"][-1].name == "researcher"
    # assert result["messages"][-1].content == "Research completed"


def test_call_copywriter_receives_accumulated_research_reports(monkeypatch):
    reports = [
        researcher_module.ResearchReport(topic="Topic A", report="Evidence A"),
        researcher_module.ResearchReport(topic="Topic B", report="Evidence B"),
    ]

    class FakeCopywriterAgent:
        async def ainvoke(self, input, config=None):
            assert input["research_reports"] == reports
            assert input["messages"][0].content == "Write a LinkedIn post"
            return {"messages": [AIMessage(content="Draft created")],
            "status": "completed",                  # 👈 ADD THIS LINE
            "output_paths": ["ai_files/mock_post.md"]} 

    monkeypatch.setattr(supervisor_module, "copywriter_agent", FakeCopywriterAgent())
    state = supervisor_module.SupervisorState(
        task_description="Write a LinkedIn post", research_reports=reports
    )

    result = asyncio.run(supervisor_module.call_copywriter(state, config={}))
    print(f"\n\n================result2 is:{result}=====================\n\n")

    assert result["messages"][-1].name == "copywriter"
    # assert result["messages"][-1].content == "Draft created"


def test_report_list_reducer_accumulates_instead_of_replacing():
    """Guard the operator.add reducer contract used by all three states."""
    from typing import get_type_hints
    from langgraph.graph import add_messages  # noqa: F401 - documents reducer dependency
    import operator

    for state_cls in (
        supervisor_module.SupervisorState,
        researcher_module.ResearcherState,
        copywriter_module.CopyWriterState,
    ):
        annotation = get_type_hints(state_cls, include_extras=True)["research_reports"]
        metadata = getattr(annotation, "__metadata__", ())
        assert operator.add in metadata, f"{state_cls.__name__} must use operator.add"
