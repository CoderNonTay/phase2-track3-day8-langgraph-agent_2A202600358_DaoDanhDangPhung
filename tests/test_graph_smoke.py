import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph not installed in local environment",
)

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How do I reset my password?", Route.SIMPLE.value),
        ("Please lookup order status for order 123", Route.TOOL.value),
        ("Refund this customer", Route.RISKY.value),
    ],
)
def test_graph_runs_basic_routes(query, expected_route):
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="smoke", query=query, expected_route=Route(expected_route))
    state = initial_state(scenario)
    result = graph.invoke(
        state, config={"configurable": {"thread_id": state["thread_id"]}}
    )
    assert result["route"] == expected_route
    assert result.get("final_answer") or result.get("pending_question")


def test_graph_error_route_dead_letters_when_max_attempts_one():
    """S07-style: error route with max_attempts=1 should reach dead_letter."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="dead",
        query="Timeout failure while processing request",
        expected_route=Route.ERROR,
        max_attempts=1,
    )
    state = initial_state(scenario)
    result = graph.invoke(
        state, config={"configurable": {"thread_id": state["thread_id"]}}
    )
    assert result["route"] == Route.ERROR.value
    assert "manual review" in (result.get("final_answer") or "").lower()


def test_graph_risky_with_mock_approval_completes():
    """Risky route with default mock approval -> tool -> evaluate -> answer."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="risky",
        query="Refund this customer and send confirmation email",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    result = graph.invoke(
        state, config={"configurable": {"thread_id": state["thread_id"]}}
    )
    assert result["route"] == Route.RISKY.value
    assert result.get("approval") is not None
    assert (result.get("final_answer") or "").strip() != ""


def test_graph_tool_route_runs_parallel_fanout():
    """Tool route should produce results from both tool_a and tool_b (fan-out)."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="fanout",
        query="Please lookup order status for order 123",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    result = graph.invoke(
        state, config={"configurable": {"thread_id": state["thread_id"]}}
    )
    joined = " ".join(result.get("tool_results") or [])
    assert "tool_a" in joined
    assert "tool_b" in joined
