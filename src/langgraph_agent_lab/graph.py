from __future__ import annotations

from typing import Any

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_a_node,
    tool_b_node,
    tool_node,
)
from .routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState


def _fanout_tool_node(state: AgentState) -> dict:
    """Identity node that emits a fan-out event before parallel tools run.

    The actual fan-out to tool_a/tool_b is wired with conditional edges + Send below.
    """
    from .state import make_event

    return {"events": [make_event("fanout_tool", "completed", "fanning out to tool_a + tool_b", latency_ms=0)]}


def _fanout_dispatch(state: AgentState) -> list:
    """Return a list of Send objects so LangGraph runs tool_a and tool_b in parallel."""
    from langgraph.types import Send

    return [Send("tool_a", state), Send("tool_b", state)]


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow.

    Architecture:
    - intake -> classify -> [simple|tool|missing_info|risky|error]
    - simple       -> answer -> finalize -> END
    - tool         -> fanout_tool -> (tool_a || tool_b) -> evaluate -> answer -> finalize
    - missing_info -> clarify -> finalize -> END
    - risky        -> risky_action -> approval -> tool -> evaluate -> answer -> finalize
    - error        -> retry -> tool -> evaluate -> (retry loop bounded by max_attempts)
    - retry exhausted -> dead_letter -> finalize -> END
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - helpful install error
        raise RuntimeError(
            "LangGraph is required. Run: pip install -e '.[dev]' or pip install langgraph"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tool", tool_node)
    graph.add_node("fanout_tool", _fanout_tool_node)
    graph.add_node("tool_a", tool_a_node)
    graph.add_node("tool_b", tool_b_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("risky_action", risky_action_node)
    graph.add_node("approval", approval_node)
    graph.add_node("retry", retry_or_fallback_node)
    graph.add_node("dead_letter", dead_letter_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer": "answer",
            "fanout_tool": "fanout_tool",
            "clarify": "clarify",
            "risky_action": "risky_action",
            "retry": "retry",
        },
    )
    # Fan-out: fanout_tool dispatches Send() to tool_a and tool_b in parallel.
    graph.add_conditional_edges("fanout_tool", _fanout_dispatch, ["tool_a", "tool_b"])
    graph.add_edge("tool_a", "evaluate")
    graph.add_edge("tool_b", "evaluate")
    # Single-tool path for risky (post-approval) and retry loop.
    graph.add_edge("tool", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_evaluate, {"retry": "retry", "answer": "answer"}
    )
    graph.add_edge("clarify", "finalize")
    graph.add_edge("risky_action", "approval")
    graph.add_conditional_edges(
        "approval", route_after_approval, {"tool": "tool", "clarify": "clarify"}
    )
    graph.add_conditional_edges(
        "retry", route_after_retry, {"tool": "tool", "dead_letter": "dead_letter"}
    )
    graph.add_edge("answer", "finalize")
    graph.add_edge("dead_letter", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
