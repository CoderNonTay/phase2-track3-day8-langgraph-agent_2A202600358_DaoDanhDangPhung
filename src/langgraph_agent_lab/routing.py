from __future__ import annotations

from .state import AgentState, Route


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Unknown routes fall back to the simple answer path to keep the graph terminating.
    The tool route enters the parallel fan-out (tool_a + tool_b) instead of a single tool.
    """
    route = state.get("route", Route.SIMPLE.value)
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "fanout_tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
    }
    return mapping.get(route, "answer")


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or escalate to dead-letter.

    Retry exhaustion uses attempt >= max_attempts to keep the loop bounded.
    """
    if int(state.get("attempt", 0)) >= int(state.get("max_attempts", 3)):
        return "dead_letter"
    return "tool"


def route_after_evaluate(state: AgentState) -> str:
    """Decide whether tool result is satisfactory or needs another retry."""
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_approval(state: AgentState) -> str:
    """Continue only if approved; rejected approvals go to clarify for safe fallback."""
    approval = state.get("approval") or {}
    return "tool" if approval.get("approved") else "clarify"
