from __future__ import annotations

import time

from .state import AgentState, ApprovalDecision, Route, make_event

# Keyword sets per route. Priority order: risky -> tool -> error -> missing_info -> simple.
# Documented in reports/lab_report.md.
RISKY_KEYWORDS = {"refund", "delete", "send", "cancel", "remove", "revoke"}
TOOL_KEYWORDS = {"status", "order", "lookup", "check", "track", "find", "search"}
ERROR_KEYWORDS = {"timeout", "fail", "failure", "failed", "error", "crash", "unavailable"}
MISSING_INFO_PRONOUNS = {"it", "this", "that", "them"}

_PUNCT = "?!.,;:()[]{}\"'"


def _tokenize(query: str) -> list[str]:
    """Lowercase + strip punctuation per token."""
    return [w.strip(_PUNCT) for w in query.lower().split() if w.strip(_PUNCT)]


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields."""
    started = time.perf_counter()
    query = state.get("query", "").strip()
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized", latency_ms=latency_ms)],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using tokenized keyword matching.

    Priority: risky > tool > error > missing_info > simple.
    Word-level matching avoids substring false positives like "send" in "sender".
    """
    started = time.perf_counter()
    tokens = _tokenize(state.get("query", ""))
    token_set = set(tokens)

    route = Route.SIMPLE
    risk_level = "low"

    if token_set & RISKY_KEYWORDS:
        route = Route.RISKY
        risk_level = "high"
    elif token_set & TOOL_KEYWORDS:
        route = Route.TOOL
    elif token_set & ERROR_KEYWORDS:
        route = Route.ERROR
        risk_level = "medium"
    elif len(tokens) < 5 and (token_set & MISSING_INFO_PRONOUNS):
        route = Route.MISSING_INFO

    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}",
                latency_ms=latency_ms,
                tokens=tokens[:10],
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    started = time.perf_counter()
    query = state.get("query", "")
    if state.get("approval") and not (state.get("approval") or {}).get("approved"):
        question = (
            "Reviewer rejected the risky action. "
            f"Please confirm intent or provide more context for: '{query}'."
        )
    else:
        question = "Can you provide the order id or the missing context for your request?"
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested", latency_ms=latency_ms)],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    """
    started = time.perf_counter()
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={scenario_id}"
    else:
        result = f"mock-tool-result for scenario={scenario_id}"
    latency_ms = int((time.perf_counter() - started) * 1000) + 5  # +5ms to simulate I/O
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}", latency_ms=latency_ms)],
    }


def tool_a_node(state: AgentState) -> dict:
    """Parallel fan-out tool A (order_lookup)."""
    started = time.perf_counter()
    scenario_id = state.get("scenario_id", "unknown")
    result = f"tool_a:order_lookup ok for scenario={scenario_id}"
    latency_ms = int((time.perf_counter() - started) * 1000) + 5
    return {
        "tool_results": [result],
        "events": [make_event("tool_a", "completed", "order_lookup executed", latency_ms=latency_ms)],
    }


def tool_b_node(state: AgentState) -> dict:
    """Parallel fan-out tool B (inventory_check)."""
    started = time.perf_counter()
    scenario_id = state.get("scenario_id", "unknown")
    result = f"tool_b:inventory_check ok for scenario={scenario_id}"
    latency_ms = int((time.perf_counter() - started) * 1000) + 5
    return {
        "tool_results": [result],
        "events": [make_event("tool_b", "completed", "inventory_check executed", latency_ms=latency_ms)],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval."""
    started = time.perf_counter()
    query = state.get("query", "")
    proposed = f"Execute risky action for: {query[:80]} (risk_level={state.get('risk_level', 'high')})"
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "proposed_action": proposed,
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                "approval required",
                latency_ms=latency_ms,
                risk_level=state.get("risk_level", "high"),
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.
    """
    import os

    started = time.perf_counter()
    used_interrupt = False
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt(
            {
                "scenario_id": state.get("scenario_id"),
                "proposed_action": state.get("proposed_action"),
                "risk_level": state.get("risk_level"),
            }
        )
        used_interrupt = True
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                "completed",
                f"approved={decision.approved}",
                latency_ms=latency_ms,
                used_interrupt=used_interrupt,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision."""
    started = time.perf_counter()
    attempt = int(state.get("attempt", 0)) + 1
    errors = [f"transient failure attempt={attempt}"]
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                latency_ms=latency_ms,
                attempt=attempt,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in tool results or approval."""
    started = time.perf_counter()
    tool_results = state.get("tool_results") or []
    approval = state.get("approval") or {}
    if approval and approval.get("approved") and tool_results:
        answer = (
            f"Approved by {approval.get('reviewer', 'reviewer')}. "
            f"Action completed. Results: {'; '.join(tool_results)}"
        )
    elif tool_results:
        answer = f"I found: {'; '.join(tool_results)}"
    else:
        answer = "Here is a safe mock answer based on your query."
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated", latency_ms=latency_ms)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results - the 'done?' check that enables retry loops.

    Returns needs_retry when any tool result contains ERROR or when tool_results is empty.
    """
    started = time.perf_counter()
    tool_results = state.get("tool_results") or []
    if not tool_results:
        result = "needs_retry"
        message = "no tool results, retry needed"
    elif any("ERROR" in r for r in tool_results):
        result = "needs_retry"
        message = "tool result indicates failure, retry needed"
    else:
        result = "success"
        message = "tool result satisfactory"
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "evaluation_result": result,
        "events": [make_event("evaluate", "completed", message, latency_ms=latency_ms)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review."""
    started = time.perf_counter()
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "final_answer": "Request could not be completed after maximum retry attempts. Logged for manual review.",
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded, attempt={state.get('attempt', 0)}",
                latency_ms=latency_ms,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished", latency_ms=0)]}
