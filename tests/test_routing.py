from langgraph_agent_lab.routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from langgraph_agent_lab.state import Route


def test_route_after_classify_simple():
    assert route_after_classify({"route": Route.SIMPLE.value}) == "answer"


def test_route_after_classify_tool_goes_to_fanout():
    # Tool route now enters the parallel fan-out node before tool_a/tool_b run.
    assert route_after_classify({"route": Route.TOOL.value}) == "fanout_tool"


def test_route_after_classify_risky():
    assert route_after_classify({"route": Route.RISKY.value}) == "risky_action"


def test_route_after_classify_missing_info():
    assert route_after_classify({"route": Route.MISSING_INFO.value}) == "clarify"


def test_route_after_classify_error():
    assert route_after_classify({"route": Route.ERROR.value}) == "retry"


def test_route_after_classify_unknown_falls_back_to_answer():
    assert route_after_classify({"route": "weird"}) == "answer"


def test_route_after_approval():
    assert route_after_approval({"approval": {"approved": True}}) == "tool"
    assert route_after_approval({"approval": {"approved": False}}) == "clarify"
    assert route_after_approval({}) == "clarify"  # no approval => safe fallback


def test_route_after_retry_bound():
    assert route_after_retry({"attempt": 0, "max_attempts": 3}) == "tool"
    assert route_after_retry({"attempt": 2, "max_attempts": 3}) == "tool"
    assert route_after_retry({"attempt": 3, "max_attempts": 3}) == "dead_letter"
    assert route_after_retry({"attempt": 5, "max_attempts": 3}) == "dead_letter"


def test_route_after_evaluate():
    assert route_after_evaluate({"evaluation_result": "success"}) == "answer"
    assert route_after_evaluate({"evaluation_result": "needs_retry"}) == "retry"
