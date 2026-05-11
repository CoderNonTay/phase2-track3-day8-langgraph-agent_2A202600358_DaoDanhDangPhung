import pytest

from langgraph_agent_lab.nodes import classify_node
from langgraph_agent_lab.state import Route


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Risky has highest priority, even if it contains a tool keyword like "order".
        ("Delete this order immediately", Route.RISKY.value),
        ("Refund customer for order 999", Route.RISKY.value),
        ("Send the confirmation email", Route.RISKY.value),
        ("Cancel my subscription", Route.RISKY.value),
        # Tool keywords without risky.
        ("Please lookup order status for order 12345", Route.TOOL.value),
        ("Search products in catalog", Route.TOOL.value),
        ("Track shipment 999", Route.TOOL.value),
        # Error keywords -- should beat missing_info even when "it" appears.
        ("Failure crash unavailable now", Route.ERROR.value),
        ("Timeout failure while processing request", Route.ERROR.value),
        ("System failure cannot recover after multiple attempts", Route.ERROR.value),
        # Missing info: short + has vague pronoun.
        ("Can you fix it?", Route.MISSING_INFO.value),
        ("Help me with this", Route.MISSING_INFO.value),
        # Default to simple.
        ("How do I reset my password?", Route.SIMPLE.value),
        ("What are your business hours", Route.SIMPLE.value),
        # Word-boundary safety: "sender" should NOT trigger risky via "send".
        ("My sender address is wrong", Route.SIMPLE.value),
        # Word-boundary safety: "item" should NOT trigger missing_info via "it".
        ("I have an item question", Route.SIMPLE.value),
    ],
)
def test_classify_priority(query, expected):
    out = classify_node({"query": query})
    assert out["route"] == expected
