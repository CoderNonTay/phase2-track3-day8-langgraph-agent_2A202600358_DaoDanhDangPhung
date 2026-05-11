"""Streamlit UI cho Day 08 lab - HITL demo voi LangGraph interrupt().

Chay bang:
    .\.venv\Scripts\python.exe -m streamlit run src/langgraph_agent_lab/streamlit_app.py
    streamlit run src/langgraph_agent_lab/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

os.environ["LANGGRAPH_INTERRUPT"] = "true"

import streamlit as st  # noqa: E402

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.persistence import build_checkpointer  # noqa: E402
from langgraph_agent_lab.state import Route, Scenario, initial_state  # noqa: E402

ROUTE_LABELS_VI: dict[str, str] = {
    "simple": "\u0110\u01a1n gi\u1ea3n (simple)",
    "tool": "G\u1ecdi c\u00f4ng c\u1ee5 (tool)",
    "missing_info": "Thi\u1ebfu th\u00f4ng tin (missing_info)",
    "risky": "H\u00e0nh \u0111\u1ed9ng r\u1ee7i ro (risky)",
    "error": "L\u1ed7i / c\u1ea7n retry (error)",
}

st.set_page_config(page_title="Day 08 LangGraph Agent - Demo HITL", layout="wide")
st.title("\U0001F916 Day 08 LangGraph Agent - Demo Duy\u1ec7t Ng\u01b0\u1eddi D\u00f9ng (HITL)")
st.caption(
    "Minh ho\u1ea1 c\u01a1 ch\u1ebf ng\u1eaft khi g\u1eb7p h\u00e0nh \u0111\u1ed9ng r\u1ee7i ro qua "
    "`langgraph.types.interrupt`. Ng\u01b0\u1eddi d\u00f9ng b\u1ea5m **Duy\u1ec7t** ho\u1eb7c "
    "**T\u1eeb ch\u1ed1i** \u0111\u1ec3 ti\u1ebfp t\u1ee5c b\u1eb1ng `Command(resume=...)`."
)


def _get_graph():
    if "graph" not in st.session_state:
        checkpointer = build_checkpointer("memory")
        st.session_state.graph = build_graph(checkpointer=checkpointer)
    return st.session_state.graph


def _initial_thread_id(scenario_id: str) -> str:
    return f"thread-{scenario_id}-{uuid.uuid4().hex[:6]}"


def _events_to_table(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Node (b\u01b0\u1edbc)": e.get("node"),
            "Lo\u1ea1i s\u1ef1 ki\u1ec7n": e.get("event_type"),
            "Th\u00f4ng \u0111i\u1ec7p": e.get("message"),
            "\u0110\u1ed9 tr\u1ec5 (ms)": e.get("latency_ms", 0),
        }
        for e in events
    ]


def _result_has_interrupt(result: Any) -> bool:
    return isinstance(result, dict) and "__interrupt__" in result


def _render_state(state: dict[str, Any]) -> None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Tuy\u1ebfn (route)**")
        st.code(state.get("route") or "(ch\u01b0a c\u00f3)")
        st.markdown("**M\u1ee9c r\u1ee7i ro**")
        st.code(state.get("risk_level") or "(kh\u00f4ng r\u00f5)")
        st.markdown("**S\u1ed1 l\u1ea7n th\u1eed**")
        st.code(f"{state.get('attempt', 0)} / {state.get('max_attempts', 3)}")
        st.markdown("**Quy\u1ebft \u0111\u1ecbnh duy\u1ec7t**")
        st.code(state.get("approval") or "(ch\u01b0a c\u00f3)")
        st.markdown("**C\u00e2u tr\u1ea3 l\u1eddi cu\u1ed1i**")
        st.write(state.get("final_answer") or "(\u0111ang ch\u1edd)")
    with col2:
        st.markdown("**Nh\u1eadt k\u00fd s\u1ef1 ki\u1ec7n**")
        st.dataframe(_events_to_table(state.get("events") or []), use_container_width=True)
        if state.get("tool_results"):
            st.markdown("**K\u1ebft qu\u1ea3 g\u1ecdi c\u00f4ng c\u1ee5**")
            st.write(state["tool_results"])


with st.sidebar:
    st.header("\u2699\ufe0f C\u1ea5u h\u00ecnh k\u1ecbch b\u1ea3n")
    scenario_id = st.text_input("M\u00e3 k\u1ecbch b\u1ea3n (scenario_id)", value="streamlit-demo")
    query = st.text_area(
        "C\u00e2u y\u00eau c\u1ea7u",
        value="Refund this customer and send confirmation email",
        help="Th\u1eed c\u00e2u ch\u1ee9a t\u1eeb kho\u00e1 r\u1ee7i ro (refund/delete/send/cancel) "
             "\u0111\u1ec3 k\u00edch ho\u1ea1t interrupt().",
    )
    route_options = [r.value for r in Route if r.value not in {"dead_letter", "done"}]
    expected_route = st.selectbox(
        "Tuy\u1ebfn k\u1ef3 v\u1ecdng (expected_route)",
        options=route_options,
        index=route_options.index("risky"),
        format_func=lambda v: ROUTE_LABELS_VI.get(v, v),
    )
    requires_approval = st.checkbox("Y\u00eau c\u1ea7u duy\u1ec7t (requires_approval)", value=True)
    max_attempts = st.number_input("S\u1ed1 l\u1ea7n retry t\u1ed1i \u0111a", min_value=1, max_value=5, value=3)
    run_clicked = st.button("\u25b6\ufe0f Ch\u1ea1y k\u1ecbch b\u1ea3n", type="primary")
    st.divider()
    st.markdown(
        "**M\u1eb9o:** \u0110\u1ec3 demo nh\u00e1nh **T\u1eeb ch\u1ed1i**, sau khi ch\u1ea1y xong nh\u00e1nh "
        "Duy\u1ec7t h\u00e3y b\u1ea5m \"Ch\u1ea1y k\u1ecbch b\u1ea3n\" l\u1ea7n n\u1eefa r\u1ed3i b\u1ea5m "
        "n\u00fat **T\u1eeb ch\u1ed1i** khi g\u1eb7p banner duy\u1ec7t."
    )
    if run_clicked:
        scenario = Scenario(
            id=scenario_id or "streamlit-demo",
            query=query,
            expected_route=Route(expected_route),
            requires_approval=requires_approval,
            max_attempts=int(max_attempts),
        )
        state = initial_state(scenario)
        thread_id = _initial_thread_id(scenario.id)
        st.session_state.thread_id = thread_id
        st.session_state.scenario = scenario
        st.session_state.result = _get_graph().invoke(
            state, config={"configurable": {"thread_id": thread_id}}
        )
        st.session_state.comment = ""


result = st.session_state.get("result")
if result is None:
    st.info(
        "\U0001F448 Ch\u1ecdn k\u1ecbch b\u1ea3n \u1edf thanh b\u00ean tr\u00e1i r\u1ed3i b\u1ea5m "
        "**\u25b6\ufe0f Ch\u1ea1y k\u1ecbch b\u1ea3n** \u0111\u1ec3 b\u1eaft \u0111\u1ea7u."
    )
    st.stop()

thread_id = st.session_state.get("thread_id", "unknown")
st.markdown(f"**M\u00e3 thread (thread_id):** `{thread_id}`")

if _result_has_interrupt(result):
    interrupt_info = result.get("__interrupt__") or []
    payload = (
        interrupt_info[0].value if interrupt_info and hasattr(interrupt_info[0], "value") else {}
    )
    st.warning(
        "\u23f8\ufe0f **C\u1ea7n duy\u1ec7t ng\u01b0\u1eddi d\u00f9ng** - Graph \u0111\u00e3 d\u1eebng "
        "t\u1ea1i `approval_node` qua `interrupt()`. H\u00e3y xem n\u1ed9i dung \u0111\u1ec1 xu\u1ea5t "
        "r\u1ed3i b\u1ea5m **Duy\u1ec7t** ho\u1eb7c **T\u1eeb ch\u1ed1i**."
    )
    st.markdown("**N\u1ed9i dung \u0111\u1ec1 xu\u1ea5t (payload c\u1ee7a interrupt):**")
    st.json(payload, expanded=True)
    full_state = _get_graph().get_state({"configurable": {"thread_id": thread_id}}).values
    _render_state(full_state)

    comment = st.text_input(
        "Ghi ch\u00fa c\u1ee7a ng\u01b0\u1eddi duy\u1ec7t (t\u00f9y ch\u1ecdn)",
        value=st.session_state.get("comment", ""),
        key="comment",
    )
    col_a, col_b = st.columns(2)
    if col_a.button("\u2705 Duy\u1ec7t", type="primary"):
        from langgraph.types import Command
        st.session_state.result = _get_graph().invoke(
            Command(resume={"approved": True, "reviewer": "streamlit-user",
                            "comment": comment or "Duyet qua Streamlit"}),
            config={"configurable": {"thread_id": thread_id}},
        )
        st.rerun()
    if col_b.button("\u274c T\u1eeb ch\u1ed1i"):
        from langgraph.types import Command
        st.session_state.result = _get_graph().invoke(
            Command(resume={"approved": False, "reviewer": "streamlit-user",
                            "comment": comment or "Tu choi qua Streamlit"}),
            config={"configurable": {"thread_id": thread_id}},
        )
        st.rerun()
else:
    st.success("\u2705 Graph \u0111\u00e3 ch\u1ea1y xong.")
    _render_state(result)
