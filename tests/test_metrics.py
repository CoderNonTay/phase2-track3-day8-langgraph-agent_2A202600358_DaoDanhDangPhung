from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics
from langgraph_agent_lab.state import make_event


def test_metric_from_state_success():
    state = {
        "scenario_id": "S",
        "route": "simple",
        "final_answer": "ok",
        "events": [
            make_event("intake", "completed", "ok", latency_ms=1),
            make_event("answer", "completed", "ok", latency_ms=2),
        ],
        "errors": [],
    }
    metric = metric_from_state(state, expected_route="simple", approval_required=False)
    assert metric.success is True
    assert metric.nodes_visited == 2
    assert metric.latency_ms == 3


def test_summarize_metrics():
    m1 = metric_from_state(
        {
            "scenario_id": "1",
            "route": "simple",
            "final_answer": "ok",
            "events": [],
            "errors": [],
        },
        "simple",
        False,
    )
    m2 = metric_from_state(
        {
            "scenario_id": "2",
            "route": "tool",
            "final_answer": None,
            "events": [],
            "errors": [],
        },
        "tool",
        False,
    )
    report = summarize_metrics([m1, m2])
    assert report.total_scenarios == 2
    assert 0 <= report.success_rate <= 1
    assert report.resume_success is False


def test_summarize_metrics_resume_success_flag():
    m1 = metric_from_state(
        {
            "scenario_id": "1",
            "route": "simple",
            "final_answer": "ok",
            "events": [],
            "errors": [],
        },
        "simple",
        False,
    )
    report = summarize_metrics([m1], resume_success=True)
    assert report.resume_success is True


def test_interrupt_count_prefers_real_interrupts():
    state = {
        "scenario_id": "S",
        "route": "risky",
        "final_answer": "done",
        "approval": {"approved": True},
        "events": [
            make_event("intake", "completed", "ok"),
            make_event("approval", "completed", "ok", used_interrupt=True),
        ],
        "errors": [],
    }
    metric = metric_from_state(state, expected_route="risky", approval_required=True)
    assert metric.interrupt_count == 1
    assert metric.approval_observed is True
