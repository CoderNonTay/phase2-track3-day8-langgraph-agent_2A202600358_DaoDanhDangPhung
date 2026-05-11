from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    history_log: list[str] = []
    resume_success = False
    # Sample threads for which we log state_history as persistence evidence.
    history_targets = {"thread-S01_simple", "thread-S05_error", "thread-S04_risky"}
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(
            metric_from_state(
                final_state, scenario.expected_route.value, scenario.requires_approval
            )
        )
        # Collect state_history evidence for a few threads.
        if checkpointer is not None and state["thread_id"] in history_targets:
            try:
                checkpoints = list(graph.get_state_history(run_config))
            except Exception as exc:  # pragma: no cover - defensive
                history_log.append(f"[{state['thread_id']}] history error: {exc}")
                continue
            history_log.append(
                f"[{state['thread_id']}] checkpoints={len(checkpoints)}"
            )
            for cp in checkpoints[:5]:
                history_log.append(
                    f"  step={cp.metadata.get('step') if cp.metadata else '?'} "
                    f"next={list(cp.next) if cp.next else []} "
                    f"route={cp.values.get('route') if isinstance(cp.values, dict) else '?'}"
                )
            if len(checkpoints) > 1:
                resume_success = True
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    if history_log:
        history_path = Path(output).with_name("state_history.log")
        history_path.write_text("\n".join(history_log), encoding="utf-8")
        typer.echo(f"Wrote state history evidence to {history_path}")
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("replay")
def replay(
    thread_id: Annotated[str, typer.Option("--thread-id")],
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
) -> None:
    """Replay a thread's state history from the checkpointer (persistence evidence).

    Re-runs the matching scenario first so the in-memory checkpointer has data,
    then prints each checkpoint (step, next, route, attempt) for inspection.
    """
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    if checkpointer is None:
        raise typer.BadParameter("Replay requires a checkpointer (set checkpointer: memory or sqlite)")
    graph = build_graph(checkpointer=checkpointer)
    matching = [s for s in scenarios if f"thread-{s.id}" == thread_id]
    if not matching:
        raise typer.BadParameter(f"No scenario maps to thread_id {thread_id}")
    scenario = matching[0]
    state = initial_state(scenario)
    run_config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(state, config=run_config)
    checkpoints = list(graph.get_state_history(run_config))
    typer.echo(f"thread_id={thread_id} checkpoints={len(checkpoints)}")
    for cp in checkpoints:
        meta = cp.metadata or {}
        values = cp.values if isinstance(cp.values, dict) else {}
        typer.echo(
            f"  step={meta.get('step')} next={list(cp.next) if cp.next else []} "
            f"route={values.get('route')} attempt={values.get('attempt')} "
            f"final_answer={(values.get('final_answer') or '')[:60]}"
        )


@app.command("export-diagram")
def export_diagram(
    output: Annotated[Path, typer.Option("--output")] = Path("reports/graph.mmd"),
) -> None:
    """Export the compiled graph as a Mermaid diagram."""
    graph = build_graph(checkpointer=None)
    diagram = graph.get_graph().draw_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(diagram, encoding="utf-8")
    typer.echo(f"Wrote Mermaid diagram to {output}")


if __name__ == "__main__":
    app()
