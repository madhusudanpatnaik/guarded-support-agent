"""`toolagent` — a ReAct tool-use agent with a gate, human review of irreversible actions, and a
trace of every step.

  offline (no key):
    toolagent demo --task damaged-refund --trajectory sloppy     # one scripted run, with its trace
    toolagent eval --scripted                                    # the harness matrix
    toolagent trace runs/demo.jsonl                              # summary + timeline of a run
  live (ANTHROPIC_API_KEY):
    toolagent run "Handle ticket T-1."                           # you approve at the terminal
    toolagent eval --trials 4 --traces runs/eval                 # Claude on every task, pass^k
    toolagent download                                           # StableToolBench test queries
    toolagent toolbench --queries data/toolbench/G1_instruction.json --limit 50 --judge
    toolagent tau2 run --domain airline --agent toolagent_gated --user-llm <model>   # tau2 extra
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from toolagent import assets
from toolagent.agent import Agent, RunResult
from toolagent.approval import Approver, ConsoleApprover, DenyAll
from toolagent.backoffice import SYSTEM, BackOffice
from toolagent.evaluate import (
    REVIEWERS,
    Outcome,
    report,
    run_task,
    scripted_matrix,
)
from toolagent.models import DEFAULT_MODEL, ClaudeModel, Model, ScriptedModel
from toolagent.policy import Policy
from toolagent.tasks import TASKS, find
from toolagent.trace import Span, read_jsonl, summarize, timeline


def _print_run(result: RunResult) -> None:
    for c in result.calls:
        review = ""
        if c.approval is not None:
            review = f" [{c.approval.reviewer}: {'yes' if c.approval.approved else 'no'}]"
        elif c.auto_approved:
            review = " [auto-approved]"
        ran = "ran" if c.executed else "not run"
        print(f"  turn {c.turn}: {c.tool} {c.args} -> {c.decision.value}{review}, {ran}")
        if not c.executed:
            print(f"           {c.observation[:160]}")
    print(f"status: {result.status} ({result.reason})")
    if result.rolled_back:
        print("rolled back: " + "; ".join(result.rolled_back))
    if result.answer:
        print(f"answer: {result.answer}")


def _show_trace(spans: Sequence[Span], otlp: bool) -> None:
    print("\n".join(summarize(spans).lines()))
    print("\n".join(timeline(spans)))
    if otlp:
        from toolagent.otel import export_with_sdk

        print(f"exported {export_with_sdk(spans)} spans over OTLP")


def _claude(args: argparse.Namespace) -> Model:
    return ClaudeModel(model=args.model, effort=args.effort)


def _cmd_demo(args: argparse.Namespace) -> int:
    task = find(args.task)
    model = ScriptedModel(getattr(task, args.trajectory))
    print(f"task {task.id}: {task.instruction}")
    if args.trajectory == "sloppy":
        print(f"(the sloppy script {task.mistake})")
    outcome, result = run_task(
        task,
        model,
        reviewer=args.reviewer,
        gate=not args.no_gate,
        capture_content=args.capture_content,
    )
    _print_run(result)
    print(
        f"scored: success={outcome.success} violations={outcome.violations} "
        f"missing={outcome.missing} unreviewed={outcome.unreviewed}\n"
    )
    if args.trace:
        result.tracer.write_jsonl(args.trace)
    _show_trace(result.tracer.spans, args.otlp)
    return 0


def _table(rows: list[tuple[str, list[Outcome]]]) -> None:
    head = f"{'':44} {'success':>7} {'violations':>10} {'unreviewed':>10} {'asked':>5} "
    print(head + f"{'declined':>8} {'refused':>7} {'turns':>5}")
    for label, outs in rows:
        r = report(outs)
        print(
            f"{label:44} {r.success:>7.0%} {r.violations:>10} {r.unreviewed:>10} {r.asked:>5} "
            f"{r.declined:>8} {r.refused:>7} {r.mean_turns:>5.1f}"
        )


def _cmd_eval(args: argparse.Namespace) -> int:
    tasks = [find(t) for t in args.tasks.split(",")] if args.tasks else list(TASKS)
    if args.scripted:
        cells = scripted_matrix(tasks, lambda t, k: ScriptedModel(getattr(t, k)))
        _table([(f"{traj:8} {label}", outs) for (traj, label), outs in cells.items()])
        print("\nScripted trajectories test the harness, not a model: see the README.")
        return 0
    outcomes: list[Outcome] = []
    for task in tasks:
        for trial in range(args.trials):
            o, result = run_task(
                task,
                _claude(args),
                reviewer=args.reviewer,
                gate=not args.no_gate,
                trial=trial,
                max_cost_usd=args.max_cost,
                capture_content=args.capture_content,
            )
            outcomes.append(o)
            cost = "?" if o.cost_usd is None else f"${o.cost_usd:.3f}"
            print(f"{task.id:16} trial {trial}: success={o.success} {o.status} "
                  f"violations={o.violations} turns={o.turns} {cost}")  # fmt: skip
            if args.traces:
                result.tracer.write_jsonl(Path(args.traces) / f"{task.id}-{trial}.jsonl")
    r = report(outcomes)
    _table([(f"{args.model} ({args.reviewer})", outcomes)])
    print("pass^k: " + ", ".join(f"k={k} {v:.2f}" for k, v in r.pass_hat.items()))
    print("pass@k: " + ", ".join(f"k={k} {v:.2f}" for k, v in r.pass_at.items()))
    print("cost: " + ("unknown" if r.cost_usd is None else f"${r.cost_usd:.2f}"))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    env = BackOffice()
    approver: Approver = ConsoleApprover() if args.reviewer == "console" else DenyAll()
    policy = Policy(auto_approve=env.auto_rules(args.instruction), max_cost_usd=args.max_cost)
    agent = Agent(
        _claude(args),
        env.toolbox(),
        approver=approver,
        policy=policy,
        system=SYSTEM,
        capture_content=args.capture_content,
    )
    result = agent.run(args.instruction)
    _print_run(result)
    print("effects: " + (", ".join(":".join(e) for e in env.effects()) or "none"))
    if args.trace:
        result.tracer.write_jsonl(args.trace)
    _show_trace(result.tracer.spans, args.otlp)
    return 0 if result.status == "completed" else 1


def _cmd_trace(args: argparse.Namespace) -> int:
    _show_trace(read_jsonl(args.file), args.otlp)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    groups = args.groups or list(assets.GROUPS)
    if args.list:
        print("\n".join(assets.describe(groups)))
        return 0
    for line in assets.download(groups, args.to):
        print(line)
    return 0


def _cmd_toolbench(args: argparse.Namespace) -> int:
    from toolagent import toolbench

    queries = toolbench.load_queries(args.queries)[: args.limit]
    server = toolbench.VirtualApiServer(args.server, key=args.toolbench_key)
    policy = Policy(max_turns=args.max_turns, max_cost_usd=args.max_cost)
    outcomes, pairs = [], []
    for q in queries:
        o, result = toolbench.run_query(q, _claude(args), server, policy, args.capture_content)
        outcomes.append(o)
        pairs.append((q, result))
        recall = "-" if o.relevant_recall is None else f"{o.relevant_recall:.2f}"
        print(f"{q.query_id}: {o.status} answered={o.answered} recall={recall} turns={o.turns}")
        if args.traces:
            result.tracer.write_jsonl(Path(args.traces) / f"{q.query_id}.jsonl")
    for k, v in toolbench.summary(outcomes).items():
        print(f"{k}: {v}")
    if args.judge:
        judge = toolbench.ClaudeJudge(model=args.model)
        pr = toolbench.solvable_pass_rate(pairs, judge, args.evaluate_times)
        print(f"solvable pass rate (SoPR-style, {args.evaluate_times} judge runs): "
              f"{pr.pass_rate:.1f} ± {pr.std:.1f}")  # fmt: skip
    return 0


def _cmd_tau2(args: argparse.Namespace) -> int:
    from toolagent import tau2_agent

    return tau2_agent.main(args.tau2_args)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="toolagent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)

    def live(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"])  # fmt: skip
        sp.add_argument("--max-cost", type=float, default=1.0, help="USD per run")
        content(sp)

    def content(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--capture-content",
            action="store_true",
            default=None,
            help="record reasoning, gate reasons and reviewer notes in traces (default: off, or "
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true)",
        )

    d = sub.add_parser("demo", help="one scripted run on a task, offline")
    d.add_argument("--task", default="damaged-refund", choices=[t.id for t in TASKS])
    d.add_argument("--trajectory", default="sloppy", choices=["careful", "sloppy"])
    d.add_argument("--reviewer", default="informed", choices=REVIEWERS)
    d.add_argument("--no-gate", action="store_true")
    d.add_argument("--trace", help="write the spans to this JSONL file")
    d.add_argument("--otlp", action="store_true", help="also export over OTLP (otel extra)")
    content(d)
    d.set_defaults(fn=_cmd_demo)

    e = sub.add_parser("eval", help="score tasks: scripted harness matrix, or Claude live")
    e.add_argument("--scripted", action="store_true", help="offline: careful/sloppy scripts")
    e.add_argument("--tasks", help="comma-separated task ids (default: all)")
    e.add_argument("--trials", type=int, default=1)
    e.add_argument("--reviewer", default="informed", choices=REVIEWERS)
    e.add_argument("--no-gate", action="store_true")
    e.add_argument("--traces", help="directory for one JSONL trace per run")
    live(e)
    e.set_defaults(fn=_cmd_eval)

    r = sub.add_parser("run", help="Claude on your own instruction against the back office")
    r.add_argument("instruction")
    r.add_argument("--reviewer", default="console", choices=["console", "unattended"])
    r.add_argument("--trace")
    r.add_argument("--otlp", action="store_true")
    live(r)
    r.set_defaults(fn=_cmd_run)

    t = sub.add_parser("trace", help="summarize a JSONL trace")
    t.add_argument("file")
    t.add_argument("--otlp", action="store_true")
    t.set_defaults(fn=_cmd_trace)

    dl = sub.add_parser("download", help="fetch StableToolBench's solvable test queries")
    dl.add_argument("groups", nargs="*", metavar="GROUP", help=", ".join(assets.GROUPS))
    dl.add_argument("--list", action="store_true")
    dl.add_argument("--to", default="data/toolbench")
    dl.set_defaults(fn=_cmd_download)

    tb = sub.add_parser(
        "toolbench", help="Claude on ToolBench queries via a StableToolBench server"
    )
    tb.add_argument("--queries", required=True)
    tb.add_argument("--server", default="http://localhost:8080/virtual")
    tb.add_argument("--toolbench-key", default=os.environ.get("TOOLBENCH_KEY", ""))
    tb.add_argument("--limit", type=int)
    tb.add_argument("--max-turns", type=int, default=12)
    tb.add_argument("--traces")
    tb.add_argument("--judge", action="store_true", help="also compute the solvable pass rate")
    tb.add_argument("--evaluate-times", type=int, default=4, help="judge runs per query")
    live(tb)
    tb.set_defaults(fn=_cmd_toolbench)

    t2 = sub.add_parser(
        "tau2", help="run τ²-bench with this agent registered as --agent toolagent_gated"
    )
    t2.add_argument("tau2_args", nargs=argparse.REMAINDER, help="passed to τ²-bench's CLI")
    t2.set_defaults(fn=_cmd_tau2)

    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except ModuleNotFoundError as e:
        hint = _install_hint(e.name)
        if hint is None:
            raise
        print(f"error: {e.name} is not installed; run: pip install -r {hint}", file=sys.stderr)
        return 2
    except TypeError as e:  # the SDK found no credentials when the request was made
        if "Could not resolve authentication method" not in str(e):
            raise
        print("error: no Anthropic credentials; set ANTHROPIC_API_KEY (see .env.example)",
              file=sys.stderr)  # fmt: skip
        return 2


# optional module -> the requirements file that installs it
_OPTIONAL = {
    "tau2": "requirements-tau2.txt",
    "opentelemetry.sdk": "requirements-otel.txt",
    "opentelemetry.exporter": "requirements-otel.txt",
}


def _install_hint(module: str | None) -> str | None:
    name = module or ""
    return next((f for m, f in _OPTIONAL.items() if name == m or name.startswith(m + ".")), None)


if __name__ == "__main__":
    sys.exit(main())
