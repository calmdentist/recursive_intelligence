"""CLI entrypoint for the recursive intelligence runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import click

from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.orchestrator import Orchestrator, get_node_tree
from recursive_intelligence.runtime.state_store import StateStore, NodeState


# ── ANSI helpers ─────────────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
CLEAR_LINE = "\033[2K\r"

STATE_STYLE = {
    "completed": f"{GREEN}completed{RESET}",
    "failed": f"{RED}failed{RESET}",
    "cancelled": f"{DIM}cancelled{RESET}",
    "paused": f"{YELLOW}paused{RESET}",
    "running": f"{CYAN}running{RESET}",
    "executing": f"{CYAN}executing{RESET}",
    "planning": f"{CYAN}planning{RESET}",
    "merging": f"{CYAN}merging{RESET}",
    "queued": f"{DIM}queued{RESET}",
    "waiting_on_children": f"{YELLOW}waiting{RESET}",
    "reviewing_children": f"{CYAN}reviewing{RESET}",
    "verifying": f"{CYAN}verifying{RESET}",
}


def _s(state: str) -> str:
    return STATE_STYLE.get(state, state)

def _dim(t: str) -> str:
    return f"{DIM}{t}{RESET}"

def _bold(t: str) -> str:
    return f"{BOLD}{t}{RESET}"

def _cyan(t: str) -> str:
    return f"{CYAN}{t}{RESET}"

def _green(t: str) -> str:
    return f"{GREEN}{t}{RESET}"

def _yellow(t: str) -> str:
    return f"{YELLOW}{t}{RESET}"

def _red(t: str) -> str:
    return f"{RED}{t}{RESET}"

def _magenta(t: str) -> str:
    return f"{MAGENTA}{t}{RESET}"

def _blue(t: str) -> str:
    return f"{BLUE}{t}{RESET}"

def _truncate(text: str, max_len: int = 80) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def _cols() -> int:
    try:
        return min(os.get_terminal_size().columns, 100)
    except OSError:
        return 80

def _hr() -> str:
    return _dim("\u2500" * _cols())


def _format_unsupported(count: int) -> str:
    if count <= 0:
        return ""
    label = "task" if count == 1 else "tasks"
    return f" {_dim(f'({count} unsupported {label})')}"

INDENT = "  "


# ── Banner ───────────────────────────────────────────────────────────────────

BANNER = f"""\
{DIM}
  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
  \u2502{RESET}{BOLD}  recursive intelligence{RESET}{DIM}              \u2502
  \u2502                                     \u2502
  \u2502{RESET}{DIM}  recursive coding-agent runtime    \u2502
  \u2502  built on claude code               \u2502
  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518
{RESET}"""


def _setup_logging(verbose: bool, chat_mode: bool, config: RuntimeConfig) -> None:
    if chat_mode and not verbose:
        config.ensure_dirs()
        log_path = config.ri_dir / "rari.log"
        handler = logging.FileHandler(str(log_path), mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
    else:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# ── Stream renderer ──────────────────────────────────────────────────────────

class StreamRenderer:
    """Renders streaming events from the root node, Claude Code style.

    Called from the background thread — all output goes through
    thread-safe sys.stderr.write to avoid interleaving with click prompts.
    """

    def __init__(self) -> None:
        self._in_text = False
        self._lock = threading.Lock()

    def on_message(self, msg_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            if msg_type == "text":
                self._render_text(data.get("text", ""))
            elif msg_type == "thinking":
                self._render_thinking(data.get("text", ""))
            elif msg_type == "tool_use":
                self._render_tool_use(data.get("tool", ""), data.get("input", {}))
            elif msg_type == "tool_result":
                self._render_tool_result(data.get("content", ""))

    def _write(self, text: str) -> None:
        sys.stderr.write(text)
        sys.stderr.flush()

    def _end_text(self) -> None:
        if self._in_text:
            self._write("\n")
            self._in_text = False

    def _render_text(self, text: str) -> None:
        if not text:
            return
        if not self._in_text:
            self._write(f"\n{INDENT}")
            self._in_text = True
        # Wrap long lines
        for line in text.split("\n"):
            self._write(f"{line}\n{INDENT}")

    def _render_thinking(self, text: str) -> None:
        self._end_text()
        # Show thinking as a compact dimmed block
        if not text.strip():
            return
        lines = text.strip().split("\n")
        preview = lines[0][:80]
        if len(lines) > 1 or len(lines[0]) > 80:
            preview += "..."
        self._write(f"{INDENT}{DIM}\u2501 {preview}{RESET}\n")

    def _render_tool_use(self, tool: str, tool_input: dict) -> None:
        self._end_text()
        # Format like Claude Code: tool name + key argument
        arg_summary = self._summarize_tool_input(tool, tool_input)
        self._write(f"{INDENT}{CYAN}\u25b8 {tool}{RESET}")
        if arg_summary:
            self._write(f" {DIM}{arg_summary}{RESET}")
        self._write("\n")

    def _render_tool_result(self, content: str) -> None:
        if not content or not content.strip():
            return
        # Show a compact preview of the result
        lines = content.strip().split("\n")
        preview = lines[0][:100]
        if len(lines) > 1:
            preview += f" {DIM}(+{len(lines)-1} lines){RESET}"
        self._write(f"{INDENT}  {DIM}{preview}{RESET}\n")

    def _summarize_tool_input(self, tool: str, inp: dict) -> str:
        if tool in ("Read", "Glob", "Grep"):
            path = inp.get("file_path") or inp.get("path") or inp.get("pattern") or ""
            return str(path)
        if tool in ("Edit", "Write"):
            return str(inp.get("file_path", ""))
        if tool == "Bash":
            cmd = inp.get("command", "")
            return cmd[:60] if cmd else ""
        if tool == "Agent":
            return inp.get("description", "")[:40]
        return ""

    def finish(self) -> None:
        """Call when the session is done to clean up."""
        with self._lock:
            self._end_text()


# ── Background runner ────────────────────────────────────────────────────────

class BackgroundRunner:
    def __init__(self, orchestrator: Orchestrator, config: RuntimeConfig) -> None:
        self.orchestrator = orchestrator
        self.config = config
        self.run_id: str | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._queued_input: str | None = None
        self._lock = threading.Lock()
        self.renderer = StreamRenderer()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_run(self, task: str) -> None:
        self.orchestrator._on_message = self.renderer.on_message

        def _work():
            try:
                self.run_id = asyncio.run(
                    self.orchestrator.start_run(task, persistent=True)
                )
            except Exception as e:
                self._error = e
            finally:
                self.renderer.finish()

        self._error = None
        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def continue_run(self, user_input: str) -> None:
        if self.run_id is None:
            return
        self.orchestrator._on_message = self.renderer.on_message

        def _work():
            try:
                asyncio.run(
                    self.orchestrator.continue_run(self.run_id, user_input)
                )
            except Exception as e:
                self._error = e
            finally:
                self.renderer.finish()

        self._error = None
        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def resolve_request(self, request_id: str, response_text: str, resolution: str = "answer") -> None:
        if self.run_id is None:
            return
        self.orchestrator._on_message = self.renderer.on_message

        def _work():
            try:
                asyncio.run(
                    self.orchestrator.resolve_request(
                        self.run_id,
                        request_id,
                        response_text,
                        resolution=resolution,
                    )
                )
            except Exception as e:
                self._error = e
            finally:
                self.renderer.finish()

        self._error = None
        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def queue_input(self, text: str) -> None:
        with self._lock:
            self._queued_input = text

    def pop_queued(self) -> str | None:
        with self._lock:
            q = self._queued_input
            self._queued_input = None
            return q

    def wait_done(self, timeout: float = 0.3) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def collect_error(self) -> Exception | None:
        e = self._error
        self._error = None
        return e


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--repo", type=click.Path(exists=True), default=".", help="Repository root")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def main(ctx: click.Context, repo: str, verbose: bool, model: str) -> None:
    """rari - recursive intelligence runtime"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = RuntimeConfig(repo_root=Path(repo).resolve())
    ctx.obj["model"] = model
    ctx.obj["verbose"] = verbose

    if ctx.invoked_subcommand is None:
        from recursive_intelligence.tui import run_tui
        run_tui(ctx.obj["config"], model)


@main.command()
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.option("--persistent", is_flag=True, help="Keep run alive for follow-up passes")
@click.pass_context
def run(ctx: click.Context, task: str, model: str, persistent: bool) -> None:
    """Start a new recursive run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)

    try:
        run_id = asyncio.run(orchestrator.start_run(task, persistent=persistent))
        _print_run_result(config, run_id)
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def baseline(ctx: click.Context, task: str, model: str) -> None:
    """Run a single flat Claude session (no recursion)."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)

    from recursive_intelligence.runtime.baseline import BaselineRunner

    adapter = _make_adapter(model)
    runner = BaselineRunner(config, adapter)

    try:
        report = asyncio.run(runner.run(task))
        click.echo(f"{INDENT}{_dim('status')}  {_s(report.status)}")
        click.echo(f"{INDENT}{_dim('cost')}    ${report.cost.total_usd:.4f}")
        click.echo(f"{INDENT}{_dim('turns')}   {report.num_turns}")
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@main.group()
@click.pass_context
def benchmark(ctx: click.Context) -> None:
    """Run benchmark suites."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)


@benchmark.command("swebench")
@click.option("--suite", default="tier-a", help="Suite to run")
@click.option("--dataset", default="SWE-bench/SWE-bench_Verified", help="Dataset name")
@click.option("--split", default="test", help="Dataset split")
@click.option("--limit", type=int, default=None, help="Optional task limit")
@click.option("--refresh", is_flag=True, help="Refresh the cached dataset")
@click.option("--keep-task-dirs", is_flag=True, help="Keep cloned task directories after the run")
@click.option("--model", default="claude-opus-4-6", help="Fallback model for both root and child nodes")
@click.option("--root-model", default=None, help="Claude model for baseline and recursive root nodes")
@click.option("--child-model", default=None, help="Claude model for recursive child nodes")
@click.option("--max-concurrency", type=int, default=2, show_default=True, help="Max concurrent benchmark tasks")
@click.option(
    "--namespace",
    default=None,
    help="Docker image namespace override for the official SWE-bench harness",
)
@click.pass_context
def benchmark_swebench(
    ctx: click.Context,
    suite: str,
    dataset: str,
    split: str,
    limit: int | None,
    refresh: bool,
    keep_task_dirs: bool,
    model: str,
    root_model: str | None,
    child_model: str | None,
    max_concurrency: int,
    namespace: str | None,
) -> None:
    """Benchmark recursive intelligence on SWE-bench."""
    from recursive_intelligence.benchmarks import BenchmarkRunner, SWEBenchLoader

    config: RuntimeConfig = ctx.obj["config"]
    loader = SWEBenchLoader(config.datasets_dir, dataset=dataset, split=split)

    try:
        tasks = loader.load_suite(suite, refresh=refresh)
        if limit is not None:
            tasks = tasks[:limit]

        runner = BenchmarkRunner(
            config,
            model=model,
            root_model=root_model,
            child_model=child_model,
            keep_task_dirs=keep_task_dirs,
            evaluation_namespace=namespace,
            max_concurrency=max_concurrency,
        )
        report = asyncio.run(
            runner.run_swebench_suite(
                tasks,
                suite=suite,
                dataset=dataset,
                split=split,
                requested_limit=limit,
            )
        )

        click.echo(f"{INDENT}{_dim('benchmark')} {report.run_id}")
        click.echo(f"{INDENT}{_dim('tasks')}      {report.task_count}")
        click.echo(
            f"{INDENT}{_dim('baseline')}   "
            f"{report.baseline.solved}/{report.baseline.eligible} solved"
            f"{_format_unsupported(report.baseline.unsupported)}"
        )
        click.echo(
            f"{INDENT}{_dim('recursive')}  "
            f"{report.recursive.solved}/{report.recursive.eligible} solved"
            f"{_format_unsupported(report.recursive.unsupported)}"
        )
        click.echo(f"{INDENT}{_dim('report')}     {config.benchmarks_dir / report.run_id / 'report.json'}")
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@benchmark.command("manifest")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--limit", type=int, default=None, help="Optional task limit")
@click.option("--refresh", is_flag=True, help="Refresh cached datasets while resolving the manifest")
@click.option("--keep-task-dirs", is_flag=True, help="Keep cloned task directories after the run")
@click.option("--model", default="claude-opus-4-6", help="Fallback model for both root and child nodes")
@click.option("--root-model", default=None, help="Claude model for baseline and recursive root nodes")
@click.option("--child-model", default=None, help="Claude model for recursive child nodes")
@click.option("--max-concurrency", type=int, default=2, show_default=True, help="Max concurrent benchmark tasks")
@click.option(
    "--namespace",
    default=None,
    help="Docker image namespace override for official harness-backed tasks",
)
@click.pass_context
def benchmark_manifest(
    ctx: click.Context,
    manifest_path: Path,
    limit: int | None,
    refresh: bool,
    keep_task_dirs: bool,
    model: str,
    root_model: str | None,
    child_model: str | None,
    max_concurrency: int,
    namespace: str | None,
) -> None:
    """Benchmark a frozen manifest that may span multiple datasets."""
    from recursive_intelligence.benchmarks import (
        BenchmarkRunner,
        load_benchmark_manifest,
        resolve_manifest_tasks,
    )

    config: RuntimeConfig = ctx.obj["config"]

    try:
        manifest = load_benchmark_manifest(manifest_path)
        tasks = resolve_manifest_tasks(manifest, config.datasets_dir, refresh=refresh)
        if limit is not None:
            tasks = tasks[:limit]

        runner = BenchmarkRunner(
            config,
            model=model,
            root_model=root_model,
            child_model=child_model,
            keep_task_dirs=keep_task_dirs,
            evaluation_namespace=namespace,
            max_concurrency=max_concurrency,
        )
        report = asyncio.run(
            runner.run_manifest_suite(
                tasks,
                suite=manifest.manifest_id,
                manifest_id=manifest.manifest_id,
                manifest_path=str(manifest_path.resolve()),
                requested_limit=limit,
            )
        )

        click.echo(f"{INDENT}{_dim('benchmark')} {report.run_id}")
        click.echo(f"{INDENT}{_dim('suite')}      {manifest.manifest_id}")
        click.echo(f"{INDENT}{_dim('tasks')}      {report.task_count}")
        click.echo(
            f"{INDENT}{_dim('baseline')}   "
            f"{report.baseline.solved}/{report.baseline.eligible} solved"
            f"{_format_unsupported(report.baseline.unsupported)}"
        )
        click.echo(
            f"{INDENT}{_dim('recursive')}  "
            f"{report.recursive.solved}/{report.recursive.eligible} solved"
            f"{_format_unsupported(report.recursive.unsupported)}"
        )
        click.echo(f"{INDENT}{_dim('report')}     {config.benchmarks_dir / report.run_id / 'report.json'}")
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@benchmark.command("build-manifest")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for the generated manifest",
)
@click.option("--manifest-id", default="recursive-hard-v1", show_default=True, help="Manifest identifier")
@click.option("--title", default="Recursive Hard v1", show_default=True, help="Manifest title")
@click.option("--refresh", is_flag=True, help="Refresh cached datasets while building the manifest")
@click.option("--swebench-pro-count", type=int, default=20, show_default=True, help="Tasks to select from SWE-Bench Pro")
@click.option("--featurebench-count", type=int, default=10, show_default=True, help="Tasks to select from FeatureBench")
@click.option("--repo-cap", type=int, default=2, show_default=True, help="Max tasks per repo within each source benchmark")
@click.pass_context
def benchmark_build_manifest(
    ctx: click.Context,
    output: Path | None,
    manifest_id: str,
    title: str,
    refresh: bool,
    swebench_pro_count: int,
    featurebench_count: int,
    repo_cap: int,
) -> None:
    """Build a deterministic recursive-hard manifest from supported datasets."""
    from recursive_intelligence.benchmarks import build_recursive_hard_manifest

    config: RuntimeConfig = ctx.obj["config"]
    output_path = output or (config.repo_root / "benchmarks" / "manifests" / f"{manifest_id}.json")

    try:
        manifest = build_recursive_hard_manifest(
            config.datasets_dir,
            output_path=output_path,
            manifest_id=manifest_id,
            title=title,
            swebench_pro_count=swebench_pro_count,
            featurebench_count=featurebench_count,
            repo_cap=repo_cap,
            refresh=refresh,
        )
        click.echo(f"{INDENT}{_dim('manifest')}  {manifest.manifest_id}")
        click.echo(f"{INDENT}{_dim('tasks')}      {len(manifest.tasks)}")
        click.echo(f"{INDENT}{_dim('output')}     {output_path}")
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@main.command("export-report")
@click.argument("run_id")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Destination directory for exported files",
)
@click.pass_context
def export_benchmark_report(ctx: click.Context, run_id: str, output_dir: Path | None) -> None:
    """Export a benchmark report as JSON and CSV."""
    from recursive_intelligence.benchmarks import export_report

    config: RuntimeConfig = ctx.obj["config"]
    report_path = config.benchmarks_dir / run_id / "report.json"
    if not report_path.exists():
        click.echo(f"{INDENT}{_red('error')} Benchmark report {run_id} not found", err=True)
        sys.exit(1)

    try:
        paths = export_report(report_path, output_dir)
        for path in paths:
            click.echo(f"{INDENT}{_dim('exported')} {path}")
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@main.command(name="continue")
@click.argument("run_id")
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def continue_run(ctx: click.Context, run_id: str, task: str, model: str) -> None:
    """Continue a paused persistent run with new instructions."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)

    try:
        asyncio.run(orchestrator.continue_run(run_id, task))
        _print_run_result(config, run_id)
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id")
@click.argument("request_id")
@click.argument("response", nargs=-1)
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def answer(ctx: click.Context, run_id: str, request_id: str, response: tuple[str, ...], model: str) -> None:
    """Answer a specific root-inbox request and continue the run."""
    _resolve_request_command(
        ctx,
        run_id,
        request_id,
        " ".join(response).strip(),
        resolution="answer",
        model=model,
    )


@main.command()
@click.argument("run_id")
@click.argument("request_id")
@click.argument("note", nargs=-1)
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def approve(ctx: click.Context, run_id: str, request_id: str, note: tuple[str, ...], model: str) -> None:
    """Approve a specific root-inbox request and continue the run."""
    _resolve_request_command(
        ctx,
        run_id,
        request_id,
        " ".join(note).strip(),
        resolution="approve",
        model=model,
    )


@main.command()
@click.argument("run_id")
@click.argument("request_id")
@click.argument("reason", nargs=-1)
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def decline(ctx: click.Context, run_id: str, request_id: str, reason: tuple[str, ...], model: str) -> None:
    """Decline a specific root-inbox request and continue the run."""
    _resolve_request_command(
        ctx,
        run_id,
        request_id,
        " ".join(reason).strip(),
        resolution="decline",
        model=model,
    )


@main.command()
@click.argument("run_id", required=False)
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def chat(ctx: click.Context, run_id: str | None, model: str) -> None:
    """Interactive session with a persistent run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=True, config=config)

    click.echo(BANNER, nl=False)

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)
    runner = BackgroundRunner(orchestrator, config)

    click.echo(f"{INDENT}{_dim('model')}  {model}")
    click.echo(f"{INDENT}{_dim('repo')}   {config.repo_root}")
    click.echo(f"{INDENT}{_dim('tips')}   /tree /domains /requests /inbox /work /status /help /quit")
    click.echo()

    if run_id:
        config.ensure_dirs()
        store = StateStore(config.db_path)
        run_record = store.get_run(run_id)
        store.close()
        if run_record is None:
            click.echo(f"{INDENT}{_red('error')} Run {run_id} not found", err=True)
            sys.exit(1)
        runner.run_id = run_id
        click.echo(f"{INDENT}{_dim('resuming')} {run_id} {_dim('pass')} {run_record.pass_count}")
        click.echo()

    # ── Main REPL loop ───────────────────────────────────────────────────
    while True:
        # If background is done, collect result
        if runner.busy:
            # Wait for background to finish — streaming happens via the
            # renderer callback, so we just poll completion
            while runner.busy:
                runner.wait_done(timeout=0.3)

            err = runner.collect_error()
            if err:
                click.echo(f"\n{INDENT}{_red('error')} {err}\n")
            elif runner.run_id:
                _print_pass_summary(config, runner.run_id)

            queued = runner.pop_queued()
            if queued:
                click.echo(f"\n{INDENT}{_dim('sending queued input...')}\n")
                runner.continue_run(queued)
                continue

        # Read input
        user_input = _read_input()
        if not user_input:
            continue

        if user_input.startswith("/"):
            if _handle_slash_command(user_input, config, runner):
                break
            continue

        if runner.busy:
            runner.queue_input(user_input)
            click.echo(f"{INDENT}{_dim('queued - will send after current pass')}")
            continue

        click.echo()

        if runner.run_id is None:
            runner.start_run(user_input)
        else:
            runner.continue_run(user_input)


@main.command()
@click.argument("run_id")
@click.pass_context
def resume(ctx: click.Context, run_id: str) -> None:
    """Resume a crashed/interrupted run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)

    adapter = _make_adapter()
    orchestrator = Orchestrator(config, adapter)

    try:
        asyncio.run(orchestrator.resume_run(run_id))
        click.echo(f"{INDENT}{_green('done')} Run resumed: {run_id}")
    except Exception as e:
        click.echo(f"{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id")
@click.pass_context
def tree(ctx: click.Context, run_id: str) -> None:
    """Display the node tree for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)
    config.ensure_dirs()
    store = StateStore(config.db_path)
    tree_data = get_node_tree(store, run_id)
    if tree_data:
        click.echo()
        _print_tree(tree_data[0])
        click.echo()
    else:
        click.echo(_dim(f"{INDENT}No nodes."))
    store.close()


@main.command()
@click.argument("node_id")
@click.pass_context
def inspect(ctx: click.Context, node_id: str) -> None:
    """Inspect a single node's state and events."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)
    config.ensure_dirs()
    store = StateStore(config.db_path)

    node = store.get_node(node_id)
    if node is None:
        click.echo(f"{INDENT}{_red('error')} Node {node_id} not found")
        store.close()
        return

    click.echo()
    click.echo(f"{INDENT}{_bold(node.node_id)}")
    click.echo(f"{INDENT}{_dim('state')}    {_s(node.state.value)}")
    click.echo(f"{INDENT}{_dim('task')}     {node.task_spec}")
    if node.worktree_path:
        click.echo(f"{INDENT}{_dim('worktree')} {node.worktree_path}")

    domain = store.get_domain_by_child(node_id)
    if domain:
        click.echo(f"{INDENT}{_dim('domain')}   {_magenta(domain.domain_name)}")
    request = store.get_latest_request(node_id)
    if request:
        payload = request["request"]
        click.echo(f"{INDENT}{_dim('request')}  {_blue(payload.get('kind', 'request'))}")
        summary = payload.get("summary") or payload.get("details", "")
        if summary:
            click.echo(f"{INDENT}{_dim('summary')}  {_truncate(summary, 88)}")
        if payload.get("action_requested"):
            click.echo(f"{INDENT}{_dim('action')}   {_truncate(payload['action_requested'], 88)}")
    downstream_task = store.get_latest_downstream_task(node_id)
    if downstream_task:
        task = downstream_task["task"]
        click.echo(f"{INDENT}{_dim('work')}     {_magenta(task.get('kind', 'task'))}")
        summary = task.get("summary") or task.get("task_spec", "")
        if summary:
            click.echo(f"{INDENT}{_dim('summary')}  {_truncate(summary, 88)}")
        if task.get("task_spec"):
            click.echo(f"{INDENT}{_dim('details')}  {_truncate(task['task_spec'], 88)}")

    events = store.get_node_events(node_id)
    if events:
        click.echo(f"\n{INDENT}{_dim('events')} ({len(events)})")
        for evt in events:
            ts = evt.timestamp.split("T")[1][:8] if "T" in evt.timestamp else evt.timestamp
            click.echo(f"{INDENT}  {_dim(ts)} {evt.event_type}")

    children = store.get_children(node_id)
    if children:
        click.echo(f"\n{INDENT}{_dim('children')} ({len(children)})")
        for child in children:
            d = store.get_domain_by_child(child.node_id)
            dtag = f" {_magenta(d.domain_name)}" if d else ""
            click.echo(f"{INDENT}  {_s(child.state.value):>20s}{dtag}  {child.task_spec[:50]}")

    click.echo()
    store.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def domains(ctx: click.Context, run_id: str) -> None:
    """Show the domain registry for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)
    config.ensure_dirs()
    store = StateStore(config.db_path)

    run_record = store.get_run(run_id)
    if run_record is None or run_record.root_node_id is None:
        click.echo(f"{INDENT}{_red('error')} Run {run_id} not found")
        store.close()
        return

    domain_list = store.get_domains(run_record.root_node_id)
    if not domain_list:
        click.echo(_dim(f"{INDENT}No domains."))
        store.close()
        return

    click.echo()
    for d in domain_list:
        child = store.get_node(d.child_node_id)
        state = child.state.value if child else "unknown"
        click.echo(f"{INDENT}{_magenta(d.domain_name)}")
        click.echo(f"{INDENT}  {_dim('node')}   {d.child_node_id[:16]}  {_s(state)}")
        if d.file_patterns:
            click.echo(f"{INDENT}  {_dim('files')}  {', '.join(d.file_patterns)}")
        click.echo()
    store.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def blockers(ctx: click.Context, run_id: str) -> None:
    """Show active blockers and root escalations for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)
    config.ensure_dirs()
    store = StateStore(config.db_path)

    run_record = store.get_run(run_id)
    if run_record is None:
        click.echo(f"{INDENT}{_red('error')} Run {run_id} not found")
        store.close()
        return

    click.echo()
    _print_blockers(store, run_id)
    click.echo()
    store.close()


@main.command("requests")
@click.argument("run_id")
@click.pass_context
def requests_cmd(ctx: click.Context, run_id: str) -> None:
    """Show active upstream requests for a run."""
    blockers(ctx, run_id)


@main.command()
@click.argument("run_id")
@click.pass_context
def inbox(ctx: click.Context, run_id: str) -> None:
    """Show unresolved root-inbox requests for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)
    config.ensure_dirs()
    store = StateStore(config.db_path)

    run_record = store.get_run(run_id)
    if run_record is None:
        click.echo(f"{INDENT}{_red('error')} Run {run_id} not found")
        store.close()
        return

    click.echo()
    _print_inbox(store, run_id)
    click.echo()
    store.close()


@main.command("work-items")
@click.argument("run_id")
@click.pass_context
def work_items(ctx: click.Context, run_id: str) -> None:
    """Show unresolved downstream work items for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)
    config.ensure_dirs()
    store = StateStore(config.db_path)

    run_record = store.get_run(run_id)
    if run_record is None:
        click.echo(f"{INDENT}{_red('error')} Run {run_id} not found")
        store.close()
        return

    click.echo()
    _print_work_items(store, run_id)
    click.echo()
    store.close()


# ── Adapter factory ──────────────────────────────────────────────────────────

def _make_adapter(model: str = "claude-sonnet-4-6"):
    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter
    return ClaudeAdapter(model=model)


def _resolution_text(resolution: str, text: str) -> str:
    if text.strip():
        return text.strip()
    if resolution == "approve":
        return "Approved."
    if resolution == "decline":
        return "Declined."
    raise click.UsageError("A response is required")


def _resolve_request_command(
    ctx: click.Context,
    run_id: str,
    request_id: str,
    response_text: str,
    resolution: str,
    model: str,
) -> None:
    config: RuntimeConfig = ctx.obj["config"]
    _setup_logging(ctx.obj["verbose"], chat_mode=False, config=config)

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)

    try:
        normalized = _resolution_text(resolution, response_text)
        asyncio.run(
            orchestrator.resolve_request(
                run_id,
                request_id,
                normalized,
                resolution=resolution,
            )
        )
        _print_run_result(config, run_id)
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"\n{INDENT}{_red('error')} {e}", err=True)
        sys.exit(1)


# ── Output helpers ───────────────────────────────────────────────────────────

def _print_pass_summary(config: RuntimeConfig, run_id: str) -> None:
    """Compact summary after a pass completes."""
    store = StateStore(config.db_path)
    run_record = store.get_run(run_id)
    nodes = store.get_run_nodes(run_id)

    if not run_record:
        store.close()
        return

    completed = sum(1 for n in nodes if n.state == NodeState.COMPLETED)
    failed = sum(1 for n in nodes if n.state == NodeState.FAILED)
    status = run_record.status
    blockers = store.get_run_blockers(run_id)
    inbox = store.get_run_inbox(run_id)
    work_items = store.get_run_downstream_tasks(run_id)
    store.close()

    click.echo()
    click.echo(f"{INDENT}{_s(status)}  {_dim(f'{len(nodes)} nodes  {completed} done')}" +
               (f"  {_red(f'{failed} failed')}" if failed else ""))
    if blockers:
        click.echo(f"{INDENT}{_yellow('blocked')} {_dim(f'{len(blockers)} active blocker(s)')}")
        _print_blocker_lines(blockers, limit=2)
    if inbox:
        click.echo(f"{INDENT}{_blue('inbox')} {_dim(f'{len(inbox)} unresolved request(s)')}")
        _print_inbox_lines(inbox, limit=2)
    if work_items:
        click.echo(f"{INDENT}{_magenta('work')} {_dim(f'{len(work_items)} downstream task(s)')}")
        _print_downstream_task_lines(work_items, limit=2)
    click.echo()

    if status == "paused":
        click.echo(f"{INDENT}{_dim('type your next instructions, or /work to inspect pending requests and tasks')}")
        click.echo()


def _print_run_result(config: RuntimeConfig, run_id: str) -> None:
    """Full run result (for non-chat commands)."""
    store = StateStore(config.db_path)
    tree_data = get_node_tree(store, run_id)
    run_record = store.get_run(run_id)
    nodes = store.get_run_nodes(run_id)

    status = run_record.status if run_record else "unknown"
    pass_num = run_record.pass_count if run_record else 1
    blockers = store.get_run_blockers(run_id) if run_record else []
    inbox = store.get_run_inbox(run_id) if run_record else []
    work_items = store.get_run_downstream_tasks(run_id) if run_record else []
    store.close()

    click.echo(_hr())
    click.echo()
    click.echo(f"{INDENT}{_dim('run')}    {run_id}")
    click.echo(f"{INDENT}{_dim('status')} {_s(status)}  {_dim('pass')} {pass_num}  {_dim('nodes')} {len(nodes)}")
    if run_record:
        click.echo(
            f"{INDENT}{_dim('telemetry')} "
            f"{run_record.telemetry.get('user_interruptions_count', 0)} interrupts  "
            f"{run_record.telemetry.get('root_requests_count', 0)} requests  "
            f"{run_record.telemetry.get('resolved_requests_count', 0)} resolved"
        )
    if inbox:
        click.echo(f"{INDENT}{_blue('inbox')} {len(inbox)} unresolved")
        _print_inbox_lines(inbox, limit=3)
    if work_items:
        click.echo(f"{INDENT}{_magenta('work')} {len(work_items)} downstream")
        _print_downstream_task_lines(work_items, limit=3)
    if blockers:
        click.echo(f"{INDENT}{_yellow('blockers')} {len(blockers)} active")
        _print_blocker_lines(blockers, limit=3)

    if tree_data:
        click.echo()
        _print_tree(tree_data[0])
    click.echo()


def _print_tree(node: dict, prefix: str = INDENT, is_last: bool = True) -> None:
    connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
    state = node["state"]
    icon = {"completed": _green("+"), "failed": _red("x"),
            "cancelled": _dim("-"), "paused": _yellow("||")}.get(state, _cyan("~"))
    dtag = f" {_magenta(node['domain'])}" if node.get("domain") else ""
    click.echo(f"{prefix}{connector}{icon} {_dim(node['node_id'][:12])}{dtag} {node['task_spec'][:55]}")

    children = node.get("children", [])
    for i, child in enumerate(children):
        ext = "   " if is_last else "\u2502  "
        _print_tree(child, prefix + ext, i == len(children) - 1)


def _print_live_tree(config: RuntimeConfig, run_id: str) -> None:
    try:
        store = StateStore(config.db_path)
        tree_data = get_node_tree(store, run_id)
        nodes = store.get_run_nodes(run_id)
        blockers = store.get_run_blockers(run_id)
        inbox = store.get_run_inbox(run_id)
        work_items = store.get_run_downstream_tasks(run_id)
        store.close()
    except Exception:
        click.echo(_dim(f"{INDENT}(could not read state)"))
        return

    if not tree_data:
        click.echo(_dim(f"{INDENT}No nodes yet."))
        return

    active = sum(1 for n in nodes if not n.state.is_idle)
    done = sum(1 for n in nodes if n.state == NodeState.COMPLETED)
    line = f"{INDENT}{_dim('nodes')} {len(nodes)}  {_green(f'{done} done')}  {_cyan(f'{active} active')}"
    if blockers:
        line += f"  {_yellow(f'{len(blockers)} blocked')}"
    if inbox:
        line += f"  {_blue(f'{len(inbox)} inbox')}"
    if work_items:
        line += f"  {_magenta(f'{len(work_items)} work')}"
    click.echo(line)
    click.echo()
    _print_tree(tree_data[0])
    if work_items:
        _print_downstream_task_lines(work_items, limit=2)
        click.echo()
    if blockers:
        _print_blocker_lines(blockers, limit=2)
        click.echo()
    click.echo()


def _read_input() -> str:
    try:
        return click.prompt(f"{INDENT}{BOLD}{CYAN}\u276f{RESET}", prompt_suffix=" ").strip()
    except (click.Abort, EOFError):
        return ""


def _parse_request_action(command: str) -> tuple[str, str, str] | None:
    raw = command.strip()
    name, _, remainder = raw.partition(" ")
    action = name.lower()
    if action not in {"/answer", "/approve", "/decline"}:
        return None

    request_id, _, response_text = remainder.strip().partition(" ")
    if not request_id:
        return action, "", ""
    return action, request_id, response_text.strip()


def _handle_slash_command(cmd: str, config: RuntimeConfig, runner: BackgroundRunner) -> bool:
    raw = cmd.strip()
    name, _, _ = raw.partition(" ")
    cmd = name.lower()
    run_id = runner.run_id

    if cmd in ("/quit", "/exit", "/q"):
        if run_id:
            click.echo(f"\n{INDENT}{_dim('run paused. resume with:')} rari chat {run_id}")
        click.echo()
        return True

    if cmd == "/done":
        if run_id:
            config.ensure_dirs()
            store = StateStore(config.db_path)
            store.finish_run(run_id, "completed")
            store.close()
        click.echo(f"\n{INDENT}{_green('done')} run finalized.\n")
        return True

    if cmd in ("/tree", "/t"):
        click.echo()
        if run_id:
            _print_live_tree(config, run_id)
        else:
            click.echo(_dim(f"{INDENT}no active run."))
        return False

    if cmd in ("/domains", "/d"):
        click.echo()
        if run_id:
            config.ensure_dirs()
            store = StateStore(config.db_path)
            rr = store.get_run(run_id)
            if rr and rr.root_node_id:
                dl = store.get_domains(rr.root_node_id)
                for d in dl:
                    ch = store.get_node(d.child_node_id)
                    st = ch.state.value if ch else "?"
                    click.echo(f"{INDENT}{_magenta(d.domain_name)}  {_s(st)}  {_dim(d.child_node_id[:12])}")
                if not dl:
                    click.echo(_dim(f"{INDENT}no domains."))
            store.close()
        else:
            click.echo(_dim(f"{INDENT}no active run."))
        click.echo()
        return False

    request_action = _parse_request_action(raw)
    if request_action is not None:
        action, request_id, response_text = request_action
        if not run_id:
            click.echo(f"\n{INDENT}{_dim('no active run.')}\n")
            return False
        if runner.busy:
            click.echo(f"\n{INDENT}{_dim('cannot resolve a request while work is still running')}\n")
            return False
        if not request_id:
            click.echo(f"\n{INDENT}{_dim(f'usage: {action} <request_id> <response>')}\n")
            return False
        resolution = {
            "/answer": "answer",
            "/approve": "approve",
            "/decline": "decline",
        }[action]
        try:
            normalized = _resolution_text(resolution, response_text)
        except click.ClickException as e:
            click.echo(f"\n{INDENT}{_red('error')} {e.format_message()}\n")
            return False
        click.echo()
        runner.resolve_request(request_id, normalized, resolution=resolution)
        return False

    if cmd in ("/blockers", "/b", "/requests", "/r"):
        click.echo()
        if run_id:
            try:
                config.ensure_dirs()
                store = StateStore(config.db_path)
                _print_blockers(store, run_id)
                store.close()
            except Exception:
                click.echo(_dim(f"{INDENT}(could not read blockers)"))
        else:
            click.echo(_dim(f"{INDENT}no active run."))
        click.echo()
        return False

    if cmd in ("/inbox", "/i"):
        click.echo()
        if run_id:
            try:
                config.ensure_dirs()
                store = StateStore(config.db_path)
                _print_inbox(store, run_id)
                store.close()
            except Exception:
                click.echo(_dim(f"{INDENT}(could not read inbox)"))
        else:
            click.echo(_dim(f"{INDENT}no active run."))
        click.echo()
        return False

    if cmd in ("/work", "/tasks", "/w"):
        click.echo()
        if run_id:
            try:
                config.ensure_dirs()
                store = StateStore(config.db_path)
                _print_work_items(store, run_id)
                store.close()
            except Exception:
                click.echo(_dim(f"{INDENT}(could not read downstream work)"))
        else:
            click.echo(_dim(f"{INDENT}no active run."))
        click.echo()
        return False

    if cmd in ("/status", "/s"):
        click.echo()
        if run_id:
            try:
                store = StateStore(config.db_path)
                rr = store.get_run(run_id)
                nodes = store.get_run_nodes(run_id)
                blockers = store.get_run_blockers(run_id)
                inbox = store.get_run_inbox(run_id)
                work_items = store.get_run_downstream_tasks(run_id)
                store.close()
                active = [n for n in nodes if not n.state.is_idle]
                click.echo(f"{INDENT}{_dim('run')}    {run_id}")
                click.echo(f"{INDENT}{_dim('status')} {_s(rr.status)}  {_dim('nodes')} {len(nodes)}  {_cyan(f'{len(active)} active')}")
                click.echo(
                    f"{INDENT}{_dim('telemetry')} "
                    f"{rr.telemetry.get('user_interruptions_count', 0)} interrupts  "
                    f"{rr.telemetry.get('root_requests_count', 0)} requests  "
                    f"{rr.telemetry.get('resolved_requests_count', 0)} resolved"
                )
                if inbox:
                    click.echo(f"{INDENT}{_blue('inbox')} {len(inbox)} unresolved")
                    _print_inbox_lines(inbox, limit=2)
                if work_items:
                    click.echo(f"{INDENT}{_magenta('work')} {len(work_items)} downstream")
                    _print_downstream_task_lines(work_items, limit=2)
                if blockers:
                    click.echo(f"{INDENT}{_yellow('blockers')} {len(blockers)} active")
                    _print_blocker_lines(blockers, limit=2)
                if runner.busy:
                    click.echo(f"{INDENT}{_cyan('working...')}")
            except Exception:
                click.echo(_dim(f"{INDENT}(could not read state)"))
        else:
            click.echo(_dim(f"{INDENT}no active run."))
        click.echo()
        return False

    if cmd in ("/log", "/logs"):
        log_path = config.ri_dir / "rari.log"
        click.echo(f"\n{INDENT}{_dim('log:')} {log_path}\n")
        return False

    if cmd == "/help":
        click.echo()
        click.echo(f"{INDENT}{_bold('/tree')}     {_dim('show the node tree (live)')}")
        click.echo(f"{INDENT}{_bold('/status')}   {_dim('show run status')}")
        click.echo(f"{INDENT}{_bold('/domains')}  {_dim('show domain registry')}")
        click.echo(f"{INDENT}{_bold('/inbox')}    {_dim('show unresolved root-inbox requests')}")
        click.echo(f"{INDENT}{_bold('/work')}     {_dim('show unresolved downstream work items')}")
        click.echo(f"{INDENT}{_bold('/answer')}   {_dim('answer a request: /answer <id> <text>')}")
        click.echo(f"{INDENT}{_bold('/approve')}  {_dim('approve a request: /approve <id> [note]')}")
        click.echo(f"{INDENT}{_bold('/decline')}  {_dim('decline a request: /decline <id> [reason]')}")
        click.echo(f"{INDENT}{_bold('/requests')} {_dim('show active upstream requests')}")
        click.echo(f"{INDENT}{_bold('/blockers')} {_dim('legacy alias for /requests')}")
        click.echo(f"{INDENT}{_bold('/log')}      {_dim('show log file path')}")
        click.echo(f"{INDENT}{_bold('/done')}     {_dim('finalize and close the run')}")
        click.echo(f"{INDENT}{_bold('/quit')}     {_dim('exit (run stays paused)')}")
        click.echo()
        return False

    click.echo(f"{INDENT}{_dim('unknown command. try /help')}")
    return False


def _print_blockers(store: StateStore, run_id: str) -> None:
    run_record = store.get_run(run_id)
    blockers = store.get_run_blockers(run_id)
    if run_record:
        click.echo(f"{INDENT}{_dim('run')}      {run_id}")
        click.echo(
            f"{INDENT}{_dim('telemetry')} "
            f"{run_record.telemetry.get('user_interruptions_count', 0)} interrupts  "
            f"{run_record.telemetry.get('root_requests_count', 0)} requests  "
            f"{run_record.telemetry.get('resolved_requests_count', 0)} resolved"
        )
    if not blockers:
        click.echo(_dim(f"{INDENT}No active blockers."))
        return
    click.echo(f"{INDENT}{_yellow('active blockers')} {len(blockers)}")
    _print_blocker_lines(blockers, limit=None)


def _print_inbox(store: StateStore, run_id: str) -> None:
    run_record = store.get_run(run_id)
    inbox = store.get_run_inbox(run_id)
    if run_record:
        click.echo(f"{INDENT}{_dim('run')}      {run_id}")
        click.echo(
            f"{INDENT}{_dim('telemetry')} "
            f"{run_record.telemetry.get('root_requests_count', 0)} requests  "
            f"{run_record.telemetry.get('resolved_requests_count', 0)} resolved"
        )
    if not inbox:
        click.echo(_dim(f"{INDENT}Inbox is empty."))
        return
    click.echo(f"{INDENT}{_blue('root inbox')} {len(inbox)} unresolved")
    _print_inbox_lines(inbox, limit=None)
    click.echo(f"{INDENT}{_dim('use /answer, /approve, or /decline with a request id to resolve one')}")


def _print_work_items(store: StateStore, run_id: str) -> None:
    run_record = store.get_run(run_id)
    work_items = store.get_run_downstream_tasks(run_id)
    if run_record:
        click.echo(f"{INDENT}{_dim('run')}      {run_id}")
    if not work_items:
        click.echo(_dim(f"{INDENT}No unresolved downstream work items."))
        return
    click.echo(f"{INDENT}{_magenta('downstream work')} {len(work_items)} unresolved")
    _print_downstream_task_lines(work_items, limit=None)


def _print_blocker_lines(blockers: list[dict[str, Any]], limit: int | None) -> None:
    subset = blockers[:limit] if limit is not None else blockers
    for blocker_entry in subset:
        blocker = blocker_entry["blocker"]
        escalation = blocker.get("escalation", {})
        title = escalation.get("summary") or blocker.get("details") or blocker.get("kind", "blocked")
        domain = blocker_entry.get("domain_name")
        domain_tag = f" {_magenta(domain)}" if domain else ""
        click.echo(
            f"{INDENT}  {_yellow('!')} {_s(blocker_entry['state'])}{domain_tag} "
            f"{_dim(blocker_entry['node_id'][:12])} {_truncate(title, 72)}"
        )
        action_requested = escalation.get("action_requested")
        if action_requested:
            click.echo(f"{INDENT}    {_dim('action')} {_truncate(action_requested, 76)}")
    if limit is not None and len(blockers) > limit:
        click.echo(f"{INDENT}    {_dim(f'+{len(blockers) - limit} more blocker(s)')}")


def _print_inbox_lines(inbox: list[dict[str, Any]], limit: int | None) -> None:
    subset = inbox[:limit] if limit is not None else inbox
    for item in subset:
        request = item["request"]
        domain = item.get("domain_name")
        domain_tag = f" {_magenta(domain)}" if domain else ""
        click.echo(
            f"{INDENT}  {_blue('>')} {request.get('kind', 'request')}{domain_tag} "
            f"{_dim(item['request_id'])} {_truncate(request.get('summary', ''), 72)}"
        )
        if request.get("action_requested"):
            click.echo(f"{INDENT}    {_dim('action')} {_truncate(request['action_requested'], 76)}")
    if limit is not None and len(inbox) > limit:
        click.echo(f"{INDENT}    {_dim(f'+{len(inbox) - limit} more inbox item(s)')}")


def _print_downstream_task_lines(work_items: list[dict[str, Any]], limit: int | None) -> None:
    subset = work_items[:limit] if limit is not None else work_items
    for item in subset:
        task = item["task"]
        domain = item.get("domain_name")
        domain_tag = f" {_magenta(domain)}" if domain else ""
        summary = task.get("summary") or task.get("task_spec", "")
        click.echo(
            f"{INDENT}  {_magenta('>')} {task.get('kind', 'task')}{domain_tag} "
            f"{_dim(item['node_id'][:12])} {_truncate(summary, 72)}"
        )
        if task.get("task_spec") and task.get("task_spec") != summary:
            click.echo(f"{INDENT}    {_dim('details')} {_truncate(task['task_spec'], 76)}")
    if limit is not None and len(work_items) > limit:
        click.echo(f"{INDENT}    {_dim(f'+{len(work_items) - limit} more work item(s)')}")
