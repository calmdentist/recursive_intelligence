"""CLI entrypoint for the recursive intelligence runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

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
}


def _styled_state(state: str) -> str:
    return STATE_STYLE.get(state, state)


def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def _bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def _cyan(text: str) -> str:
    return f"{CYAN}{text}{RESET}"


def _green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def _yellow(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


def _red(text: str) -> str:
    return f"{RED}{text}{RESET}"


def _magenta(text: str) -> str:
    return f"{MAGENTA}{text}{RESET}"


def _cols() -> int:
    try:
        return min(os.get_terminal_size().columns, 100)
    except OSError:
        return 80


def _hr() -> str:
    return _dim("\u2500" * _cols())


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


def _print_banner() -> None:
    click.echo(BANNER, nl=False)


def _print_chat_help() -> None:
    click.echo(f"  {_dim('Commands:')} /tree  /domains  /status  /help  /done  /quit")
    click.echo()


PROMPT = f"  {BOLD}{CYAN}\u276f{RESET} "


# ── Background runner ────────────────────────────────────────────────────────

class BackgroundRunner:
    """Runs orchestrator passes in a background thread.

    The prompt stays live while the orchestrator works. Slash commands
    read from the SQLite DB which is updated in real-time.
    """

    def __init__(self, orchestrator: Orchestrator, config: RuntimeConfig) -> None:
        self.orchestrator = orchestrator
        self.config = config
        self.run_id: str | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._queued_input: str | None = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_run(self, task: str) -> None:
        """Start a new persistent run in the background."""
        def _work():
            try:
                self.run_id = asyncio.run(
                    self.orchestrator.start_run(task, persistent=True)
                )
            except Exception as e:
                self._error = e

        self._error = None
        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def continue_run(self, user_input: str) -> None:
        """Continue a paused run in the background."""
        if self.run_id is None:
            return

        def _work():
            try:
                asyncio.run(
                    self.orchestrator.continue_run(self.run_id, user_input)
                )
            except Exception as e:
                self._error = e

        self._error = None
        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def queue_input(self, text: str) -> None:
        """Queue input to be sent after the current pass finishes."""
        with self._lock:
            self._queued_input = text

    def pop_queued(self) -> str | None:
        with self._lock:
            q = self._queued_input
            self._queued_input = None
            return q

    def wait_done(self, timeout: float = 0.3) -> bool:
        """Wait briefly for the background task. Returns True if done."""
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def collect_error(self) -> Exception | None:
        e = self._error
        self._error = None
        return e

    def get_status_line(self) -> str:
        """Build a one-line status from live DB state."""
        if not self.run_id:
            return ""
        try:
            store = StateStore(self.config.db_path)
            nodes = store.get_run_nodes(self.run_id)
            store.close()
        except Exception:
            return ""

        active = [n for n in nodes if not n.state.is_idle]
        if not active:
            return ""

        parts = []
        for n in active[:3]:
            domain = ""
            task = n.task_spec[:25]
            parts.append(f"{_styled_state(n.state.value)} {_dim(task)}")

        suffix = f" {_dim(f'+{len(active)-3} more')}" if len(active) > 3 else ""
        return "  " + "  ".join(parts) + suffix


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--repo", type=click.Path(exists=True), default=".", help="Repository root")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def main(ctx: click.Context, repo: str, verbose: bool, model: str) -> None:
    """rari - recursive intelligence runtime"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    ctx.ensure_object(dict)
    ctx.obj["config"] = RuntimeConfig(repo_root=Path(repo).resolve())
    ctx.obj["model"] = model

    if ctx.invoked_subcommand is None:
        ctx.invoke(chat, run_id=None, model=model)


@main.command()
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.option("--persistent", is_flag=True, help="Keep run alive for follow-up passes")
@click.option("--cleanup/--no-cleanup", default=False, help="Remove worktrees after run")
@click.pass_context
def run(ctx: click.Context, task: str, model: str, persistent: bool, cleanup: bool) -> None:
    """Start a new recursive run."""
    config: RuntimeConfig = ctx.obj["config"]
    _print_banner()

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)

    click.echo(f"  {_dim('model')}  {model}")
    click.echo(f"  {_dim('repo')}   {config.repo_root}")
    click.echo(f"  {_dim('mode')}   {'persistent' if persistent else 'one-shot'}")
    click.echo()

    try:
        run_id = asyncio.run(orchestrator.start_run(task, persistent=persistent))
        _print_run_result(config, run_id)

        if cleanup:
            orchestrator.cleanup_worktrees(run_id)
            click.echo(_dim("  Worktrees cleaned up."))
    except Exception as e:
        click.echo(f"\n  {_red('error')} {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def baseline(ctx: click.Context, task: str, model: str) -> None:
    """Run a single flat Claude session (no recursion)."""
    config: RuntimeConfig = ctx.obj["config"]
    _print_banner()

    from recursive_intelligence.runtime.baseline import BaselineRunner

    adapter = _make_adapter(model)
    runner = BaselineRunner(config, adapter)

    click.echo(f"  {_dim('mode')}   baseline (flat, no recursion)")
    click.echo()

    try:
        report = asyncio.run(runner.run(task))
        click.echo()
        click.echo(f"  {_dim('run')}      {report.run_id}")
        click.echo(f"  {_dim('status')}   {_styled_state(report.status)}")
        click.echo(f"  {_dim('cost')}     ${report.cost.total_usd:.4f}")
        click.echo(f"  {_dim('turns')}    {report.num_turns}")
        click.echo(f"  {_dim('time')}     {report.duration_ms}ms")
        if report.changed_files:
            click.echo(f"  {_dim('changed')}  {', '.join(report.changed_files)}")
        click.echo()
    except Exception as e:
        click.echo(f"\n  {_red('error')} {e}", err=True)
        sys.exit(1)


@main.command(name="continue")
@click.argument("run_id")
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def continue_run(ctx: click.Context, run_id: str, task: str, model: str) -> None:
    """Continue a paused persistent run with new instructions."""
    config: RuntimeConfig = ctx.obj["config"]

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)

    try:
        asyncio.run(orchestrator.continue_run(run_id, task))
        _print_run_result(config, run_id)
    except Exception as e:
        click.echo(f"\n  {_red('error')} {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id", required=False)
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def chat(ctx: click.Context, run_id: str | None, model: str) -> None:
    """Interactive session with a persistent run."""
    config: RuntimeConfig = ctx.obj["config"]
    _print_banner()

    adapter = _make_adapter(model)
    orchestrator = Orchestrator(config, adapter)
    runner = BackgroundRunner(orchestrator, config)

    click.echo(f"  {_dim('model')}  {model}")
    click.echo(f"  {_dim('repo')}   {config.repo_root}")
    click.echo()

    if run_id:
        config.ensure_dirs()
        store = StateStore(config.db_path)
        run_record = store.get_run(run_id)
        store.close()
        if run_record is None:
            click.echo(f"  {_red('error')} Run {run_id} not found", err=True)
            sys.exit(1)
        runner.run_id = run_id
        click.echo(f"  {_dim('resuming')} {run_id} {_dim('(pass')} {run_record.pass_count}{_dim(')')}")
        click.echo()
        _print_chat_help()
    else:
        _print_chat_help()

    # ── Main REPL loop ───────────────────────────────────────────────────
    while True:
        # If busy, show status and poll for completion
        if runner.busy:
            _wait_with_status(runner, config)
            err = runner.collect_error()
            if err:
                click.echo(f"  {_red('error')} {err}")
            elif runner.run_id:
                _print_run_result(config, runner.run_id)

            # Check for queued input
            queued = runner.pop_queued()
            if queued:
                click.echo(f"  {_dim('sending queued input...')}")
                click.echo()
                runner.continue_run(queued)
                continue

        # Read input
        user_input = _read_input()
        if not user_input:
            continue

        # Slash commands work any time
        if user_input.startswith("/"):
            if _handle_slash_command(user_input, config, runner):
                break
            continue

        # If orchestrator is busy, queue the input
        if runner.busy:
            runner.queue_input(user_input)
            click.echo(f"  {_dim('Queued. Will send after current pass completes.')}")
            continue

        click.echo()

        # First message starts a new run
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

    adapter = _make_adapter()
    orchestrator = Orchestrator(config, adapter)

    try:
        asyncio.run(orchestrator.resume_run(run_id))
        click.echo(f"  {_green('done')} Run resumed: {run_id}")
    except Exception as e:
        click.echo(f"  {_red('error')} {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id")
@click.pass_context
def tree(ctx: click.Context, run_id: str) -> None:
    """Display the node tree for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    config.ensure_dirs()
    store = StateStore(config.db_path)

    tree_data = get_node_tree(store, run_id)
    if not tree_data:
        click.echo(_dim("  No nodes."))
        store.close()
        return

    click.echo()
    _print_tree(tree_data[0])
    click.echo()
    store.close()


@main.command()
@click.argument("node_id")
@click.pass_context
def inspect(ctx: click.Context, node_id: str) -> None:
    """Inspect a single node's state and events."""
    config: RuntimeConfig = ctx.obj["config"]
    config.ensure_dirs()
    store = StateStore(config.db_path)

    node = store.get_node(node_id)
    if node is None:
        click.echo(f"  {_red('error')} Node {node_id} not found")
        store.close()
        return

    click.echo()
    click.echo(f"  {_bold(node.node_id)}")
    click.echo(f"  {_dim('run')}      {node.run_id}")
    click.echo(f"  {_dim('parent')}   {node.parent_id or _dim('(root)')}")
    click.echo(f"  {_dim('state')}    {_styled_state(node.state.value)}")
    click.echo(f"  {_dim('task')}     {node.task_spec}")

    if node.worktree_path:
        click.echo(f"  {_dim('worktree')} {node.worktree_path}")
    if node.branch_name:
        click.echo(f"  {_dim('branch')}   {node.branch_name}")

    domain = store.get_domain_by_child(node_id)
    if domain:
        click.echo(f"  {_dim('domain')}   {_magenta(domain.domain_name)}")
        if domain.file_patterns:
            click.echo(f"  {_dim('files')}    {', '.join(domain.file_patterns)}")
        if domain.module_scope:
            click.echo(f"  {_dim('scope')}    {domain.module_scope}")

    events = store.get_node_events(node_id)
    if events:
        click.echo(f"\n  {_dim('events')} ({len(events)})")
        for evt in events:
            ts = evt.timestamp.split("T")[1][:8] if "T" in evt.timestamp else evt.timestamp
            click.echo(f"    {_dim(ts)} {evt.event_type}")

    children = store.get_children(node_id)
    if children:
        click.echo(f"\n  {_dim('children')} ({len(children)})")
        for child in children:
            d = store.get_domain_by_child(child.node_id)
            domain_tag = f" {_magenta(d.domain_name)}" if d else ""
            click.echo(f"    {_styled_state(child.state.value):>20s}{domain_tag}  {child.task_spec[:50]}")

    click.echo()
    store.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def domains(ctx: click.Context, run_id: str) -> None:
    """Show the domain registry for a run."""
    config: RuntimeConfig = ctx.obj["config"]
    config.ensure_dirs()
    store = StateStore(config.db_path)

    run_record = store.get_run(run_id)
    if run_record is None or run_record.root_node_id is None:
        click.echo(f"  {_red('error')} Run {run_id} not found")
        store.close()
        return

    domain_list = store.get_domains(run_record.root_node_id)
    if not domain_list:
        click.echo(_dim("  No domains registered."))
        store.close()
        return

    click.echo()
    click.echo(f"  {_dim('run')}  {run_id}  {_dim('pass')} {run_record.pass_count}")
    click.echo()

    for d in domain_list:
        child = store.get_node(d.child_node_id)
        state = child.state.value if child else "unknown"
        click.echo(f"  {_magenta(d.domain_name)}")
        click.echo(f"    {_dim('node')}   {d.child_node_id[:16]}  {_styled_state(state)}")
        if d.file_patterns:
            click.echo(f"    {_dim('files')}  {', '.join(d.file_patterns)}")
        if d.module_scope:
            click.echo(f"    {_dim('scope')}  {d.module_scope}")
        click.echo()

    store.close()


# ── Adapter factory ──────────────────────────────────────────────────────────

def _make_adapter(model: str = "claude-sonnet-4-6"):
    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter
    return ClaudeAdapter(model=model)


# ── Output helpers ───────────────────────────────────────────────────────────

def _wait_with_status(runner: BackgroundRunner, config: RuntimeConfig) -> None:
    """Poll the background runner, showing a live status line."""
    spinner = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
    i = 0
    while runner.busy:
        frame = _cyan(spinner[i % len(spinner)])
        status = runner.get_status_line()
        sys.stderr.write(f"{CLEAR_LINE}  {frame} {_dim('working...')} {status}")
        sys.stderr.flush()
        i += 1
        if runner.wait_done(timeout=0.4):
            break
    sys.stderr.write(CLEAR_LINE)
    sys.stderr.flush()


def _print_run_result(config: RuntimeConfig, run_id: str) -> None:
    """Print a compact run summary after each pass."""
    store = StateStore(config.db_path)
    tree_data = get_node_tree(store, run_id)
    run_record = store.get_run(run_id)
    nodes = store.get_run_nodes(run_id)
    store.close()

    status = run_record.status if run_record else "unknown"
    pass_num = run_record.pass_count if run_record else 1

    click.echo(_hr())
    click.echo()
    click.echo(f"  {_dim('run')}    {run_id}")
    click.echo(f"  {_dim('status')} {_styled_state(status)}  {_dim('pass')} {pass_num}  {_dim('nodes')} {len(nodes)}")

    if tree_data:
        click.echo()
        _print_tree(tree_data[0])

    click.echo()

    if status == "paused":
        click.echo(f"  {_dim('Waiting for input. Type your next instructions or /help.')}")
        click.echo()


def _print_tree(node: dict, prefix: str = "  ", is_last: bool = True) -> None:
    connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
    state = node["state"]

    indicator_map = {
        "completed": _green("+"),
        "failed": _red("x"),
        "cancelled": _dim("-"),
        "paused": _yellow("||"),
    }
    indicator = indicator_map.get(state, _cyan("~"))

    domain_tag = f" {_magenta(node['domain'])}" if node.get("domain") else ""
    node_short = node["node_id"][:12]
    task_short = node["task_spec"][:55]

    click.echo(f"{prefix}{connector}{indicator} {_dim(node_short)}{domain_tag} {task_short}")

    children = node.get("children", [])
    for i, child in enumerate(children):
        extension = "   " if is_last else "\u2502  "
        _print_tree(child, prefix + extension, i == len(children) - 1)


def _print_live_tree(config: RuntimeConfig, run_id: str) -> None:
    """Print the tree from live DB state."""
    try:
        store = StateStore(config.db_path)
        tree_data = get_node_tree(store, run_id)
        nodes = store.get_run_nodes(run_id)
        store.close()
    except Exception:
        click.echo(_dim("  (could not read state)"))
        return

    if not tree_data:
        click.echo(_dim("  No nodes yet."))
        return

    active = sum(1 for n in nodes if not n.state.is_idle)
    done = sum(1 for n in nodes if n.state == NodeState.COMPLETED)
    click.echo(f"  {_dim('nodes')} {len(nodes)}  {_green(str(done) + ' done')}  {_cyan(str(active) + ' active')}")
    click.echo()
    _print_tree(tree_data[0])
    click.echo()


def _read_input() -> str:
    try:
        return click.prompt(
            f"  {BOLD}{CYAN}\u276f{RESET}",
            prompt_suffix=" ",
        ).strip()
    except (click.Abort, EOFError):
        return ""


def _handle_slash_command(cmd: str, config: RuntimeConfig, runner: BackgroundRunner) -> bool:
    """Handle slash commands in the REPL. Returns True if should exit."""
    cmd = cmd.strip().lower()
    run_id = runner.run_id

    if cmd in ("/quit", "/exit", "/q"):
        if run_id:
            click.echo(f"\n  {_dim('Run paused. Resume with:')} rari chat {run_id}")
        click.echo()
        return True

    if cmd == "/done":
        if run_id:
            config.ensure_dirs()
            store = StateStore(config.db_path)
            store.finish_run(run_id, "completed")
            store.close()
        click.echo(f"\n  {_green('done')} Run finalized.")
        click.echo()
        return True

    if cmd in ("/tree", "/t"):
        click.echo()
        if run_id:
            _print_live_tree(config, run_id)
        else:
            click.echo(_dim("  No active run."))
        return False

    if cmd in ("/domains", "/d"):
        click.echo()
        if run_id:
            config.ensure_dirs()
            store = StateStore(config.db_path)
            run_record = store.get_run(run_id)
            if run_record and run_record.root_node_id:
                domain_list = store.get_domains(run_record.root_node_id)
                if domain_list:
                    for d in domain_list:
                        child = store.get_node(d.child_node_id)
                        state = child.state.value if child else "?"
                        click.echo(f"  {_magenta(d.domain_name)}  {_styled_state(state)}  {_dim(d.child_node_id[:12])}")
                else:
                    click.echo(_dim("  No domains."))
            store.close()
        else:
            click.echo(_dim("  No active run."))
        click.echo()
        return False

    if cmd in ("/status", "/s"):
        click.echo()
        if run_id:
            try:
                store = StateStore(config.db_path)
                run_record = store.get_run(run_id)
                nodes = store.get_run_nodes(run_id)
                store.close()
                status = run_record.status if run_record else "unknown"
                active = [n for n in nodes if not n.state.is_idle]
                click.echo(f"  {_dim('run')}    {run_id}")
                click.echo(f"  {_dim('status')} {_styled_state(status)}  {_dim('nodes')} {len(nodes)}  {_cyan(str(len(active)) + ' active')}")
                if runner.busy:
                    click.echo(f"  {_cyan('working...')}")
            except Exception:
                click.echo(_dim("  (could not read state)"))
        else:
            click.echo(_dim("  No active run."))
        click.echo()
        return False

    if cmd == "/help":
        click.echo()
        click.echo(f"  {_bold('/tree')}     {_dim('Show the node tree (live)')}")
        click.echo(f"  {_bold('/status')}   {_dim('Show run status')}")
        click.echo(f"  {_bold('/domains')}  {_dim('Show domain registry')}")
        click.echo(f"  {_bold('/done')}     {_dim('Finalize and close the run')}")
        click.echo(f"  {_bold('/quit')}     {_dim('Exit (run stays paused)')}")
        click.echo()
        return False

    click.echo(f"  {_dim('Unknown command. Try /help')}")
    return False
