"""CLI entrypoint for the recursive intelligence runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.orchestrator import Orchestrator, get_node_tree
from recursive_intelligence.runtime.state_store import StateStore


@click.group()
@click.option("--repo", type=click.Path(exists=True), default=".", help="Repository root")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, repo: str, verbose: bool) -> None:
    """ri – recursive intelligence runtime."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    ctx.ensure_object(dict)
    ctx.obj["config"] = RuntimeConfig(repo_root=Path(repo).resolve())


@main.command()
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.option("--persistent", is_flag=True, help="Keep run alive for follow-up passes")
@click.option("--cleanup/--no-cleanup", default=False, help="Remove worktrees after run")
@click.pass_context
def run(ctx: click.Context, task: str, model: str, persistent: bool, cleanup: bool) -> None:
    """Start a new recursive run."""
    config: RuntimeConfig = ctx.obj["config"]

    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

    adapter = ClaudeAdapter(model=model)
    orchestrator = Orchestrator(config, adapter)

    try:
        run_id = asyncio.run(orchestrator.start_run(task, persistent=persistent))
        _print_run_summary(config, run_id)

        if cleanup:
            orchestrator.cleanup_worktrees(run_id)
            click.echo("Worktrees cleaned up.")
    except Exception as e:
        click.echo(f"Run failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def baseline(ctx: click.Context, task: str, model: str) -> None:
    """Run a single flat Claude session (no recursion). Control group for benchmarks."""
    config: RuntimeConfig = ctx.obj["config"]

    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter
    from recursive_intelligence.runtime.baseline import BaselineRunner

    adapter = ClaudeAdapter(model=model)
    runner = BaselineRunner(config, adapter)

    try:
        report = asyncio.run(runner.run(task))
        click.echo(f"\nBaseline run: {report.run_id}")
        click.echo(f"Status:       {report.status}")
        click.echo(f"Session:      {report.session_id}")
        click.echo(f"Branch:       {report.branch_name}")
        click.echo(f"Cost:         ${report.cost.total_usd:.4f}")
        click.echo(f"Turns:        {report.num_turns}")
        click.echo(f"Duration:     {report.duration_ms}ms (API: {report.duration_api_ms}ms)")
        click.echo(f"Stop reason:  {report.stop_reason}")
        if report.changed_files:
            click.echo(f"Changed:      {', '.join(report.changed_files)}")
        else:
            click.echo("Changed:      (none)")
        click.echo(f"\nReport saved: .ri/runs/{report.run_id}/report.json")
    except Exception as e:
        click.echo(f"Baseline failed: {e}", err=True)
        sys.exit(1)


@main.command(name="continue")
@click.argument("run_id")
@click.argument("task")
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def continue_run(ctx: click.Context, run_id: str, task: str, model: str) -> None:
    """Continue a paused persistent run with new instructions."""
    config: RuntimeConfig = ctx.obj["config"]

    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

    adapter = ClaudeAdapter(model=model)
    orchestrator = Orchestrator(config, adapter)

    try:
        asyncio.run(orchestrator.continue_run(run_id, task))
        _print_run_summary(config, run_id)
    except Exception as e:
        click.echo(f"Continue failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id", required=False)
@click.option("--model", default="claude-sonnet-4-6", help="Model to use")
@click.pass_context
def chat(ctx: click.Context, run_id: str | None, model: str) -> None:
    """Interactive REPL — conversational interface to a persistent run."""
    config: RuntimeConfig = ctx.obj["config"]

    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

    adapter = ClaudeAdapter(model=model)
    orchestrator = Orchestrator(config, adapter)

    if run_id:
        # Resume existing run
        store = StateStore(config.db_path)
        run_record = store.get_run(run_id)
        store.close()
        if run_record is None:
            click.echo(f"Run {run_id} not found", err=True)
            sys.exit(1)
        click.echo(f"Resuming run {run_id} (pass {run_record.pass_count})")
    else:
        # Start a new persistent run with the first message
        click.echo("Starting new persistent run. Type your task:")
        first_msg = _read_input()
        if not first_msg:
            return
        try:
            run_id = asyncio.run(orchestrator.start_run(first_msg, persistent=True))
            _print_run_summary(config, run_id)
        except Exception as e:
            click.echo(f"Failed: {e}", err=True)
            sys.exit(1)

    # REPL loop
    while True:
        click.echo("")
        user_input = _read_input()
        if not user_input:
            continue

        if user_input.startswith("/"):
            if _handle_slash_command(user_input, config, run_id):
                break
            continue

        try:
            asyncio.run(orchestrator.continue_run(run_id, user_input))
            _print_run_summary(config, run_id)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)


@main.command()
@click.argument("run_id")
@click.pass_context
def resume(ctx: click.Context, run_id: str) -> None:
    """Resume a crashed/interrupted run."""
    config: RuntimeConfig = ctx.obj["config"]

    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

    adapter = ClaudeAdapter()
    orchestrator = Orchestrator(config, adapter)

    try:
        asyncio.run(orchestrator.resume_run(run_id))
        click.echo(f"Run resumed successfully: {run_id}")
    except Exception as e:
        click.echo(f"Resume failed: {e}", err=True)
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
        click.echo(f"No nodes found for run {run_id}")
        store.close()
        return

    _print_tree(tree_data[0])
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
        click.echo(f"Node {node_id} not found")
        store.close()
        return

    click.echo(f"Node:       {node.node_id}")
    click.echo(f"Run:        {node.run_id}")
    click.echo(f"Parent:     {node.parent_id or '(root)'}")
    click.echo(f"State:      {node.state.value}")
    click.echo(f"Task:       {node.task_spec}")
    click.echo(f"Worktree:   {node.worktree_path or '(none)'}")
    click.echo(f"Branch:     {node.branch_name or '(none)'}")
    click.echo(f"Session:    {node.session_id or '(none)'}")

    domain = store.get_domain_by_child(node_id)
    if domain:
        click.echo(f"Domain:     {domain.domain_name}")
        click.echo(f"Files:      {', '.join(domain.file_patterns)}")
        click.echo(f"Scope:      {domain.module_scope}")

    events = store.get_node_events(node_id)
    if events:
        click.echo(f"\nEvents ({len(events)}):")
        for evt in events:
            click.echo(f"  [{evt.timestamp}] {evt.event_type}: {json.dumps(evt.data, indent=None)[:120]}")

    children = store.get_children(node_id)
    if children:
        click.echo(f"\nChildren ({len(children)}):")
        for child in children:
            d = store.get_domain_by_child(child.node_id)
            domain_tag = f" [{d.domain_name}]" if d else ""
            click.echo(f"  {child.node_id} [{child.state.value}]{domain_tag} {child.task_spec[:50]}")

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
        click.echo(f"Run {run_id} not found")
        store.close()
        return

    domain_list = store.get_domains(run_record.root_node_id)
    if not domain_list:
        click.echo("No domains registered yet.")
        store.close()
        return

    click.echo(f"Domains for run {run_id} (pass {run_record.pass_count}):\n")
    for d in domain_list:
        child = store.get_node(d.child_node_id)
        state = child.state.value if child else "unknown"
        click.echo(f"  {d.domain_name}")
        click.echo(f"    Child:    {d.child_node_id}")
        click.echo(f"    State:    {state}")
        click.echo(f"    Files:    {', '.join(d.file_patterns) or '(none)'}")
        click.echo(f"    Scope:    {d.module_scope or '(none)'}")
        click.echo()

    store.close()


# --- Helpers ---

def _print_run_summary(config: RuntimeConfig, run_id: str) -> None:
    store = StateStore(config.db_path)
    tree_data = get_node_tree(store, run_id)
    run_record = store.get_run(run_id)
    nodes = store.get_run_nodes(run_id)
    store.close()

    click.echo(f"\nRun: {run_id}")
    click.echo(f"Status: {run_record.status if run_record else 'unknown'}")
    click.echo(f"Pass: {run_record.pass_count if run_record else 1}")
    click.echo(f"Nodes: {len(nodes)}")

    if tree_data:
        click.echo("\nTree:")
        _print_tree(tree_data[0])

    if run_record and run_record.status == "paused":
        click.echo(f"\nRun is paused. Continue with:")
        click.echo(f"  ri continue {run_id} \"your next instructions\"")
        click.echo(f"  ri chat {run_id}")


def _print_tree(node: dict, prefix: str = "", is_last: bool = True) -> None:
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    state = node["state"]
    indicator = {
        "completed": "+", "failed": "x", "cancelled": "-", "paused": "||",
    }.get(state, "~")
    domain_tag = f" [{node['domain']}]" if node.get("domain") else ""
    click.echo(f"{prefix}{connector}[{indicator}] {node['node_id'][:16]} ({state}){domain_tag} {node['task_spec']}")

    children = node.get("children", [])
    for i, child in enumerate(children):
        extension = "    " if is_last else "\u2502   "
        _print_tree(child, prefix + extension, i == len(children) - 1)


def _read_input() -> str:
    try:
        return click.prompt("ri", prompt_suffix="> ").strip()
    except (click.Abort, EOFError):
        return ""


def _handle_slash_command(cmd: str, config: RuntimeConfig, run_id: str) -> bool:
    """Handle slash commands in the REPL. Returns True if should exit."""
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        click.echo("Exiting (run remains paused).")
        return True

    if cmd == "/done":
        click.echo("Finalizing run.")
        store = StateStore(config.db_path)
        store.finish_run(run_id, "completed")
        store.close()
        return True

    if cmd == "/tree":
        store = StateStore(config.db_path)
        tree_data = get_node_tree(store, run_id)
        store.close()
        if tree_data:
            _print_tree(tree_data[0])
        else:
            click.echo("No nodes.")
        return False

    if cmd == "/domains":
        store = StateStore(config.db_path)
        run_record = store.get_run(run_id)
        if run_record and run_record.root_node_id:
            domain_list = store.get_domains(run_record.root_node_id)
            for d in domain_list:
                child = store.get_node(d.child_node_id)
                state = child.state.value if child else "?"
                click.echo(f"  {d.domain_name} ({state}) -> {d.child_node_id[:12]}")
        store.close()
        return False

    if cmd == "/help":
        click.echo("Commands: /tree /domains /done /quit /help")
        return False

    click.echo(f"Unknown command: {cmd}. Try /help")
    return False
