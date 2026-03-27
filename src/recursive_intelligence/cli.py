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
@click.option("--cleanup/--no-cleanup", default=False, help="Remove worktrees after run")
@click.pass_context
def run(ctx: click.Context, task: str, model: str, cleanup: bool) -> None:
    """Start a new recursive run."""
    config: RuntimeConfig = ctx.obj["config"]

    from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

    adapter = ClaudeAdapter(model=model)
    orchestrator = Orchestrator(config, adapter)

    try:
        run_id = asyncio.run(orchestrator.start_run(task))

        # Print the tree
        store = StateStore(config.db_path)
        tree_data = get_node_tree(store, run_id)
        run_record = store.get_run(run_id)
        nodes = store.get_run_nodes(run_id)
        store.close()

        click.echo(f"\nRun: {run_id}")
        click.echo(f"Status: {run_record.status if run_record else 'unknown'}")
        click.echo(f"Nodes: {len(nodes)}")

        if tree_data:
            click.echo("\nTree:")
            _print_tree(tree_data[0])

        if cleanup:
            orchestrator.cleanup_worktrees(run_id)
            click.echo("\nWorktrees cleaned up.")
        else:
            click.echo(f"\nWorktrees preserved in {config.worktrees_dir}")
            click.echo("Use --cleanup to remove, or ri tree/inspect to explore.")
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


@main.command()
@click.argument("run_id")
@click.pass_context
def resume(ctx: click.Context, run_id: str) -> None:
    """Resume an existing run."""
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
    click.echo(f"Created:    {node.created_at}")
    click.echo(f"Updated:    {node.updated_at}")

    events = store.get_node_events(node_id)
    if events:
        click.echo(f"\nEvents ({len(events)}):")
        for evt in events:
            click.echo(f"  [{evt.timestamp}] {evt.event_type}: {json.dumps(evt.data, indent=None)}")

    children = store.get_children(node_id)
    if children:
        click.echo(f"\nChildren ({len(children)}):")
        for child in children:
            click.echo(f"  {child.node_id} [{child.state.value}] {child.task_spec[:60]}")

    store.close()


def _print_tree(node: dict, prefix: str = "", is_last: bool = True) -> None:
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    state = node["state"]
    indicator = {"completed": "+", "failed": "x", "cancelled": "-"}.get(state, "~")
    click.echo(f"{prefix}{connector}[{indicator}] {node['node_id']} ({state}) {node['task_spec']}")

    children = node.get("children", [])
    for i, child in enumerate(children):
        extension = "    " if is_last else "\u2502   "
        _print_tree(child, prefix + extension, i == len(children) - 1)
