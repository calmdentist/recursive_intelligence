"""Textual TUI for recursive intelligence – split-pane Claude Code style interface."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Footer,
    Header,
    Input,
    RichLog,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.orchestrator import Orchestrator, get_node_tree
from recursive_intelligence.runtime.state_store import StateStore, NodeState

log = logging.getLogger(__name__)

# ── State icons ──────────────────────────────────────────────────────────────

STATE_ICON = {
    "completed": "[green]✓[/]",
    "failed": "[red]✗[/]",
    "cancelled": "[dim]−[/]",
    "paused": "[yellow]⏸[/]",
    "executing": "[cyan]●[/]",
    "planning": "[cyan]◎[/]",
    "merging": "[cyan]⊕[/]",
    "queued": "[dim]○[/]",
    "waiting_on_children": "[yellow]◌[/]",
    "reviewing_children": "[cyan]◉[/]",
}


def _icon(state: str) -> str:
    return STATE_ICON.get(state, "[dim]?[/]")


# ── App ──────────────────────────────────────────────────────────────────────

class RariApp(App):
    """Recursive Intelligence TUI."""

    CSS = """
    #main {
        height: 1fr;
    }
    #output-pane {
        width: 2fr;
        border-right: solid $surface-lighten-2;
    }
    #side-pane {
        width: 1fr;
        min-width: 30;
    }
    #output {
        height: 1fr;
    }
    #node-tree {
        height: 1fr;
        border-bottom: solid $surface-lighten-2;
    }
    #detail {
        height: auto;
        max-height: 40%;
        overflow-y: auto;
        padding: 0 1;
    }
    #prompt {
        dock: bottom;
        height: auto;
        padding: 0 1;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+t", "toggle_tree", "Tree", show=True),
    ]

    def __init__(
        self,
        config: RuntimeConfig,
        model: str = "claude-sonnet-4-6",
        run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.ri_config = config
        self.model = model
        self.run_id = run_id
        self._orchestrator: Orchestrator | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            with Vertical(id="output-pane"):
                yield RichLog(id="output", wrap=True, highlight=True, markup=True)
            with Vertical(id="side-pane"):
                yield Tree("nodes", id="node-tree")
                yield RichLog(id="detail", wrap=True, highlight=True, markup=True)
        yield Static("", id="status-bar")
        yield Input(
            placeholder="type a task, or /tree /inbox /work /status /answer <id> ... /help /quit",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        output = self.query_one("#output", RichLog)
        output.write("[bold]recursive intelligence[/]")
        output.write(f"[dim]model[/]  {self.model}")
        output.write(f"[dim]repo[/]   {self.ri_config.repo_root}")
        output.write("")

        self.query_one("#prompt", Input).focus()

        if self.run_id:
            output.write(f"[dim]resuming run[/] {self.run_id}")
            self._refresh_tree()

    # ── Input handling ───────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""

        if not text:
            return

        output = self.query_one("#output", RichLog)

        if text.startswith("/"):
            self._handle_command(text)
            return

        if self._busy:
            output.write("[dim]busy — please wait for the current pass to finish[/]")
            return

        output.write(f"\n[bold cyan]❯[/] {text}\n")
        self._run_task(text)

    def _handle_command(self, cmd: str) -> None:
        raw = cmd.strip()
        name, _, _ = raw.partition(" ")
        cmd = name.lower()
        output = self.query_one("#output", RichLog)

        if cmd in ("/quit", "/q", "/exit"):
            if self.run_id:
                output.write(f"[dim]run paused. resume with:[/] rari chat {self.run_id}")
            self.exit()
            return

        if cmd in ("/tree", "/t"):
            self._refresh_tree()
            return

        if cmd in ("/domains", "/d"):
            self._show_domains()
            return

        if cmd in ("/blockers", "/b", "/requests", "/r"):
            self._show_blockers()
            return

        if cmd in ("/inbox", "/i"):
            self._show_inbox()
            return

        if cmd in ("/work", "/tasks", "/w"):
            self._show_work_items()
            return

        request_action = self._parse_request_action(raw)
        if request_action is not None:
            action, request_id, response_text = request_action
            if not self.run_id:
                output.write("[dim]no active run[/]")
                return
            if self._busy:
                output.write("[dim]cannot resolve a request while work is still running[/]")
                return
            if not request_id:
                output.write(f"[dim]usage: {action} <request_id> <response>[/]")
                return
            resolution = {
                "/answer": "answer",
                "/approve": "approve",
                "/decline": "decline",
            }[action]
            try:
                normalized = self._resolution_text(resolution, response_text)
            except ValueError as e:
                output.write(f"[red]error[/] {e}")
                return
            output.write(f"[dim]resolving {request_id} ({resolution})...[/]")
            self._resolve_request(request_id, normalized, resolution)
            return

        if cmd in ("/status", "/s"):
            self._show_status()
            return

        if cmd == "/done":
            if self.run_id:
                self.ri_config.ensure_dirs()
                store = StateStore(self.ri_config.db_path)
                store.finish_run(self.run_id, "completed")
                store.close()
                output.write("[green]run finalized.[/]")
            self.exit()
            return

        if cmd == "/help":
            output.write("[bold]/tree[/]     [dim]show/refresh the node tree[/]")
            output.write("[bold]/domains[/]  [dim]show domain registry[/]")
            output.write("[bold]/inbox[/]    [dim]show unresolved root-inbox requests[/]")
            output.write("[bold]/work[/]     [dim]show unresolved downstream work items[/]")
            output.write("[bold]/answer[/]   [dim]answer a request: /answer <id> <text>[/]")
            output.write("[bold]/approve[/]  [dim]approve a request: /approve <id> [note][/]")
            output.write("[bold]/decline[/]  [dim]decline a request: /decline <id> [reason][/]")
            output.write("[bold]/requests[/] [dim]show active upstream requests[/]")
            output.write("[bold]/blockers[/] [dim]legacy alias for /requests[/]")
            output.write("[bold]/status[/]   [dim]show run status[/]")
            output.write("[bold]/done[/]     [dim]finalize and exit[/]")
            output.write("[bold]/quit[/]     [dim]exit (run stays paused)[/]")
            output.write("[bold]ctrl+t[/]    [dim]refresh tree[/]")
            return

        output.write(f"[dim]unknown command: {cmd}. try /help[/]")

    # ── Orchestrator integration ─────────────────────────────────────────

    @work(thread=True)
    def _run_task(self, task: str) -> None:
        """Run a task in a background thread."""
        self._busy = True
        self._update_status("working...")

        try:
            from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

            adapter = ClaudeAdapter(model=self.model)
            orch = Orchestrator(self.ri_config, adapter, on_message=self._on_stream_message)
            self._orchestrator = orch

            if self.run_id is None:
                self.run_id = asyncio.run(orch.start_run(task, persistent=True))
            else:
                asyncio.run(orch.continue_run(self.run_id, task))

            self.call_from_thread(self._on_pass_complete)
        except Exception as e:
            self.call_from_thread(self._on_error, str(e))
        finally:
            self._busy = False
            self._update_status("")

    @work(thread=True)
    def _resolve_request(self, request_id: str, response_text: str, resolution: str) -> None:
        """Resolve a specific inbox request in a background thread."""
        self._busy = True
        self._update_status(f"resolving {request_id}...")

        try:
            from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

            adapter = ClaudeAdapter(model=self.model)
            orch = Orchestrator(self.ri_config, adapter, on_message=self._on_stream_message)
            self._orchestrator = orch

            if self.run_id is None:
                raise ValueError("No active run")

            asyncio.run(
                orch.resolve_request(
                    self.run_id,
                    request_id,
                    response_text,
                    resolution=resolution,
                )
            )

            self.call_from_thread(self._on_pass_complete)
        except Exception as e:
            self.call_from_thread(self._on_error, str(e))
        finally:
            self._busy = False
            self._update_status("")

    def _on_stream_message(self, msg_type: str, data: dict[str, Any]) -> None:
        """Callback from adapter — called from the background thread."""
        self.call_from_thread(self._render_stream_message, msg_type, data)

    def _render_stream_message(self, msg_type: str, data: dict[str, Any]) -> None:
        """Render a streaming message in the output pane (main thread)."""
        output = self.query_one("#output", RichLog)

        if msg_type == "text":
            text = data.get("text", "")
            if text.strip():
                output.write(text)

        elif msg_type == "thinking":
            text = data.get("text", "")
            if text.strip():
                preview = text.strip().split("\n")[0][:90]
                output.write(f"[dim]━ {preview}{'...' if len(text) > 90 else ''}[/]")

        elif msg_type == "tool_use":
            tool = data.get("tool", "")
            inp = data.get("input", {})
            arg = self._summarize_tool_input(tool, inp)
            line = f"[cyan]▸ {tool}[/]"
            if arg:
                line += f" [dim]{arg}[/]"
            output.write(line)

        elif msg_type == "tool_result":
            content = data.get("content", "")
            if content.strip():
                lines = content.strip().split("\n")
                preview = lines[0][:100]
                if len(lines) > 1:
                    preview += f" [dim](+{len(lines)-1} lines)[/]"
                output.write(f"  [dim]{preview}[/]")

        # Auto-refresh tree periodically during streaming
        self._refresh_tree()

    def _on_pass_complete(self) -> None:
        output = self.query_one("#output", RichLog)
        output.write("")

        if not self.run_id:
            return

        try:
            store = StateStore(self.ri_config.db_path)
            run = store.get_run(self.run_id)
            nodes = store.get_run_nodes(self.run_id)
            blockers = store.get_run_blockers(self.run_id)
            inbox = store.get_run_inbox(self.run_id)
            work_items = store.get_run_downstream_tasks(self.run_id)
            store.close()
        except Exception:
            return

        if not run:
            return

        completed = sum(1 for n in nodes if n.state == NodeState.COMPLETED)
        failed = sum(1 for n in nodes if n.state == NodeState.FAILED)

        status_color = {"paused": "yellow", "completed": "green", "failed": "red"}.get(run.status, "cyan")
        line = f"[{status_color}]{run.status}[/]  [dim]{len(nodes)} nodes  {completed} done[/]"
        if failed:
            line += f"  [red]{failed} failed[/]"
        if blockers:
            line += f"  [yellow]{len(blockers)} blocked[/]"
        if inbox:
            line += f"  [blue]{len(inbox)} inbox[/]"
        if work_items:
            line += f"  [magenta]{len(work_items)} work[/]"
        output.write(line)

        if run.status == "paused":
            output.write("[dim]type your next instructions, or /work to inspect pending requests and tasks[/]")
            if blockers:
                self._write_blocker_lines(output, blockers, limit=2)
            if work_items:
                self._write_downstream_task_lines(output, work_items, limit=2)
        output.write("")

        self._refresh_tree()

    def _on_error(self, msg: str) -> None:
        output = self.query_one("#output", RichLog)
        output.write(f"\n[red]error[/] {msg}\n")

    def _update_status(self, text: str) -> None:
        try:
            self.call_from_thread(self._set_status, text)
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
            bar.update(text)
        except NoMatches:
            pass

    # ── Tree ─────────────────────────────────────────────────────────────

    def _refresh_tree(self) -> None:
        """Rebuild the tree widget from SQLite state."""
        if not self.run_id:
            return

        try:
            tree_widget = self.query_one("#node-tree", Tree)
        except NoMatches:
            return

        try:
            store = StateStore(self.ri_config.db_path)
            tree_data = get_node_tree(store, self.run_id)
            store.close()
        except Exception:
            return

        if not tree_data:
            return

        tree_widget.clear()
        tree_widget.root.expand()
        self._build_tree_node(tree_widget.root, tree_data[0])

    def _build_tree_node(self, parent: TreeNode, data: dict) -> None:
        icon = _icon(data["state"])
        domain = data.get("domain")
        domain_tag = f" [magenta]{domain}[/]" if domain else ""
        task = data["task_spec"][:45]
        label = f"{icon} [dim]{data['node_id'][:10]}[/]{domain_tag} {task}"

        node = parent.add(label, data=data, expand=True)

        for child in data.get("children", []):
            self._build_tree_node(node, child)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Show node detail when clicked."""
        data = event.node.data
        if not data or not isinstance(data, dict):
            return

        node_id = data.get("node_id")
        if not node_id:
            return

        self._show_node_detail(node_id)

    def _show_node_detail(self, node_id: str) -> None:
        try:
            detail = self.query_one("#detail", RichLog)
        except NoMatches:
            return

        detail.clear()

        try:
            store = StateStore(self.ri_config.db_path)
            node = store.get_node(node_id)
            if not node:
                detail.write(f"[red]node {node_id} not found[/]")
                store.close()
                return

            state_color = {"completed": "green", "failed": "red", "paused": "yellow"}.get(
                node.state.value, "cyan"
            )
            detail.write(f"[bold]{node.node_id}[/]")
            detail.write(f"[dim]state[/]    [{state_color}]{node.state.value}[/]")
            detail.write(f"[dim]task[/]     {node.task_spec[:80]}")

            if node.worktree_path:
                detail.write(f"[dim]worktree[/] {node.worktree_path}")
            if node.branch_name:
                detail.write(f"[dim]branch[/]   {node.branch_name}")

            domain = store.get_domain_by_child(node_id)
            if domain:
                detail.write(f"[dim]domain[/]   [magenta]{domain.domain_name}[/]")
                if domain.file_patterns:
                    detail.write(f"[dim]files[/]    {', '.join(domain.file_patterns)}")
                if domain.module_scope:
                    detail.write(f"[dim]scope[/]    {domain.module_scope}")

            request = store.get_latest_request(node_id)
            if request:
                payload = request["request"]
                detail.write(f"[dim]request[/]  [blue]{payload.get('kind', 'request')}[/]")
                title = payload.get("summary") or payload.get("details", "")
                if title:
                    detail.write(f"[dim]summary[/]  {title[:100]}")
                action = payload.get("action_requested")
                if action:
                    detail.write(f"[dim]action[/]   {action[:100]}")
            downstream_task = store.get_latest_downstream_task(node_id)
            if downstream_task:
                payload = downstream_task["task"]
                detail.write(f"[dim]work[/]     [magenta]{payload.get('kind', 'task')}[/]")
                title = payload.get("summary") or payload.get("task_spec", "")
                if title:
                    detail.write(f"[dim]summary[/]  {title[:100]}")
                task_spec = payload.get("task_spec")
                if task_spec and task_spec != title:
                    detail.write(f"[dim]details[/]  {task_spec[:100]}")

            events = store.get_node_events(node_id)
            if events:
                detail.write(f"\n[dim]events ({len(events)})[/]")
                for evt in events[-10:]:  # last 10
                    ts = evt.timestamp.split("T")[1][:8] if "T" in evt.timestamp else ""
                    detail.write(f"  [dim]{ts}[/] {evt.event_type}")

            store.close()
        except Exception as e:
            detail.write(f"[red]error: {e}[/]")

    # ── Commands ─────────────────────────────────────────────────────────

    def _show_domains(self) -> None:
        output = self.query_one("#output", RichLog)
        if not self.run_id:
            output.write("[dim]no active run[/]")
            return

        try:
            store = StateStore(self.ri_config.db_path)
            run = store.get_run(self.run_id)
            if not run or not run.root_node_id:
                output.write("[dim]no root node[/]")
                store.close()
                return

            domains = store.get_domains(run.root_node_id)
            if not domains:
                output.write("[dim]no domains registered[/]")
                store.close()
                return

            output.write("")
            for d in domains:
                child = store.get_node(d.child_node_id)
                state = child.state.value if child else "?"
                state_color = {"completed": "green", "failed": "red"}.get(state, "cyan")
                output.write(
                    f"[magenta]{d.domain_name}[/]  [{state_color}]{state}[/]  "
                    f"[dim]{d.child_node_id[:12]}[/]"
                )
                if d.file_patterns:
                    output.write(f"  [dim]{', '.join(d.file_patterns)}[/]")
            output.write("")
            store.close()
        except Exception as e:
            output.write(f"[red]error: {e}[/]")

    def _show_status(self) -> None:
        output = self.query_one("#output", RichLog)
        if not self.run_id:
            output.write("[dim]no active run[/]")
            return

        try:
            store = StateStore(self.ri_config.db_path)
            run = store.get_run(self.run_id)
            nodes = store.get_run_nodes(self.run_id)
            blockers = store.get_run_blockers(self.run_id)
            inbox = store.get_run_inbox(self.run_id)
            work_items = store.get_run_downstream_tasks(self.run_id)
            store.close()

            if not run:
                output.write("[dim]run not found[/]")
                return

            active = sum(1 for n in nodes if not n.state.is_idle)
            done = sum(1 for n in nodes if n.state == NodeState.COMPLETED)
            failed = sum(1 for n in nodes if n.state == NodeState.FAILED)

            status_color = {"paused": "yellow", "completed": "green", "failed": "red"}.get(
                run.status, "cyan"
            )
            output.write(f"[dim]run[/]    {self.run_id}")
            output.write(
                f"[dim]status[/] [{status_color}]{run.status}[/]  "
                f"[dim]pass[/] {run.pass_count}  "
                f"[dim]nodes[/] {len(nodes)}  "
                f"[green]{done} done[/]  [cyan]{active} active[/]"
                + (f"  [red]{failed} failed[/]" if failed else "")
            )
            output.write(
                f"[dim]telemetry[/] {run.telemetry.get('user_interruptions_count', 0)} interrupts  "
                f"{run.telemetry.get('root_requests_count', 0)} requests  "
                f"{run.telemetry.get('resolved_requests_count', 0)} resolved"
            )
            if inbox:
                output.write(f"[blue]root inbox[/] {len(inbox)} unresolved")
                self._write_inbox_lines(output, inbox, limit=2)
            if work_items:
                output.write(f"[magenta]downstream work[/] {len(work_items)} unresolved")
                self._write_downstream_task_lines(output, work_items, limit=2)
            if blockers:
                output.write(f"[yellow]active blockers[/] {len(blockers)}")
                self._write_blocker_lines(output, blockers, limit=2)
            if self._busy:
                output.write("[cyan]working...[/]")
            output.write("")
        except Exception as e:
            output.write(f"[red]error: {e}[/]")

    def _show_blockers(self) -> None:
        output = self.query_one("#output", RichLog)
        if not self.run_id:
            output.write("[dim]no active run[/]")
            return

        try:
            store = StateStore(self.ri_config.db_path)
            run = store.get_run(self.run_id)
            blockers = store.get_run_blockers(self.run_id)
            store.close()
        except Exception as e:
            output.write(f"[red]error: {e}[/]")
            return

        if not run:
            output.write("[dim]run not found[/]")
            return

        output.write(f"[dim]run[/]      {self.run_id}")
        output.write(
            f"[dim]telemetry[/] {run.telemetry.get('user_interruptions_count', 0)} interrupts  "
            f"{run.telemetry.get('root_requests_count', 0)} requests  "
            f"{run.telemetry.get('resolved_requests_count', 0)} resolved"
        )
        if not blockers:
            output.write("[dim]no active blockers[/]")
            output.write("")
            return
        output.write(f"[yellow]active blockers[/] {len(blockers)}")
        self._write_blocker_lines(output, blockers, limit=None)
        output.write("")

    def _show_inbox(self) -> None:
        output = self.query_one("#output", RichLog)
        if not self.run_id:
            output.write("[dim]no active run[/]")
            return

        try:
            store = StateStore(self.ri_config.db_path)
            run = store.get_run(self.run_id)
            inbox = store.get_run_inbox(self.run_id)
            store.close()
        except Exception as e:
            output.write(f"[red]error: {e}[/]")
            return

        if not run:
            output.write("[dim]run not found[/]")
            return

        output.write(f"[dim]run[/]      {self.run_id}")
        output.write(
            f"[dim]telemetry[/] {run.telemetry.get('root_requests_count', 0)} requests  "
            f"{run.telemetry.get('resolved_requests_count', 0)} resolved"
        )
        if not inbox:
            output.write("[dim]inbox is empty[/]")
            output.write("")
            return
        output.write(f"[blue]root inbox[/] {len(inbox)} unresolved")
        self._write_inbox_lines(output, inbox, limit=None)
        output.write("[dim]use /answer, /approve, or /decline with a request id to resolve one[/]")
        output.write("")

    def _show_work_items(self) -> None:
        output = self.query_one("#output", RichLog)
        if not self.run_id:
            output.write("[dim]no active run[/]")
            return

        try:
            store = StateStore(self.ri_config.db_path)
            work_items = store.get_run_downstream_tasks(self.run_id)
            store.close()
        except Exception as e:
            output.write(f"[red]error: {e}[/]")
            return

        if not work_items:
            output.write("[dim]no unresolved downstream work items[/]")
            output.write("")
            return

        output.write(f"[magenta]downstream work[/] {len(work_items)} unresolved")
        self._write_downstream_task_lines(output, work_items, limit=None)
        output.write("")

    def action_toggle_tree(self) -> None:
        self._refresh_tree()

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _summarize_tool_input(tool: str, inp: dict) -> str:
        if tool in ("Read", "Glob", "Grep"):
            return str(inp.get("file_path") or inp.get("path") or inp.get("pattern") or "")
        if tool in ("Edit", "Write"):
            return str(inp.get("file_path", ""))
        if tool == "Bash":
            cmd = inp.get("command", "")
            return cmd[:60] if cmd else ""
        if tool == "Agent":
            return inp.get("description", "")[:40]
        return ""

    def _write_blocker_lines(self, output: RichLog, blockers: list[dict[str, Any]], limit: int | None) -> None:
        subset = blockers[:limit] if limit is not None else blockers
        for blocker_entry in subset:
            blocker = blocker_entry["blocker"]
            escalation = blocker.get("escalation", {})
            title = escalation.get("summary") or blocker.get("details") or blocker.get("kind", "blocked")
            domain = blocker_entry.get("domain_name")
            domain_tag = f" [magenta]{domain}[/]" if domain else ""
            output.write(
                f"  [yellow]![/] [{self._state_color(blocker_entry['state'])}]{blocker_entry['state']}[/]"
                f"{domain_tag} [dim]{blocker_entry['node_id'][:12]}[/] {title[:80]}"
            )
            action = escalation.get("action_requested")
            if action:
                output.write(f"    [dim]action[/] {action[:88]}")
        if limit is not None and len(blockers) > limit:
            output.write(f"    [dim]+{len(blockers) - limit} more blocker(s)[/]")

    def _write_inbox_lines(self, output: RichLog, inbox: list[dict[str, Any]], limit: int | None) -> None:
        subset = inbox[:limit] if limit is not None else inbox
        for item in subset:
            request = item["request"]
            domain = item.get("domain_name")
            domain_tag = f" [magenta]{domain}[/]" if domain else ""
            output.write(
                f"  [blue]>[/] {request.get('kind', 'request')}{domain_tag} "
                f"[dim]{item['request_id']}[/] {request.get('summary', '')[:80]}"
            )
            action = request.get("action_requested")
            if action:
                output.write(f"    [dim]action[/] {action[:88]}")
        if limit is not None and len(inbox) > limit:
            output.write(f"    [dim]+{len(inbox) - limit} more inbox item(s)[/]")

    def _write_downstream_task_lines(self, output: RichLog, work_items: list[dict[str, Any]], limit: int | None) -> None:
        subset = work_items[:limit] if limit is not None else work_items
        for item in subset:
            task = item["task"]
            domain = item.get("domain_name")
            domain_tag = f" [magenta]{domain}[/]" if domain else ""
            summary = task.get("summary") or task.get("task_spec", "")
            output.write(
                f"  [magenta]>[/] {task.get('kind', 'task')}{domain_tag} "
                f"[dim]{item['node_id'][:12]}[/] {summary[:80]}"
            )
            details = task.get("task_spec")
            if details and details != summary:
                output.write(f"    [dim]details[/] {details[:88]}")
        if limit is not None and len(work_items) > limit:
            output.write(f"    [dim]+{len(work_items) - limit} more work item(s)[/]")

    @staticmethod
    def _state_color(state: str) -> str:
        return {"paused": "yellow", "failed": "red", "completed": "green"}.get(state, "cyan")

    @staticmethod
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

    @staticmethod
    def _resolution_text(resolution: str, text: str) -> str:
        normalized = text.strip()
        if normalized:
            return normalized
        if resolution == "approve":
            return "Approved."
        if resolution == "decline":
            return "Declined."
        raise ValueError("A response is required")


def run_tui(config: RuntimeConfig, model: str, run_id: str | None = None) -> None:
    """Launch the TUI app."""
    # Route all logging to file in TUI mode
    config.ensure_dirs()
    log_path = config.ri_dir / "rari.log"
    handler = logging.FileHandler(str(log_path), mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

    app = RariApp(config=config, model=model, run_id=run_id)
    app.run()
