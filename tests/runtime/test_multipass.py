"""Tests for multi-pass persistent runs, domain registry, and routing."""

import sqlite3
import subprocess
from pathlib import Path

import pytest

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.orchestrator import Orchestrator, get_node_tree
from recursive_intelligence.runtime.state_store import NodeState, StateStore


class ScriptedAdapter(AgentAdapter):
    """Mock adapter with scripted responses."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._call_log: list[dict] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[dict]:
        return self._call_log

    async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
        if not self._responses:
            raise RuntimeError(f"ScriptedAdapter ran out of responses (call #{len(self._call_log) + 1}, mode={mode})")

        resp = self._responses.pop(0)
        self._call_log.append({"prompt": prompt[:200], "mode": mode, "worktree": str(worktree)})

        if resp.get("_commit"):
            _make_commit(Path(worktree), resp.get("_commit_file", "out.txt"), resp.get("_commit_msg", "mock"))

        raw = {k: v for k, v in resp.items() if not k.startswith("_")}
        return NodeResult(
            session_id=resp.get("_session_id", f"session-{len(self._call_log)}"),
            raw=raw, result_text="", cost=CostRecord(total_usd=0.01),
            stop_reason="end_turn",
        )


def _make_commit(worktree: Path, filename: str, message: str) -> None:
    target = worktree / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"content of {filename}\n")
    subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=str(worktree), capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


@pytest.fixture
def config(git_repo):
    return RuntimeConfig(repo_root=git_repo)


class TestPersistentRun:
    """Persistent runs pause after pass 1 and can be continued."""

    @pytest.mark.asyncio
    async def test_persistent_run_pauses(self, config):
        adapter = ScriptedAdapter([
            # Root plans: spawn child with domain
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "auth", "objective": "build auth module",
                  "success_criteria": ["auth works"],
                  "domain_name": "auth", "file_patterns": ["src/auth/**"],
                  "module_scope": "Authentication and sessions"},
             ]},
            # Child plans + executes
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "built auth",
             "_commit": True, "_commit_file": "src/auth/auth.py", "_commit_msg": "auth"},
            # Root reviews: accept
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("build an app", persistent=True)

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "paused"
        assert run.pass_count == 1

        root = store.get_node(run.root_node_id)
        assert root.state == NodeState.PAUSED

        # Domain should be registered
        domains = store.get_domains(root.node_id)
        assert len(domains) == 1
        assert domains[0].domain_name == "auth"
        assert domains[0].file_patterns == ["src/auth/**"]
        store.close()

    @pytest.mark.asyncio
    async def test_continue_routes_to_child(self, config):
        """Pass 1 spawns a child. Pass 2 routes follow-up to that child."""
        adapter = ScriptedAdapter([
            # --- Pass 1 ---
            # Root: spawn auth child
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "auth", "objective": "build auth",
                  "success_criteria": ["works"],
                  "domain_name": "auth", "file_patterns": ["src/auth/**"],
                  "module_scope": "Auth module"},
             ]},
            # Auth child: plan + execute
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "built auth",
             "_commit": True, "_commit_file": "src/auth/auth.py", "_commit_msg": "auth v1"},
            # Root reviews: accept
            {"verdict": "accept", "reason": "ok"},

            # --- Pass 2 (continue_run) ---
            # Root routing: route to existing auth child
            {"action": "route_to_children", "rationale": "auth domain",
             "routes": [{"child_node_id": "PLACEHOLDER", "domain_name": "auth",
                         "task_spec": "add password hashing"}]},
            # Auth child re-executes
            {"status": "implemented", "summary": "added hashing",
             "_commit": True, "_commit_file": "src/auth/hashing.py", "_commit_msg": "add hashing"},
            # Root reviews reactivated child: accept
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)

        # Pass 1
        run_id = await orch.start_run("build an app", persistent=True)

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "paused"

        # Get the child's actual node_id to patch into the routing response
        root = store.get_node(run.root_node_id)
        children = store.get_children(root.node_id)
        assert len(children) == 1
        child_id = children[0].node_id

        # Patch the routing response with the real child_id
        adapter._responses[0]["routes"][0]["child_node_id"] = child_id
        store.close()

        # Pass 2
        await orch.continue_run(run_id, "add password hashing to the auth module")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "paused"  # still persistent
        assert run.pass_count == 2

        # Verify child was reactivated and executed again
        child = store.get_node(child_id)
        assert child.state == NodeState.COMPLETED

        child_events = store.get_node_events(child_id)
        exec_events = [e for e in child_events if e.event_type == "execution_result"]
        assert len(exec_events) == 2  # original + reactivation

        downstream_tasks = [e for e in child_events if e.event_type == "downstream_task"]
        assert len(downstream_tasks) == 1
        assert downstream_tasks[0].data["kind"] == "reactivation"
        assert downstream_tasks[0].data["task_spec"] == "add password hashing"

        # Root worktree should have both files
        root = store.get_node(run.root_node_id)
        root_wt = Path(root.worktree_path)
        assert (root_wt / "src/auth/auth.py").exists()
        assert (root_wt / "src/auth/hashing.py").exists()
        store.close()

    @pytest.mark.asyncio
    async def test_continue_reuses_existing_planning_session(self, config):
        adapter = ScriptedAdapter([
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "auth", "objective": "build auth",
                  "success_criteria": ["works"],
                  "domain_name": "auth", "file_patterns": ["src/auth/**"],
                  "module_scope": "Auth module"},
             ],
             "_session_id": "root-plan"},
            {"action": "solve_directly", "rationale": "ok", "_session_id": "child-plan"},
            {"status": "implemented", "summary": "built auth",
             "_commit": True, "_commit_file": "src/auth/auth.py", "_commit_msg": "auth v1"},
            {"verdict": "accept", "reason": "ok", "_session_id": "root-review"},
            {"action": "route_to_children", "rationale": "auth domain",
             "routes": [{"child_node_id": "PLACEHOLDER", "domain_name": "auth",
                         "task_spec": "add password hashing"}],
             "_session_id": "root-review"},
            {"status": "implemented", "summary": "added hashing",
             "_commit": True, "_commit_file": "src/auth/hashing.py", "_commit_msg": "add hashing"},
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("build an app", persistent=True)

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        child_id = store.get_children(root.node_id)[0].node_id
        adapter._responses[0]["routes"][0]["child_node_id"] = child_id
        store.close()

        await orch.continue_run(run_id, "add password hashing to the auth module")

        conn = sqlite3.connect(config.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
                ("root-review",),
            ).fetchone()[0]
        finally:
            conn.close()

        assert count == 1

    @pytest.mark.asyncio
    async def test_continue_spawns_new_child(self, config):
        """Pass 2 spawns a new child for work outside existing domains."""
        adapter = ScriptedAdapter([
            # --- Pass 1 ---
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "api", "objective": "build API",
                  "success_criteria": ["works"],
                  "domain_name": "api", "file_patterns": ["api.py"],
                  "module_scope": "API routes"},
             ]},
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "built API",
             "_commit": True, "_commit_file": "api.py", "_commit_msg": "api"},
            {"verdict": "accept", "reason": "ok"},

            # --- Pass 2 ---
            # Root: spawn new child for DB (outside API domain)
            {"action": "spawn_children", "rationale": "new domain",
             "children": [
                 {"idempotency_key": "db", "objective": "add database layer",
                  "success_criteria": ["db works"],
                  "domain_name": "database", "file_patterns": ["db.py"],
                  "module_scope": "Database layer"},
             ]},
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "added DB",
             "_commit": True, "_commit_file": "db.py", "_commit_msg": "db"},
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("build app", persistent=True)

        store = StateStore(config.db_path)
        assert store.get_run(run_id).status == "paused"
        store.close()

        await orch.continue_run(run_id, "add a database layer")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "paused"
        assert run.pass_count == 2

        root = store.get_node(run.root_node_id)
        children = store.get_children(root.node_id)
        assert len(children) == 2  # api + db

        domains = store.get_domains(root.node_id)
        domain_names = {d.domain_name for d in domains}
        assert domain_names == {"api", "database"}
        store.close()


class TestDomainRegistry:
    """Domain registration and auto-update from changed files."""

    @pytest.mark.asyncio
    async def test_domains_registered_on_spawn(self, config):
        adapter = ScriptedAdapter([
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "fe", "objective": "frontend",
                  "success_criteria": ["works"],
                  "domain_name": "frontend", "file_patterns": ["src/ui/**"],
                  "module_scope": "React frontend"},
                 {"idempotency_key": "be", "objective": "backend",
                  "success_criteria": ["works"],
                  "domain_name": "backend", "file_patterns": ["src/api/**"],
                  "module_scope": "Express backend"},
             ]},
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "fe done",
             "_commit": True, "_commit_file": "src/ui/index.tsx", "_commit_msg": "fe"},
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "be done",
             "_commit": True, "_commit_file": "src/api/server.ts", "_commit_msg": "be"},
            {"verdict": "accept", "reason": "ok"},
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("full stack app", persistent=True)

        store = StateStore(config.db_path)
        root = store.get_node(store.get_run(run_id).root_node_id)
        domains = store.get_domains(root.node_id)
        assert len(domains) == 2
        names = {d.domain_name for d in domains}
        assert names == {"frontend", "backend"}
        store.close()


class TestNonPersistentUnchanged:
    """Non-persistent runs should still complete as before."""

    @pytest.mark.asyncio
    async def test_non_persistent_completes(self, config):
        adapter = ScriptedAdapter([
            {"action": "solve_directly", "rationale": "simple"},
            {"status": "implemented", "summary": "done",
             "_commit": True, "_commit_file": "out.txt", "_commit_msg": "work"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("simple task", persistent=False)

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "completed"
        assert store.get_node(run.root_node_id).state == NodeState.COMPLETED
        store.close()
