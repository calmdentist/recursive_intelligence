"""Integration tests for the orchestrator – full recursive flow with mock adapter."""

import subprocess
from pathlib import Path

import pytest

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.orchestrator import Orchestrator, get_node_tree
from recursive_intelligence.runtime.state_store import NodeState, StateStore


class ScriptedAdapter(AgentAdapter):
    """Mock adapter that returns scripted responses based on mode and call count.

    Responses are popped from the queue in order. Each response is a dict
    that becomes the NodeResult.raw, plus optional commit behavior.
    """

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
            raise RuntimeError("ScriptedAdapter ran out of responses")

        resp = self._responses.pop(0)
        self._call_log.append({"prompt": prompt[:100], "mode": mode, "worktree": str(worktree)})

        # Optionally make a commit in the worktree
        if resp.get("_commit"):
            _make_commit(Path(worktree), resp.get("_commit_file", "output.txt"), resp.get("_commit_msg", "mock"))

        raw = {k: v for k, v in resp.items() if not k.startswith("_")}
        return NodeResult(
            session_id=f"session-{len(self._call_log)}",
            raw=raw,
            result_text="",
            cost=CostRecord(total_usd=0.01),
            stop_reason="end_turn",
        )


class NoopAdapter(AgentAdapter):
    @property
    def name(self) -> str:
        return "noop"

    async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
        raise RuntimeError(f"Unexpected adapter call in mode={mode}")


def _make_commit(worktree: Path, filename: str, message: str) -> None:
    (worktree / filename).write_text(f"content of {filename}\n")
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


class TestSolveDirectly:
    """Root plans to solve directly, executes, completes."""

    @pytest.mark.asyncio
    async def test_solve_directly_flow(self, config):
        adapter = ScriptedAdapter([
            # Plan: solve directly
            {"action": "solve_directly", "rationale": "simple task"},
            # Execute: implemented
            {"action": "solve_directly", "status": "implemented", "summary": "done",
             "changed_files": ["output.txt"], "commit_sha": "abc",
             "_commit": True, "_commit_file": "output.txt", "_commit_msg": "implement feature"},
        ])
        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("add a feature")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "completed"

        nodes = store.get_run_nodes(run_id)
        assert len(nodes) == 1
        assert nodes[0].state == NodeState.COMPLETED
        store.close()

        # Verify adapter was called twice: plan + execute
        assert len(adapter.calls) == 2
        assert adapter.calls[0]["mode"] == "plan"
        assert adapter.calls[1]["mode"] == "execute"

    @pytest.mark.asyncio
    async def test_resume_run_from_verifying_completes(self, config):
        (config.repo_root / "ok.txt").write_text("ok")
        config.ensure_dirs()
        store = StateStore(config.db_path)
        run = store.create_run(str(config.repo_root), "verify task", test_command="sh -c 'test -f ok.txt'")
        node = store.create_node(
            run.run_id,
            "verify task",
            worktree_path=str(config.repo_root),
            branch_name="ri/verify",
        )
        store.set_root_node(run.run_id, node.node_id)
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.EXECUTING)
        store.transition_node(node.node_id, NodeState.VERIFYING)
        store.close()

        orch = Orchestrator(config, NoopAdapter())
        await orch.resume_run(run.run_id)

        store = StateStore(config.db_path)
        run = store.get_run(run.run_id)
        root = store.get_node(node.node_id)
        verify_events = [
            e for e in store.get_node_events(node.node_id)
            if e.event_type == "verification_result"
        ]
        assert run.status == "completed"
        assert root.state == NodeState.COMPLETED
        assert len(verify_events) == 1
        assert verify_events[0].data["passed"] is True
        store.close()


class TestRecursiveFlow:
    """Root spawns children, children execute, parent reviews and merges."""

    @pytest.mark.asyncio
    async def test_spawn_review_merge(self, config):
        adapter = ScriptedAdapter([
            # 1. Root plans: spawn 2 children
            {
                "action": "spawn_children",
                "rationale": "complex task",
                "children": [
                    {"idempotency_key": "child-a", "objective": "implement feature A",
                     "success_criteria": ["A works"]},
                    {"idempotency_key": "child-b", "objective": "implement feature B",
                     "success_criteria": ["B works"]},
                ],
            },
            # 2. Child A plans: solve directly
            {"action": "solve_directly", "rationale": "simple subtask"},
            # 3. Child A executes
            {"status": "implemented", "summary": "added A",
             "_commit": True, "_commit_file": "feature_a.txt", "_commit_msg": "add feature A"},
            # 4. Child B plans: solve directly
            {"action": "solve_directly", "rationale": "simple subtask"},
            # 5. Child B executes
            {"status": "implemented", "summary": "added B",
             "_commit": True, "_commit_file": "feature_b.txt", "_commit_msg": "add feature B"},
            # 6. Root reviews child A: accept
            {"verdict": "accept", "reason": "looks good"},
            # 7. Root reviews child B: accept
            {"verdict": "accept", "reason": "looks good"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("build features A and B")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "completed"

        nodes = store.get_run_nodes(run_id)
        assert len(nodes) == 3  # root + 2 children

        root = store.get_node(run.root_node_id)
        assert root.state == NodeState.COMPLETED

        children = store.get_children(root.node_id)
        assert len(children) == 2
        assert all(c.state == NodeState.COMPLETED for c in children)

        # Verify the root worktree has both features merged
        root_wt = Path(root.worktree_path)
        assert (root_wt / "feature_a.txt").exists()
        assert (root_wt / "feature_b.txt").exists()

        store.close()

        # Verify call sequence
        modes = [c["mode"] for c in adapter.calls]
        assert modes == ["plan", "plan", "execute", "plan", "execute", "review", "review"]

    @pytest.mark.asyncio
    async def test_verification_runs_once_after_root_merge(self, git_repo):
        config = RuntimeConfig(repo_root=git_repo, max_parallel_children=1)
        adapter = ScriptedAdapter([
            {
                "action": "spawn_children",
                "rationale": "split task",
                "children": [
                    {"idempotency_key": "child-a", "objective": "implement A", "success_criteria": ["A works"]},
                    {"idempotency_key": "child-b", "objective": "implement B", "success_criteria": ["B works"]},
                ],
            },
            {"action": "solve_directly", "rationale": "simple"},
            {"status": "implemented", "summary": "added A",
             "_commit": True, "_commit_file": "a.txt", "_commit_msg": "add A"},
            {"action": "solve_directly", "rationale": "simple"},
            {"status": "implemented", "summary": "added B",
             "_commit": True, "_commit_file": "b.txt", "_commit_msg": "add B"},
            {"verdict": "accept", "reason": "looks good"},
            {"verdict": "accept", "reason": "looks good"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run(
            "build A and B",
            test_command="sh -c 'test -f a.txt && test -f b.txt'",
        )

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        children = store.get_children(root.node_id)
        root_verify = [
            e for e in store.get_node_events(root.node_id)
            if e.event_type == "verification_result"
        ]

        assert run.status == "completed"
        assert root.state == NodeState.COMPLETED
        assert len(root_verify) == 1
        assert root_verify[0].data["passed"] is True
        assert all(child.state == NodeState.COMPLETED for child in children)
        for child in children:
            child_verify = [
                e for e in store.get_node_events(child.node_id)
                if e.event_type == "verification_result"
            ]
            assert child_verify == []
        store.close()

    @pytest.mark.asyncio
    async def test_tree_output(self, config):
        adapter = ScriptedAdapter([
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "c1", "objective": "task 1", "success_criteria": ["done"]},
             ]},
            {"action": "solve_directly", "rationale": "simple"},
            {"status": "implemented", "summary": "done",
             "_commit": True, "_commit_file": "out.txt", "_commit_msg": "work"},
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("parent task")

        store = StateStore(config.db_path)
        tree = get_node_tree(store, run_id)
        assert len(tree) == 1
        root = tree[0]
        assert root["state"] == "completed"
        assert len(root["children"]) == 1
        assert root["children"][0]["state"] == "completed"
        store.close()


class TestReviseLoop:
    """Parent requests revision, child re-executes, parent accepts on second review."""

    @pytest.mark.asyncio
    async def test_revise_then_accept(self, config):
        adapter = ScriptedAdapter([
            # 1. Root plans: spawn one child
            {"action": "spawn_children", "rationale": "delegate",
             "children": [
                 {"idempotency_key": "c1", "objective": "write feature",
                  "success_criteria": ["has tests", "passes lint"]},
             ]},
            # 2. Child plans: solve directly
            {"action": "solve_directly", "rationale": "straightforward"},
            # 3. Child executes (first attempt — missing tests)
            {"status": "implemented", "summary": "added feature but no tests",
             "_commit": True, "_commit_file": "feature.py", "_commit_msg": "add feature v1"},
            # 4. Root reviews: revise (needs tests)
            {"verdict": "revise", "reason": "missing tests",
             "follow_up": "add unit tests for the feature"},
            # 5. Child re-executes (second attempt — with tests)
            {"status": "implemented", "summary": "added tests",
             "_commit": True, "_commit_file": "test_feature.py", "_commit_msg": "add tests"},
            # 6. Root reviews again: accept
            {"verdict": "accept", "reason": "now has tests"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("build feature with tests")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "completed"

        root = store.get_node(run.root_node_id)
        assert root.state == NodeState.COMPLETED

        children = store.get_children(root.node_id)
        assert len(children) == 1
        child = children[0]
        assert child.state == NodeState.COMPLETED

        # Verify child went through revision
        child_events = store.get_node_events(child.node_id)
        event_types = [e.event_type for e in child_events]
        downstream_tasks = [e for e in child_events if e.event_type == "downstream_task"]
        assert len(downstream_tasks) == 1
        assert downstream_tasks[0].data["kind"] == "revision"
        assert downstream_tasks[0].data["task_spec"] == "add unit tests for the feature"
        # Should have two execution_result events (initial + revision)
        assert event_types.count("execution_result") == 2

        # Verify root worktree has both files from the child
        root_wt = Path(root.worktree_path)
        assert (root_wt / "feature.py").exists()
        assert (root_wt / "test_feature.py").exists()

        store.close()

        # Verify call sequence: plan, plan, execute, review, execute(revision), review
        modes = [c["mode"] for c in adapter.calls]
        assert modes == ["plan", "plan", "execute", "review", "execute", "review"]

    @pytest.mark.asyncio
    async def test_revise_twice_then_accept(self, config):
        """Child gets revised twice before being accepted."""
        adapter = ScriptedAdapter([
            # Root spawns child
            {"action": "spawn_children", "rationale": "delegate",
             "children": [
                 {"idempotency_key": "c1", "objective": "task",
                  "success_criteria": ["correct"]},
             ]},
            # Child plans + executes (v1)
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "v1",
             "_commit": True, "_commit_file": "v1.txt", "_commit_msg": "v1"},
            # Review 1: revise
            {"verdict": "revise", "reason": "not quite", "follow_up": "fix X"},
            # Child re-executes (v2)
            {"status": "implemented", "summary": "v2",
             "_commit": True, "_commit_file": "v2.txt", "_commit_msg": "v2"},
            # Review 2: revise again
            {"verdict": "revise", "reason": "still wrong", "follow_up": "fix Y"},
            # Child re-executes (v3)
            {"status": "implemented", "summary": "v3",
             "_commit": True, "_commit_file": "v3.txt", "_commit_msg": "v3"},
            # Review 3: accept
            {"verdict": "accept", "reason": "finally correct"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("iterative task")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "completed"

        children = store.get_children(store.get_run(run_id).root_node_id)
        child_events = store.get_node_events(children[0].node_id)
        assert [e.event_type for e in child_events].count("execution_result") == 3
        downstream_tasks = [e for e in child_events if e.event_type == "downstream_task"]
        assert len(downstream_tasks) == 2
        assert all(e.data["kind"] == "revision" for e in downstream_tasks)
        store.close()

    @pytest.mark.asyncio
    async def test_multi_child_revise_one(self, config):
        """Two children: A accepted, B revised. Stale verdicts must not cause
        premature MERGING — B must be re-reviewed after revision."""
        adapter = ScriptedAdapter([
            # Root spawns 2 children
            {"action": "spawn_children", "rationale": "split",
             "children": [
                 {"idempotency_key": "ca", "objective": "feature A",
                  "success_criteria": ["A works"]},
                 {"idempotency_key": "cb", "objective": "feature B",
                  "success_criteria": ["B works"]},
             ]},
            # Child A: plan + execute
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "A done",
             "_commit": True, "_commit_file": "a.txt", "_commit_msg": "A"},
            # Child B: plan + execute (first attempt)
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "B v1",
             "_commit": True, "_commit_file": "b_v1.txt", "_commit_msg": "B v1"},
            # Round 1 review: accept A, revise B
            {"verdict": "accept", "reason": "A looks good"},
            {"verdict": "revise", "reason": "B needs tests", "follow_up": "add tests for B"},
            # Child B re-executes (second attempt)
            {"status": "implemented", "summary": "B v2 with tests",
             "_commit": True, "_commit_file": "b_v2.txt", "_commit_msg": "B v2"},
            # Round 2 review: only B needs re-review (A was auto-accepted, no new work)
            {"verdict": "accept", "reason": "B now has tests"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("features A and B")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        assert run.status == "completed"

        root = store.get_node(run.root_node_id)
        assert root.state == NodeState.COMPLETED

        children = store.get_children(root.node_id)
        assert len(children) == 2
        assert all(c.state == NodeState.COMPLETED for c in children)

        # Verify both children's work was merged into root worktree
        root_wt = Path(root.worktree_path)
        assert (root_wt / "a.txt").exists()
        # B should have both v1 and v2 files (all commits cherry-picked)
        assert (root_wt / "b_v1.txt").exists()
        assert (root_wt / "b_v2.txt").exists()

        # Round 1: 2 reviews (A accepted, B revised)
        # Round 2: 1 review (B only — A auto-accepted since it has no new work)
        modes = [c["mode"] for c in adapter.calls]
        assert modes.count("review") == 3
        store.close()


class TestCherryPickConflict:
    """Cherry-pick fails due to conflicting changes from two children."""

    @pytest.mark.asyncio
    async def test_conflict_fails_merge(self, config, git_repo):
        """Two children modify the same file on the same line, causing a conflict."""

        class ConflictAdapter(AgentAdapter):
            """Adapter that creates conflicting edits to the same file."""

            def __init__(self):
                self._responses = [
                    # Root spawns 2 children
                    {"action": "spawn_children", "rationale": "parallel",
                     "children": [
                         {"idempotency_key": "c1", "objective": "edit A", "success_criteria": ["done"]},
                         {"idempotency_key": "c2", "objective": "edit B", "success_criteria": ["done"]},
                     ]},
                    # Child A: solve directly
                    {"action": "solve_directly", "rationale": "ok"},
                    # Child A: execute (writes "version A" to README)
                    {"status": "implemented", "summary": "A"},
                    # Child B: solve directly
                    {"action": "solve_directly", "rationale": "ok"},
                    # Child B: execute (writes "version B" to README — conflicts)
                    {"status": "implemented", "summary": "B"},
                    # Reviews
                    {"verdict": "accept", "reason": "ok"},
                    {"verdict": "accept", "reason": "ok"},
                ]
                self._call_count = 0

            @property
            def name(self):
                return "conflict"

            async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
                resp = self._responses.pop(0)
                self._call_count += 1
                raw = {k: v for k, v in resp.items() if not k.startswith("_")}

                # On execute calls, write conflicting content
                if mode == "execute" and self._call_count == 3:
                    # Child A
                    (Path(worktree) / "README.md").write_text("version A\n")
                    subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
                    subprocess.run(["git", "commit", "-m", "A"], cwd=str(worktree), capture_output=True)
                elif mode == "execute" and self._call_count == 5:
                    # Child B — same file, different content
                    (Path(worktree) / "README.md").write_text("version B\n")
                    subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
                    subprocess.run(["git", "commit", "-m", "B"], cwd=str(worktree), capture_output=True)

                return NodeResult(
                    session_id=f"s-{self._call_count}",
                    raw=raw, result_text="", cost=CostRecord(total_usd=0.01),
                    stop_reason="end_turn",
                )

        orch = Orchestrator(config, ConflictAdapter())
        run_id = await orch.start_run("parallel readme edits")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        # Merge should fail due to conflict on second cherry-pick
        assert run.status == "failed"

        root = store.get_node(run.root_node_id)
        events = store.get_node_events(root.node_id)
        conflict_events = [
            e for e in events
            if e.event_type == "child_integrated" and e.data.get("status") == "conflict"
        ]
        conflict_tasks = [
            e for e in events
            if e.event_type == "downstream_task" and e.data.get("kind") == "merge_conflict"
        ]
        assert len(conflict_events) == 1
        assert len(conflict_tasks) == 1
        store.close()

    @pytest.mark.asyncio
    async def test_conflict_resolution_records_typed_work_and_completes(self, config):
        class ConflictResolvingAdapter(AgentAdapter):
            def __init__(self):
                self._responses = [
                    {"action": "spawn_children", "rationale": "parallel",
                     "children": [
                         {"idempotency_key": "c1", "objective": "edit A", "success_criteria": ["done"]},
                         {"idempotency_key": "c2", "objective": "edit B", "success_criteria": ["done"]},
                     ]},
                    {"action": "solve_directly", "rationale": "ok"},
                    {"status": "implemented", "summary": "A"},
                    {"action": "solve_directly", "rationale": "ok"},
                    {"status": "implemented", "summary": "B"},
                    {"verdict": "accept", "reason": "ok"},
                    {"verdict": "accept", "reason": "ok"},
                    {"status": "resolved", "summary": "merged README"},
                ]
                self.calls = 0

            @property
            def name(self):
                return "conflict-resolver"

            async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
                resp = self._responses.pop(0)
                self.calls += 1
                raw = {k: v for k, v in resp.items() if not k.startswith("_")}

                if mode == "execute" and self.calls == 3:
                    (Path(worktree) / "README.md").write_text("version A\n")
                    subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
                    subprocess.run(["git", "commit", "-m", "A"], cwd=str(worktree), capture_output=True)
                elif mode == "execute" and self.calls == 5:
                    (Path(worktree) / "README.md").write_text("version B\n")
                    subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
                    subprocess.run(["git", "commit", "-m", "B"], cwd=str(worktree), capture_output=True)
                elif mode == "execute" and self.calls == 8:
                    (Path(worktree) / "README.md").write_text("resolved version\n")
                    subprocess.run(["git", "add", "README.md"], cwd=str(worktree), capture_output=True)

                return NodeResult(
                    session_id=f"s-{self.calls}",
                    raw=raw,
                    result_text="",
                    cost=CostRecord(total_usd=0.01),
                    stop_reason="end_turn",
                )

        orch = Orchestrator(config, ConflictResolvingAdapter())
        run_id = await orch.start_run("parallel readme edits")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        events = store.get_node_events(root.node_id)
        task_events = [
            e for e in events
            if e.event_type == "downstream_task" and e.data.get("kind") == "merge_conflict"
        ]
        task_results = [
            e for e in events
            if e.event_type == "downstream_task_result" and e.data.get("kind") == "merge_conflict"
        ]

        assert run.status == "completed"
        assert root.state == NodeState.COMPLETED
        assert len(task_events) == 1
        assert task_events[0].data["child_id"].startswith("node-")
        assert len(task_results) == 1
        assert task_results[0].data["status"] == "completed"
        assert task_results[0].data["resolved_sha"]
        assert (Path(root.worktree_path) / "README.md").read_text() == "resolved version\n"
        store.close()


class TestChildFailure:
    """Child fails, parent handles it gracefully."""

    @pytest.mark.asyncio
    async def test_child_execution_failure(self, config):
        adapter = ScriptedAdapter([
            # Root: spawn one child
            {"action": "spawn_children", "rationale": "delegate",
             "children": [
                 {"idempotency_key": "c1", "objective": "do thing", "success_criteria": ["works"]},
             ]},
            # Child plans: solve directly
            {"action": "solve_directly", "rationale": "ok"},
            # Child executes: blocked
            {"status": "blocked", "kind": "missing dependency", "details": "can't find lib"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("task that fails")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        # Run fails because all children failed and none accepted
        assert run.status == "failed"

        nodes = store.get_run_nodes(run_id)
        child_nodes = [n for n in nodes if n.parent_id is not None]
        assert len(child_nodes) == 1
        assert child_nodes[0].state == NodeState.FAILED
        store.close()

    @pytest.mark.asyncio
    async def test_review_failure_fails_run(self, config):
        class ReviewFailureAdapter(ScriptedAdapter):
            async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
                if mode == "review":
                    raise RuntimeError("review backend unavailable")
                return await super().run(
                    prompt,
                    worktree,
                    mode,
                    system_prompt=system_prompt,
                    resume_session_id=resume_session_id,
                    on_message=on_message,
                    is_root=is_root,
                )

        adapter = ReviewFailureAdapter([
            {"action": "spawn_children", "rationale": "delegate",
             "children": [
                 {"idempotency_key": "c1", "objective": "do thing", "success_criteria": ["works"]},
             ]},
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "done",
             "_commit": True, "_commit_file": "out.txt", "_commit_msg": "work"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("task needing review")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        assert run.status == "failed"
        assert root.state == NodeState.FAILED
        store.close()


class TestEscalations:
    @pytest.mark.asyncio
    async def test_root_request_upstream_pauses_run(self, config):
        adapter = ScriptedAdapter([
            {"action": "solve_directly", "rationale": "need credentials"},
            {
                "status": "request_upstream",
                "request": {
                    "kind": "capability",
                    "summary": "Need STRIPE_SECRET_KEY to continue",
                    "details": "Provide the production or test Stripe secret key.",
                    "action_requested": "Set STRIPE_SECRET_KEY",
                    "requires_input": True,
                    "urgency": "high",
                },
            },
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("add Stripe billing")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        request_events = [
            e for e in store.get_node_events(root.node_id)
            if e.event_type == "root_request_upstream"
        ]

        assert run.status == "paused"
        assert root.state == NodeState.PAUSED
        assert run.telemetry["user_interruptions_count"] == 1
        assert run.telemetry["root_escalations_count"] == 1
        assert run.telemetry["root_requests_count"] == 1
        assert len(request_events) == 1
        assert request_events[0].data["request"]["kind"] == "capability"
        store.close()

    @pytest.mark.asyncio
    async def test_child_request_bubbles_to_root(self, config):
        adapter = ScriptedAdapter([
            {
                "action": "spawn_children",
                "rationale": "delegate setup work",
                "children": [
                    {"idempotency_key": "billing", "objective": "set up billing integration"},
                ],
            },
            {"action": "solve_directly", "rationale": "need credentials"},
            {
                "status": "request_upstream",
                "request": {
                    "kind": "capability",
                    "summary": "Need a Stripe account and API key",
                    "details": "Create a Stripe account and provide test credentials.",
                    "action_requested": "Create Stripe account",
                    "requires_input": True,
                    "urgency": "normal",
                },
            },
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("add billing")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        child = store.get_children(root.node_id)[0]
        root_events = store.get_node_events(root.node_id)

        assert run.status == "paused"
        assert root.state == NodeState.PAUSED
        assert child.state == NodeState.PAUSED
        assert run.telemetry["user_interruptions_count"] == 1
        assert any(e.event_type == "root_request_upstream" for e in root_events)
        assert any(e.event_type == "request_upstream" for e in store.get_node_events(child.node_id))
        store.close()

    @pytest.mark.asyncio
    async def test_continue_routes_response_downstream_to_child_without_root_replan(self, config):
        adapter = ScriptedAdapter([
            {
                "action": "spawn_children",
                "rationale": "delegate billing setup",
                "children": [
                    {"idempotency_key": "billing", "objective": "set up billing integration"},
                ],
            },
            {"action": "solve_directly", "rationale": "need credentials"},
            {
                "status": "request_upstream",
                "request": {
                    "kind": "capability",
                    "summary": "Need STRIPE_SECRET_KEY",
                    "details": "Provide a Stripe test key",
                    "action_requested": "Set STRIPE_SECRET_KEY",
                    "requires_input": True,
                },
            },
            {
                "status": "implemented",
                "summary": "configured billing",
                "_commit": True,
                "_commit_file": "billing.py",
                "_commit_msg": "billing configured",
            },
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("add billing")
        await orch.continue_run(run_id, "Use STRIPE_SECRET_KEY from the environment")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        child = store.get_children(root.node_id)[0]
        child_events = store.get_node_events(child.node_id)
        response_events = [e for e in child_events if e.event_type == "response_downstream"]

        assert run.status == "completed"
        assert root.state == NodeState.COMPLETED
        assert child.state == NodeState.COMPLETED
        assert len(response_events) == 1
        assert response_events[0].data["response_text"] == "Use STRIPE_SECRET_KEY from the environment"
        modes = [c["mode"] for c in adapter.calls]
        assert modes == ["plan", "plan", "execute", "execute", "review"]
        store.close()

    @pytest.mark.asyncio
    async def test_resolve_request_by_id_routes_approval_downstream(self, config):
        adapter = ScriptedAdapter([
            {
                "action": "spawn_children",
                "rationale": "delegate billing setup",
                "children": [
                    {"idempotency_key": "billing", "objective": "set up billing integration"},
                ],
            },
            {"action": "solve_directly", "rationale": "need approval"},
            {
                "status": "request_upstream",
                "request": {
                    "kind": "approval",
                    "summary": "Can I enable Stripe webhooks in dev?",
                    "details": "This creates a local webhook endpoint and test signing secret.",
                    "action_requested": "Approve Stripe webhook setup",
                    "requires_input": True,
                },
            },
            {
                "status": "implemented",
                "summary": "configured webhooks",
                "_commit": True,
                "_commit_file": "billing.py",
                "_commit_msg": "billing configured",
            },
            {"verdict": "accept", "reason": "ok"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("add billing")

        store = StateStore(config.db_path)
        inbox = store.get_run_inbox(run_id)
        assert len(inbox) == 1
        request_id = inbox[0]["request_id"]
        store.close()

        await orch.resolve_request(run_id, request_id, "", resolution="approve")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        child = store.get_children(root.node_id)[0]
        child_events = store.get_node_events(child.node_id)
        response_events = [e for e in child_events if e.event_type == "response_downstream"]
        root_events = store.get_node_events(root.node_id)
        resolved_events = [e for e in root_events if e.event_type == "request_resolved"]

        assert run.status == "completed"
        assert root.state == NodeState.COMPLETED
        assert child.state == NodeState.COMPLETED
        assert len(response_events) == 1
        assert response_events[0].data["resolution"] == "approve"
        assert response_events[0].data["response_text"] == "Approved."
        assert len(resolved_events) == 1
        assert resolved_events[0].data["resolution"] == "approve"
        modes = [c["mode"] for c in adapter.calls]
        assert modes == ["plan", "plan", "execute", "execute", "review"]
        store.close()


class TestOwnershipEnforcement:
    @pytest.mark.asyncio
    async def test_merge_fails_when_child_changes_out_of_scope_files(self, config):
        adapter = ScriptedAdapter([
            {
                "action": "spawn_children",
                "rationale": "split frontend work",
                "children": [
                    {
                        "idempotency_key": "frontend",
                        "objective": "update frontend",
                        "success_criteria": ["frontend works"],
                        "domain_name": "frontend",
                        "file_patterns": ["src/ui/**"],
                    },
                ],
            },
            {"action": "solve_directly", "rationale": "straightforward"},
            {
                "status": "implemented",
                "summary": "changed the server instead",
                "_commit": True,
                "_commit_file": "server.py",
                "_commit_msg": "oops out of scope",
            },
            {"verdict": "accept", "reason": "accepted by reviewer"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("update the frontend")

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        root = store.get_node(run.root_node_id)
        violation_events = [
            e for e in store.get_node_events(root.node_id)
            if e.event_type == "ownership_violation"
        ]

        assert run.status == "failed"
        assert root.state == NodeState.FAILED
        assert run.telemetry["ownership_violations_count"] == 1
        assert len(violation_events) == 1
        assert violation_events[0].data["violations"] == ["server.py"]
        store.close()


class TestWorktreeCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_worktrees(self, config):
        adapter = ScriptedAdapter([
            {"action": "solve_directly", "rationale": "simple"},
            {"status": "implemented", "summary": "done",
             "_commit": True, "_commit_file": "out.txt", "_commit_msg": "work"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("task")

        store = StateStore(config.db_path)
        nodes = store.get_run_nodes(run_id)
        wt_path = Path(nodes[0].worktree_path)
        assert wt_path.exists()
        store.close()

        orch.cleanup_worktrees(run_id)
        assert not wt_path.exists()


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_completed_run_raises(self, config):
        adapter = ScriptedAdapter([
            {"action": "solve_directly", "rationale": "simple"},
            {"status": "implemented", "summary": "done",
             "_commit": True, "_commit_file": "out.txt", "_commit_msg": "work"},
        ])

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("task")

        with pytest.raises(ValueError, match="not resumable"):
            await orch.resume_run(run_id)
