"""Prompt templates for Claude node sessions."""

from __future__ import annotations

from typing import Any

SYSTEM_CONTRACT = """\
You are a node in a recursive coding-agent runtime. You work in an isolated git worktree.

- Prefer direct execution when you can complete the task reliably after a brief repo exploration.
- Delegate only when parallelism or context reduction clearly helps.
- Parallel children should own substantial, mostly disjoint domains rather than tiny tasks.
- Same-wave children must be runnable from the same parent snapshot and must not depend on sibling output.
- Route follow-up work back to the existing domain owner instead of spawning duplicate children.
- Stop recursing when another layer would add orchestration overhead more than clarity or throughput.
- Commit your work before finishing.
- End each phase with a single JSON object. No wrapping text or markdown around it.
"""


ROOT_SYSTEM_CONTRACT = """\
You are the root node in a recursive coding-agent runtime. You are also the only node
the human sees directly in the UI.

- Keep your streamed messages conversational and plain-English.
- Do not expose raw JSON envelopes, tool chatter, or internal control-plane jargon to the human.
- Give short progress updates before or between major steps when useful.
- Apply the same delegation rules as other nodes: parallelize only across substantial, mostly disjoint domains.
- Prefer a small number of meaningful children over many tiny ones.
- Commit work before finishing.
- End each phase with a single JSON object. No wrapping text or markdown around it.
"""


DECOMPOSITION_POLICY = """\
## Decomposition Policy
- Solve directly if you can comfortably understand and complete the work within your own context after exploring the repo.
- Delegate only when there are multiple substantial, mostly disjoint domains that can advance in parallel, or when a prerequisite wave must land first.
- Prefer a small number of meaningful children. Avoid many tiny children, single-file micro-tasks, or splits that only separate tightly coupled layers of one feature.
- Children should own outcomes or domains, not implementation fragments.
- Same-wave children must be independent against the current parent snapshot. If child B would need child A's output, do not make them siblings in the same live wave.
- If follow-up work stays inside an existing child domain, keep it with that child.
- Stop decomposing when the next node could reasonably execute the work end-to-end without needing another management layer.
"""


def _completion_option(allow_pause: bool) -> str:
    if allow_pause:
        return 'Pause and wait for more instructions:\n{"action": "pause", "rationale": "..."}'
    return 'Done:\n{"action": "done", "rationale": "..."}'


def _bullet_section(title: str, lines: list[str] | None, empty_message: str) -> str:
    rendered = [str(line).strip() for line in (lines or []) if str(line).strip()]
    if not rendered:
        rendered = [empty_message]
    body = "\n".join(f"- {line}" for line in rendered)
    return f"## {title}\n{body}"


def _ownership_section(title: str, entries: list[dict[str, Any]] | None, empty_message: str) -> str:
    if not entries:
        return _bullet_section(title, None, empty_message)

    lines = []
    for entry in entries:
        scope = entry.get("domain_name") or entry.get("label") or "unnamed scope"
        details = [entry.get("availability", "").strip()]
        if entry.get("state"):
            details.append(f"state: {entry['state']}")
        if entry.get("module_scope"):
            details.append(str(entry["module_scope"]).strip())
        if entry.get("file_patterns"):
            details.append(f"files: {', '.join(entry['file_patterns'])}")
        if entry.get("summary"):
            details.append(f"summary: {entry['summary']}")
        detail_text = "; ".join(part for part in details if part)
        lines.append(f"{scope} — {detail_text}" if detail_text else scope)
    return _bullet_section(title, lines, empty_message)


def planning_prompt(
    task_spec: str,
    file_scope: list[str] | None = None,
    snapshot_summary: list[str] | None = None,
    ownership_context: list[dict[str, Any]] | None = None,
    *,
    allow_pause: bool = False,
) -> str:
    scope = ""
    if file_scope:
        scope = f"\nRelevant files: {', '.join(file_scope)}"
    completion = _completion_option(allow_pause)
    snapshot_block = _bullet_section(
        "Current Snapshot",
        snapshot_summary,
        "No snapshot summary is available yet.",
    )
    ownership_block = _ownership_section(
        "Nearby Ownership And Availability",
        ownership_context,
        "No nearby ownership context is recorded yet.",
    )

    return f"""\
Plan how to accomplish this task. Explore the repo first.
{scope}
{snapshot_block}

{ownership_block}

## Task
{task_spec}

{DECOMPOSITION_POLICY}

Decompose work into snapshot-independent waves, not just non-overlapping file scopes.
A valid wave contains tasks that each child can complete using only the CURRENT parent
snapshot. If some work requires new shared foundation, interfaces, or scaffolding,
spawn only that prerequisite wave now. After those children merge, you will plan the
next wave from the updated snapshot.
Treat only ownership entries marked as merged/available as usable dependencies.
Do not recreate foundation or modules that are already owned upstream or by a sibling.
If another node owns a needed scope but it is not available yet, plan around that
constraint instead of rebuilding it.
If follow-up work belongs to a domain that is already owned nearby, keep that work with
the existing owner instead of spawning a second child for the same domain.
Base the decision on your actual context needs: if the work still fits clearly in one node, do it yourself.

## Respond with ONE of:

Solve it yourself:
{{"action": "solve_directly", "rationale": "..."}}

Split into children (give each child a distinct file scope):
{{
  "action": "spawn_children",
  "rationale": "...",
  "more_waves_expected": false,
  "children": [
    {{
      "idempotency_key": "slug",
      "objective": "what this child does",
      "success_criteria": ["..."],
      "domain_name": "short-name",
      "file_patterns": ["src/module/**"],
      "module_scope": "what this domain owns"
    }}
  ]
}}

Set "more_waves_expected": true only when this wave is intentionally a prerequisite wave
and you expect to plan another wave after these children merge.

{completion}
"""


def execution_prompt(task_spec: str) -> str:
    return f"""\
## Task
{task_spec}

Implement this task. Commit when done.

Return a concise manager handoff that helps the parent decide what to do next.
The handoff should capture what changed, any risks, deviations, discoveries, and
recommended follow-ups. Use empty arrays when a section has nothing important.

On success:
{{
  "status": "implemented",
  "summary": "one-sentence outcome",
  "changed_files": ["..."],
  "commit_sha": "...",
  "handoff": {{
    "deliverables": ["what shipped"],
    "notes": ["important implementation notes"],
    "concerns": ["risks or caveats"],
    "deviations": ["where you intentionally differed from the task"],
    "findings": ["useful discoveries for the planner"],
    "suggested_next_steps": ["recommended follow-up work"]
  }}
}}

If blocked:
{{"status": "blocked", "kind": "...", "recoverable": true, "details": "..."}}
"""


def review_prompt(
    child_id: str,
    diff: str,
    summary: str,
    success_criteria: list[str],
    handoff: str = "",
) -> str:
    criteria_str = "\n".join(f"  - {c}" for c in success_criteria) if success_criteria else "  - (none specified)"
    handoff_block = handoff or "(none)"

    return f"""\
Review child {child_id}'s work against these criteria:
{criteria_str}

Summary: {summary or "(none)"}

Worker handoff:
{handoff_block}

```diff
{diff}
```

Respond with one of:
{{"verdict": "accept", "child_id": "{child_id}", "reason": "..."}}
{{"verdict": "revise", "child_id": "{child_id}", "reason": "...", "follow_up": "what to fix"}}
{{"verdict": "reject", "child_id": "{child_id}", "reason": "..."}}
"""


def revision_prompt(follow_up: str) -> str:
    return f"""\
Revision requested. Make these changes, commit, and return the result.

Feedback: {follow_up}

{{
  "status": "implemented",
  "summary": "...",
  "changed_files": ["..."],
  "commit_sha": "...",
  "handoff": {{
    "deliverables": ["..."],
    "notes": ["..."],
    "concerns": [],
    "deviations": [],
    "findings": [],
    "suggested_next_steps": []
  }}
}}
"""


def routing_prompt(
    user_input: str,
    domains: list[dict[str, Any]],
    pass_number: int,
    snapshot_summary: list[str] | None = None,
    ownership_context: list[dict[str, Any]] | None = None,
    merged_work: list[dict[str, Any]] | None = None,
    *,
    allow_pause: bool = False,
) -> str:
    """Route follow-up work to existing children or spawn new ones."""

    if domains:
        rows = []
        for d in domains:
            patterns = ", ".join(d.get("file_patterns", []))
            rows.append(
                f"| {d['domain_name']} | {d['child_node_id'][:12]} | {d.get('availability', '?')} | "
                f"{patterns} | {d.get('child_state', '?')} | {d.get('last_summary', '')[:50]} |"
            )
        domain_table = (
            "| Domain | Child | Availability | Files | State | Summary |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(rows)
        )
    else:
        domain_table = "(no children yet)"
    completion = _completion_option(allow_pause)
    snapshot_block = _bullet_section(
        "Current Snapshot",
        snapshot_summary,
        "No snapshot summary is available yet.",
    )
    ownership_block = _ownership_section(
        "Upstream Ownership Already In Scope",
        ownership_context,
        "No upstream ownership context is recorded yet.",
    )
    merged_block = _ownership_section(
        "Previously Merged Child Work In This Branch",
        merged_work,
        "No child work has been merged into this branch yet.",
    )

    return f"""\
## Department (pass {pass_number})

{snapshot_block}

{ownership_block}

{merged_block}

{domain_table}

## User request
{user_input}

You are a manager node with existing child domains. Do not implement code yourself.
Previously accepted child work is already merged into your current worktree. New child
tasks must be executable immediately against this snapshot. If a requested outcome depends
on foundation that is not yet merged, spawn or route ONLY that prerequisite wave first.
Only work marked as merged/available can be treated as a live dependency.
Do not recreate domains that are already owned upstream or already merged here.
If follow-up stays inside an existing child domain, route it back to that same child.
Do not spawn a second child for dependent follow-up in the same domain.
Route at most one task to each child in a single wave.
Prefer the minimum number of child tasks needed to keep work parallel and domain-disjoint.
Do not fan out into many small routes when one child can own the next meaningful step.

## Respond with ONE of:

Route to existing children:
{{"action": "route_to_children", "rationale": "...", "more_waves_expected": false, "routes": [{{"child_node_id": "...", "domain_name": "...", "task_spec": "..."}}]}}

Spawn new children (for new scope):
{{"action": "spawn_children", "rationale": "...", "more_waves_expected": false, "children": [{{"idempotency_key": "...", "objective": "...", "success_criteria": [...], "domain_name": "...", "file_patterns": ["..."], "module_scope": "..."}}]}}

Set "more_waves_expected": true only when this wave is intentionally a prerequisite wave
and you need another planning pass after these children merge.

{completion}
"""


def reactivation_prompt(original_task: str, previous_handoff: str, new_task: str) -> str:
    """Re-activate a child with follow-up work."""

    return f"""\
You previously completed: {original_task}
Previous handoff:
{previous_handoff or "(unknown)"}

New task: {new_task}

Implement the changes, commit, and return:
{{
  "status": "implemented",
  "summary": "...",
  "changed_files": ["..."],
  "commit_sha": "...",
  "handoff": {{
    "deliverables": ["..."],
    "notes": ["..."],
    "concerns": [],
    "deviations": [],
    "findings": [],
    "suggested_next_steps": []
  }}
}}
"""


def verification_retry_prompt(
    task_spec: str,
    test_command: str,
    test_output: str,
    has_children: bool = False,
) -> str:
    """Re-plan after test verification failed."""

    if has_children:
        options = """\
## Respond with ONE of:

Route fixes to the responsible children:
{{
  "action": "route_to_children",
  "rationale": "...",
  "routes": [{{"child_node_id": "...", "domain_name": "...", "task_spec": "describe what the child should fix"}}]
}}

Spawn a new child if no existing domain cleanly owns the fix:
{{"action": "spawn_children", "rationale": "...", "children": [{{"idempotency_key": "...", "objective": "...", "success_criteria": ["..."], "domain_name": "...", "file_patterns": ["..."], "module_scope": "..."}}]}}"""
    else:
        options = """\
## Respond with:
{{"action": "solve_directly", "rationale": "..."}}"""

    return f"""\
Tests failed after implementation. Fix the failures and try again.

## Original task
{task_spec}

## Test command
{test_command}

## Test output (truncated)
```
{test_output}
```

Read the failing test output carefully. Identify exactly what went wrong and fix it.
Commit your changes when done.

{options}
"""


def conflict_resolution_prompt(child_id: str, conflict_files: list[str], conflict_diff: str) -> str:
    """Resolve merge conflicts from a child's cherry-pick."""

    files_list = ", ".join(conflict_files)

    return f"""\
Cherry-picking child {child_id} conflicted in: {files_list}

```diff
{conflict_diff[:10000]}
```

Resolve all conflicts, stage, and commit. Return:
{{"status": "resolved", "summary": "...", "resolved_files": ["..."]}}

If irreconcilable:
{{"status": "irreconcilable", "reason": "..."}}
"""
