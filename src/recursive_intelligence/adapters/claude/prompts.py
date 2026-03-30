"""Prompt templates for Claude node sessions."""

from __future__ import annotations

from typing import Any

SYSTEM_CONTRACT = """\
You are a node in a recursive coding-agent runtime. You work in an isolated git worktree.

- Solve small tasks directly. Decompose large tasks into children with independent file scopes.
- Commit your work before finishing.
- End each phase with a single JSON object. No wrapping text or markdown around it.
"""


def planning_prompt(task_spec: str, file_scope: list[str] | None = None) -> str:
    scope = ""
    if file_scope:
        scope = f"\nRelevant files: {', '.join(file_scope)}"

    return f"""\
Plan how to accomplish this task. Explore the repo first.
{scope}
## Task
{task_spec}

## Respond with ONE of:

Solve it yourself:
{{"action": "solve_directly", "rationale": "..."}}

Split into children (give each child a distinct file scope):
{{
  "action": "spawn_children",
  "rationale": "...",
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
"""


def execution_prompt(task_spec: str) -> str:
    return f"""\
## Task
{task_spec}

Implement this task. Commit when done.

On success:
{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "..."}}

If blocked:
{{"status": "blocked", "kind": "...", "recoverable": true, "details": "..."}}
"""


def review_prompt(child_id: str, diff: str, summary: str, success_criteria: list[str]) -> str:
    criteria_str = "\n".join(f"  - {c}" for c in success_criteria) if success_criteria else "  - (none specified)"

    return f"""\
Review child {child_id}'s work against these criteria:
{criteria_str}

Summary: {summary or "(none)"}

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

{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "..."}}
"""


def routing_prompt(user_input: str, domains: list[dict[str, Any]], pass_number: int) -> str:
    """Route follow-up work to existing children or spawn new ones."""

    if domains:
        rows = []
        for d in domains:
            patterns = ", ".join(d.get("file_patterns", []))
            rows.append(
                f"| {d['domain_name']} | {d['child_node_id'][:12]} | {patterns} | "
                f"{d.get('child_state', '?')} | {d.get('last_summary', '')[:50]} |"
            )
        domain_table = (
            "| Domain | Child | Files | State | Summary |\n"
            "| --- | --- | --- | --- | --- |\n"
            + "\n".join(rows)
        )
    else:
        domain_table = "(no children yet)"

    return f"""\
## Department (pass {pass_number})

{domain_table}

## User request
{user_input}

## Respond with ONE of:

Route to existing children:
{{"action": "route_to_children", "rationale": "...", "routes": [{{"child_node_id": "...", "domain_name": "...", "task_spec": "..."}}]}}

Spawn new children (for new scope):
{{"action": "spawn_children", "rationale": "...", "children": [{{"idempotency_key": "...", "objective": "...", "success_criteria": [...], "domain_name": "...", "file_patterns": ["..."], "module_scope": "..."}}]}}

Solve directly:
{{"action": "solve_directly", "rationale": "..."}}

Done:
{{"action": "done", "rationale": "..."}}
"""


def reactivation_prompt(original_task: str, previous_summary: str, new_task: str) -> str:
    """Re-activate a child with follow-up work."""

    return f"""\
You previously completed: {original_task}
What you did: {previous_summary or "(unknown)"}

New task: {new_task}

Implement the changes, commit, and return:
{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "..."}}
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

Fix it yourself:
{{"action": "solve_directly", "rationale": "..."}}

Route fixes to the responsible children:
{{
  "action": "route_to_children",
  "rationale": "...",
  "routes": [{{"child_node_id": "...", "domain_name": "...", "task_spec": "describe what the child should fix"}}]
}}"""
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
