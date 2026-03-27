"""Prompt templates for Claude node sessions."""

from __future__ import annotations

from typing import Any

SYSTEM_CONTRACT = """\
You are a node in a recursive coding-agent runtime. You are working inside an isolated git worktree.

## Rules
- Work only inside your current worktree. Never modify files outside it.
- If the task is small enough to solve directly, solve it yourself.
- If the task is large (many files, distinct subtasks), decompose into independent children.
- Make child tasks narrow, specific, and independently testable.
- Always commit your work before finishing.
- At the end of each phase, emit ONLY a JSON object — no markdown, no explanation outside the JSON.

## JSON Envelope
Your final message in each phase MUST be a single JSON object. No wrapping text.
"""


def planning_prompt(task_spec: str, file_scope: list[str] | None = None) -> str:
    scope = ""
    if file_scope:
        scope = f"\nRelevant files/directories: {', '.join(file_scope)}"

    return f"""\
Inspect the repository and decide how to accomplish this task.
{scope}
## Task
{task_spec}

## Instructions
1. Explore the repo structure and relevant files.
2. Decide: can you solve this directly, or should it be split into subtasks?
3. Return your decision as a JSON object.

## If solving directly
```json
{{"action": "solve_directly", "rationale": "why this is simple enough to do yourself"}}
```

## If spawning children
Each child should own a clear domain (a set of files/modules it is responsible for).
```json
{{
  "action": "spawn_children",
  "rationale": "why this needs decomposition",
  "children": [
    {{
      "idempotency_key": "short-unique-slug",
      "objective": "specific task description for child",
      "success_criteria": ["criterion 1", "criterion 2"],
      "domain_name": "short-domain-slug",
      "file_patterns": ["src/module/**", "tests/test_module*"],
      "module_scope": "Human-readable description of what this domain owns"
    }}
  ]
}}
```

Return ONLY the JSON object. No other text.
"""


def execution_prompt(task_spec: str) -> str:
    return f"""\
## Task
{task_spec}

## Instructions
1. Implement the task in your worktree.
2. Run any relevant tests if they exist.
3. Stage and commit all changes with a descriptive message.
4. Return a JSON result.

## On success
```json
{{"status": "implemented", "summary": "what you did", "changed_files": ["file1.py", "file2.py"], "commit_sha": "abc123..."}}
```

## If blocked
```json
{{"status": "blocked", "kind": "description of blocker", "recoverable": true, "details": "what went wrong"}}
```

Return ONLY the JSON object after committing.
"""


def review_prompt(child_id: str, diff: str, summary: str, success_criteria: list[str]) -> str:
    criteria_str = "\n".join(f"  - {c}" for c in success_criteria) if success_criteria else "  - (none specified)"

    return f"""\
## Review child work

Child ID: {child_id}

### Success criteria
{criteria_str}

### Child's summary
{summary or "(no summary provided)"}

### Diff
```diff
{diff}
```

## Instructions
Review the diff against the success criteria. Return a JSON verdict:

### If the work meets criteria
```json
{{"verdict": "accept", "child_id": "{child_id}", "reason": "why it's acceptable"}}
```

### If it needs changes
```json
{{"verdict": "revise", "child_id": "{child_id}", "reason": "what's wrong", "follow_up": "specific instructions for revision"}}
```

### If it's fundamentally wrong
```json
{{"verdict": "reject", "child_id": "{child_id}", "reason": "why it should be discarded"}}
```

Return ONLY the JSON object.
"""


def revision_prompt(follow_up: str) -> str:
    return f"""\
## Revision requested

Your previous work needs changes.

### Feedback
{follow_up}

## Instructions
1. Make the requested changes in your worktree.
2. Stage and commit all changes.
3. Return an updated JSON result.

```json
{{"status": "implemented", "summary": "what you changed", "changed_files": ["..."], "commit_sha": "..."}}
```

Return ONLY the JSON object after committing.
"""


def routing_prompt(user_input: str, domains: list[dict[str, Any]], pass_number: int) -> str:
    """Prompt for the root node on pass 2+ — route work to existing children or spawn new ones."""

    if domains:
        table_rows = []
        for d in domains:
            patterns = ", ".join(d.get("file_patterns", []))
            table_rows.append(
                f"| {d['domain_name']} | {d['child_node_id'][:12]} | {patterns} | "
                f"{d.get('module_scope', '')} | {d.get('child_state', 'unknown')} | "
                f"{d.get('last_summary', '')[:60]} |"
            )
        domain_table = (
            "| Domain | Child ID | Files | Scope | State | Last Summary |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(table_rows)
        )
    else:
        domain_table = "(no children spawned yet)"

    return f"""\
## Your Department (pass {pass_number})

{domain_table}

## New Instructions from User
{user_input}

## Instructions
Decide how to handle this follow-up request. You have these options:

### Route to existing children
If the work falls within existing children's domains, reactivate them:
```json
{{
  "action": "route_to_children",
  "rationale": "why these children should handle it",
  "routes": [
    {{"child_node_id": "node-...", "domain_name": "...", "task_spec": "specific follow-up task for this child"}}
  ]
}}
```

### Spawn new children
If this is genuinely new scope that no existing child covers:
```json
{{
  "action": "spawn_children",
  "rationale": "why a new child is needed",
  "children": [
    {{
      "idempotency_key": "...", "objective": "...", "success_criteria": [...],
      "domain_name": "...", "file_patterns": ["..."], "module_scope": "..."
    }}
  ]
}}
```

### Solve directly
If this is a small follow-up you can handle yourself:
```json
{{"action": "solve_directly", "rationale": "..."}}
```

### Done
If the user is signaling completion or you have nothing more to do:
```json
{{"action": "done", "rationale": "..."}}
```

Return ONLY the JSON object.
"""


def reactivation_prompt(original_task: str, previous_summary: str, new_task: str) -> str:
    """Prompt for a child being re-activated with follow-up work."""

    return f"""\
## Reactivation

You previously worked on this domain and completed the following:

### Original task
{original_task}

### What you did
{previous_summary or "(no summary available)"}

### New follow-up task
{new_task}

## Instructions
1. Review your previous work in the worktree.
2. Implement the follow-up changes.
3. Stage and commit all changes.
4. Return a JSON result.

```json
{{"status": "implemented", "summary": "what you changed", "changed_files": ["..."], "commit_sha": "..."}}
```

Return ONLY the JSON object after committing.
"""
