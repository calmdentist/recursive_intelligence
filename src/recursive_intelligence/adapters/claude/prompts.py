"""Prompt templates for Claude node sessions."""

from __future__ import annotations

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
```json
{{
  "action": "spawn_children",
  "rationale": "why this needs decomposition",
  "children": [
    {{
      "idempotency_key": "short-unique-slug",
      "objective": "specific task description for child",
      "success_criteria": ["criterion 1", "criterion 2"]
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
