"""Prompt templates for Claude node sessions."""

from __future__ import annotations

from typing import Any

SYSTEM_CONTRACT = """\
You are a node in a recursive coding-agent runtime. You work in an isolated git worktree.

- Solve small tasks directly. Decompose large tasks into children with independent file scopes.
- Commit your work before finishing.
- If you need clarification, credentials, third-party signup, or approval, send a typed request upstream.
- Child nodes must never ask the user directly. They send requests to their parent. Only the root may surface user-facing requests.
- End each phase with a single JSON object. No wrapping text or markdown around it.
"""


def _request_upstream_schema(is_root: bool) -> str:
    user_line = (
        "If you need user action, return a `request_upstream` result with `requires_input: true`."
        if is_root
        else "Do not ask the user directly. If you need user action, return a `request_upstream` result for your parent or the root."
    )
    return f"""\
If you need something from upstream:
{user_line}
{{
  "status": "request_upstream",
  "request": {{
    "kind": "...",
    "summary": "...",
    "details": "...",
    "action_requested": "...",
    "requires_input": true,
    "urgency": "normal"
  }}
}}"""


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
      "module_scope": "what this domain owns",
      "depends_on": ["domain-this-child-needs"],
      "interface_contract": "what this child must expose or consume for siblings",
      "handoff_artifacts": ["shared/types.ts", "src/module/api.ts"],
      "verification_command": "pytest tests/module/test_one.py",
      "verification_notes": "focused check for this child"
    }}
  ]
}}
"""


def _contract_section(
    success_criteria: list[str] | None = None,
    verification_command: str | None = None,
    verification_notes: str = "",
) -> str:
    sections: list[str] = []

    if success_criteria:
        criteria = "\n".join(f"- {item}" for item in success_criteria)
        sections.append(f"## Success Criteria\n{criteria}")

    command = (verification_command or "").strip()
    notes = verification_notes.strip()
    if command or notes:
        verification = ["## Local Verification"]
        if command:
            verification.append(
                "Use this focused local check before declaring success if the repo state allows it."
            )
            verification.append(f"Command: {command}")
        if notes:
            verification.append(f"Notes: {notes}")
        sections.append("\n".join(verification))

    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _ownership_section(
    domain_name: str | None = None,
    module_scope: str = "",
    file_patterns: list[str] | None = None,
    changed_files: list[str] | None = None,
) -> str:
    lines: list[str] = []
    if domain_name:
        lines.append(f"Domain: {domain_name}")
    if module_scope:
        lines.append(f"Module scope: {module_scope}")
    if file_patterns:
        lines.append(f"Allowed files: {', '.join(file_patterns)}")
    if changed_files:
        lines.append(f"Changed files: {', '.join(changed_files)}")
    if not lines:
        return ""
    return "\n\n## Ownership Boundaries\n" + "\n".join(lines)


def _coordination_section(
    depends_on: list[str] | None = None,
    interface_contract: str = "",
    handoff_artifacts: list[str] | None = None,
    dependency_context: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    if depends_on:
        lines.append(f"Depends on: {', '.join(depends_on)}")
    if interface_contract.strip():
        lines.append(f"Interface contract: {interface_contract.strip()}")
    if handoff_artifacts:
        lines.append(f"Handoff artifacts: {', '.join(handoff_artifacts)}")
    if dependency_context:
        lines.append("Dependency context:")
        for item in dependency_context:
            summary = item.get("summary", "").strip() or "(no summary)"
            lines.append(
                f"- {item.get('dependency', 'dependency')}: {summary}"
            )
            handoff = item.get("handoff_summary", "").strip()
            if handoff:
                lines.append(f"  Handoff: {handoff}")
            interfaces = item.get("interfaces", [])
            if interfaces:
                lines.append(f"  Interfaces: {', '.join(interfaces)}")
            breaking = item.get("breaking_changes", [])
            if breaking:
                lines.append(f"  Breaking changes: {', '.join(breaking)}")
    if not lines:
        return ""
    return "\n\n## Coordination Contract\n" + "\n".join(lines)


def execution_prompt(
    task_spec: str,
    is_root: bool = False,
    success_criteria: list[str] | None = None,
    verification_command: str | None = None,
    verification_notes: str = "",
    domain_name: str | None = None,
    module_scope: str = "",
    file_patterns: list[str] | None = None,
    depends_on: list[str] | None = None,
    interface_contract: str = "",
    handoff_artifacts: list[str] | None = None,
    dependency_context: list[dict[str, Any]] | None = None,
) -> str:
    contract = _contract_section(success_criteria, verification_command, verification_notes)
    ownership = _ownership_section(domain_name, module_scope, file_patterns)
    coordination = _coordination_section(
        depends_on,
        interface_contract,
        handoff_artifacts,
        dependency_context,
    )
    return f"""\
## Task
{task_spec}
{contract}
{ownership}
{coordination}

Implement this task. Commit when done.

On success:
{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "...", "handoff": {{"summary": "...", "interfaces": ["..."], "artifacts": ["..."], "breaking_changes": ["..."]}}}}

{_request_upstream_schema(is_root)}
"""


def _review_verification_section(verification: dict[str, Any] | None = None) -> str:
    if not verification or verification.get("status") == "not_required":
        return ""

    lines = [f"Status: {verification.get('status', 'unknown')}"]
    command = verification.get("command", "").strip()
    if command:
        lines.append(f"Command: {command}")
    notes = verification.get("notes", "").strip()
    if notes:
        lines.append(f"Notes: {notes}")
    output = verification.get("output", "").strip()
    if output:
        lines.append("Latest output:")
        lines.append("```")
        lines.append(output[:2000])
        lines.append("```")
    return "\n\n## Local Verification\n" + "\n".join(lines)


def review_prompt(
    child_id: str,
    diff: str,
    summary: str,
    success_criteria: list[str],
    verification: dict[str, Any] | None = None,
    ownership: dict[str, Any] | None = None,
    coordination: dict[str, Any] | None = None,
) -> str:
    criteria_str = "\n".join(f"  - {c}" for c in success_criteria) if success_criteria else "  - (none specified)"
    verification_section = _review_verification_section(verification)
    ownership_section = _ownership_section(
        ownership.get("domain_name") if ownership else None,
        ownership.get("module_scope", "") if ownership else "",
        ownership.get("file_patterns") if ownership else None,
        ownership.get("changed_files") if ownership else None,
    )
    coordination_section = _coordination_section(
        coordination.get("depends_on") if coordination else None,
        coordination.get("interface_contract", "") if coordination else "",
        coordination.get("handoff_artifacts") if coordination else None,
        coordination.get("dependency_context") if coordination else None,
    )

    return f"""\
Review child {child_id}'s work against these criteria:
{criteria_str}
{verification_section}
{ownership_section}
{coordination_section}

Summary: {summary or "(none)"}

```diff
{diff}
```

Respond with one of:
{{"verdict": "accept", "child_id": "{child_id}", "reason": "..."}}
{{"verdict": "revise", "child_id": "{child_id}", "reason": "...", "follow_up": "what to fix"}}
{{"verdict": "reject", "child_id": "{child_id}", "reason": "..."}}
"""


def revision_prompt(
    follow_up: str,
    is_root: bool = False,
    success_criteria: list[str] | None = None,
    verification_command: str | None = None,
    verification_notes: str = "",
    domain_name: str | None = None,
    module_scope: str = "",
    file_patterns: list[str] | None = None,
    depends_on: list[str] | None = None,
    interface_contract: str = "",
    handoff_artifacts: list[str] | None = None,
    dependency_context: list[dict[str, Any]] | None = None,
) -> str:
    contract = _contract_section(success_criteria, verification_command, verification_notes)
    ownership = _ownership_section(domain_name, module_scope, file_patterns)
    coordination = _coordination_section(
        depends_on,
        interface_contract,
        handoff_artifacts,
        dependency_context,
    )
    return f"""\
Revision requested. Make these changes, commit, and return the result.

Feedback: {follow_up}
{contract}
{ownership}
{coordination}

{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "...", "handoff": {{"summary": "...", "interfaces": ["..."], "artifacts": ["..."], "breaking_changes": ["..."]}}}}

{_request_upstream_schema(is_root)}
"""


def routing_prompt(user_input: str, domains: list[dict[str, Any]], pass_number: int) -> str:
    """Route follow-up work to existing children or spawn new ones."""

    if domains:
        rows = []
        for d in domains:
            patterns = ", ".join(d.get("file_patterns", []))
            depends = ", ".join(d.get("depends_on", [])) or "-"
            handoff = d.get("handoff_summary", "")[:40] or "-"
            rows.append(
                f"| {d['domain_name']} | {d['child_node_id'][:12]} | {patterns} | "
                f"{depends} | {d.get('child_state', '?')} | {handoff} | {d.get('last_summary', '')[:40]} |"
            )
        domain_table = (
            "| Domain | Child | Files | Depends On | State | Handoff | Summary |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
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
{{"action": "spawn_children", "rationale": "...", "children": [{{"idempotency_key": "...", "objective": "...", "success_criteria": [...], "domain_name": "...", "file_patterns": ["..."], "module_scope": "...", "depends_on": ["..."], "interface_contract": "...", "handoff_artifacts": ["..."], "verification_command": "pytest tests/module/test_one.py", "verification_notes": "focused check for this child"}}]}}

Solve directly:
{{"action": "solve_directly", "rationale": "..."}}

Done:
{{"action": "done", "rationale": "..."}}
"""


def reactivation_prompt(
    original_task: str,
    previous_summary: str,
    new_task: str,
    is_root: bool = False,
    success_criteria: list[str] | None = None,
    verification_command: str | None = None,
    verification_notes: str = "",
    domain_name: str | None = None,
    module_scope: str = "",
    file_patterns: list[str] | None = None,
    depends_on: list[str] | None = None,
    interface_contract: str = "",
    handoff_artifacts: list[str] | None = None,
    dependency_context: list[dict[str, Any]] | None = None,
) -> str:
    """Re-activate a child with follow-up work."""
    contract = _contract_section(success_criteria, verification_command, verification_notes)
    ownership = _ownership_section(domain_name, module_scope, file_patterns)
    coordination = _coordination_section(
        depends_on,
        interface_contract,
        handoff_artifacts,
        dependency_context,
    )

    return f"""\
You previously completed: {original_task}
What you did: {previous_summary or "(unknown)"}

New task: {new_task}
{contract}
{ownership}
{coordination}

Implement the changes, commit, and return:
{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "...", "handoff": {{"summary": "...", "interfaces": ["..."], "artifacts": ["..."], "breaking_changes": ["..."]}}}}

{_request_upstream_schema(is_root)}
"""


def downstream_response_prompt(
    request_summary: str,
    response_text: str,
    original_details: str = "",
    resolution: str = "answer",
    is_root: bool = False,
    success_criteria: list[str] | None = None,
    verification_command: str | None = None,
    verification_notes: str = "",
    domain_name: str | None = None,
    module_scope: str = "",
    file_patterns: list[str] | None = None,
    depends_on: list[str] | None = None,
    interface_contract: str = "",
    handoff_artifacts: list[str] | None = None,
    dependency_context: list[dict[str, Any]] | None = None,
) -> str:
    resolution_label = {
        "approve": "Approval from upstream",
        "decline": "Decline from upstream",
        "answer": "Response from upstream",
    }.get(resolution, "Response from upstream")
    contract = _contract_section(success_criteria, verification_command, verification_notes)
    ownership = _ownership_section(domain_name, module_scope, file_patterns)
    coordination = _coordination_section(
        depends_on,
        interface_contract,
        handoff_artifacts,
        dependency_context,
    )

    return f"""\
You previously sent an upstream request.

## Request summary
{request_summary}

## Original details
{original_details or "(none)"}

## {resolution_label}
{response_text}
{contract}
{ownership}
{coordination}

Use this response to continue the task. Commit when done.

On success:
{{"status": "implemented", "summary": "...", "changed_files": ["..."], "commit_sha": "...", "handoff": {{"summary": "...", "interfaces": ["..."], "artifacts": ["..."], "breaking_changes": ["..."]}}}}

{_request_upstream_schema(is_root)}
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
