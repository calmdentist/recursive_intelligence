# Recursive Intelligence

Recursive Intelligence is a local-first runtime for recursive coding agents.

The thesis is simple: a capable model can do better than its flat, single-session baseline when you give it structure to decompose work, loop on mistakes, review child output, and merge accepted changes upward through isolated git worktrees.

This repo is built to test that thesis directly.

## What It Does

- `rari baseline "<task>"` runs one flat Claude session with no recursion.
- `rari run "<task>" --persistent` runs the recursive runtime with decomposition, child worktrees, review, and merge.
- `rari chat [run-id]` opens the interactive TUI for a persistent run, with the root conversation on the left and the node tree on the right.
- `rari resume <run-id>` resumes a crashed or interrupted persistent run from the last durable state.
- `rari benchmark swebench ...` compares flat vs recursive on a representative SWE-bench slice using the official SWE-bench evaluation harness.

Each run persists structured artifacts under `.ri/`, including costs, durations, sessions, patches, and benchmark reports.

## Vision

The goal is not just “more agents.”

The goal is a runtime where:

- the root node decides when to solve directly vs recurse
- child work is isolated in separate git worktrees
- parents review, loop, and integrate child output
- decomposition happens in staged waves when later work depends on earlier foundation
- manager nodes stay in management mode after they delegate, routing revisions back to the right child instead of dropping into direct execution
- the whole process is benchmarkable against a flat control

If this works, recursion and looping should shift the cost-quality frontier for coding agents. A recursive system may be able to augment a cheaper base model or get more leverage out of a stronger one.

## Setup

Requirements:

- Python `>=3.11`
- `git`
- Docker, if you want SWE-bench evaluation
- Claude Code / Anthropic access in your local environment

Install:

```bash
pip install -e .
```

Or with `pipx`:

```bash
pipx install --editable .
```

Check the CLI:

```bash
rari --help
```

## Quickstart

Run a flat baseline:

```bash
rari baseline "fix the failing test in this repo"
```

Run the recursive runtime:

```bash
rari run "fix the failing test in this repo" --persistent
```

Open the interactive chat UI for a new or existing persistent run:

```bash
rari chat
rari chat <run-id>
```

Inspect a run:

```bash
rari tree <run-id>
rari domains <run-id>
rari inspect <node-id>
rari resume <run-id>
```

## Runtime Model

- Planning is decision-only. Nodes inspect the repo and choose whether to solve directly, route to an existing child, or spawn a new wave of children.
- Parallel children should own substantial, mostly disjoint domains. Same-wave children are expected to be runnable against the same parent snapshot.
- If later work depends on new foundation, the parent should spawn that prerequisite wave first, merge it, and then plan the next wave.
- Once a node has delegated, it acts as a manager for that slice: it reviews child work, requests revisions, routes follow-up back to the current domain owner, and merges accepted results upward.
- Worker nodes return a structured handoff so the parent can replan using concrete deliverables, findings, concerns, and suggested next steps.

## Interactive UI

- The left pane is the human-facing conversation with the root node.
- The right pane shows the live node tree plus details for the selected node.
- Internal tool chatter and raw control-plane JSON are hidden by default. Use `/debug` only when you want the internal trace.
- Leaving the chat does not discard a persistent run. `/quit` exits and leaves the run paused; `/done` finalizes it.

## Benchmarking

Run a representative SWE-bench slice:

```bash
rari benchmark swebench --suite tier-a --limit 2
```

Compare model configurations inside the Claude family:

```bash
rari benchmark swebench --suite tier-a --limit 2 \
  --root-model claude-sonnet-4-6 \
  --child-model claude-haiku-4-5
```

This keeps baseline and recursive root nodes on the same model while making recursive children cheaper.

Export a completed benchmark report:

```bash
rari export-report <benchmark-run-id>
```

## Notes

- Persistent runs store their state in `.ri/state.db` and their worktrees under `.ri/worktrees/`.
- The root conversation is intentionally more human-facing than child control traffic. Internal planning and routing still use structured JSON contracts behind the scenes.
- The recursive runtime is optimized for benchmarkable prototypes today, not arbitrary production autonomy. The architecture is still evolving around planning quality, review quality, and scheduling.
- SWE-bench scoring uses the official Docker harness, not a host-local test runner.
- Benchmark runs can be slow and disk-heavy.
- On Apple Silicon, the harness uses a local namespace override so images can be built locally when needed.

## More Context

- [architecture.md](architecture.md)
- [claude_runtime_plan.md](claude_runtime_plan.md)
