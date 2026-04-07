# Recursive Intelligence

Recursive Intelligence is a local-first runtime for recursive coding agents.

The thesis is simple: a capable model can do better than its flat, single-session baseline when you give it structure to decompose work, loop on mistakes, review child output, and merge accepted changes upward through isolated git worktrees.

This repo is built to test that thesis directly.

## What It Does

- `rari baseline "<task>"` runs one flat Claude session with no recursion.
- `rari run "<task>" --persistent` runs the recursive runtime with decomposition, child worktrees, review, and merge.
- `rari benchmark swebench ...` compares flat vs recursive on a representative SWE-bench slice using the official SWE-bench evaluation harness.

Each run persists structured artifacts under `.ri/`, including costs, durations, sessions, patches, and benchmark reports.

## Vision

The goal is not just “more agents.”

The goal is a runtime where:

- the root node decides when to solve directly vs recurse
- child work is isolated in separate git worktrees
- parents review, loop, and integrate child output
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

Inspect a run:

```bash
rari tree <run-id>
rari inspect <node-id>
```

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

- SWE-bench scoring uses the official Docker harness, not a host-local test runner.
- Benchmark runs can be slow and disk-heavy.
- On Apple Silicon, the harness uses a local namespace override so images can be built locally when needed.

## More Context

- [architecture.md](architecture.md)
- [claude_runtime_plan.md](claude_runtime_plan.md)
- [productization_plan.md](productization_plan.md)
