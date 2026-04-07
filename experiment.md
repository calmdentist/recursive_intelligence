# experiment.md

## Objective
Test whether a **root-managed recursive coding-agent runtime** materially outperforms both:

- a **flat coding agent**
- a **human-orchestrated parallel-agent workflow**

for building complex software from either existing codebases or a single product spec.

## Core Hypothesis
A recursive runtime improves task success by enabling:
- narrower local contexts
- parallel hypothesis testing
- better repo navigation
- cleaner intermediate outputs

The key question is not whether recursion is elegant, but whether the runtime improves:

- feature completion
- search efficiency
- cost-normalized performance
- time to a working preview
- human interruption burden
- human orchestration time

Repo issue benchmarks are useful, but they are not sufficient. The product thesis is about taking a prompt, spec, or markdown document and driving a working product with minimal back-and-forth.

## Benchmark
Use a layered benchmark stack.

### Layer 1: Engine benchmarks
Purpose: validate the recursive runtime as an engine for repo-scale reasoning, decomposition, refactors, and integration.

Recommended suites:
- SWE-bench Verified for rapid iteration
- SWE-Bench Pro for harder repo-scale issue solving
- FeatureBench for complex feature implementation inside real repositories

Why this layer exists:
- it tests recursive reasoning in existing repositories
- it is useful for runtime ablations
- it measures decomposition and merge quality better than prompt-to-app suites

### Layer 2: Product benchmarks
Purpose: validate "single spec to working app" behavior.

Recommended suites:
- Vibe Code Bench or a similar browser-evaluated app-development benchmark
- App-Bench style rubric-driven app-building tasks

Why this layer exists:
- it is much closer to the actual product promise
- it measures whether the system can build a functional application from a single prompt or spec
- it allows hidden end-to-end workflows instead of only patch-level scoring

### Layer 3: Internal Spec-to-Ship benchmark
Purpose: test the exact claim this product is making.

Required properties:
- one prompt, spec, or markdown file starts the run
- bounded root-level clarification is allowed
- child agents never ask the user directly
- the output must be deployable or locally runnable
- hidden browser, API, and verification checks score the final product
- human interruption count and human orchestration time are first-class metrics

Suggested task mix:
- greenfield product builds
- large cross-cutting refactors
- feature expansions on nontrivial existing apps

### Layer 4: Dogfood and freshness checks
Purpose: reduce contamination risk and validate real-world value.

Recommended sources:
- recent internal tasks
- fresh GitHub issues
- real products and refactors built with the system

## Systems Compared
Hold the **base model constant** where possible for engine ablations. For product comparisons, also measure realistic tool-default workflows because the user buys products, not just model weights.

### Baseline 0: Flat agent
- Single agent session
- No child calls
- Standard tools and verification

### Baseline 1: Human-orchestrated parallel agents
- Parallel agents in isolated worktrees or equivalent environments
- Human decides when to branch, redirect, retry, and merge
- Models and tools should match best-practice usage for the comparison tool

### Baseline 2: Non-recursive decomposition
- Parent can call helper workers
- Helpers cannot call themselves

### Treatment 1: Root-managed runtime, single worker
- Root remains manager
- One implementation worker does the coding
- Used to test manager/worker overhead on small and medium tasks

### Treatment 2: True recursion, depth 1
- Parent can spawn fresh instances of itself
- Children cannot recurse further

### Treatment 3: True recursion, depth 2+
- Children may recurse under strict budget decay

### Optional comparison: Prompt-to-app builders
- Include builder tools where the task is a greenfield app and the comparison is relevant
- Measure them as product workflows rather than repo-engine workflows

## Metrics

Track both engine metrics and product metrics.

Engine metrics:
- solve rate
- feature pass rate
- cost-normalized solve rate
- merge conflict rate
- review churn
- verification pass rate

Product metrics:
- time to first working preview
- time to green verification
- total wall-clock time
- user interruption count
- root escalation count
- human minutes spent orchestrating
- severe defects after QA
- deploy success rate

## Telemetry Requirements

The runtime must record the following from early milestones onward:

- each user interruption and its reason
- each root escalation and whether it required user action
- each human action taken to advance the run
- merge conflicts, retries, and review loops
- timestamps for preview-ready, verification-ready, and release-ready states
- model, provider, and role assignment for each node

## Minimum Prototype
Build a runtime with one additional primitive:

```python
call_self(
    objective,
    file_scope,
    success_criteria,
    budget,
    writable=False|True,
    max_depth_remaining
) -> TypedResult
