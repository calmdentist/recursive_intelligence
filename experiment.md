# experiment.md

## Objective
Test whether a **recursively self-calling coding-agent runtime** outperforms a **flat coding agent** on repo-scale software tasks, holding the base model fixed.

## Core Hypothesis
A recursive runtime improves task success by enabling:
- narrower local contexts
- parallel hypothesis testing
- better repo navigation
- cleaner intermediate outputs

The key question is not whether recursion is elegant, but whether it improves **solve rate**, **search efficiency**, and **cost-normalized performance** on realistic coding tasks. SWE-bench and its newer variants are appropriate starting points because they evaluate issue resolution over real repositories, including human-filtered and freshness-oriented subsets. 

## Benchmark
Use a three-tier stack:

### Tier A: Fast iteration
- ~30–50 tasks from **SWE-bench Verified**
- Stratified toward multi-file and repo-navigation-heavy issues
- Purpose: rapid runtime debugging and ablations

### Tier B: Main offline eval
- ~100–200 held-out tasks from **SWE-bench Verified** and, if available, **SWE-bench Pro**
- Purpose: main comparison under controlled conditions

### Tier C: Freshness check
- ~25–50 tasks from **SWE-bench-Live** or recent internal GitHub issues
- Purpose: reduce contamination risk and test real-world generalization 

## Systems Compared
Hold the **base model constant** across all comparisons.

### Baseline 0: Flat agent
- Single agent session
- No child calls
- Standard tools and verification

### Baseline 1: Non-recursive decomposition
- Parent can call helper workers
- Helpers cannot call themselves

### Treatment 1: True recursion, depth 1
- Parent can spawn fresh instances of itself
- Children cannot recurse further

### Treatment 2: True recursion, depth 2+
- Children may recurse under strict budget decay

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
