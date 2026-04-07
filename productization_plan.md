# Productization Plan: Recursive Runtime to Agent Software Org

Status: Draft  
Date: 2026-04-07

## Objective

Turn the current recursive runtime into a product that can take a single prompt, spec, or markdown doc and drive a complex software project to a shippable state with minimal user interruption.

The step-function is not "more agents." The step-function is a managed software org:

- one root agent owns product intent, user communication, and escalation
- specialist child agents work in isolated git worktrees
- review, integration, QA, and release are first-class roles
- the system ships outcomes, not just recursive transcripts

## What Exists Today

The current codebase already provides a real recursive execution kernel:

- recursive parent/child execution with persisted node state
- child work in isolated git worktrees
- parent review, revise, and merge loops
- persistent multi-pass runs
- a benchmark harness comparing flat vs recursive execution

Core implementation lives in:

- [src/recursive_intelligence/runtime/orchestrator.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/orchestrator.py)
- [src/recursive_intelligence/runtime/node_fsm.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/node_fsm.py)
- [src/recursive_intelligence/runtime/state_store.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/state_store.py)
- [src/recursive_intelligence/git/worktrees.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/git/worktrees.py)
- [src/recursive_intelligence/benchmarks/runner.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/benchmarks/runner.py)

What it is missing is product behavior:

- the root is not yet a true manager with a formal escalation contract
- ownership boundaries are advisory rather than enforced
- the control plane is runtime-centric instead of outcome-centric
- shipping stops at intra-run integration instead of landing/deploy/canary
- provider staffing is still Claude-only
- benchmark evidence is not yet strong enough to prove a category-level advantage

## Product Principles

1. Root manages; workers implement.
   For any code-changing task, the root agent remains a manager, even when only one worker is needed.

2. The user talks to one agent.
   Child agents never ask the user for clarification, credentials, or service signups directly.

3. Escalations are typed.
   Third-party signup, missing secrets, ambiguous product decisions, and blocked deploys must surface as structured root-level requests.

4. Ownership must be enforced, not suggested.
   File scope, test scope, and acceptance criteria need hard gates before work can promote upward.

5. The interface must speak in outcomes.
   The primary UX should be spec, blockers, previews, quality, and ship status, not nodes and event logs.

6. The thesis must be measurable.
   Product claims need ablations, controls, user-visible metrics, and competitive baselines, not just demos.

## Target User Experience

1. User provides a prompt or spec.
2. Root agent triages:
   - answer directly if no code changes are needed
   - otherwise create a run and spawn one or more workers
3. Root extracts missing high-leverage decisions and asks the user only when necessary.
4. Root creates a task graph with explicit roles, ownership, and acceptance criteria.
5. Workers implement in isolated worktrees.
6. Review and integration agents gate promotion.
7. QA agents run product-level checks.
8. Release agent lands to trunk and drives deploy verification.
9. User sees a single timeline with blocker cards, previews, and ship status.

## Definition of Step-Function Improvement

This product is meaningfully better than today's coding-agent status quo only if all of the following are true:

- one top-level prompt can drive multi-step internal execution without repeated "phase 1 complete" user prompts
- user interruptions are limited to true ambiguities and external-service setup
- workers can operate in parallel without clobbering each other
- integration quality improves rather than regresses as parallelism increases
- the system can carry work from planning through verification and release
- benchmark and dogfood evidence show a material improvement over both:
  - a flat single-agent control
  - human-orchestrated parallel-agent tools where the user decides when to branch, retry, or merge

## Benchmark Strategy

The benchmark strategy should match the product thesis instead of over-indexing on repo patch benchmarks.

Use a layered stack:

1. Engine benchmarks
   Purpose: validate recursive decomposition, repo navigation, review, merge discipline, and refactor quality inside existing codebases.
   Recommended suites:
   - SWE-bench Verified
   - SWE-Bench Pro
   - FeatureBench

2. Product benchmarks
   Purpose: validate "single spec to working product" behavior.
   Recommended suites:
   - Vibe Code Bench or equivalent browser-evaluated app-building benchmarks
   - App-Bench style rubric-driven app-generation tasks

3. Internal moat benchmark: Spec-to-Ship
   Purpose: test the exact behavior this product claims as its advantage.
   Required properties:
   - one prompt, spec, or markdown file starts the run
   - bounded root-level clarification is allowed
   - child agents never ask the user directly
   - hidden end-to-end tests and deployment checks score the output
   - human interruption count and human orchestration time are first-class metrics

4. Dogfood builds
   Purpose: validate performance on real internal products and large refactors.

SWE-Bench Pro remains useful, but it is an engine benchmark, not the north-star product benchmark.

## Telemetry Requirements

Instrumentation cannot wait until the end of the roadmap. The runtime must begin recording product-proof telemetry early.

Required telemetry from the first productization milestones:

- user interruption count
- root escalation count
- human minutes spent orchestrating the run
- time to first preview
- time to green verification
- total wall-clock time
- total model cost
- merge conflict count
- review churn
- ownership violations caught before merge
- deploy blockers and release retries

## Phase Plan

## Phase 1: Formalize the Operating Model

Goal: turn the current tree runtime into an explicit manager/worker org.

Deliverables:

- add typed blocker and escalation records to node results
- add a run-level telemetry schema for interruptions, escalations, and orchestration time
- codify root-only user communication for code-changing runs
- introduce explicit role types: `manager`, `implementer`, `reviewer`, `integrator`, `qa`, `release`
- replace simple parent-child waiting with a task graph model that supports dependencies and promotion gates
- add a manager fast path: root still manages simple code tasks but spawns only one worker

Primary files:

- [src/recursive_intelligence/runtime/orchestrator.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/orchestrator.py)
- [src/recursive_intelligence/runtime/node_fsm.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/node_fsm.py)
- [src/recursive_intelligence/runtime/state_store.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/state_store.py)
- [src/recursive_intelligence/adapters/base.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/adapters/base.py)
- [src/recursive_intelligence/adapters/claude/prompts.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/adapters/claude/prompts.py)

Acceptance criteria:

- every node outcome is one of: `implemented`, `blocked`, `needs_review`, `needs_revision`, `ready_for_release`, `failed`
- blockers are structured with kind, owner, urgency, and whether user input is required
- only the root node can emit user-facing clarification or signup requests
- simple code-changing tasks still run through a root-manager plus one-worker flow
- run records include interruption and escalation telemetry from day one

## Phase 2: Enforce Ownership and Quality Gates

Goal: make delegation safe enough to improve quality, not just throughput.

Deliverables:

- enforce `file_scope` and `file_patterns` against actual git diffs
- add per-node verification contracts:
  - focused tests
  - lint/type checks
  - artifact summary
- require review verdicts before upward integration
- add merge promotion gates:
  - ownership check
  - test check
  - conflict check
  - acceptance-criteria check
- record merge conflicts, retries, and review churn as first-class metrics

Primary files:

- [src/recursive_intelligence/runtime/orchestrator.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/orchestrator.py)
- [src/recursive_intelligence/runtime/node_fsm.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/node_fsm.py)
- [src/recursive_intelligence/git/diffing.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/git/diffing.py)
- [src/recursive_intelligence/git/merge.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/git/merge.py)
- [src/recursive_intelligence/runtime/artifact_store.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/artifact_store.py)

Acceptance criteria:

- a worker cannot silently modify files outside its assigned scope
- every merged child has a recorded review verdict and verification artifact
- the runtime can fail closed when review or verification requirements are missing
- merge/review failure reasons are visible to the root as structured data

## Phase 3: Build the Outcome-Oriented Control Plane

Goal: make the system feel like a product manager for software delivery, not a runtime debugger.

Deliverables:

- define a first-class run spec:
  - user objective
  - optional markdown spec
  - constraints
  - design expectations
  - deploy target
- add a root-level backlog for open questions, blockers, and external-service requests
- redesign the primary UX around:
  - run overview
  - blockers
  - workstreams
  - previews
  - quality status
  - release status
- keep low-level node/event views as secondary diagnostics
- add spec-aware commands such as:
  - `rari create`
  - `rari continue`
  - `rari blockers`
  - `rari ship`

Primary files:

- [src/recursive_intelligence/cli.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/cli.py)
- [src/recursive_intelligence/tui.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/tui.py)
- [src/recursive_intelligence/runtime/state_store.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/state_store.py)
- [README.md](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/README.md)

Acceptance criteria:

- a new user can start from a spec rather than a runtime command
- blockers and signup requests are visible without inspecting node internals
- the primary run screen answers:
  - what is being built
  - what is blocked
  - what is done
  - what remains before shipping

## Phase 4: Add Release, Deploy, and Verification as First-Class Workflow

Goal: finish the job after implementation.

Deliverables:

- add a release role that owns trunk landing decisions
- support runtime-managed branch landing and merge queue behavior
- add deploy configuration and environment readiness checks
- add post-deploy verification and canary checks
- route missing credentials, auth setup, billing/signup, and deploy approvals back to root

Primary files:

- [src/recursive_intelligence/git/merge.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/git/merge.py)
- [src/recursive_intelligence/runtime/orchestrator.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/orchestrator.py)
- [src/recursive_intelligence/runtime/state_store.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/runtime/state_store.py)
- new deploy/release modules under `src/recursive_intelligence/runtime/` or `src/recursive_intelligence/release/`

Acceptance criteria:

- a run can progress from accepted code to landed code
- deploy blockers are typed and root-visible
- post-deploy verification status is attached to the run record

## Phase 5: Introduce Multi-Provider Staffing

Goal: make model choice part of org design instead of a hardcoded implementation detail.

Deliverables:

- generalize the adapter contract for multi-provider execution
- add at least one second provider adapter
- let the root assign provider and model by role:
  - planner
  - implementer
  - reviewer
  - QA
  - release
- add staffing policies for cost, latency, and quality tiers

Primary files:

- [src/recursive_intelligence/adapters/base.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/adapters/base.py)
- [src/recursive_intelligence/cli.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/cli.py)
- adapter-specific modules under [src/recursive_intelligence/adapters/](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/adapters)

Acceptance criteria:

- the runtime can execute one run with mixed providers by role
- benchmarks can compare same-provider and mixed-staffing configurations
- provider selection is recorded as part of run artifacts and reports

## Phase 6: Prove the Thesis with Harder Evidence

Goal: validate that the product is materially better than the status quo.

Deliverables:

- implement the full layered benchmark stack described in [experiment.md](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/experiment.md):
  - engine benchmarks for repo-scale issue solving and feature work
  - product benchmarks for spec-to-app generation
  - an internal Spec-to-Ship benchmark
- implement the recursive-runtime ablation matrix:
  - Baseline 0: flat single agent
  - Baseline 1: non-recursive decomposition
  - Treatment 1: recursive depth-1
  - Treatment 2: recursive depth-2+
- add competitive baselines:
  - human-orchestrated parallel-agent workflow
  - prompt-to-app builder workflow where relevant
- add repeated trials and confidence reporting
- add metrics for:
  - user interruption count
  - human minutes spent orchestrating
  - merge conflict rate
  - review churn
  - clarification burden
  - cost-normalized success
  - time-to-first-preview
  - time-to-green
- add dogfood suites for full application builds, not just benchmark patches

Primary files:

- [src/recursive_intelligence/benchmarks/runner.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/benchmarks/runner.py)
- [src/recursive_intelligence/benchmarks/models.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/benchmarks/models.py)
- [src/recursive_intelligence/benchmarks/reporting.py](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/src/recursive_intelligence/benchmarks/reporting.py)
- [experiment.md](/Users/calmdentist/conductor/workspaces/recursive_intelligence/banjul/experiment.md)

Acceptance criteria:

- reports show statistically grounded comparisons instead of single-run anecdotes
- product metrics capture user experience, not just patch success
- reports distinguish engine gains from product-delivery gains
- reports distinguish recursion gains from human-orchestration savings
- we can identify which parts of the runtime create the measured gains

## P0 Build Sequence

These are the minimum changes required to create a visible product discontinuity.

1. Formalize root-only escalation and typed blockers.
2. Enforce file ownership and verification gates.
3. Replace runtime-first UX with spec, blocker, and ship views.
4. Add release and deploy flow.

If these four are not done, the system remains an advanced recursive runtime rather than a productized software org.

## P1 Build Sequence

These changes deepen the moat after the product loop works end to end.

1. Add multi-provider staffing.
2. Add the full layered benchmark stack and repeated trials.
3. Add better dependency scheduling beyond simple parent-child barriers.

## Explicit Non-Goals for This Plan

- hosted multi-tenant control plane before the local product loop works
- open-ended peer-to-peer worker communication without manager mediation
- broad provider abstraction before adding one concrete second adapter
- autonomous purchasing or signup decisions without explicit root-mediated user approval

## Suggested Implementation Order by Repo Area

1. Runtime contracts
   - `state_store.py`
   - `node_fsm.py`
   - `adapters/base.py`
   - `adapters/claude/prompts.py`

2. Enforcement and integration
   - `orchestrator.py`
   - `git/diffing.py`
   - `git/merge.py`
   - `artifact_store.py`

3. Product surface
   - `cli.py`
   - `tui.py`
   - `README.md`

4. Release and evidence
   - release/deploy modules
   - `benchmarks/*`
   - `experiment.md`

## Go/No-Go Metrics

Before calling this a step-function improvement, the product should demonstrate:

- median user interruption count below 3 for complex builds, excluding credentials and irreducible product ambiguity
- materially lower human orchestration time than human-managed parallel-agent workflows on the same tasks
- enforced ownership violations caught before merge in more than 95 percent of seeded tests
- end-to-end runs that reach implementation, verification, and release without manual orchestration between each phase
- statistically defensible gains over a flat baseline on at least one engine benchmark slice
- statistically defensible gains over human-orchestrated parallel-agent baselines on at least one product benchmark slice
- at least one dogfood app build completed from a single spec with root-only escalations

## Immediate Next Build

The next implementation milestone should combine the first two P0 steps:

- add typed blocker and escalation objects to the runtime
- add run-level telemetry for interruptions, escalations, and orchestration time
- make the root the only node allowed to request user action
- enforce worker file ownership at merge time
- add per-node verification artifacts and merge gates

That is the smallest change set that starts converting the current engine into a real managed software org.
