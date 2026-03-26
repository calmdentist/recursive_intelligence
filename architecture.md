# architecture.md

## Overview

This system is a **recursive coding-agent runtime** built on top of **Codex** or **Claude Code**.

The idea is simple:

1. every node works in a child git worktree derived from its parent
2. every parent reviews child work, handles merge conflicts, and can loop a child over multiple turns
3. any node can decide whether it should solve the task itself or spawn children
4. an external orchestrator facilitates communication between parents and children

That is the whole model.

---

## Core model

Each node is the same kind of thing:

- a coding agent instance
- given a task
- given a git worktree
- given a parent
- optionally allowed to spawn children

A node does one of two things:

- solve the task directly
- decide the task should be split up and spawn children

If it spawns children, it becomes responsible for:

- reviewing child output
- deciding whether child work is acceptable
- handling merge conflicts between children
- sending follow-up instructions to children when needed
- synthesizing child progress into a result for its own parent

---

## System components

## 1. Orchestrator

The orchestrator is external to the model.

Its job is simple:

- create child nodes
- create child worktrees
- route messages between parents and children
- wake parent nodes when children finish a turn
- keep track of the tree

The orchestrator does **not** do the reasoning.
It only facilitates execution and communication.

---

## 2. Nodes

A node is a Codex or Claude Code instance working on a task.

Each node has:

- a task
- a git worktree
- a parent node, unless it is the root
- zero or more child nodes

Each node can decide:

- “I can handle this myself”
- or “this should be broken into smaller subtasks”

If it chooses the second option, it spawns children.

---

## 3. Root node

The root node is the node that talks to the human.

It receives the top-level task and either:

- solves it directly
- or decomposes it into child tasks

The root is also just a node.
It follows the same logic as every other node.

---

## Worktree model

Every node works in a git worktree derived from its parent’s current state.

That means:

- parent has its own worktree
- each child gets a child worktree based on the parent worktree
- child changes are isolated from the parent until reviewed and merged
- siblings do not write directly into the same workspace

This gives the system:

- isolation
- easy rollback
- clear parent/child lineage
- parent-controlled integration

The important rule is:

> a child inherits the parent’s state, but the parent decides whether the child’s changes are accepted.

---

## Parent-child relationship

A parent owns its children.

That means the parent is responsible for:

- deciding when to spawn a child
- deciding what task the child should do
- reviewing the child’s commits / diff / output
- deciding whether the child should continue
- deciding whether to merge the child’s work
- resolving conflicts between multiple children

Children do work.
Parents synthesize.

---

## Child looping

A child is not necessarily one-shot.

A parent can loop a child by giving it another turn after reviewing its output.

This is useful when:

- the child’s work is incomplete
- the child made a mistake
- the child needs to revise its approach
- the child needs multiple turns to finish the task

So the basic loop is:

1. parent spawns child
2. child works and halts
3. parent reviews child output
4. parent either:
   - accepts it,
   - rejects it,
   - or messages the child again with further direction

This lets work continue across multiple turns without creating a new child every time.

---

## Recursion

Any node can decide whether it needs to recurse.

That decision is local.

A node should ask:

- can I finish this task myself?
- or is it better to split this into child tasks?

This decision can be guided by prompt and tuned over time.

The core recursive pattern is:

1. inspect task
2. choose solve vs recurse
3. if recurse, spawn children
4. review child work
5. merge accepted child work
6. return result upward

This same logic applies at every level of the tree.

---

## Communication model

The orchestrator facilitates communication between parents and children.

At minimum, the system needs two actions:

### `spawn_child()`
Create a child node with:
- a task
- a child worktree
- a link to the parent

### `message_child()`
Send another instruction to an existing child after reviewing its latest output

The child does not need a special `message_parent()` tool.

Instead:
- the child finishes its turn
- the orchestrator records the result
- the orchestrator wakes the parent
- the parent sees the child’s latest output and decides what to do next

So upward communication is event-driven.

---

## Merge model

Children do not directly modify the parent’s accepted state.

Instead:

- each child works in its own child worktree
- the parent reviews the child’s changes
- the parent decides whether to merge them into the parent worktree

If multiple children produce conflicting work, the parent handles it.

This means merge conflict resolution is part of the parent’s job.

The parent is also responsible for code review.

So parent responsibilities are:

- code review
- merge decisions
- conflict resolution
- follow-up direction

---

## Universal node loop

Every node follows the same simple loop:

1. receive a task and a worktree
2. inspect the task
3. decide whether to:
   - solve directly, or
   - spawn children
4. if solving directly:
   - make changes
   - commit / produce output
   - halt
5. if spawning children:
   - create child tasks
   - wait for children to halt
   - review child work
   - loop children if needed
   - merge accepted work
   - halt

That is the whole recursive system.

---

## Why this may work

This structure turns software work into a tree:

- each node either handles a task directly
- or breaks it into smaller tasks
- parents review and integrate child work
- accepted progress moves upward through the tree

This is a natural way to handle complex coding tasks because it allows:

- decomposition
- parallel execution
- iterative refinement
- hierarchical integration

without requiring one agent to do everything in one long trajectory.

---

## Minimal assumptions

This architecture assumes only that Codex or Claude Code can already:

- read code
- edit code
- run commands/tests
- work inside a git worktree

The recursive system adds only:

- child spawning
- parent/child communication
- child worktree management
- parent-controlled review and merge

That keeps the design minimal.

---

## Summary

The system is defined by four rules:

1. every node works in a child git worktree derived from its parent
2. the parent reviews child work, handles merge conflicts, and can loop the child
3. any node can decide whether it should solve directly or spawn children
4. an external orchestrator facilitates communication between parents and children

That is the full architecture.
