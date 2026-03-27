"""Scheduler – V1 will add concurrency control here. V0 is serial."""

from __future__ import annotations

from recursive_intelligence.runtime.state_store import NodeState, StateStore


def get_ready_nodes(store: StateStore, run_id: str) -> list[str]:
    """Return node IDs that are queued and ready to be claimed."""
    return [n.node_id for n in store.get_nodes_in_state(run_id, NodeState.QUEUED)]


def get_waiting_parents(store: StateStore, run_id: str) -> list[str]:
    """Return node IDs waiting on children where all children are terminal."""
    waiting = store.get_nodes_in_state(run_id, NodeState.WAITING_ON_CHILDREN)
    ready = []
    for parent in waiting:
        children = store.get_children(parent.node_id)
        if children and all(c.state.is_terminal or c.state == NodeState.COMPLETED for c in children):
            ready.append(parent.node_id)
    return ready
