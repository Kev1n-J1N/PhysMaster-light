"""MCTS tree structures and node utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCTSNode:
    """MCTS 搜索树节点。"""

    subtask_id: int
    node_id: int
    node_type: str  # "virtual" | "draft" | "revise"
    subtask_description: str

    subtask_payload: Optional[Dict[str, Any]] = None
    created_by: str = "supervisor"

    # "open" | "completed" | "completed_expended" | "completed_closed" | "failed"
    status: str = "open"

    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    reward: float = 0.0

    # Tree links
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)

    # Node artifacts
    result: Optional[Any] = None
    evaluation: Optional[Dict[str, Any]] = None
    supervisor_dispatch: Optional[Dict[str, Any]] = None
    supervisor_feedback: Optional[Dict[str, Any]] = None
    theoretician_output: Optional[Any] = None
    memory: str = ""
    selected_round: Optional[int] = None
    log_path: Optional[str] = None

    def __post_init__(self):
        if self.node_id is None:
            self.node_id = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        # This project controls expansion in supervisor; keep semantic helper.
        return self.status in {"completed_closed", "failed"}

    def get_reward_value(self) -> float:
        if self.evaluation and self.evaluation.get("reward") is not None:
            try:
                return float(self.evaluation["reward"])
            except Exception:
                pass
        if self.reward is not None:
            return float(self.reward)
        return float(self.average_reward or 0.0)

    def get_ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """Compute UCB1 value for child selection."""
        if self.visits <= 0:
            return float("inf")

        if self.parent is None:
            return self.get_reward_value()

        parent_visits = max(1, int(self.parent.visits))
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )
        return exploitation + exploration

    def select_best_child(self, exploration_constant: float = 1.414) -> Optional["MCTSNode"]:
        """Use UCB1 to select best child while skipping closed/failed nodes."""
        if not self.children:
            return None

        candidates = [c for c in self.children if c.status not in {"completed_closed", "failed"}]
        if not candidates:
            return None

        return max(candidates, key=lambda c: c.get_ucb1_value(exploration_constant))

    def add_child(self, child: "MCTSNode"):
        child.parent = self
        self.children.append(child)

    def update_stats(self, reward: float):
        self.visits += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.visits

    def backpropagate(self, reward: float):
        current: Optional["MCTSNode"] = self
        while current is not None:
            current.update_stats(reward)
            current = current.parent

    def get_depth(self) -> int:
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def is_subtask_complete(self) -> bool:
        feedback = self.evaluation or {}
        decision = str(feedback.get("decision", "")).strip().lower()
        verdict = str(feedback.get("verdict", "")).strip().lower()
        return decision == "complete" or verdict == "accept"

    def node_id_number(self) -> int:
        return int(self.node_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "subtask_id": self.subtask_id,
            "node_type": self.node_type,
            "status": self.status,
            "visits": self.visits,
            "reward": self.reward,
            "average_reward": self.average_reward,
            "children_count": len(self.children),
            "has_result": self.result is not None,
        }


class MCTSTree:
    """MCTS 搜索树管理器。"""

    def __init__(self, root_subtask_id: int, root_description: str):
        self.root = MCTSNode(
            subtask_id=root_subtask_id,
            node_id=0,
            node_type="virtual",
            subtask_description=root_description,
            status="completed_expended",
            created_by="root",
        )
        self.nodes: Dict[int, MCTSNode] = {self.root.node_id: self.root}
        self.subtask_roots: Dict[int, MCTSNode] = {root_subtask_id: self.root}

    def get_node(self, node_id: int) -> Optional[MCTSNode]:
        return self.nodes.get(node_id)

    def get_node_by_id(self, node_id: int) -> Optional[MCTSNode]:
        return self.nodes.get(node_id)

    def add_node(self, node: MCTSNode):
        self.nodes[node.node_id] = node
        if node.subtask_id not in self.subtask_roots:
            self.subtask_roots[node.subtask_id] = node

    def selection(self, exploration_constant: float = 1.414) -> MCTSNode:
        """Selection phase: walk from root with UCB1 until leaf."""
        current = self.root
        visited = set()
        while current.children:
            if current.node_id in visited:
                break
            visited.add(current.node_id)
            nxt = current.select_best_child(exploration_constant)
            if nxt is None:
                break
            current = nxt
        return current

    def get_subtask_root(self, subtask_id: int) -> Optional[MCTSNode]:
        return self.subtask_roots.get(subtask_id)

    def get_all_nodes(self) -> List[MCTSNode]:
        return list(self.nodes.values())

    def get_tree_stats(self) -> Dict[str, Any]:
        depths: Dict[int, int] = {}
        node_type_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            depths[node.get_depth()] = depths.get(node.get_depth(), 0) + 1
            node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "subtasks": len(self.subtask_roots),
            "nodes_by_subtask": {
                sid: len([n for n in self.nodes.values() if n.subtask_id == sid])
                for sid in self.subtask_roots
            },
            "nodes_by_depth": depths,
            "node_type_counts": node_type_counts,
        }
