"""
MCTS节点类，用于构建搜索树
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCTSNode:
    """MCTS搜索树节点"""
    subtask_id: int
    node_index: int
    node_type: str  # "draft", "revise", "complete"
    
    # 节点状态
    subtask_description: str
    status: str = "pending"  # "pending", "completed", "failed", "pruned"
    
    # MCTS统计信息
    visits: int = 0  # 访问次数
    total_reward: float = 0.0  # 累计reward
    average_reward: float = 0.0  # 平均reward
    
    # 树结构
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    # 节点执行结果
    result: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    log_path: Optional[str] = None
    
    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """判断是否已完全扩展"""
        # 待修改，暂时简单返回False
        return False
    
    def get_ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """计算UCB1值，用于选择最有希望的节点"""
        if self.visits == 0:
            return float('inf')  # 未访问的节点优先探索
        
        if self.parent is None:
            return self.average_reward
        
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def select_best_child(self, exploration_constant: float = 1.414) -> Optional['MCTSNode']:
        """使用UCB1选择最佳子节点，忽略被剪枝的节点"""
        if not self.children:
            return None

        candidates = [c for c in self.children if c.status != "pruned"]
        if not candidates:
            return None

        best_child = max(
            candidates,
            key=lambda c: c.get_ucb1_value(exploration_constant)
        )
        return best_child
    
    def add_child(self, child: 'MCTSNode'):
        """添加子节点"""
        child.parent = self
        self.children.append(child)
    
    def update_stats(self, reward: float):
        """更新节点统计信息，反向传播时调用"""
        self.visits += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.visits
    
    def backpropagate(self, reward: float):
        """反向传播：更新从当前节点到根节点路径上的所有节点统计信息"""
        current = self
        while current is not None:
            current.update_stats(reward)
            current = current.parent
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        return {
            "subtask_id": self.subtask_id,
            "node_index": self.node_index,
            "node_type": self.node_type,
            "status": self.status,
            "visits": self.visits,
            "average_reward": self.average_reward,
            "children_count": len(self.children),
            "has_result": self.result is not None,
        }

    def get_depth(self) -> int:
        """返回节点在树中的深度（根节点深度为0）。

        通过沿着父指针向上统计层级获得深度。这比使用 subtask_id 可靠，
        因为同一 subtask 可在不同深度重复出现（redraft/revise/improve）。
        """
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
