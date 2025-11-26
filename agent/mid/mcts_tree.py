"""
MCTS搜索树管理类
"""
from typing import Dict, List, Optional
from .mcts_node import MCTSNode


class MCTSTree:
    """MCTS搜索树管理器"""
    
    def __init__(self, root_subtask_id: int, root_description: str):
        """初始化MCTS树，创建根节点"""
        self.root = MCTSNode(
            subtask_id=root_subtask_id,
            node_index=0,
            node_type="draft",
            subtask_description=root_description,
            status="pending",
        )
        self.nodes: Dict[int, MCTSNode] = {self.root.node_index: self.root}
        self.subtask_roots: Dict[int, MCTSNode] = {root_subtask_id: self.root}
    
    def get_node(self, node_index: int) -> Optional[MCTSNode]:
        """根据 node_index 获取节点"""
        return self.nodes.get(node_index)
    
    def add_node(self, node: MCTSNode):
        """添加节点到树中"""
        self.nodes[node.node_index] = node
        if node.subtask_id not in self.subtask_roots:
            self.subtask_roots[node.subtask_id] = node
    
    def selection(self, exploration_constant: float = 1.414) -> MCTSNode:
        """
        选择阶段：从根节点开始，使用UCB1策略选择最有希望的叶子节点
        """
        current = self.root
        
        # 向下遍历直到叶子节点
        while current.children:
            current = current.select_best_child(exploration_constant)
            if current is None:
                break
        
        return current if current else self.root
    
    def get_subtask_root(self, subtask_id: int) -> Optional[MCTSNode]:
        """获取某个subtask的根节点"""
        return self.subtask_roots.get(subtask_id)
    
    def get_all_nodes(self) -> List[MCTSNode]:
        """获取所有节点"""
        return list(self.nodes.values())
    
    def get_tree_stats(self) -> Dict:
        """获取树的统计信息"""
        return {
            "total_nodes": len(self.nodes),
            "subtasks": len(self.subtask_roots),
            "nodes_by_subtask": {
                sid: len([n for n in self.nodes.values() if n.subtask_id == sid])
                for sid in self.subtask_roots.keys()
            },
        }

