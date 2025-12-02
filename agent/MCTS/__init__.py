"""
MCTS (Monte Carlo Tree Search) module for multi-agent physics reasoning.

This module implements the complete MCTS algorithm with:
- Selection (UCB1)
- Expansion (parallel node generation)
- Simulation (node execution)
- Backpropagation (reward propagation)
"""

from .mcts_node import MCTSNode
from .mcts_tree import MCTSTree
from .supervisor import SupervisorOrchestrator
from .theoretician import run_theo_node

__all__ = ["MCTSNode", "MCTSTree", "SupervisorOrchestrator", "run_theo_node"]

