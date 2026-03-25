"""Generate MCTS visualization HTML with the new PhysMaster template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mcts import MCTSTree

TEMPLATE_FILE = Path(__file__).resolve().parent / "visualization_template.html"
INJECT_MARKER = "<!-- __DATA_INJECT__ -->"


def _compute_tree_layout(nodes: list[dict[str, Any]], root_id: int = 0) -> dict[int, tuple[float, float]]:
    by_id = {int(n["node_id"]): n for n in nodes}
    children_map: dict[int, list[int]] = {
        int(n["node_id"]): [int(c) for c in n.get("children", [])] for n in nodes
    }

    levels: dict[int, list[int]] = {}
    queue = [(int(root_id), 0)]
    visited = set()
    while queue:
        node_id, depth = queue.pop(0)
        if node_id in visited or node_id not in by_id:
            continue
        visited.add(node_id)
        levels.setdefault(depth, []).append(node_id)
        for child_id in children_map.get(node_id, []):
            queue.append((child_id, depth + 1))

    if not levels:
        return {}

    max_depth = max(levels.keys())
    coords: dict[int, tuple[float, float]] = {}
    for depth, ids in levels.items():
        count = max(1, len(ids))
        y = 0.08 + (0.84 * (depth / max(1, max_depth)))
        for i, node_id in enumerate(ids):
            x = 0.1 + (0.8 * ((i + 1) / (count + 1)))
            coords[node_id] = (x, y)
    return coords


def _safe_short(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def _serialize_tree(tree: MCTSTree) -> list[dict[str, Any]]:
    nodes_payload: list[dict[str, Any]] = []
    for node in sorted(tree.get_all_nodes(), key=lambda n: int(n.node_id)):
        node_id = int(getattr(node, "node_id", 0))
        parent = getattr(node, "parent", None)
        parent_id = int(parent.node_id) if parent is not None else None
        children = [int(c.node_id) for c in node.children]

        subtask = node.subtask_payload
        if subtask is None:
            subtask = {
                "id": int(node.subtask_id) if int(node.subtask_id) > 0 else None,
                "description": node.subtask_description,
            }

        reward = float(getattr(node, "reward", 0.0) or 0.0)
        if node.node_type != "virtual":
            try:
                reward = float(node.get_reward_value())
            except Exception:
                reward = float(getattr(node, "reward", 0.0) or 0.0)

        nodes_payload.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "children": children,
                "depth": node.get_depth(),
                "node_type": node.node_type,
                "subtask": subtask,
                "description": _safe_short(node.subtask_description, 4000),
                "reward": reward,
                "visits": int(getattr(node, "visits", 0) or 0),
                "status": getattr(node, "status", "open"),
                "created_by": getattr(node, "created_by", "supervisor"),
                "memory": _safe_short(getattr(node, "memory", ""), 4000),
                "theoretician_output": _safe_short(getattr(node, "theoretician_output", ""), 48000),
                "supervisor_dispatch": getattr(node, "supervisor_dispatch", {}) or {},
                "critic_feedback": getattr(node, "evaluation", {}) or {},
                "supervisor_feedback": getattr(node, "supervisor_feedback", {}) or {},
                "selected_round": getattr(node, "selected_round", None),
            }
        )
    return nodes_payload


def build_payload(
    *,
    nodes: list[dict[str, Any]],
    root_id: int,
    task_description: str,
    subtasks: Any = None,
    summary: str = "",
) -> dict[str, Any]:
    coords = _compute_tree_layout(nodes, root_id=root_id)

    edges: list[list[int]] = []
    for node in nodes:
        for child_id in node.get("children", []):
            if child_id in coords and node["node_id"] in coords:
                edges.append([int(node["node_id"]), int(child_id)])

    payload_nodes = []
    for node in nodes:
        x, y = coords.get(node["node_id"], (0.5, 0.5))
        copied = dict(node)
        copied["viz_x"] = x
        copied["viz_y"] = y
        payload_nodes.append(copied)

    return {
        "task_description": task_description,
        "subtasks": subtasks if subtasks is not None else [],
        "summary": summary or "",
        "root_id": int(root_id),
        "nodes": payload_nodes,
        "edges": edges,
    }


def build_mcts_html(
    *,
    nodes: list[dict[str, Any]],
    root_id: int,
    task_description: str,
    subtasks: Any = None,
    summary: str = "",
) -> str:
    payload = build_payload(
        nodes=nodes,
        root_id=root_id,
        task_description=task_description,
        subtasks=subtasks,
        summary=summary,
    )
    payload_json = json.dumps(payload, ensure_ascii=False)

    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    inject = f"<script>window.__PHY_MCTS_DATA__ = {payload_json};</script>"
    return template.replace(INJECT_MARKER, inject)


def write_mcts_html(
    output_path: str | Path,
    *,
    nodes: list[dict[str, Any]],
    root_id: int = 0,
    task_description: str = "",
    subtasks: Any = None,
    summary: str = "",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_mcts_html(
        nodes=nodes,
        root_id=root_id,
        task_description=task_description,
        subtasks=subtasks,
        summary=summary,
    )
    out.write_text(html, encoding="utf-8")
    return out


def generate_vis(
    output_path: str | Path,
    tree: MCTSTree,
    task_description: str = "",
    subtasks: Any = None,
    summary: str = "",
) -> Path:
    nodes = _serialize_tree(tree)
    root_id = int(getattr(tree.root, "node_id", 0))
    return write_mcts_html(
        output_path=output_path,
        nodes=nodes,
        root_id=root_id,
        task_description=task_description,
        subtasks=subtasks if subtasks is not None else [],
        summary=summary,
    )
