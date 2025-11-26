import json
import yaml
import sys
import os
from pathlib import Path
from typing import Any, Dict

from agent.mid.supervisor import SupervisorOrchestrator
from agent.pre.clarifier import Clarifier


class EmptyLocalKB:
    def to_brief(self, n: int = 3):
        return []

class EmptyGlobalKB:
    def to_brief(self, n: int = 3):
        return []

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_task_name(structured_problem) -> str:
    raw_name = str(
        structured_problem.get("instruction_filename")
        or structured_problem.get("topic")
    )
    task_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(raw_name))
    return task_name


def clarify_query(query_path: str, clr_cfg) -> Dict[str, Any]:
    """
    兼容 main() 里的调用：
        structured_problem, task_dir = clarify_query(query_path, clarifier_cfg)
    """
    # 用 main 传进来的 query_path
    path = query_path

    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Clarifier 吃 clarifier 区那一整块 config（里面有 schema_file_path 等）
    clr = Clarifier(clr_cfg)
    structured_problem = clr.run(content)

    # 从文件名推一个 task_name
    instruction_filename = Path(path).stem
    structured_problem["instruction_filename"] = instruction_filename
    task_name = get_task_name(structured_problem)

    # 输出根目录：优先用 config 里的 output_path
    output_root = clr_cfg.get("output_path", "outputs")
    task_dir = Path(output_root) / task_name
    os.makedirs(task_dir, exist_ok=True)

    # 把 structured_problem 存下来，方便后面 para 那套来读
    with open(task_dir / "structured_problem.json", "w", encoding="utf-8") as f:
        json.dump(structured_problem, f, ensure_ascii=False, indent=2)

    return structured_problem, task_dir


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    clarifier_cfg = cfg.get("clarifier", {})
    pipeline_cfg = cfg.get("pipeline", {})
    mcts_cfg = cfg.get("mcts", {})
    query_path = cfg.get("query_file", "instructions/test.txt")

    structured_problem, task_dir = clarify_query(query_path,clarifier_cfg)

    kb_cfg = cfg.get("knowledge_base", {})
    if not kb_cfg.get("enabled", False):
        local_kb = EmptyLocalKB()
        global_kb = EmptyGlobalKB()
    else:
        local_kb = EmptyLocalKB()
        global_kb = EmptyGlobalKB()

    processes = pipeline_cfg.get("parallel_processes", 2)
    max_nodes = pipeline_cfg.get("max_nodes", 4)

    supervisor = SupervisorOrchestrator(
        structured_problem=structured_problem,
        local_kb=local_kb,
        global_kb=global_kb,
        task_dir=task_dir,
        processes=processes,
        max_nodes=max_nodes,
        draft_expansion=mcts_cfg.get("draft_expansion", 2),
        revise_expansion=mcts_cfg.get("revise_expansion", 2),
        improve_expansion=mcts_cfg.get("improve_expansion", 1),
        exploration_constant=mcts_cfg.get("exploration_constant", 1.414),
        complete_score_threshold=mcts_cfg.get("complete_score_threshold", 0.9),
    )

    summary = supervisor.run()

    summary_file = task_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Supervisor finished. Summary saved to:", summary_file)


if __name__ == "__main__":
    cfg_file = "config.yaml"
    main(cfg_file)
