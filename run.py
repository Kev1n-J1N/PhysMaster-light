import json
import yaml
import sys
import os

from pathlib import Path
from typing import Any, Dict, Tuple

from agent.MCTS.supervisor import SupervisorOrchestrator
from agent.clarifier.clarifier import Clarifier
from agent.librarian.librarian import run_pasax_for_kb


class EmptyLocalKB:
    def to_brief(self, n: int = 3):
        return []

    def search(self, query: str, top_k: int = 5):
        """
        占位 KB 搜索函数。
        真实 KB 实现后会替换这个逻辑。
        现在为了不让程序报错，统一返回空列表。
        """
        return []


class EmptyGlobalKB:
    def to_brief(self, n: int = 3):
        return []

    def search(self, query: str, top_k: int = 5):
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
    path = query_path
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    clr = Clarifier(clr_cfg)
    structured_problem = clr.run(content)

    instruction_filename = Path(path).stem
    structured_problem["instruction_filename"] = instruction_filename
    task_name = get_task_name(structured_problem)

    output_root = clr_cfg.get("output_path", "outputs")
    task_dir = Path(output_root) / task_name
    os.makedirs(task_dir, exist_ok=True)

    with open(task_dir / "contract.json", "w", encoding="utf-8") as f:
        json.dump(structured_problem, f, ensure_ascii=False, indent=2)

    return structured_problem, task_dir


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    clarifier_cfg = cfg.get("clarifier", {})
    pipeline_cfg = cfg.get("pipeline", {})
    mcts_cfg = cfg.get("mcts", {})

    query_path = clarifier_cfg.get("query_file", "instructions/test.txt")

    structured_problem, task_dir = clarify_query(query_path, clarifier_cfg)
    task_name = get_task_name(structured_problem)

    kb_cfg = cfg.get("knowledge_base", {}) or {}
    task_local_kb_dir = None  

    if not kb_cfg.get("enabled", False):
        print("[KB] knowledge_base.enabled = False，跳过 PasaX 和 KB 加载")
        local_kb = EmptyLocalKB()
        global_kb = EmptyGlobalKB()
    else:
        task_local_kb_dir = run_pasax_for_kb(task_dir, task_name, kb_cfg)

        from knowledge_base.local_store import LocalKnowledgeBase
        from knowledge_base.global_store import GlobalKnowledgeBase

        project_root = Path(__file__).resolve().parent
        base_local_root = kb_cfg.get("local_root", "knowledge_base/local_knowledge_base")

        if task_local_kb_dir is not None:
            local_root_dir = task_local_kb_dir
        else:
            local_root_dir = (project_root / base_local_root / task_name).resolve()

        print(f"[LocalKB] Using local KB dir: {local_root_dir}")
        local_kb = LocalKnowledgeBase(str(local_root_dir))

        global_cfg = kb_cfg.get("global", {}) or {}
        if global_cfg.get("enabled", False):
            global_kb = GlobalKnowledgeBase(global_cfg)
        else:
            global_kb = EmptyGlobalKB()

        if kb_cfg.get("merge_local_into_global", False):
            print("[KB] merge_local_into_global = True，global_kb 将与 local_kb 共用。")
            global_kb = local_kb

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
