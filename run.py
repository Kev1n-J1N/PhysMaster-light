import json
import yaml
import os

from pathlib import Path
from typing import Any, Dict, Tuple

from agent.MCTS.supervisor import SupervisorOrchestrator
from agent.clarifier.clarifier import Clarifier
from visualization.generate_html import generate_vis


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


def clarify_query(query_path: str, clr_cfg, workflow_enabled: bool = True) -> Dict[str, Any]:
    """
    兼容 main() 里的调用：
        structured_problem, task_dir = clarify_query(query_path, clarifier_cfg)
    """
    path = query_path
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    print(f"[Clarifier] Clarifying query from: {path}")

    if workflow_enabled:
        print(f"[LANDAU] Methodology-Workflow: enabled")
    else:
        print("[LANDAU] Methodology-Workflow: disabled")

    clr = Clarifier(clr_cfg, workflow_enabled=workflow_enabled)
    structured_problem = clr.run(content)

    instruction_filename = Path(path).stem
    structured_problem["instruction_filename"] = instruction_filename
    task_name = get_task_name(structured_problem)

    output_root = clr_cfg.get("output_path", "outputs")
    task_dir = Path(output_root) / task_name
    os.makedirs(task_dir, exist_ok=True)

    with open(task_dir / "contract.json", "w", encoding="utf-8") as f:
        json.dump(structured_problem, f, ensure_ascii=False, indent=2)
    print(f"[CLR] Structured problem saved to: {task_dir / 'contract.json'}")
    
    return structured_problem, task_dir, task_name


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    clarifier_cfg = cfg.get("clarifier", {})
    query_path = clarifier_cfg.get("query_file", "instructions/test.txt")

    landau_cfg = cfg.get("landau", {})
    library_enabled = bool(landau_cfg.get("library_enabled", True))
    workflow_enabled = bool(landau_cfg.get("workflow_enabled", True))
    skills_enabled = bool(landau_cfg.get("skills_enabled", True))
    prior_enabled = bool(landau_cfg.get("prior_enabled", True))

    pipeline_cfg = cfg.get("pipeline", {})
    mcts_cfg = cfg.get("mcts", {})
    vis_cfg = cfg.get("visualization",{})

    structured_problem, task_dir, task_name = clarify_query(
        query_path, clarifier_cfg, workflow_enabled=workflow_enabled
    )

    project_root = Path(__file__).resolve().parent
    library_root = (project_root / landau_cfg.get("library", "LANDAU/library")).resolve()
    methodology_root = (project_root / landau_cfg.get("methodology", "LANDAU/global_methodology")).resolve()
    prior_root = (project_root / landau_cfg.get("prior", "LANDAU/global_prior")).resolve()

    if library_enabled:
        print(f"[LANDAU] Library: {library_root}")
    else:
        print("[LANDAU] Library: disabled")
    if skills_enabled:
        print(f"[LANDAU] Methodology-Skills: {methodology_root}/skills")
    else:
        print("[LANDAU] Methodology-Skills: disabled")
    if prior_enabled:
        print(f"[LANDAU] Prior: {prior_root}")
    else:
        print("[LANDAU] Prior: disabled")

    processes = pipeline_cfg.get("parallel_processes", 2)
    max_nodes = pipeline_cfg.get("max_nodes", 4)

    supervisor = SupervisorOrchestrator(
        structured_problem=structured_problem,
        task_dir=task_dir,
        processes=processes,
        max_nodes=max_nodes,
        draft_expansion=mcts_cfg.get("draft_expansion", 2),
        revise_expansion=mcts_cfg.get("revise_expansion", 2),
        improve_expansion=mcts_cfg.get("improve_expansion", 1),
        exploration_constant=mcts_cfg.get("exploration_constant", 1.414),
        beam_width=mcts_cfg.get("beam_width"),
        landau_prior_enabled=prior_enabled,
    )

    summary = supervisor.run()

    if vis_cfg.get("enabled",False):
        vis_path = task_dir / "visualization.html"
        generate_vis(vis_path,supervisor.tree)
        print("visualization succeed")

    summary_file = task_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Supervisor finished. Summary saved to:", summary_file)


if __name__ == "__main__":
    cfg_file = "config.yaml"
    main(cfg_file)
