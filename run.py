import json
import yaml
import os

from pathlib import Path
from typing import Any, Dict, Tuple

from core.supervisor import SupervisorOrchestrator
from core.summarizer import TrajectorySummarizer
from core.clarifier import Clarifier
from core.visualization import generate_vis
from utils.skill_loader import resolve_skill_roots


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
        print(f"[LANDAU] Workflow: enabled")
    else:
        print("[LANDAU] Workflow: disabled")

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
    print(f"[Clarifier] Structured problem saved to: {task_dir / 'contract.json'}")
    
    return structured_problem, task_dir, task_name


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    pipeline_cfg = cfg.get("pipeline", {})
    clarifier_cfg = cfg.get("clarifier", {})
    query_path = pipeline_cfg.get("query_file", "instructions/test.txt")
    output_root = pipeline_cfg.get("output_path", "outputs")
    clarifier_cfg["output_path"] = output_root

    landau_cfg = cfg.get("landau", {})
    library_enabled = bool(landau_cfg.get("library_enabled", True))
    workflow_enabled = bool(landau_cfg.get("workflow_enabled", True))
    prior_enabled = bool(landau_cfg.get("prior_enabled", True))
    skills_cfg = cfg.get("skills", {})
    skills_enabled = bool(skills_cfg.get("enabled", True))

    mcts_cfg = cfg.get("mcts", {})
    vis_cfg = cfg.get("visualization",{})

    project_root = Path(__file__).resolve().parent
    library_root = (project_root / landau_cfg.get("library", "LANDAU/library")).resolve()
    workflow_root = (
        project_root
        / landau_cfg.get("workflow", landau_cfg.get("methodology", "LANDAU/workflow"))
    ).resolve()
    prior_root = (project_root / landau_cfg.get("prior", "LANDAU/prior")).resolve()

    clarifier_cfg["workflow_dir"] = str(workflow_root)

    structured_problem, task_dir, task_name = clarify_query(
        query_path, clarifier_cfg, workflow_enabled=workflow_enabled
    )

    if library_enabled:
        print(f"[LANDAU] Library: {library_root}")
    else:
        print("[LANDAU] Library: disabled")
    if workflow_enabled:
        print(f"[LANDAU] Workflow Dir: {workflow_root}")
    else:
        print("[LANDAU] Workflow Dir: disabled")
    if skills_enabled:
        print("[Skills] Roots:")
        for root in resolve_skill_roots(config_path):
            print(f"  - {root}")
    else:
        print("[Skills] disabled")
    if prior_enabled:
        print(f"[LANDAU] Prior: {prior_root}")
    else:
        print("[LANDAU] Prior: disabled")

    processes = pipeline_cfg.get("parallel_processes", 2)
    max_rounds = pipeline_cfg.get("max_rounds", 8)

    supervisor = SupervisorOrchestrator(
        structured_problem=structured_problem,
        task_dir=task_dir,
        processes=processes,
        max_rounds=max_rounds,
        draft_expansion=mcts_cfg.get("draft_expansion", 2),
        revise_expansion=mcts_cfg.get("revise_expansion", 2),
        exploration_constant=mcts_cfg.get("exploration_constant", 1.414),
        active_beam_width=mcts_cfg.get("active_beam_width", 0),
        landau_library_enabled=library_enabled,
        landau_prior_enabled=prior_enabled,
    )

    mcts_result = supervisor.run()
    trajectory = mcts_result.get("trajectory", []) or []

    summarizer = TrajectorySummarizer(prompts_path="prompts/")
    summary_md_path = task_dir / "summary.md"
    contract_for_summary = json.dumps(structured_problem, ensure_ascii=False, indent=2)
    summarizer.write_summary_markdown(
        summary_md_path,
        task_description=contract_for_summary,
        trajectory=trajectory,
    )
    summary_text = summary_md_path.read_text(encoding="utf-8")
    print("[Summrizer] Summary generated:", summary_md_path)

    if vis_cfg.get("enabled",False):
        vis_path = task_dir / "visualization.html"
        generate_vis(
            vis_path,
            supervisor.tree,
            task_description=structured_problem.get("task_description", ""),
            subtasks=supervisor.subtasks,
            summary=summary_text,
        )
        print("[Visulization] visualization saved:", vis_path)
    

if __name__ == "__main__":
    cfg_file = "config.yaml"
    main(cfg_file)
