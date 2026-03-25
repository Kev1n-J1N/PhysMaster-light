import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from e


def _safe_read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping (dict): {path}")
    return data


def discover_skill_manifests(skills_root: Path) -> List[Dict[str, Any]]:
    """
    Find all skill.yaml under skills_root/**/skill.yaml and parse them.
    Returns a list of dicts with resolved paths and normalized fields.
    """
    skills = []
    for skill_path in skills_root.rglob("skill.yaml"):
        data = _safe_read_yaml(skill_path)

        # Normalize fields
        skill_id = data.get("skill_id") or data.get("id")
        domain = data.get("domain") or ""
        goal = data.get("goal") or ""
        scope = data.get("scope") or []
        inputs = data.get("inputs") or {}
        outputs = data.get("outputs") or {}
        method = data.get("method") or []
        quality_gate = data.get("quality_gate") or []


        skills.append({
            "skill_id": skill_id,
            "domain": domain,
            "goal": goal,
            "scope": scope,
            "inputs": inputs,
            "outputs": outputs,
            "method": method,
            "quality_gate": quality_gate,
            "skill_path": str(skill_path.resolve()),
        })

    # Basic validation
    missing_id = [m for m in skills if not m["skill_id"]]
    if missing_id:
        bad = "\n".join(m["skill_path"] for m in missing_id[:10])
        raise ValueError(f"Some skill files missing skill_id (showing up to 10):\n{bad}")

    # Detect duplicates
    seen = {}
    dup = []
    for m in skills:
        sid = m.get("skill_id")
        if not sid:
            continue  # 跳过缺 skill_id 的（前面已经会报错）

        if sid in seen:
            dup.append((sid, seen[sid], m.get("skill_path", "<unknown>")))
        else:
            seen[sid] = m.get("skill_path", "<unknown>")

    if dup:
        msg = "\n".join([f"{sid}\n  - {a}\n  - {b}" for sid, a, b in dup])
        raise ValueError(f"Duplicate skill_id found:\n{msg}")

    return skills

def _format_scope(scope: Any, max_items: int = 6) -> str:
    if not scope:
        return ""
    if isinstance(scope, str):
        scope_items = [scope]
    elif isinstance(scope, list):
        scope_items = [str(s) for s in scope]
    else:
        scope_items = [str(scope)]

    scope_items = scope_items[:max_items]
    return "\n".join([f"    - {s}" for s in scope_items])


def build_progressive_disclosure_prompt(skills: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("[SKILL MANIFESTS]")
    lines.append("Below are available Skill manifests (summaries). Use them to decide which skills to load via load_skill_spec if needed.")
    lines.append("")

    for m in sorted(skills, key=lambda x: ((x.get("domain") or "").lower(), x["skill_id"].lower())):
        goal = (m.get("goal") or "").strip()
        domain = (m.get("domain") or "").strip()
        scope_txt = _format_scope(m.get("scope"))

        lines.append(f"- skill_id: {m['skill_id']}")
        if domain:
            lines.append(f"  domain: {domain}")
        if goal:
            goal_one = " ".join(goal.split())
            if len(goal_one) > 240:
                goal_one = goal_one[:237] + "..."
            lines.append(f"  goal: {goal_one}")
        if scope_txt:
            lines.append("  scope:")
            lines.append(scope_txt)
        lines.append("")

    return "\n".join(lines)



def main():
    skills_root = Path(r"./LANDAU/skills").resolve()
    if not skills_root.exists():
        raise SystemExit(f"skills_root not found: {skills_root}")

    skills = discover_skill_manifests(skills_root)
    prompt = build_progressive_disclosure_prompt(skills)

    current_dir = Path(__file__).resolve().parent
    out_dir = current_dir.parent / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = out_dir / "skills_manifest_prompt.txt"
    # index_path = out_dir / "skills_index.json"

    prompt_path.write_text(prompt, encoding="utf-8")
    # index_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"  - {prompt_path}")
    # print(f"  - {index_path}")
    print(f"Discovered {len(skills)} skill manifests under: {skills_root}")


if __name__ == "__main__":
    main()
