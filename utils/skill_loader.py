from pathlib import Path
from typing import List
import yaml  # (kept, even if unused here, in case you want to parse later)

current_dir = Path(__file__).resolve().parent
SKILLS_ROOT = current_dir.parent / "LANDAU" / "methodology" / "skills"

class _PrettyDumper(yaml.SafeDumper):
    pass

def _str_presenter(dumper, data: str):
    # 如果字符串里有换行，就用 block style 输出
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

_PrettyDumper.add_representer(str, _str_presenter)

def remove_yaml_comments_and_prettify(yaml_text: str) -> str:
    data = yaml.safe_load(yaml_text)
    return yaml.dump(
        data,
        Dumper=_PrettyDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()

def load_skill_specs(skill_ids: List[str]) -> str:
    # de-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for sid in skill_ids:
        if sid not in seen:
            seen.add(sid)
            uniq.append(sid)

    # Big header so the model knows this is injected skill context, not a normal tool output
    blocks = [
        "[SKILL SPECS]",
        "The following content is Skill specifications (not ordinary tool output). "
        "Treat them as authoritative procedural knowledge and follow them when solving the task.",
        "",
    ]

    for skill_id in uniq:
        candidate_p =[
            SKILLS_ROOT / "general" / skill_id / "skill.yaml",
            SKILLS_ROOT / "mathematics" / skill_id / "skill.yaml",
        ]
        for p in candidate_p:
            if p.exists():
                break
        else:
            raise FileNotFoundError(f"Skill not found: {skill_id}")


        yaml_text = p.read_text(encoding="utf-8")
        yaml_text = remove_yaml_comments_and_prettify(yaml_text)


        blocks.append(
            "<SKILL_SPEC>\n"
            f"{yaml_text}\n"
            "</SKILL_SPEC>"
        )
        blocks.append("")  # blank line between skills

    return "\n".join(blocks).rstrip() + "\n"
