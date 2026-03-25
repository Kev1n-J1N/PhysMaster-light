from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else (_project_root() / "config.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_skill_roots(config_path: str | Path | None = None) -> List[Path]:
    config = _load_config(config_path)
    configured = (((config.get("skills") or {}).get("roots")) or [])

    roots: List[Path] = []
    for raw in configured:
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = (_project_root() / candidate).resolve()
        roots.append(candidate)

    roots.append(_project_root() / "LANDAU" / "skills")

    unique: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    return parts[2].lstrip()


def _extract_summary(markdown_text: str) -> str:
    body = _strip_frontmatter(markdown_text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            continue
        summary = " ".join(paragraph.split())
        if summary:
            return summary[:280] + ("..." if len(summary) > 280 else "")
    return ""


def discover_skills(config_path: str | Path | None = None) -> List[Dict[str, Any]]:
    skills: List[Dict[str, Any]] = []
    for root in resolve_skill_roots(config_path):
        if not root.exists():
            continue
        for skill_file in sorted(root.rglob("SKILL.md")):
            raw_text = skill_file.read_text(encoding="utf-8")
            frontmatter = _parse_frontmatter(raw_text)
            skill_name = (
                frontmatter.get("name")
                or skill_file.parent.name
            )
            skills.append(
                {
                    "name": str(skill_name).strip(),
                    "description": str(frontmatter.get("description") or "").strip(),
                    "summary": _extract_summary(raw_text),
                    "path": skill_file.resolve(),
                    "root": root,
                }
            )
    skills.sort(key=lambda item: item["name"].lower())
    return skills


def build_skill_brief_prompt(config_path: str | Path | None = None) -> str:
    skills = discover_skills(config_path)
    lines = [
        "[SKILL BRIEF]",
        "Below are installed Codex-style skills discovered from configured skill roots.",
        "Use them to decide which skills to load in full via load_skill_specs(skill_names) if needed.",
        "",
    ]

    if not skills:
        lines.append("No installed skills were found.")
        return "\n".join(lines)

    for skill in skills:
        lines.append(f"- skill_name: {skill['name']}")
        if skill["description"]:
            lines.append(f"  description: {skill['description']}")
        elif skill["summary"]:
            lines.append(f"  description: {skill['summary']}")
        lines.append(f"  path: {skill['path']}")
    return "\n".join(lines)


def _resolve_skill_entries(skill_names: List[str], config_path: str | Path | None = None) -> List[Dict[str, Any]]:
    requested = [str(name).strip() for name in skill_names if str(name).strip()]
    if not requested:
        return []

    discovered = discover_skills(config_path)
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in discovered:
        by_name.setdefault(entry["name"], entry)

    missing = [name for name in requested if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Skill not found: {', '.join(missing)}")

    return [by_name[name] for name in requested]


def load_skill_specs(skill_names: List[str], config_path: str | Path | None = None) -> str:
    entries = _resolve_skill_entries(skill_names, config_path=config_path)
    blocks = [
        "[SKILL FULL]",
        "The following content is loaded from installed Codex-style SKILL.md files.",
        "Treat these skills as authoritative workflow guidance when they are relevant to the task.",
        "",
    ]

    for entry in entries:
        content = entry["path"].read_text(encoding="utf-8").strip()
        blocks.append(
            "<SKILL_FULL>\n"
            f"name: {entry['name']}\n"
            f"path: {entry['path']}\n"
            "---\n"
            f"{content}\n"
            "</SKILL_FULL>"
        )
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"
