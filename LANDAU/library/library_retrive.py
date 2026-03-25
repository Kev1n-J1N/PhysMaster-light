from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

from utils.llm_client import call_model_without_tools


class LibraryRetriever:
    """Web-backed library retriever.

    It exposes two capabilities:
    - search: discover relevant web sources
    - parse: fetch a concrete page/PDF and extract task-relevant content
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.config = self._load_project_config()
        self._api_base_url = self._resolve_api_base_url()
        self._search_defaults = self._load_web_defaults()

    def _load_project_config(self) -> Dict[str, Any]:
        path = self.project_root / "config.yaml"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_api_base_url(self) -> str:
        library_cfg = ((self.config.get("landau") or {}).get("library_config")) or {}
        configured = library_cfg.get("api_base_url")
        if configured:
            return str(configured).rstrip("/")

        legacy_path = self.project_root / "x_master" / "mcp_sandbox" / "configs" / "mcp_config.json"
        if legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as f:
                    return str((json.load(f) or {}).get("tool_api_url", "http://127.0.0.1:1234")).rstrip("/")
            except Exception:
                pass
        return "http://127.0.0.1:1234"

    def _load_web_defaults(self) -> Dict[str, Any]:
        defaults = {
            "serper_api_key": "",
            "search_region": "us",
            "search_lang": "en",
            "parse_model": None,
        }
        library_cfg = ((self.config.get("landau") or {}).get("library_config")) or {}
        defaults.update(
            {
                "serper_api_key": library_cfg.get("serper_api_key", defaults["serper_api_key"]),
                "search_region": library_cfg.get("search_region", defaults["search_region"]),
                "search_lang": library_cfg.get("search_lang", defaults["search_lang"]),
                "parse_model": library_cfg.get("parse_model", defaults["parse_model"]),
            }
        )

        legacy_path = self.project_root / "x_master" / "mcp_sandbox" / "configs" / "web_agent.json"
        if legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as f:
                    legacy = json.load(f) or {}
                defaults["serper_api_key"] = defaults["serper_api_key"] or legacy.get("serper_api_key", "")
                defaults["search_region"] = defaults["search_region"] or legacy.get("search_region", "us")
                defaults["search_lang"] = defaults["search_lang"] or legacy.get("search_lang", "en")
                defaults["parse_model"] = defaults["parse_model"] or legacy.get("USE_MODEL")
            except Exception:
                pass
        return defaults

    def _post_json(self, route: str, payload: Dict[str, Any], timeout: int = 30) -> Any:
        url = f"{self._api_base_url}/{route.lstrip('/')}"
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sources: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        return self.search(query=query, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: int = 5,
        region: str | None = None,
        lang: str | None = None,
        depth: int = 0,
    ) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "serper_api_key": self._search_defaults["serper_api_key"],
            "top_k": int(top_k),
            "region": region or self._search_defaults["search_region"],
            "lang": lang or self._search_defaults["search_lang"],
            "depth": int(depth),
        }
        results = self._post_json("search", payload, timeout=30)
        if isinstance(results, dict):
            results = results.get("organic", [])
        if not isinstance(results, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in results[:top_k]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "text": item.get("snippet", ""),
                }
            )
        return normalized

    def parse(
        self,
        link: str,
        user_prompt: str,
        llm: str | None = None,
    ) -> Dict[str, Any]:
        is_pdf = ".pdf" in link or "arxiv.org/abs" in link or "arxiv.org/pdf" in link
        route = "read_pdf" if is_pdf else "fetch_web"
        timeout = 60 if is_pdf else 30
        raw_content = self._post_json(route, {"url": link}, timeout=timeout)

        if isinstance(raw_content, dict):
            source_text = (
                raw_content.get("markdown")
                or raw_content.get("text")
                or raw_content.get("content")
                or json.dumps(raw_content, ensure_ascii=False)
            )
        else:
            source_text = str(raw_content)

        source_text = (source_text or "").strip()
        if not source_text or source_text.startswith("Failed to"):
            return {"content": "", "urls": [], "score": -1, "error": source_text or "Failed to fetch content"}

        prompt = (
            "You are a library retrieval assistant. Read the provided source content and answer the user request "
            "without fabricating information. Return a JSON object with keys content, urls, score. "
            "The urls field must be a list of objects with keys url and description. "
            "If no extra related URLs are available, return an empty list. "
            "Score must be a float between 0 and 1.\n\n"
            f"Source URL: {link}\n"
            f"User request: {user_prompt}\n"
            f"Source content:\n{source_text[:120000]}"
        )
        response = call_model_without_tools(
            system_prompt="Return valid JSON only.",
            user_prompt=prompt,
            model_name=llm or self._search_defaults["parse_model"],
        )
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            parsed = json.loads(response[start:end])
            if isinstance(parsed, dict):
                parsed.setdefault("content", "")
                parsed.setdefault("urls", [])
                parsed.setdefault("score", 0.0)
                return parsed
        except Exception:
            pass
        return {"content": response.strip(), "urls": [], "score": 0.5}

    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            link = r.get("link", "")
            text = r.get("text", "") or r.get("snippet", "")
            lines.append(f"{i}. {title}\nURL: {link}\n{text}")
        return "\n".join(lines)

    def format_parsed_for_llm(self, parsed: Dict[str, Any]) -> str:
        if not parsed:
            return ""
        content = str(parsed.get("content", "")).strip()
        score = parsed.get("score", "")
        urls = parsed.get("urls", []) or []
        lines = [f"score: {score}", content]
        if urls:
            lines.append("related_urls:")
            for item in urls:
                if not isinstance(item, dict):
                    continue
                lines.append(f"- {item.get('url', '')}: {item.get('description', '')}")
        return "\n".join(line for line in lines if line)
