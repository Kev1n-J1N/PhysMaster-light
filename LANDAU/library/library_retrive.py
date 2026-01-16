from __future__ import annotations

from typing import Any, Dict, List


class LibraryRetriever:
    """Placeholder retriever for the library KB.

    This stub exists to keep imports and tool calls from failing.
    Replace the implementation with a real RAG-backed retriever later.
    """

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sources: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        return []

    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            text = r.get("text", "")
            lines.append(f"{i}. {title}\n{text}")
        return "\n".join(lines)
