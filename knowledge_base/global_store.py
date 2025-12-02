from typing import Any, Dict, List


class GlobalKnowledgeBase:
    """
    暂时的占位 Global KB：
    - 以后可以接 FAISS / 向量索引
    - 现在只是提供一个接口，避免 import 报错
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def to_brief(self, n: int = 3) -> List[str]:
        return [f"[GlobalKB] Placeholder entry {i+1}" for i in range(n)]

    def search(self, query: str, top_k: int = 5) -> List[str]:
        return [
            f"[GlobalKB] Placeholder search result {i+1} for query: {query}"
            for i in range(top_k)
        ]
