import json
import os
from typing import Any, Dict, List


class LocalKnowledgeBase:
    """
    一个最小可用的本地知识库：
    - 从某个文件夹里读取 librarian 生成的 JSON 文件
    - 把其中的论文条目展开成一个 list
    - 提供最简单的关键词 search(query, top_k)
    """

    def __init__(self, root_dir: str):
        """
        root_dir: 存放 JSON 的目录，比如 'librarian'
        """
        self.root_dir = root_dir
        self.entries: List[Dict[str, Any]] = []
        self._load_from_dir()

    def _load_from_dir(self):
        print(f"[LocalKB] Loading from: {self.root_dir}")
        if not os.path.isdir(self.root_dir):
            print(f"[LocalKB] Directory not found: {self.root_dir}")
            return

        total_files = 0
        total_entries = 0

        for fname in os.listdir(self.root_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.root_dir, fname)
            total_files += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[LocalKB] Failed to load {path}: {e}")
                continue

            candidates = []

            # 情形 1：data 是 dict，尝试 extra.recall_papers / extra.top_papers
            if isinstance(data, dict):
                extra = data.get("extra", {}) or {}
                if isinstance(extra, dict):
                    rp = extra.get("recall_papers") or extra.get("top_papers")
                    if isinstance(rp, list):
                        candidates.extend(rp)

                # 情形 2：顶层就有 papers / results / items
                for key in ["papers", "results", "items"]:
                    if isinstance(data.get(key), list):
                        candidates.extend(data[key])

            # 情形 3：整个 data 就是一个 list
            if isinstance(data, list):
                candidates.extend(data)

            # 抽取每个 candidate 里的 title / abstract 等
            for p in candidates:
                if not isinstance(p, dict):
                    continue
                entry = {
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "venue": p.get("venue", "") or p.get("journal", ""),
                    "year": p.get("year", ""),
                    "arxiv_id": p.get("arxiv_id", "") or p.get("id", ""),
                    "url": p.get("url", "") or p.get("pdf_url", ""),
                }
                if entry["title"] or entry["abstract"]:
                    self.entries.append(entry)
                    total_entries += 1

        print(f"[LocalKB] Loaded {total_entries} entries from {total_files} files under {self.root_dir}")

    def to_brief(self, n: int = 3) -> List[str]:
        """
        返回前 n 条论文的简单摘要，用于没有 search 的兜底情况。
        """
        briefs = []
        for e in self.entries[:n]:
            briefs.append(
                f"{e.get('title', '')} "
                f"({e.get('year', '')}, {e.get('venue', '')})\n"
                f"Abstract: {e.get('abstract', '')[:300]}..."
            )
        return briefs

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        最朴素的关键词检索：
        - 在 title + abstract 里用大小写不敏感的 substring 搜索
        - 命中次数多的优先
        """
        if not query:
            return self.to_brief(n=top_k)

        q = query.lower()
        scored: List[tuple[int, Dict[str, Any]]] = []

        for e in self.entries:
            text = (e.get("title", "") + " " + e.get("abstract", "")).lower()
            score = text.count(q)
            if score > 0:
                scored.append((score, e))

        # 如果一个都没匹配到，就退回 to_brief
        if not scored:
            return self.to_brief(n=top_k)

        # 按匹配次数从大到小排序
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[str] = []
        for score, e in scored[:top_k]:
            snippet = (
                f"{e.get('title', '')} "
                f"({e.get('year', '')}, {e.get('venue', '')})\n"
                f"[score: {score}] "
                f"Abstract: {e.get('abstract', '')[:400]}..."
            )
            results.append(snippet)

        return results

