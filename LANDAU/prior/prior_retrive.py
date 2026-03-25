from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


CURRENT_DIR = Path(__file__).resolve().parent

CFG = {
    "dirs": {
        "knowledge": CURRENT_DIR / "knowledge",
        "index": CURRENT_DIR / "index",
    },
    "embedding": {"model": "BAAI/bge-small-en-v1.5"},
    "dense_search": {"candidate_k": 24},
    "sparse_search": {"candidate_k": 24, "k1": 1.5, "b": 0.75},
    "hybrid": {"dense_weight": 0.65, "sparse_weight": 0.35},
    "rerank": {
        "dense_weight": 0.45,
        "sparse_weight": 0.2,
        "overlap_weight": 0.2,
        "keyword_weight": 0.1,
        "phrase_weight": 0.05,
    },
}

QUERY_ALIASES = {
    "lamet": ["large momentum effective theory", "large-momentum effective theory"],
    "qcd": ["quantum chromodynamics"],
    "pdf": ["parton distribution function"],
    "tmd": ["transverse momentum dependent"],
    "pz": ["p_z", "hadron momentum"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", (text or "").lower())


class PriorRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] 正在加载检索模型 ({self.device})...")
        self.model = SentenceTransformer(CFG["embedding"]["model"], device=self.device)

        self.index_dir = Path(CFG["dirs"]["index"])
        self.knowledge_dir = Path(CFG["dirs"]["knowledge"])
        self.faiss_path = self.index_dir / "index.faiss"
        self.id_map_path = self.index_dir / "id_map.jsonl"
        self.meta_path = self.index_dir / "index_meta.json"
        self.knowledge_path = self.knowledge_dir / "chunks.jsonl"

        self.knowledge_base = self._load_knowledge()
        self.id_map = self._load_id_map()
        self.index = self._load_or_rebuild_index()

        self._token_cache: Dict[str, List[str]] = {
            chunk_id: _tokenize(data.get("text", ""))
            for chunk_id, data in self.knowledge_base.items()
        }
        self._build_sparse_stats()

    def _load_knowledge(self) -> Dict[str, Dict[str, Any]]:
        kb: Dict[str, Dict[str, Any]] = {}
        with self.knowledge_path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                kb[data["chunk_id"]] = data
        return kb

    def _load_id_map(self) -> Dict[int, str]:
        if not self.id_map_path.exists():
            return {}
        id_map: Dict[int, str] = {}
        with self.id_map_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                id_map[int(item["index_id"])] = item["chunk_id"]
        return id_map

    def _load_index_meta(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}

    def _load_or_rebuild_index(self):
        meta = self._load_index_meta()
        needs_rebuild = (
            not self.faiss_path.exists()
            or not self.id_map
            or meta.get("index_type") != "IndexFlatIP"
            or meta.get("embedding_model") != CFG["embedding"]["model"]
            or meta.get("num_vectors") != len(self.knowledge_base)
            or len(self.id_map) != len(self.knowledge_base)
        )
        if needs_rebuild:
            return self._rebuild_dense_index()
        index = faiss.read_index(str(self.faiss_path))
        if index.ntotal != len(self.knowledge_base):
            return self._rebuild_dense_index()
        return index

    def _rebuild_dense_index(self):
        texts = [self.knowledge_base[cid]["text"] for cid in self.knowledge_base]
        chunk_ids = list(self.knowledge_base.keys())
        embs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embs = np.asarray(embs, dtype="float32")
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.faiss_path))
        with self.id_map_path.open("w", encoding="utf-8") as f:
            for idx, chunk_id in enumerate(chunk_ids):
                f.write(json.dumps({"chunk_id": chunk_id, "index_id": idx}, ensure_ascii=False) + "\n")
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_type": "IndexFlatIP",
                    "embedding_model": CFG["embedding"]["model"],
                    "normalized_embeddings": True,
                    "num_vectors": len(chunk_ids),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.id_map = {idx: chunk_id for idx, chunk_id in enumerate(chunk_ids)}
        return index

    def _build_sparse_stats(self):
        self.doc_len: Dict[str, int] = {}
        self.doc_freq: Dict[str, int] = {}
        total_len = 0
        for chunk_id, tokens in self._token_cache.items():
            token_set = set(tokens)
            self.doc_len[chunk_id] = len(tokens)
            total_len += len(tokens)
            for token in token_set:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.num_docs = max(1, len(self._token_cache))
        self.avg_doc_len = total_len / self.num_docs if self.num_docs else 1.0

    def _rewrite_query(self, query: str) -> List[str]:
        tokens = _tokenize(query)
        variants = [query.strip()]
        expanded_terms: List[str] = []
        for token in tokens:
            expanded_terms.append(token)
            expanded_terms.extend(QUERY_ALIASES.get(token, []))
        if expanded_terms:
            variants.append(" ".join(dict.fromkeys(expanded_terms)))
        normalized = query.replace("P_z", "Pz").replace("PZ", "Pz")
        if normalized not in variants:
            variants.append(normalized)
        return [variant for variant in dict.fromkeys(v for v in variants if v.strip())]

    def _dense_search(self, query_variants: List[str], candidate_k: int) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for variant in query_variants:
            q_emb = self.model.encode([variant], normalize_embeddings=True).astype("float32")
            similarities, indices = self.index.search(q_emb, candidate_k)
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1:
                    continue
                chunk_id = self.id_map.get(int(idx))
                if not chunk_id or chunk_id not in self.knowledge_base:
                    continue
                scores[chunk_id] = max(scores.get(chunk_id, -1.0), float(sim))
        return scores

    def _bm25_score(self, query_tokens: List[str], chunk_id: str) -> float:
        tokens = self._token_cache.get(chunk_id, [])
        if not tokens:
            return 0.0
        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        score = 0.0
        k1 = CFG["sparse_search"]["k1"]
        b = CFG["sparse_search"]["b"]
        doc_len = self.doc_len.get(chunk_id, 0)
        norm = k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1e-6)))
        for token in query_tokens:
            if token not in freq:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            tf = freq[token]
            score += idf * (tf * (k1 + 1)) / (tf + norm)
        return score

    def _sparse_search(self, query_variants: List[str], candidate_k: int) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for variant in query_variants:
            query_tokens = _tokenize(variant)
            for chunk_id in self.knowledge_base:
                score = self._bm25_score(query_tokens, chunk_id)
                if score <= 0:
                    continue
                scores[chunk_id] = max(scores.get(chunk_id, 0.0), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k]
        return dict(ranked)

    def _apply_filters(
        self,
        chunk_ids: List[str],
        source_ids: List[str] | None = None,
        chapter: str | None = None,
        section_prefix: str | None = None,
        keywords: List[str] | None = None,
    ) -> List[str]:
        filtered: List[str] = []
        keyword_tokens = {k.lower() for k in (keywords or [])}
        source_set = set(source_ids or [])
        for chunk_id in chunk_ids:
            item = self.knowledge_base.get(chunk_id, {})
            if source_set and item.get("source", {}).get("source_id") not in source_set:
                continue
            locator = item.get("locator", {})
            if chapter is not None and str(locator.get("chapter")) != str(chapter):
                continue
            if section_prefix is not None and not str(locator.get("section", "")).startswith(str(section_prefix)):
                continue
            if keyword_tokens:
                haystack = {kw.lower() for kw in item.get("keywords", [])}
                haystack.update(_tokenize(item.get("text", "")))
                if not keyword_tokens.intersection(haystack):
                    continue
            filtered.append(chunk_id)
        return filtered

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        min_score = min(scores.values())
        if math.isclose(max_score, min_score):
            return {key: 1.0 for key in scores}
        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }

    def _rerank(self, query: str, chunk_ids: List[str], dense_scores: Dict[str, float], sparse_scores: Dict[str, float]) -> List[str]:
        query_tokens = set(_tokenize(query))
        query_text = query.lower()
        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)
        ranked: List[tuple[str, float]] = []
        for chunk_id in chunk_ids:
            item = self.knowledge_base[chunk_id]
            text = item.get("text", "").lower()
            chunk_tokens = set(self._token_cache.get(chunk_id, []))
            overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            keyword_overlap = len(query_tokens & {kw.lower() for kw in item.get("keywords", [])}) / max(len(query_tokens), 1)
            phrase_bonus = 1.0 if any(phrase in text for phrase in [query_text, query_text.replace("_", " ")]) else 0.0
            score = (
                CFG["rerank"]["dense_weight"] * norm_dense.get(chunk_id, 0.0)
                + CFG["rerank"]["sparse_weight"] * norm_sparse.get(chunk_id, 0.0)
                + CFG["rerank"]["overlap_weight"] * overlap
                + CFG["rerank"]["keyword_weight"] * keyword_overlap
                + CFG["rerank"]["phrase_weight"] * phrase_bonus
            )
            ranked.append((chunk_id, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        expand_context: bool = False,
        source_ids: List[str] | None = None,
        chapter: str | None = None,
        section_prefix: str | None = None,
        keywords: List[str] | None = None,
        rewrite_query: bool = True,
    ) -> List[Dict[str, Any]]:
        query_variants = self._rewrite_query(query) if rewrite_query else [query]
        dense_scores = self._dense_search(query_variants, CFG["dense_search"]["candidate_k"])
        sparse_scores = self._sparse_search(query_variants, CFG["sparse_search"]["candidate_k"])

        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)
        combined: Dict[str, float] = {}
        for chunk_id in set(norm_dense) | set(norm_sparse):
            combined[chunk_id] = (
                CFG["hybrid"]["dense_weight"] * norm_dense.get(chunk_id, 0.0)
                + CFG["hybrid"]["sparse_weight"] * norm_sparse.get(chunk_id, 0.0)
            )

        candidate_ids = [chunk_id for chunk_id, _ in sorted(combined.items(), key=lambda item: item[1], reverse=True)]
        candidate_ids = self._apply_filters(
            candidate_ids,
            source_ids=source_ids,
            chapter=chapter,
            section_prefix=section_prefix,
            keywords=keywords,
        )
        reranked_ids = self._rerank(query, candidate_ids, dense_scores, sparse_scores)[:top_k]

        results: List[Dict[str, Any]] = []
        for chunk_id in reranked_ids:
            chunk_data = self.knowledge_base[chunk_id]
            res_item: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "score": round(combined.get(chunk_id, 0.0), 4),
                "dense_score": round(dense_scores.get(chunk_id, 0.0), 4),
                "sparse_score": round(sparse_scores.get(chunk_id, 0.0), 4),
                "text": chunk_data["text"],
                "citation": chunk_data["citation"],
                "locator": chunk_data["locator"],
                "source": chunk_data["source"],
                "keywords": chunk_data["keywords"],
            }
            if expand_context:
                res_item["context_prev"] = self.knowledge_base.get(chunk_data["prev_chunk_id"], {}).get("text", "")
                res_item["context_next"] = self.knowledge_base.get(chunk_data["next_chunk_id"], {}).get("text", "")
            results.append(res_item)
        return results

    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        blocks: List[str] = []
        for i, item in enumerate(results, 1):
            source = item.get("source", {}) or {}
            locator = item.get("locator", {}) or {}
            block_lines = [
                f"[Retrieved Source {i}]",
                f"score={item.get('score', '')} dense={item.get('dense_score', '')} sparse={item.get('sparse_score', '')}",
                f"title={source.get('title', '')}",
                f"source_id={source.get('source_id', '')}",
                f"citation={item.get('citation', '')}",
                f"locator=chapter:{locator.get('chapter', '')} section:{locator.get('section', '')} page:{locator.get('page_start', '')}",
                f"keywords={', '.join(item.get('keywords', []))}",
                f"text={item.get('text', '')}",
            ]
            if item.get("context_prev"):
                block_lines.append(f"context_prev={item['context_prev']}")
            if item.get("context_next"):
                block_lines.append(f"context_next={item['context_next']}")
            blocks.append("\n".join(block_lines))
        return "\n\n".join(blocks)
