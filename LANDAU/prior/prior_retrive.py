import os
import json
import faiss
import numpy as np
import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# ===================== 核心配置 =====================
CFG = {
    "dirs": {
        "knowledge": "./prior/knowledge", 
        "index": "./prior/index",
    },
    "embedding": {"model": "all-MiniLM-L6-v2"},
}

class PriorRetriever:
    def __init__(self):
        # 1. 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. 加载 Embedding 模型
        print(f"[*] 正在加载检索模型 ({self.device})...")
        self.model = SentenceTransformer(CFG["embedding"]["model"], device=self.device)
        
        # 3. 加载 FAISS 索引
        self.faiss_path = os.path.join(CFG["dirs"]["index"], "index.faiss")
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"FAISS file not found {self.faiss_path}. Please build the index first.")
        self.index = faiss.read_index(self.faiss_path)
        
        # 4. 加载 ID 映射表
        self.id_map = self._load_id_map()
        
        # 5. 加载全文知识库
        self.knowledge_base = self._load_knowledge()

    def _load_id_map(self) -> Dict[int, str]:
        path = os.path.join(CFG["dirs"]["index"], "id_map.jsonl")
        id_map = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                id_map[item["index_id"]] = item["chunk_id"]
        return id_map

    def _load_knowledge(self) -> Dict[str, Dict]:
        path = os.path.join(CFG["dirs"]["knowledge"], "chunks.jsonl")
        kb = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                kb[data["chunk_id"]] = data
        return kb

    def retrieve(self, query: str, top_k: int = 3, expand_context: bool = False) -> List[Dict]:
        """
        核心检索函数
        :param query: 用户的物理问题
        :param top_k: 返回最相关的块数量
        :param expand_context: 是否开启上下文扩展（拉取前后的 chunk）
        """
        # 1. 向量化查询
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        
        # 2. 在 FAISS 中搜索
        # distances 是 L2 距离，索引数值越小越相似
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue # 过滤无效结果
            
            chunk_id = self.id_map.get(idx)
            if not chunk_id or chunk_id not in self.knowledge_base:
                continue
                
            chunk_data = self.knowledge_base[chunk_id]
            
            # 3. 构造返回结构 (符合 RAG 需求)
            res_item = {
                "chunk_id": chunk_id,
                "score": round(float(1 / (1 + dist)), 4), # 将距离转化为 0-1 的相似度评分
                "text": chunk_data["text"],
                "citation": chunk_data["citation"],
                "locator": chunk_data["locator"],
                "source": chunk_data["source"],
                "keywords": chunk_data["keywords"]
            }
            
            # 4. 可选：上下文扩展 (Context Expansion)
            # 如果物理公式需要前后文解释，这个功能非常有用
            if expand_context:
                res_item["context_prev"] = self.knowledge_base.get(chunk_data["prev_chunk_id"], {}).get("text", "")
                res_item["context_next"] = self.knowledge_base.get(chunk_data["next_chunk_id"], {}).get("text", "")
                
            results.append(res_item)
            
        return results

    def format_for_llm(self, results: List[Dict]) -> str:
        """
        将检索结果转化为可以直接喂给 LLM 的 Prompt 背景材料
        """
        prompt_texts = []
        for i, r in enumerate(results, 1):
            source_info = f"Source {i}: {r['citation']}"
            content = r['text']
            # 如果有上下文扩展，可以拼接在这里
            prompt_texts.append(f"[{source_info}]\nContent: {content}")
            
        return "\n\n".join(prompt_texts)
