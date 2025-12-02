# -*- coding: utf-8 -*-
"""
加速版：Query 主题聚类（新增：LLM 学术子领域关键词提取）
- 小规模(N<=1500)：complete linkage + 相似度阈值
- 大规模(N> 1500)：MiniBatchKMeans（线性可扩展）
- 句向量支持 GPU/批量 & sqlite 缓存
- 新增：用 llm_call 为每条 query 提取简短“学术子领域关键词/短语”
  （默认开启 USE_LLM_KEYWORDS），再基于该短语做向量化与聚类，
  以更稳地按“子领域”聚合
"""

import json
import os
import sqlite3
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import random

import pandas as pd
from numpy.linalg import norm
import math
import time
import sys
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer, util
import torch

# ==== 你的工程路径与 llm ====
current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_path))))
from tools.llm_call import llm_call
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 可调参数 =====================

DEFAULT_EMB_MODEL = "/data/public_model/Qwen3-Embedding-8B"  # 你的 embedding 模型
LLM_MODEL_FOR_KW = "qwen-72b"                             # 关键词抽取用哪个模型
USE_LLM_KEYWORDS = True                                      # ✅ 开启：对每条 query 先抽取“子领域关键词/短语”
KW_BATCH = 64                                                # 关键词批处理（逐条也可，用这个做进度切片）

MIN_SIM = 0.6
K_MIN, K_MAX = 5, 1000

# 编码相关
BATCH_SIZE = 256               # 适当变大以吃满 GPU；CPU 可降到 64~128
CACHE_PATH = "./emb_cache.sqlite"  # 向量/关键词 缓存 sqlite
USE_CACHE = True

# 策略阈值
LARGE_N_SWITCH = 1500          # N 超过此值改用 MiniBatchKMeans
MBKMEANS_K = None              # 不指定则自动用 sqrt(N)//2 裁剪到 [K_MIN, K_MAX]

# ===================== 缓存实现（sqlite） =====================

def _ensure_cache(conn: sqlite3.Connection):
    # 向量缓存
    conn.execute("""
    CREATE TABLE IF NOT EXISTS emb_cache (
        key TEXT PRIMARY KEY,
        dim INTEGER NOT NULL,
        vec BLOB NOT NULL
    )""")
    # 关键词缓存（新增）
    conn.execute("""
    CREATE TABLE IF NOT EXISTS kw_cache (
        key TEXT PRIMARY KEY,
        kw  TEXT NOT NULL
    )""")
    conn.commit()

def _hash_key(model_name: str, text: str, prefix: str = "emb") -> str:
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8") + b'||' + model_name.encode('utf-8') + b'||' + text.encode('utf-8'))
    return h.hexdigest()

def _get_cached_embeddings(conn: sqlite3.Connection, model_name: str, texts: List[str]) -> Tuple[List[Optional[np.ndarray]], List[int]]:
    """返回与 texts 等长的列表；命中为 np.ndarray，未命中为 None；以及未命中索引列表"""
    res = [None] * len(texts)
    missing = []
    for i, t in enumerate(texts):
        k = _hash_key(model_name, t, prefix="emb")
        cur = conn.execute("SELECT dim, vec FROM emb_cache WHERE key=?", (k,))
        row = cur.fetchone()
        if row is None:
            missing.append(i)
        else:
            dim, blob = row
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == dim:
                res[i] = vec
            else:
                missing.append(i)
    return res, missing

def _put_cached_embeddings(conn: sqlite3.Connection, model_name: str, texts: List[str], vecs: np.ndarray, idxs: List[int]):
    if not idxs:
        return
    with conn:
        for j, i in enumerate(idxs):
            key = _hash_key(model_name, texts[i], prefix="emb")
            vec = vecs[j].astype(np.float32).tobytes()
            conn.execute("REPLACE INTO emb_cache (key, dim, vec) VALUES (?, ?, ?)", (key, vecs.shape[1], vec))

# ---- 关键词缓存（新增） ----
def _get_cached_keywords(conn: sqlite3.Connection, model_name: str, texts: List[str]) -> Tuple[List[Optional[str]], List[int]]:
    res = [None] * len(texts)
    missing = []
    for i, t in enumerate(texts):
        k = _hash_key(model_name, t, prefix="kw")
        cur = conn.execute("SELECT kw FROM kw_cache WHERE key=?", (k,))
        row = cur.fetchone()
        if row is None:
            missing.append(i)
        else:
            res[i] = row[0]
    return res, missing

def _put_cached_keywords(conn: sqlite3.Connection, model_name: str, texts: List[str], kws: List[str], idxs: List[int]):
    if not idxs:
        return
    with conn:
        for j, i in enumerate(idxs):
            key = _hash_key(model_name, texts[i], prefix="kw")
            conn.execute("REPLACE INTO kw_cache (key, kw) VALUES (?, ?)", (key, kws[j]))

# ===================== LLM 学术子领域关键词提取（新增） =====================

_KW_PROMPT = (
    "You are an academic domain expert. Given one user query, output a concise academic subfield keyword/phrase "
    "(ideally 1–5 words, lowercase, no extra text). If multiple are relevant, pick the most specific one.\n"
    "Query: {q}\n"
    "Output: "
)

def _normalize_kw(s: str) -> str:
    s = (s or "").strip()
    # 去掉可能的标签/引号/多余标点
    s = s.strip("<>\"' \t\r\n")
    # 只保留第一行
    s = s.splitlines()[0]
    # 保守清洗，逗号分隔时取前一个也可（你也可以保留整串）
    s = s.replace("；", ";").replace("，", ",")
    s = s.strip().strip(".;")
    # 统一小写
    return s.lower()

def _extract_domain_keywords_single(q: str, model_name: str = LLM_MODEL_FOR_KW) -> str:
    prompt = _KW_PROMPT.format(q=q)
    try:
        out = llm_call(prompt, model_name)
        return _normalize_kw(out)
    except Exception:
        # 失败回退：直接用原文本（尽量不阻断流程）
        return _normalize_kw(q)

def _extract_domain_keywords_batch(
    texts: List[str],
) -> List[str]:
    def _one(q: str) -> str:
        prompt = (
            "You are an academic expert. Given a query, assign it to ONE broad AI research area "
            "(not a specific subfield, keep it coarse). "
            "Output format: <topic> sub_topic </topic>\n\n"
            f"Query: {q}"
        )
        resp = llm_call(prompt, LLM_MODEL_FOR_KW)
        begin = resp.find("<topic>")
        end = resp.find("</topic>")
        kw = resp[begin+7:end] if (begin != -1 and end != -1 and end > begin) else resp
        kw = (kw or "").strip().lower()
        parts = kw.split()
        if len(parts) > 5:
            kw = " ".join(parts[:5])
        return kw

    if not texts:
        return []

    max_workers = 20
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, q): i for i, q in enumerate(texts)}
        out = [None] * len(texts)
        for fut in tqdm(as_completed(futures), total=len(futures)):
            i = futures[fut]
            out[i] = fut.result()
    return out
    
# ===================== 编码 & 聚类 =====================

def _embed_queries(queries: List[str], model_name: str, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """支持 GPU / 批量 / 缓存"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    if USE_CACHE:
        conn = sqlite3.connect(CACHE_PATH)
        _ensure_cache(conn)
        cached, missing_idx = _get_cached_embeddings(conn, model_name, queries)
    else:
        conn, cached, missing_idx = None, [None]*len(queries), list(range(len(queries)))

    # 批量编码未命中部分
    if missing_idx:
        to_encode = [queries[i] for i in missing_idx]
        new_vecs = model.encode(to_encode, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        # 回填到结果
        for j, i in enumerate(missing_idx):
            cached[i] = new_vecs[j]
        # 写缓存
        if USE_CACHE:
            _put_cached_embeddings(conn, model_name, queries, new_vecs, missing_idx)

    if USE_CACHE:
        conn.close()

    X = np.vstack(cached).astype(np.float32, copy=False)
    # 已经 normalize 了（normalize_embeddings=True），此处无需再归一化
    return X


def _complete_linkage_threshold(X: np.ndarray, min_sim: float) -> np.ndarray:
    dist_thresh = 1.0 - min_sim
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            linkage="average",
            metric="cosine",
            compute_full_tree="auto"
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            linkage="average",
            affinity="cosine"
        )
    return clustering.fit_predict(X)


def _labels_ok(labels: np.ndarray) -> bool:
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) <= 1: return False
    if np.all(counts == 1): return False
    return True


def _pick_k_by_silhouette(X: np.ndarray, k_min: int, k_max: int) -> Tuple[np.ndarray, int, float]:
    best_score, best_k, best_labels = -1.0, None, None
    n = len(X)
    if n <= 2:
        return np.zeros(n, dtype=int), 1, -1.0
    upper = min(k_max, n)
    for k in range(max(k_min, 2), max(3, upper + 1)):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        except TypeError:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        _, counts = np.unique(labels, return_counts=True)
        if np.any(counts < 2):
            continue
        try:
            score = silhouette_score(X, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    if best_labels is None:
        try:
            km = KMeans(n_clusters=2, n_init="auto", random_state=42)
        except TypeError:
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
        best_labels = km.fit_predict(X)
        best_k = 2
    return best_labels, best_k, best_score


def _topic_label_for_cluster(queries: List[str]) -> str:
    # 用更“关键词化”的输出格式，尽量得到短语主题
    prompt = (
        "You are an academic expert. I will provide you with several queries related to the same research field. "
        "Summarize the specific academic subfield (as concise keywords/phrase, 1–5 words, lowercase). "
        "Output format: <topic> specific academic subfield </topic>\n\n"
        f"Queries: {queries}"
    )
    response = llm_call(prompt, LLM_MODEL_FOR_KW)
    begin = response.find("<topic>")
    end = response.find("</topic>")
    res = response[begin+7:end] if (begin != -1 and end != -1 and end > begin) else response
    return (res or "").strip()


def _ensure_unique_keys(kv: Dict[str, List[str]]) -> Dict[str, List[str]]:
    seen, out = {}, {}
    for k, v in kv.items():
        if k not in seen:
            seen[k] = 1; out[k] = v
        else:
            seen[k] += 1
            new_k = f"{k} (#{seen[k]})"
            while new_k in out:
                seen[k] += 1
                new_k = f"{k} (#{seen[k]})"
            out[new_k] = v
    return out


# ===================== 对外接口 =====================

def cluster_queries_to_json(
    query_list: List[str],
    emb_model: str = DEFAULT_EMB_MODEL,
    min_sim: float = MIN_SIM,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, List[str]]:

    # 1) 预处理（去空、去重但保留原顺序）
    seen = set()
    raw = []
    for q in query_list:
        if not isinstance(q, str): continue
        s = q.strip()
        if not s: continue
        if s in seen:  # 纯重复无需重复编码
            continue
        seen.add(s)
        raw.append(s)
    if not raw:
        return {}

    # 1.5) ✅ 先用 LLM 抽取“学术子领域关键词/短语”，再用于嵌入与聚类
    print(f"Extracting domain keywords for {len(raw)} queries...")
    begin = time.time()
    texts_for_embedding = _extract_domain_keywords_batch(raw) if USE_LLM_KEYWORDS else raw
    end = time.time()
    print(f"domain keyword extraction time: {end-begin:.3f}s")
    # 2) 编码（GPU/缓存/批量）
    print(f"Embedding {len(texts_for_embedding)} queries...")
    begin = time.time()
    X = _embed_queries(texts_for_embedding, emb_model, batch_size=batch_size)
    end = time.time()
    print(f"embedding time: {end-begin:.3f}s")

    begin = time.time()
    # 3) 聚类策略
    N = len(raw)
    if N > LARGE_N_SWITCH:
        # ====== 大规模：MiniBatchKMeans ======
        if MBKMEANS_K is None:
            k_guess = int(np.clip(int(np.sqrt(N) // 2), K_MIN, K_MAX))
        else:
            k_guess = MBKMEANS_K
        mbk = MiniBatchKMeans(
            n_clusters=k_guess,
            batch_size=max(1024, batch_size),
            n_init=3,
            random_state=42,
            reassignment_ratio=0.01
        )
        labels = mbk.fit_predict(X)
    else:
        # ====== 小规模：complete linkage + 阈值 ======
        labels = _complete_linkage_threshold(X, min_sim)
        if not _labels_ok(labels):
            labels, _, _ = _pick_k_by_silhouette(X, k_min, k_max)
    end_time = time.time()
    print(f"clustering time: {end_time-begin:.3f}s")

    begin = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    topic2queries: Dict[str, List[str]] = {}
    sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))

    def process_cluster(kv):
        """对单个簇执行 LLM 主题总结"""
        _, idxs = kv
        queries = [raw[i] for i in idxs]
        try:
            topic = _topic_label_for_cluster(queries)
        except Exception as e:
            topic = f"error_topic_{_}"  
        return topic, queries

    MAX_WORKERS = 20 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_cluster, kv) for kv in sorted_clusters]
        desc = f"Topic labeling (workers={MAX_WORKERS})"
        with tqdm(total=len(futures), desc=desc, ncols=80) as pbar:
            for fut in as_completed(futures):
                topic, qs = fut.result()
                topic2queries[topic] = qs
                pbar.update(1)

    end = time.time()
    print(f"topic summarization time: {end-begin:.3f}s")
    return _ensure_unique_keys(topic2queries)


# ===================== 示例 =====================

if __name__ == "__main__":
    N_SAMPLES_PER_BUCKET = 20

    current_path = os.path.abspath(__file__)

    file_path = f"{os.path.dirname(current_path)}/../data/ai.csv"
    out_dir   = f"{os.path.dirname(current_path)}/../result"
    save_path = os.path.join(out_dir, "ai_cluster.json")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(file_path)

    # 更健壮地读取 query type，避免 math.isnan 在字符串上报错
    def _is_nan(v) -> bool:
        if v is None:
            return True
        if isinstance(v, float):
            try:
                return math.isnan(v)
            except Exception:
                return False
        if isinstance(v, str):
            return v.strip().lower() in {"", "nan", "none"}
        return False

    if 'query' not in df.columns:
        raise ValueError("CSV 必须包含 'query' 列")

    query_types = df['query type'].tolist() if 'query type' in df.columns else [1]*len(df)
    queries     = df['query'].tolist()

    # 三个桶：同时记录“query 字符串列表”和“原 CSV 行号列表”（用于回溯）
    buckets_queries: List[List[str]] = [[], [], []]
    buckets_indices: List[List[int]] = [[], [], []]

    prev_valid = 1
    for i in range(len(query_types)):
        qt = query_types[i]
        if _is_nan(qt):
            cur = prev_valid
        else:
            try:
                cur = int(qt)
            except (TypeError, ValueError):
                cur = prev_valid
        cur = max(1, min(3, cur))
        prev_valid = cur

        q = queries[i]
        if isinstance(q, str) and q.strip():
            buckets_queries[cur-1].append(q.strip())
            buckets_indices[cur-1].append(i)

    # 统计：筛选前（有效 query 总数）
    total_before = sum(len(bq) for bq in buckets_queries)

    # 汇总所有 bucket 的代表行
    sampled_rows: List[dict] = []

    # 逐 bucket 聚类 + 采样代表行 + 写各自 JSON
    all_result_json = {}
    for bidx, (qs, idxs) in enumerate(zip(buckets_queries, buckets_indices), 1):
        if not qs:
            continue

        print(f"\n=== Bucket #{bidx} | N={len(qs)} ===")
        # 你项目里的聚类函数：返回 {topic: [query, ...]}
        result = cluster_queries_to_json(qs, emb_model=DEFAULT_EMB_MODEL, min_sim=0.80)
        all_result_json[f"bucket_{bidx}"] = result

        # 为该 bucket 写 JSON
        save_i = os.path.join(out_dir, f"ai_cluster.bucket{bidx}.json")
        with open(save_i, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[info] 写入聚类结果: {save_i}")

        # —— 构造“query -> 原CSV行号列表”的映射（仅在该 bucket 内）
        q2rows: Dict[str, List[int]] = {}
        for i_csv in idxs:
            qstr = str(df.at[i_csv, 'query']).strip()
            q2rows.setdefault(qstr, []).append(i_csv)

       # ========== 每簇随机抽取，若有富余名额则随机分配 ==========
        topics = list(result.keys())
        num_clusters = len(topics)
        if num_clusters == 0:
            continue

        # 平均分配 + 余数随机分配
        base, rem = divmod(N_SAMPLES_PER_BUCKET, num_clusters)
        per_cluster_counts = [base] * num_clusters
        if rem > 0:
            for i_rand in random.sample(range(num_clusters), k=rem):
                per_cluster_counts[i_rand] += 1

        used_row_indices: set = set()
        sampled_this_bucket = 0

        # 预先统计每个簇的可用行
        cluster_available = []
        for topic in topics:
            qlist = result.get(topic, []) or []
            avail = []
            for qstr in set(qlist):
                for ridx in q2rows.get(qstr, []):
                    if ridx not in used_row_indices:
                        avail.append((qstr, ridx))
            cluster_available.append(avail)

        # 第一轮采样：尽量取满，记录未使用名额
        unused_quota = 0
        topic_sampled_count = {t: 0 for t in topics}  # ✅ 记录每个topic实际取样数量

        for t_idx, topic in enumerate(topics):
            avail = cluster_available[t_idx]
            qlist = result.get(topic, []) or []
            if not avail:
                unused_quota += per_cluster_counts[t_idx]
                continue

            k_take = min(per_cluster_counts[t_idx], len(avail))
            if k_take < per_cluster_counts[t_idx]:
                unused_quota += (per_cluster_counts[t_idx] - k_take)

            random.shuffle(avail)
            selected = avail[:k_take]

            for qstr, ridx in selected:
                used_row_indices.add(ridx)
                row_dict = df.loc[ridx].to_dict()
                row_dict["cluster_topic"] = topic
                row_dict["cluster_size"] = len(qlist)
                row_dict["cluster_bucket"] = bidx
                row_dict["cluster_representative_query"] = qstr
                sampled_rows.append(row_dict)
                sampled_this_bucket += 1
                topic_sampled_count[topic] += 1  # ✅ 计数

        # 第二轮：把回收的名额随机发给仍有剩余的簇
        if unused_quota > 0:
            big_clusters = []
            for t_idx, topic in enumerate(topics):
                avail = [pair for pair in cluster_available[t_idx] if pair[1] not in used_row_indices]
                if len(avail) > 0:
                    big_clusters.append((t_idx, avail))

            random.shuffle(big_clusters)
            for _ in range(unused_quota):
                if not big_clusters:
                    break
                t_idx, avail = random.choice(big_clusters)
                if not avail:
                    big_clusters.remove((t_idx, avail))
                    continue
                qstr, ridx = avail.pop(random.randrange(len(avail)))
                used_row_indices.add(ridx)
                qlist = result.get(topics[t_idx], []) or []
                row_dict = df.loc[ridx].to_dict()
                row_dict["cluster_topic"] = topics[t_idx]
                row_dict["cluster_size"] = len(qlist)
                row_dict["cluster_bucket"] = bidx
                row_dict["cluster_representative_query"] = qstr
                sampled_rows.append(row_dict)
                sampled_this_bucket += 1
                topic_sampled_count[topics[t_idx]] += 1  # ✅ 第二轮也统计
        # ========== 结束采样 ==========

        # ✅ 输出每个 topic 实际采样数量
        print("\n[采样分布]")
        for topic, cnt in sorted(topic_sampled_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {topic[:60]:<60} : {cnt} 条")

    # —— 汇总 JSON（可选）
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_result_json, f, ensure_ascii=False, indent=2)
    print(f"\n[info] 合并聚类 JSON 写入: {save_path}")

    # —— 保存“筛选后 CSV”
    filtered_csv_path = os.path.join(out_dir, "ai_cluster.filtered.csv")
    if sampled_rows:
        filtered_df = pd.DataFrame(sampled_rows)
        # 让输出列更友好（把聚类信息放前面）
        front_cols = ["cluster_bucket", "cluster_topic", "cluster_size", "cluster_representative_query"]
        other_cols = [c for c in filtered_df.columns if c not in front_cols]
        filtered_df = filtered_df[front_cols + other_cols]
        filtered_df.to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")
        print(f"[info] 筛选后的 CSV 写入: {filtered_csv_path}")
    else:
        print("[warn] 未采样到任何代表行，未生成筛选后 CSV。")

    # —— 输出筛选前/后的数目
    total_after = len(sampled_rows)  # 采样后条目数（≈ 每 bucket 实际采样数之和）
    print(f"\n=== 压缩统计 ===")
    print(f"总计：before={total_before}  after={total_after}  压缩率={(1 - (total_after / max(1, total_before))) * 100:.2f}%")