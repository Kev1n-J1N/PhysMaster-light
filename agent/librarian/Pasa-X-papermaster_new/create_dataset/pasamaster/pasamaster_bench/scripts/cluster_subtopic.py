# -*- coding: utf-8 -*-
"""
Embedding-based phrase clustering + combination subtopics (comma-only)
- 只按逗号（英文 , 与中文 ，）分短语，保留原始短语用于输出命名
- 用 sentence-transformers 对“清洗短语”做嵌入，对短语进行 AHC（cosine, average）聚类
- 在每个短语簇内，枚举 1..N 个短语的组合，形成子topic，并输出所有满足支持度的子topic -> paper list
- 严格相关：论文必须同时包含该组合的所有短语才计入该子topic
- 不做任何簇合并；可通过阈值/上限控制规模，避免组合爆炸
"""

import json
import re
import itertools
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import time  # For measuring time
from tqdm import tqdm  # For progress bar

import numpy as np
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from sklearn.metrics import silhouette_score, calinski_harabasz_score
import sys,os
current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from tools.llm_call import llm_call


# ===================== 可调参数 =====================

DATA_PATH = "/data/duyuwen/Pasa-X/create_dataset/pasamaster/pasamaster_bench/output/20251009/cite_graphs.jsonl"

# 嵌入模型（小而快，够用）
EMB_MODEL_NAME = "/data/public_model/Qwen3-Embedding-8B"

# 只按逗号切分（英文 , 与中文 ，）；不按空格/分号/顿号等切分
COMMA_ONLY_REGEX = r"[,\uff0c]+"

# 关键词聚类的簇数范围（对“短语”聚类）
K_MIN, K_MAX = 4, 12

# 组合大小：从 1 到 MAX_COMB_SIZE（建议 2 或 3；更大组合会急剧增多）
MAX_COMB_SIZE = 3

# 每篇论文用于组合的“最多短语数”（按全局 df 排序截取），防止组合爆炸；None 表示不限制
TOP_PHRASES_PER_PAPER = 15

# 每个“短语簇”用于组合的“最多短语数”（按簇内 df 排序截取），防止组合爆炸；None 表示不限制
TOP_PHRASES_PER_CLUSTER = 30

# 每种组合大小（在每个簇内）的候选上限，防止爆炸；None 表示不限制
MAX_COMBINATIONS_PER_CLUSTER_SIZE = None  # e.g., 5000

# 最小支持度（至少多少篇论文同时包含该组合）
MIN_SUPPORT_PAPERS = 2

# 是否把“支持度=1 的组合”也输出（一般不建议；保持严格相关但会很多）
OUTPUT_SINGLETON_SUPPORT = False

# —— 高相似度聚类：complete linkage + 距离阈值切分 ——
USE_COMPLETE_DISTANCE_CUT = True

COMPLETE_MIN_SIM = 0.80      # 同簇任意两词最小余弦相似度阈值
MIN_CLUSTER_SIZE = 0         # 小簇阈值（低于这个视为碎簇）
REASSIGN_TINY_CLUSTERS = False  # 是否回填小簇到大簇
REASSIGN_MIN_SIM = 0.8      # 小簇回填到大簇的最低质心相似度



# ===================== 基础函数 =====================

def load_graph_first_line(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline().strip())
    return first["graph"]


def strip_outer_parens(s: str) -> str:
    s = s.strip()
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("（") and s.endswith("）")):
        return s[1:-1].strip()
    return s


def parse_phrases_comma_only(desc: str) -> List[Tuple[str, str]]:
    """
    只按逗号分短语；返回 [(原始短语, 清洗短语lower)]。
    - 保留原始短语（大小写/空格）用于输出命名
    - 清洗短语（lower + 去最外层括号）用于嵌入和匹配
    """
    parts = re.split(COMMA_ONLY_REGEX, desc)
    result = []
    for p in parts:
        original = p.strip()
        if not original:
            continue
        cleaned = strip_outer_parens(original).lower()
        if not cleaned:
            continue
        result.append((original, cleaned))
    return result


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(a @ b / (norm(a) * norm(b) + eps))



# FP-Growth减枝
def simple_fp_growth_mining(transactions, min_support, max_size):
    """
    简化版FP-Growth，专门为短语组合挖掘优化
    transactions: 列表的列表，每个内层列表是一个事务（论文的短语集合）
    min_support: 最小支持度
    max_size: 最大组合大小
    """
    # 确保所有事务项都是字符串
    cleaned_transactions = []
    for transaction in transactions:
        cleaned_transaction = []
        for item in transaction:
            # 确保每个项都是字符串，并且不为空
            if item is not None:
                # 强制转换为字符串，并去除前后空格
                str_item = str(item).strip()
                if str_item:  # 过滤掉空字符串
                    cleaned_transaction.append(str_item)
        if cleaned_transaction:  # 只添加非空事务
            cleaned_transactions.append(cleaned_transaction)

    transactions = cleaned_transactions
    if not transactions:
        return []

    # 计算单项频率
    item_freq = {}
    for transaction in transactions:
        for item in transaction:
            # 确保item是字符串
            if not isinstance(item, str):
                item = str(item)
            item_freq[item] = item_freq.get(item, 0) + 1

    # 过滤低频项
    frequent_items = {item for item, count in item_freq.items()
                      if count >= min_support}

    # 按频率降序排序（重要：FP-Growth的标准做法）
    sorted_frequent_items = sorted(frequent_items,
                                   key=lambda x: item_freq[x], reverse=True)

    # 存储所有发现的频繁模式
    all_patterns = []

    def mine_patterns(conditional_db, prefix, depth):
        """
        递归挖掘频繁模式
        conditional_db: 条件数据库
        prefix: 当前前缀模式
        depth: 当前深度（组合大小）
        """
        if depth > max_size:
            return

        # 计算当前条件数据库中的单项频率
        local_freq = {}
        for transaction in conditional_db:
            for item in transaction:
                # 确保item是字符串
                if not isinstance(item, str):
                    item = str(item)
                local_freq[item] = local_freq.get(item, 0) + 1

        # 过滤并排序
        local_frequent_items = [item for item in sorted_frequent_items
                                if item in local_freq and local_freq[item] >= min_support]

        for item in local_frequent_items:
            # 确保item是字符串
            if not isinstance(item, str):
                item = str(item)

            # 确保prefix中的所有元素都是字符串
            str_prefix = [str(p) for p in prefix]

            # 新模式 = 前缀 + 当前项
            new_pattern = str_prefix + [item]
            support = local_freq[item]

            # 记录模式
            all_patterns.append((new_pattern, support))

            # 构建新的条件数据库：只包含包含当前项的事务，并移除当前项
            new_conditional_db = []
            for transaction in conditional_db:
                if item in transaction:
                    # 创建新事务：移除当前项，并只保留在当前项之后出现的项（按频率排序）
                    item_index = transaction.index(item)
                    # 获取当前项在sorted_frequent_items中的位置
                    if item in sorted_frequent_items:
                        item_pos = sorted_frequent_items.index(item)
                        new_transaction = [i for i in transaction[item_index + 1:]
                                           if i in sorted_frequent_items[item_pos + 1:]]
                    else:
                        new_transaction = []
                    if new_transaction:
                        new_conditional_db.append(new_transaction)

            # 递归挖掘
            if new_conditional_db:
                # 确保new_pattern中的所有元素都是字符串
                str_new_pattern = [str(item) for item in new_pattern]
                mine_patterns(new_conditional_db, str_new_pattern, depth + 1)

    # 为每个频繁项构建初始条件数据库
    for i, item in enumerate(sorted_frequent_items):
        # 确保item是字符串
        if not isinstance(item, str):
            item = str(item)

        # 构建条件数据库：只包含包含当前项的事务
        conditional_db = []
        for transaction in transactions:
            if item in transaction:
                # 创建条件事务：只保留在当前项之后出现的频繁项
                item_index = transaction.index(item)
                # 获取当前项在sorted_frequent_items中的位置
                if item in sorted_frequent_items:
                    item_pos = sorted_frequent_items.index(item)
                    conditional_transaction = [i for i in transaction[item_index + 1:]
                                               if i in sorted_frequent_items[item_pos + 1:]]
                else:
                    conditional_transaction = []
                if conditional_transaction:
                    conditional_db.append(conditional_transaction)

        # 挖掘以当前项为后缀的模式
        if conditional_db:
            mine_patterns(conditional_db, [item], 2)

    # 添加单项频繁模式
    for item in sorted_frequent_items:
        if not isinstance(item, str):
            item = str(item)
        all_patterns.append(([item], item_freq[item]))

    return all_patterns


# 增加回退函数，防止FP减枝失败
def fallback_to_original_combinations(cluster_phrases, inv_index, paper_ids,
                                      cluster_info, display_phrase):
    """FP-Growth失败时的回退方案"""
    comb_count = 0
    valid_count = 0

    for size in range(1, MAX_COMB_SIZE + 1):
        candidates_iter = itertools.combinations(cluster_phrases, size)

        # 应用组合数量限制
        if MAX_COMBINATIONS_PER_CLUSTER_SIZE is not None:
            candidates_iter = itertools.islice(candidates_iter, MAX_COMBINATIONS_PER_CLUSTER_SIZE)

        for comb in candidates_iter:
            comb_count += 1
            if comb_count % 1000 == 0:
                print(f"    已处理 {comb_count} 个组合...")

            # 计算支持论文
            support_set = inv_index[comb[0]].copy()
            for w in comb[1:]:
                support_set &= inv_index[w]
                if not support_set:
                    break

            if support_set and len(support_set) >= MIN_SUPPORT_PAPERS:
                if len(support_set) == 1 and not OUTPUT_SINGLETON_SUPPORT:
                    continue

                key = ", ".join(display_phrase(w) for w in sorted(comb))
                cluster_info["subtopics"][key] = sorted(support_set)
                valid_count += 1

    print(f"  回退方法完成: {comb_count} 个组合，{valid_count} 个有效")


from sklearn.metrics.pairwise import cosine_similarity

def cluster_with_complete_threshold(X: np.ndarray,
                                    min_sim: float = COMPLETE_MIN_SIM,
                                    min_cluster_size: int = MIN_CLUSTER_SIZE,
                                    reassign_tiny: bool = REASSIGN_TINY_CLUSTERS,
                                    reassign_min_sim: float = REASSIGN_MIN_SIM) -> np.ndarray:
    """
    用 complete linkage + distance_threshold 做聚类，保证类内最小相似度 >= min_sim。
    对于过小簇，可选择回填到最近的大簇（相似度不足则保留为独立簇）。
    返回：labels（0..K-1）
    """
    dist_thresh = max(0.0, 1.0 - float(min_sim))  # 余弦距离阈值

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,                     # 用距离阈值切分，不固定 K
            distance_threshold=dist_thresh,
            linkage="complete",
            metric="cosine"
        )
    except TypeError:
        # 兼容旧版 sklearn
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            linkage="complete",
            affinity="cosine"
        )

    labels = clustering.fit_predict(X)
    unique_labs = np.unique(labels)

    if not reassign_tiny:
        return labels

    # 统计簇大小
    lab2idx = {lab: np.where(labels == lab)[0] for lab in unique_labs}
    big_labs = [lab for lab, idxs in lab2idx.items() if len(idxs) >= min_cluster_size]
    tiny_labs = [lab for lab, idxs in lab2idx.items() if len(idxs) <  min_cluster_size]

    if not tiny_labs or not big_labs:
        return labels

    # 计算“大簇”质心（单位向量）
    centroids = {}
    for lab in big_labs:
        vecs = X[lab2idx[lab]]
        c = vecs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids[lab] = c.reshape(1, -1)

    # 把每个“小簇”的成员回填到最近的大簇（若相似度达标）
    big_centroid_mat = np.vstack([centroids[lab] for lab in big_labs])  # [B, d]
    for tlab in tiny_labs:
        idxs = lab2idx[tlab]
        sims = cosine_similarity(X[idxs], big_centroid_mat)  # [m, B]
        best_big_idx = sims.argmax(axis=1)
        best_sims = sims[np.arange(len(idxs)), best_big_idx]
        for i, pid in enumerate(idxs):
            if best_sims[i] >= reassign_min_sim:
                labels[pid] = big_labs[best_big_idx[i]]
            # 否则保留原标签（小簇），确保不把明显离群点硬塞进大簇

    # 重新压缩标签（0..K-1）
    new_labs = np.unique(labels)
    remap = {lab: i for i, lab in enumerate(new_labs)}
    labels = np.array([remap[lab] for lab in labels], dtype=int)
    return labels


# ===================== 主流程 =====================

def build_subtopics_by_embedding_and_combinations() -> Dict[str, List[str]]:
    # 1) 读取 graph
    graph = load_graph_first_line(DATA_PATH)

    # 2) paper -> phrases（只按逗号分）
    paper2phrases_orig: Dict[str, List[str]] = defaultdict(list)
    paper2phrases_clean: Dict[str, List[str]] = defaultdict(list)
    paper_infos: Dict[str, List[str]] = defaultdict(list)

    # 记录 clean->orig 的“最常见显示形式”（用于子topic命名时用原始样式）
    clean2orig_counter: Dict[str, Counter] = defaultdict(Counter)

    for e in graph:
        to_id = e["to"]
        pairs = parse_phrases_comma_only(e["description"])
        # 同一条边去重
        seen = set()
        clean_str = ""
        for orig, clean in pairs:
            if clean in seen:
                continue
            seen.add(clean)
            paper2phrases_orig[to_id].append(orig)
            paper2phrases_clean[to_id].append(clean)
            clean2orig_counter[clean][orig] += 1
            clean_str += clean + ","
        paper_infos[to_id] = {
            "title": e["to_title"],
            "abstract": e["to_abstract"],
            "description": clean_str,
        }
    paper_ids = sorted(paper2phrases_clean.keys())
    if not paper_ids:
        return {}

    # 3) 全局 df（包含该短语的论文数），用于后续筛选 top 短语
    df_counter = Counter()
    for pid in paper_ids:
        df_counter.update(set(paper2phrases_clean[pid]))

    # 4) 每篇论文的短语集合（可限制数量以控爆）
    paper2phrase_set: Dict[str, Set[str]] = {}
    for pid in paper_ids:
        phrases = list(set(paper2phrases_clean[pid]))
        if TOP_PHRASES_PER_PAPER and len(phrases) > TOP_PHRASES_PER_PAPER:
            phrases.sort(key=lambda w: df_counter[w], reverse=True)
            phrases = phrases[:TOP_PHRASES_PER_PAPER]
        paper2phrase_set[pid] = set(phrases)
    # 5) 所有唯一短语（clean 版）+ 对应原始显示
    unique_cleans = sorted(df_counter.keys())
    if not unique_cleans:
        return {}

    # clean -> 代表性的原始短语（最高频写法）
    def display_phrase(clean: str) -> str:
        cnt = clean2orig_counter.get(clean, None)
        if cnt and len(cnt) > 0:
            return cnt.most_common(1)[0][0]
        return clean

    # 6) 短语嵌入 + 对短语进行 AHC 聚类（形成“短语簇”）
    model = SentenceTransformer(EMB_MODEL_NAME)
    phrase_emb = model.encode(unique_cleans, normalize_embeddings=True)  # [P, d]

    phrase_labels = cluster_with_complete_threshold(
        phrase_emb,
        min_sim=COMPLETE_MIN_SIM,
        min_cluster_size=MIN_CLUSTER_SIZE,
        reassign_tiny=REASSIGN_TINY_CLUSTERS,
        reassign_min_sim=REASSIGN_MIN_SIM
    )
    k = len(np.unique(phrase_labels))
    print(f"[complete-cut] auto K={k}, min_sim={COMPLETE_MIN_SIM}")

    clusters_result = []
    # label -> 该簇内的短语索引列表
    label2idxs: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(phrase_labels):
        label2idxs[lab].append(i)

    # 7) 在每个"短语簇"内使用简化版FP-Growth挖掘频繁组合
    result: Dict[str, List[str]] = {}

    for cluster_idx, (lab, idxs) in tqdm(enumerate(label2idxs.items())):

        # 存储当前簇的信息
        cluster_info = {
            "cluster_id": cluster_idx,
            "cluster_size": len(idxs),
            "phrases": [display_phrase(unique_cleans[i]) for i in idxs],
            "subtopics": {}
        }
        # 该簇内的短语（clean）
        cluster_phrases = [unique_cleans[i] for i in idxs]

        # 按"簇内df"排序，选TOP_PHRASES_PER_CLUSTER
        cluster_phrases.sort(key=lambda w: df_counter[w], reverse=True)
        transactions = []
        cluster_paper_list = []
        for pid in paper_ids:
            for p in paper2phrase_set[pid]:
                if p in cluster_phrases:
                    paper_info = {
                        "arxiv_id": pid,
                        "title": paper_infos[pid]["title"],
                        "description": paper_infos[pid]["description"],
                        "abstract": paper_infos[pid]["abstract"],
                    }
                    cluster_paper_list.append(paper_info)
                    break

        cluster_info = {
            "cluster_phrases": cluster_phrases,
            "cluster_paper_list": cluster_paper_list,
        }
        if len(cluster_paper_list) >= 5:
            clusters_result.append(cluster_info)
    # 8) 返回（键：短语组合，值：paper id 列表）
    return clusters_result

def generate_simple_subtopics(subtopic, paper_list):
    abstract_list = []
    for paper in paper_list:
        abstract_list.append(paper["abstract"])
    abstract_total = " ".join(abstract_list)
    MAX_LEN = 30000
    if len(abstract_total) > MAX_LEN:
        print(f"\033[33mwaring: abstract 过长，截断...\033[0m")
        abstract_list = abstract_list[:MAX_LEN/len(abstract_list)]
    for idx, abstract in enumerate(abstract_list):
        paper_list[idx]['abstract'] = abstract
    prompt = f"""You are an expert at writing natural, human-like literature search queries. 
Given the subtopic: {subtopic} and the related papers: {paper_list}, write ONE concise question that a researcher would type into a search bar. The question must be strongly relevant to EVERY paper listed.

Then, for EACH paper, explain why this question fits it in NO MORE THAN 200 words.

Return EXACTLY one JSON object in this format:
{{"question": "<the search-style question>",
  "reason": {{"<title1>": "<why it aligns>", "<title2>": "<why it aligns>", ...}}}}

Requirements:
- The question MUST be a literature-search query that, when entered into scholarly search engines (e.g., Google Scholar), would retrieve these specific papers; use shared, discriminative terminology they have in common (tasks, methods, datasets, problem settings), without naming titles/IDs/authors.
- Make the question general enough to cover the whole set.
- English only. Output nothing except the JSON object.
- The question doesn’t need to be complicated; it should capture the commonalities of these papers, and the main goal is to ensure each paper fully satisfies the question’s requirements.
"""
    response = llm_call(prompt,"deepseek-r1")
    json_begin = response.find("{")
    json_end = response.rfind("}")
    response = json.loads(response[json_begin:json_end+1])
    return response
    


if __name__ == "__main__":
    OUTPUT_PATH = "/data/duyuwen/Pasa-X/create_dataset/pasamaster/pasamaster_bench/output/"
    begin = time.time()
    subtopics = build_subtopics_by_embedding_and_combinations()
    end = time.time()
    print(f"聚类总耗时：{end - begin:.2f}s, 共聚类 {len(subtopics)} 个子主题")
    # 直接输出 JSON
    with open(f"{OUTPUT_PATH}/subtopics.json", "w", encoding="utf-8") as f:
        json.dump(subtopics, f, ensure_ascii=False, indent=2)
        
    question_list = []
    # 输出简化版的子主题
    begin = time.time()
    for subtopic in subtopics:
        topic_info = subtopic["cluster_phrases"]
        paper_list = subtopic["cluster_paper_list"]
        res = generate_simple_subtopics(topic_info, paper_list)
        answer = []
        res['answer'] = subtopic["cluster_paper_list"]
        question_list.append(res)
    end = time.time()
    print(f"生成简化版子主题总耗时：{end - begin:.2f}s")
    with open(f"{OUTPUT_PATH}/benchmark_questions.json", "w", encoding="utf-8") as f:
        json.dump(question_list, f, ensure_ascii=False, indent=2)