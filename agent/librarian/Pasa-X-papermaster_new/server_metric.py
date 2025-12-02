import math
from typing import List, Dict, Tuple

def keep_letters(s: str) -> str:
    """与原脚本一致：只保留英文字母并小写，弱化大小写/标点/空格影响。"""
    letters = [c for c in s if c.isalpha()]
    return ''.join(letters).lower()

def _dedup_preserve_order(items: List[str]) -> List[str]:
    """按顺序去重。"""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def _dcg_at_k(rels: List[int], k: int) -> float:
    """DCG（log2(i+2) 分母，与原脚本一致）"""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

def _ndcg_at_k(ranked_norm: List[str], gt_norm_set: set, k: int) -> float:
    hits = [1 if t in gt_norm_set else 0 for t in ranked_norm[:k]]
    dcg = _dcg_at_k(hits, k)
    m = min(k, len(gt_norm_set))
    ideal = [1]*m + [0]*max(0, k - m)
    idcg = _dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

def evaluate_query_topk(query: str,
                        gt: List[str],
                        searched_paper: List[str],
                        ks: Tuple[int, ...] = (5, 10, 20)
                       ) -> Dict[str, Dict[int, float]]:
    """
    输入：
      - query: 查询（用于回显）
      - gt: 标答标题列表（字符串）
      - searched_paper: 检索结果标题列表（已按相关度降序）
      - ks: 需要计算的 K（默认 5/10/20）

    返回：
      {
        "query": query,
        "precision": {5: ..., 10: ..., 20: ...},
        "ndcg": {5: ..., 10: ..., 20: ...}
      }
    """
    # 归一化
    gt_norm = {keep_letters(t) for t in gt if isinstance(t, str) and t.strip()}
    ranked_norm = _dedup_preserve_order(
        [keep_letters(t) for t in searched_paper if isinstance(t, str) and t.strip()]
    )

    precision, ndcg = {}, {}
    for k in ks:
        topk = ranked_norm[:k]
        tp = sum(1 for t in topk if t in gt_norm)
        precision[k] = tp / float(k)                 # 分母固定为 K
        ndcg[k] = _ndcg_at_k(ranked_norm, gt_norm, k)

    return {"query": query, "precision": precision, "ndcg": ndcg}

if __name__ == '__main__':
    q = "graph contrastive learning"
    gt = ["When Less is More: Investigating Data Pruning for Pretraining LLMs at   Scale",
      "How to Train Data-Efficient LLMs",
      "Deduplicating Training Data Makes Language Models Better",
      "AlpaGasus: Training A Better Alpaca with Fewer Data",
      "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes",
      "LESS: Selecting Influential Data for Targeted Instruction Tuning",
      "Automatic Document Selection for Efficient Encoder Pretraining",
      "Farewell to aimless large-scale pretraining: Influential subset selection for language model",
      "Babyllama-2: Ensemble-distilled models consistently outperform teachers with limited data."]
    searched = [
        "How to Train Data-Efficient LLMs",
      "When Less is More: Investigating Data Pruning for Pretraining LLMs at\n  Scale",
      "Automatic Document Selection for Efficient Encoder Pretraining",
      "Perplexed by Perplexity: Perplexity-Based Data Pruning With Small\n  Reference Models",
      "LESS: Selecting Influential Data for Targeted Instruction Tuning",
      "Efficient Continual Pre-training by Mitigating the Stability Gap",
      "An Empirical Exploration in Quality Filtering of Text Data",
      "Scaling Laws for Neural Language Models",
      "Rephrasing the Web: A Recipe for Compute and Data-Efficient Language\n  Modeling",
      "Deduplicating Training Data Makes Language Models Better",
      "A Survey on Data Selection for Language Models",
      "Deep Double Descent: Where Bigger Models and More Data Hurt",
      "Brevity is the soul of wit: Pruning long files for code generation",
      "D4: Improving LLM Pretraining via Document De-Duplication and\n  Diversification",
      "From Quantity to Quality: Boosting LLM Performance with Self-Guided Data\n  Selection for Instruction Tuning",
      "Efficient Continual Pre-training for Building Domain Specific Large\n  Language Models",
      "The FineWeb Datasets: Decanting the Web for the Finest Text Data at\n  Scale",
      "SemDeDup: Data-efficient learning at web-scale through semantic\n  deduplication",
      "Your Vision-Language Model Itself Is a Strong Filter: Towards\n  High-Quality Instruction Tuning with Data Selection",
      "Data Selection for Language Models via Importance Resampling",
      "Too Large; Data Reduction for Vision-Language Pre-Training",
      "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora\n  with Web Data, and Web Data Only",
      "Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs",
      "Data pruning and neural scaling laws: fundamental limitations of\n  score-based algorithms",
      "Measuring Sample Importance in Data Pruning for Language Models based on\n  Information Entropy",
      "Data-Efficient Contrastive Language-Image Pretraining: Prioritizing Data\n  Quality over Quantity",
      "Scaling (Down) CLIP: A Comprehensive Analysis of Data, Architecture, and\n  Training Strategies",
      "CodeACT: Code Adaptive Compute-efficient Tuning Framework for Code LLMs",
      "Insights into Pre-training via Simpler Synthetic Tasks",
      "Pre-train or Annotate? Domain Adaptation with a Constrained Budget",
      "Superfiltering: Weak-to-Strong Data Filtering for Fast\n  Instruction-Tuning",
      "INGENIOUS: Using Informative Data Subsets for Efficient Pre-Training of\n  Language Models",
      "JetMoE: Reaching Llama2 Performance with 0.1M Dollars"
    ]
    print(evaluate_query_topk(q, gt, searched))
