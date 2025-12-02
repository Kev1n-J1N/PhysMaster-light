import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

training_data_path = "sft_training_data/result_APASbech_comp_train_20250905_r1.jsonl"
gt_data_path = "output/APASbench_comp_splits/APASbech_comp_train_converted.jsonl"

def get_sft_qa_pairs(sft_data):
    sft_qa_pairs = {}
    # extract unique query and its multiple paper info with scores
    for item in sft_data:
        query = item.get("input", "").get("user_query", "")
        paper_title = item.get("input", "").get("paper", "").get("title", "")
        score = item.get("output", {}).get("score", 0)
        if query and paper_title and score:
            if query not in sft_qa_pairs:
                sft_qa_pairs[query] = [{"answer": paper_title, "score": score}]
            else:
                sft_qa_pairs[query].append({"answer": paper_title, "score": score})
    return sft_qa_pairs
def get_gt_qa_pairs(gt_data):
    gt_qa_pairs = {}
    for item in gt_data:
        query = item.get("question", "")
        answer_titles = item.get("answer", [])
        if query and answer_titles:
            gt_qa_pairs[query] = answer_titles
    return gt_qa_pairs


# ==================== Core Evaluation Metrics ====================

def calculate_ndcg_at_k(sft_papers: List[Dict], gt_papers: List[str], k: int = 10) -> float:
    """
    Calculate NDCG@k treating GT papers as relevant with graded relevance
    First GT paper has highest relevance, others have lower relevance
    """
    # Create relevance mapping
    relevance_map = {}
    for idx, paper in enumerate(gt_papers):
        # First paper gets highest score, decreasing for others
        relevance_map[paper] = len(gt_papers) - idx
    
    # Sort SFT papers by score
    sorted_sft = sorted(sft_papers, key=lambda x: x['score'], reverse=True)
    
    # Calculate DCG@k
    dcg = 0.0
    for i, paper_info in enumerate(sorted_sft[:k]):
        rel = relevance_map.get(paper_info['answer'], 0)
        dcg += rel / np.log2(i + 2)  # i+2 because ranking starts at 1
    
    # Calculate IDCG@k (ideal DCG)
    ideal_relevances = sorted([len(gt_papers) - i for i in range(len(gt_papers))], reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances[:k]))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def calculate_recall_at_k(sft_papers: List[Dict], gt_papers: List[str], k: int = 10) -> float:
    """Calculate what fraction of GT papers appear in top-k SFT results"""
    sorted_sft = sorted(sft_papers, key=lambda x: x['score'], reverse=True)
    top_k_titles = [p['answer'] for p in sorted_sft[:k]]
    
    found = sum(1 for gt_paper in gt_papers if gt_paper in top_k_titles)
    return found / len(gt_papers) if gt_papers else 0.0

def calculate_mrr(sft_papers: List[Dict], gt_papers: List[str]) -> float:
    """Calculate Mean Reciprocal Rank for the best GT paper"""
    if not gt_papers:
        return 0.0
    
    sorted_sft = sorted(sft_papers, key=lambda x: x['score'], reverse=True)
    best_gt_paper = gt_papers[0]  # First paper is the best
    
    for rank, paper_info in enumerate(sorted_sft, 1):
        if paper_info['answer'] == best_gt_paper:
            return 1.0 / rank
    
    return 0.0  # Best paper not found

def calculate_pairwise_accuracy(sft_papers: List[Dict], gt_papers: List[str]) -> float:
    """Check if relative ordering of GT papers is preserved in SFT scores, gt[0] > others"""
    # Create score mapping
    score_map = {p['answer']: p['score'] for p in sft_papers}
    
    correct_pairs = 0
    total_pairs = 0
    
    # gt[0] should be ranked higher than all others
    best_paper = gt_papers[0]
    best_score = score_map.get(best_paper, -np.inf)
    for i in range(len(gt_papers)-1):
        other_paper = gt_papers[i+1]
        other_score = score_map.get(other_paper, -np.inf)
        if best_score > other_score:
            correct_pairs += 1
        total_pairs += 1
        
    
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def get_gt_paper_ranks(sft_papers: List[Dict], gt_papers: List[str]) -> Dict[str, int]:
    """Get ranks of GT papers in SFT ranking"""
    sorted_sft = sorted(sft_papers, key=lambda x: x['score'], reverse=True)
    rank_map = {p['answer']: rank + 1 for rank, p in enumerate(sorted_sft)}
    
    gt_ranks = {}
    for paper in gt_papers:
        gt_ranks[paper] = rank_map.get(paper, len(sorted_sft) + 1)  # If not found, worst rank
    
    return gt_ranks

# ==================== Comprehensive Evaluation ====================

def evaluate_single_query(sft_papers: List[Dict], gt_papers: List[str]) -> Dict:
    """Evaluate SFT ranking against GT for a single query"""
    
    if not gt_papers or not sft_papers:
        return {}
    
    # Get ranks of GT papers
    gt_ranks = get_gt_paper_ranks(sft_papers, gt_papers)
    rank_values = list(gt_ranks.values())
    
    # Score analysis
    score_map = {p['answer']: p['score'] for p in sft_papers}
    gt_scores = [score_map.get(p, 0) for p in gt_papers]
    all_scores = [p['score'] for p in sft_papers]
    non_gt_scores = [p['score'] for p in sft_papers if p['answer'] not in gt_papers]
    
    metrics = {
        # Ranking metrics
        # 'ndcg@5': calculate_ndcg_at_k(sft_papers, gt_papers, k=5),
        'ndcg@10': calculate_ndcg_at_k(sft_papers, gt_papers, k=10),
        # 'ndcg@20': calculate_ndcg_at_k(sft_papers, gt_papers, k=20),
        
        # Recall metrics
        # 'recall@5': calculate_recall_at_k(sft_papers, gt_papers, k=5),
        'recall@10': calculate_recall_at_k(sft_papers, gt_papers, k=10),
        # 'recall@20': calculate_recall_at_k(sft_papers, gt_papers, k=20),
        
        # Position metrics
        'mrr': calculate_mrr(sft_papers, gt_papers),
        'best_paper_rank': gt_ranks.get(gt_papers[0], len(sft_papers) + 1) if gt_papers else None,
        'mean_rank_gt': np.mean(rank_values),
        
        # Relative ranking
        'pairwise_consistency': calculate_pairwise_accuracy(sft_papers, gt_papers),
        
        # Score distribution
        'avg_score_gt': np.mean(gt_scores) if gt_scores else 0,
        'avg_score_non_gt': np.mean(non_gt_scores) if non_gt_scores else 0,
        'score_separation': (np.mean(gt_scores) - np.mean(non_gt_scores)) if gt_scores and non_gt_scores else 0,
        
        # Coverage
        # 'gt_papers_found': sum(1 for p in gt_papers if p in score_map),
        # 'gt_papers_total': len(gt_papers),
        'coverage_rate': sum(1 for p in gt_papers if p in score_map) / len(gt_papers),
        
        # Additional info
        'total_sft_papers': len(sft_papers),
        'gt_in_top_20_percent': sum(1 for r in rank_values if r <= len(sft_papers) * 0.2) / len(gt_papers),
    }
    
    return metrics

def evaluate_all_queries(sft_qa_pairs: Dict, gt_qa_pairs: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate all queries and return detailed results
    
    Returns:
        - DataFrame with per-query metrics
        - Dictionary with aggregated metrics
    """
    results = []
    assert set(sft_qa_pairs.keys()) == set(gt_qa_pairs.keys())
    # Find common queries
    common_queries = set(sft_qa_pairs.keys()) & set(gt_qa_pairs.keys())


    for query in common_queries:
        sft_papers = sft_qa_pairs[query]
        gt_papers = gt_qa_pairs[query]
        
        metrics = evaluate_single_query(sft_papers, gt_papers)
        metrics['query'] = query
        results.append(metrics)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate aggregated metrics
    aggregated = {
        'num_queries': len(common_queries),
        'gt_in_top_20_percent': (df_results['gt_in_top_20_percent'] >= 0.2).sum(),
        'avg_score_gt': df_results['avg_score_gt'].mean(),
        
        # Average metrics
        'avg_ndcg@5': df_results['ndcg@5'].mean(),
        'avg_ndcg@10': df_results['ndcg@10'].mean(),
        'avg_ndcg@20': df_results['ndcg@20'].mean(),
        
        'avg_recall@5': df_results['recall@5'].mean(),
        'avg_recall@10': df_results['recall@10'].mean(),
        'avg_recall@20': df_results['recall@20'].mean(),
        
        'avg_mrr': df_results['mrr'].mean(),
        'avg_best_paper_rank': df_results['best_paper_rank'].mean(),
        
        # # Distribution metrics
        # 'queries_with_perfect_best': (df_results['best_paper_rank'] == 1).sum(),
        # 'queries_with_best_in_top5': (df_results['best_paper_rank'] <= 5).sum(),
        # 'queries_with_best_in_top10': (df_results['best_paper_rank'] <= 10).sum(),
        
        # Coverage
        # 'avg_coverage_rate': df_results['coverage_rate'].mean(),
        'perfect_coverage_queries': (df_results['coverage_rate'] == 1.0).sum(),
    }
    
    return df_results, aggregated

# ==================== Analysis and Reporting ====================

def print_evaluation_report(df_results: pd.DataFrame, aggregated: Dict):
    """Print a comprehensive evaluation report"""
    
    print("=" * 80)
    print("SFT vs GT RANKING EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  - Common queries evaluated: {aggregated['num_queries']}")
    print(f"  - Queries only in SFT: {aggregated['num_queries_sft_only']}")
    print(f"  - Queries only in GT: {aggregated['num_queries_gt_only']}")
    
    print(f"\n🎯 Primary Metrics (Averaged across queries):")
    print(f"  - NDCG@10: {aggregated['avg_ndcg@10']:.3f}")
    print(f"  - MRR: {aggregated['avg_mrr']:.3f}")
    print(f"  - Recall@10: {aggregated['avg_recall@10']:.3f}")
    
    print(f"\n📈 Best Paper Performance:")
    print(f"  - Average rank of best paper: {aggregated['avg_best_paper_rank']:.1f}")
    print(f"  - Median rank of best paper: {aggregated['median_best_paper_rank']:.1f}")
    print(f"  - Queries with perfect ranking (best=1): {aggregated['queries_with_perfect_best']}/{aggregated['num_queries']}")
    print(f"  - Queries with best in top-5: {aggregated['queries_with_best_in_top5']}/{aggregated['num_queries']}")
    
    print(f"\n🔍 Ranking Quality:")
    print(f"  - Pairwise accuracy: {aggregated['avg_pairwise_accuracy']:.3f}")
    print(f"  - Coverage rate: {aggregated['avg_coverage_rate']:.3f}")
    
    print(f"\n📊 Detailed Metrics Table:")
    print("-" * 40)
    metrics_table = [
        ("Metric", "@5", "@10", "@20"),
        ("NDCG", f"{aggregated['avg_ndcg@5']:.3f}", f"{aggregated['avg_ndcg@10']:.3f}", f"{aggregated['avg_ndcg@20']:.3f}"),
        ("Recall", f"{aggregated['avg_recall@5']:.3f}", f"{aggregated['avg_recall@10']:.3f}", f"{aggregated['avg_recall@20']:.3f}"),
    ]
    
    for row in metrics_table:
        print(f"  {row[0]:<10} {row[1]:<8} {row[2]:<8} {row[3]:<8}")
    
    # Identify problematic queries
    print(f"\n⚠️  Queries Needing Attention:")
    worst_queries = df_results.nlargest(5, 'best_paper_rank')[['query', 'best_paper_rank', 'ndcg@10']]
    for _, row in worst_queries.iterrows():
        print(f"  - Query: '{row['query'][:50]}...' | Best rank: {row['best_paper_rank']:.0f} | NDCG@10: {row['ndcg@10']:.3f}")

# ==================== Main Evaluation Function ====================

def evaluate_sft_gt_consistency(sft_qa_pairs: Dict[str, List[Dict]], gt_qa_pairs: Dict[str, List[Dict]], verbose=True):
    """
    Main function to evaluate SFT and GT data consistency
    
    Args:
        sft_qa_pairs: Dictionary of SFT QA pairs
        gt_qa_pairs: Dictionary of GT QA pairs
        verbose: Whether to print detailed report
    
    Returns:
        Tuple of (per-query DataFrame, aggregated metrics dict)
    """
    
    # Run evaluation
    df_results, aggregated = evaluate_all_queries(sft_qa_pairs, gt_qa_pairs)
    
    if verbose and not df_results.empty:
        print_evaluation_report(df_results, aggregated)
    
    return df_results, aggregated

def main():
    with open(training_data_path, 'r', encoding='utf-8') as f:
        sft_data = [json.loads(line) for line in f]
    with open(gt_data_path, 'r', encoding='utf-8') as f:
        gt_data = [json.loads(line) for line in f]
    sft_queries = get_sft_qa_pairs(sft_data)
    # sort the paper info by score in descending order
    sorted_sft_queries = {k: sorted(v, key=lambda x: x['score'], reverse=True) for k, v in sft_queries.items()}
    gt_queries = get_gt_qa_pairs(gt_data)
    assert set(sorted_sft_queries.keys()) == set(gt_queries.keys()), "Mismatch in queries between SFT data and GT data"
    # compare the sft training data with ground truth data in ranking
    df_results, aggregated = evaluate_sft_gt_consistency(sft_data, gt_data)

    recall_at_k = recall_at_k(
        sorted_sft_queries,
        gt_queries,
        k=10
    )

    