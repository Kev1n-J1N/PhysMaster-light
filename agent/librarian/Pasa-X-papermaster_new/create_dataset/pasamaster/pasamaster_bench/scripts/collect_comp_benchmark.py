"""
Split generated benchmarks into train/test sets with non-overlapping paper lists
Maintains approximately 3:1 ratio between train and test sets
"""

import json
import os
import random
from typing import List, Dict, Tuple, Set
from pathlib import Path
import argparse
from collections import defaultdict

def load_benchmark_from_dir(benchmark_dir: str) -> Dict:
    """
    Load benchmark data from a directory containing the generated files
    
    Args:
        benchmark_dir: Directory containing benchmark files
    
    Returns:
        Dictionary with benchmark data
    """
    benchmark_path = os.path.join(benchmark_dir, "benchmark_entry.json")
    
    if os.path.exists(benchmark_path):
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Benchmark file not found in {benchmark_dir}")
def collect_all_benchmarks(output_dir: str) -> List[Dict]:
    """
    Collect all benchmark entries from subdirectories
    
    Args:
        output_dir: Base output directory containing benchmark subdirectories
    
    Returns:
        List of benchmark entries
    """
    all_entries = []
    
    # Iterate through all subdirectories
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"Processing benchmark directory: {subdir}")
        
        try:
            benchmark_data = load_benchmark_from_dir(subdir_path)
            
            # Create entries with question, queries, target_paper, and paper_list
            questions = benchmark_data.get("questions", [])
            queries = benchmark_data.get("queries", [])
            papers = benchmark_data.get("papers", [])
            
            # Match questions with queries (assuming they're aligned by index)
            for i, question in enumerate(questions):
                if i < len(queries):
                    entry = {
                        "question": question["question"],
                        "queries": queries,
                        "target_paper": question["paper_id"],  # Assuming first paper is target
                        "paper_list": papers
                        }
                    all_entries.append(entry)
        
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            continue
    
    return all_entries

def split_papers_disjoint(all_entries: List[Dict], train_ratio: float = 0.75) -> Tuple[Set[str], Set[str]]:
    """
    Split papers into disjoint train and test sets
    
    Args:
        all_entries: List of all benchmark entries
        train_ratio: Ratio of papers to include in training set
    
    Returns:
        Tuple of (train_paper_ids, test_paper_ids)
    """
    # Collect all unique papers
    all_papers = set()
    for entry in all_entries:
        if entry.get("paper_list"):
            for paper in entry["paper_list"]:
                # Extract paper ID (assuming papers have an 'id' field or use the whole object as ID)
                if isinstance(paper, dict):
                    paper_id = paper.get("id", str(paper))
                else:
                    paper_id = str(paper)
                all_papers.add(paper_id)
    
    # Convert to list and shuffle
    all_papers_list = list(all_papers)
    random.shuffle(all_papers_list)
    
    # Split papers
    split_point = int(len(all_papers_list) * train_ratio)
    train_papers = set(all_papers_list[:split_point])
    test_papers = set(all_papers_list[split_point:])
    
    print(f"Total unique papers: {len(all_papers)}")
    print(f"Train papers: {len(train_papers)}")
    print(f"Test papers: {len(test_papers)}")
    
    return train_papers, test_papers

def assign_entries_to_splits(
    all_entries: List[Dict], 
    train_papers: Set[str], 
    test_papers: Set[str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Assign entries to train or test based on their paper lists
    
    Args:
        all_entries: List of all benchmark entries
        train_papers: Set of paper IDs for training
        test_papers: Set of paper IDs for testing
    
    Returns:
        Tuple of (train_entries, test_entries)
    """
    train_entries = []
    test_entries = []
    ambiguous_entries = []
    
    for entry in all_entries:
        paper_list = entry.get("paper_list", [])
        if not paper_list:
            ambiguous_entries.append(entry)
            continue
        
        # Get paper IDs from this entry
        entry_paper_ids = set()
        for paper in paper_list:
            if isinstance(paper, dict):
                paper_id = paper.get("id", str(paper))
            else:
                paper_id = str(paper)
            entry_paper_ids.add(paper_id)
        
        # Check if all papers belong to train or test
        in_train = entry_paper_ids.intersection(train_papers)
        in_test = entry_paper_ids.intersection(test_papers)
        
        if in_train and not in_test:
            # All papers in training set
            train_entries.append(entry)
        elif in_test and not in_train:
            # All papers in test set
            test_entries.append(entry)
        else:
            # Mixed or ambiguous - need to handle
            # Strategy: assign based on majority
            if len(in_train) >= len(in_test):
                # Filter paper_list to only include train papers
                filtered_papers = [
                    p for p in paper_list 
                    if (p.get("id", str(p)) if isinstance(p, dict) else str(p)) in train_papers
                ]
                if filtered_papers:
                    entry_copy = entry.copy()
                    entry_copy["paper_list"] = filtered_papers
                    train_entries.append(entry_copy)
            else:
                # Filter paper_list to only include test papers
                filtered_papers = [
                    p for p in paper_list 
                    if (p.get("id", str(p)) if isinstance(p, dict) else str(p)) in test_papers
                ]
                if filtered_papers:
                    entry_copy = entry.copy()
                    entry_copy["paper_list"] = filtered_papers
                    test_entries.append(entry_copy)
    
    # Handle ambiguous entries (no paper list) - distribute according to ratio
    random.shuffle(ambiguous_entries)
    split_point = int(len(ambiguous_entries) * 0.75)
    train_entries.extend(ambiguous_entries[:split_point])
    test_entries.extend(ambiguous_entries[split_point:])
    
    return train_entries, test_entries

def save_splits_to_jsonl(
    train_entries: List[Dict], 
    test_entries: List[Dict], 
    output_dir: str,
    prefix: str = "benchmark"
):
    """
    Save train and test splits to JSONL files
    
    Args:
        train_entries: Training set entries
        test_entries: Test set entries
        output_dir: Directory to save the files
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training set
    train_file = os.path.join(output_dir, f"{prefix}_train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)

    # Save test set
    test_file = os.path.join(output_dir, f"{prefix}_test.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_entries, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(train_entries)} training entries to: {train_file}")
    print(f"Saved {len(test_entries)} test entries to: {test_file}")
    
    # Save split statistics
    stats_file = os.path.join(output_dir, f"{prefix}_split_stats.json")
    stats = {
        "train_entries": len(train_entries),
        "test_entries": len(test_entries),
        "ratio": f"{len(train_entries)}:{len(test_entries)}",
        "train_percentage": len(train_entries) / (len(train_entries) + len(test_entries)) * 100
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Split statistics saved to: {stats_file}")

def verify_disjoint_papers(train_entries: List[Dict], test_entries: List[Dict]) -> bool:
    """
    Verify that paper lists in train and test sets are disjoint
    
    Args:
        train_entries: Training set entries
        test_entries: Test set entries
    
    Returns:
        True if disjoint, False otherwise
    """
    train_papers = set()
    for entry in train_entries:
        for paper in entry.get("paper_list", []):
            if isinstance(paper, dict):
                paper_id = paper.get("id", str(paper))
            else:
                paper_id = str(paper)
            train_papers.add(paper_id)
    
    test_papers = set()
    for entry in test_entries:
        for paper in entry.get("paper_list", []):
            if isinstance(paper, dict):
                paper_id = paper.get("id", str(paper))
            else:
                paper_id = str(paper)
            test_papers.add(paper_id)
    
    overlap = train_papers.intersection(test_papers)
    
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping papers between train and test sets!")
        print(f"Overlapping papers: {list(overlap)[:5]}...")  # Show first 5
        return False
    else:
        print(f"✓ Verification passed: Train and test paper lists are disjoint")
        print(f"  - Train papers: {len(train_papers)}")
        print(f"  - Test papers: {len(test_papers)}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Split benchmark data into train/test sets")
    parser.add_argument("--input_dir", type=str, default="output\\APASbech_comp",
                        help="Directory containing benchmark subdirectories")
    parser.add_argument("--output_dir", type=str, default="output\\APASbech_comp_splits",
                        help="Directory to save the split files")
    parser.add_argument("--prefix", type=str, default="APASbech_comp",
                        help="Prefix for output filenames")
    parser.add_argument("--train_ratio", type=float, default=0.55,
                        help="Ratio of data for training set (default: 0.55 for 11:9 split)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("="*60)
    print("Benchmark Dataset Splitter")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio} ({args.train_ratio}:{1-args.train_ratio})")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Collect all benchmarks
    print("\nCollecting benchmark entries...")
    all_entries = collect_all_benchmarks(args.input_dir)
    print(f"Total entries collected: {len(all_entries)}")
    
    if not all_entries:
        print("No entries found! Please check your input directory.")
        return
    
    # Split papers into disjoint sets
    print("\nSplitting papers into disjoint sets...")
    train_papers, test_papers = split_papers_disjoint(all_entries, args.train_ratio)
    
    # Assign entries to train/test based on papers
    print("\nAssigning entries to train/test sets...")
    train_entries, test_entries = assign_entries_to_splits(all_entries, train_papers, test_papers)
    
    print(f"\nFinal split:")
    print(f"  - Training entries: {len(train_entries)}")
    print(f"  - Test entries: {len(test_entries)}")
    print(f"  - Actual ratio: {len(train_entries)/len(test_entries):.2f}:1")
    
    # Verify disjoint paper lists
    print("\nVerifying disjoint paper lists...")
    verify_disjoint_papers(train_entries, test_entries)
    
    # Save to JSONL files
    print("\nSaving splits to JSONL files...")
    save_splits_to_jsonl(train_entries, test_entries, args.output_dir, args.prefix)
    
    print("\n" + "="*60)
    print("Split completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()