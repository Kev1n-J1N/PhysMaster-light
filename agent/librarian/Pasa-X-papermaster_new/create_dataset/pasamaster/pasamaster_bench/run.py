from typing import List, Dict
import json
import asyncio
from benchmark_generator import BenchmarkGenerator
import random
import os
from llm import LLMClient
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
import time

from config import API_KEY, MODEL, API_BASE

# Global lock for thread-safe console output
print_lock = threading.Lock()

def safe_print(message: str):
    """Thread-safe printing"""
    with print_lock:
        print(message)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process paper files for benchmark generation")
    parser.add_argument("--papers_dir", type=str, 
                        default="papers\\APASbech_comp", 
                        help="Path to the directory containing paper files")
    parser.add_argument("--output_dir", type=str, 
                        default="output\\APASbech_comp", 
                        help="Path to the output directory")
    parser.add_argument("--question_type", type=str, 
                        default="compound", 
                        help="Type of questions to generate")
    parser.add_argument("--max_workers", type=int, 
                        default=5, 
                        help="Maximum number of parallel workers")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip papers that already have output directories")
    parser.add_argument("--rate_limit_delay", type=float,
                        default=1.0,
                        help="Delay in seconds between API calls to avoid rate limits")
    return parser.parse_args()

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait(self):
        """Wait if necessary to maintain rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call_time = time.time()

# Global rate limiter
rate_limiter = None

async def process_single_paper(
    paper_file: str,
    papers_dir_path: str,
    output_basedir: str,
    question_type: str,
    worker_id: int,
    skip_existing: bool = False
) -> Dict:
    """
    Process a single paper file and generate benchmark
    
    Args:
        paper_file: Name of the paper file
        papers_dir_path: Directory containing paper files
        output_basedir: Base output directory
        question_type: Type of questions to generate
        worker_id: ID of the worker processing this paper
        skip_existing: Whether to skip existing outputs
    
    Returns:
        Dictionary with results and statistics
    """
    paper_file_name = os.path.basename(paper_file)
    output_dir = os.path.join(output_basedir, paper_file_name.replace(".json", "").replace(".jsonl", ""))
    
    # Check if should skip
    if skip_existing and os.path.exists(output_dir):
        safe_print(f"[Worker {worker_id}] Output directory for {paper_file_name} already exists, skipping...")
        return {
            "paper_file": paper_file_name,
            "status": "skipped",
            "reason": "output_exists"
        }
    
    paper_path = os.path.join(papers_dir_path, paper_file)
    
    try:
        # Rate limiting
        if rate_limiter:
            rate_limiter.wait()
        
        # Create LLM client for this worker
        llm = LLMClient(api_key=API_KEY, model=MODEL, api_base=API_BASE)
        
        start_time = time.time()
        safe_print(f"[Worker {worker_id}] Processing paper: {paper_file_name}")
        
        # Generate benchmark
        benchmark_generator = BenchmarkGenerator(
            llm=llm, 
            paper_path=paper_path,
            base_path=output_dir, 
            question_type=question_type
        )
        
        benchmark = await benchmark_generator.generate_benchmark_entry()
        
        elapsed_time = time.time() - start_time
        
        # Collect statistics
        result = {
            "paper_file": paper_file_name,
            "status": "success",
            "elapsed_time": elapsed_time,
            "statistics": {
                "papers": len(benchmark.get('papers', [])),
                "questions": len(benchmark.get('questions', [])),
                "queries": len(benchmark.get('queries', []))
            }
        }
        
        safe_print(f"[Worker {worker_id}] Completed {paper_file_name} in {elapsed_time:.2f}s")
        safe_print(f"[Worker {worker_id}] Stats - Papers: {result['statistics']['papers']}, "
                  f"Questions: {result['statistics']['questions']}, "
                  f"Queries: {result['statistics']['queries']}")
        
        return result
        
    except Exception as e:
        safe_print(f"[Worker {worker_id}] Error processing {paper_file_name}: {str(e)}")
        return {
            "paper_file": paper_file_name,
            "status": "error",
            "error": str(e)
        }

def process_paper_wrapper(args):
    """Wrapper to run async function in thread pool"""
    paper_file, papers_dir_path, output_basedir, question_type, worker_id, skip_existing = args
    
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            process_single_paper(
                paper_file, 
                papers_dir_path, 
                output_basedir, 
                question_type, 
                worker_id,
                skip_existing
            )
        )
        return result
    finally:
        loop.close()

async def process_papers_parallel(
    papers_dir_path: str,
    output_basedir: str,
    question_type: str,
    max_workers: int = 5,
    skip_existing: bool = False,
    rate_limit_delay: float = 1.0
) -> List[Dict]:
    """
    Process multiple papers in parallel
    
    Args:
        papers_dir_path: Directory containing paper files
        output_basedir: Base output directory
        question_type: Type of questions to generate
        max_workers: Maximum number of parallel workers
        skip_existing: Whether to skip existing outputs
        rate_limit_delay: Delay between API calls
    
    Returns:
        List of results for each paper
    """
    # Get all paper files
    paper_files = [
        f for f in os.listdir(papers_dir_path)
        if f.endswith(".json") or f.endswith(".jsonl")
    ]
    
    if not paper_files:
        safe_print("No paper files found!")
        return []
    
    safe_print(f"Found {len(paper_files)} paper files to process")
    safe_print(f"Using {max_workers} parallel workers")
    
    # Prepare work items
    work_items = [
        (paper_file, papers_dir_path, output_basedir, question_type, i % max_workers, skip_existing)
        for i, paper_file in enumerate(paper_files)
    ]
    
    results = []
    completed = 0
    failed = 0
    skipped = 0
    
    # Process papers in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = list(executor.map(process_paper_wrapper, work_items))
        
        # Collect results
        for result in futures:
            results.append(result)
            completed += 1
            
            if result['status'] == 'error':
                failed += 1
            elif result['status'] == 'skipped':
                skipped += 1
            
            # Progress update
            safe_print(f"\nProgress: {completed}/{len(paper_files)} completed "
                      f"({failed} failed, {skipped} skipped)")
    
    return results

def save_summary(results: List[Dict], output_basedir: str):
    """Save processing summary to file"""
    summary_path = os.path.join(output_basedir, "processing_summary.json")
    
    # Calculate summary statistics
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    total_time = sum(r.get('elapsed_time', 0) for r in results if r['status'] == 'success')
    avg_time = total_time / successful if successful > 0 else 0
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": total,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "total_processing_time": total_time,
        "average_time_per_paper": avg_time,
        "details": results
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_basedir, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    safe_print(f"\nSummary saved to: {summary_path}")
    
    return summary

# Main execution
async def main():
    global rate_limiter
    
    args = parse_arguments()
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(min_interval=args.rate_limit_delay)
    
    print("="*60)
    print("PARALLEL Paper Search Benchmark Generation")
    print("="*60)
    print(f"Papers Directory: {args.papers_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Question Type: {args.question_type}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Skip Existing: {args.skip_existing}")
    print(f"Rate Limit Delay: {args.rate_limit_delay}s")
    print("="*60)
    
    start_time = time.time()
    
    # Process papers in parallel
    results = await process_papers_parallel(
        papers_dir_path=args.papers_dir,
        output_basedir=args.output_dir,
        question_type=args.question_type,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
        rate_limit_delay=args.rate_limit_delay
    )
    
    total_time = time.time() - start_time
    
    # Save and print summary
    if results:
        summary = save_summary(results, args.output_dir)
        
        print("\n" + "="*60)
        print("FINAL BENCHMARK GENERATION STATISTICS:")
        print("="*60)
        print(f"Total Files Processed: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Total Processing Time: {total_time:.2f} seconds")
        print(f"Average Time per Paper: {summary['average_time_per_paper']:.2f} seconds")
        
        # Print detailed statistics for successful papers
        if summary['successful'] > 0:
            print("\nDetailed Statistics for Successful Papers:")
            total_papers = sum(r['statistics']['papers'] for r in results if r['status'] == 'success')
            total_questions = sum(r['statistics']['questions'] for r in results if r['status'] == 'success')
            total_queries = sum(r['statistics']['queries'] for r in results if r['status'] == 'success')
            
            print(f"- Total Papers in Benchmarks: {total_papers}")
            print(f"- Total Questions Generated: {total_questions}")
            print(f"- Total Queries Generated: {total_queries}")
        
        # Print errors if any
        errors = [r for r in results if r['status'] == 'error']
        if errors:
            print("\nFailed Papers:")
            for error in errors:
                print(f"- {error['paper_file']}: {error.get('error', 'Unknown error')}")
        
        print("="*60)
        print(f"Speedup: {summary['successful'] / (total_time / 3600):.1f}x compared to sequential processing")
    else:
        print("No papers were processed.")

if __name__ == "__main__":
    asyncio.run(main())