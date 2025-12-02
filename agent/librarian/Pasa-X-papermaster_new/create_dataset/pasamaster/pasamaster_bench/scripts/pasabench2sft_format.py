import json
from datetime import datetime
from typing import List, Dict, Any

def convert_json_format(input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert JSON from input format to output format.
    
    Args:
        input_data: List of dictionaries containing question, queries, target_paper, and paper_list
    
    Returns:
        Dictionary in the output format with papers, arxiv_ids, and metadata
    """
    
    output = []
    # Process each entry in the input
    for entry in input_data:
        target_paper_id = entry.get("target_paper", "")
        all_papers = entry.get("paper_list", [])
        # Process all papers in the paper_list
        for paper in all_papers:
            paper_id = paper.get("id", "")
            # Store paper information
            if paper_id not in all_papers:
                all_papers[paper_id] = {
                    "title": paper.get("title", ""),
                    "published": paper.get("published", ""),
                    "is_target": False
                }
            
            # Mark if this is a target paper
            if paper_id == target_paper_id:
                all_papers[paper_id]["is_target"] = True
            
            # Track the latest publication date
            paper_date = paper.get("published", "")
            if paper_date:
                try:
                    # Parse the ISO format date
                    current_date = datetime.fromisoformat(paper_date.replace("+00:00", ""))
                    if latest_date is None or current_date > latest_date:
                        latest_date = current_date
                except:
                    pass
    
    # Prepare the output
    paper_titles = []
    paper_arxiv_ids = []
    
    # First, add all target papers (maintaining order from input)
    for entry in input_data:
        target_paper_id = entry.get("target_paper", "")
        if target_paper_id in all_papers:
            title = all_papers[target_paper_id]["title"]
            if title not in paper_titles:  # Avoid duplicates
                paper_titles.append(title)
                paper_arxiv_ids.append(target_paper_id)
    
            # Then add non-target papers
            for paper_id, paper_info in all_papers.items():
                if not paper_info["is_target"]:
                    title = paper_info["title"]
                    if title not in paper_titles:  # Avoid duplicates
                        paper_titles.append(title)
                        paper_arxiv_ids.append(paper_id)
    
            # Format the publication date
            published_time = ""
            if latest_date:
                published_time = latest_date.strftime("%Y%m%d")
    
            # Create output structure
            output = {
                "question": "Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.",
                "answer": paper_titles,
                "answer_arxiv_id": paper_arxiv_ids,
                "source_meta": {
                    "published_time": published_time
                },
                "qid": "RealScholarQuery_0"
            }
    
    return output


def process_file(input_file: str, output_file: str):
    """
    Read input JSON file, convert format, and write to output file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Convert format
    output_data = convert_json_format(input_data)
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {input_file} to {output_file}")


# Example usage
if __name__ == "__main__":
    input_file = "output/APASbench_comp_splits/APASbech_comp_test.json"
    output_file = "output/APASbench_comp_splits/APASbech_comp_test_converted.json"
    process_file(input_file, output_file)