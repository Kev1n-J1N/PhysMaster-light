import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from utils import *
import json, os

if __name__ == "__main__":
    arxiv_list = ["2103.09217"]
    cite = r"~\\cite\{(.*?)\}"
    save_dir = f"{current_dir}/../datasets/paper_database"
    
    id_path = os.path.join(save_dir, "id2paper.json")
    paper_dir = os.path.join(save_dir, "paper_db")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(paper_dir):
        os.makedirs(paper_dir)
        
    for arxiv_id in arxiv_list:
        result = search_paper_by_arxiv_id(arxiv_id)
        result["source"] = result["source"]+f"_{arxiv_id}"
        result["sections"] = search_section_by_arxiv_id(arxiv_id,cite)
        with open(os.path.join(paper_dir, f"{keep_letters(result['title'])}"), "w") as f:
            json.dump(result, f)
            path = os.path.join(paper_dir, f"{keep_letters(result['title'])}.json")
            print(f"✅ 已保存到 {path}")
        with open(id_path, "a") as f:
            f.write(f"{arxiv_id}: {result['title']}\n")
            print(f"✅ 已保存到 {id_path}")
