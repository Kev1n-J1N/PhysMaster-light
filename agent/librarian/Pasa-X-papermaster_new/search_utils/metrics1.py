import os
import json
import glob
import argparse
from tqdm import tqdm

def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default="results/pasa_crawer_pasa_selector_20250702")
args = parser.parse_args()

pred_files = [f for f in glob.glob(args.output_folder + "/*.json") if not os.path.basename(f).startswith("mtr") and not os.path.basename(f).startswith("eval")]
# 用于统计所有论文的集合
all_papers_set = set()
total_papers_count = 0

# 用于统计crawler_recall_papers的集合
all_crawler_recall_papers_set = set()
total_crawler_recall_papers_count = 0

for pred_file in tqdm(pred_files):    
    with open(pred_file) as f:
        paper_root = json.load(f)
    
    # 递归读取每个文件的每个title，加入crawled_paper_set
    crawled_paper_set = set()
    queue = [paper_root]
    
    while len(queue) > 0:
        node, queue = queue[0], queue[1:]
        for _, v in node["child"].items():
            for i in v:
                queue.append(i)
                if keep_letters(i["title"]) not in crawled_paper_set:
                    crawled_paper_set.add(keep_letters(i["title"]))
    
    # 将当前文件的论文添加到全局集合中
    all_papers_set.update(crawled_paper_set)
    total_papers_count += len(crawled_paper_set)
    
    # 统计crawler_recall_papers
    if "extra" in paper_root and "crawler_recall_papers" in paper_root["extra"]:
        crawler_recall_papers = paper_root["extra"]["crawler_recall_papers"]
        crawler_recall_papers_set = set([keep_letters(paper) for paper in crawler_recall_papers])
        all_crawler_recall_papers_set.update(crawler_recall_papers_set)
        total_crawler_recall_papers_count += len(crawler_recall_papers_set)

print(f"所有文件加起来的总论文数（去重前）: {total_papers_count}")
print(f"所有文件加起来的总论文数（去重后）: {len(all_papers_set)}")
print(f"重复论文数: {total_papers_count - len(all_papers_set)}")
print(f"处理的文件数量: {len(pred_files)}")
print(f"平均每个文件的论文数: {total_papers_count / len(pred_files):.2f}")

print("\n=== Crawler Recall Papers 统计 ===")
print(f"所有文件crawler_recall_papers总数（去重前）: {total_crawler_recall_papers_count}")
print(f"所有文件crawler_recall_papers总数（去重后）: {len(all_crawler_recall_papers_set)}")
print(f"crawler_recall_papers重复数: {total_crawler_recall_papers_count - len(all_crawler_recall_papers_set)}")
print(f"平均每个文件的crawler_recall_papers数: {total_crawler_recall_papers_count / len(pred_files):.2f}")
