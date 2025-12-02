import json
import os
import sys
from optim_utils import search_id_by_title
from search_utils import search_paper_by_arxiv_id
from model_vllm import Agent
from paper_agent import PaperAgent
from datetime import datetime
import openai
from prompt import selector_true_sample_prompt
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from llm_call import *
MODELNAME = "qwen-72b"

def get_true_sample(data, save_path):
    question = data["question"]  # 从数据中提取用户问题

    # 遍历每个论文标题
    for paper_title in data["answer"]:
        # 使用标题搜索论文信息（调用 utils 中的函数）
        title_id = search_id_by_title(paper_title)
        if title_id is None:
            return None
        title_id = title_id.split('v')[0]
        paper=search_paper_by_arxiv_id(title_id)

        if paper is not None:
            abstract = paper["abstract"]  # 获取摘要
            title = paper["title"]        # 获取标题
            print(f"Processing paper: {title}")
            print(f"Abstract: {abstract}")

            # 构造提示词模板（selector_true_sample_prompt）
            prompt = selector_true_sample_prompt(question, title, abstract)

            # 调用大模型进行判断
            decision = llm_call(prompt).split("\n\n")[0].strip().lower()

            print(decision)

            # 如果模型判断为“true”，则将该样本保存为正样本
            if "true" in decision:
                sample = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": decision}
                    ]
                }

                # 将样本追加写入 JSONL 文件
                with open(save_path, "a", encoding="utf-8") as f:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write("\n")

def get_false_sample(question, crawler, selector, args, data, save_path):
    end_date = datetime.now().strftime("%Y%m%d")
    
    # 初始化 PaperAgent 对象
    paper_agent = PaperAgent(
        user_query     = question,          # 用户的问题
        crawler        = crawler,           # 爬虫代理
        selector       = selector,          # 选择器代理
        end_date       = end_date,          # 结束日期
        expand_layers  = args.expand_layers,# 展开层数
        search_queries = args.expand_papers,# 扩展查询数量
        search_papers  = args.search_papers,# 搜索论文数量
        expand_papers  = args.expand_papers,# 扩展论文数量
        threads_num    = args.threads_num,  # 线程数量
        max_num        = args.max_num,      # 最大结果数量
        time_limit     = args.time_limit,   # 时间限制
        build_selector_sample=True         # 是否构建选择器样本
    )
    
    # 运行 PaperAgent
    paper_agent.run()
    
    # 收集潜在的负样本
    false_sample_list = []
    for prompt, decision, title in zip(
        paper_agent.root.extra["false_sample"]["prompt"],
        paper_agent.root.extra["false_sample"]["answer"],
        paper_agent.root.extra["false_sample"]["title"]
    ):
        sample = {
            "title": title,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": decision}
            ]
        }
        false_sample_list.append(sample)
    
    # 过滤掉与正样本重复的论文
    final_false_sample_list = []
    paper_list = data["answer"]  # 正样本的论文标题列表
    for paper in false_sample_list:
        if paper["title"] not in paper_list:
            final_false_sample_list.append(paper["messages"])
            # 将样本追加写入 JSONL 文件
            with open(save_path, "a", encoding="utf-8") as f:
                json.dump({"messages": paper["messages"]}, f, ensure_ascii=False)
                f.write("\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    sys.path.append(current_dir)

    parser = argparse.ArgumentParser(description="Process some papers.")
    parser.add_argument("--expand_layers", type=int, default=0, help="Number of expansion layers")
    parser.add_argument("--expand_papers", type=int, default=10, help="Number of expanded papers")
    parser.add_argument("--search_papers", type=int, default=10, help="Number of search papers")
    parser.add_argument("--threads_num", type=int, default=10, help="Number of threads")
    parser.add_argument("--max_num", type=int, default=10, help="Maximum number of results")
    parser.add_argument("--time_limit", type=int, default=10, help="Time limit for processing")
    args = parser.parse_args()

    path = f"{current_dir}/result/dataset/AQS"
    true_save_path = f"{path}/../sft_selector/math_ids_ASQ_10000_{MODELNAME}_true_sample2.jsonl"
    false_save_path = f"{path}/../sft_selector/math_ids_ASQ_10000_{MODELNAME}_false_sample2.jsonl"

    with open(f"{path}/math_ids_ASQ_10000_qwen-72b.jsonl", "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f]

    # 初始化 agents
    crawler = Agent(model_name="crawer", url="http://172.26.104.240:30016/v1")
    selector = Agent(model_name="selector", url="http://172.26.104.240:30001/v1")

    # === 并发执行 get_true_sample ===
    def process_true(item):
        get_true_sample(item, true_save_path)

    print("Processing true samples...")
    with ThreadPoolExecutor(max_workers=args.threads_num) as executor:
        futures = [executor.submit(process_true, item) for item in data_list]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in true sample processing: {e}")

    # === 并发执行 get_false_sample ===
    def process_false(item):
        get_false_sample(item["question"], crawler, selector, args, item, false_save_path)

    print("Processing false samples...")
    with ThreadPoolExecutor(max_workers=args.threads_num) as executor:
        futures = [executor.submit(process_false, item) for item in data_list]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in false sample processing: {e}")

    print("All tasks completed.")


