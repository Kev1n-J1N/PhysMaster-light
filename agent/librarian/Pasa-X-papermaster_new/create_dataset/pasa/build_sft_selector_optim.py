import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)
from optim_utils import *
from utils import  search_paper_by_arxiv_id
from model_vllm import Agent
from paper_agent import PaperAgent
from datetime import datetime
import openai
from prompt   import selector_true_sample_prompt
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from threading import Lock, get_ident
MODELNAME = "qwen-72b"
from llm_call import *
from datetime import datetime
from threading import get_ident

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        thread_id = get_ident()
        message = f"[Thread-{thread_id}] {message}"  # 自动添加线程 ID
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
def get_true_sample(data, save_path):
    thread_id = get_ident()
    true_samples = []

    for paper_title in data["answer"]:
        print(f"[Thread-{thread_id}] 📝 Trying to search paper title: '{paper_title}'")

        title_id = search_id_by_title(paper_title)
        if title_id is None:
            print(f"[Thread-{thread_id}] ❌ Failed to get arXiv ID for paper: '{paper_title}'")
            continue  # 跳过这个论文
        else:
            print(f"[Thread-{thread_id}] ✅ Got arXiv ID: {title_id} for paper: '{paper_title}'")

        # 去除版本号
        title_id = title_id.split('v')[0]

        print(f"[Thread-{thread_id}] 🕵️‍♂️ Fetching paper details for arXiv ID: {title_id}")
        paper = search_paper_by_arxiv_id(title_id)

        if paper is not None:
            abstract = paper["abstract"]
            title = paper["title"]

            print(f"[Thread-{thread_id}] 📄 Successfully fetched paper: '{title}'")
            print(f"[Thread-{thread_id}] 📝 Abstract: {abstract[:100]}...")  # 显示摘要前100字符

            prompt = selector_true_sample_prompt(question, title, abstract)
            decision = llm_call(prompt).split("\n\n")[0].strip().lower()

            print(f"[Thread-{thread_id}] 🔍 Decision for paper '{title}': {decision}")

            if "true" in decision:
                sample = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": decision}
                    ]
                }
                true_samples.append(sample)
        else:
            print(f"[Thread-{thread_id}] ❌ Failed to fetch paper details for arXiv ID: {title_id}")

    # 写入文件
    if true_samples:
        with open(save_path, "a", encoding="utf-8") as f:
            for sample in true_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")

def get_false_sample(question, crawler, selector, args, data, save_path):
    end_date = datetime.now().strftime("%Y%m%d")
    paper_agent = PaperAgent(
        user_query=question,
        crawler=crawler,
        selector=selector,
        end_date=end_date,
        expand_layers=args.expand_layers,
        search_queries=args.expand_papers,
        search_papers=args.search_papers,
        expand_papers=args.expand_papers,
        threads_num=args.threads_num,
        max_num=args.max_num,
        time_limit=args.time_limit,
        build_selector_sample=True
    )

    paper_agent.run()

    false_sample_list = []
    for prompt, decision, title in zip(
        paper_agent.root.extra["false_sample"]["prompt"],
        paper_agent.root.extra["false_sample"]["answer"],
        paper_agent.root.extra["false_sample"]["title"]
    ):
        print(f"[Thread-{thread_id}] 📝 Found potential false sample: '{title}'")
        sample = {
            "title": title,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": decision}
            ]
        }
        false_sample_list.append(sample)

    paper_list = data["answer"]
    false_samples = []
    for paper in false_sample_list:
        title = paper["title"]
        if title not in paper_list:
            print(f"[Thread-{thread_id}] 🆕 Adding false sample: '{title}' (not in answer list)")
            false_samples.append({"messages": paper["messages"]})
        else:
            print(f"[Thread-{thread_id}] 🚫 Skipping false sample: '{title}' (already in answer list)")

    # 写入文件
    if false_samples:
        with open(save_path, "a", encoding="utf-8") as f:
            for sample in false_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    sys.path.append(current_dir)

    # 创建日志目录
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名：log_年月日_时分秒.log
    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 重定向标准输出到日志文件
    sys.stdout = Logger(log_file)

    print("✅ 程序启动，日志已写入:", log_file)
    parser = argparse.ArgumentParser(description="Process some papers.")
    parser.add_argument("--expand_layers", type=int, default=0, help="Number of expansion layers")
    parser.add_argument("--expand_papers", type=int, default=10, help="Number of expanded papers")
    parser.add_argument("--search_papers", type=int, default=10, help="Number of search papers")
    parser.add_argument("--threads_num", type=int, default=1, help="Number of threads")
    parser.add_argument("--max_num", type=int, default=10, help="Maximum number of results")
    parser.add_argument("--time_limit", type=int, default=10, help="Time limit for processing")
    args = parser.parse_args()

    MODELNAME = "qwen-72b"  # 示例值，根据你的实际变量调整

    path = f"{current_dir}/result/dataset/AQS"
    true_save_path = f"{path}/../sft_selector/math_ids_ASQ_1000_{MODELNAME}_true_sample2.jsonl"
    false_save_path = f"{path}/../sft_selector/math_ids_ASQ_1000_{MODELNAME}_false_sample2.jsonl"

    with open(f"{path}/math_ids_ASQ_1000_qwen-72b.jsonl", "r", encoding="utf-8") as f:
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
