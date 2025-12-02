import re
import json
import arxiv
import warnings
import requests
from datetime   import datetime
from pdf_read import *
from paper_download import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from build_sft_crawler_expand import *
GOOGLE_KEY   = 'ec52ebf6f00f92df6070705691e0dc2fd06b3cf1'
arxiv_client = arxiv.Client(delay_seconds = 0.05)

def google_search_arxiv_id(query, num=None, end_date=None):
    url = "https://google.serper.dev/search"

    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    
    payload = json.dumps({
        "q": search_query, 
        "num": num, 
        "page": 1, 
    })

    headers = {
        'X-API-KEY': GOOGLE_KEY,
        'Content-Type': 'application/json'
    }
    assert headers['X-API-KEY'] != 'your google keys', "add your google search key!!!"

    for attempt in range(3):
        try:
            print(f"\n[Attempt {attempt+1}] Sending query: {search_query}")
            response = requests.request("POST", url, headers=headers, data=payload)
            print(f"Status code: {response.status_code}")
            if response.status_code != 200:
                print(f"Response text: {response.text}")
                continue
            
            results = json.loads(response.text)
            # print("Raw search results received:")
            # print(json.dumps(results, indent=2)[:1000])  # 限制输出长度
            
            arxiv_id_list = []
            for i, paper in enumerate(results.get('organic', [])):
                if len(arxiv_id_list) >= num:
                    break  # 严格控制最大提取量

                link = paper.get("link", "")
                print(f"[{i+1}] Checking link: {link}")
                match = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', link)
                if match:
                    arxiv_id = match.group(1)
                    arxiv_id_list.append(arxiv_id)
                    print(f"Matched arXiv ID: {arxiv_id}")
                else:
                    print("No arXiv ID matched in this link.")

            print(f"Total matched arXiv IDs: {len(arxiv_id_list)}")
            return list(set(arxiv_id_list))
        except Exception as e:
            warnings.warn(f"google search failed, query: {query}, error: {e}")
            continue

    print("No results found after 3 attempts.")
    return []

def extract_questions_from_file(input_json_path):
    question_list = []
    with open(input_json_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if 'question' in item:
                question_list.append(item['question'])
    print(f"提取到 {len(question_list)} 条问题")
    return question_list


def load_existing_results(output_path):
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"检测到 {output_path} 损坏，将重建")
    return {}

def process_question_paper_pair(q, arxiv_id, paper_data, output_path, model_name, output_llm_result_path):
    # 下载论文
    if arxiv_id not in paper_data:
        try:
            paper_id, metadata = process_one(arxiv_id)
            paper_data[paper_id] = metadata
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)
            print(f"已下载并保存: {paper_id}")
        except Exception as e:
            print(f"下载失败 {arxiv_id}：{e}")
            return

    metadata = paper_data.get(arxiv_id)
    if not metadata:
        print(f"缺少元数据，跳过: {arxiv_id}")
        return

    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")
    sections = metadata.get("section_titles", [])
    sections_str = json.dumps(sections, ensure_ascii=False)

    user_prompt = (
        f"You are conducting research on the topic:\n"
        f"`{q}`\n\n"
        "You are provided with the title, abstract, and section titles of a research paper.\n\n"
        "Your task:\n"
        "- Identify which sections are most likely to contain citations relevant to this topic.\n"
        "- For each relevant section, output in the format: [Expand]section_name[StopExpand]\n"
        "- If no section is relevant, output exactly: [Expand][StopExpand]\n\n"
        "❗ Do NOT output anything else.\n"
        "❗ Do NOT explain your reasoning.\n"
        "❗ Do NOT output any JSON, comments, or extra text.\n"
        "❗ Output must contain only one or more lines of the form: [Expand]...section name...[StopExpand]\n\n"
        "=== Paper Information ===\n"
        f"Title: {title}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Sections:\n{sections_str}\n"
    )

    write_prompt = (
        f"You are conducting research on `{q}`. "
        "You need to predict which sections to look at for getting more relevant papers. "
        f"Title: {title}\n"
        f"Abstract:  {abstract}\n"
        f"Sections: {sections_str}"
    )

    try:
        response = llm_call(user_prompt, model_name=model_name)

        # 写入 JSONL（追加）
        with open(output_llm_result_path, 'a', encoding='utf-8') as f:
            json.dump({
                "messages": [
                    {"role": "user", "content": write_prompt},
                    {"role": "assistant", "content": response}
                ]
            }, f, ensure_ascii=False)
            f.write("\n")

        print(f"已处理并保存 LLM 评估: {arxiv_id}")
    except Exception as e:
        print(f"LLM 处理失败 {arxiv_id}: {e}")

from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    input_json_path = "result/dataset/AQS/math_ids_ASQ_5000_qwen-72b.jsonl"
    output_path = "result/dataset/sft_crawler/combined_arxiv_papers.json"
    output_llm_result_path = "result/dataset/sft_crawler/math_ids_sft_crawler_expand_5000_qwen-72b.jsonl.jsonl"
    end_date = "20250101"
    max_per_query = 5
    max_workers = 10
    model_name = "qwen-turbo"

    questions = extract_questions_from_file(input_json_path)
    paper_data = load_existing_results(output_path)

    print(f"当前缓存论文数：{len(paper_data)}")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, q in enumerate(questions):
            print(f"\n问题 [{idx + 1}/{len(questions)}]: {q}")
            arxiv_ids = google_search_arxiv_id(q, num=max_per_query, end_date=end_date)

            for arxiv_id in arxiv_ids:
                task = executor.submit(
                    process_question_paper_pair,
                    q,
                    arxiv_id,
                    paper_data,
                    output_path,
                    model_name,
                    output_llm_result_path
                )
                tasks.append(task)

        # 等待所有任务完成
        for future in as_completed(tasks):
            _ = future.result()
