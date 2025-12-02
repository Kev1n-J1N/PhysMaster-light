import json
import openai
import concurrent.futures
import os
from extract_section import *
from llm_call import *
def process_question(index: int, total: int, entry: dict, model_name: str = "qwen-72b"):
    question = entry.get("question", "").strip()
    if not question:
        return None

    print(f"[{index + 1}/{total}] Processing question: {question[:60]}...")

    user_prompt = (
        "You are an elite researcher in the field of AI. You are conducting research on the following topic: \n"
        f"{question}\n\n"
        "Please generate a list of mutually exclusive queries to retrieve highly relevant papers. "
        "Searching for survey papers is encouraged.\n\n"
        "You must strictly output only a list of search queries in the format below, without any explanations or additional text:\n"
        "[Search]...\n"
        "[Search]...\n"
        "[Search]...\n"
        "[Search]...\n"
        "[StopSearch]\n\n"
        "Example:\n"
        "User Query: What studies initiated variational inference techniques for graphical models?\n"
        "[Search]Foundational papers on variational inference for graphical models\n"
        "[Search]Survey papers on variational inference techniques in graphical models\n"
        "[Search]Initiation of variational inference in graphical models research\n"
        "[Search]Chronological progression of variational inference techniques in graphical models\n"
        "[StopSearch]\n\n"
        "Now, based on the User Query above, generate the list:"
    )
    write_prompt = (
        "You are an elite researcher in the field of AI. You are conducting research on the following topic: "
        f"{question}\n\n"
        "Please generate a list of mutually exclusive queries to retrieve highly relevant papers. "
        "Searching for survey papers is encouraged.\n\n"
    )
    try:
        assistant_response = llm_call(user_prompt, model_name=model_name)
        return {
            "messages": [
                {"role": "user", "content": write_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    except Exception as e:
        print(f"Error with question [{index + 1}]: {e}")
        return None

def generate_search_conversations_parallel(input_jsonl, output_jsonl, model_name="gpt-4o", max_workers=10):
    # 读取输入数据
    with open(input_jsonl, 'r', encoding='utf-8') as infile:
        items = [json.loads(line.strip()) for line in infile if "question" in line]

    total = len(items)

    # 检查已处理数量（断点续跑）
    processed_count = 0
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as outf:
            processed_count = sum(1 for _ in outf)

    if processed_count >= total:
        print(f"All {total} entries already processed. Nothing to do.")
        return

    print(f"⚙️  Resuming from item [{processed_count + 1}/{total}] ...")

    remaining_items = items[processed_count:]

    with open(output_jsonl, 'a', encoding='utf-8') as outfile:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_question, processed_count + idx, total, entry, model_name)
                for idx, entry in enumerate(remaining_items)
            ]

            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outfile.flush()

    print(f"\nDone. {total} questions processed. Output saved to {output_jsonl}")

################## 下面是提取title、abstract、sections的代码 ##################
from concurrent.futures import ThreadPoolExecutor, as_completed
def get_html_by_arxiv_id(arxiv_id):
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch HTML for {arxiv_id}")
    return resp.text

def parse_html_no_citation(html_file):
    soup = bs4.BeautifulSoup(html_file, "lxml")

    # parse title
    title = soup.head.title.get_text().replace("\n", " ") if soup.head and soup.head.title else ""

    # parse abstract
    abstract_tag = soup.find(class_='ltx_abstract')
    abstract = abstract_tag.get_text(separator="\n", strip=True) if abstract_tag else ""

    # parse introduction section
    introduction = extract_section_by_title(soup, "Introduction")
    #print("introduction:", introduction)

    # generate and process other sections
    sections = generate_full_toc(soup)
    sections = remove_stop_word_sections_and_extract_text(sections, soup)

    document = {
        "title": title, 
        "abstract": abstract, 
        "introduction": introduction,
        "sections": sections, 
    }
    return document

def process_one(paper_id):
    """
    拉取并解析单篇论文，返回 (paper_id, item_dict or None)
    """
    try:
        html = get_html_by_arxiv_id(paper_id)
        doc = parse_html_no_citation(html)
        raw_title = doc.get("title", "")
        clean_title = re.sub(r"^\[\d{4}\.\d{5}\]\s*", "", raw_title)
        
        # get_2nd_section 返回 { section_title: section_text, ... }
        secs = get_2nd_section(doc["sections"][0]["subsections"])
        section_titles = list(secs.keys())

        return paper_id, {
            "title": clean_title,
            "abstract": doc.get("abstract", ""),
            "section_titles": section_titles
        }
    except Exception as e:
        warnings.warn(f"Failed on {paper_id}: {e}")
        return paper_id, None

def process_arxiv_ids(input_path, output_path, max_count=None, max_workers=8):
    """
    并行处理 arXiv ID 列表，提取 title, abstract, section_titles，
    支持断点续跑，总共最多处理 max_count 个 ID。
    """
    # 读取待处理 ID
    with open(input_path, 'r', encoding='utf-8') as f:
        paper_list = json.load(f)

    # 读取已有结果
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        result = {}

    processed_ids = set(result.keys())
    unprocessed_ids = [pid for pid in paper_list if pid not in processed_ids]

    total_target = max_count if max_count is not None else len(paper_list)
    remaining = max(0, total_target - len(result))
    ids_to_process = unprocessed_ids[:remaining]

    print(f"Already processed: {len(result)} entries.")
    print(f"Will now process: {len(ids_to_process)} more (target total: {total_target}).\n")

    # 并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_id = {executor.submit(process_one, pid): pid for pid in ids_to_process}

        for idx, future in enumerate(as_completed(future_to_id), start=1):
            pid, item = future.result()
            overall_index = len(processed_ids) + idx
            print(f"[{overall_index}/{total_target}] Completed {pid}")

            # 更新结果字典
            result[pid] = item

            # 实时保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nAll done. Total entries now: {len(result)}. Saved to {output_path}")

# 主执行入口
if __name__ == "__main__":
    model_name = "qwen-72b"  # 可以根据需要修改为 "gpt-4o"
    mode="search"
    max_count = 10000
    generate_search_conversations_parallel(
        input_jsonl=f"../create_dataset/result/dataset/AQS/math_ids_ASQ_{max_count}_qwen-72b.json",
        output_jsonl=f"../create_dataset/result/dataset/sft_crawler/math_ids_sft_crawler_{mode}_{max_count}_{model_name}.json",
        model_name=model_name,
        max_workers=20
    )
    # max_count = 10000
    # section_title="abstract"
    # process_arxiv_ids(
    # input_path="../create_dataset/result/train_data/math_ids.json",
    # output_path=f"../create_dataset/result/dataset/sft_crawler/math_ids_allmetatdata_{max_count}.json",
    # max_count=max_count,
    # max_workers=20
    # )