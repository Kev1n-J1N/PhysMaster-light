import json
import openai
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_call import *

lock = threading.Lock()
qid_index = 0
def load_existing_qids(path):
    existing = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = obj.get("arxiv_id", "") + obj.get("question", "")
                    existing.add(key)
                except:
                    continue
    return existing

def get_existing_qid_index(path):
    """Returns max qid index already used"""
    max_id = -1
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    qid = obj.get("qid", "")
                    if qid and qid.startswith("AutoScholarQuery_train_"):
                        idx = int(qid.split("_")[-1])
                        max_id = max(max_id, idx)
                except:
                    continue
    return max_id + 1

def build_ASQ_parallel(input_json_path, output_jsonl_path, max_count=None, max_total_qa=None, model_name="qwen-72b", num_workers=5):
    global qid_index
    qid_index = get_existing_qid_index(output_jsonl_path)
    print(f"Starting from qid index: {qid_index}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    existing = load_existing_qids(output_jsonl_path)
    print(f"Found {len(existing)} existing QA pairs. Skipping duplicates...")

    prompt_header = (
        "You are provided a ‘Introduction' section of a research paper. The researcher reviewed the relevant work, "
        "conducted a literature survey, and cited corresponding references in this text. "
        "Can you guess what research questions the researcher might have posed when preparing this text? "
        "The answers to these questions should be the references cited in this passage. "
        "Please list questions and provide the corresponding answers.\n\n"
        "[Requirements:]\n"
        "1. Craft questions similar to those a researcher would pose when reviewing related works, such as “Which paper studied ...?”, “Any works about...?”, “Could you provide me some works...?”\n"
        "2. Construct the question-answer pairs based on [Section from A Research Paper]. The answer should be the cited papers in [Section from A Research Paper].\n"
        "3. Do not ask questions including \"or\" or \"and\" that may involve more than one condition.\n"
        "4. Clarity: Formulate questions clearly and unambiguously to prevent confusion.\n"
        "5. Contextual Definitions: Include explanations or definitions for specialized terms and concepts used in the questions.\n"
        "6. A question should be broader and correspond to multiple articles. However, if the text is too long, more than one question can be asked.\n\n"
        "Here are some examples:\n"
        "[Begin of examples]\n"
        "{\"arxiv_id\":\"11111\", \"question\": \"What works are related to the field of image retrieval?\", \"answer\": [\"UNITER: UNiversal Image-TExt Representation Learning\"], \"qid\": \"AutoScholarQuery_train_0\"}\n"
        "{\"arxiv_id\":\"11111\", \"question\": \"Could you provide me some works employs image patches and superpixels in region-based methods for semantic segmentation?\", \"answer\": [\"CEREALS – Cost-Effective REgion-based Active Learning for Semantic Segmentation\", \"Reinforced Active Learning for Image Segmentation\", \"MetaBox+: A new Region Based Active Learning Method for Semantic Segmentation using Priority Maps\", \"ViewAL: Active Learning With Viewpoint Entropy for Semantic Segmentation\"], \"qid\": \"AutoScholarQuery_train_1\"}\n"
        "[End of examples]\n\n"
    )

    items = list(data.items())
    if max_count:
        items = items[:max_count]

    def process_entry(index, arxiv_id, content):
        global qid_index
        intro_text = content.get("text", "").strip()
        titles = content.get("titles", [])
        if not intro_text or not titles:
            return []

        title_block = "\n".join(f"- {t}" for t in titles)
        prompt = (
            prompt_header +
            f"Here is the text of the introduction part of the article and the citation titles contained therein:\n\n"
            f"[text]:\n{intro_text}\n\n"
            f"[citation titles]:\n{title_block}\n\n"
            "{OUTPUT}:"
        )

        print(f"[{index + 1}/{len(items)}] Querying arXiv:{arxiv_id}...")

        try:
            raw_output = llm_call(prompt, model_name=model_name)
            local_results = []
            for line in raw_output.splitlines():
                line = line.strip()
                if line.startswith("{") and "question" in line:
                    try:
                        entry = json.loads(line.replace("“", "\"").replace("”", "\""))
                        unique_key = arxiv_id + entry["question"]
                        if unique_key in existing:
                            continue
                        with lock:
                            entry["arxiv_id"] = arxiv_id
                            entry["qid"] = f"AutoScholarQuery_train_{qid_index}"
                            qid_index += 1
                            existing.add(unique_key)
                        local_results.append(entry)

                        if max_total_qa is not None and qid_index >= max_total_qa:
                            return local_results
                    except Exception as e:
                        print(f"JSON parse error: {e}")
            return local_results
        except Exception as e:
            print(f"Error calling LLM for arXiv:{arxiv_id}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_entry, idx, arxiv_id, content): (arxiv_id, content)
            for idx, (arxiv_id, content) in enumerate(items)
        }

        for future in as_completed(futures):
            qa_entries = future.result()
            if qa_entries:
                with lock:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
                        for entry in qa_entries:
                            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if max_total_qa is not None and qid_index >= max_total_qa:
                        print(f"\n✅ Reached QA limit {max_total_qa}, stopping early.")
                        break

    print(f"\n✅ Done. Total QA pairs written: {qid_index}. Output saved to {output_jsonl_path}")

if __name__ == "__main__":
    max_count = 10000  # 设置最大处理数量
    model_name= "qwen-72b"  # 使用的模型名称
    build_ASQ_parallel(
        input_json_path=f"../create_dataset/result/dataset/math_ids_combined_output_{max_count}.json",
        output_jsonl_path=f"../create_dataset/result/dataset/AQS/math_ids_ASQ_{max_count}_{model_name}.json",
        max_count=None,
        max_total_qa=None,
        model_name=model_name,
        num_workers=20  # 设置并发线程数
    )
