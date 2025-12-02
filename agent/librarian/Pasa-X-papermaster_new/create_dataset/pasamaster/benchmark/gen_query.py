import sys,os
import threading,json
from concurrent.futures import ThreadPoolExecutor, as_completed
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_dir}/../../..")
from tools.llm_call import llm_call

from utils import search_abs_by_id,search_content_by_arxiv_id

prompt_general = "You're a senior expert in scientific research. Based on the text provided below, generate several literature-search queries (they should look like what a researcher would actually type into Google Scholar/Scopus, e.g. \"works on …\", \"papers that investigate …\", \"studies about …\", \"recommend related work on …\", \"give me papers that …\"), not open-ended questions. Write the queries in a way that mimics how a real researcher would phrase a literature search request—natural, concise, and retrieval-oriented. Do not mention specific paper titles or author names; refer only to the general technical ideas. For example: \"papers that use AI agents with tool calling to achieve performance gains on HLE tasks.\" Follow these requirements:\n\n1. Each query should have a clear and narrow focus so that a retrieval system can target the specific technical idea or setup described in the text.\n2. The queries may be reasonably specific and include some details; they should not be overly broad, high-level topic queries.\n3. The difficulty should be relatively high: ideally, each query should retrieve a small set of truly relevant papers (around 10), rather than hundreds of loosely related results.\n4. At the same time, do not make the query so narrow that it matches only a single paper.\n5. Write the queries in a style typical for academic search (you may include method names, task definitions, data types, constraints, etc.), and vary the phrasing across queries (e.g. \"papers on …\", \"studies that propose …\", \"work that combines …\", \"recommend related work on …\", \"give me papers that …\").\n6. Your final output must be in JSON format like this:\n {{\n \"queries\": [\n {{\"query\": \"...\", \"checklist\": [\"...\", \"...\"],\"evidence\":evidence}},\n {{\"query\": \"...\", \"checklist\": [\"...\", \"...\"],\"evidence\":evidence}}\n ]\n }}\n7. For each query you generate, also generate a corresponding checklist. The checklist describes the concrete aspects that must be checked to determine whether a paper actually satisfies this query (for example: problem setting, model type, supervision signal, data/modality, evaluation setup, constraints/assumptions). Evidence is the original text from which you generated this query. It must directly show the corresponding fragments from the original text. If you referenced multiple sources, use \n to separate them. The article content you can refer to is {content}."

prompt_description =  "You are a senior expert in scientific research and technical literature. Based on the input text provided below, generate a set of advanced, descriptive literature search queries that a researcher would realistically type into Google Scholar, Scopus, or similar platforms.\n\nEach query must follow this format:\n- Begin with a natural-language, scenario-style description of the problem or research challenge (e.g., “I'm working on...” or “I've found that...”).\n- Then pose a research-oriented search query, asking for papers or studies relevant to that issue.\n\nStrictly follow these requirements:\n\n1. Each query must focus on a specific technical setup or problem, not a general topic.\n2. Queries should be moderately specific — detailed enough to filter results to a manageable number (~10 highly relevant papers), but not so narrow that they return only one or two papers.\n3. Use phrasing common in academic literature searches, including technical terms like model types, task formulations, data modalities, evaluation setups, constraints, and dataset names.\n4. Avoid repeating query structures — vary the sentence construction and research framing.\n5. For each query, also include a corresponding checklist — a list of key elements to check when determining whether a paper satisfies the query (e.g., problem setting, data modality, supervision method, assumptions, etc.).\n\nOutput format (strictly JSON):\n\n{{\"queries\": [{{\"query\": \"I'm working on [descriptive problem statement]. Are there any studies that [describe desired solution]?\",\"checklist\": [\"problem setting\", \"methodology\", \"data used\", \"constraints\", \"evaluation\"]}},{{\"query\": \"I've encountered [technical issue or limitation] in my research. Can you find papers that [suggest resolution or method]?\",\"checklist\": [\"issue addressed\", \"approach taken\", \"domain specificity\", \"data type\", \"performance metrics\"]}}]}}\n\nContent to refer to: {content}"

def call_llm(prompt, max_retries=3, sleep_sec=1.0):
    """
    调 LLM 拿到 {"queries": [...]}，失败最多重试 max_retries 次。
    成功返回 queries 列表，失败最后一次还是解析不了就抛错。
    """

    for attempt in range(1, max_retries + 1):
        try:
            # 你原来的调用
            response = llm_call(prompt, "glm")

            # 尝试从响应里截取 json
            json_begin = response.find('{')
            json_end = response.rfind('}')
            if json_begin == -1 or json_end == -1:
                raise ValueError("no JSON braces found in response")

            json_str = response[json_begin:json_end+1]
            response_dict = json.loads(json_str)

            if "queries" not in response_dict:
                raise KeyError("key 'queries' not found in response JSON")

            return response_dict["queries"]

        except Exception as e:
            last_err = e
            print(f"[call_llm] attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(sleep_sec)
        return []


def gen_query(arxiv_id,prompt):
    final_queries = []
    # abstract = search_abs_by_id(arxiv_id)['abstract']
    try:
        content = search_content_by_arxiv_id(arxiv_id)['sections']
        intro_content = ""
        related_work_content = ""
        for key,section in content.items():
            if 'introduction' in key.lower():
                intro_content += section
            elif'related work' in key.lower():
                related_work_content += section
        if intro_content != "":
            final_prompt = prompt.format(content=intro_content)
            queries = call_llm(final_prompt)
            final_queries.extend(queries)
        if related_work_content != "":
            final_prompt = prompt.format(content=related_work_content)
            queries = call_llm(final_prompt)
            final_queries.extend(queries)
    except:
        pass
    return final_queries

if __name__ == '__main__':
    # queries = gen_query("2501.10120",prompt_general)
    # queries = gen_query("2501.10120",prompt_description)
    BASE_DIR = "/data/Tianjin/pasamaster_bench/paper_db/tex"
    OUTPUT_JSON = f"{current_dir}/data/queries_icml_general.json"

    all_queries = {}
    final_success = 0
    for item in sorted(os.listdir(BASE_DIR)):
        arxiv_id = item

        try:
            queries = gen_query(arxiv_id, prompt_general)
            if len(queries) > 0:
                all_queries[arxiv_id] = queries
                print(f"[OK] {arxiv_id}")
                final_success += 1
            else:
                raise("return []")
        except Exception as e:
            print(f"[ERROR] {arxiv_id}: {e}")

    print(f"成功处理{final_success}，总文章：{len(os.listdir(BASE_DIR))}")
    with open("w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)

    print(f"done, saved to: {OUTPUT_JSON}") 
