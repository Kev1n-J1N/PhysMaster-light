import fitz  # PyMuPDF
import re,sys
import json,openai
from collections import defaultdict, OrderedDict
from pdfminer.high_level import extract_text
import os
from concurrent.futures import ThreadPoolExecutor
import time
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../..')
from tools.web_search import title_search_rag
from rapidfuzz import fuzz, process
import json
import tqdm

from typing import List, Tuple
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from loguru import logger

with open(f"{current_path}/../../config/config.json", "r") as f:
    config = json.load(f)
model_config = config["model_config"]
with open(f"{current_path}/../../papermaster_prompt.json", "r") as f:
    prompts = json.load(f)

def embedding_retrieval(queries, candidates, threshold=80):
    """
    在 title_list 中找到所有匹配的 candidates，使用模糊匹配来判断是否相似。
    """
    try:
        candidates = candidates
        begin = time.time()
        title_list = title_search_rag(queries, 10000)
        for title in title_list:
            title = title.replace("\n", " ").strip()
        end = time.time()
        print(f"🔍 [Embedding Retrieval] 标题搜索耗时: {end-begin:.2f}秒")
        matches = []
        begin = time.time()
        for candidate in tqdm.tqdm(candidates):
            cleaned_candidate = candidate.replace("\n", " ").strip()
            best_match, score,idx = process.extractOne(cleaned_candidate, title_list, scorer=fuzz.ratio)
            if score >= threshold:
                if title_list[idx] not in matches:
                    matches.append(title_list[idx])
        end = time.time()
        print(f"🔍 [Embedding Retrieval] 模糊匹配耗时: {end-begin:.2f}秒")
    except Exception as e:
        logger.error(f"rag 检索失败,{e}")
        matches = []
    return matches

def llm_call(query: str, model_name: str = "qwen-72b"):
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

def llm_batch_call(queries: list, batch_size = 100, model_name: str = "qwen-72b"):
    batch_size = min(batch_size, len(queries))
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        responses = list(executor.map(lambda query: client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": query}]
        ), queries))
    results = [choice.message.content for response in responses for choice in response.choices]
    return results

def extract_title(refs):
    prompt_list = []
    prompt = prompts['extract_title']
    for ref in refs:
        prompt_list.append(prompt.format(ref_info=ref))
    results = llm_batch_call(prompt_list)
    title_list = []
    for result in results:
        begin = result.find('<title>')
        end = result.find('</title>')
        title_list.append(result[begin+7:end])
    return title_list

def find_paper_path(arxiv_id, base_path=config['arxiv_database_path']):
    """
    根据给定的 arxiv_id，先根据 ID 的前四位找到对应目录，然后查找该目录下的 PDF 文件路径。
    :param arxiv_id: arxiv paper id (如 "0704.0001")
    :param base_path: 存储 PDF 文件的基础路径
    :return: 对应的 PDF 文件路径，若未找到则返回 None
    """
    base_path = os.path.join(base_path, 'arxiv/pdf')  
    folder_name = arxiv_id[:4]  
    
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found.")
        return None
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_arxiv_id = file.split('v')[0]  
            if file_arxiv_id == arxiv_id:
                print(f"Found PDF for {arxiv_id}: {os.path.join(root, file)}")
                return os.path.join(root, file)
            
    print(f"PDF for {arxiv_id} not found in {folder_path}.")
    return None

# ========= 预处理：标准化括号/分页/页码 =========
def normalize_text(s: str) -> str:
    s = s.translate(str.maketrans({
        "℄": "]",   # U+2104
        "⁆": "]",   # U+2046
        "⟧": "]",   # U+27E7
        "〗": "]",   # U+3017
    }))
    s = s.replace("\x0c", "\n")
    s = re.sub(r"(?m)^\s*\d+\s*$", "", s)
    return s

# ========= 正则：参考文献条目、文内引用 =========
HEADING_RE = re.compile(r"^(?P<num>(?:\d+(?:\.\d+)*))\s+(?P<title>.+?)\s*$", re.MULTILINE)
REF_BLOCK_RE = re.compile(r"(?m)^\s*\[(?P<idx>\d+)\](?P<body>.*?)(?=\n\[\d+\]|\Z)", re.DOTALL)
CITE_RE = re.compile(r"\[(?P<cites>\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\]")

def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def build_reference_map(refs: list):
    """
    构建参考文献映射，将每个参考文献的序号映射到其标题，去重并清理无关信息。
    :param refs_text: 包含参考文献的文本
    :return: 参考文献序号到标题的映射字典
    """
    ref_map = {}
    if len(refs) == 0:
        return ref_map
    
    # 提取每篇参考文献的标题（假设标题在正文的第一部分）
    title_list = extract_title(refs)

    # 将序号和标题建立映射
    idx = 1
    for title in title_list:
        if "</None>" in title:
           continue
        ref_map[str(idx)] = title
        idx += 1

    return ref_map

import re

def find_references_block(full_text: str) -> str:
    # 使用正则表达式查找所有以 \n[数字] 开头并且以 \n[数字] 结尾的句子
    pattern = r"(\n\[\d+\].*?)(?=\n\[\d+\]|\Z)"  # 找到一个句子的结束，并确保接下来的句子开头为 \n[数字] 或文本结束
    
    # 找到所有匹配的句子
    matches = re.findall(pattern, full_text, re.DOTALL)
    
    result = []
    
    for match in matches:
        if len(match) > 500:
            truncated = match[:500]
            last_period_pos = truncated.rfind('.')
            if last_period_pos != -1:
                truncated = truncated[:last_period_pos+1]  # 包含句号
            result.append(truncated)
        else:
            result.append(match)
    
    
    return result


def extract_references(refs: dict, query: str) -> OrderedDict:
    prompt = f"""
   You are a literature recommendation expert. You will be given a list of references, each with a numbered ID. Based on the titles of these references, your task is to determine which papers are relevant to the user's question.

Please follow these instructions:
1. Review the titles of the references and compare them with the user's question.
2. For each relevant reference, explain why it is relevant in up to 150 words.
3. Return the IDs of all relevant references in the format: <id> id1,id2,id3 </id>, where each id corresponds to the ID of a reference.
Note: Include any papers that you think could potentially be relevant. Do not overlook any paper that may have a connection to the user's question. 
The list of references is:
{str(refs)}

The user's question is:
{query}

Please return your results as described above. Only return the IDs of the relevant references, it must more than 10 references.
    """
    result = llm_call(prompt)
    print(result)
    json_start = result.find('<id>')
    json_end = result.find('</id>')
    result = result[json_start+4:json_end]
    result = result.split(',')
    extracted_refs = []
    for res in result:
        i = res.strip()
        extracted_refs.append(refs[i])
    return extracted_refs

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    
    :param pdf_path: Path to the PDF file
    :return: Extracted text from the PDF
    """
    doc = fitz.open(pdf_path)
    
    full_text = ""
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        full_text += page.get_text("text")  # Extract text from the page
    
    return full_text

def search_info_by_local_db(arxiv_id):
    pdf_path = find_paper_path(arxiv_id)
    raw_text = extract_text_from_pdf(pdf_path)
    full_text = normalize_text(raw_text)
    refs_text = find_references_block(full_text)
    ref_map = build_reference_map(refs_text)
    query = ["Nerf",'3dgs','diffusion']
    # extract_ref = extract_references(ref_map, query)
    ref_list = []
    for ref in ref_map.values():
        ref_list.append(ref)
    
    extract_ref = embedding_retrieval(query,ref_list)
    return extract_ref
    
# ========= 使用示例 =========
if __name__ == "__main__":
    # import time
    begin = time.time()
    arxiv_id = '2503.10410'  
    citations = search_info_by_local_db(arxiv_id)
    print(citations)
    end = time.time()
    print(f"Time cost: {end-begin:.2f}s")
    # print(llm_call("hello"))




