import json
import time
import warnings
import requests
import re
from bs4 import BeautifulSoup
from arxiv_utils import *

def extract_first_n_pages_text(soup, n=2):
    # 假设每页内容被 <div class="ltx_page"> 
    pages = soup.find_all("div", class_="ltx_page")
    text = ""
    count = 0
    for page in pages:
        text += page.get_text(separator="\n", strip=True) + "\n"
        count += 1
        if count >= n:
            break
    # 若未找到分页，则前若干段落
    if not text.strip():
        paras = soup.find_all(['p', 'div'])
        for para in paras[:30]:  # 取前30个段落
            text += para.get_text(separator="\n", strip=True) + "\n"
    return text.strip()

def parse_html_for_section(html_text, section_title):
    soup = BeautifulSoup(html_text, "lxml")
    section = extract_section_by_title(soup, section_title)
    if section and section.strip():
        return section
    # 未找到分节，则取前两页内容
    return extract_first_n_pages_text(soup, n=2)

def get_section_by_arxiv_id(entry_id, section_title):
    try:
        assert re.match(r'^\d+\.\d+$', entry_id)
        url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            warnings.warn(f"Failed to retrieve {entry_id}, status code: {response.status_code}")
            return ""
        html_content = response.text
        section_text = parse_html_for_section(html_content, section_title)
        return section_text.strip()
    except Exception as e:
        warnings.warn(f"Error processing {entry_id}: {e}")
        return ""

def process_arxiv_ids_for_section(input_path, output_path, section_title="Introduction", max_count=10000, sleep_time=0):
    with open(input_path, 'r', encoding='utf-8') as f:
        arxiv_ids = json.load(f)

    result = {}
    for idx, arxiv_id in enumerate(arxiv_ids[:max_count]):
        print(f"[{idx+1}/{min(max_count, len(arxiv_ids))}] Processing {arxiv_id} ...")
        section_text = get_section_by_arxiv_id(arxiv_id, section_title)
        result[arxiv_id] = section_text
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Finished saving to {output_path}")

import json

def merge_intro_with_titles(intro_path, reference_path, output_path):
    import json

    with open(intro_path, 'r', encoding='utf-8') as f:
        intro_data = json.load(f)

    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)

    result = {}

    for arxiv_id, intro_text in intro_data.items():
        titles = []
        ref_entry = reference_data.get(arxiv_id)

        if not isinstance(ref_entry, dict):
            continue

        for section_name, content in ref_entry.items():
            if "introduction" in section_name.lower():
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "title" in item:
                            titles.append(item["title"])

        print(f"Processing {arxiv_id} - Found {len(titles)} titles")

        if intro_text.strip() and titles:
            result[arxiv_id] = {
                "text": intro_text,
                "titles": titles
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"完成！共提取 {len(result)} 篇论文的引文标题。结果写入：{output_path}")


if __name__ == "__main__":
    # max_count = 10000
    # process_arxiv_ids_for_section(
    # input_path="../create_dataset/result/train_data/math_ids.json",
    # output_path=f"../create_dataset/result/dataset/math_ids_introduction_{max_count}.json",
    # section_title="Introduction",  # 可以换成“Related Work”等
    # max_count=max_count,)
# 示例调用
    max_count = 10000
    merge_intro_with_titles(
        intro_path=f"../create_dataset/result/dataset/math_ids_introduction_{max_count}.json",
        reference_path=f"../create_dataset/result/dataset/math_ids_metadata_{max_count}.json",
        output_path=f"../create_dataset/result/dataset/math_ids_combined_output_{max_count}.json",
    )
