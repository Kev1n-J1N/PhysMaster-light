# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Please note that:
1. You need to first apply for a Google Search API key at https://serpapi.com/,
   and replace the 'your google keys' below before you can use it.
2. The service for searching arxiv and obtaining paper contents is relatively simple. 
   If there are any bugs or improvement suggestions, you can submit pull requests.
   We would greatly appreciate and look forward to your contributions!!
"""
import re
import bs4
import json
import arxiv
import urllib
import zipfile
import warnings
import requests
from datetime   import datetime
warnings.simplefilter("always")
import threading
import time
from llm_call import *
GOOGLE_KEY   = 'a5d20c46dc0c1a7a926cb8491ca9e459610d4a02'
arxiv_client = arxiv.Client(delay_seconds = 0.05)
# id2paper     = json.load(open("..\pasa-dataset\paper_database\id2paper.json"))
# paper_db     = zipfile.ZipFile("..\pasa-dataset\paper_database\cs_paper_2nd.zip", "r")

# openai调用模型回答
import openai
def parse_metadata(metas):
    """
    使用阿里百炼模型解析文献元数据为 authors, title, journal。
    """
    # ##并发验证
    # thread_id = threading.get_ident()
    # start_time = time.time()
    # print(f"[Thread {thread_id}] Start parsing at {start_time:.2f} → {metas[:1]}")
    # ###
    metas = [item.replace('\n', ' ') for item in metas]
    meta_string = ' '.join(metas)
    #print(f"元数据字符串: {meta_string}")  # 打印前100个字符以检查

    prompt = f"""
请从以下文献元数据中提取作者（authors）、标题（title）和期刊（journal），只以标准 JSON 格式返回结果，不要多余解释：
"{meta_string}"
要求返回格式为：
{{
  "authors": "...",
  "title": "...",
  "journal": "...",
  "published_date": "..." (格式为 YYYY-MM-DD, 例如 "2023-10-01" 或 "2023-10" 或 "2023" 等)
}}
    """

    try:
        content = llm_call(prompt,model_name="qwen-72b")
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            parsed = {"authors": "", "title": "", "journal": "", "published_date": ""}

    except Exception as e:
        print(f"API调用失败: {e}")
        parsed = {"authors": "", "title": "", "journal": "", "published_date": ""}
        print("失败时的prompt:", prompt)

    return {
        "meta_list": metas,
        "meta_string": meta_string,
        "authors": parsed.get("authors", ""),
        "title": parsed.get("title", ""),
        "journal": parsed.get("journal", ""),
        "published_date": parsed.get("published_date", "")
    }

def create_dict_for_citation(ul_element):
    citation_dict, futures, id_attrs = {}, [], []
    for li in ul_element.find_all("li", recursive=False):
        id_attr = li['id']
        metas = [x.text.strip() for x in li.find_all('span', class_='ltx_bibblock')]
        id_attrs.append(id_attr)
        futures.append(parse_metadata(metas))
    results = list(zip(id_attrs, futures))
    citation_dict = dict(results)
    return citation_dict

##并发版本
import concurrent.futures
def create_dict_for_citation(ul_element, max_workers=20):
    citation_dict = {}
    id_attrs = []
    metas_list = []

    for li in ul_element.find_all("li", recursive=False):
        id_attr = li['id']
        # print(f"Processing {id_attr} ...")
        metas = [x.text.strip() for x in li.find_all('span', class_='ltx_bibblock')]
        id_attrs.append(id_attr)
        metas_list.append(metas)

    # 并发执行 parse_metadata
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(parse_metadata, metas): id_attr
            for metas, id_attr in zip(metas_list, id_attrs)
        }

        for future in concurrent.futures.as_completed(future_to_id):
            id_attr = future_to_id[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"[{id_attr}] 解析失败: {e}")
                result = {
                    "meta_list": [],
                    "meta_string": "",
                    "authors": "",
                    "title": "",
                    "journal": "",
                    "published_data":""
                }
            citation_dict[id_attr] = result

    return citation_dict

def generate_full_toc(soup):
    toc = []
    stack = [(0, toc)]
    
    # Mapping of heading tags to their levels
    heading_tags = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5}
    
    for tag in soup.find_all(heading_tags.keys()):
        level = heading_tags[tag.name]
        title = tag.get_text()
        
        # Ensure the stack has the correct level
        while stack and stack[-1][0] >= level:
            stack.pop()
        
        current_level = stack[-1][1]

        # Find the nearest enclosing section with an id
        section = tag.find_parent('section', id=True)
        section_id = section.get('id') if section else None
        
        # Create the new entry
        new_entry = {'title': title, 'id': section_id, 'subsections': []}
        
        current_level.append(new_entry)
        stack.append((level, new_entry['subsections']))
    
    return toc

def parse_text(local_text, tag):
    ignore_tags = ['a', 'figure', 'center', 'caption', 'td', 'h1', 'h2', 'h3', 'h4']
    # latexmlc
    ignore_tags += ['sup']
    max_math_length = 300000

    for child in tag.children:
        child_type = type(child)
        if child_type == bs4.element.NavigableString:
                txt = child.get_text()
                local_text.append(txt)

        elif child_type == bs4.element.Comment:
            continue
        elif child_type == bs4.element.Tag:

                if child.name in ignore_tags or (child.has_attr('class') and child['class'][0] == 'navigation'):
                    continue
                elif child.name == 'cite':
                    # add hrefs
                    hrefs = [a.get('href').strip('#') for a in child.find_all('a', class_='ltx_ref')]
                    local_text.append('~\cite{' + ', '.join(hrefs) + '}')
                elif child.name == 'img' and child.has_attr('alt'):
                    math_txt = child.get('alt')
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.has_attr('class') and (child['class'][0] == 'ltx_Math' or child['class'][0] == 'ltx_equation'):
                    math_txt = child.get_text()
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.name == 'section':
                    return
                else:
                    parse_text(local_text, child)
        else:
            raise RuntimeError('Unhandled type')

def clean_text(text):
    delete_items = ['=-1', '\t', u'\xa0', '[]', '()', 'mathbb', 'mathcal', 'bm', 'mathrm', 'mathit', 'mathbf', 'mathbfcal', 'textbf', 'textsc', 'langle', 'rangle', 'mathbin']
    for item in delete_items:
        text = text.replace(item, '')
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[[,]+]', '', text)
    text = re.sub(r'\.(?!\d)', '. ', text)
    text = re.sub('bib. bib', 'bib.bib', text)
    return text

def remove_stop_word_sections_and_extract_text(toc, soup, stop_words=['references', 'acknowledgments', 'about this document', 'apopendix']):
    def has_stop_word(title, stop_words):
        return any(stop_word.lower() in title.lower() for stop_word in stop_words)
    
    def extract_text(entry, soup):
        section_id = entry['id']
        if section_id: # section_id
            section = soup.find(id=section_id)
            if section is not None:
                local_text = []
                parse_text(local_text, section)
                if local_text:
                    processed_text = clean_text(''.join(local_text))
                    entry['text'] = processed_text
        return 0 
    
    def filter_and_update_toc(entries):
        filtered_entries = []
        for entry in entries:
            if not has_stop_word(entry['title'], stop_words):
                # Get clean text
                extract_text(entry, soup)                
                entry['subsections'] = filter_and_update_toc(entry['subsections'])
                filtered_entries.append(entry)
        return filtered_entries
    
    return filter_and_update_toc(toc)

def parse_html(html_file):
    soup = bs4.BeautifulSoup(html_file, "lxml")

    # parse title
    title = soup.head.title.get_text().replace("\n", " ") if soup.head and soup.head.title else ""

    # parse abstract
    abstract_tag = soup.find(class_='ltx_abstract')
    abstract = abstract_tag.get_text(separator="\n", strip=True) if abstract_tag else ""

    # parse introduction section
    introduction = extract_section_by_title(soup, "Introduction")
    #print("introduction:", introduction)

    # parse citation
    citation = soup.find(class_='ltx_biblist')
    citation_dict = create_dict_for_citation(citation)

    # generate and process other sections
    sections = generate_full_toc(soup)
    sections = remove_stop_word_sections_and_extract_text(sections, soup)

    document = {
        "title": title, 
        "abstract": abstract, 
        "introduction": introduction,
        "sections": sections, 
        "references": citation_dict,
    }
    return document


def extract_section_by_title(soup, main_keyword, fallback_keyword="Overview"):
    """
    从HTML中提取包含main_keyword的section内容，
    如果没找到，尝试用fallback_keyword（如 Overview）找子section
    """
    # 先尝试找主标题匹配的section
    sections = soup.find_all("section", class_="ltx_section")
    
    def get_clean_title(header_tag):
        # 去掉span标签，只取纯文本作为标题判断
        if not header_tag:
            return ""
        # clone header内容，去掉所有span
        for span in header_tag.find_all("span"):
            span.extract()
        return header_tag.get_text(strip=True)
    
    # 尝试匹配主关键词的section
    for section in sections:
        # 找主标题：h2带class ltx_title
        header = section.find("h2", class_="ltx_title")
        title_text = get_clean_title(header).lower()
        if main_keyword.lower() in title_text:
            # 抓所有p段落
            paragraphs = section.find_all("p")
            content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if content:
                return content

    # 主关键词找不到，尝试用fallback关键词找子section
    if fallback_keyword:
        # 找所有子section
        subsections = soup.find_all("section", class_="ltx_subsection")
        for subsection in subsections:
            header = subsection.find(["h2", "h3", "h4"], class_="ltx_title")
            title_text = get_clean_title(header).lower()
            if fallback_keyword.lower() in title_text:
                paragraphs = subsection.find_all("p")
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                if content:
                    return content

    # 都找不到就返回空字符串
    return ""



def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def search_section_ref_by_arxiv_id(entry_id, cite_pattern):
    assert re.match(r'^\d+\.\d+$', entry_id)
    url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'

    try:
        response = requests.get(url)
        if response.status_code != 200:
            warnings.warn(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
        
        html_content = response.text
        if 'https://ar5iv.labs.arxiv.org/html' not in html_content:
            warnings.warn(f'Invalid ar5iv HTML document: {url}')
            return None
        
        try:
            document = parse_html(html_content)
        except:
            warnings.warn(f'Wrong format HTML document: {url}')
            return None
        
        try:
            sections = get_2nd_section(document["sections"][0]["subsections"])
        except:
            warnings.warn(f'Get subsections error')
            return None

        sections2refs = {}

        for sec_title, sec_text in sections.items():
            norm_title = " ".join(sec_title.split("\n"))
            matched_refs = set()
            refs_in_section = []

            bibs = re.findall(cite_pattern, sec_text, re.DOTALL)
            for bib in bibs:
                for b in bib.split(","):
                    b = b.strip()
                    if not b or b not in document["references"]:
                        continue
                    ref = document["references"][b]

                    # 构造包含所需信息的引用条目
                    ref_info = {
                        #"arxiv_id": ref.get("arxiv_id", ""),
                        "title": ref.get("title", ""),
                        #"publish_date": ref.get("published_date", "")
                    }

                    # 避免重复
                    ref_tuple = tuple(ref_info.items())
                    if ref_tuple not in matched_refs:
                        matched_refs.add(ref_tuple)
                        refs_in_section.append(ref_info)

            if refs_in_section:
                sections2refs[norm_title] = refs_in_section

        return sections2refs

    except requests.RequestException as e:
        warnings.warn(f"An error occurred: {e}")
        return None

def get_subsection(sections):
    res = {}
    for section in sections:
        if "text" in section and section["text"].strip() != "":
            res[section["title"].strip()] = section["text"].strip()
        subsections = get_subsection(section["subsections"])
        for k, v in subsections.items():
            res[k] = v
    return res

def get_1st_section(sections):
    res = {}
    for section in sections:
        subsections = get_subsection(section["subsections"])
        if "text" in section and section["text"].strip() != "" or len(subsections) > 0:
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
            else:
                res[section["title"].strip()] = ""
            for k, v in subsections.items():
                res[section["title"].strip()] += v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def get_2nd_section(sections):
    res = {}
    for section in sections:
        subsections = get_1st_section(section["subsections"])
        if "text" in section and section["text"].strip() != "":
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
        for k, v in subsections.items():
            res[section["title"].strip() + " " + k.strip()] = v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

import os
import json

def process_arxiv_ids(input_path, output_path, max_count=None):
    """
    从 input_path 加载 arXiv ID 列表，逐条处理，结果保存在 output_path。
    支持断点续跑，总共最多处理 max_count 个 ID。
    """

    with open(input_path, 'r', encoding='utf-8') as f:
        paper_list = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        result = {}

    processed_ids = set(result.keys())
    unprocessed_ids = [pid for pid in paper_list if pid not in processed_ids]

    already_done = len(result)
    total_to_process = max_count if max_count else len(paper_list)
    remaining = max(0, total_to_process - already_done)
    ids_to_process = unprocessed_ids[:remaining]

    print(f"Already done: {already_done}")
    print(f"Will now process {len(ids_to_process)} more (target total: {total_to_process})")

    for i, paper_id in enumerate(ids_to_process):
        print(f"[{already_done + i + 1}/{total_to_process}] Processing: {paper_id}")
        try:
            ref_data = search_section_ref_by_arxiv_id(paper_id, r"~\\cite\{(.*?)\}")
            result[paper_id] = ref_data
        except Exception as e:
            print(f"Failed to process {paper_id}: {e}")
            result[paper_id] = None

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nAll done. Total processed: {len(result)} entries. Output saved to: {output_path}")

if __name__ == "__main__":
    #print(search_section_ref_by_arxiv_id("2501.10120", r"~\\cite\{(.*?)\}"))#2404.03447
    #print(search_paper_by_arxiv_id("2501.10120"))
    #print(search_paper_by_title("A hybrid approach to CMB lensing reconstruction on all-sky intensity maps"))

    ### 提取某章节的原文


    ##### 提取各章节的引用（标题、作者、期刊等）信息
    max_count=10000
    input_path="../create_dataset/result/train_data/math_ids.json"
    output_path=f"../create_dataset/result/dataset/math_ids_metadata_{max_count}.json"
    process_arxiv_ids(input_path, output_path,max_count)
    # process_arxiv_ids(input_path, output_path)