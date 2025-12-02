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
import json
import os
import re
import sys
import time
import unicodedata
import urllib
import uuid
import warnings
import zipfile
import arxiv
import bs4
import requests
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.llm_call import *
from search_utils.call_mtr import CallMetricsTracker
metrics_tracker = CallMetricsTracker()

warnings.simplefilter("always")

arxiv_client = arxiv.Client(delay_seconds = 0.05)
base_path = os.path.dirname(os.path.abspath(__file__))

with open(f"{base_path}/data/paper_database/id2paper.json", "r") as f:
    id2paper  = json.load(f)
paper_db     = zipfile.ZipFile(f"{base_path}/data/paper_database/cs_paper_2nd.zip", "r")

# MODEL: 本地模型调用，解析ref
from model_vllm import Agent
current_path = os.path.dirname(os.path.abspath(__file__))
with open(f'{current_path}/config/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
# qwen_agent = Agent(model_name=config["tool_model"])
# thread_num = 100
with open(f"{base_path}/data/arxiv_dataset/arxiv_idx_aid2pos.json", encoding="utf-8") as f:
    db3_idx_aid2pos = json.load(f)
with open(f"{base_path}/data/arxiv_dataset/arxiv_idx_sim2pos.json", encoding="utf-8") as f:
    db3_idx_sim2pos = json.load(f)
db3_path = f"{base_path}/data/arxiv_dataset/arxiv-metadata-oai-snapshot.json"
db3_file = open(db3_path,encoding="utf-8")

def read_line(pos):
    fd = db3_file.fileno()
    chunk_size = 65536
    buf = bytearray()
    off = pos
    while True:
        chunk = os.pread(fd, chunk_size, off)
        if not chunk:  # EOF
            break
        nl = chunk.find(b'\n')
        if nl != -1:
            buf.extend(chunk[:nl + 1])
            break
        buf.extend(chunk)
        off += len(chunk)

    line = bytes(buf).rstrip(b'\r\n').decode('utf-8')
    return line

def search_abs_by_id(arxiv_id):
    pos = db3_idx_aid2pos.get(arxiv_id)
    if pos is None:
        return None
    line = read_line(pos)
    data = json.loads(line)
    publish_date_str = data['versions'][0]["created"].split(',')[-1].strip().split(' ')
    convert = {
        "Jan": "01","Feb": "02","Mar": "03","Apr": "04","May": "05","Jun": "06",
        "Jul": "07","Aug": "08","Sep": "09","Oct": "10","Nov": "11","Dec": "12"
    }
    publish_date = f"{publish_date_str[2]}-{convert[publish_date_str[1]]}-{publish_date_str[0]}"
    return {
        'arxiv_id': data['id'],
        'title': data['title'],
        'abstract': data['abstract'],
        "journal": data['journal-ref'],
        "authors": data['authors'],
        "year": publish_date,
        'sections': "",
        'source': 'SearchFrom:db3'
    }

def normalize_title(text):
    """将标题转换为纯字符"""
    text = unicodedata.normalize('NFKC', text) # Unicode标准化
    text = text.lower() # 转小写
    text = re.sub(r'[^a-z0-9]', '', text) # 只保留字母和数字
    return text

def search_id_by_title(title):
    simtitle = normalize_title(title)
    pos = db3_idx_sim2pos.get(simtitle)
    if pos is None:
        return None
    line = read_line(pos)
    data = json.loads(line)
    return data['id']
 

def parse_metadata(metas):
    """
    Parse concatenated metadata string into authors, title, and journal.
    """
    # Get and clean metas
    metas = [item.replace('\n', ' ') for item in metas]
    meta_string = ' '.join(metas)
    
    authors, title, journal = "", "", ""
        
    if len(metas) == 3: # author / title / journal
        authors, title, journal = metas
    else:
        # Remove the year suffix (e.g., 2022a) from the metadata string
        meta_string = re.sub(r'\.\s\d{4}[a-z]?\.', '.', meta_string)
        # Regular expression to match the pattern
        regex = r"^(.*?\.\s)(.*?)(\.\s.*|$)"
        match = re.match(regex, meta_string, re.DOTALL)
        if match:
            authors = match.group(1).strip() if match.group(1) else ""
            title = match.group(2).strip() if match.group(2) else ""
            journal = match.group(3).strip() if match.group(3) else ""

            if journal.startswith('. '):
                journal = journal[2:]

    return {
        "meta_list": metas, 
        "meta_string": meta_string, 
        "authors": authors,
        "title": title,
        "journal": journal
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
    begin = time.time()
    soup = bs4.BeautifulSoup(html_file, "lxml")
    end = time.time()
    logger.info(f"\033[94m soup解析耗时：{end-begin} \033[0m")
    
    begin = time.time()
    # parse title
    title = soup.head.title.get_text().replace("\n", " ")
    # parse abstract
    abstract = soup.find(class_='ltx_abstract').get_text()
    # parse citation
    citation = soup.find(class_='ltx_biblist')
    citation_dict = create_dict_for_citation(citation)
    # generate the full toc without text
    sections = generate_full_toc(soup)
    # remove the sections need to skip and extract the text of the rest sections
    sections = remove_stop_word_sections_and_extract_text(sections, soup)
    document = {
        "title": title, 
        "abstract": abstract, 
        "sections": sections, 
        "references": citation_dict,
    }
    end = time.time()
    logger.info(f"\033[94m 结构化解析HTML耗时：{end-begin} \033[0m")
    return document 

def parse_ar5iv(entry_id):
    begin = time.time()
    hit = 0
    url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'
    data_path = f"{config['arxiv_database_path']}/arxiv/ar5iv_html/{entry_id.split('.')[0]}/{entry_id}.html"
    if os.path.exists(data_path) and config['use_local_db']:
        with open(data_path, "r", encoding="utf-8") as f:
            logger.info(f"\033[94m 解析内容，{entry_id}.html已存在，直接读取 \033[0m")
            html_content = f.read()
            hit = 1
    else:
        # response = requests.get(url, timeout=20)
        # if response.status_code == 200:
        #     html_content = response.text
        # else:
        #     warnings.warn(f'Invalid ar5iv HTML document: {url}')
        #     return None,0
        logger.error(f"\033[91m ar5ix html不存在\033[0m")
        return None,0
    end = time.time()
    # print(f"\033[94m 网页下载耗时：{end-begin} \033[0m")
    begin = time.time()
    if html_content:
        if not 'https://ar5iv.labs.arxiv.org/html' in html_content:
            logger.error(f'Failed to get Invalid ar5iv HTML document: {url}')
            return html_content,0
        else:
            try:
                document = parse_html(html_content)
                if config["save_data"]:
                    save_path = data_path
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    if not os.path.exists(save_path):
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(html_content)
                end = time.time()
                print(f"{entry_id}解析耗时：{end-begin} ")
                return document,hit
            except:
                logger.error(f'Failed to get Wrong format HTML document: {url}')
                return None,0
    return None,0

# def parse_pdf_download(entry_id):
#     link = f"https://arxiv.org/pdf/{entry_id}"
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     if not os.path.exists(os.path.join(current_dir, "pdf_temp")):
#         os.makedirs(os.path.join(current_dir, "pdf_temp"))
#     paper_id = uuid.uuid4()
#     pdf_path = paper_downloader(link,f"{current_dir}/pdf_temp/{paper_id}.pdf")
#     document = pdf_reader(pdf_path)
#     os.remove(pdf_path)
#     return document

def search_content_by_arxiv_id(entry_id, max_chars=300):
    """根据aid搜索ar5iv，返回论文章节内容信息（前max_chars字），未找到返回None"""
    arxiv_id = entry_id.split('v')[0] if 'v' in entry_id else entry_id
    start_time = time.time()
    url = f'https://ar5iv.labs.arxiv.org/html/{arxiv_id}'
    try:
        data_path = f"{config['arxiv_database_path']}/arxiv/ar5iv_html/{entry_id.split('.')[0]}/{entry_id}.html"
        if os.path.exists(data_path) and config['use_local_db']:
            with open(data_path, "r", encoding="utf-8") as f:
                logger.info(f"解析内容，{entry_id}.html已存在，直接读取")
                html_content = f.read()
        else:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                html_content = response.text
                if not 'https://ar5iv.labs.arxiv.org/html' in html_content:
                    warnings.warn(f'Invalid ar5iv HTML document: {url}')
                    metrics_tracker.org_add_id2cnt(success=False, time_cost=time.time() - start_time)
                    return None
                else:
                    if config["save_data"]:
                        save_path = data_path
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        if not os.path.exists(save_path):
                            with open(save_path, "w", encoding="utf-8") as f:
                                f.write(html_content)
            else:
                logger.error(f'Failed to retrieve content. Status code: {response.status_code}')
                metrics_tracker.org_add_id2cnt(success=False, time_cost=time.time() - start_time)
                return None
        try:
            document = parse_html(html_content)
        except:
            logger.error(f'failed to get Wrong format HTML document: {url}')
            metrics_tracker.org_add_id2cnt(success=False, time_cost=time.time() - start_time)
            return None
        try:
            sections = get_2nd_section(document["sections"][0]["subsections"])
        except:
            logger.error(f'Failed to get subsections')
            metrics_tracker.org_add_id2cnt(success=False, time_cost=time.time() - start_time)
            return None
        
        sections2content = {}
        for k, v in sections.items():
            k = " ".join(k.split("\n"))
            # 深度清理HTML内容
            import re
            clean_content = v
            
            # 移除HTML标签
            clean_content = re.sub(r'<[^>]+>', '', clean_content)
            
            # 移除HTML实体
            clean_content = re.sub(r'&[a-zA-Z]+;', '', clean_content)
            clean_content = re.sub(r'&#\d+;', '', clean_content)
            
            # 移除特殊符号和多余标点
            clean_content = re.sub(r'[~&$#^_{}\\]', '', clean_content)
            
            # 移除引用格式 (数字引用)
            clean_content = re.sub(r'\[\d+\]', '', clean_content)
            clean_content = re.sub(r'\(\d+\)', '', clean_content)
            
            # 规范化空白字符
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            # 移除过短的内容（可能是无意义的残留）
            if len(clean_content.strip()) < 10:
                clean_content = "Content not available"
            
            # 获取前max_chars字符
            preview_content = clean_content if len(clean_content) > max_chars else clean_content
            
            sections2content[k] = preview_content
        
        metrics_tracker.org_add_id2cnt(success=True, time_cost=time.time() - start_time)
        return {
            "arxiv_id": arxiv_id,
            "sections": sections2content,
        }
        
    except Exception as e:
        warnings.warn(f"Function search_content_by_arxiv_id failed for entry_id: {entry_id}, error: {e}")
        metrics_tracker.org_add_id2cnt(success=False, time_cost=time.time() - start_time)
        return None

def search_section_by_arxiv_id(entry_id, cite=r"~\\cite\{(.*?)\}"):
    start_time = time.time()
    # warnings.warn("Using search_section_by_arxiv_id function may return wrong title because ar5iv parsing citation error. To solve this, You can prompt any LLM to extract the paper title from the reference string")
    assert re.match(r'^\d+\.\d+$', entry_id)
    hit = 0
    try:
        try:
            # try to use ar5iv to parse the paper first
            document,hit = parse_ar5iv(entry_id)
            if document is None:
                raise Exception(f'Failed to parse ar5iv: {entry_id}')
        except:
            logger.error(f'Failed to parse ar5iv: {entry_id}')
            metrics_tracker.org_add_id2ref(success=False, time_cost=time.time() - start_time)
            return None,hit
        try:
            sections = get_2nd_section(document["sections"][0]["subsections"])
        except:
            logger.error(f'Failed to Get subsections')
            metrics_tracker.org_add_id2ref(success=False, time_cost=time.time() - start_time)
            return None,hit
        
        # begin = time.time()
        # 使用LLM批量解析ref，替换原来的title
        references = document["references"]
        bib_keys = list(references.keys())
        # queries = []
        # for bib in bib_keys:
        #     meta_string = references[bib]["meta_string"]
        #     query = f"Get the title from the following reference: {meta_string}, and return the title in a json format:{{'title': 'title'}}. For example, if the reference is '1. [1] Smith, J. (2021). A study on the effects of AI on society. Journal of AI Research, 10(2), 1-10.', you should return {{'title': 'A study on the effects of AI on society'}}."
        #     queries.append(query)
        # responses = qwen_agent.batch_infer_safe(queries, batch_size=min(thread_num, len(queries)))
        # for bib, response in zip(bib_keys, responses):
        #     json_begin = response.find("{")
        #     json_end = response.rfind("}") + 1
        #     result = ""
        #     try:
        #         json_str = response[json_begin:json_end].replace("'", '"')
        #         result_json = json.loads(json_str)
        #         result = result_json["title"]
        #     except:
        #         pass
        #     if result and isinstance(result, str) and result.strip():
        #         references[bib]["title"] = result.strip()
        # end = time.time()
        # print(f"LLM解析耗时：{end-begin}")
        sections2title = {}
        for k, v in sections.items():
            k = " ".join(k.split("\n"))
            sections2title[k] = set()
            bibs = re.findall(cite, v, re.DOTALL)
            for bib in bibs:
                bib = bib.split(",")
                for b in bib:
                    if b not in document["references"]:
                        continue
                    # title = get_title_from_reference(document["references"][b]["meta_string"])
                    # if title != "":
                    #     sections2title[k].add(title)
                    # else:
                    
                    sections2title[k].add(document["references"][b]["title"])
            if len(sections2title[k]) == 0:
                del sections2title[k]
            else:
                sections2title[k] = list(sections2title[k])

        metrics_tracker.org_add_id2ref(success=True, time_cost=time.time() - start_time)
        return sections2title,hit
    except Exception as e:
        logger.error(f'Failed to parse paper: {entry_id}, error: {str(e)}')
        metrics_tracker.org_add_id2ref(success=False, time_cost=time.time() - start_time)
        return None,hit


def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def search_paper_by_arxiv_id(arxiv_id):
    """
    Search paper by arxiv id.
    :param arxiv_id: arxiv id of the paper
    :return: paper list
    """
    total_start = time.time()
    db_start = time.time()
    if arxiv_id in id2paper:
        title_key = keep_letters(id2paper[arxiv_id])
        if title_key in paper_db.namelist():
            with paper_db.open(title_key) as f:
                data = json.loads(f.read().decode("utf-8"))
            metrics_tracker.add_id2paper(db_success=True, db_time=time.time() - db_start, total_time=time.time() - total_start)
            return {
                "arxiv_id": arxiv_id,
                "title": data["title"].replace("\n", " "),
                "abstract": data["abstract"],
                "sections": data["sections"],
                "source": 'SearchFrom:local_paper_db',
            }
    axv_start = time.time()
    search = arxiv.Search(
        query = "",
        id_list = [arxiv_id],
        max_results = 10,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending,
    )

    try:
        results = list(arxiv_client.results(search, offset=0))
    except:
        warnings.warn(f"Failed to search arxiv id: {arxiv_id}")
        metrics_tracker.add_id2paper(axv_call=True, axv_success=False, db_time=axv_start-db_start, axv_time=time.time() - axv_start, total_time=time.time() - total_start)
        return None

    res = None
    for arxiv_id_result in results:
        entry_id = arxiv_id_result.entry_id.split("/")[-1]
        entry_id = entry_id.split('v')[0]
        if entry_id == arxiv_id:
            res = {
                "arxiv_id": arxiv_id,
                "title": arxiv_id_result.title.replace("\n", " "),
                "abstract": arxiv_id_result.summary.replace("\n", " "),
                "sections": "",
                "source": 'SearchFrom:arxiv',
            }
            break
    if res is not None:
        metrics_tracker.add_id2paper(axv_call=True, axv_success=True, db_time=axv_start-db_start, axv_time=time.time() - axv_start, total_time=time.time() - total_start)
    else:
        metrics_tracker.add_id2paper(axv_call=True, axv_success=False, db_time=axv_start-db_start, axv_time=time.time() - axv_start, total_time=time.time() - total_start)
    return res


# 原版的title2id方法
def search_arxiv_id_by_title(title):
    start_time = time.time()
    try:
        url = "https://arxiv.org/search/?" + urllib.parse.urlencode({
            'query': title,
            'searchtype': 'title', 
            'abstracts': 'hide', 
            'size': 200, 
        })
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            html_content = response.text
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
            results = []
            if soup.find('meta', charset=True):
                if soup.find('p', class_="is-size-4 has-text-warning") and "Sorry" in soup.find('p', class_="is-size-4 has-text-warning").text.strip():
                    metrics_tracker.add_ti2id_axv(success=False, time_cost=time.time() - start_time)
                    warnings.warn(f"Failed to find results by Arxiv Advanced Search: {title}")
                    return None
                p_tags = soup.find_all("li", class_="arxiv-result")
                for p_tag in p_tags:
                    title_ = p_tag.find("p", class_="title is-5 mathjax").text.strip()
                    id = p_tag.find('p', class_='list-title is-inline-block').find('a').text.strip('arXiv:')
                    if title_ and id:
                        results.append((title_, id))
            if soup.find('html', xmlns=True):
                p_tag = soup.find("head").find("title")
                match = re.match(r'\[(.*?)\]\s*(.*)', soup.title.string)
                if match:
                    id = match.group(1)
                    title_ = match.group(2)
                    if title_ and id:
                        results = [(title_, id)]
            if results:
                for (result, id) in results:
                    title_find = result.lower().strip('.').replace(' ', '').replace('\n', '')
                    title_search = title.lower().strip('.').replace(' ', '').replace('\n', '')
                    if title_find == title_search:
                        metrics_tracker.add_ti2id_axv(success=True, time_cost=time.time() - start_time)
                        return id
                metrics_tracker.add_ti2id_axv(success=False, time_cost=time.time() - start_time)
                return None
            metrics_tracker.add_ti2id_axv(success=False, time_cost=time.time() - start_time)
            warnings.warn(f"Failed to parse the html: {url}")
            return None
        else:
            metrics_tracker.add_ti2id_axv(success=False, time_cost=time.time() - start_time)
            warnings.warn(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        metrics_tracker.add_ti2id_axv(success=False, time_cost=time.time() - start_time)
        warnings.warn(f"An error occurred while search_arxiv_id_by_title: {e}")
        return None

def search_paper_by_title(title):
    """
    Search paper by title.
    :param title: title of the paper
    :return: paper list
    """
    title_id = search_id_by_title(title) # 改进方法,从db3/db1搜索
    
    if not title_id:
        return None
    title_id = title_id.split('v')[0]
    
    return search_abs_by_id(title_id) # 改进方法,从数据库db3/db1搜索paper,用arxiv兜底

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

def cal_micro(pred_set, label_set):
    if len(label_set) == 0:
        return 0, 0, 0

    if len(pred_set) == 0:
        return 0, 0, len(label_set)

    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)

    assert tp + fn == len(label_set)
    assert len(label_set) != 0
    return tp, fp, fn

def search_ref_by_id(arxiv_id):
    ref,hit = search_section_by_arxiv_id(arxiv_id, r"~\\cite\{(.*?)\}")
    refs = {}
    if ref == None:
        return [],0
    idx = 1
    for key,values in ref.items():
        for value in values:
            refs[str(idx)] = value
            idx += 1
    return refs,hit

if __name__ == "__main__":
    # print(search_section_by_arxiv_id("2507.05241", r"~\\cite\{(.*?)\}"))
    # print(search_abs_by_id("2501.10120"))
    print(search_id_by_title("DeepAgent: A General Reasoning Agent with Scalable Toolsets"))
    # print(search_content_by_arxiv_id("2501.10120"))
