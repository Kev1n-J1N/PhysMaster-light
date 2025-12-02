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
import urllib
import uuid
import warnings
import zipfile
from datetime import datetime
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut 
import arxiv
import bs4
import requests
from paper_download import *
from pdf_read import *
from search_utils_source import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.llm_call import *

warnings.simplefilter("always")

arxiv_client = arxiv.Client(delay_seconds = 0.05)
# with open("data/paper_database/id2paper.json", "r") as f:
#     id2paper  = json.load(f)
# paper_db     = zipfile.ZipFile("data/paper_database/cs_paper_2nd.zip", "r")

def search_paper_by_title(title):
    """
    Search paper by title.
    :param title: title of the paper
    :return: paper list
    """
    title_id = search_arxiv_id_by_title_simi(title)
    if title_id is None:
        return None
    title_id = title_id.split('v')[0]
    return search_paper_by_arxiv_id(title_id)


def search_paper_by_arxiv_id(arxiv_id):
    """
    Search paper by arxiv id.
    :param arxiv_id: arxiv id of the paper
    :return: paper list
    """
    # if arxiv_id in id2paper:
    #     title_key = keep_letters(id2paper[arxiv_id])
    #     if title_key in paper_db.namelist():
    #         with paper_db.open(title_key) as f:
    #             data = json.loads(f.read().decode("utf-8"))
    #         return {
    #             "arxiv_id": arxiv_id,
    #             "title": data["title"].replace("\n", " "),
    #             "abstract": data["abstract"],
    #             "sections": data["sections"],
    #             "source": 'SearchFrom:local_paper_db',
    #         }

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
    return res
    
def search_arxiv_id_by_title_html(title: str, return_debug=False, threshold=85, return_best_always=True):
    """
    Search arXiv ID from HTML search results by exact title match or fuzzy match.
    :param title: Title of the paper
    :param return_debug: If True, return dict with debug info
    :param threshold: Fuzzy matching threshold (0-100)
    :param return_best_always: If no match exceeds threshold, still return best match
    :return: arXiv ID or debug info
    """
    query_url = "https://arxiv.org/search/?" + urllib.parse.urlencode({
        'query': title,
        'searchtype': 'title',
        'abstracts': 'hide',
        'size': 200,
    })

    headers = {'User-Agent': 'Mozilla/5.0'}
    print(f"[信息] 正在请求页面：{query_url}")

    try:
        response = requests.get(query_url, headers=headers)
        if response.status_code != 200:
            warning_msg = f"Failed to retrieve content. Status code: {response.status_code}"
            warnings.warn(warning_msg)
            return "" if not return_debug else {"arxiv_id": "", "error": warning_msg}

        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        entries = []

        # 多结果页面
        if soup.find('li', class_="arxiv-result"):
            for result in soup.find_all("li", class_="arxiv-result"):
                result_title = result.find("p", class_="title is-5 mathjax").text.strip()
                id_text = result.find('p', class_='list-title is-inline-block').find('a').text.strip('arXiv:')
                entries.append((result_title, id_text))

        # 单结果页面（直接跳转）
        elif soup.find("head") and soup.find("title"):
            match = re.match(r'\[(.*?)\]\s*(.*)', soup.title.string)
            if match:
                id_text = match.group(1)
                result_title = match.group(2)
                entries.append((result_title, id_text))

        if not entries:
            warning_msg = f"Failed to parse the html: {query_url}"
            warnings.warn(warning_msg)
            return "" if not return_debug else {"arxiv_id": "", "error": warning_msg}

        norm_input = normalize(title)
        best_match = None
        best_score = 0

        for result_title, arxiv_id in entries:
            score = fuzz.token_sort_ratio(norm_input, normalize(result_title))
            print(f"[比对] 相似度 {score} | 标题：{result_title}")
            if score > best_score:
                best_score = score
                best_match = (arxiv_id, result_title, score)

        if best_match and (best_score >= threshold or return_best_always):
            print(f"[成功] 匹配成功，arXiv ID: {best_match[0]} (相似度 {best_score})")
            return {"arxiv_id": best_match[0], "title": best_match[1], "score": best_score} if return_debug else best_match[0]

        print("[警告] 没有找到足够匹配的结果")
        return "" if not return_debug else {"arxiv_id": "", "reason": "no good match"}

    except Exception as e:
        error_msg = f"An error occurred during search: {e}"
        warnings.warn(error_msg)
        return "" if not return_debug else {"arxiv_id": "", "error": str(e)}

def search_arxiv_id_by_title_api(title: str, max_results=10, threshold=85, return_debug=False, return_best_always=True):
    """
    通过标题调用 arXiv API 进行搜索，返回最匹配的 arXiv ID。

    Args:
        title (str): 待搜索的论文标题
        max_results (int): API 搜索返回的最大结果数
        threshold (int): 最小相似度阈值（0-100）
        return_debug (bool): 是否返回调试信息
        return_best_always (bool): 如果未达阈值，是否返回最相近匹配

    Returns:
        str or dict: arXiv ID，或包含调试信息的字典
    """
    import urllib.request as libreq
    import urllib.parse
    import xml.etree.ElementTree as ET

    encoded_title = urllib.parse.quote(title)
    url = f'http://export.arxiv.org/api/query?search_query=ti:{encoded_title}&start=0&max_results={max_results}'
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ArxivBot/1.0)'}
    req = libreq.Request(url, headers=headers)

    print(f"[信息] 正在请求 API：{url}")

    try:
        with libreq.urlopen(req) as response:
            data = response.read()
    except Exception as e:
        error_msg = f"请求失败: {e}"
        print(f"[错误] {error_msg}")
        return "" if not return_debug else {"arxiv_id": "", "error": error_msg}

    try:
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('.//atom:entry', ns)

        print(f"[信息] 候选条目数：{len(entries)}")
        norm_input = normalize(title)

        best_match = None
        best_score = 0

        for entry in entries:
            entry_title = entry.find('atom:title', ns).text
            entry_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            score = fuzz.token_sort_ratio(norm_input, normalize(entry_title))
            print(f"[比对] 相似度 {score} | 标题：{entry_title}")
            if score > best_score:
                best_score = score
                best_match = (entry_id, entry_title, score)

        if best_match and (best_score >= threshold or return_best_always):
            print(f"[成功] 匹配成功，arXiv ID: {best_match[0]} (相似度 {best_score})")
            return {"arxiv_id": best_match[0], "title": best_match[1], "score": best_score} if return_debug else best_match[0]

        print("[警告] 没有找到足够匹配的结果")
        return "" if not return_debug else {"arxiv_id": "", "reason": "no good match"}

    except Exception as e:
        error_msg = f"解析 XML 失败: {e}"
        print(f"[错误] {error_msg}")
        return "" if not return_debug else {"arxiv_id": "", "error": error_msg}


def search_arxiv_id_by_title_tfidf(title: str, max_results: int = 20, threshold: float = 0.5, return_debug: bool = False) -> str:
    import urllib.request as libreq
    import urllib.parse
    import xml.etree.ElementTree as ET
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    import time

    def normalize(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip().lower()

    def build_query_from_title(title):
        # 提取有意义的关键词，长度大于2
        words = re.findall(r'\b\w+\b', title)
        keywords = [w for w in words if len(w) > 2]
        # 构造关键词查询，多个关键词用 AND 连接
        if not keywords:
            return urllib.parse.quote(title)
        query = '+AND+'.join([f'ti:{urllib.parse.quote(w)}' for w in keywords])
        return query

    # 构造 API 查询
    normalized_title = normalize(title)
    query_str = build_query_from_title(title)
    url = f'http://export.arxiv.org/api/query?search_query={query_str}&start=0&max_results={max_results}'
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ArxivBot/1.0)'}
    req = libreq.Request(url, headers=headers)

    print(f"[信息] 请求 API：{url}")

    # 处理 API 请求，最多重试3次
    for attempt in range(3):
        try:
            with libreq.urlopen(req) as response:
                data = response.read()
            break
        except Exception as e:
            print(f"[重试] 第 {attempt + 1} 次失败：{e}")
            time.sleep(1)
    else:
        print("[错误] 所有重试均失败")
        return "" if not return_debug else {"arxiv_id": "", "error": "Network error"}

    try:
        root = ET.fromstring(data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('.//atom:entry', namespace)

        if not entries:
            print("[警告] 没有找到候选论文")
            return "" if not return_debug else {"arxiv_id": "", "reason": "no entries"}

        candidate_titles = []
        arxiv_ids = []
        raw_titles = []

        for entry in entries:
            entry_title = entry.find('atom:title', namespace).text
            entry_id = entry.find('atom:id', namespace).text.split('/abs/')[-1]
            raw_titles.append(entry_title)
            candidate_titles.append(normalize(entry_title))
            arxiv_ids.append(entry_id)

        # 计算 TF-IDF 相似度
        vectorizer = TfidfVectorizer().fit([normalized_title] + candidate_titles)
        vectors = vectorizer.transform([normalized_title] + candidate_titles)
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        best_idx = cosine_sim.argmax()
        best_score = cosine_sim[best_idx]

        print(f"[相似度] 最佳匹配相似度：{best_score:.4f}")
        print(f"[标题] 原始：{title}")
        print(f"[标题] 匹配：{raw_titles[best_idx]}")
        print(f"[arXiv ID]：{arxiv_ids[best_idx]}")

        if best_score >= threshold:
            if return_debug:
                return {
                    "arxiv_id": arxiv_ids[best_idx],
                    "similarity": best_score,
                    "matched_title": raw_titles[best_idx]
                }
            return arxiv_ids[best_idx]
        else:
            print("[提示] 相似度未达到阈值")
            return "" if not return_debug else {
                "arxiv_id": "",
                "best_candidate": arxiv_ids[best_idx],
                "similarity": best_score,
                "matched_title": raw_titles[best_idx],
                "reason": "below threshold"
            }

    except Exception as e:
        print(f"[错误] 解析或计算失败：{e}")
        return "" if not return_debug else {"arxiv_id": "", "error": str(e)}
    


from rapidfuzz import fuzz
import urllib.request as libreq
import urllib.parse
import xml.etree.ElementTree as ET
import re
import time

def normalize(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text.strip().lower()

def search_arxiv_id_by_title_keywords(title: str, max_results=20, threshold=85, use_summary=True, return_debug=False, return_best_always=True):
    keywords = re.findall(r'\b\w+\b', title)
    keywords_query = '+AND+'.join([f'ti:{k}' for k in keywords if len(k) > 2])
    url = f'http://export.arxiv.org/api/query?search_query={keywords_query}&start=0&max_results={max_results}'

    # url = f"http://export.arxiv.org/api/query?search_query=ti:\"{encoded_title}\"+OR+all:\"{encoded_title}\"&start=0&max_results={max_results}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ArxivBot/1.0)'}
    req = libreq.Request(url, headers=headers)

    print(f"[信息] 正在请求 API：{url}")

    for attempt in range(3):
        try:
            with libreq.urlopen(req) as response:
                data = response.read()
            break
        except Exception as e:
            print(f"[重试] 第 {attempt+1} 次失败：{e}")
            time.sleep(1)
    else:
        return "" if not return_debug else {"arxiv_id": "", "error": "Network error"}

    try:
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('.//atom:entry', ns)

        print(f"[信息] 候选条目数：{len(entries)}")
        norm_input = normalize(title)

        best_match = None
        best_score = 0

        for entry in entries:
            entry_title = entry.find('atom:title', ns).text
            entry_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            score = fuzz.token_sort_ratio(norm_input, normalize(entry_title))
            print(f"[比对] 相似度 {score} | 标题：{entry_title}")

            if score > best_score:
                best_score = score
                best_match = (entry_id, entry_title, score)

        if best_match and (best_score >= threshold or return_best_always):
            print(f"[成功] 匹配成功，arXiv ID: {best_match[0]} (相似度 {best_score})")
            return {"arxiv_id": best_match[0], "title": best_match[1], "score": best_score} if return_debug else best_match[0]

        if use_summary:
            for entry in entries:
                summary = entry.find('atom:summary', ns).text.lower()
                if norm_input in normalize(summary):
                    entry_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
                    entry_title = entry.find('atom:title', ns).text
                    print(f"[备用匹配] 摘要中找到匹配，arXiv ID: {entry_id}")
                    return {"arxiv_id": entry_id, "title": entry_title, "score": "via-summary"} if return_debug else entry_id

        print("[警告] 没有找到足够匹配的结果")
        return "" if not return_debug else {"arxiv_id": "", "reason": "no good match"}

    except Exception as e:
        print(f"[错误] XML解析失败：{e}")
        return "" if not return_debug else {"arxiv_id": "", "error": str(e)}

if __name__ == "__main__":
    # print(search_section_by_arxiv_id("2307.00235", r"~\\cite\{(.*?)\}"))
    # print(search_paper_by_arxiv_id("2307.00235"))
    print(search_paper_by_title("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"))
