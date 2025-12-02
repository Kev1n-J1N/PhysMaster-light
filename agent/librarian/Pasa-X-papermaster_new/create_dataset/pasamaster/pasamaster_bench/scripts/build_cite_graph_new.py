import os
import re
import sys
import time
import json
import queue
import random
import textwrap
import threading
from datetime import datetime
from typing import List, Dict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from copy import deepcopy
import string
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

SERPER_API_KEY = os.getenv("SERPER_API_KEY") or "6ba0aabc5205d8e298682df7b7bd57ac05bf88a3"
SERPER_SEARCH_URL = "https://google.serper.dev/search"
REQUEST_DELAY = 2

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or "http://127.0.0.1:30020/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY") or "EMPTY"
LLM_MODEL = os.getenv("LLM_MODEL") or "qwen-72b"

DEFAULT_RETRY_STATUS = {403, 429, 500, 502, 503, 504}

arxiv_id_resume_path = "/data/duyuwen/Pasa-X/search_utils/data/paper_db/db3.json"
title_id_data = []
with open(arxiv_id_resume_path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        title_id_data.append(obj)
title2id = {re.sub(r'\s+', '', item["title"]).lower().translate(str.maketrans('', '', string.punctuation)): item["id"] for item in title_id_data}
# ==== Rate limiter ====
html_resume_path = "/data/public_data/arxiv_database/arxiv/ar5iv_html"
htmls = []
sub_folders = os.listdir(html_resume_path)
sub_folders = [f for f in sub_folders if f.isdigit()]
for sub_folder in sub_folders:
    folder_path = os.path.join(html_resume_path, sub_folder)
    sub_htmls = os.listdir(folder_path)
    htmls.extend([f for f in sub_htmls if f.endswith(".html")])


class RateLimiter:
    def __init__(self, min_interval=2):
        self.min_interval = min_interval
        self.last_call = {}
        self.lock = threading.Lock()
    
    def wait(self, key="default"):
        with self.lock:
            now = time.time()
            if key in self.last_call:
                elapsed = now - self.last_call[key]
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed + random.uniform(0, 0.5))
            self.last_call[key] = time.time()

# Global rate limiters
serper_limiter = RateLimiter(min_interval=1.0)   # Serper
arxiv_limiter  = RateLimiter(min_interval=0.5)   # arXiv / ar5iv
llm_limiter    = RateLimiter(min_interval=0.5) 


# ==== Visualization ====

def visualize_citation_graph(
    graph: List[Dict],
    root_id: str,
    output_path: str = "citation_graph.jpg",
    max_label_len: int = 60,
    show_edge_labels: bool = False,
    seed: int = 42
):
    """
    将 build_* 构建的 graph 可视化并保存为 .jpg
    graph: [{'from': 'paperA', 'to': 'paperB', 'description': '...'}, ...]
    root_id: 根论文 arxiv_id（用于高亮）
    """
    if not graph:
        raise ValueError("graph 为空，无法可视化。")

    G = nx.DiGraph()
    for e in graph:
        u, v = e["from"], e["to"]
        desc = e.get("description", "")
        if G.has_edge(u, v):
            G[u][v]["count"] += 1
            if desc:
                G[u][v]["description"].append(desc)
        else:
            G.add_edge(u, v, description=[desc] if desc else [], count=1)

    indeg = Counter({n: G.in_degree(n) for n in G.nodes})
    outdeg = Counter({n: G.out_degree(n) for n in G.nodes})
    node_sizes, node_colors = [], []
    for n in G.nodes:
        base = 300 + 50 * (indeg[n] + outdeg[n])
        node_sizes.append(base)
        node_colors.append("#ff7f0e" if n == root_id else "#1f77b4")

    random.seed(seed)
    pos = nx.spring_layout(G, k=0.7, seed=seed)

    fig = plt.figure(figsize=(14, 10), dpi=200)
    ax = plt.gca()
    ax.set_title("Citation Graph", fontsize=16)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0.5, edgecolors="#333333")
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=12, width=1.2, alpha=0.8, edge_color="#555555")

    def short_node(n: str, keep: int = 16):
        return n if len(n) <= keep else f"{n[:8]}…{n[-7:]}"
    node_labels = {n: short_node(n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    if show_edge_labels:
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            descs = [d for d in data.get("description", []) if d]
            text = "; ".join(descs) if descs else ""
            if text:
                wrapped = "\n".join(textwrap.wrap(text, width=max_label_len))
                if data.get("count", 1) > 1:
                    wrapped = f"[x{data['count']}] " + wrapped
                edge_labels[(u, v)] = wrapped
            else:
                if data.get("count", 1) > 1:
                    edge_labels[(u, v)] = f"[x{data['count']}]"
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存到：{output_path}")


# fetch URL + retries

def fetch_url(url, retries=3, delay=5, timeout=30,
              retry_status=DEFAULT_RETRY_STATUS):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code in retry_status:
                raise requests.exceptions.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r.text
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            if attempt == retries:
                return None
            time.sleep(delay)


# ==== section extraction helpers ====

def get_clean_title(header_tag) -> str:
    if not header_tag:
        return ""
    for span in header_tag.find_all("span"):
        span.extract()
    return header_tag.get_text(strip=True)

def extract_year_from_arxiv_id(arxiv_id: str) -> int:
    # 新式 arXiv ID 形如 YYMM.xxxxx
    try:
        y = int(arxiv_id[:2])
        return 2000 + y
    except Exception:
        return 0

def extract_section_citations(text: str, references: Dict) -> List[Dict]:
    """提取文本里引用的文献，并返回 [{'title','authors','journal','description'}]"""
    matched_refs = set()
    refs_in_section = []
    cite_pattern = r"\[(\d+)\]"
    bibs = re.findall(cite_pattern, text)

    for bib in bibs:
        key = bib.strip()
        if not key or key not in references:
            continue
        ref = references[key]
        ref_info = {
            "title": ref.get("title",""),
            "authors": ref.get("authors",""),
            "journal": ref.get("journal",""),
        }
        # description: 截取引用前后一句
        idx = text.find(f"[{key}]")
        desc = ""
        if idx >= 0:
            start = max(0, text.rfind('.', 0, idx) + 1)
            end = text.find('.', idx)
            if end == -1:
                end = len(text)
            desc = text[start:end+1].strip()
        ref_info["description"] = desc
        ref_tuple = tuple(ref_info.items())
        if ref_tuple not in matched_refs:
            matched_refs.add(ref_tuple)
            refs_in_section.append(ref_info)
    return refs_in_section

def extract_sections(soup: BeautifulSoup, references=None) -> List[Dict[str,str]]:
    sections = []
    section_tags = soup.find_all(["section", "div"], class_=re.compile(".*section.*", re.I))
    wrapper = textwrap.TextWrapper(width=80)

    if not section_tags:
        body_text = soup.get_text(separator="\n", strip=True)
        if body_text:
            section = {"title": "Full Text", "content": "\n".join(wrapper.wrap(body_text))}
            if references:
                refs = extract_section_citations(body_text, references)
                section["references"] = refs
            sections.append(section)
        return sections

    for idx, node in enumerate(section_tags):
        header = node.find(re.compile("h[1-6]"))
        title = get_clean_title(header) if header else f"Section {idx+1}"
        paras = node.find_all("p")
        wrapped_paras = [
            "\n".join(wrapper.wrap(p.get_text(strip=True)))
            for p in paras if p.get_text(strip=True)
        ]
        content = "\n\n".join(wrapped_paras)
        if not content.strip():
            continue
        section = {"title": title, "content": content}
        if references:
            refs = extract_section_citations(content, references)
            section["references"] = refs
        sections.append(section)
    return sections

def enrich_paper_with_sections(paper: Dict, worker_id: int = 0) -> Dict:
    """根据 arXiv ID 抓取 ar5iv HTML，提取 sections 和引用"""
    arxiv_id = paper.get("id")
    if not arxiv_id:
        return None
    prefix = arxiv_id[:4]
    if arxiv_id + ".html" in htmls:
        html_path = os.path.join(html_resume_path, prefix, arxiv_id+".html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            soup = BeautifulSoup(html, "lxml")
            print(f"[Worker {worker_id}] Successfully parsed in resume path: {html_path}")
    

    else:
        url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        arxiv_limiter.wait(f"arxiv_{worker_id % 3}")

        html = fetch_url(url, retries=3, delay=5, timeout=30)
        if not html:
            print(f"[Worker {worker_id}] Failed to fetch {url} after retries")
            return None

        print(f"[Worker {worker_id}] Successfully fetched {url}")
        soup = BeautifulSoup(html, "lxml")

        
    # 提取参考文献
    refs = {}
    biblist = soup.find(class_=re.compile(".*biblist.*", re.I))
    if biblist:
        for idx, li in enumerate(biblist.find_all("li")):
            key = str(idx+1)
            refs[key] = {
                "title": li.get_text(strip=True),
                "authors": "",
                "journal": ""
            }

    sections = extract_sections(soup, refs)
    print(f"[Worker {worker_id}] Found {len(sections)} sections")
    return {"sections": sections}


# ==== arXiv 搜索 ====

def search_arxiv_for_title(title: str, max_results: int = 5, worker_id: int = 0) -> List[str]:
    if not title:
        return []
    if title.lower() in title2id.keys():
        print(f"skip search for title: {title}")
        return [title2id[title.lower()]]

    serper_limiter.wait(f"serper_{worker_id % 2}")
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": f"site:arxiv.org {title}", "num": max_results, "page": 1, "engine": "google"}
    results = []
    
    try:
        r = requests.post(SERPER_SEARCH_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data.get("organic", []):
            url = item.get("link") or item.get("url") or ""
            if url:
                m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?', url)
                if m:
                    results.append(m.group(1))
                    if len(results) >= max_results:
                        break
    except Exception as e:
        print(f"[Worker {worker_id}] Serper search failed: {e}")

    return results

def fetch_arxiv_abstract(arxiv_id: str, worker_id: int = 0) -> str:
    """
    通过 arXiv Atom API 获取摘要；失败返回空串
    """
    if not arxiv_id:
        return ""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    arxiv_limiter.wait(f"arxiv_api_{worker_id % 3}")

    xml = fetch_url(url, retries=3, delay=5, timeout=15)
    if not xml:
        print(f"[Worker {worker_id}] Failed to fetch arXiv abstract for {arxiv_id}")
        return ""

    try:
        soup = BeautifulSoup(xml, "xml")
        entry = soup.find("entry")
        if not entry:
            return ""
        summary_tag = entry.find("summary")
        if not summary_tag:
            return ""
        abstract = summary_tag.get_text(separator=" ", strip=True)
        return abstract
    except Exception as e:
        print(f"[Worker {worker_id}] fetch_arxiv_abstract error for {arxiv_id}: {e}")
        return ""

@lru_cache(maxsize=1024)
def fetch_arxiv_title_abstract(arxiv_id: str) -> Dict[str, str]:
    """
    利用 arXiv Atom API 返回 {'title': str, 'abstract': str}
    带 LRU 缓存，减少重复请求。
    """
    if not arxiv_id:
        return {"title": "", "abstract": ""}

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    arxiv_limiter.wait("arxiv_api_meta")

    xml = fetch_url(url, retries=3, delay=5, timeout=15)
    if not xml:
        print(f"[meta] Failed to fetch arXiv metadata for {arxiv_id}")
        return {"title": "", "abstract": ""}

    try:
        soup = BeautifulSoup(xml, "xml")
        entry = soup.find("entry")
        if not entry:
            return {"title": "", "abstract": ""}

        title_tag = entry.find("title")
        summary_tag = entry.find("summary")
        title = (title_tag.get_text(separator=" ", strip=True) if title_tag else "").strip()
        abstract = (summary_tag.get_text(separator=" ", strip=True) if summary_tag else "").strip()
        return {"title": title, "abstract": abstract}
    except Exception as e:
        print(f"[meta] fetch_arxiv_title_abstract error for {arxiv_id}: {e}")
        return {"title": "", "abstract": ""}


# ==== LLM keyword extraction ====

def llm_extract_keywords(context: str, abstract: str, max_keywords: int = 6, worker_id: int = 0) -> str:

    llm_limiter.wait(f"llm_{worker_id % 2}")
    prompt = (
    "Read the paper and return 4 to 6 extracted English keywords, which should be highly specific, discipline-relevant noun phrases, decribing (i) the core algorithm/model name (e.g., DP-Transformer), (ii) the related theoretical concept or applied method (e.g., ResNets), (iii) specific potential application scenarios if mentioned (eg. Medical Image Processing), etc.\n" 
        "Output only the comma-separated list, no numbers or periods. Avoid redundancy and exclude evaluative and ageneric terms such as state-of-the-art, novel, comprehensive, etc.\n"
        f"[Citation Context]\n{(context or '').strip() or 'None'}\n\n"
        f"[Cited Paper Abstract]\n{(abstract or '').strip() or 'None'}\n"
    )
    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a rigorous academic assistant, only output English keyword."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 128
        }
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
        resp = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        text = re.sub(r"[\n\r]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip("，,;； ")
        parts = re.split(r"[，,；;]", text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return ""
        if len(parts) > max_keywords:
            parts = parts[:max_keywords]
        return "，".join(parts)
    except Exception as e:
        print(f"[Worker {worker_id}] llm_extract_keywords error: {e}")
        return ""


# ==== 缓存机制 ====

parsed_cache = {}  # type: Dict[str, List[Dict]]
cache_lock = threading.Lock()

def get_refs_from_paper(arxiv_id: str, target_count: int = 10, worker_id: int = 0) -> List[Dict]:
    """
    从 ar5iv 抽取本论文的被引条目，尽量定位到 arXiv id；
    对每条引用，用“引用上下文 + 被引论文摘要”经 LLM 提炼关键词，填入 description。
    同时返回：title、abstract、raw_description、year。
    若论文已被解读过，则直接从缓存返回，避免重复调用llm解析
    """
    if not arxiv_id:
        return []

    # 检查缓存，若论文已解析则直接返回
    with cache_lock:
        if arxiv_id in parsed_cache:
            cached = deepcopy(parsed_cache[arxiv_id])
            for ref in cached:
                ref["raw_description"] = "[Previously analyzed]"
            return cached

    # 对新论文正常解析
    enriched = enrich_paper_with_sections({"id": arxiv_id}, worker_id)
    if enriched is None:
        return []

    refs_out = []
    collected = 0

    for sec in enriched.get("sections", []):
        for ref in sec.get("references", []):
            if collected >= target_count:
                break

            cite_context = ref.get("description", "") or ""

            # 搜索arXiv id
            bib_title = ref.get("title", "")
            # 提取bib_title中双引号之间的内容，去掉换行符与多余空格，全部小写，去掉所有标点符号
            match = re.search(r'“([^”]+)”', bib_title) or re.search(r'"([^"]+)"', bib_title)
            if match:
                bib_title = re.sub(r'\s+', ' ', match.group(1).replace('\n', ' ')).strip().lower()
                bib_title = ''.join(bib_title.split()).lower()   
                bib_title = ''.join(ch for ch in bib_title if ch not in string.punctuation)
            else:
                bib_title = re.sub(r'\s+', ' ', bib_title.replace('\n', ' ')).strip().lower()
            aids = search_arxiv_for_title(bib_title, worker_id=worker_id) if bib_title else None
            aid = aids[0] if aids else None
            if not aid:
                continue

            year = extract_year_from_arxiv_id(aid)

            # 提取title + abstract
            meta = fetch_arxiv_title_abstract(aid)
            official_title = meta.get("title", "") or bib_title
            abstract = meta.get("abstract", "")

            # LLM 提取关键词
            llm_desc = llm_extract_keywords(cite_context, abstract, max_keywords=6, worker_id=worker_id)
            final_desc = llm_desc if llm_desc else (cite_context or "")

            refs_out.append({
                "arxiv_id": aid,
                "title": official_title,        # 官方题名（或 bib_title 回退）
                "abstract": abstract,           # 官方摘要
                "description": final_desc,      # 关键词/短语（用于边标签）
                "raw_description": cite_context,# 原始引用上下文（首次解析保留）
                "year": year
            })
            collected += 1

        if collected >= target_count:
            break

    refs_out.sort(key=lambda x: x.get("year", 0), reverse=True)

    # 写入缓存
    with cache_lock:
        parsed_cache[arxiv_id] = deepcopy(refs_out)

    return refs_out


# ==== Graph building ====

def build_circle(root_id: str, worker_id: int = 0, expand_num: int = 4) -> List[Dict]:
    """
    构建引用圈图（root -> refs -> refs -> ...）递归扩展。
    root_id: 根论文 arXiv ID
    expand_num: 扩展层数，包括根层
    """
    graph = []

    # 获取根论文元数据
    root_meta = fetch_arxiv_title_abstract(root_id)
    root_year = extract_year_from_arxiv_id(root_id)

    # 第1层引用
    refs = get_refs_from_paper(root_id, worker_id=worker_id)
    for ref in refs:
        if ref["arxiv_id"]:
            graph.append({
                "from": root_id,
                "to": ref["arxiv_id"],
                "description": ref.get("description",""),
                "raw_description": ref.get("raw_description",""),
                "from_title": root_meta.get("title",""),
                "from_abstract": root_meta.get("abstract",""),
                "from_year": root_year,
                "to_title": ref.get("title",""),
                "to_abstract": ref.get("abstract",""),
                "to_year": ref.get("year",0),
            })
    print(f"第1层扩充后，论文数目：{len(graph)}")

    # 用上一层的 refs 作为下一层扩展起点
    current_layer_refs = refs
    for layer in range(2, expand_num + 1):
        next_layer_refs = []
        time_start = time.time()

        for ref in current_layer_refs:
            if not ref.get("arxiv_id"):
                continue
            subrefs = get_refs_from_paper(ref["arxiv_id"], worker_id=worker_id)
            for subref in subrefs:
                if not subref.get("arxiv_id"):
                    continue
                graph.append({
                    "from": ref["arxiv_id"],
                    "to": subref["arxiv_id"],
                    "description": subref.get("description",""),
                    "raw_description": subref.get("raw_description",""),
                    "from_title": ref.get("title",""),
                    "from_abstract": ref.get("abstract",""),
                    "from_year": ref.get("year",0),
                    "to_title": subref.get("title",""),
                    "to_abstract": subref.get("abstract",""),
                    "to_year": subref.get("year",0),
                })
            next_layer_refs.extend(subrefs)

        current_layer_refs = next_layer_refs
        time_end = time.time()
        print(f"第{layer}层扩充用时：{time_end - time_start:.2f} 秒")
        print(f"第{layer}层扩充后，论文数目：{len(graph)}")

        if not current_layer_refs:
            break

    current_date = datetime.now().strftime("%Y%m%d")
    result_dir_name = f"{current_path}/../output/{current_date}/vis_graph"
    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)

    visualize_citation_graph(
        graph,
        root_id=root_id,
        output_path=f"{result_dir_name}/{root_id}.jpg",
        show_edge_labels=False
    )

    return graph


def split_patterns_for_100(topics):
    patterns = ["circle"]
    return [(t, patterns[0]) for i, t in enumerate(topics)]

def build_for_topic(topic, pattern, max_target: int = 10, worker_id: int = 0):
    print(f"[Worker {worker_id}] Processing topic: {topic} with pattern: {pattern}")

    root_arxiv = None
    
    candidate_papers = search_arxiv_for_title(topic, max_results=max_target, worker_id=worker_id)

    for idx, candidate in enumerate(candidate_papers):
        refs = get_refs_from_paper(candidate, target_count=5, worker_id=worker_id)
        if refs:
            print(f"[Worker {worker_id}] Found root paper: {candidate} with {len(refs)} refs")
            root_arxiv = candidate
            break
        else:
            print(f"[Worker {worker_id}] Candidate {candidate} has no refs, trying next")
    
    if not root_arxiv:
        return {"topic": topic, "pattern": pattern, "error":"No root found", "graph":[]}
    
    if pattern=="circle":
        graph = build_circle(root_arxiv, worker_id=worker_id)
    else:
        graph = []
    
    if not graph:
        print(f"[Worker {worker_id}] Failed to build {pattern} for topic '{topic}', skipping")
    
    return {"topic": topic, "pattern": pattern, "root": root_arxiv, "graph": graph}

def process_topic_wrapper(args):
    topic, pattern, worker_id = args
    try:
        result = build_for_topic(topic, pattern, worker_id=worker_id)
        return result
    except Exception as e:
        print(f"[Worker {worker_id}] Error processing topic {topic}: {e}")
        return {"topic": topic, "pattern": pattern, "error": str(e), "graph": []}


# ==== Main =====

def main():
    start_time_main = time.time()

    paper_list_file = f"{current_path}/../papers/root_papers.json"
    with open(paper_list_file, "r", encoding="utf-8") as f:
        papers = json.load(f)
    
    topics = [p["topic"] for p in papers][:1]
    assignments = split_patterns_for_100(topics)
    from datetime import datetime
    current_date = datetime.now().strftime("%Y%m%d")
    result_dir_name = f"{current_path}/../output/{current_date}"
    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)
    
    max_workers = 1
    out_path = f"{result_dir_name}/cite_graphs.jsonl"
    
    work_items = [(topic, pattern, i % max_workers) for i, (topic, pattern) in enumerate(assignments)]
    
    write_lock = threading.Lock()
    results_queue: "queue.Queue" = queue.Queue()
    
    def write_results():
        with open(out_path, "w", encoding="utf-8") as f:
            while True:
                result = results_queue.get()
                if result is None:
                    break
                with write_lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
    
    writer_thread = threading.Thread(target=write_results)
    writer_thread.start()
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_topic_wrapper, item): item for item in work_items}
        with tqdm(total=len(work_items), desc="Processing topics") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=300)
                    results_queue.put(result)
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix({"completed": completed})
                    import pdb;pdb.set_trace()
                except Exception as e:
                    topic, pattern, worker_id = futures[future]
                    print(f"\n[ERROR] Failed to process {topic}: {e}")
                    results_queue.put({"topic": topic, "pattern": pattern, "error": str(e), "graph": []})
                    pbar.update(1)
    
    results_queue.put(None)
    writer_thread.join()
    
    end_time_main = time.time()
    print(f"\nCompleted! Saved {completed} results to {out_path}")
    total_time = end_time_main - start_time_main
    print(f"Total time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()