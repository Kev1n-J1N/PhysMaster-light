import re
import os
import sys
import json
import requests
import warnings
from threading import Lock
sys.path.append((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from search_utils.optimize_utils.paraline import exec_parallel
from search_utils.optimize_utils.fetch_arxiv import is_similar_title, arxiv_search

with open('zyz/utils/apikeys.json', 'r', encoding='utf-8') as f:
    google_key = json.load(f)['google']

def google_search(query):
    """根据query搜索，返回结果"""
    url = "https://google.serper.dev/search"
    search_query = f"{query} site:arxiv.org"
    payload = json.dumps({
        "q": search_query, 
        "num": 10, 
        "page": 1, 
    })
    headers = {
        'X-API-KEY': google_key,
        'Content-Type': 'application/json'
    }
    assert headers['X-API-KEY'] != 'your google keys', "add your google search key!!!"

    for _ in range(3):
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code == 200:
                results = json.loads(response.text)
                return results
        except:
            warnings.warn(f"google search failed, query: {query[:10]}")
            continue
    return None

def fetch_google(line, stop_event):
    """查找title，返回title-id-abstract/None"""
    if stop_event.is_set():
        return None    
    print("Dealing", line['title'][:10])
    
    tar_title = line['title']
    # 根据title在google搜索arxiv_id
    res = google_search(tar_title)
    for paper in res['organic']:
        if is_similar_title(tar_title, paper['title']):
            if re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]):
                arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]).group(1)
                line['arxiv_id'] = arxiv_id
                del line['title']
                line['title'] = tar_title
                break
    # 根据arxiv_id在arxiv搜索abstract
    if 'arxiv_id' in line:
        pass
        # entries = arxiv_search(ids=[line['arxiv_id']])
        # ns = {'atom': 'http://www.w3.org/2005/Atom'}
        # for entry in entries:
        #     # 提取
        #     title_elem = entry.find('atom:title', ns)
        #     arxiv_id_elem = entry.find('atom:id', ns)
        #     abstract_elem = entry.find('atom:summary', ns)
        #     arxiv_id = arxiv_id_elem.text.strip().split('/')[-1] if arxiv_id_elem is not None else None
        #     if arxiv_id is not None:
        #         arxiv_id = arxiv_id.split('v')[0]
        #     title = title_elem.text.strip() if title_elem is not None else None
        #     abstract = abstract_elem.text.strip() if abstract_elem is not None else None
        #     # 匹配
        #     if arxiv_id == line['arxiv_id'] and is_similar_title(title, line['title']):
        #         line['abstract'] = abstract
        #         break
    else:
        line['arxiv_id'] = None
        # line['abstract'] = None
    return line

# 提取输入
def input_func(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            fin_lines = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        fin_lines = []
    fin_ids = set(line.get('title') for line in fin_lines)
    res = [line for line in all_lines if line.get('title') not in fin_ids]
    print('total:', len(all_lines), 'finish:', len(fin_ids), 'todo:', len(res))
    return res

# 处理输出
results = []
results_lock = Lock()
batch_size = 10
def output_func(result, output_path):
    # 只保留arxiv_id不为None的数据
    # if result.get('arxiv_id') is not None and result.get('abstract') is not None:
    if result.get('arxiv_id') is not None:
        print('Save:', result['arxiv_id'])
        with results_lock:
            results.append(result)
            if len(results) >= batch_size:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                results.clear()


if __name__ == '__main__':
    """
    input_func(input_path, output_path)
    process_func(line, stop_event)
    output_func(result, output_path)
    """
    input_path = r'optimize\evaluate\ti2id\data\test2\all.jsonl'
    output_path = r'optimize\evaluate\ti2id\result\test2\goo_fnd.jsonl'
    max_workers = 50
    exec_parallel(input_path, output_path, max_workers, input_func, fetch_google, output_func)
    # 在主流程结束后，记得将剩余未满batch_size的结果写入文件
    if results:
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')