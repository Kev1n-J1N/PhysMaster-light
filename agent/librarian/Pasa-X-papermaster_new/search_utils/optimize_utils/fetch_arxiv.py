import os
import sys
import json
import requests
from threading import Lock
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
import urllib.request as libreq
import urllib.parse
sys.path.append((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from search_utils.optimize_utils.sim_title import normalize_title, get_main_title, jaccard_similarity
from search_utils.optimize_utils.paraline import exec_parallel


def is_similar_title(title1, title2, jac_threshold=0.85, edit_threshold=0.85):
    """判断标题是否相同"""
    # 原代码做法
    title1 = title1.lower().strip('.').replace(' ', '').replace('\n', '')
    title2 = title2.lower().strip('.').replace(' ', '').replace('\n', '')
    if title1 == title2:
        return True
    # 转纯字符比较
    t1 = normalize_title(title1)
    t2 = normalize_title(title2)
    if t1 == t2:
        return True
    # jaccard相似度比较
    jac = jaccard_similarity(t1, t2)
    if jac >= jac_threshold:
        return True
    # 编辑距离比较
    seq = SequenceMatcher(None, t1, t2)
    if seq.ratio() >= edit_threshold:
        return True
    # 如果不相似，target_title去掉冒号或破折号后的内容再比较
    main1 = normalize_title(get_main_title(title1))
    main2 = normalize_title(get_main_title(title2))
    jac_main = jaccard_similarity(main1, main2)
    if jac_main >= jac_threshold:
        return True
    seq_main = SequenceMatcher(None, main1, main2)
    if seq_main.ratio() >= edit_threshold:
        return True
    return False

def arxiv_search(query=None, ids=None):
    base_url = "http://export.arxiv.org/api/query"
    params = {"max_results": 25}
    if query:
        params['search_query'] = query
    if ids:
        params['id_list'] = ids
    for _ in range(3):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            return entries
        except requests.exceptions.RequestException as e:
            print(f"arxiv请求出错: {e}")
    return None

def fetch_arxiv(lines, stop_event):
    """查找所有title，返回title-id-abstract/None列表"""
    if stop_event.is_set():
        return None
    print("Dealing", lines[0]['title'][:10])
    
    # 若有多个title，拼接起来搜索
    titles = [item['title'] for item in lines]
    search_query = " OR ".join([f'ti:\"{t}\"' for t in titles])
    entries = arxiv_search(query=search_query)
    if entries is None:
        for item in lines:
            item['arxiv_id'] = None
            item['abstract'] = None
        return lines
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entry_infos = []
    # 提取所有返回的title-id-abstract
    for entry in entries:
        title_elem = entry.find('atom:title', ns)
        arxiv_id_elem = entry.find('atom:id', ns)
        abstract_elem = entry.find('atom:summary', ns)
        if title_elem is not None and title_elem.text:
            entry_infos.append({
                'title': title_elem.text.strip(),
                'arxiv_id': arxiv_id_elem.text.strip().split('/')[-1] if arxiv_id_elem is not None else None,
                'abstract': abstract_elem.text.strip() if abstract_elem is not None else None
            })
    # 对每个title与返回结果比较
    for item in lines:
        found = False
        for info in entry_infos:
            if is_similar_title(info['title'], item['title']):
                item['arxiv_id'] = info['arxiv_id']
                item['abstract'] = info['abstract']
                found = True
                break
        if not found:
            item['arxiv_id'] = None
            item['abstract'] = None
    return lines

def fetch_arxiv1(lines, stop_event):
    """通过论文标题在 arXiv 上搜索并返回最匹配的结果的 ID"""
    if stop_event.is_set():
        return None
    print("Dealing", lines[0]['title'][:10])
    
    title = lines[0]['title']
    lines[0]['arxiv_id'] = None
    lines[0]['abstract'] = None
    # 对标题进行 URL 编码
    encoded_title = urllib.parse.quote(title)
    # 增加搜索结果数量，以便进行更好的匹配
    url = f'http://export.arxiv.org/api/query?search_query=ti:{encoded_title}&start=0&max_results=5'
    try:
        with libreq.urlopen(url) as response:
            data = response.read()
        
        root = ET.fromstring(data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        # 查找所有条目
        entries = root.findall('.//atom:entry', namespace)
        best_match = None
        highest_ratio = 0
        for entry in entries:
            entry_title = entry.find('atom:title', namespace).text.lower()
            # 使用字符串相似度比较
            ratio = SequenceMatcher(None, title.lower(), entry_title).ratio()
            # 如果相似度超过阈值，更新最佳匹配
            if ratio > highest_ratio and ratio > 0.85:  # 设置较高的阈值确保匹配质量
                highest_ratio = ratio
                best_match = entry
            # if is_similar_title(entry_title, title.lower()):
            #     best_match = entry
            #     break
        if best_match is not None:
            id_url = best_match.find('atom:id', namespace).text
            arxiv_id = id_url.split('/abs/')[-1]
            lines[0]['arxiv_id'] = arxiv_id
            summary = best_match.find('atom:summary', namespace).text
            lines[0]['abstract'] = summary
        return lines
    except Exception as e:
        print(f"搜索时发生错误: {str(e)}")
        return lines

# 提取输入
def input_func(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f if line.strip()]
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            fin_lines = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        fin_lines = []
    # 根据title去除已处理项
    fin_ids = set(line.get('title') for line in fin_lines)
    res = [line for line in all_lines if line.get('title') not in fin_ids]
    print('total:', len(all_lines), 'finish:', len(fin_ids), 'todo:', len(res))
    # 分组
    group_size = 1
    grouped = [res[i:i+group_size] for i in range(0, len(res), group_size)]
    print('groups:', len(grouped))
    return grouped[600:]

# 处理输出
results = []
results_lock = Lock()
batch_size = 10
def output_func(result, output_path):
    # 只保留arxiv_id不为None的数据
    valid_items = [item for item in result if item.get('arxiv_id') is not None]
    print('result:', len(result), 'avai:', len(valid_items))
    if not valid_items:
        return
    with results_lock:
        results.extend(valid_items)
        if len(results) >= batch_size:
            with open(output_path, 'a', encoding='utf-8') as f:
                for item in results:
                    res = {}
                    res['arxiv_id'] = item['arxiv_id']
                    res['title'] = item['title']
                    res['abstract'] = item['abstract']
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
            results.clear()


if __name__ == "__main__":
    """
    input_func(input_path, output_path)
    process_func(line, stop_event)
    output_func(result, output_path)
    """
    input_path = r'search_utils\data\evaluate\ti2id\data\test900.jsonl'
    output_path = r'search_utils\data\evaluate\ti2id\result\test2\axv_fnd4.jsonl'
    max_workers = 5
    exec_parallel(input_path, output_path, max_workers, input_func, fetch_arxiv, output_func)
    # 在主流程结束后，记得将剩余未满batch_size的结果写入文件
    if results:
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in results:
                res = {}
                res['arxiv_id'] = item['arxiv_id']
                res['title'] = item['title']
                res['abstract'] = item['abstract']
                f.write(json.dumps(res, ensure_ascii=False) + '\n')