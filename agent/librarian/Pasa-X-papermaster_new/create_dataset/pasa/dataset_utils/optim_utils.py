import os
import sys
import threading
import urllib.parse
import urllib.request as libreq
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zyz.optimize.call_db import search_db3_title, search_db1_title
from zyz.optimize.fetch_arxiv import is_similar_title
from zyz.optimize.call_ar5iv import search_section_by_arxiv_id
from zyz.optimize.parse_utils import parse_tex_folder, fetch_eprint, extract_tar_gz


def fetch_arxiv(title):
    """【不直接调用的工具函数】根据title搜索arXiv，匹配则返回arxiv_id/title/abstract，否则返回None"""
    encoded_title = urllib.parse.quote(title) # 对标题进行 URL 编码
    url = f'http://export.arxiv.org/api/query?search_query=ti:{encoded_title}&start=0&max_results=5'
    try:
        with libreq.urlopen(url, timeout=60) as response:
            data = response.read()
        root = ET.fromstring(data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('.//atom:entry', namespace) # 查找所有条目
        best_match = None
        highest_ratio = 0
        for entry in entries:
            entry_title = entry.find('atom:title', namespace).text.lower()
            # 原比对方法
            # ratio = SequenceMatcher(None, title.lower(), entry_title).ratio() # 使用字符串相似度比较
            # if ratio > highest_ratio and ratio > 0.85:  # 如果相似度超过阈值，更新最佳匹配
            #     highest_ratio = ratio
            #     best_match = entry
            
            # 更鲁棒的比对方法
            if is_similar_title(entry_title, title.lower()):
                best_match = entry
                break
        if best_match is not None:
            id_url = best_match.find('atom:id', namespace).text
            arxiv_id = id_url.split('/abs/')[-1]
            summary = best_match.find('atom:summary', namespace).text
            return {
                'arxiv_id': arxiv_id.split('v')[0],
                'title': title,
                'abstract': summary
            }
        else:
            return None
    except Exception as e:
        print(f"搜索 {title} 时发生错误: {str(e)}")
        return None

id2ref_call_cnt = 0
id2ref_suc_cnt = 0
id2ref_tex_suc_cnt = 0
id2ref_ar5iv_call_cnt = 0
id2ref_ar5iv_suc_cnt = 0
id2ref_lock = threading.Lock()

def get_id2ref():
    global id2ref_call_cnt, id2ref_suc_cnt, id2ref_tex_suc_cnt, id2ref_ar5iv_call_cnt, id2ref_ar5iv_suc_cnt
    return {
        'id2ref call': id2ref_call_cnt,
        'id2ref success': id2ref_suc_cnt,
        'id2ref success rate': id2ref_suc_cnt / id2ref_call_cnt if id2ref_call_cnt else 0,
        'tex success': id2ref_tex_suc_cnt,
        'tex success rate': id2ref_tex_suc_cnt / id2ref_call_cnt if id2ref_call_cnt else 0,
        'ar5iv call': id2ref_ar5iv_call_cnt,
        'ar5iv success': id2ref_ar5iv_suc_cnt,
        'ar5iv success rate': id2ref_ar5iv_suc_cnt / id2ref_ar5iv_call_cnt if id2ref_ar5iv_call_cnt else 0
    }

def fetch_ref(arxiv_id):
    """【不直接调用的工具函数】根据arxiv_id搜索tex和ar5iv，匹配则返回arxiv_id/sections，否则返回None"""
    global id2ref_call_cnt, id2ref_suc_cnt, id2ref_tex_suc_cnt, id2ref_ar5iv_call_cnt, id2ref_ar5iv_suc_cnt
    with id2ref_lock:
        id2ref_call_cnt += 1
    base_dir = 'zyz/optimize/paper_cache'
    os.makedirs(base_dir, exist_ok=True)
    paper_dir = os.path.join(base_dir, f'arXiv-{arxiv_id}')
    tar_path = fetch_eprint(arxiv_id, base_dir)
    if tar_path is None:
        print('call ar5iv')
        with id2ref_lock:
            id2ref_ar5iv_call_cnt += 1
        res = search_section_by_arxiv_id(arxiv_id)
        if res is not None:
            with id2ref_lock:
                id2ref_ar5iv_suc_cnt += 1
                id2ref_suc_cnt += 1
        return res
    try:
        extract_tar_gz(tar_path, paper_dir)
    except Exception as e:
        print('call ar5iv')
        with id2ref_lock:
            id2ref_ar5iv_call_cnt += 1
        res = search_section_by_arxiv_id(arxiv_id)
        if res is not None:
            with id2ref_lock:
                id2ref_ar5iv_suc_cnt += 1
                id2ref_suc_cnt += 1
        return res
    try:
        result = parse_tex_folder(paper_dir)
        if not result:
            with id2ref_lock:
                id2ref_ar5iv_call_cnt += 1
            res = search_section_by_arxiv_id(arxiv_id)
            if res is not None:
                with id2ref_lock:
                    id2ref_ar5iv_suc_cnt += 1
                    id2ref_suc_cnt += 1
            return res
        with id2ref_lock:
            id2ref_tex_suc_cnt += 1
            id2ref_suc_cnt += 1
        return {
            'arxiv_id': arxiv_id,
            'sections': result,
        }
    except Exception as e:
        print(f'tex解析失败: {arxiv_id}, 错误: {e}')
        print('call ar5iv')
        with id2ref_lock:
            id2ref_ar5iv_call_cnt += 1
        res = search_section_by_arxiv_id(arxiv_id)
        if res is not None:
            with id2ref_lock:
                id2ref_ar5iv_suc_cnt += 1
                id2ref_suc_cnt += 1
        return res

ti2id_lock = threading.Lock()
ti2id_call_cnt = 0
ti2id_suc_cnt = 0
db3_suc_cnt = 0
db1_call_cnt = 0
db1_suc_cnt = 0
axv_call_cnt = 0
axv_suc_cnt = 0

def get_ti2id():
    global ti2id_call_cnt, ti2id_suc_cnt, db_suc_cnt, axv_call_cnt, axv_suc_cnt
    return {
        'ti2id call': ti2id_call_cnt,
        'ti2id success': ti2id_suc_cnt,
        'ti2id success rate': ti2id_suc_cnt / ti2id_call_cnt if ti2id_call_cnt else 0,
        'db3 success': db3_suc_cnt,
        'db3 success rate': db3_suc_cnt / ti2id_call_cnt if ti2id_call_cnt else 0,
        'db1 call': db1_call_cnt,
        'db1 success': db1_suc_cnt,
        'db1 success rate': db1_suc_cnt / db1_call_cnt if db1_call_cnt else 0,
        'axv call': axv_call_cnt,
        'axv success': axv_suc_cnt,
        'axv success rate': axv_suc_cnt / axv_call_cnt if axv_call_cnt else 0
    }

def search_id_by_title(title):
    """根据title搜索本地数据库db3、db1和arxiv，匹配则返回arxiv_id/title/abstract，否则返回None"""
    global ti2id_call_cnt, ti2id_suc_cnt, db3_suc_cnt, db1_call_cnt, db1_suc_cnt, axv_call_cnt, axv_suc_cnt
    with ti2id_lock:
        ti2id_call_cnt+=1
    res = search_db3_title(title)
    if res is None:
        with ti2id_lock:
            db1_call_cnt+=1
        res = search_db1_title(title)
        if res is None:
            with ti2id_lock:
                axv_call_cnt+=1
            res = fetch_arxiv(title)
            if res is not None:
                with ti2id_lock:
                    axv_suc_cnt+=1
                    ti2id_suc_cnt+=1
        else:
            with ti2id_lock:
                db1_suc_cnt+=1
                ti2id_suc_cnt+=1
    else:
        with ti2id_lock:
            db3_suc_cnt+=1
            ti2id_suc_cnt+=1
    # res包括id/title/abstract，为对接仅返回arxiv_id
    if res is not None:
        res = res['arxiv_id']
    return res

def search_ref_by_id(arxiv_id):
    """根据arxiv_id搜索tex和ar5iv，匹配则返回arxiv_id/sections，否则返回None"""
    res = fetch_ref(arxiv_id.split('v')[0])
    # res包括id/sections原方法仅返回sections
    if res is not None:
        res = res['sections']
    return res


if __name__ == '__main__':
    #print(search_ref_by_id('2103.15808'))
    print(search_id_by_title('Directional Statistics'))
    print(search_id_by_title('Prime numbers: Emergence and victories of bilinear forms decomposition'))