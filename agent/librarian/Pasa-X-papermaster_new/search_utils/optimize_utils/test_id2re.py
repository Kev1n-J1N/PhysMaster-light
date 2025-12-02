import os
import sys
import json
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_utils.optim_utils import search_ref_by_id, get_id2ref
from search_utils.optimize_utils.paraline import exec_parallel
from search_utils.optimize_utils.call_db import search_db3_id
from utils import search_paper_by_arxiv_id, get_db_axv_cnt, search_section_by_arxiv_id

import arxiv
arxiv_client = arxiv.Client(delay_seconds = 0.05)

def search_axv_id(arxiv_id):
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
        print(f"Failed to search arxiv id: {arxiv_id}")
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
            }
            break
    return res

id2ab_lock = threading.Lock()
id2ab_call_cnt = 0
id2ab_suc_cnt = 0
db3_suc_cnt = 0
axv_call_cnt = 0
axv_suc_cnt = 0
def get_id2ab():
    global id2ab_call_cnt, id2ab_suc_cnt, db3_suc_cnt, axv_call_cnt, axv_suc_cnt
    return {
        'id2ab call': id2ab_call_cnt,
        'id2ab success': id2ab_suc_cnt,
        'id2ab success rate': id2ab_suc_cnt / id2ab_call_cnt if id2ab_call_cnt else 0,
        'db3 success': db3_suc_cnt,
        'db3 success rate': db3_suc_cnt / id2ab_call_cnt if id2ab_call_cnt else 0,
        'axv call': axv_call_cnt,
        'axv success': axv_suc_cnt,
        'axv success rate': axv_suc_cnt / axv_call_cnt if axv_call_cnt else 0
    }

def search_abs_by_id(arxiv_id):
    """根据aid搜索paper，返回aid/title/abstract，或者None"""
    global id2ab_call_cnt, id2ab_suc_cnt, db3_suc_cnt, axv_call_cnt, axv_suc_cnt
    with id2ab_lock:
        id2ab_call_cnt += 1
    res = search_db3_id(arxiv_id)
    if res is None:
        with id2ab_lock:
            axv_call_cnt += 1
        res = search_axv_id(arxiv_id)
        if res is not None:
            with id2ab_lock:
                axv_suc_cnt += 1
                id2ab_suc_cnt += 1
    else:
        with id2ab_lock:
            db3_suc_cnt += 1
            id2ab_suc_cnt += 1
    return res

def process_func(arxiv_id, stop_event):
    """改进方法：先用db3和arxiv搜索abstract，再用tex和ar5iv搜索sections"""
    if stop_event.is_set():
        return None
    print("Dealing", arxiv_id)
    # id搜索abstract
    res = search_abs_by_id(arxiv_id)
    if res is None:
        return None
    result = {}
    result['arxiv_id'] = res['arxiv_id']
    result['title'] = res['title'].replace('\n', ' ')
    result['abstract'] = res['abstract']
    result['sections'] = ""
    # id搜索sections
    res = search_ref_by_id(arxiv_id)
    if res is not None:
        result['sections'] = res
    return result


a5v_lock = threading.Lock()
call_a5v_cnt = 0
a5v_sucs_cnt = 0
def get_a5v_cnt():
    global call_a5v_cnt, a5v_sucs_cnt
    return {
        'call a5v': call_a5v_cnt,
        'a5v success': a5v_sucs_cnt,
        'a5v success rate': (a5v_sucs_cnt/call_a5v_cnt) if call_a5v_cnt else 0,
    }

def process_func1(arxiv_id, stop_event):
    """原方法：用db1或者arxiv+ar5iv，根据id搜索paper"""
    global call_a5v_cnt, a5v_sucs_cnt
    if stop_event.is_set():
        return None
    print("Dealing", arxiv_id)
    # id搜索内容
    res = search_paper_by_arxiv_id(arxiv_id)
    if res is None:
        return None
    # 若没有session再搜ar5iv
    if res['sections'] == "":
        try:
            with a5v_lock:
                call_a5v_cnt+=1
            res1 = search_section_by_arxiv_id(arxiv_id, r"~\\cite\{(.*?)\}")
            if res1 is not None:
                with a5v_lock:
                    a5v_sucs_cnt+=1
                res['sections'] = res1
        except:
            print(f"\033[91mError searching sections for paper: {res['title']}\033[0m")
    return res

def input_func(input_path, output_path):
    ids = []
    fin_ids = set()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                ids.append(json.loads(line.strip())['arxiv_id'])
            except Exception as e:
                print(f"input_path解析出错: {e}, 行内容: {line.strip()}")
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    fin_ids.add(json.loads(line.strip())['arxiv_id'])
                except Exception as e:
                    print(f"output_path解析出错: {e}, 行内容: {line.strip()}")
    except FileNotFoundError:
        pass
    res = [id for id in ids if id not in fin_ids]
    print('total:', len(ids), 'finish:', len(fin_ids), 'todo:', len(res))
    return res


results = []
results_lock = threading.Lock()
batch_size = 10
def output_func(result, output_path):
    if result is None:
        return
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
    input_path = 'zyz/data/evaluate/id2ref/test2/ee_1.jsonl'
    output_path = 'zyz/data/evaluate/id2ref/result/db1_axv_a5v_ee.jsonl'
    max_workers = 20
    exec_parallel(input_path, output_path, max_workers, input_func, process_func1, output_func)
    # 在主流程结束后，记得将剩余未满batch_size的结果写入文件
    if results:
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('id2ab', get_db_axv_cnt())
    print('id2ref', get_a5v_cnt())
