import os
import sys
import json
from threading import Lock
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from search_utils.optimize_utils.paraline import exec_parallel
from search_utils.optimize_utils.call_db import search_db1_title, search_db2_title, search_db3_title


def fetch_db(line, stop_event):
    """查找title，返回title-id-abstract/None列表"""
    if stop_event.is_set():
        return None
    print("Dealing", line['title'][:10])
    
    title = line['title']
    return search_db3_title(title)

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
    if result:
        print('Save:', result['arxiv_id'])
        with results_lock:
            results.append(result)
            if len(results) >= batch_size:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                results.clear()


if __name__ == "__main__":
    """
    input_func(input_path, output_path)
    process_func(line, stop_event)
    output_func(result, output_path)
    """
    input_path = r'optimize\evaluate\ti2id\data\test2\all.jsonl'
    output_path = r'optimize\evaluate\ti2id\result\test2\db_fnd.jsonl'
    max_workers = 1
    exec_parallel(input_path, output_path, max_workers, input_func, fetch_db, output_func)
    # 在主流程结束后，记得将剩余未满batch_size的结果写入文件
    if results:
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')