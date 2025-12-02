import os
import sys
import json
import time
import requests
import tarfile
from threading import Lock
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from search_utils.optimize_utils.paraline import exec_parallel
from search_utils.optimize_utils.parse_utils import parse_tex_folder
from search_utils.optimize_utils.call_ar5iv import search_section_by_arxiv_id


def fetch_eprint(arxiv_id, save_dir):
    """下载arxiv tar.gz源码"""
    url = f'https://arxiv.org/e-print/{arxiv_id}'
    local_path = os.path.join(save_dir, f'{arxiv_id}.tar.gz')
    if os.path.exists(local_path):
        print(f'{local_path} 已存在，跳过下载')
        return local_path
    # headers = {
    #     "User-Agent": "Lynx"
    # }
    resp = requests.get(url, stream=True, timeout=60)
    if 'html' in resp.headers.get('Content-Type', ''):
        print('被 arXiv 拦截，需要验证码，未能下载源码。')
        return None
    if resp.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    else:
        print(f'arxiv tar.gz 下载失败: {arxiv_id}, 状态码: {resp.status_code}')
        return None

def extract_tar_gz(tar_path, extract_dir):
    if os.path.exists(extract_dir):
        print(f'{extract_dir} 已存在，跳过解压')
        return
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError:
        # 
        print(f"{tar_path}不是gzip格式，尝试普通tar")
        try:
            with tarfile.open(tar_path, 'r:') as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"解压失败: {tar_path}，错误: {e}")
            raise
    for i in range(3):
        try:
            os.remove(tar_path)
            return
        except PermissionError as e:
            print(f"第{i+1}次尝试删除失败: {e}")
            time.sleep(0.5)
    print(f"最终删除失败: {tar_path}")

def process_func(arxiv_id, stop_event):
    # tex
    start_time = time.time()
    if stop_event.is_set():
        return None
    print("tex Dealing", arxiv_id)
    
    base_dir = 'optimize/papers'
    paper_dir = os.path.join(base_dir, f'arXiv-{arxiv_id}')
    tar_path = fetch_eprint(arxiv_id, base_dir)
    if tar_path is None:
        res = process_func1(arxiv_id, stop_event)
        if res and res.get('time'):
            res['time'] = time.time() - start_time
        return res
    try:
        extract_tar_gz(tar_path, paper_dir)
    except Exception as e:
        res = process_func1(arxiv_id, stop_event)
        if res and res.get('time'):
            res['time'] = time.time() - start_time
        return res
    try:
        result = parse_tex_folder(paper_dir)
        if not result:
            res = process_func1(arxiv_id, stop_event)
            if res and res.get('time'):
                res['time'] = time.time() - start_time
            return res
        return {
            'arxiv_id': arxiv_id,
            'sections': result,
            'time': time.time() - start_time
        }
    except Exception as e:
        print(f'解析失败: {arxiv_id}, 错误: {e}')
        res = process_func1(arxiv_id, stop_event)
        if res and res.get('time'):
            res['time'] = time.time() - start_time
        return res

def process_func1(arxiv_id, stop_event):
    # ar5iv
    if stop_event.is_set():
        return None
    print("axv Dealing", arxiv_id)
    res = search_section_by_arxiv_id(arxiv_id)
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
    # return ['2304.10147']

# 处理输出
results = []
results_lock = Lock()
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
    exit(0)
