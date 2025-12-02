import requests
import os
import re
import json
import time
import arxiv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import warnings
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "../../config/config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
# def proxy_request_get(url, stream=False, verify=True, params=None, headers=None, timeout=None):
#     entry = 'http://{}:{}@27144f3588738de8.arq.na.ipidea.online:2336'.format(
#         "dpzhongtai-zone-custom-region-us", "Aa12345678")
#     return requests.get(url, stream=stream, verify=verify, proxies={
#         'http': entry,
#         'https': entry,
#     }, params=params, headers=headers, timeout=timeout)

JSONL_PATH = "/data/duyuwen/Pasa-X/data/arxiv_dataset/arxiv-metadata-oai-snapshot.json"
ARXIV_ROOT = "/data/duyuwen/Pasa-X/public_data/arxiv_database/arxiv"  # 你图里那个根目录
AR5IV_ROOT = os.path.join(ARXIV_ROOT, "ar5iv_html")

def fetch_eprint(arxiv_id, save_dir, dl_timeout=30):
    """下载arxiv论文压缩包"""
    url = f'https://arxiv.org/e-print/{arxiv_id}'
    # resp = proxy_request_get(url, stream=True, timeout=dl_timeout)
    resp = requests.get(url, stream=True, timeout=dl_timeout)
    if 'html' in resp.headers.get('Content-Type', ''):
        print('被 arXiv 拦截，需要验证码，未能下载源码。')
        return None
    # 优先从Content-Disposition获取原始文件名
    fname = None
    cd = resp.headers.get('Content-Disposition')
    if cd:
        m = re.search('filename="?([^";]+)"?', cd)
        if m:
            fname = m.group(1)
    if not fname:
        # 兜底：用原有逻辑
        fname = f'{arxiv_id}.tar.gz'
    local_path = os.path.join(save_dir, fname)
    if '.tar.gz' not in local_path:
        print(f'文件名不正确: {fname}')
        return None
    if os.path.exists(local_path):
        print(f'{local_path} 已存在，跳过下载')
        return local_path
    if resp.status_code == 200:
        start = time.time()
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=4096):
                if time.time() - start > dl_timeout:
                    print(f"tex下载超时，自动退出: {arxiv_id}")
                    return None
                f.write(chunk)
        return local_path
    else:
        print(f'arxiv 源码包下载失败: {arxiv_id}, 状态码: {resp.status_code}')
        return None
    
def download_tex(arxiv_id,save_dir):
    local_path = fetch_eprint(arxiv_id, save_dir)
    if local_path is None:
        return arxiv_id, False
    else:
        return arxiv_id, True
    
def parse_ar5iv(entry_id,save_dir):
    try:
        url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'
        data_path = f"{config['arxiv_database_path']}/arxiv/ar5iv_html/{entry_id.split('.')[0]}/{entry_id}.html"
        if os.path.exists(data_path):
            return entry_id,False
        else:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                html_content = response.text
            else:
                warnings.warn(f'Request failed: {url}, status code: {response.status_code}')
                return entry_id,False
        if html_content:
            if not 'https://ar5iv.labs.arxiv.org/html' in html_content:
                warnings.warn(f'Invalid ar5iv HTML document: {url}')
                return entry_id,False
            else:
                print(f"save {data_path}")
                save_path = data_path
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                if not os.path.exists(save_path):
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                return entry_id,True
    except Exception as e:  
        print(f"parse ar5iv failed: {entry_id}, {e}")
        return entry_id,False

pattern = re.compile(r"^\d{4}\.")  # 前4位数字，后面一个点

def build_prefix_to_ids(jsonl_path):
    """从 arxiv jsonl 里把所有 id 按前缀分好，只保留形如 1234.xxx 的老式id"""
    prefix2ids = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            arxiv_id = d.get("id")
            if not arxiv_id:
                continue

            # 只保留前4位是数字且第5位是'.'的
            if not pattern.match(arxiv_id):
                continue

            prefix = arxiv_id.split(".")[0]  # "0704.0001" -> "0704"
            prefix2ids[prefix].append(arxiv_id)
    return prefix2ids


def main():
    prefix2ids = build_prefix_to_ids(JSONL_PATH)
    success_log = []
    failed_log = []
    count = 0

    for idx, prefix in enumerate(sorted(prefix2ids.keys(),reverse=True), start=1):
        print(f"================= 处理第 {idx} 个前缀目录：{prefix} ==================")

        # 存放 ar5iv 的目录
        save_dir = os.path.join(AR5IV_ROOT, prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            downloaded_ids = set()
        else:
            # 已经下载的文件名去掉后缀（.html/.htm）
            downloaded_ids = {fname.split(".h")[0] for fname in os.listdir(save_dir)}

        # 得到这个前缀下的所有 id
        all_ids_this_prefix = prefix2ids[prefix]
        # 关键：按降序来（最新在前）
        all_ids_this_prefix.sort(reverse=True)

        # 从最新往前下，一旦遇到已经有的就停
        wait_arxiv_ids = []
        for aid in all_ids_this_prefix:
            if aid in downloaded_ids:
                # 遇到已经下过的就认为这个前缀的旧的都不用再下了
                break
            wait_arxiv_ids.append(aid)

        if not wait_arxiv_ids:
            print(f"[{prefix}] 没有需要下载的，跳过")
            continue

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            progress_bar = tqdm(
                total=len(wait_arxiv_ids),
                desc=f"{prefix} 下载进度",
                ncols=100,
                position=0,
                leave=True,
            )

            for arxiv_id in wait_arxiv_ids:
                futures.append(executor.submit(parse_ar5iv, arxiv_id, save_dir))

            for future in as_completed(futures):
                try:
                    arxiv_id, success = future.result()
                except Exception as e:
                    # 线程里面炸掉
                    msg = f"[{prefix}] 下载异常：{e}"
                    print(msg)
                    failed_log.append(msg)
                    count += 1
                    progress_bar.update(1)
                    continue

                if success:
                    print(f"[{prefix}] {arxiv_id} 下载成功")
                    success_log.append(arxiv_id)
                else:
                    print(f"[{prefix}] {arxiv_id} 下载失败")
                    failed_log.append(arxiv_id)

                count += 1
                progress_bar.update(1)

            progress_bar.close()

    print(f"总共下载 {count} 个文件")
    print(f"成功下载: {len(success_log)}, 失败: {len(failed_log)}")

    with open("success_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(success_log))
    with open("failed_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(failed_log))

if __name__ == "__main__":
    main()
                    
# if __name__ == "__main__":
#     # dataset_path = "/data/public_data/arxiv_database/arxiv/pdf"
#     # dir_list = os.listdir(dataset_path)
#     arxiv_log = []
#     failed_log = []
#     success_log = []
#     count = 0
#     idx = 1
#     # for dir_name in sorted(dir_list):
#     #     print(f"================= 下载第{idx}个目录：{dir_name} ==================")
#     #     dir_path = os.path.join(dataset_path, dir_name)
#     #     files_name = os.listdir(dir_path)
#     #     save_dir = dir_path.replace("pdf", "tex")
#     #     if not os.path.exists(save_dir):
#     #         os.makedirs(save_dir)
#     #         download_files = []
#     #     else:
#     #         download_files = os.listdir(save_dir)
#     #         download_files = [f.split('-')[-1].split('v')[0] for f in download_files]
            
#     #     wait_arxiv_id = []
#     #     for file_name in sorted(files_name):
#     #         arxiv_id = file_name.split('v')[0]
#     #         if arxiv_id not in download_files:
#     #             wait_arxiv_id.append(arxiv_id)
                
#     #     with ThreadPoolExecutor(max_workers=1) as executor:
#     #         futures = []
#     #         progress_bar = tqdm(total=len(wait_arxiv_id), desc="下载进度", ncols=100, position=0, leave=True)
#     #         for arxiv_id in wait_arxiv_id:
#     #             futures.append(executor.submit(download_tex, arxiv_id, save_dir))
                
#     #         for future in as_completed(futures):
#     #             arxiv_id, success = future.result()
#     #             if success:
#     #                 success_log.append(arxiv_id)
#     #             else:
#     #                 failed_log.append(arxiv_id)
#     #             count += 1
#     #             progress_bar.update(1)

#     #         progress_bar.close()
        
#     # print(f"总共下载{count}个文件")
#     # print(f"成功下载: {len(success_log)}, 失败: {len(failed_log)}")
#     # with open("success_log.txt", "w") as f:
#     #     f.write("\n".join(success_log))
#     # with open("failed_log.txt", "w") as f:
#     #     f.write("\n".join(failed_log))
    
#     for dir_name in sorted(dir_list):
#         print(f"================= 下载第{idx}个目录：{dir_name} ==================")
#         dir_path = os.path.join(dataset_path, dir_name)
#         files_name = os.listdir(dir_path)
#         save_dir = dir_path.replace("pdf", "ar5iv_html")
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#             download_files = []
#         else:
#             download_files = os.listdir(save_dir)
#             download_files = [f.split('.h')[0] for f in download_files]
            
#         wait_arxiv_id = []
#         for file_name in sorted(files_name):
#             arxiv_id = file_name.split('v')[0]
#             if arxiv_id not in download_files:
#                 wait_arxiv_id.append(arxiv_id)
                
#         with ThreadPoolExecutor(max_workers=10) as executor:
#             futures = []
#             progress_bar = tqdm(total=len(wait_arxiv_id), desc="下载进度", ncols=100, position=0, leave=True)
#             for arxiv_id in wait_arxiv_id:
#                 futures.append(executor.submit(parse_ar5iv, arxiv_id,save_dir))
                
#             for future in as_completed(futures):
#                 arxiv_id, success = future.result()
#                 if success:
#                     success_log.append(arxiv_id)
#                 else:
#                     failed_log.append(arxiv_id)
#                 count += 1
#                 progress_bar.update(1)

#             progress_bar.close()
        
#     print(f"总共下载{count}个文件")
#     print(f"成功下载: {len(success_log)}, 失败: {len(failed_log)}")
#     with open("success_log.txt", "w") as f:
#         f.write("\n".join(success_log))
#     with open("failed_log.txt", "w") as f:
#         f.write("\n".join(failed_log))