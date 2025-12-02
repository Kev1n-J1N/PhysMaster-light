#!/usr/bin/env python3
import os
import csv
import io
import json
import subprocess
import unicodedata,re
from tqdm import tqdm
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_REF = "Cornell-University/arxiv"
DOWNLOAD_DIR = f"{current_dir}/../data/arxiv_dataset"
STATE_FILE = os.path.join(DOWNLOAD_DIR, "kaggle_arxiv_state.json")

def ensure_dir():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def list_remote_files():
    """
    列出这个 Kaggle 数据集里所有文件，返回 {name: {size: ..., date: ...}}
    kaggle datasets files ... --csv 会给 name/size/date 这些列
    """
    result = subprocess.run(
        ["kaggle", "datasets", "files", DATASET_REF, "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    files = {}
    for row in reader:
        name = row.get("name")
        if not name:
            continue
        files[name] = {
            "size": row.get("size", ""),
            "date": row.get("date", ""),  # 有的 kaggle 版本叫 "date" 或 "creationDate"
        }
    return files


def load_local_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_local_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def download_one_file(fname: str):
    """
    只下载数据集里的某一个文件
    """
    cmd = [
        "kaggle",
        "datasets",
        "download",
        DATASET_REF,
        "-f",
        fname,
        "-p",
        DOWNLOAD_DIR,
        "--unzip",
    ]
    print("running:", " ".join(cmd))
    # 让 kaggle 自己打进度条
    subprocess.run(cmd, check=True)


def main():
    ensure_dir()

    # 1. 远端有哪些文件
    remote_files = list_remote_files()

    # 2. 本地状态（已经下过哪些）
    state = load_local_state()
    downloaded = state.get("downloaded_files", {})  # {name: {size: ...}}

    # 3. 计算需要下载的文件：远端有，本地没有，或者大小变了
    to_download = []
    for name, meta in remote_files.items():
        remote_size = meta.get("size", "")
        local_meta = downloaded.get(name)
        if not local_meta:
            to_download.append((name, remote_size))
        else:
            # 如果你想更严一点，可以比较 size
            if local_meta.get("size") != remote_size:
                to_download.append((name, remote_size))

    if not to_download:
        print("没有需要增量下载的文件，本地是最新的。")
    else:
        print(f"发现 {len(to_download)} 个新文件/变动文件，需要下载：")
        for name, _ in to_download:
            print(" -", name)
        for name, size in to_download:
            download_one_file(name)
            # 下载成功就写到状态里
            downloaded[name] = {"size": size, "downloadedAt": datetime.utcnow().isoformat() + "Z"}

        # 写回状态
        state["downloaded_files"] = downloaded
        state["last_checked"] = datetime.utcnow().isoformat() + "Z"
        save_local_state(state)
        print("增量下载完成。")

def normalize_title(text):
    """将标题转换为纯字符"""
    text = unicodedata.normalize('NFKC', text) # Unicode标准化
    text = text.lower() # 转小写
    text = re.sub(r'[^a-z0-9]', '', text) # 只保留字母和数字
    return text

def gen_title_idx_db3():
    
    input_path = f'{current_dir}/../data/arxiv_dataset/arxiv-metadata-oai-snapshot.json'
    output_path = f'{current_dir}/../data/arxiv_dataset/arxiv_idx_sim2pos.json'

    file_size = os.path.getsize(input_path)
    index = {}

    with open(input_path, 'r', encoding='utf-8') as f:
        offset = 0
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="build title->offset")
        for line in f:
            # 用 utf-8 的字节长度来更新 offset，和你的写法一致
            line_bytes = line.encode('utf-8')
            try:
                data = json.loads(line)
                title = normalize_title(data.get('title'))
                if title:
                    index[title] = offset
            except Exception as e:
                print(f"跳过异常行: {e}")
            offset += len(line_bytes)
            pbar.update(len(line_bytes))
        pbar.close()

    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(index, fout, ensure_ascii=False)
    print("total titles:", len(index))

def gen_db3_aid2pos():
    """
    为 arxiv jsonl 生成 {arxiv_id: byte_offset} 的索引文件
    输出格式示例：
    {
        "0704.0001": 0,
        "0704.0002": 1689,
        ...
    }
    """
    jsonl_path = f"{current_dir}/../data/arxiv_dataset/arxiv-metadata-oai-snapshot.json"
    index_path = f"{current_dir}/../data/arxiv_dataset/arxiv_idx_aid2pos.json"

    file_size = os.path.getsize(jsonl_path)
    aid2offset = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="build aid->offset")
        while True:
            offset = f.tell()      # 这一行的起始字节位置
            line = f.readline()
            if not line:
                break
            if line.strip():
                try:
                    d = json.loads(line)
                    # arxiv 每行格式基本都是 {"id": "...", ...}
                    aid = d.get("id")
                    if aid:
                        aid2offset[aid] = offset
                except Exception as e:
                    # 出问题就跳过这一行
                    # print(f"skip line at offset {offset}: {e}")
                    pass
            # 用 utf-8 字节数更新进度条，和 offset 计算保持一致
            pbar.update(len(line.encode("utf-8")))
        pbar.close()

    # 把 {id: offset} 整个写成 JSON
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(aid2offset, f, ensure_ascii=False)

    print(f"total id: {len(aid2offset)}")
    print(f"index saved to: {index_path}")

if __name__ == "__main__":
    # main()
    gen_title_idx_db3()
    # gen_db3_aid2pos()
