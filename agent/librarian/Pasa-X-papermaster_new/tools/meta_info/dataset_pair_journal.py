import os
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from Journal_Agent import JournalClient

data_path = '/data/cxliu/arxiv_dataset_metainfo_boh_1029.jsonl'
PAIR_FIELDS = [
    "paired_score", "paired_journal", "journal_alias",
    "h5_index", "h5_median", "h5_rank",
    "IF", "CiteScore", "CCF", "CORE", "JCR", "CAS"
]

CHUNK_SIZE = 1000
MAX_WORKERS = 64

# 全局共享 client（只加载一次）
shared_client = JournalClient()

def process_chunk(chunk_lines, chunk_index, client):
    updated = errors = pair = 0
    results = []

    for line in chunk_lines:
        s = line.strip()
        if not s:
            results.append(line)
            continue

        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            results.append(line)
            errors += 1
            continue

        jr = data.get("journal")
        try:
            info = client.pair_journal_info(jr)
            if isinstance(info, dict):
                for k in PAIR_FIELDS:
                    data[k] = info.get(k)
                updated += 1
                if info.get('paired_score', 0) > 0:
                    pair += 1
        except Exception:
            errors += 1

        results.append(json.dumps(data, ensure_ascii=False) + "\n")

    return chunk_index, results, (updated, errors, pair), len(chunk_lines)

def read_in_chunks(file_obj, chunk_size):
    chunk = []
    for line in file_obj:
        chunk.append(line)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def main(path: str):
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="metainfo_", suffix=".jsonl", dir=dir_name)
    os.close(fd)

    total_lines = 0
    total_updated = total_errors = total_pair = 0

    try:
        with open(path, "r", encoding="utf-8", buffering=1024*1024) as fin:
            chunks = list(read_in_chunks(fin, CHUNK_SIZE))
            total_lines = sum(len(c) for c in chunks)
            total_chunks = len(chunks)

        print(f"📁 总行数: {total_lines}, 分成 {total_chunks} 个批次 (每批 {CHUNK_SIZE} 行)")

        results_map = {}
        futures = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            for i, chunk in enumerate(chunks):
                future = executor.submit(process_chunk, chunk, i, shared_client)
                futures.append(future)

            # 使用 tqdm 显示进度
            with tqdm(total=total_chunks, desc="🔄 处理中", unit="chunk") as pbar:
                for future in as_completed(futures):
                    idx, lines, (upd, err, pr), chunk_line_count = future.result()
                    results_map[idx] = (lines, (upd, err, pr))
                    total_updated += upd
                    total_errors += err
                    total_pair += pr
                    pbar.update(1)

        # 按顺序写入
        with open(tmp_path, "w", encoding="utf-8", buffering=1024*1024) as fout:
            for i in range(total_chunks):
                lines, _ = results_map[i]
                fout.writelines(lines)

        shutil.move(tmp_path, path)
        print(f"\n✅ 完成！总行数={total_lines}, 已更新={total_updated}, 成功匹配={total_pair}, 错误={total_errors}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

if __name__ == "__main__":
    main(data_path)