import os
import json
import tempfile
import shutil
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
from tqdm import tqdm
from rapidfuzz import fuzz

os.environ["SERPER_API_KEY"] = "30249291c44b5e8ddd00fdbe2c0b3e9fdc0a1d15"

# ========= 路径 & 配置 =========
INPUT_PATH   = '/data/duyuwen/Pasa-X/result/arxiv_datase_metainfo_new.jsonl'   # 新的，需要补充的
OLD_PATH     = '/data/cxliu/arxiv_dataset_metainfo_boh_1029.jsonl'             # 旧的，已经有meta的
ERROR_LOG    = '/data/duyuwen/Pasa-X/result/serper_errors.log'
MAX_WORKERS  = 300
BATCH_SIZE   = MAX_WORKERS
BUFFER_SIZE  = 1024 * 1024
PARTS_EVERY  = 100      # 每 100 个 batch 写一个分片
OUTPUT_BASE  = INPUT_PATH

SERPER_URL = os.environ.get("SERPER_URL", "https://google.serper.dev/scholar")
SERPER_HEADERS = {
    "X-API-KEY": os.environ.get("SERPER_API_KEY", ""),
    "Content-Type": "application/json"
}

TARGET_KEYS = ["snippet", "citedBy", "link", "pdfUrl", "scholar_id", "publication_info"]


# ========= 工具函数 =========
def preprocess_journal_name(journal_name: str) -> str:
    journal_name = journal_name.lower()
    journal_name = journal_name.replace(".", "").replace("-", "").replace(",", "")
    journal_name = journal_name.strip()
    journal_name = " ".join(journal_name.split())
    return (
        journal_name.replace("\\'", "'")
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
        .replace("$", "")
        .replace("_", "")
        .replace("&", "and")
    )


def serper_search_organic(title: str, url: str = SERPER_URL, headers: Dict = SERPER_HEADERS) -> Dict:
    payload = json.dumps({"q": title})
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"_error": f"request_error: {e.__class__.__name__}: {e}"}
    except json.JSONDecodeError:
        return {"_error": "json_decode_error"}


def get_info_serper(title: str) -> Dict:
    data = serper_search_organic(title)
    if "_error" in data:
        return {"_error": data["_error"]}

    organic = data.get("organic", [])
    if not organic:
        return {"_error": "no_organic_results"}

    item = organic[0]
    result = {
        "title": item.get("title", ""),
        "snippet": item.get("snippet"),
        "citedBy": item.get("citedBy"),
        "link": item.get("link"),
        "pdfUrl": item.get("pdfUrl"),
        "scholar_id": item.get("id"),
        "publication_info": item.get("publicationInfo"),
    }
    for k in TARGET_KEYS:
        result.setdefault(k, None)
    return result


def needs_fill(record: Dict) -> bool:
    # 只有没有 scholar_id 才需要去查
    return record.get("scholar_id") is None


def log_error(logf, line_no: int, reason: str):
    logf.write(f"line {line_no}: {reason}\n")


def _letters_only_len(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha())


def worker(idx: int, title: str):
    """
    返回: (行号, 覆盖patch或None, 错误或None)
    """
    try:
        result = get_info_serper(title)
        if "_error" in result:
            return idx, None, result["_error"]

        title_out = result.get("title", "") or ""
        tin = preprocess_journal_name(title)
        tout = preprocess_journal_name(title_out)
        sim = fuzz.token_set_ratio(tin, tout)
        diff = abs(_letters_only_len(title) - _letters_only_len(title_out))

        if not (sim >= 85 and diff <= 30):
            return idx, None, f"title_not_similar: score={sim}, len_diff={diff}, input:{tin}, searched:{tout}"

        patch = {k: result.get(k, None) for k in TARGET_KEYS}
        return idx, patch, None

    except Exception as e:
        return idx, None, f"exception: {e.__class__.__name__}: {e}"


def _part_filename(base_path: str, part_no: int) -> str:
    root, ext = os.path.splitext(base_path)
    return f"{root}.part{part_no:05d}{ext}"


def _open_new_part(base_path: str, part_no: int):
    dir_name = os.path.dirname(base_path) or "."
    final_part_path = _part_filename(base_path, part_no)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(final_part_path) + ".", suffix=".tmp", dir=dir_name)
    os.close(fd)
    fout = open(tmp_path, "w", encoding="utf-8", buffering=BUFFER_SIZE)
    return fout, tmp_path, final_part_path


def _close_part(fout, tmp_path: str, final_part_path: str, parts_list: list):
    if fout and not fout.closed:
        fout.flush()
        fout.close()
    shutil.move(tmp_path, final_part_path)
    parts_list.append(final_part_path)


def _merge_parts(parts_list: list, final_output_path: str):
    dir_name = os.path.dirname(final_output_path) or "."
    fd, merged_tmp = tempfile.mkstemp(prefix="merged_", suffix=".jsonl", dir=dir_name)
    os.close(fd)
    with open(merged_tmp, "w", encoding="utf-8", buffering=BUFFER_SIZE) as out:
        for p in parts_list:
            with open(p, "r", encoding="utf-8", buffering=BUFFER_SIZE) as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
    shutil.move(merged_tmp, final_output_path)


# ====== 新增：加载旧数据，按 arxiv_id 建索引 ======
def load_old_meta(path: str) -> Dict[str, Dict]:
    """
    把旧文件读进来，按 arxiv_id 索引，
    后面如果新文件里有这个 arxiv_id，就直接用旧的 meta，不调 API。
    """
    old = {}
    if not os.path.exists(path):
        return old
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            aid = d.get("arxiv_id") or d.get("id")
            if not aid:
                continue
            old[aid] = d
    print(f"[old-meta] loaded {len(old)} records from {path}")
    return old


# ========= 主流程 =========
def main():
    total = updated = errors = 0
    next_to_write = 0
    buffer = {}

    batch_no = 0
    batch_futures = []
    parts_paths = []

    # 1) 读取旧数据
    old_meta = load_old_meta(OLD_PATH)

    file_size = os.path.getsize(INPUT_PATH)

    with open(ERROR_LOG, "w", encoding="utf-8") as logf:
        try:
            with open(INPUT_PATH, "r", encoding="utf-8", buffering=BUFFER_SIZE) as fin, \
                 ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, \
                 tqdm(total=file_size, unit='B', unit_scale=True, desc="process new jsonl") as pbar:

                # 开第一个分片
                part_no = 1
                fout, part_tmp, part_final = _open_new_part(OUTPUT_BASE, part_no)

                def _drain_batch_and_maybe_rotate():
                    nonlocal batch_no, batch_futures, fout, part_tmp, part_final
                    nonlocal part_no, next_to_write, errors, updated

                    if not batch_futures:
                        return

                    batch_no += 1
                    t0 = time.perf_counter()
                    for fut in as_completed(batch_futures):
                        i, patch, err = fut.result()
                        d = fut.data
                        if err:
                            errors += 1
                            log_error(logf, i + 1, err)
                        else:
                            for k in TARGET_KEYS:
                                d[k] = patch.get(k, None)
                            updated += 1
                        buffer[i] = json.dumps(d, ensure_ascii=False) + "\n"
                        # 顺序写出
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1
                    t1 = time.perf_counter()
                    print(f"[Batch {batch_no}] {len(batch_futures)} items finished in {t1 - t0:.2f}s "
                          f"({len(batch_futures)/(t1 - t0 + 1e-9):.2f} items/s)")
                    batch_futures.clear()

                    if batch_no % PARTS_EVERY == 0:
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1
                        _close_part(fout, part_tmp, part_final, parts_paths)
                        part_no += 1
                        fout, part_tmp, part_final = _open_new_part(OUTPUT_BASE, part_no)

                # 2) 逐行读新的文件
                for idx, line in enumerate(fin):
                    total += 1
                    line_bytes = len(line.encode('utf-8'))

                    data = json.loads(line)
                    arxiv_id = data.get("arxiv_id") or data.get("id")

                    # 2.1 若旧文件里有这个 arxiv_id，直接用旧的 META 覆盖
                    if arxiv_id and arxiv_id in old_meta:
                        old_d = old_meta[arxiv_id]
                        for k in TARGET_KEYS:
                            data[k] = old_d.get(k)
                        # 不需要调 API，直接写 buffer
                        buffer[idx] = json.dumps(data, ensure_ascii=False) + "\n"
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1
                        pbar.update(line_bytes)
                        continue

                    # 2.2 没有旧数据，按你原来的逻辑决定要不要调 API
                    if needs_fill(data):
                        title = data.get("title", "")
                        fut = ex.submit(worker, idx, title)
                        fut.data = data
                        batch_futures.append(fut)

                        if len(batch_futures) >= BATCH_SIZE:
                            _drain_batch_and_maybe_rotate()
                    else:
                        buffer[idx] = line if line.endswith("\n") else (line + "\n")
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1

                    pbar.update(line_bytes)

                # 文件读完，把最后一批跑完
                _drain_batch_and_maybe_rotate()

                # 把还没写出去的刷掉
                while next_to_write in buffer:
                    fout.write(buffer.pop(next_to_write))
                    next_to_write += 1

                # 关当前分片
                _close_part(fout, part_tmp, part_final, parts_paths)

            # 3) 合并分片
            _merge_parts(parts_paths, OUTPUT_BASE)

        except Exception:
            try:
                if 'fout' in locals() and fout and not fout.closed:
                    fout.close()
                if 'part_tmp' in locals() and os.path.exists(part_tmp):
                    os.remove(part_tmp)
            except Exception:
                pass
            raise

    print(f"Done. total={total}, updated={updated}, errors={errors}")
    print(f"Error log: {ERROR_LOG}")


if __name__ == "__main__":
    main()
