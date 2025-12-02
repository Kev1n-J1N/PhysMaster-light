import os
import json
import tempfile
import shutil
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
import hashlib
import pytz
from datetime import datetime
from tqdm import tqdm
from rapidfuzz import fuzz
import glob
import re

# ========= 新旧文件路径 =========
NEW_INPUT_PATH = '/data/duyuwen/Pasa-X/result/arxiv_datase_metainfo_new.jsonl'   # 新的，需要补
OLD_META_PATH  = '/data/cxliu/arxiv_dataset_metainfo_boh_1029.jsonl'            # 旧的，已经查过的
ERROR_LOG      = '/data/cxliu/bohrium_errors.log'

# 输出还是按新的来
OUTPUT_BASE = '/data/cxliu/arxiv_dataset_metainfo_boh.jsonl'

# ========= 并发 & 分片配置 =========
MAX_WORKERS  = 500
BATCH_SIZE   = MAX_WORKERS
BUFFER_SIZE  = 1024 * 1024
PARTS_EVERY  = 1000  # 每多少个 batch 落一个分片

# ========= Bohrium API 鉴权 =========
os.environ["PAPER_ACCESS_KEY"] = "CPfApvbkFUsDtSaSLYvrwPW5D8gJBgrJ"
os.environ["PAPER_ACCESS_SECRET"] = "93BrrMVL3EVnseyYf09jv1rTmSYeTHax"
PAPER_ACCESS_KEY = os.getenv("PAPER_ACCESS_KEY")
PAPER_ACCESS_SECRET = os.getenv("PAPER_ACCESS_SECRET")

TARGET_KEYS = ["doi", "journal", "keyword", "citedBy", "cited_update_date"]

# 统计用
ALL_PAPER = 0
FIND_ARXIV_PAPER = 0
FIND_JOURNAL_PAPER = 0
FIND_CITATION = 0
FIND_KEYWORD = 0


def preprocess_journal_name(journal_name: str) -> str:
    journal_name = journal_name.lower()
    journal_name = journal_name.replace(".", "").replace("-", "").replace(",", "")
    journal_name = journal_name.strip()
    journal_name = " ".join(journal_name.split())
    return journal_name.replace("\\'", "'").replace("{", "").replace("}", "").replace("\\", "").replace("$", "").replace("_", "").replace("&", "and")


def get_digester_info(access_key, access_secret):
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(shanghai_tz)
    current_minutes = current_time.strftime('%Y%m%d%H%M')
    data = access_key + access_secret[:10] + current_minutes
    digester = hashlib.sha512(data.encode()).hexdigest()
    return digester


def submit_paper_title(title: str, max_retries: int = 5, retry_delay: float = 1.0):
    digester = get_digester_info(PAPER_ACCESS_KEY, PAPER_ACCESS_SECRET)
    url = "https://engine.bohrium.com/paper/pass/title"
    headers = {"Content-Type": "application/json"}
    payload = {
        "accessKey": PAPER_ACCESS_KEY,
        "digester": digester,
        "title": title,
        "status": None,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"request_error (attempt {attempt}/{max_retries}): {e.__class__.__name__}: {e}"
            if attempt == max_retries:
                return {"_error": error_msg}
            else:
                time.sleep(retry_delay)
        except json.JSONDecodeError:
            return {"_error": "json_decode_error"}
    return {"_error": f"unexpected failure after {max_retries} retries"}


def get_info(title: str) -> Dict:
    response = submit_paper_title(title)
    if "_error" in response:
        return {"_error": response["_error"]}

    item = response.get("data", [])
    if not item:
        return {"_error": response}

    result = {
        "title": item.get("enName", ""),
        "doi": item.get("doi", None),
        "journal": item.get("publicationEnName", None),
        "keyword": [d['enName'] for d in item.get("keywordList", [])] if item.get("keywordList") else None,
        "citedBy": item.get("citationNums", None),
        "cited_update_date": item.get("coverDateStart", None),
    }
    return result


def needs_fill(record: Dict) -> bool:
    # 你之前是 journal 为 None 就补
    return record.get("journal") is None


def log_error(logf, line_no: int, reason: str):
    logf.write(f"line {line_no}: {reason}\n")


def _letters_only_len(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha())


def worker(idx: int, title: str, data: Dict):
    global FIND_ARXIV_PAPER, FIND_JOURNAL_PAPER, FIND_CITATION, FIND_KEYWORD
    try:
        result = get_info(title)
        if "_error" in result:
            return idx, None, result["_error"]

        title_out = result.get("title", "") or ""
        tin = preprocess_journal_name(title)
        tout = preprocess_journal_name(title_out)
        sim = fuzz.token_set_ratio(tin, tout)
        diff = abs(_letters_only_len(title) - _letters_only_len(title_out))

        if not (sim >= 85 and diff <= 30):
            return idx, None, f"title_not_similar: score={sim}, len_diff={diff}, input:{tin}, searched:{tout}"

        # 如果 DOI 看起来就是 arXiv 的，跳过
        if result.get('doi', '').lower().find('arxiv') >= 0:
            FIND_ARXIV_PAPER += 1
            return idx, None, "arxiv"

        patch = {}
        patch["doi"] = result["doi"]

        if data.get("journal") is None and result["journal"] is not None:
            patch["journal"] = result["journal"]
        if result["journal"] is not None:
            FIND_JOURNAL_PAPER += 1

        patch["keyword"] = result["keyword"]
        if patch["keyword"] is not None:
            FIND_KEYWORD += 1

        if data.get("citedBy") is None and result["citedBy"] is not None:
            patch["citedBy"] = result["citedBy"]
            patch["cited_update_date"] = result["cited_update_date"]
            if patch["citedBy"] is not None:
                FIND_CITATION += 1

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


# ====== 新增：加载旧的数据集，按 arxiv_id 索引 ======
def load_old_meta(path: str) -> Dict[str, Dict]:
    old = {}
    if not os.path.exists(path):
        print(f"[old-meta] {path} not found, skip.")
        return old
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as f, \
         tqdm(total=size, unit="B", unit_scale=True, desc="load old meta") as pbar:
        for line in f:
            pbar.update(len(line.encode("utf-8")))
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


def main():
    global FIND_ARXIV_PAPER, FIND_JOURNAL_PAPER, FIND_CITATION, FIND_KEYWORD, ALL_PAPER

    # 1) 先读旧的数据，后面能直接用
    old_meta = load_old_meta(OLD_META_PATH)

    total = updated = errors = 0
    next_to_write = 0
    buffer = {}

    batch_no = 0
    batch_futures = []
    parts_paths = []

    # ====== 恢复处理逻辑（如果之前跑过，有分片）======
    root, ext = os.path.splitext(OUTPUT_BASE)
    part_pattern = f"{root}.part*.jsonl"
    part_files = glob.glob(part_pattern)

    total_processed = 0
    max_part_no = 0
    part_re = re.compile(rf"{re.escape(root)}\.part(\d{{5}}){re.escape(ext)}$")

    for f in part_files:
        match = part_re.search(f)
        if match:
            pn = int(match.group(1))
            max_part_no = max(max_part_no, pn)
            with open(f, 'r', encoding='utf-8') as ff:
                total_processed += sum(1 for line in ff if line.strip())

    part_no = max_part_no + 1
    print(f"[Resume] Found {len(part_files)} part files. processed={total_processed}, next part={part_no}")

    file_size = os.path.getsize(NEW_INPUT_PATH)

    with open(ERROR_LOG, "w", encoding="utf-8") as logf:
        try:
            with open(NEW_INPUT_PATH, "r", encoding="utf-8", buffering=BUFFER_SIZE) as fin, \
                 ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, \
                 tqdm(total=file_size, unit='B', unit_scale=True, desc="process new jsonl") as pbar:

                # 跳过之前已经处理过的行
                for _ in range(total_processed):
                    try:
                        line = next(fin)
                        pbar.update(len(line.encode('utf-8')))
                    except StopIteration:
                        break

                fout, part_tmp, part_final = _open_new_part(OUTPUT_BASE, part_no)

                def _drain_batch_and_maybe_rotate():
                    nonlocal batch_no, batch_futures, fout, part_tmp, part_final, part_no, next_to_write, errors, updated
                    if not batch_futures:
                        return
                    batch_no += 1
                    t0 = time.perf_counter()
                    for fut in as_completed(batch_futures):
                        i, patch, err = fut.result()
                        d = fut.data
                        if err == "arxiv":
                            # 不写入 patch，但也不算错误
                            pass
                        elif err:
                            errors += 1
                            log_error(logf, i + 1, err)
                        else:
                            for k, v in patch.items():
                                d[k] = v
                            updated += 1
                        buffer[i] = json.dumps(d, ensure_ascii=False) + "\n"
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1
                    t1 = time.perf_counter()
                    print(f"[Batch {batch_no}] {len(batch_futures)} items in {t1 - t0:.2f}s "
                          f"arxiv: {FIND_ARXIV_PAPER}, journal: {FIND_JOURNAL_PAPER}, "
                          f"keyword: {FIND_KEYWORD}, citation: {FIND_CITATION}")
                    batch_futures.clear()

                    if batch_no % PARTS_EVERY == 0:
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1
                        _close_part(fout, part_tmp, part_final, parts_paths)
                        part_no += 1
                        fout, part_tmp, part_final = _open_new_part(OUTPUT_BASE, part_no)

                # 逐行处理新的文件
                for idx, line in enumerate(fin):
                    total += 1
                    line_bytes = len(line.encode('utf-8'))
                    data = json.loads(line)
                    arxiv_id = data.get("arxiv_id") or data.get("id")

                    # ---- 如果这条在旧文件里，就直接贴旧的元信息，不调 API ----
                    if arxiv_id and arxiv_id in old_meta:
                        old_d = old_meta[arxiv_id]
                        for k in TARGET_KEYS:
                            data[k] = old_d.get(k)
                        buffer[idx] = json.dumps(data, ensure_ascii=False) + "\n"
                        while next_to_write in buffer:
                            fout.write(buffer.pop(next_to_write))
                            next_to_write += 1
                        pbar.update(line_bytes)
                        continue

                    # ---- 不在旧文件里，按需要再调 API ----
                    if needs_fill(data):
                        ALL_PAPER += 1
                        title = data.get("title", "")
                        fut = ex.submit(worker, idx, title, data)
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

                # 刷剩下的
                while next_to_write in buffer:
                    fout.write(buffer.pop(next_to_write))
                    next_to_write += 1

                _close_part(fout, part_tmp, part_final, parts_paths)

            # 合并所有分片
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
