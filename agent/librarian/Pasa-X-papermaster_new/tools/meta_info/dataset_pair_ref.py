import os
import re
import json
import shutil
import tempfile
from tqdm import tqdm
from MetaInfo_Agent import MetaInfoClient

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = f"{current_dir}/../../result/arxiv_datase_metainfo_new.jsonl"
client = MetaInfoClient(data_path)

# 截断：从开头取到第一个 逗号 / 左括号 / 数字 为止
CUT_RE = re.compile(r'^\s*([^,(\d]+)', re.UNICODE)

PAIR_FIELDS = [
    "paired_score", "paired_journal", "journal_alias",
    "h5_index", "h5_median", "h5_rank",
    "IF", "CiteScore", "CCF", "CORE", "JCR", "CAS"
]


def get_journal_cut(journal_ref: str):
    if not isinstance(journal_ref, str):
        return None
    m = CUT_RE.match(journal_ref)
    if not m:
        return None
    cut = m.group(1).strip(' .;:-\t')
    return cut or None


def main(path: str):
    # 大缓冲区加速 I/O
    buf_size = 1024 * 1024
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="metainfo_", suffix=".jsonl", dir=dir_name)
    os.close(fd)

    total = updated = errors = pair = 0

    file_size = os.path.getsize(path)

    try:
        with open(path, "r", encoding="utf-8", buffering=buf_size) as fin, \
             open(tmp_path, "w", encoding="utf-8", buffering=buf_size) as fout, \
             tqdm(total=file_size, unit="B", unit_scale=True, desc="pairing journals") as pbar:

            for line in fin:
                s = line.strip()
                if not s:
                    fout.write("\n")
                    pbar.update(len(line.encode("utf-8")))
                    continue

                total += 1
                try:
                    data = json.loads(s)
                except Exception:
                    errors += 1
                    fout.write(line)  # 原样写回
                    pbar.update(len(line.encode("utf-8")))
                    continue

                jr = data.get("journal-ref")
                if jr is not None:
                    cut = get_journal_cut(jr)
                    if cut:
                        try:
                            info = client.pair_journal_info(cut)
                            if isinstance(info, dict):
                                for k in PAIR_FIELDS:
                                    data[k] = info.get(k)
                                updated += 1
                                if info.get("paired_score", 0) > 0:
                                    pair += 1
                        except Exception:
                            errors += 1

                fout.write(json.dumps(data, ensure_ascii=False))
                fout.write("\n")
                pbar.update(len(line.encode("utf-8")))

        shutil.move(tmp_path, path)
        print(f"Done. total={total}, updated={updated}, pair={pair}, errors={errors}")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    main(data_path)
