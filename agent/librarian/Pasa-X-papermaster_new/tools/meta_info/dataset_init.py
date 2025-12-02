import json
import os
from tqdm import tqdm  # <- 新增

current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = f"{current_dir}/../../data/arxiv_dataset/arxiv-metadata-oai-snapshot.json"
output_path = f"{current_dir}/../../result/arxiv_datase_metainfo_new.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

def empty_meta():
    return {
        "snippet": None,
        "citedBy": None,
        "link": None,
        "pdfUrl": None,
        "scholar_id": None,
        "publication_info": None,
        "journal": None,
        "paired_score": None,
        "paired_journal": None,
        "journal_alias": None,
        "h5_index": None,
        "h5_median": None,
        "h5_rank": None,
        "IF": None,
        "CiteScore": None,
        "CCF": None,
        "CORE": None,
        "JCR": None,
        "CAS": None,
    }

count = 0
file_size = os.path.getsize(input_path)  # 拿到文件大小，给进度条用

with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout, \
     tqdm(total=file_size, unit='B', unit_scale=True, desc="convert arxiv jsonl") as pbar:

    for line in fin:
        line_stripped = line.strip()
        if not line_stripped:
            pbar.update(len(line.encode('utf-8')))
            continue

        arxiv = json.loads(line_stripped)

        record = {
            "title": arxiv.get("title", "").replace('\n  ', ' '),
            "arxiv_id": arxiv.get("id"),
            "submitter": arxiv.get("submitter"),
            "authors": arxiv.get("authors", "").replace('\n  ', ' '),
            "comments": arxiv.get("comments"),
            "journal-ref": arxiv.get("journal-ref"),
            "doi": arxiv.get("doi"),
            "categories": arxiv.get("categories"),
            "abstract": arxiv.get("abstract", "").replace('\n', ' '),
            "update_date": arxiv.get("update_date"),
            "year": (arxiv.get("update_date") or "0000-00-00").split('-')[0]
        }
        record.update(empty_meta())

        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        count += 1

        # 用当前行的字节数更新进度条，和 file_size 对得上
        pbar.update(len(line.encode('utf-8')))

print(f"✅ Total papers: {count}")
