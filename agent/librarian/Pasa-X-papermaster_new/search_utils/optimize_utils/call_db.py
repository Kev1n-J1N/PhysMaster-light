import os
import sys
import json
import zipfile
import threading
sys.path.append((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from search_utils.optimize_utils.sim_title import normalize_title

def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

current_path = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

# db1
with open(f"{base_path}/data/paper_database/id2paper.json", encoding="utf-8") as f:
    db1_idx_aid2title = json.load(f)
with open(f"{base_path}/search_utils/data/paper_db/db1_idx_sim2aidtitle.json", encoding="utf-8") as f:
    db1_idx_sim2aidtitle = json.load(f)

db1_path = f"{base_path}/data/paper_database/cs_paper_2nd.zip"
db1_zip = zipfile.ZipFile(db1_path, 'r')
db1_namelist = set(db1_zip.namelist())
db1_lock = threading.Lock()

# db2_idx_aid2pos = load_jsonl_to_dict(f"{base_path}/pasa1/data/paper_db/db2_idx_aid2pos.jsonl")
# db2_idx_sim2aid = load_jsonl_to_dict(f"{base_path}/pasa1/data/paper_db/db2_idx_sim2aid.jsonl")
# db2_path = f"{base_path}/pasa1/data/paper_db/db2.jsonl"
# db2_lock = threading.Lock()
# db2_file = open(db2_path, 'r', encoding="utf-8")
# db2_file_lock = threading.Lock()

# db3
with open(f"{base_path}/search_utils/data/paper_db/db3_idx_sim2pos.json", encoding="utf-8") as f:
    db3_idx_sim2pos = json.load(f)
with open(f"{base_path}/search_utils/data/paper_db/db3_idx_aid2pos.json", encoding="utf-8") as f:
    db3_idx_aid2pos = json.load(f)
db3_path = f"{base_path}/search_utils/data/paper_db/db3.json"
db3_file = open(db3_path, 'r', encoding="utf-8")
db3_file_lock = threading.Lock()


def search_db1_id(aid):
    """输入aid，返回论文完整信息，找不到返回None"""
    if aid in db1_idx_aid2title:
        title_key = keep_letters(db1_idx_aid2title[aid])
        if title_key in db1_namelist:
            with db1_lock:
                with db1_zip.open(title_key) as f:
                    data = json.loads(f.read().decode("utf-8"))
            # 获取sections字段
            # sv = data.get('sections', {})
            # 合并所有title
            # all_titles = []
            # for v in sv.values():
            #     if isinstance(v, list):
            #         all_titles.extend(v)
            # all_titles = list(set(all_titles))
            # 修改为kv对
            # references_kv = {f"ref{i+1}": v for i, v in enumerate(all_titles)}
            # data['references'] = references_kv
            return {
                'arxiv_id': aid,
                "title": data["title"].replace("\n", " "),
                "abstract": data["abstract"],
                "sections": data["sections"],
                "source": 'SearchFrom:db1',
                "journal": "",
                "authors": "",
                "year": ""
            }
    return None

def search_db1_title(title):
    simtitle = normalize_title(title)
    if simtitle in db1_idx_sim2aidtitle:
        aid = db1_idx_sim2aidtitle[simtitle][0]
        return {
            'arxiv_id': aid
        }
        # return search_db1_aid(aid)
    return None

# def search_db2_aid(aid):
#     """输入aid，返回论文完整信息，找不到返回None"""
#     pos = db2_idx_aid2pos.get(aid)
#     if pos is None:
#         return None
#     with db2_file_lock:
#         db2_file.seek(pos)
#         line = db2_file.readline()
#         data = json.loads(line)
#         return data.get(aid)

# def search_db2_title(title):
#     simtitle = normalize_title(title)
#     aid = db2_idx_sim2aid.get(simtitle)
#     if aid is None:
#         return None
#     return search_db2_aid(aid)

# def add_db2(result):
#     if not isinstance(result, dict) or 'arxiv_id' not in result or 'title' not in result:
#         print('db: data error')
#         return False
#     aid = result['arxiv_id']
#     title = result['title']
#     simtitle = normalize_title(title)
#     try:
#         with db2_lock:
#             # 1. 写入db2.jsonl
#             with open(db2_path, "ab+") as f:
#                 f.seek(0, 2)
#                 pos = f.tell()
#                 line = (json.dumps({aid: result}, ensure_ascii=False) + '\n').encode('utf-8')
#                 f.write(line)
#             # 2. 更新db2_idx_aid2pos.jsonl
#             with open("pasa1/data/paper_db/db2_idx_aid2pos.jsonl", "a", encoding="utf-8") as f:
#                 f.write(json.dumps({aid: pos}, ensure_ascii=False) + '\n')
#             # 3. 更新db2_idx_sim2aid.jsonl
#             with open("pasa1/data/paper_db/db2_idx_sim2aid.jsonl", "a", encoding="utf-8") as f:
#                 f.write(json.dumps({simtitle: aid}, ensure_ascii=False) + '\n')
#         print('db: success')
#         return True
#     except Exception as e:
#         print('db: ', e)
#         return False

def search_db3_title(title):
    simtitle = normalize_title(title)
    pos = db3_idx_sim2pos.get(simtitle)
    if pos is None:
        return None
    with db3_file_lock:
        db3_file.seek(pos)
        line = db3_file.readline()
        data = json.loads(line)
        return {
            'arxiv_id': data['id'],
            'title': data['title'],
            'abstract': data['abstract']
        }

def search_db3_id(arxiv_id):
    pos = db3_idx_aid2pos.get(arxiv_id)
    if pos is None:
        return None
    with db3_file_lock:
        db3_file.seek(pos)
        line = db3_file.readline()
        data = json.loads(line)
        publish_date_str = data['versions'][0]["created"].split(',')[-1].strip().split(' ')
        convert = {
            "Jan": "01","Feb": "02","Mar": "03","Apr": "04","May": "05","Jun": "06",
            "Jul": "07","Aug": "08","Sep": "09","Oct": "10","Nov": "11","Dec": "12"
        }
        publish_date = f"{publish_date_str[2]}-{convert[publish_date_str[1]]}-{publish_date_str[0]}"
        return {
            'arxiv_id': data['id'],
            'title': data['title'],
            'abstract': data['abstract'],
            "journal": data['journal-ref'],
            "authors": data['authors'],
            "year": publish_date,
            'sections': "",
            'source': 'SearchFrom:db3'
        }

def cleanup_resources():
    """清理资源，关闭文件句柄"""
    if hasattr(cleanup_resources, 'db3_file_closed') and not cleanup_resources.db3_file_closed:
        db3_file.close()
        cleanup_resources.db3_file_closed = True

# 标记文件未关闭
cleanup_resources.db3_file_closed = False

if __name__ == '__main__':
    # exit(0)
    print(search_db1_id('2007.06843'))
    # print(search_db3_title('DDI-COCO: A DATASET FOR UNDERSTANDING THE EFFECT OF COLOR CONTRAST IN MACHINE-ASSISTED SKIN DISEASE DETECTION'))