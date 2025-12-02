import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from search_utils.optimize_utils.sim_title import normalize_title


def clean_null():
    in_path = 'pasa1/data/paper_db/db2.jsonl'
    out_path = 'pasa1/data/paper_db/db2_clean.jsonl'
    seen = set()
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict) and len(d) == 1:
                    key = list(d.keys())[0]
                    value = d[key]
                    # 跳过形如{aid: null}的行
                    if value is None:
                        continue
                    # 按aid去重
                    if key not in seen:
                        seen.add(key)
                        fout.write(json.dumps(d, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f'跳过异常行: {e}')

def gen_title_idx():
    # db1
    input_path = "pasa1/data/paper_db/db1_idx_aid2title.json"
    out_path = "pasa1/data/paper_db/db1_idx_simtitle2aidtitle1.json"
    with open(input_path, "r", encoding="utf-8") as f:
        db1_idx_aid2title = json.load(f)
    simtitle2aidtitle = {}
    for aid, title in db1_idx_aid2title.items():
        simtitle = normalize_title(title)
        simtitle2aidtitle[simtitle] = [aid, title]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(simtitle2aidtitle, f, ensure_ascii=False, indent=2)
    # db2
    # jsonl_path = 'pasa1/data/paper_db/db2.jsonl'
    # out_path = 'pasa1/data/paper_db/db2_idx_sim2aid1.jsonl'
    # simtitle2aid = {}
    # with open(jsonl_path, 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         if not line.strip():
    #             continue
    #         try:
    #             d = json.loads(line)
    #             if isinstance(d, dict) and len(d) == 1:
    #                 aid, info = list(d.items())[0]
    #                 if info is not None and 'title' in info:
    #                     title = info['title']
    #                     simtitle = normalize_title(title)
    #                     simtitle2aid[simtitle] = aid
    #         except Exception as e:
    #             print(f"跳过异常行: {e}")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     for sim, aid in simtitle2aid.items():
    #         f.write(json.dumps({sim: aid}, ensure_ascii=False) + "\n")
    # print(f"已生成simtitle2aid索引文件 {out_path}，共 {len(simtitle2aid)} 条记录")

def gen_title_idx_db3():
    input_path = r'pasa1\data\paper_db\db3.json'
    output_path = r'pasa1\data\paper_db\db3_idx_sim2pos1.json'
    index = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        offset = 0
        for line in f:
            try:
                data = json.loads(line)
                title = normalize_title(data.get('title'))
                if title:
                    index[title] = offset
            except Exception as e:
                print(f"跳过异常行: {e}")
            offset += len(line.encode('utf-8'))  # 用utf-8编码长度
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(index, fout, ensure_ascii=False)
    print(len(index))

def gen_db2_aid2pos():
    """为jsonl文件生成aid到字节偏移量的索引文件"""
    jsonl_path = 'pasa1\data\paper_db\db2.jsonl'
    index_path = 'pasa1\data\paper_db\db2_idx_aid2pos1.jsonl'
    aid2offset = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                try:
                    d = json.loads(line)
                    for aid in d.keys():
                        aid2offset[aid] = offset
                except Exception:
                    pass
    with open(index_path, "w", encoding="utf-8") as f:
        for aid, offset in aid2offset.items():
            f.write(json.dumps({aid: offset}, ensure_ascii=False) + "\n")

def check_data():
    """db3是一个jsonl文件"""
    input_path = r'pasa1\data\paper_db\db3.json'
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    print(f"总行数：{count}")

def extract_last_data(input_path):
    """从JSON文件中提取最后一个数据项"""
    with open(input_path, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)
    
    # 获取最后一个键值对
    if isinstance(data, dict) and data:
        last_key = list(data.keys())[-1]
        last_value = data[last_key]
        print(f"已提取最后一个数据: {last_key} -> {last_value}")
    else:
        print("数据为空或格式不正确")
        return None
    

if __name__ == '__main__':
    extract_last_data(r'pasa1\data\paper_db\db3_idx_sim2pos.json')