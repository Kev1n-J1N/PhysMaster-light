import csv
import json
import random
import zipfile
import os

def get_ai_auto_id(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            item = {
                "arxiv_id": data['answer_arxiv_id'][0].split('v')[0],
                "title": data['answer'][0]
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_csee_id(input_path, output_path):
    rows = []    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            title = row.get('Title')
            arxiv_id = row.get('arXiv_id')
            rows.append({'arxiv_id':arxiv_id, 'title':title})
    random.shuffle(rows)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for data in rows:
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
def sample_id(input_path, output_path):
    lines = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            lines.append(line.rstrip('\n'))
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in lines[:100]:
            f_out.write(line + '\n')

def sample_title_from_csv(input_path, output_path):
    titles = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            title = row.get('Title')
            if title:
                titles.append(title)
    random.shuffle(titles)
    with open(output_path, 'a', encoding='utf-8') as f_out:
        for title in titles[:200]:
            item = {"title": title}
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

def remove_arxiv_id(input_path, output_path):
    """读取 input_path 的 jsonl 文件，去除每行的 arxiv_id 字段，只保留 title 字段，写入 output_path"""
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            if 'title' in data:
                f_out.write(json.dumps({'title': data['title']}, ensure_ascii=False) + '\n')

def zip_file():
    file_paths = [
        r'zyz\data\paper_db\db3.json',
        r'zyz\data\paper_db\db3_idx_sim2pos.json',
        r'zyz\utils\apikeys.json'
    ]
    zip_path = r'zyz/utils/zip_files.zip'  # 输出zip文件路径
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            if os.path.exists(file):
                zipf.write(file, arcname=os.path.basename(file))
            else:
                print(f"文件不存在: {file}")
    print(f"已打包为: {zip_path}")

if __name__ == '__main__':
    # input_path = r'pasa1\data\\top_paper\CS_all_nover.csv'
    # input_path = r'data\AutoScholarQuery\train.jsonl'
    # output_path = r'optimize\evaluate\id2sec\data\all\ai.jsonl'
    # get_ai_auto_id(input_path, output_path)
    # input_path = r'pasa1\data\top_paper\EE_all_nover.csv'
    # output_path = r'optimize\evaluate\ti2id\data\test2\ee.jsonl'
    # sample_title_from_csv(input_path, output_path)
    zip_file()