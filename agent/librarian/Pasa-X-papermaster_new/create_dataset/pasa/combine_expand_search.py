import json
import random

def mix_jsonl(file1_path, file2_path, output_path):
    # 读取第一个 jsonl 文件
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = [json.loads(line) for line in f1 if line.strip()]
    
    # 读取第二个 jsonl 文件
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = [json.loads(line) for line in f2 if line.strip()]
    
    # 合并并打乱顺序
    combined = data1 + data2
    random.shuffle(combined)
    
    # 写入新的 jsonl 文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in combined:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

expand_jsonl = 'create_dataset/result/dataset/sft_crawler/math_ids_sft_crawler_expand_5000_qwen-72b.jsonl'
search_jsonl = 'create_dataset/result/dataset/sft_crawler/math_ids_sft_crawler_search_10000_qwen-72b.jsonl'
output_jsonl = 'create_dataset/result/dataset/sft_crawler/math_ids_sft_crawler_10000_qwen-72b.jsonl'
mix_jsonl(expand_jsonl, search_jsonl, output_jsonl)
