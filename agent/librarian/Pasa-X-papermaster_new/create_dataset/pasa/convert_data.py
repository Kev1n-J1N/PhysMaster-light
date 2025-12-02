import pandas as pd
import json
import re

# 文件路径
csv_file = '/home/ma-user/modelarts/work/duyuwen/pasa/create_dataset/math.csv'
jsonl_file = '/home/ma-user/modelarts/work/duyuwen/pasa/create_dataset/result/dataset/real/math.jsonl'

# 尝试不同编码读取 CSV
encodings = ['utf-8-sig', 'gbk', 'utf-8']
df = None

for enc in encodings:
    try:
        df = pd.read_csv(csv_file, encoding=enc)
        print(f"成功读取，编码为：{enc}")
        break
    except Exception as e:
        print(f"尝试编码 {enc} 失败: {e}")

if df is not None:
    # 清洗文本内容：去除中文引号、空格
    def clean_text(text):
        if pd.isna(text):
            return ""
        return str(text).replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").strip()

    # 字符串转数组：把形如 "\"a\", \"b\"" 的字符串变成 ["a", "b"]
    def parse_string_list(text):
        text = clean_text(text)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = text.replace('\\"', '"')  # 去掉转义
        parts = re.split(r'"\s*,\s*"|,\s*"', text.strip('"'))
        return [p.strip().strip('"') for p in parts if p.strip()]

    df['question'] = df['question'].map(clean_text)
    df['answer'] = df['answer'].map(parse_string_list)
    df['answer_arxiv_id'] = df['answer_arxiv_id'].map(parse_string_list)

    # 写入 JSONL 文件
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump({
                "question": row['question'],
                "answer": row['answer'],
                "answer_arxiv_id": row['answer_arxiv_id']
            }, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ 已成功写入 JSONL 文件：{jsonl_file}")
else:
    print("❌ 读取 CSV 文件失败")
