import json

def clean_arxiv_result_file(input_path, output_path):
    """
    读取一个包含 {arxiv_id: 内容或 None} 的 JSON 文件，删除值为 None 的项，并写入新文件。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("type of data is not dict")

    cleaned_data = {paper_id: content for paper_id, content in data.items() if content is not None}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
def filter_arxiv_ids_by_year(input_path, output_path, year_threshold=20):
    """
    读取包含 arXiv ID 列表的 JSON 文件，
    只保留前两位数字（年份）大于等于 year_threshold 的 ID，
    并写入新文件。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        arxiv_list = json.load(f)

    def extract_year(arxiv_id):
        # arXiv ID 格式如 "1303.3704" 或 "2105.1234"，取前两位数字作为年份
        try:
            # 有些ID可能长度不一，取开头两个数字
            year_str = arxiv_id[:2]
            return int(year_str)
        except Exception:
            return -1  # 非法ID返回-1，自动过滤

    filtered_ids = [arxiv_id for arxiv_id in arxiv_list if extract_year(arxiv_id) >= year_threshold]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_ids, f, indent=2, ensure_ascii=False)

# 示例调用
# filter_arxiv_ids_by_year("arxiv_ids.json", "arxiv_ids_filtered.json")

# 示例用法
# input_path="../create_dataset/result/train_data/id2paper_with_metadata.json"
# output_path="../create_dataset/result/train_data/id2paper_with_metadata.json"
# clean_arxiv_result_file(input_path, output_path)

input_path="../create_dataset/result/train_data/math_ids.json"
output_path="../create_dataset/result/train_data/math_ids.json"
# input_path="../create_dataset/result/train_data/physics_ids.json"
# output_path="../create_dataset/result/train_data/physics_ids.json"
filter_arxiv_ids_by_year(input_path, output_path)