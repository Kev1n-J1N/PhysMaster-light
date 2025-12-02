import json

import json

def convert_json_to_jsonl(input_json_file, output_jsonl_file):
    """
    将每行一个 JSON 的文件直接复制为 JSONL 文件。
    适用于原始文件已经是 JSONL 格式或每行一个 JSON 的情况。
    """
    try:
        print(f"🚀 开始处理文件: {input_json_file}")
        print(f"🔍 正在按行读取并写入 JSONL 格式到: {output_jsonl_file}")

        with open(input_json_file, 'r', encoding='utf-8') as infile, \
             open(output_jsonl_file, 'w', encoding='utf-8') as outfile:

            count = 0
            for line in infile:
                # 确保每一行是合法的 JSON（可选：验证）
                try:
                    obj = json.loads(line.strip())
                    json.dump(obj, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    count += 1

                    if count % 1000 == 0:
                        print(f"📝 已处理 {count} 行数据...")
                except json.JSONDecodeError as je:
                    print(f"⚠️ 跳过无效行: {je}")
                    continue

        print(f"🎉 成功将 {count} 条记录写入 JSONL 文件。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    # 输入 JSON 文件路径
    input_file = './result/dataset/sft_crawler/math_ids_sft_crawler_search_10000_qwen-72b.json'
    
    # 输出 JSONL 文件路径
    output_file = './result/dataset/sft_crawler/math_ids_sft_crawler_search_10000_qwen-72b.jsonl'
    
    # 调用转换函数
    print("🚀 开始转换 JSON 到 JSONL...")
    convert_json_to_jsonl(input_file, output_file)
    print("🏁 转换结束。")