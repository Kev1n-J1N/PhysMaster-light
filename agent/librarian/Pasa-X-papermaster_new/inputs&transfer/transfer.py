import json
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--published_time", type=str, default=None)
    args = parser.parse_args()

    # 读 clarifier 输出
    with open(args.input_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 检索截止时间
    if args.published_time is not None:
        pub_time = args.published_time
    else:
        pub_time = datetime.today().strftime("%Y%m%d")

    rk_list = obj.get("related_knowledge") or []

    # 输出文件所在目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for rk in rk_list:
            record = {
                "question": str(rk),
                "source_meta": {
                    "published_time": pub_time
                }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✔ transfer 转换完成，共生成 {len(rk_list)} 条检索 query")
    print(f"→ 输出文件: {args.output_file}")


if __name__ == "__main__":
    main()
