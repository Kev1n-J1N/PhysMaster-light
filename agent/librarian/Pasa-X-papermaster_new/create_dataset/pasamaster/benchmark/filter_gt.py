import csv
import json
import sys,os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_dir}/../../..")
from tools.llm_call import llm_call
import difflib
csv_path = f"{current_dir}/data/data_3.csv"
save_path = f"{current_dir}/data/data_3_new.csv"

def find_best_key(check, data_keys, cutoff=0.6):
    """
    在 data_keys 里找最接近 check 的 key。
    返回找到的 key；找不到就返回 None
    """
    # 1. 完全一致
    if check in data_keys:
        return check

    # 2. 不区分大小写一致
    lower_map = {k.lower(): k for k in data_keys}
    if check.lower() in lower_map:
        return lower_map[check.lower()]

    # 3. 子串匹配（任意方向）
    for k in data_keys:
        if check in k or k in check:
            return k

    # 4. 相似度匹配
    # get_close_matches 会返回最像的一些
    matches = difflib.get_close_matches(check, data_keys, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    return None



rows = []  

with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    original_fieldnames = reader.fieldnames[:]  # 保存原始列顺序
    for r in reader:
        rows.append(r)

prompt = "You are an academic expert. Your task is to determine whether the given paper fully satisfies the query requirements. The query is: {query}. The given paper has the title: {title}, and the abstract: {abstract}. You must examine the following key points: {check_list}. You should strictly check whether the paper satisfies each key point and provide supporting evidence from the original text. Return the result in JSON format as follows: {{\"write the key point directly\": {{\"evidence\": <original supporting evidence>, \"reason\":give your reason for the decision,limit in 100 words,\"satisfied\": <1 if the key point is satisfied, otherwise 0>}}, \"write the key point directly\": {{\"evidence\": <original supporting evidence>, \"reason\":give your reason for the decision,limit in 100 words,\"satisfied\": <1 if the key point is satisfied, otherwise 0>}}, ... , \"gt\": <1 if all key points are satisfied, otherwise 0>}}. You must evaluate each key point one by one. The evidence must be the original words of the article. If no direct evidence is found to support the viewpoint, the output for \"evidence\" is \"no relavant evidence,\" and the output for \"reason\" is whether the checkpoint is met based on the analysis of the entire abstract."

# 👇 这里不再是 []，而是固定长度的列表，这样索引就能对上
new_rows = [None] * len(rows)
extra_cols = []

extra_cols_lock = threading.Lock()
max_workers = 10

def process_one_row(row_idx, row, prompt):
    check_list = []
    for key, value in row.items():
        if key not in ["query", "title", "url", "abstract", "gt or not"]:
            check_list.append(key)

    max_retries = 3
    success = False
    local_new_cols = []

    for attempt in range(1, max_retries + 1):
        try:
            final_prompt = prompt.format(
                query=row.get("query", ""),
                title=row.get("title", ""),
                abstract=row.get("abstract", ""),
                check_list=check_list,
            )

            # 你这里用了 llm_call(final_prompt, 'glm')，我保留
            response = llm_call(final_prompt, "glm")

            json_begin = response.find("{")
            json_end = response.rfind("}")
            response_json = response[json_begin:json_end + 1]
            data = json.loads(response_json)

            for idx, check in enumerate(check_list):
                match_key = find_best_key(check, data.keys())
                if match_key is not None:
                    row[check] = data[match_key]["satisfied"]
                    reason_val = data[match_key]["reason"]
                    evidence_val = data[match_key]["evidence"]
                else:
                    print("\033[91m[ERROR]\033[0m no check key found for", check, file=sys.stderr)
                    row[check] = 0
                    reason_val = "no relevant evidence"
                    evidence_val = "no relevant evidence"

                reason_col = f"check_{idx + 1}_reason"
                evidence_col = f"check_{idx + 1}_evidence"

                row[reason_col] = reason_val
                row[evidence_col] = evidence_val

                if reason_col not in local_new_cols:
                    local_new_cols.append(reason_col)
                if evidence_col not in local_new_cols:
                    local_new_cols.append(evidence_col)

            row["gt or not"] = data.get("gt", 0)

            print(f"\033[92m[OK]\033[0m 第 {row_idx} 行处理成功")
            success = True
            break

        except Exception as e:
            # 注意 response 可能在异常前就没定义，做个兜底
            resp_str = locals().get("response", "<no response>")
            print(
                f"\033[91m[ERROR]\033[0m 第 {row_idx} 行，第 {attempt} 次处理失败：{e}, response: {resp_str}",
                file=sys.stderr,
            )

    if not success:
        for idx, check in enumerate(check_list):
            row[check] = 0
            reason_col = f"check_{idx + 1}_reason"
            evidence_col = f"check_{idx + 1}_evidence"

            row[reason_col] = "failed"
            row[evidence_col] = "failed"

            if reason_col not in local_new_cols:
                local_new_cols.append(reason_col)
            if evidence_col not in local_new_cols:
                local_new_cols.append(evidence_col)

        row["gt or not"] = 0
        print(
            f"\033[91m[ERROR]\033[0m 第 {row_idx} 行最终失败，已使用兜底结果。",
            file=sys.stderr,
        )

    # ⚠️ 把行号也返回出去
    return row_idx, row, local_new_cols


# 并发处理
futures = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for row_idx, row in enumerate(rows, start=0):  # 用0开头，这样能直接当索引
        futures.append(executor.submit(process_one_row, row_idx, row, prompt))

    for fut in as_completed(futures):
        row_idx, row_result, new_cols = fut.result()

        # 按原始顺序放回去
        new_rows[row_idx] = row_result

        # 合并新增列，保持顺序
        if new_cols:
            with extra_cols_lock:
                for col in new_cols:
                    if col not in extra_cols:
                        extra_cols.append(col)

# 写回 CSV
final_fieldnames = original_fieldnames + extra_cols
with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=final_fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print("\033[92m[OK]\033[0m CSV 已成功写入到:", save_path)