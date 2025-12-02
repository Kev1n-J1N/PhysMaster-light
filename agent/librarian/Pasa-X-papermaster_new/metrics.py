# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os,math
import json
import glob
import argparse
from tqdm import tqdm

# 直接在这里切换是否打印中间结果：True/False
VERBOSE = False
def dcg_at_k(rels, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(ranked_list, relevant_set, k):
    # 实际 DCG：看 top-k 命中
    hits = [1 if x in relevant_set else 0 for x in ranked_list[:k]]
    dcg = dcg_at_k(hits, k)

    # 理想 DCG：只看 ground truth，有多少相关文档就放在最前面
    m = min(k, len(relevant_set))       # 真实相关文档数 ∧ k
    ideal = [1]*m + [0]*max(0, k - m)
    idcg = dcg_at_k(ideal, k)

    return 0.0 if idcg == 0 else dcg / idcg

def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def cal_micro(pred_set, label_set):
    if len(label_set) == 0:
        return 0, 0, 0, set(), set(), set()

    if len(pred_set) == 0:
        return 0, 0, len(label_set), set(), set(), label_set

    tp_set = pred_set & label_set  # 真正例（预测正确的）
    fp_set = pred_set - label_set  # 假正例（预测错误的）
    fn_set = label_set - pred_set  # 假负例（遗漏的真实标签）

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    assert tp + fn == len(label_set)
    assert len(label_set) != 0
    return tp, fp, fn, tp_set, fp_set, fn_set

parser = argparse.ArgumentParser(description='评测文献搜索系统的性能指标')
parser.add_argument('--input_folder', type=str, default="/data/duyuwen/Pasa-X/result/result_RealScholarQuery_20251107",
                   help='输入的JSON结果文件夹路径 (例如: .//home/ma-user/modelarts/work/TianJin/papermaster/PASAX-jt/Pasa-X/result/result_RealScholarQuery_20250902)')
parser.add_argument('--output_file', type=str, default="/data/duyuwen/Pasa-X/result/result_RealScholarQuery_20250917_emb_combine/eval_result.json",
                   help='评测结果保存路径 (例如: ./home/ma-user/modelarts/work/TianJin/papermaster/PASAX-jt/Pasa-X/result/result_RealScholarQuery_20250902/eval_result.json)，默认为输入文件夹同级目录下的eval_result.json')
parser.add_argument('--output_folder_ensemble', type=str, default=None,
                   help='集成评测的第二个结果文件夹路径（可选）')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--verbose', action='store_true', help='打印每个查询的中间结果（集合内容、tp/fp/fn、@K统计等）')
args = parser.parse_args()
args.output_file = args.input_folder + "/eval_result.json"
# 确保输出目录存在
output_dir = os.path.dirname(args.output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

pred_files = glob.glob(args.input_folder + "/*.json")

crawler_recalls, precisions, recalls, recalls_100, recalls_50, recalls_20, actions, recalls_10,recalls_5,scores = [], [], [], [], [], [], [], [],[], []
ndcg_20, ndcg_10, ndcg_5 = [], [], []

precisions_5, precisions_10, precisions_20 = [], [], []
crawler_recall_cnt, recall_cnt, answer_cnt = [], [], []
total_papers_count = []  # 新增：统计每个查询搜索到的总论文数量
execution_times = []
all_papers_hit_counts = []  # 新增：每题在all_papers中命中的标答数量
all_papers_total_counts = []  # 新增：仅统计存在 all_papers 时的论文总数

detailed_query_results = []  # 新增：存储每个查询的详细tp/fp/fn结果
for pred_file in tqdm(pred_files):    
    if "eval" in pred_file:
        continue
    id = int(pred_file.split("/")[-1].split(".")[0])
    if id >= args.test_num:
        continue
    with open(pred_file) as f:
        paper_root = json.load(f)
    
    # 统计召回论文数量，后续计算平均值，以做更详细比较
    # crawler_recall_cnt.append(len(paper_root['extra']['crawler_recall_papers']))
    # recall_cnt.append(len(paper_root['extra']['recall_papers']))
    # answer_cnt.append(len(paper_root['extra']['answer']))
    
    # 统计搜索到的总论文数量
    # if "all_papers_title" in paper_root["extra"]:
    #     total_papers_count.append(len(paper_root["extra"]["all_papers_title"]))
    # else:
    #     total_papers_count.append(0)
    # 统计执行时间
    if "execution_time" in paper_root:
        execution_time = paper_root["execution_time"]
        paper_agent_run_duration = execution_time.get("paper_agent_run_duration", 0)
        total_batch_time = execution_time.get("total_batch_time", 0)
        execution_times.append({
            "paper_agent_run_duration": paper_agent_run_duration,
            "total_batch_time": total_batch_time,
            "timestamp": execution_time.get("timestamp", "")
        })
    else:
        # 如果没有执行时间信息，添加默认值
        execution_times.append({
            "paper_agent_run_duration": 0,
            "total_batch_time": 0,
            "timestamp": ""
        })
    
    crawled_papers, crawled_paper_set, selected_paper_set = [], set(), set()
    answer_list_original = paper_root["extra"].get("answer", [])
    answer_paper_set = set([keep_letters(paper) for paper in answer_list_original])
    normalized_answer_to_original = {keep_letters(ans): ans for ans in answer_list_original}
    
    # 兼容新框架：从extra字段中获取论文数据
    all_papers_hit_titles = set()  # 新增：记录本题在all_papers命中的原始答案标题
    if "all_papers" in paper_root["extra"] and paper_root["extra"]["all_papers"]:
        # 新框架：使用all_papers（所有搜索到的论文）
        total_count_this_query = len(paper_root["extra"]["all_papers"]) 
        total_papers_count.append(total_count_this_query)
        all_papers_total_counts.append(total_count_this_query)
        for paper_data in paper_root["extra"]["all_papers"]:
            title = keep_letters(paper_data["title"])
            score = paper_data.get("select_score", 0.0)
            
            if title not in crawled_paper_set:
                crawled_paper_set.add(title)
                crawled_papers.append([title, score])
            
            if score > 3.8:
                selected_paper_set.add(title)
            # 统计all_papers命中标答
            if title in answer_paper_set:
                all_papers_hit_titles.add(normalized_answer_to_original.get(title, paper_data.get("title", "")))
    
    elif "sorted_papers" in paper_root["extra"] and paper_root["extra"]["sorted_papers"]:
        # 备用：使用sorted_papers（仅高分论文）
        for paper_data in paper_root["extra"]["sorted_papers"]:
            title = keep_letters(paper_data["title"])
            score = paper_data.get("select_score", 0.0)
            
            if title not in crawled_paper_set:
                crawled_paper_set.add(title)
                crawled_papers.append([title, score])
            
            if score > 0.5:
                selected_paper_set.add(title)
    
    elif "child" in paper_root and paper_root["child"]:
        # 原版框架：使用child树结构
        queue, action, score = [paper_root], 0, []
        num = 0
        while len(queue) > 0:
            node, queue = queue[0], queue[1:]
            action += len(node["child"])
            total_score = 0
            for _, v in node["child"].items():
                num += len(v)
                total_score -= 0.1
                for i in v:
                    queue.append(i)
                    if i["select_score"] > 0.5:
                        selected_paper_set.add(keep_letters(i["title"]))
                        total_score += 1
                    if keep_letters(i["title"]) not in crawled_paper_set:
                        crawled_paper_set.add(keep_letters(i["title"]))
                        crawled_papers.append([keep_letters(i["title"]), i["select_score"]])
            score.append(total_score)
        actions.append(action)
        scores.append(sum(score) / len(score) if len(score) > 0 else 0)
        total_papers_count.append(num)
    else:
        print(f"⚠️ 警告: 文件 {pred_file} 中没有找到论文数据")
    
    # 为新框架设置默认值
    if "all_papers" in paper_root["extra"]:
        actions.append(len(paper_root["extra"]["all_papers"]))
        scores.append(1.0 if paper_root["extra"]["all_papers"] else 0.0)
    elif "sorted_papers" in paper_root["extra"]:
        actions.append(len(paper_root["extra"]["sorted_papers"]))
        scores.append(1.0 if paper_root["extra"]["sorted_papers"] else 0.0)
    
    # ensemble
    if args.output_folder_ensemble is not None:
        paper_root = json.load(open(os.path.join(args.output_folder_ensemble, pred_file.split("/")[-1])))
        queue = [paper_root]
        while len(queue) > 0:
            node, queue = queue[0], queue[1:]
            for _, v in node["child"].items():
                for i in v:
                    queue.append(i)
                    if i["select_score"] > 0.5:
                        selected_paper_set.add(keep_letters(i["title"]))
                    if keep_letters(i["title"]) not in crawled_paper_set:
                        crawled_paper_set.add(keep_letters(i["title"]))
                        crawled_papers.append([keep_letters(i["title"]), i["select_score"]])
    crawled_papers.sort(key=lambda x: x[1], reverse=True)
    crawled_20, crawled_50, crawled_100 = set(), set(), set()
    crawled_5, crawled_10 = set(), set()
    ranked_titles = [item[0] for item in crawled_papers]
    # 新增：计算不同K值的NDCG
    k_values = [20, 10, 5]
    for k in k_values:
        ndcg_score = ndcg_at_k(ranked_titles, answer_paper_set, k)
        if k == 20:
            ndcg_20.append(ndcg_score)
        elif k == 10:
            ndcg_10.append(ndcg_score)
        elif k == 5:
            ndcg_5.append(ndcg_score)
            
    for i in range(100):
        if i >= len(crawled_papers):
            break
        if i < 5:
            crawled_5.add(crawled_papers[i][0])
        if i < 10:
            crawled_10.add(crawled_papers[i][0])
        if i < 20:
            crawled_20.add(crawled_papers[i][0])
        if i < 50:
            crawled_50.add(crawled_papers[i][0])
        crawled_100.add(crawled_papers[i][0])
        
    crawled_res = cal_micro(crawled_paper_set, answer_paper_set)
    selected_res = cal_micro(selected_paper_set, answer_paper_set)
    crawled_20_res = cal_micro(crawled_20, answer_paper_set)
    crawled_50_res = cal_micro(crawled_50, answer_paper_set)
    crawled_100_res = cal_micro(crawled_100, answer_paper_set)
    crawled_5_res = cal_micro(crawled_5, answer_paper_set)
    crawled_10_res = cal_micro(crawled_10, answer_paper_set)

    if VERBOSE or args.verbose:
        print("\n========== 中间结果: 文件 {} ==========".format(pred_file))
        # 基础集合信息
        print("answer(标准答案) 数量: {}".format(len(answer_paper_set)))
        print("answer 示例(最多前10条): {}".format(list(answer_paper_set)[:10]))

        print("\n去重后的爬取论文(crawled_paper_set) 数量: {}".format(len(crawled_paper_set)))
        print("选中论文(selected_paper_set) 数量(阈值>0.8 或 >0.5 取决于来源): {}".format(len(selected_paper_set)))

        # 排序后的前若干条
        preview_top = min(10, len(crawled_papers))
        print("\n按分数降序的前{}条 crawled_papers: (title, score)".format(preview_top))
        print(crawled_papers[:preview_top])

        # @K 集合
        print("\nTop@5 数量: {}, 样例: {}".format(len(crawled_5), list(crawled_5)[:10]))
        print("Top@10 数量: {}, 样例: {}".format(len(crawled_10), list(crawled_10)[:10]))
        print("Top@20 数量: {}, 样例: {}".format(len(crawled_20), list(crawled_20)[:10]))
        print("Top@50 数量: {}, 样例: {}".format(len(crawled_50), list(crawled_50)[:10]))
        print("Top@100 数量: {}, 样例: {}".format(len(crawled_100), list(crawled_100)[:10]))

        # 打印论文集合详细匹配信息
        print("\n========== 论文匹配详情 ==========")
        
        def print_paper_details(name, res):
            tp, fp, fn, tp_set, fp_set, fn_set = res
            print(f"\n【{name}】:")
            print(f"  tp={tp}, fp={fp}, fn={fn}")
            if tp > 0:
                print(f"  ✅ 正确识别的论文 (TP, {tp}篇):")
                for i, paper in enumerate(sorted(tp_set)[:10]):  # 最多显示10篇
                    print(f"    {i+1}. {paper}")
                if len(tp_set) > 10:
                    print(f"    ... 还有{len(tp_set)-10}篇")
            
            if fp > 0:
                print(f"  ❌ 错误识别的论文 (FP, {fp}篇):")
                for i, paper in enumerate(sorted(fp_set)[:10]):  # 最多显示10篇
                    print(f"    {i+1}. {paper}")
                if len(fp_set) > 10:
                    print(f"    ... 还有{len(fp_set)-10}篇")
            
            if fn > 0:
                print(f"  ⚠️  遗漏的标准答案 (FN, {fn}篇):")
                for i, paper in enumerate(sorted(fn_set)[:10]):  # 最多显示10篇
                    print(f"    {i+1}. {paper}")
                if len(fn_set) > 10:
                    print(f"    ... 还有{len(fn_set)-10}篇")

        print_paper_details("crawled_paper_set (所有爬取论文)", crawled_res)
        print_paper_details("selected_paper_set (筛选后论文)", selected_res)
        print_paper_details("Top@5", crawled_5_res)
        print_paper_details("Top@10", crawled_10_res)
        print_paper_details("Top@20", crawled_20_res)
        print_paper_details("Top@50", crawled_50_res)
        print_paper_details("Top@100", crawled_100_res)

    # 保存当前查询的详细结果
    def convert_set_to_list(res):
        """将集合转换为列表以便JSON序列化"""
        tp, fp, fn, tp_set, fp_set, fn_set = res
        return {
            "tp": tp,
            "fp": fp, 
            "fn": fn,
            "tp_papers": sorted(list(tp_set)),
            "fp_papers": sorted(list(fp_set)),
            "fn_papers": sorted(list(fn_set))
        }
    
    # 记录并汇总 all_papers 命中情况
    all_papers_hit_count = len(all_papers_hit_titles)
    all_papers_hit_counts.append(all_papers_hit_count)

    query_detail = {
        "query_id": id,
        "query_file": pred_file.split("/")[-1],
        "answer_papers": sorted(list(answer_paper_set)),
        "answer_count": len(answer_paper_set),
        "total_crawled_papers": len(crawled_paper_set),
        "total_selected_papers": len(selected_paper_set),
        "all_papers_hits": {
            "hit_count": all_papers_hit_count,
            "hit_titles": sorted(list(all_papers_hit_titles))
        },
        "results": {
            "crawled_paper_set": convert_set_to_list(crawled_res),
            "selected_paper_set": convert_set_to_list(selected_res),
            "top_5": convert_set_to_list(crawled_5_res),
            "top_10": convert_set_to_list(crawled_10_res),
            "top_20": convert_set_to_list(crawled_20_res),
            "top_50": convert_set_to_list(crawled_50_res),
            "top_100": convert_set_to_list(crawled_100_res)
        },
        "performance_metrics": {
            "crawler_recall": crawled_res[0] / (crawled_res[0] + crawled_res[2] if (crawled_res[0] + crawled_res[2]) > 0 else 1e-9),
            "selected_precision": selected_res[0] / (selected_res[0] + selected_res[1] if (selected_res[0] + selected_res[1]) > 0 else 1e-9),
            "selected_recall": selected_res[0] / (selected_res[0] + selected_res[2] if (selected_res[0] + selected_res[2]) > 0 else 1e-9),
            "recall_at_5": crawled_5_res[0] / (crawled_5_res[0] + crawled_5_res[2] if (crawled_5_res[0] + crawled_5_res[2]) > 0 else 1e-9),
            "recall_at_10": crawled_10_res[0] / (crawled_10_res[0] + crawled_10_res[2] if (crawled_10_res[0] + crawled_10_res[2]) > 0 else 1e-9),
            "recall_at_20": crawled_20_res[0] / (crawled_20_res[0] + crawled_20_res[2] if (crawled_20_res[0] + crawled_20_res[2]) > 0 else 1e-9),
            "recall_at_50": crawled_50_res[0] / (crawled_50_res[0] + crawled_50_res[2] if (crawled_50_res[0] + crawled_50_res[2]) > 0 else 1e-9),
            "recall_at_100": crawled_100_res[0] / (crawled_100_res[0] + crawled_100_res[2] if (crawled_100_res[0] + crawled_100_res[2]) > 0 else 1e-9),
            "precision_at_5": crawled_5_res[0] / 5,
            "precision_at_10": crawled_10_res[0] / 10,
            "precision_at_20": crawled_20_res[0] / 20
        }
    }
    detailed_query_results.append(query_detail)

    crawler_recalls.append(crawled_res[0] / (crawled_res[0] + crawled_res[2] if (crawled_res[0] + crawled_res[2]) > 0 else 1e-9))
    precisions.append(selected_res[0] / (selected_res[0] + selected_res[1] if (selected_res[0] + selected_res[1]) > 0 else 1e-9))
    recalls.append(selected_res[0] / (selected_res[0] + selected_res[2] if (selected_res[0] + selected_res[2]) > 0 else 1e-9))
    recalls_100.append(crawled_100_res[0] / (crawled_100_res[0] + crawled_100_res[2] if (crawled_100_res[0] + crawled_100_res[2]) > 0 else 1e-9))
    recalls_50.append(crawled_50_res[0] / (crawled_50_res[0] + crawled_50_res[2] if (crawled_50_res[0] + crawled_50_res[2]) > 0 else 1e-9))
    recalls_20.append(crawled_20_res[0] / (crawled_20_res[0] + crawled_20_res[2] if (crawled_20_res[0] + crawled_20_res[2]) > 0 else 1e-9))
    recalls_10.append(crawled_10_res[0] / (crawled_10_res[0] + crawled_10_res[2] if (crawled_10_res[0] + crawled_10_res[2]) > 0 else 1e-9))
    recalls_5.append(crawled_5_res[0] / (crawled_5_res[0] + crawled_5_res[2] if (crawled_5_res[0] + crawled_5_res[2]) > 0 else 1e-9))
    # 计算 Precision@K
    precisions_5.append(crawled_5_res[0] / 5)  # 分母固定为5
    precisions_10.append(crawled_10_res[0] / 10)  # 分母固定为10
    precisions_20.append(crawled_20_res[0] / 20)  # 分母固定为20

    if VERBOSE or args.verbose:
        # 单条查询的比例指标
        def safe_div(a, b):
            return a / (b if b > 0 else 1e-9)

        print("\n比例指标(单条查询):")
        print("crawler_recall (爬取集合的召回率) = {:.4f}".format(safe_div(crawled_res[0], crawled_res[0] + crawled_res[2])))
        print("selected_precision (选中集合的精确度) = {:.4f}".format(safe_div(selected_res[0], selected_res[0] + selected_res[1])))
        print("selected_recall (选中集合的召回率) = {:.4f}".format(safe_div(selected_res[0], selected_res[0] + selected_res[2])))
        print("Recall@5/10/20/50/100 = {:.4f} / {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
            safe_div(crawled_5_res[0], crawled_5_res[0] + crawled_5_res[2]),
            safe_div(crawled_10_res[0], crawled_10_res[0] + crawled_10_res[2]),
            safe_div(crawled_20_res[0], crawled_20_res[0] + crawled_20_res[2]),
            safe_div(crawled_50_res[0], crawled_50_res[0] + crawled_50_res[2]),
            safe_div(crawled_100_res[0], crawled_100_res[0] + crawled_100_res[2]),
        ))
        print("Precision@5/10/20 = {:.4f} / {:.4f} / {:.4f}".format(
            precisions_5[-1], precisions_10[-1], precisions_20[-1]
        ))
    
    # 每轮计算的tp/fp/fn指标
    # print('crawler mtr', crawled_res)
    # print('select mtr', selected_res)
    
# 计算平均指标
avg_crawler_recall = round(sum(crawler_recalls) / len(crawler_recalls), 4)
avg_precision = round(sum(precisions) / len(precisions), 4)
avg_recall = round(sum(recalls) / len(recalls), 4)
avg_recall_100 = round(sum(recalls_100) / len(recalls_100), 4)
avg_recall_50 = round(sum(recalls_50) / len(recalls_50), 4)
avg_recall_20 = round(sum(recalls_20) / len(recalls_20), 4)
avg_recall_10 = round(sum(recalls_10) / len(recalls_10), 4)
avg_recall_5 = round(sum(recalls_5) / len(recalls_5), 4)
avg_actions = round(sum(actions) / len(actions), 4)
avg_scores = round(sum(scores) / len(scores), 4)
avg_precision_5 = round(sum(precisions_5) / len(precisions_5), 4)
avg_precision_10 = round(sum(precisions_10) / len(precisions_10), 4)
avg_precision_20 = round(sum(precisions_20) / len(precisions_20), 4)
avg_total_papers = round(sum(total_papers_count) / len(total_papers_count), 2)  # 新增：平均搜索论文数量
avg_paper_agent_run_duration = round(sum([t["paper_agent_run_duration"] for t in execution_times]) / len(execution_times), 2)
avg_total_batch_time = round(sum([t["total_batch_time"] for t in execution_times]) / len(execution_times), 2)
avg_all_papers_hit = round(sum(all_papers_hit_counts) / len(all_papers_hit_counts), 4) if all_papers_hit_counts else 0.0
avg_all_papers_total = round(sum(all_papers_total_counts) / len(all_papers_total_counts), 2) if all_papers_total_counts else 0.0

# 打印原有格式的结果
print("{} & {} & {} & {} & {} & {} & {} & {} & {}".format(
    avg_crawler_recall,
    avg_precision,
    avg_recall,
    avg_recall_100,
    avg_recall_50,
    avg_recall_20,
    round(sum(ndcg_5) / len(ndcg_5), 4),  # NDCG@100
    round(sum(ndcg_10) / len(ndcg_10), 4),   # NDCG@50
    round(sum(ndcg_20) / len(ndcg_20), 4)    # NDCG@20
))
print("{} & {} & {} & {}  & {}".format(
    avg_crawler_recall,
    avg_actions,
    avg_scores,
    avg_precision,
    avg_recall,
))
print("Precision@5/10/20: {} / {} / {}".format(
    avg_precision_5,
    avg_precision_10,
    avg_precision_20,
))
print("平均搜索论文数量: {}".format(avg_total_papers))
print("平均执行时间: paper_agent.run() = {:.2f}秒, 总批次时间 = {:.2f}秒".format(
    avg_paper_agent_run_duration,
    avg_total_batch_time
))
print("All-Papers 平均命中标答数量: {}".format(avg_all_papers_hit))
print("All-Papers 平均总数量: {}".format(avg_all_papers_total))

# 保存详细评测结果到JSON文件
eval_result = {
    "evaluation_summary": {
        "total_queries": len(pred_files),
        "input_folder": args.input_folder,
        "output_file": args.output_file,
        "ensemble_folder": args.output_folder_ensemble
    },
    "average_metrics": {
        "crawler_recall": avg_crawler_recall,
        "precision": avg_precision,
        "recall": avg_recall,
        "recall_at_100": avg_recall_100,
        "recall_at_50": avg_recall_50,
        "recall_at_20": avg_recall_20,
        "recall_at_10": avg_recall_10,
        "recall_at_5": avg_recall_5,
        "precision_at_5": avg_precision_5,
        "precision_at_10": avg_precision_10,
        "precision_at_20": avg_precision_20,
        "average_actions": avg_actions,
        "average_scores": avg_scores,
        "average_total_papers": avg_total_papers,
        "average_all_papers_hit": avg_all_papers_hit,
        "average_all_papers_total": avg_all_papers_total,
        "average_execution_time": {
            "paper_agent_run_duration": avg_paper_agent_run_duration,
            "total_batch_time": avg_total_batch_time
        }
    },
    "detailed_metrics": {
        "crawler_recalls": crawler_recalls,
        "precisions": precisions,
        "recalls": recalls,
        "recalls_100": recalls_100,
        "recalls_50": recalls_50,
        "recalls_20": recalls_20,
        "recalls_10": recalls_10,
        "recalls_5": recalls_5,
        "precisions_5": precisions_5,
        "precisions_10": precisions_10,
        "precisions_20": precisions_20,
        "actions": actions,
        "scores": scores,
        "total_papers_count": total_papers_count,
        "execution_times": execution_times,
        "all_papers_hit_counts": all_papers_hit_counts,
        "all_papers_total_counts": all_papers_total_counts
    },
    "performance_summary": {
        "best_recall": max(recalls) if recalls else 0,
        "worst_recall": min(recalls) if recalls else 0,
        "best_precision": max(precisions) if precisions else 0,
        "worst_precision": min(precisions) if precisions else 0,
        "best_precision_at_5": max(precisions_5) if precisions_5 else 0,
        "best_precision_at_10": max(precisions_10) if precisions_10 else 0,
        "best_precision_at_20": max(precisions_20) if precisions_20 else 0,
        "max_papers_searched": max(total_papers_count) if total_papers_count else 0,
        "min_papers_searched": min(total_papers_count) if total_papers_count else 0,
        "queries_with_high_recall": sum(1 for r in recalls if r > 0.8),
        "queries_with_high_precision": sum(1 for p in precisions if p > 0.8),
        "execution_time_stats": {
            "fastest_paper_agent_run": min([t["paper_agent_run_duration"] for t in execution_times]) if execution_times else 0,
            "slowest_paper_agent_run": max([t["paper_agent_run_duration"] for t in execution_times]) if execution_times else 0,
            "fastest_total_batch": min([t["total_batch_time"] for t in execution_times]) if execution_times else 0,
            "slowest_total_batch": max([t["total_batch_time"] for t in execution_times]) if execution_times else 0,
            "queries_under_30s": sum(1 for t in execution_times if t["paper_agent_run_duration"] < 30),
            "queries_under_60s": sum(1 for t in execution_times if t["paper_agent_run_duration"] < 60),
            "queries_over_120s": sum(1 for t in execution_times if t["paper_agent_run_duration"] > 120)
        }
    },
    "detailed_query_results": detailed_query_results
}

# 保存评测结果
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(eval_result, f, indent=2, ensure_ascii=False)

print(f"\n📊 评测结果已保存到: {args.output_file}")
print(f"📋 总查询数: {len(total_papers_count)}")
print(f"📈 平均召回率: {avg_recall}")
print(f"📈 平均精确度: {avg_precision}")

# 输出召回论文数量的平均值
# def avg(a):
#     return sum(a)/len(a) if len(a)>0 else 0
# print('avg crawler recall', avg(crawler_recall_cnt))
# print('avg recall', avg(recall_cnt))
# print('avg answer', avg(answer_cnt))