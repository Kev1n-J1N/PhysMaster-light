from search_utils import search_paper_by_title  
import time
import csv
import re

test_data = [
    ("Attention Is All You Need", "1706.03762"),
    ("Deep Residual Learning for Image Recognition", "1512.03385"),
    ("Auto-Encoding Variational Bayes", "1312.6114"),
    ("Generative Adversarial Nets", "1406.2661"),
    ("Optimal Speed and Accuracy of Object Detection", "2004.10934"),
    ("Denoising Diffusion Probabilistic Models", "2006.11239"),
    ("Segment Anything", "2304.02643"),
    ("Language Models are Few-Shot Learners", "2005.14165"),
    ("A Survey on Image Captioning", "2003.10198"),
    ("Diffusion Models Beat GANs on Image Synthesis", "2105.05233"),
    ("Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805"),
    ("ImageNet Classification with Deep Convolutional Neural Networks", "1409.1556"),
    ("A distilled version of BERT smaller, faster, cheaper and lighter", "1910.01108"),
    ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "2010.11929"),
    ("Masked Language Modeling for Unified Pretraining", "2008.39053"),
    ("Connecting Text and Images", "2103.00020"),
    ("The Semantic Scholar Open Research Corpus", "1911.02782"),
    ("The COVID-19 Open Research Dataset", "2004.10706"),
    ("A Large-Scale Dataset of AI-Generated Summaries for Scientific Papers", "2502.20582"),
    ("NLLG Quarterly arXiv Report 09/23 What are the most influential current AI Papers", "2312.05688"),
    ("Neural Machine Translation by Jointly Learning to Align and Translate", "1409.0473"),
    ("A Method for Stochastic Optimization", "1412.6980"),
    ("Towards Real-Time Object Detection with Region Proposal Networks", "1506.01497"),
    ("Convolutional Networks for Biomedical Image Segmentation", "1505.04597"),
    ("Mask R-CNN", "1703.06870"),
    ("Attentive Language Models Beyond a Fixed-Length Context", "1901.02860"),
    ("Analyzing and Improving the Image Quality of StyleGAN", "1912.04958"),
    ("Densely Connected Convolutional Networks", "1608.06993"),
    ("Generalized Autoregressive Pretraining for Language Understanding", "1906.08237"),
    ("Neural Ordinary Differential Equations", "1806.07366"),
    ("Large Scale GAN Training for High Fidelity Natural Image Synthesis", "1809.11096"),
    ("Hierarchical Vision Transformer using Shifted Windows", "2103.14030"),
    ("A Robustly Optimized BERT Pretraining Approach", "1907.11692"),
    ("Graph Attention Networks", "1710.10903"),
    ("Learning Transferable Visual Models From Natural Language Supervision", "2103.00020"),
    ("DeepMind Control Suite", "1801.00690"),
    ("Beyond Empirical Risk Minimization", "1710.09412"),
    ("The Efficient Transformer", "2001.04451"),
    ("Inverted Residuals and Linear Bottlenecks", "1801.04381"),
    ("Rethinking Model Scaling for Convolutional Neural Networks", "1905.11946"),
    ("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "1910.10683"),
    ("Search with Reinforcement Learning", "1611.01578"),
    ("Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "1703.03400"),
    ("Deep Contextualized Word Representations", "1802.05365"),
    ("Smaller, Faster, Cheaper and Lighter", "1910.01108"),
    ("AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size", "1602.07360"),
    ("An Incremental Improvement", "1804.02767"),
    ("Focal Loss for Dense Object Detection", "1708.02002"),
    ("Using AI for scientific discovery", "2101.11465"),
    ("Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", "1910.13461"),
    ("Language Models are Unsupervised Multitask Learners", "1906.08237"),
    ("Vision Transformer", "2010.11929"),
    ("Convolutional Networks for Biomedical Image Segmentation", "1505.04597"),
    ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "2010.11929"),
    ("Masked Language Modeling for Unified Pretraining", "2008.39053"),
    ("Connecting Text and Images", "2103.00020"),
    ("The Semantic Scholar Open Research Corpus", "1911.02782"),
    ("The COVID-19 Open Research Dataset", "2004.10706"),
    ("Rethinking Model Scaling for Convolutional Neural Networks", "1905.11946"),
    ("The Efficient Transformer", "2001.04451"),
    ("Hierarchical Vision Transformer using Shifted Windows", "2103.14030"),
    ("Generalized Autoregressive Pretraining for Language Understanding", "1906.08237"),
    ("Search with Reinforcement Learning", "1611.01578"),
    ("Deep Contextualized Word Representations", "1802.05365"),
    ("Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "1703.03400"),
    ("AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size", "1602.07360"),
    ("An Incremental Improvement", "1804.02767"),
    ("Focal Loss for Dense Object Detection", "1708.02002"),
    ("Using AI for scientific discovery", "2101.11465"),
    ("Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", "1910.13461"),
    ("Neural Machine Translation by Jointly Learning to Align and Translate", "1409.0473"),
    ("A Method for Stochastic Optimization", "1412.6980"),
    ("Towards Real-Time Object Detection with Region Proposal Networks", "1506.01497"),
]


# def strip_version(arxiv_id):
#     if arxiv_id is None:
#         return ""
#     return re.sub(r'v\d+$', '', arxiv_id)

# correct = 0
# total = len(test_data)
# results = []
# start_time = time.time()  # ⏱️ 开始计时

# for title, expected_id in test_data:
#     try:
#         result = search_paper_by_title(title)  # 这里可能返回 dict 或 str
#         if isinstance(result, dict):
#             result_id = result.get("arxiv_id", "")
#         else:
#             result_id = result

#         is_correct = strip_version(expected_id) == strip_version(result_id)
#         if is_correct:
#             correct += 1
#         results.append((title, expected_id, result_id, is_correct))
#         print(f"[{'✅' if is_correct else '❌'}] Title: {title}\n\tExpected: {expected_id} | Found: {result_id}")
#     except Exception as e:
#         print(f"[⚠️] Error processing '{title}': {e}")
#     # time.sleep(1)  # 限速，避免 API 被限流

# end_time = time.time()  # ⏱️ 结束计时
# total_duration = end_time - start_time
# avg_duration = total_duration / total if total else 0

# accuracy = correct / total * 100
# print(f"\n✅ 总体正确率：{correct}/{total} = {accuracy:.2f}%")
# print(f"⏱️ 总耗时：{total_duration:.2f} 秒")
# print(f"⏱️ 平均每个样本耗时：{avg_duration:.2f} 秒")

import time
import re
import json
import csv
import os
from search_utils import search_paper_by_title  # ✅ 替换成你自己的模块路径

# ⏳ 清理 arXiv ID 的版本号
def strip_version(arxiv_id):
    if arxiv_id is None:
        return ""
    return re.sub(r'v\d+$', '', arxiv_id)

# ✅ 加载 JSON 引用标题
with open("create_dataset/result/dataset/process_json/math_ids_metadata_1000.json", "r", encoding="utf-8") as f:
    ref_data = json.load(f)

# ✅ 提取所有 reference 中的 title（去重）
seen_titles = set()
test_titles = []

for paper_id, sections in ref_data.items():
    if not isinstance(sections, dict):
        continue
    for section, papers in sections.items():
        for paper in papers:
            title = paper.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                test_titles.append(title)

# ✅ 写入 CSV（行缓冲 + 实时 flush）
csv_path = "json_reference_search_results_second.csv"
with open(csv_path, "w", newline='', encoding='utf-8', buffering=1) as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Title", "Predicted_ID", "Matched"])  # 写表头
    f_csv.flush()  # 确保表头已写入

    correct = 0
    total = len(test_titles)
    start_time = time.time()

    for idx, title in enumerate(test_titles, 1):
        try:
            result = search_paper_by_title(title)
            result_id = result.get("arxiv_id") if isinstance(result, dict) else result
            is_correct = bool(result_id)

            if is_correct:
                correct += 1

            writer.writerow([title, result_id, is_correct])
            f_csv.flush()             # 强制将该行写入磁盘
            # os.fsync(f_csv.fileno()) # 可选：让操作系统也立即同步

            print(f"[{'✅' if is_correct else '❌'}] ({idx}/{total}) Title: {title}\n\tFound: {result_id}")

        except Exception as e:
            writer.writerow([title, "", False])
            f_csv.flush()
            print(f"[⚠️] Error processing '{title}': {e}")

    end_time = time.time()
    total_duration = end_time - start_time
    avg_duration = total_duration / total if total else 0
    accuracy = correct / total * 100

    print(f"\n✅ 总体正确率：{correct}/{total} = {accuracy:.2f}%")
    print(f"⏱️ 总耗时：{total_duration:.2f} 秒")
    print(f"⏱️ 平均每个样本耗时：{avg_duration:.2f} 秒")
