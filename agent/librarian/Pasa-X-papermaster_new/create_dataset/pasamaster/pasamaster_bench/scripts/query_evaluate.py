# -*- coding: utf-8 -*-
"""
查询生成准确性评估脚本
使用完整的论文信息评估查询与论文的匹配质量
"""

import json
import time
import asyncio
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import os

# 添加路径以导入llm_call
current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from tools.llm_call import llm_call

# ===================== 配置参数 =====================

# 输入文件路径
BENCHMARK_JSON_PATH = "/data/duyuwen/Pasa-X/create_dataset/pasamaster/pasamaster_bench/output/benchmark_questions.json"

# 输出文件路径
EVALUATION_OUTPUT_PATH = "/data/duyuwen/Pasa-X/create_dataset/pasamaster/pasamaster_bench/output/query_accuracy_evaluation.json"

# LLM模型
LLM_MODEL = "deepseek-r1"

# 批次大小（基于完整信息，可能需要更小的批次大小）
BATCH_SIZE = 2  # 减小批次大小以容纳完整信息

# 批次间延迟（秒，避免API限制）
BATCH_DELAY = 3

# 是否启用详细日志
VERBOSE = True


# ===================== 完整信息评估函数 =====================

def create_query_accuracy_prompt(queries_batch: List[Dict]) -> str:
    """
    为一批查询创建准确性评估prompt

    Args:
        queries_batch: 一批查询数据（包含完整论文信息）

    Returns:
        str: 格式化的评估prompt
    """

    base_prompt = """You are an expert in information retrieval (IR), natural language processing (NLP), and academic knowledge evaluation.  
You are asked to evaluate the **retrieval accuracy** of automatically generated academic queries.  

Each query is paired with a list of academic papers (with title and abstract).  
Your task is to judge **how accurately and comprehensively the papers reflect the intent of the query** — that is, whether the retrieved papers match the query's meaning, topic scope, and research focus.

---

###  Your Evaluation Task

For each entry (query + paper list):

1. Read the **query** carefully.
2. Read each paper title and abstract.
3. Judge how relevant and consistent these papers are with the query's semantic intent.
4. Assign a **retrieval accuracy score (1–5)** following the criteria below.

---

###  Scoring Rubric (1–5 Scale)

**5 — Excellent match:**  
All or nearly all papers directly address the query's intent.  
The retrieved papers collectively provide strong coverage of the main topic, with high topical alignment and minimal noise.  
*(e.g., query about "CNNs in image recognition" retrieves papers like AlexNet, VGGNet, ResNet — all directly relevant.)*

**4 — Good match:**  
Most papers are highly relevant; a few are tangential or peripheral (e.g., same field but different subtask).  
The query meaning is well captured, but coverage might be slightly incomplete or mixed between subtopics.  

**3 — Moderate match:**  
Rough thematic overlap — the papers are related to the query domain (e.g., "deep learning"),  
but many do not specifically focus on the intended scope. The query's focus is partially lost.  

**2 — Poor match:**  
The retrieved papers are weakly related or only share generic terms with the query.  
Semantic alignment is low — e.g., papers in neighboring domains, or unrelated methods within the same general field.  

**1 — Very poor match:**  
Almost none of the papers address the query's intent.  
The set appears random or driven by keyword overlap only.  

---

###  Output Format

Return only valid JSON, with one entry per query:

[
  {
    "query_index": <integer>,
    "score": <1–5>,
    "reason": "<2–3 concise sentences explaining why this score was given, referencing specific topical alignment or mismatch>"
  },
  ...
]

---

### Important Instructions
- Base your judgment **only on the information provided** (query and complete paper abstracts).
- Do NOT summarize papers; focus on semantic alignment between query intent and paper content.
- Use your expertise in NLP and academic research to judge whether the papers collectively match the query's research focus.
- Be consistent across all queries.
- Evaluate each query independently.

Scoring Reference Example
Below is a concrete example of a query that would receive a score of ​5 (Excellent)​. Use this to calibrate your scoring.
 Score 5 Example:{
    "question": "What are the recent advances, key challenges, and historical developments in recommender systems and models?",
    "reason": {
      "Deep Learning based Recommender System: A Survey and New Perspectives": "The paper surveys advances in deep learning for recommender systems, discusses challenges like data sparsity, and contextualizes its historical growth in the field.",
      "Bias and Debias in Recommender System: A Survey and Future Directions": "It addresses key challenges (e.g., biases in data), reviews debiasing methods as advances, and situates bias research within historical system limitations.",
      "ColdGAN: Resolving Cold Start User Recommendation by using Generative Adversarial Networks": "Proposes an advance (GAN-based model) to address the cold-start challenge, aligning with the question's focus on innovative solutions to key problems.",
      "A Comprehensive Overview of Recommender System and Sentiment Analysis": "Explores advances (integration of sentiment analysis into recommendations) and challenges (reliance on user-generated content), fitting the question's scope.",
      "A Brief History of Recommender Systems": "Explicitly covers historical developments of recommender systems and frameworks, while touching on advances (web architectures) and challenges (big data processing)."
    },
    "answer": [
      {
        "arxiv_id": "1707.07435",
        "title": "Deep Learning based Recommender System: A Survey and New Perspectives",
        "description": "deep learning,recommender systems,information retrieval,online information,data sparsity,web applications,",
        "abstract": "With the ever-growing volume of online information, recommender systems have\nbeen an effective strategy to overcome such information overload. The utility\nof recommender systems cannot be overstated, given its widespread adoption in\nmany web applications, along with its potential impact to ameliorate many\nproblems related to over-choice. In recent years, deep learning has garnered\nconsiderable interest in many research fields such as computer vision and\nnatural language processing, owing not only to stellar performance but also the\nattractive property of learning feature representations from scratch. The\ninfluence of deep learning is also pervasive, recently demonstrating its\neffectiveness when applied to information retrieval and recommender systems\nresearch. Evidently, the field of deep learning in recommender system is\nflourishing. This article aims to provide a comprehensive review of recent\nresearch efforts on deep learning based recommender systems. More concretely,\nwe provide and devise a taxonomy of deep learning based recommendation models,\nalong with providing a comprehensive summary of the state-of-the-art. Finally,\nwe expand on current trends and provide new perspectives pertaining to this new\nexciting development of the field."
      },
      {
        "arxiv_id": "2010.03240",
        "title": "Bias and Debias in Recommender System: A Survey and Future Directions",
        "description": "recommender system,bias,debiasing,selection bias,position bias,exposure bias,",
        "abstract": "While recent years have witnessed a rapid growth of research papers on\nrecommender system (RS), most of the papers focus on inventing machine learning\nmodels to better fit user behavior data. However, user behavior data is\nobservational rather than experimental. This makes various biases widely exist\nin the data, including but not limited to selection bias, position bias,\nexposure bias, and popularity bias. Blindly fitting the data without\nconsidering the inherent biases will result in many serious issues, e.g., the\ndiscrepancy between offline evaluation and online metrics, hurting user\nsatisfaction and trust on the recommendation service, etc. To transform the\nlarge volume of research models into practical improvements, it is highly\nurgent to explore the impacts of the biases and perform debiasing when\nnecessary. When reviewing the papers that consider biases in RS, we find that,\nto our surprise, the studies are rather fragmented and lack a systematic\norganization. The terminology ``bias'' is widely used in the literature, but\nits definition is usually vague and even inconsistent across papers. This\nmotivates us to provide a systematic survey of existing work on RS biases. In\nthis paper, we first summarize seven types of biases in recommendation, along\nwith their definitions and characteristics. We then provide a taxonomy to\nposition and organize the existing work on recommendation debiasing. Finally,\nwe identify some open challenges and envision some future directions, with the\nhope of inspiring more research work on this important yet less investigated\ntopic. The summary of debiasing methods reviewed in this survey can be found at\n\\url{https://github.com/jiawei-chen/RecDebiasing}."
      },
      {
        "arxiv_id": "2011.12566",
        "title": "ColdGAN: Resolving Cold Start User Recommendation by using Generative\n  Adversarial Networks",
        "description": "coldgan,gan based model,new user cold-start problem,rating distributions,time-based function,recommendation system,",
        "abstract": "Mitigating the new user cold-start problem has been critical in the\nrecommendation system for online service providers to influence user experience\nin decision making which can ultimately affect the intention of users to use a\nparticular service. Previous studies leveraged various side information from\nusers and items; however, it may be impractical due to privacy concerns. In\nthis paper, we present ColdGAN, an end-to-end GAN based model with no use of\nside information to resolve this problem. The main idea of the proposed model\nis to train a network that learns the rating distributions of experienced users\ngiven their cold-start distributions. We further design a time-based function\nto restore the preferences of users to cold-start states. With extensive\nexperiments on two real-world datasets, the results show that our proposed\nmethod achieves significantly improved performance compared with the\nstate-of-the-art recommenders."
      },
      {
        "arxiv_id": "2109.08794",
        "title": "A Comprehensive Overview of Recommender System and Sentiment Analysis",
        "description": "recommender systems,sentiment analysis,user-generated reviews,aspect-based sentiment analysis,e-commerce platforms,numeric ratings,",
        "abstract": "Recommender system has been proven to be significantly crucial in many fields\nand is widely used by various domains. Most of the conventional recommender\nsystems rely on the numeric rating given by a user to reflect his opinion about\na consumed item; however, these ratings are not available in many domains. As a\nresult, a new source of information represented by the user-generated reviews\nis incorporated in the recommendation process to compensate for the lack of\nthese ratings. The reviews contain prosperous and numerous information related\nto the whole item or a specific feature that can be extracted using the\nsentiment analysis field. This paper gives a comprehensive overview to help\nresearchers who aim to work with recommender system and sentiment analysis. It\nincludes a background of the recommender system concept, including phases,\napproaches, and performance metrics used in recommender systems. Then, it\ndiscusses the sentiment analysis concept and highlights the main points in the\nsentiment analysis, including level, approaches, and focuses on aspect-based\nsentiment analysis."
      },
      {
        "arxiv_id": "2209.01860",
        "title": "A Brief History of Recommender Systems",
        "description": "recommender system,recommendation models,web architectures,content recommendation,big data value,internet history,",
        "abstract": "Soon after the invention of the Internet, the recommender system emerged and\nrelated technologies have been extensively studied and applied by both academia\nand industry. Currently, recommender system has become one of the most\nsuccessful web applications, serving billions of people in each day through\nrecommending different kinds of contents, including news feeds, videos,\ne-commerce products, music, movies, books, games, friends, jobs etc. These\nsuccessful stories have proved that recommender system can transfer big data to\nhigh values. This article briefly reviews the history of web recommender\nsystems, mainly from two aspects: (1) recommendation models, (2) architectures\nof typical recommender systems. We hope the brief review can help us to know\nthe dots about the progress of web recommender systems, and the dots will\nsomehow connect in the future, which inspires us to build more advanced\nrecommendation services for changing the world better."
      }
    ]
  }
    reason: Excellent match: Paper 1 surveys deep learning advances (recent advances), Paper 2 addresses key bias challenges, Paper 5 covers historical developments, and Papers 3-4 broaden the scope (cold-start, sentiment analysis). All align directly with the query's request for advances, challenges, and history. The papers collectively span foundational models, temporal evolution, and technical hurdles while maintaining topical cohesion.
Now, here are the queries to evaluate with their COMPLETE information:
"""

    # 添加每个查询的完整信息
    for i, query_data in enumerate(queries_batch):
        batch_index = query_data["batch_index"]
        query_text = query_data["question"]
        papers = query_data["answer"]

        base_prompt += f"\n\n=== Query {batch_index} ===\n"
        base_prompt += f"Query: {query_text}\n"
        base_prompt += f"Papers ({len(papers)}):\n"

        # 添加完整的论文信息
        for j, paper in enumerate(papers):
            base_prompt += f"  Paper {j + 1}:\n"
            base_prompt += f"    Title: {paper.get('title', 'N/A')}\n"
            base_prompt += f"    Abstract: {paper.get('abstract', 'No abstract available')}\n"

    base_prompt += "\n\nNow, please provide your evaluations in the specified JSON format based on the COMPLETE information above."

    return base_prompt


def process_batch_evaluation_response(response: str, queries_batch: List[Dict]) -> List[Dict]:
    """
    处理批次评估的LLM响应

    Args:
        response: LLM的响应文本
        queries_batch: 对应的查询批次数据

    Returns:
        List[Dict]: 处理后的评估结果
    """
    try:
        # 解析JSON响应
        json_begin = response.find("[")
        json_end = response.rfind("]")
        if json_begin != -1 and json_end != -1:
            evaluation_data = json.loads(response[json_begin:json_end + 1])

            # 验证响应格式
            if isinstance(evaluation_data, list):
                results = []
                for eval_item in evaluation_data:
                    if all(key in eval_item for key in ["query_index", "score", "reason"]):
                        # 找到对应的原始查询数据
                        original_query = None
                        for q in queries_batch:
                            if q["batch_index"] == eval_item["query_index"]:
                                original_query = q
                                break

                        if original_query:
                            result = {
                                "query_index": eval_item["query_index"],
                                "original_index": original_query["original_index"],
                                "question": original_query["question"],
                                "paper_count": len(original_query["answer"]),
                                "score": eval_item["score"],
                                "reason": eval_item["reason"]
                            }
                            results.append(result)

                return results
            else:
                print("Warning: Invalid response format from LLM")
                return []
        else:
            print("Warning: Could not find JSON in LLM response")
            return []

    except Exception as e:
        print(f"Error processing batch evaluation response: {e}")
        return []


def evaluate_queries_with_full_info(queries_data: List[Dict]) -> List[Dict]:
    """
    使用完整信息批次评估所有查询

    Args:
        queries_data: 所有查询数据（包含完整论文信息）

    Returns:
        List[Dict]: 所有评估结果
    """
    all_evaluation_results = []

    # 准备批次数据
    batches = []
    for i in range(0, len(queries_data), BATCH_SIZE):
        batch_data = []
        for j, query in enumerate(queries_data[i:i + BATCH_SIZE]):
            batch_item = query.copy()
            batch_item["batch_index"] = i + j  # 批次内的索引
            batch_item["original_index"] = i + j  # 原始索引
            batch_data.append(batch_item)
        batches.append(batch_data)

    print(f"总共 {len(batches)} 个批次，每批 {BATCH_SIZE} 个查询")
    print(f"使用完整论文信息（无长度限制）")

    # 处理每个批次
    for batch_idx, batch in enumerate(tqdm(batches, desc="处理批次")):
        if VERBOSE:
            print(f"\n处理批次 {batch_idx + 1}/{len(batches)}")
            print(f"批次包含查询: {[item['batch_index'] for item in batch]}")
            # 显示批次信息统计
            total_papers = sum(len(item['answer']) for item in batch)
            print(f"批次总论文数: {total_papers}")

        try:
            # 创建完整信息的批次prompt
            prompt = create_query_accuracy_prompt(batch)

            if VERBOSE:
                print(f"Prompt长度: {len(prompt)} 字符")
                # 估计token数量（粗略估算）
                estimated_tokens = len(prompt) // 4  # 常见估算：1 token ≈ 4字符
                print(f"估计token数量: ~{estimated_tokens}")

            # 调用LLM
            response = llm_call(prompt, LLM_MODEL)

            if VERBOSE:
                print(f"响应长度: {len(response)} 字符")

            # 处理响应
            batch_results = process_batch_evaluation_response(response, batch)

            if batch_results:
                all_evaluation_results.extend(batch_results)
                if VERBOSE:
                    print(f"批次 {batch_idx + 1} 成功评估 {len(batch_results)} 个查询")
            else:
                print(f"警告: 批次 {batch_idx + 1} 未获得有效评估结果")
                # 为失败的批次创建默认结果
                for item in batch:
                    all_evaluation_results.append({
                        "query_index": item["batch_index"],
                        "original_index": item["original_index"],
                        "question": item["question"],
                        "paper_count": len(item["answer"]),
                        "score": -1,
                        "reason": "Batch evaluation failed - no valid response"
                    })

            # 批次间延迟
            if batch_idx < len(batches) - 1:  # 不是最后一个批次
                time.sleep(BATCH_DELAY)

        except Exception as e:
            print(f"处理批次 {batch_idx + 1} 时出错: {e}")
            # 为出错的批次创建默认结果
            for item in batch:
                all_evaluation_results.append({
                    "query_index": item["batch_index"],
                    "original_index": item["original_index"],
                    "question": item["question"],
                    "paper_count": len(item["answer"]),
                    "score": -1,
                    "reason": f"Error: {str(e)}"
                })

    return all_evaluation_results


def calculate_comprehensive_statistics(evaluation_results: List[Dict]) -> Dict[str, Any]:
    """
    计算全面的评估统计信息

    Args:
        evaluation_results: 所有评估结果

    Returns:
        Dict: 统计信息
    """
    valid_scores = [result["score"] for result in evaluation_results if result["score"] > 0]
    failed_evaluations = len([result for result in evaluation_results if result["score"] <= 0])

    if not valid_scores:
        return {
            "total_queries": len(evaluation_results),
            "valid_evaluations": 0,
            "failed_evaluations": failed_evaluations,
            "average_score": 0,
            "score_distribution": {},
            "quality_breakdown": {}
        }

    # 基础统计
    statistics = {
        "total_queries": len(evaluation_results),
        "valid_evaluations": len(valid_scores),
        "failed_evaluations": failed_evaluations,
        "success_rate": len(valid_scores) / len(evaluation_results),
        "average_score": sum(valid_scores) / len(valid_scores),
        "score_std": (sum((x - sum(valid_scores) / len(valid_scores)) ** 2 for x in valid_scores) / len(valid_scores)) ** 0.5,
        "min_score": min(valid_scores),
        "max_score": max(valid_scores),
        "score_distribution": {
            "5_excellent": len([s for s in valid_scores if s == 5]),
            "4_good": len([s for s in valid_scores if s == 4]),
            "3_moderate": len([s for s in valid_scores if s == 3]),
            "2_poor": len([s for s in valid_scores if s == 2]),
            "1_very_poor": len([s for s in valid_scores if s == 1]),
        }
    }

    # 质量分类
    statistics["quality_breakdown"] = {
        "high_accuracy": statistics["score_distribution"]["5_excellent"] + statistics["score_distribution"]["4_good"],
        "medium_accuracy": statistics["score_distribution"]["3_moderate"],
        "low_accuracy": statistics["score_distribution"]["2_poor"] + statistics["score_distribution"]["1_very_poor"]
    }

    # 计算百分比
    statistics["accuracy_percentages"] = {
        "high_accuracy_pct": statistics["quality_breakdown"]["high_accuracy"] / len(valid_scores) * 100,
        "medium_accuracy_pct": statistics["quality_breakdown"]["medium_accuracy"] / len(valid_scores) * 100,
        "low_accuracy_pct": statistics["quality_breakdown"]["low_accuracy"] / len(valid_scores) * 100
    }

    return statistics


def analyze_query_patterns(evaluation_results: List[Dict], queries_data: List[Dict]) -> Dict[str, Any]:
    """
    分析查询模式

    Args:
        evaluation_results: 评估结果
        queries_data: 原始查询数据

    Returns:
        Dict: 查询模式分析
    """
    # 按得分分组
    high_accuracy = [r for r in evaluation_results if r["score"] >= 4]
    medium_accuracy = [r for r in evaluation_results if r["score"] == 3]
    low_accuracy = [r for r in evaluation_results if r["score"] <= 2]

    # 分析高质量查询的特征
    high_accuracy_features = {
        "avg_paper_count": sum(r["paper_count"] for r in high_accuracy) / len(high_accuracy) if high_accuracy else 0,
        "avg_query_length": sum(len(r["question"]) for r in high_accuracy) / len(high_accuracy) if high_accuracy else 0,
    }

    # 分析低质量查询的问题
    low_accuracy_issues = {
        "common_reasons": [],
        "avg_paper_count": sum(r["paper_count"] for r in low_accuracy) / len(low_accuracy) if low_accuracy else 0,
        "avg_query_length": sum(len(r["question"]) for r in low_accuracy) / len(low_accuracy) if low_accuracy else 0
    }

    return {
        "high_accuracy_queries": len(high_accuracy),
        "medium_accuracy_queries": len(medium_accuracy),
        "low_accuracy_queries": len(low_accuracy),
        "high_accuracy_features": high_accuracy_features,
        "low_accuracy_issues": low_accuracy_issues
    }


# ===================== 主函数 =====================

def main():
    """
    主函数：加载完整数据，使用完整信息评估查询准确性，保存详细结果
    """
    try:
        # 1. 加载benchmark_question.json数据
        print("正在加载benchmark_question.json数据...")
        with open(BENCHMARK_JSON_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)

        print(f"成功加载 {len(queries_data)} 个查询")

        # 显示详细的查询概览
        total_papers = sum(len(query['answer']) for query in queries_data)
        print(f"总论文数量: {total_papers}")

        for i, query in enumerate(queries_data):
            papers = query['answer']
            print(f"  查询 {i}: {query['question']}")
            print(f"    包含 {len(query['answer'])} 篇论文")

            if papers:
                # 提取所有arxiv_id
                arxiv_ids = [paper.get('arxiv_id', 'N/A') for paper in papers]
                print(f"    所有论文的arxiv_id: {arxiv_ids}")
            else:
                print("    该查询没有论文")

            print()  # 空行分隔

        # 2. 使用完整信息批次评估所有查询
        print(f"\n开始完整信息批次评估")
        print(f"批次大小: {BATCH_SIZE}, 批次延迟: {BATCH_DELAY}秒")
        print(f"使用完整论文标题和摘要（无长度限制）")

        start_time = time.time()
        evaluation_results = evaluate_queries_with_full_info(queries_data)
        end_time = time.time()

        print(f"\n评估完成！总耗时: {end_time - start_time:.2f}秒")

        # 3. 计算统计信息
        statistics = calculate_comprehensive_statistics(evaluation_results)

        # 4. 分析查询模式
        # pattern_analysis = analyze_query_patterns(evaluation_results, queries_data)

        # 5. 保存完整评估结果
        output_data = {
            "evaluation_config": {
                "batch_size": BATCH_SIZE,
                "batch_delay": BATCH_DELAY,
                "model": LLM_MODEL,
                "evaluation_time_seconds": end_time - start_time,
                "full_information_used": True
            },
            "queries_overview": {
                "total_queries": len(queries_data),
                "total_papers": total_papers,
                "papers_per_query_avg": total_papers / len(queries_data) if queries_data else 0
            },
            "evaluation_results": evaluation_results,
            "statistics": statistics,
            # "pattern_analysis": pattern_analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(EVALUATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n完整评估结果已保存至: {EVALUATION_OUTPUT_PATH}")

        # 6. 显示详细总结信息
        if statistics["valid_evaluations"] > 0:
            print(f"\n=== 查询准确性评估总结 ===")
            print(f"总查询数: {statistics['total_queries']}")
            print(f"成功评估: {statistics['valid_evaluations']} ({statistics['success_rate']:.1%})")
            print(f"失败评估: {statistics['failed_evaluations']}")
            print(f"平均得分: {statistics['average_score']:.2f} ± {statistics['score_std']:.2f}")
            print(f"得分范围: {statistics['min_score']} - {statistics['max_score']}")
            print(f"\n得分分布:")
            print(f"  5 (优秀): {statistics['score_distribution']['5_excellent']}")
            print(f"  4 (良好): {statistics['score_distribution']['4_good']}")
            print(f"  3 (中等): {statistics['score_distribution']['3_moderate']}")
            print(f"  2 (较差): {statistics['score_distribution']['2_poor']}")
            print(f"  1 (很差): {statistics['score_distribution']['1_very_poor']}")
            print(f"\n准确性分类:")
            print(
                f"  高准确性 (4-5分): {statistics['quality_breakdown']['high_accuracy']} ({statistics['accuracy_percentages']['high_accuracy_pct']:.1f}%)")
            print(
                f"  中等准确性 (3分): {statistics['quality_breakdown']['medium_accuracy']} ({statistics['accuracy_percentages']['medium_accuracy_pct']:.1f}%)")
            print(
                f"  低准确性 (1-2分): {statistics['quality_breakdown']['low_accuracy']} ({statistics['accuracy_percentages']['low_accuracy_pct']:.1f}%)")

        # 7. 显示查询模式分析
        # print(f"\n=== 查询模式分析 ===")
        # print(f"高准确性查询: {pattern_analysis['high_accuracy_queries']}")
        # print(f"中等准确性查询: {pattern_analysis['medium_accuracy_queries']}")
        # print(f"低准确性查询: {pattern_analysis['low_accuracy_queries']}")

        # 8. 显示所有评估结果
        print(f"\n=== 所有查询评估结果 ===")
        valid_results = [r for r in evaluation_results if r["score"] > 0]
        for i, result in enumerate(valid_results):
            print(f"\n查询 {result['original_index']}:")
            print(f"  问题: {result['question']}")
            print(f"  论文数量: {result['paper_count']}")
            print(f"  准确性得分: {result['score']}/5")
            print(f"  评估理由: {result['reason']}")
            print("-" * 80)  # 添加分隔线

    except FileNotFoundError:
        print(f"错误: 找不到文件 {BENCHMARK_JSON_PATH}")
    except json.JSONDecodeError:
        print(f"错误: {BENCHMARK_JSON_PATH} 不是有效的JSON文件")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()