# create dataset_说明
##  AutoScholarQuery 构建
### arxiv_utils.py
>*从arXiv上的论文HTML页面中提取每个章节对应的引用信息（包括引用的标题、作者、期刊和发表日期），具体包括以下几个步骤：*

* 对每个论文 ID：从ar5iv 获取 HTML 页面；
* 使用 BeautifulSoup 分析结构，提取章节标题、正文和引用标记（\cite{}）；
* 通过LLM（如 qwen-72b）提取引用元数据（title、authors、journal 等）；
* 最终保存json文件：章节title与该章节内引用title（也可选author、journal等）。

### extract_section.py 
>*从arXiv 论文中提取 “Introduction” 段落内容，并结合引用信息，构建一个带有原文与被引论文标题的数据集。整体流程分两步：*
#### 第一步：提取论文的 Introduction 内容（可以改为其他section）
函数：process_arxiv_ids_for_section(...)

输入：
math_ids.json：包含 arXiv ID 的列表。

操作：
* 遍历每个 arXiv ID，访问其对应的 HTML 页面（通过 ar5iv）。
* 用 extract_section_by_title 函数提取指定章节（默认为 "Introduction"）的正文。

输出：保存为 math_ids_introduction_10000.json，格式如：
```
{
  "2401.00001": "This paper studies...",
  "2401.00002": "We propose a new method..."
}
```
#### 第二步：合并 Introduction 内容与引用的标题
函数：merge_intro_with_titles(...)

输入：
* math_ids_introduction_10000.json：每篇论文的引言文本。
* math_ids_metadata_10000.json：每篇论文章节引用的元信息（如标题）。

操作：
* 从引用信息中提取出 “Introduction” 章节引用的所有论文标题；
* 与对应的 Introduction 文本配对。
  
输出：
保存为 math_ids_combined_output_10000.json，格式如下：
```
{
  "2401.00001": {
    "text": "This paper studies...",
    "titles": [
      "A Neural Network Approach to Optimization",
      "Understanding Gradient Descent"
    ]
  }
}
```
### build_ASQ.py
>*利用LLM，从论文的引言部分和引用文献中自动生成“query-answer”问答对，用于构建学术问答数据集。它支持多线程并发生成、去重、断点续跑，并将结果保存为 .jsonl 文件。*
* 加载数据：从 math_ids_combined_output_*.json 中读取每篇论文的引言文本和其对应的引用文献标题列表。
* 构建一个自然语言提示词，引导LLM（如 qwen-72b 或 gpt-4o）从引言文本和引用标题中推断可能的研究问题，并以问答对形式返回。

形式如下：
```
{"arxiv_id": "2001.00002", "question": "Which papers have studied the classification of two-dimensional PI-algebras?", "answer": ["Classification of 222-dimensional evolution algebras, their groups of automorphisms and derivation algebras", "Classification of two-dimensional Jordan algebras", "On Two-Dimensional Power Associative Algebras Over Algebraically Closed Fields and ℝ", "Classification of two-dimensional Jordan algebras over ℝ", "Classifying quadratic maps from plane to plane"], "qid": "AutoScholarQuery_train_0"}
```

### build_sft_crawler.py
>*用于构建学术搜索对话数据集，通过大语言模型生成搜索查询，并提取 arXiv 论文的元信息（如标题、摘要、章节名等）。*
主要有两个功能：
* process_arxiv_ids 函数用于提取文章title、abstract、各个sections的引用文章title
* generate_search_conversations_paralle 函数用于从ASQ dataset中提取query并判断可以提升成哪些[search]语句。
  示例：
```
{"messages": [{"role": "user", "content": "You are an elite researcher in the field of AI. You are conducting research on the following topic: What work offers detailed information on the Von Mises distribution?\n\nPlease generate a list of mutually exclusive queries to retrieve highly relevant papers. Searching for survey papers is encouraged.\n\n"}, {"role": "assistant", "content": "[Search]Detailed papers on Von Mises distribution\n[Search]Survey papers on Von Mises distribution\n[Search]Comprehensive studies on Von Mises distribution\n[Search]Review articles on Von Mises distribution\n[StopSearch]"}]}
```
```
{"messages": [{"role": "user", "content": "You are an elite researcher in the field of AI. You are conducting research on the following topic: Which paper discussed the early history of the singular value decomposition?\n\nPlease generate a list of mutually exclusive queries to retrieve highly relevant papers. Searching for survey papers is encouraged.\n\n"}, {"role": "assistant", "content": "[Search]Early history of singular value decomposition\n[Search]Survey papers on the history of singular value decomposition\n[Search]Origins and development of singular value decomposition\n[Search]Historical review of singular value decomposition methods\n[StopSearch]"}]}
```

### expand.py
>*利用Google搜索和LLM，完成“问题 → arXiv论文 → 相关章节判断”的流程。*

主要流程：
* 从已有问答数据中提取研究问题，使用 Google 搜索 site:arxiv.org 限定，获取与问题相关的论文 arXiv ID。
* 对每篇论文，提取标题、摘要、章节标题。
* 以“问题 + 论文元数据”为输入，让模型判断哪些章节可能包含与问题相关的引用，按 [Expand]section_name[StopExpand] 格式输出，并记录下来。、

示例：
```
{"messages": [{"role": "user", "content": "You are conducting research on `Which paper provides comprehensive information on the Cardioid and Von Mises distributions?`. You need to predict which sections to look at for getting more relevant papers. Title: Information geometry and synchronization phase transition in Kuramoto model\nAbstract:  Abstract\nWe discuss how the synchronization in Kuramoto model can be treated in terms of information geometry. We argue that the Fisher information is sensitive to synchronization transition, specifically components of Fisher metric diverge at the critical point. Our approach is based on the recently proposed relation between Kuramoto model and geodesics on hyperbolic space.\n
Sections: [\"Introduction\", \"II Brief review of Kuramoto model\", \"III Information geometry and Kuramoto model III.1 Basic concepts of information geometry\", \"III Information geometry and Kuramoto model III.2 Gradient flow on statistical manifold\", \"III Information geometry and Kuramoto model III.3 Cauchy distribution\", \"IV Fisher metric as the order parameter for synchronization transition IV.1 Fisher metric and phase transitions\", \"IV Fisher metric as the order parameter for synchronization transition IV.2 Fisher metric in Kuramoto model\", \"IV Fisher metric as the order parameter for synchronization transition IV.3 Fisher metric in Kuramoto-Sakaguchi model\", \"IV Fisher metric as the order parameter for synchronization transition IV.4 Kullback-Leibler divergence in Kuramoto model\", \"V Conclusion\", \"Acknowledgements\"]"}, 
{"role": "assistant", "content": "[Expand]III Information geometry and Kuramoto model III.3 Cauchy distribution[StopExpand]"}]}
```