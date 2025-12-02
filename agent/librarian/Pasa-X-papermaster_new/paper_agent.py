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
import json
import ast
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from log import logger
from model_vllm import Agent
from paper_node import PaperNode

from tools.web_search import duckduckgo_search_arxiv_id, google_search_arxiv_id, arxiv_search_id
from utils import search_paper_by_title
import os
import traceback
current_path = os.path.dirname(os.path.abspath(__file__))

#from utils import search_abs_by_id,search_ref_by_id
from utils import search_abs_by_id, search_ref_by_id, search_content_by_arxiv_id

from search_utils.call_mtr import CallMetricsTracker
metrics_tracker = CallMetricsTracker()
from tools.search_info import search_meta_info

with open(f"{current_path}/config/config.json", "r") as f:
    config = json.load(f)
USE_LOCAL_SEARCH = config["use_local_search"]
FREEZE_MODE = False
FREEZE_MODE2 = False
FREEZE_PATH = "/data/duyuwen/Pasa-X/result/result_RealScholarQuery_20250928_30b"
# FREEZE_PATH = "/data/Tianjin/Pasa-X/result/result_APAS_bench_20250905"
def infer_score(agent, prompts):
    """
    对输入的prompts批量推理，返回相关性分数（0.0-1.0）
    专门用于计算论文摘要与关键词的相关性分数
    
    Args:
        agent: Agent实例，用于调用模型
        prompts: prompt列表
    
    Returns:
        list: 相关性分数列表（每个分数在0.0-1.0之间）
    """
    try:
        if not prompts:
            return []
        
        def get_score(prompt):
            # logger.info(f"🔍 [DEBUG] 计算分数的prompt: {prompt}")
            response = agent._call_llm(prompt, max_tokens=10, temperature=0.0, top_p=1.0)
            content = response.choices[0].message.content.strip()
            
            # 尝试提取数值分数
            score_match = re.search(r'([0-1](?:\.\d+)?)', content)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    # logger.info(f"🔍 [DEBUG] 解析分数: {score}")
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass
            
            # logger.info(f"⚠️  无法解析分数，返回内容: '{content[:50]}...'")
            return 0.5
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            scores = list(executor.map(get_score, prompts))
        return scores
    except Exception as e:
        # logger.info(f"❌ infer_score执行失败: {e}")
        return [0.5] * len(prompts)

class PaperAgent:
    def __init__(
        self,
        task_id:        str,
        user_query:     str,
        keyminder:      Agent,  # 新架构：关键词提取器和查询生成器
        top_ranker:     Agent,  # 新架构：定位器
        ranker:         Agent,  # 评分器
        end_date:       str = datetime.now().strftime("%Y%m%d"),
        prompts_path:   str = "papermaster_prompt.json",
        expand_layers:  int = 1,
        search_queries: int = 5,  # 向后兼容参数
        search_papers:  int = 10,
        expand_papers:  int = 20,
        threads_num:    int = 20,
        max_num:        int = -1,
        time_limit:     int = -1,
        build_ranker_sample: bool = False,
        content_max_chars: int = 300,  # 章节内容最大字符数
        further_sample_limit: int = 50,  # further_search中抽样的论文数上限
        idx:    int = 0,
        output_dir: str =  f"{current_path}/result"
    ) -> None:
        self.user_query = user_query
        self.keyminder = keyminder
        self.top_ranker = top_ranker
        self.ranker = ranker
        self.end_date = end_date
        self.content_max_chars = content_max_chars

        self.summarizer = self.ranker  # 默认用 ranker 这路模型来做“核心知识抽取”

        self.idx = idx
        self.output_dir = output_dir
        self.further_sample_limit = further_sample_limit
        self.hit = 0
        self.search_times = 0
        self.logger = logger.bind(task_id=task_id)
        self.task_id = task_id
        self.queries = {}
        # 添加必要的参数检查
        if self.keyminder is None:
            raise ValueError("keyminder参数不能为None，请传入有效的Agent实例")
        if self.top_ranker is None:
            raise ValueError("top_ranker参数不能为None，请传入有效的Agent实例")
        if self.ranker is None:
            raise ValueError("ranker参数不能为None，请传入有效的Agent实例")
        
        with open(prompts_path, "r") as f:
            self.prompts = json.load(f)
        
        # 验证必要的prompt是否存在
        required_prompts = [
            ("keyminder", "extract_keywords_and_queries"),
            ("locater", "locate_sections"),
            ("ranker", "keyword_relevance")
        ]
        
        for module, prompt_key in required_prompts:
            if module not in self.prompts:
                raise ValueError(f"prompts文件中缺少'{module}'模块")
            if prompt_key not in self.prompts[module]:
                raise ValueError(f"prompts文件中缺少'{module}.{prompt_key}'提示词")
            
        self.root = PaperNode({
            "title": user_query,
            "extra": {
                "touch_ids": [],
                "recall_papers": [],
                "recall_papers_abstract": [],
                "arxiv_ids": [],
                "score": [],
                "sorted_papers": [],  # 存储排序后的完整论文信息
                "final_selected_papers_title": [],  # 存储排序后的论文标题
                "all_papers": [],  # 存储所有搜索到的论文（不论分数）
                "all_papers_title": [],  # 存储所有搜索到的论文标题
                "subqueries": {},  # 存储keyminder生成的子查询
                "false_sample": {
                    "prompt": [],
                    "answer": [],
                    "title": [],
                }
            },
            "relevant_section": {},  # 初始化relevant_section字段
            "relevant_section_ref": {}  # 初始化relevant_section_ref字段
        })

        # hyperparameters
        self.max_num         = max_num
        self.time_limit      = time_limit
        self.start_time      = datetime.now()
        self.expand_layers   = expand_layers
        self.search_queries  = search_queries  # 保留兼容参数
        self.search_papers   = search_papers
        self.expand_papers   = expand_papers
        self.threads_num     = threads_num
        self.build_ranker_sample = build_ranker_sample
        self.papers_queue    = []
        self.expand_start    = 0
        self.lock            = threading.Lock()
        
        # 深度管理
        self.depth_stats     = {}  # 记录每个深度的论文数量
        # 记录已搜索过的查询，避免重复
        self.searched_queries = set()
        # 记录用户原始生成的初始查询
        self.original_queries = []

    def get_paper_info(self,paper,meta_info,depth):
        try:
            if meta_info is not None:
                if meta_info['paired_journal'] is not None and meta_info['paired_journal'] != "":
                    journal = meta_info['paired_journal']
                elif meta_info['journal'] is not None and meta_info['journal'] != "":
                    journal = meta_info['journal']
                else:
                    journal = "arxiv"
            else:
                journal = "arxiv"
            paper_node = PaperNode({
                "title": paper["title"],
                "arxiv_id": paper["arxiv_id"],
                "depth": depth,
                "abstract": paper["abstract"],
                "sections": paper.get("sections", ""),
                "journal": journal,
                "authors": paper.get("authors", ""),
                "year": paper.get("year", ""),
                "h5": meta_info['h5_index'] if meta_info is not None and meta_info['h5_index'] is not None else -1,
                "IF": meta_info['IF'] if meta_info is not None and meta_info['IF'] is not None else -1,
                'CCF': meta_info['CCF'] if meta_info is not  None and meta_info['CCF'] is not None else "",
                "citations": meta_info["citedBy"] if meta_info is not None and meta_info["citedBy"] is not None else -1,
                "source": "Search " + paper.get("source", ""),
                "select_score": -1,
                "extra": {},
                "relevant_section": {},  # 初始化relevant_section字段
                "relevant_section_ref": {}  # 初始化relevant_section_ref字段
            })
        except:
            paper_node = PaperNode({
                "title": paper["title"],
                "arxiv_id": paper["arxiv_id"],
                "depth": depth,
                "abstract": paper["abstract"],
                "sections": paper.get("sections", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year", ""),
                "source": "Search " + paper.get("source", ""),
                "select_score": -1,
                "extra": {},
                "relevant_section": {},  # 初始化relevant_section字段
                "relevant_section_ref": {}  # 初始化relevant_section_ref字段
            })
            
        return paper_node
    
    @staticmethod
    def do_parallel(func, args, num):
        threads = []
        for _ in range(num):
            thread = threading.Thread(target=func, args=args)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def check_limit(self):
        if self.max_num != -1 and len(self.root.extra["recall_papers"]) >= self.max_num:
            return True
        if self.time_limit != -1 and (datetime.now() - self.start_time).seconds >= self.time_limit:
            return True
        return False
    
    def get_depth_statistics(self):
        """获取当前深度统计信息"""
        depth_stats = {}
        for paper in self.papers_queue:
            depth_stats[paper.depth] = depth_stats.get(paper.depth, 0) + 1
        return depth_stats
    
    def print_progress_summary(self, current_depth=None):
        """打印进度摘要"""
        stats = self.get_depth_statistics()
        total_papers = len(self.papers_queue)
        
        self.logger.info(f"📈 [PROGRESS] =============== 进度摘要 ===============")
        self.logger.info(f"📊 [PROGRESS] 总论文数: {total_papers}")
        self.logger.info(f"📈 [PROGRESS] 深度分布: {stats}")
        if current_depth is not None:
            self.logger.info(f"🎯 [PROGRESS] 当前处理深度: {current_depth}")
        self.logger.info(f"📈 [PROGRESS] =========================================\n")

    def translate(self, query):
        prompt_template = self.prompts["keyminder"]["translate"]
        prompt = prompt_template.format(query=query)
        
        response = self.keyminder.infer(prompt,use_cot=False)
        m = re.search(r"<query>\s*(.*?)\s*<(?:\\|/)query>", response, re.IGNORECASE | re.DOTALL)
        return m.group(1) if m else query

    def extract_keywords_and_queries(self):
        """Keyminder: 从用户查询中提取关键词+权重+搜索查询"""
        self.logger.info(f"\n🔍 [Keyminder] 开始提取关键词和查询...")
        self.logger.info(f"📝 [Keyminder] 输入查询: {self.user_query}")
        
        start_time = time.time()
        prompt_template = self.prompts["keyminder"]["extract_keywords_and_queries"]
        prompt = prompt_template.format(user_query=self.user_query)
        
        response = self.keyminder.infer(prompt,use_cot=True)
        elapsed_time = time.time() - start_time
        metrics_tracker.add_agent_time(keyminder=elapsed_time)
        
        # logger.info(f"📥 [Keyminder] 输出response长度: {len(response)} 字符")
        # logger.info(f"📥 [Keyminder] 输出response内容:")
        # logger.info("-" * 40)
        # logger.info(response)
        # logger.info("-" * 40)
        # logger.info(f"⏱️ [Keyminder] 处理耗时: {elapsed_time:.2f}秒")
        # logger.info("="*80)
        
        try:
            # 解析JSON格式的关键词、权重和查询
            # logger.info(f"🔄 [Keyminder] 开始解析JSON响应...")
            
            # 处理可能包含```json的响应
            json_text = response.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]  # 移除开头的```json
            if json_text.endswith("```"):
                json_text = json_text[:-3]  # 移除结尾的```
            json_text = json_text.strip()
            
            result = json.loads(json_text)
            queries = result.get("queries", [])
            
            if not queries:
                logger.info(f"⚠️ [Keyminder] 没有生成查询，使用原始查询")
                queries = [self.user_query]
            self.queries["0"] = queries  # 保存原始查询
            self.root.extra["subqueries"]["0"] = queries  # 保存keyminder生成的子查询
            # 记录初始查询
            self.original_queries = list(queries)
            
            self.logger.info(f"📊 \033[32m[Keyminder] 最终结果:")
            self.logger.info(f"  查询: {queries}\033[0m")
            
            return queries
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.exception(f"❌ [Keyminder] JSON解析失败: {e}")
            self.logger.info(f"🔄 [Keyminder] 使用后备方案: 原始查询")
            fallback_result = ([self.user_query])
            self.logger.info(f"📊 [Keyminder] 后备结果: {fallback_result}")
            return fallback_result

    def search_papers_with_queries(self, queries):
        """直接用查询搜索论文并放入队列"""
        self.logger.info(f"\n🔍 [DirectSearch] 开始搜索论文...")
        
        start_time = time.time()
        
        # 限制查询数量
        original_count = len(queries)
        queries = queries[:self.search_queries] if self.search_queries > 0 else queries[:5]
        # logger.info(f"📊 [DirectSearch] 查询数量限制: {original_count} -> {len(queries)}")
        
        # logger.info(f"🚀 [DirectSearch] 开始并行搜索，使用 {min(len(queries), self.threads_num)} 个线程")
        
        # 记录查询
        for q in queries:
            self.searched_queries.add(q)

        # 并行搜索
        PaperAgent.do_parallel(self.search_paper_worker, (queries,), min(len(queries), self.threads_num))
        
        elapsed_time = time.time() - start_time
        metrics_tracker.add_agent_time(search=elapsed_time)
        
        self.logger.info(f"⏱️ [DirectSearch] 搜索耗时: {elapsed_time:.2f}秒")
        self.logger.info(f"📊 [DirectSearch] 当前队列中论文数量: {len(self.papers_queue)}")

    def search_paper_worker(self, queries,num=-1):
        """搜索论文的工作线程"""
        thread_id = threading.current_thread().name
        # logger.info(f"🧵 [SearchWorker-{thread_id}] 线程启动")
        
        while queries:
            with self.lock:
                if queries:
                    query = queries.pop()
                    # logger.info(f"🎯 [SearchWorker-{thread_id}] 获取查询: {query}")
                else:
                    # logger.info(f"🏁 [SearchWorker-{thread_id}] 没有更多查询，线程退出")
                    break
                    
            # logger.info(f"🌐 [SearchWorker-{thread_id}] 尝试Google搜索...")
            # 首先尝试Google搜索
            if USE_LOCAL_SEARCH:
                arxiv_ids = arxiv_search_id(query, self.search_papers, self.end_date)
                self.logger.info(f"📊 [SearchWorker-{thread_id}] 本地搜索到 {len(arxiv_ids)} 篇论文ID")
            else:
                arxiv_ids = google_search_arxiv_id(query, self.search_papers, self.end_date)
                self.logger.info(f"📊 [SearchWorker-{thread_id}] Google搜索到 {len(arxiv_ids)} 篇论文ID")
            if not arxiv_ids:
                self.logger.error(f"📊 \033[091m [SearchWorker-{thread_id}] Google搜索失败!!!\033[0m")
                return
            if num != -1:
                arxiv_ids = arxiv_ids[:num]
            searched_papers = []
            # logger.info(f"🔍 [SearchWorker-{thread_id}] 开始获取论文详情...")
            for i, arxiv_id in enumerate(arxiv_ids):
                arxiv_id = arxiv_id.split('v')[0]
                
                with self.lock:
                    if arxiv_id not in self.root.extra["touch_ids"]:
                        self.root.extra["touch_ids"].append(arxiv_id)
                    else:
                        continue
                
                paper = search_abs_by_id(arxiv_id)
                if paper is not None:
                    searched_papers.append(paper)
                    self.logger.info(f"✅ [SearchWorker-{thread_id}] 成功获取: {paper['title']}")
                else:
                    self.logger.info(f"❌ [SearchWorker-{thread_id}] 获取失败: {arxiv_id}")
            
            # 将搜索到的论文加入队列
            with self.lock:
                searched_paper_titles = [p["title"] for p in searched_papers]
                meta_infos = search_meta_info(searched_paper_titles)
                for paper, meta_info in zip(searched_papers,meta_infos):
    
                    paper_node = self.get_paper_info(paper,meta_info,0)
                    self.papers_queue.append(paper_node)
                    # 更新深度统计
                    self.depth_stats[0] = self.depth_stats.get(0, 0) + 1
            self.logger.info(f"📊 [SearchWorker-{thread_id}] 本次查询完成，获得 {len(searched_papers)} 篇有效论文，任务ID: {self.task_id}")
        
        # logger.info(f"🏁 [SearchW orker-{thread_id}] 线程结束")

    def search_paper_worker_with_depth(self, args):
        """带指定深度的搜索论文工作线程"""
        queries, target_depth = args
        thread_id = threading.current_thread().name
        while queries:
            with self.lock:
                if queries:
                    query = queries.pop()
                else:
                    break
            try:
                if USE_LOCAL_SEARCH:
                    arxiv_ids = arxiv_search_id(query, self.search_papers, self.end_date)
                    self.logger.info(f"📊 [SearchWorkerDepth-{thread_id}] 本地搜索到 {len(arxiv_ids)} 篇论文ID")
                else:
                    arxiv_ids = google_search_arxiv_id(query, self.search_papers, self.end_date)
                    self.logger.info(f"📊 [SearchWorkerDepth-{thread_id}] Google搜索到 {len(arxiv_ids)} 篇论文ID")
                if not arxiv_ids:
                    raise Exception("No papers found by google search")
            except Exception as e:
                self.logger.info(f"❌ [SearchWorkerDepth-{thread_id}] Google搜索失败: {e}")
                self.logger.info(f"🔄 [SearchWorkerDepth-{thread_id}] 尝试DuckDuckGo搜索...")
                arxiv_ids = duckduckgo_search_arxiv_id(query, self.search_papers, self.end_date)
                self.logger.info(f"📊 [SearchWorkerDepth-{thread_id}] DuckDuckGo搜索到 {len(arxiv_ids)} 篇论文ID")

            searched_papers = []
            for arxiv_id in arxiv_ids:
                arxiv_id = arxiv_id.split('v')[0]
                with self.lock:
                    if arxiv_id not in self.root.extra["touch_ids"]:
                        self.root.extra["touch_ids"].append(arxiv_id)
                    else:
                        continue
                paper = search_abs_by_id(arxiv_id)
                if paper is not None:
                    searched_papers.append(paper)
                    self.logger.info(f"✅ [SearchWorkerDepth-{thread_id}] 成功获取: {paper['title']}")
                else:
                    self.logger.info(f"❌ [SearchWorkerDepth-{thread_id}] 获取失败: {arxiv_id}")

            with self.lock:
                searched_paper_titles = [p["title"] for p in searched_papers]
                meta_infos = search_meta_info(searched_paper_titles)
                for paper, meta_info in zip(searched_papers,meta_infos):
                    paper_node = self.get_paper_info(paper,meta_info,target_depth)
                    self.papers_queue.append(paper_node)
                    self.depth_stats[target_depth] = self.depth_stats.get(target_depth, 0) + 1

    def search_papers_with_queries_at_depth(self, queries, target_depth):
        """用查询搜索论文并设置指定的深度"""
        if not queries:
            return
        self.logger.info(f"\n🔍 [FurtherSearch] 按深度搜索论文，目标深度: {target_depth}")
        start_time = time.time()
        # 限制数量，进一步搜索可以适当放宽
        queries = queries[:max(self.search_queries, 5)]
        for q in queries:
            self.searched_queries.add(q)
        PaperAgent.do_parallel(self.search_paper_worker_with_depth, ((queries, target_depth),), min(len(queries), self.threads_num))
        elapsed_time = time.time() - start_time
        metrics_tracker.add_agent_time(search=elapsed_time)
        self.logger.info(f"✅ [FurtherSearch] 完成，耗时: {elapsed_time:.2f}秒")

    def further_search(self, papers_with_content, target_depth):
        """根据上一轮论文摘要生成额外搜索query并检索

        - 每20篇论文组成一个prompt，批量生成额外查询
        - 将已搜索过的query传给模型避免重复
        - 对生成的新query进行检索，并将得到的论文加入本轮（target_depth）
        """
        try:
            if not papers_with_content:
                self.logger.info("⚠️ [FurtherSearch] 无论文内容可用于生成额外查询，跳过")
                return
            # 仅取前 self.further_sample_limit 篇
            papers_sample = papers_with_content[: self.further_sample_limit]
            # 组装摘要文本，20篇一组
            def chunk(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i+size]

            prompts = []
            # 统计批次数并打印一行摘要
            batch_size_cfg = 20
            num_batches = (len(papers_sample) + batch_size_cfg - 1) // batch_size_cfg
            self.logger.info(f"      📦 [FurtherSearch] 本轮论文: {len(papers_with_content)} | 抽样: {len(papers_sample)} | 分组大小: {batch_size_cfg} | 批次数: {num_batches}")
            for batch_idx, batch in enumerate(chunk(papers_sample, batch_size_cfg)):
                abstracts_lines = []
                for i, paper in enumerate(batch, 1):
                    title = getattr(paper, 'title', '') or ''
                    abstract = getattr(paper, 'abstract', '') or ''
                    abstracts_lines.append(f"{i}. Title: {title}\nAbstract: {abstract}")
                abstracts_block = "\n\n".join(abstracts_lines)
                # 区分原始查询与后续查询
                original_qs = list(self.original_queries) if self.original_queries else []
                expanded_qs = [q for q in sorted(self.searched_queries) if q not in set(original_qs)]
                original_queries_text = "\n".join(original_qs) if original_qs else "(none)"
                expanded_queries_text = "\n".join(expanded_qs) if expanded_qs else "(none)"
                prompt_template = self.prompts.get("further", {}).get("generate_extra_queries", "")
                if not prompt_template:
                    self.logger.info("⚠️ [FurtherSearch] 缺少 further.generate_extra_queries 提示词，跳过进一步搜索")
                    return
                prompt = prompt_template.format(
                    user_query=self.user_query,
                    original_queries=original_queries_text,
                    expanded_queries=expanded_queries_text,
                    abstracts_block=abstracts_block
                )
                prompts.append(prompt)
            if not prompts:
                self.logger.info("⚠️ [FurtherSearch] 无可用prompt，跳过")
                return

            batch_size = min(len(prompts), self.threads_num)
            responses = self.keyminder.batch_infer_safe(prompts, batch_size=batch_size, use_cot=False)
            
            # 解析响应，提取查询（每个批次最多取5条）
            new_queries = []
            # 打印一行式的 prompt 和 response 对应关系
    
            for idx_r, resp in enumerate(responses):
                text = str(resp).strip()
                # 去除代码块
                if text.startswith('```json'):
                    text = text[7:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()
                # 尝试提取方括号中的JSON数组
                l = text.find('[')
                r = text.rfind(']')
                if l != -1 and r != -1 and r > l:
                    candidate = text[l:r+1]
                else:
                    candidate = text
                parsed = None
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    self.logger.exception(f"❌ [FurtherSearch] JSON解析失败: {e}")
                    try:
                        parsed = ast.literal_eval(candidate)
                    except Exception:
                        self.logger.exception(f"❌ [FurtherSearch] 解析失败: {e}")
                        parsed = None
                # 每个批次上限为5条
                per_batch_added = 0
                if isinstance(parsed, list):
                    for q in parsed:
                        if per_batch_added >= 5:
                            break
                        if isinstance(q, str):
                            qs = q.strip()
                            if not qs:
                                continue
                            # 不重复且不与已用冲突
                            if qs in self.searched_queries:
                                continue
                            new_queries.append(qs)
                            per_batch_added += 1
                else:
                    self.logger.info(f"⚠️ [FurtherSearch] 错误响应无法解析为查询列表，原文片段: {text}")
            # 去重并过滤已使用查询
            filtered = []
            seen = set()
            for q in new_queries:
                if q in self.searched_queries:
                    continue
                if q in seen:
                    continue
                seen.add(q)
                filtered.append(q)

            if not filtered:
                self.logger.info("⚠️ [FurtherSearch] 错误无新的有效查询，跳过检索")
                return

            # 不再做全局5条限制，按批次已限制
            self.logger.info(f"📝 [FurtherSearch] 新生成查询数: {len(filtered)}")
            self.logger.info(f"📝 \033[92m [FurtherSearch] 新生成查询问题: {filtered}\033[0m")
            self.queries[f"{target_depth}"] = filtered
            self.root.extra["subqueries"][f"{target_depth}"] = filtered

            # 检索并加入指定深度
            prev_queue_size = len(self.papers_queue)
            self.search_papers_with_queries_at_depth(filtered, target_depth)
            added = len(self.papers_queue) - prev_queue_size
            self.logger.info(f"📊 [FurtherSearch] 通过额外查询新增论文: {added}")
        except Exception as e:
            logger.exception(f"❌ [FurtherSearch] 发生错误: {e}")
            
    def locate_relevant_papers(self, papers_with_content, target_depth=None):
        
        from search_utils.optimize_utils.arxiv_database import embedding_retrieval
        refs = []
        for paper in papers_with_content:
            section_ref = paper.sections
            for key,value in section_ref.items():
                refs.append(value)
        search_query = [self.user_query]
        for key,value in self.queries.items():
            search_query.extend(value)
        extend_refs = embedding_retrieval(search_query, refs)
        for ref in extend_refs:
            self.add_paper_by_title(ref, target_depth)

    def add_paper_by_title(self, title, depth):
        """根据标题搜索并添加论文到队列"""
        
        searched_paper = search_paper_by_title(title)
        if searched_paper is None:
            self.logger.info(f"❌ [AddPaper] 未找到论文: {title}")
            return False
        
        arxiv_id = searched_paper["arxiv_id"]
        
        with self.lock:
            if arxiv_id not in self.root.extra["touch_ids"]:
                self.root.extra["touch_ids"].append(arxiv_id)
            else:
                # self.logger.info(f"♻️ [AddPaper] 重复论文，跳过: {arxiv_id}")
                return False
        searched_paper_titles = [searched_paper["title"]]
        meta_info = search_meta_info(searched_paper_titles)[0]
        paper_node = self.get_paper_info(searched_paper,meta_info,depth)
        
        with self.lock:
            self.papers_queue.append(paper_node)
            # 更新深度统计
            self.depth_stats[depth] = self.depth_stats.get(depth, 0) + 1
        
        return True

    def _parse_text_response(self, response, paper_title):
        """解析纯文本格式的响应，提取分数和推理"""
        import re
        
        try:
            # 尝试从纯文本中提取分数
            score_patterns = [
                r'Score:\s*([0-9]*\.?[0-9]+)',
                r'score:\s*([0-9]*\.?[0-9]+)',
                r'评分:\s*([0-9]*\.?[0-9]+)',
                r'分数:\s*([0-9]*\.?[0-9]+)',
                r'Relevance score:\s*([0-9]*\.?[0-9]+)',
                r'relevance score:\s*([0-9]*\.?[0-9]+)'
            ]
            
            score = None
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # 确保分数在0-1范围内
                    score = max(0.0, min(1.0, score))
                    break
            
            if score is None:
                # 如果没有找到分数，尝试查找数字
                number_matches = re.findall(r'\b([0-9]*\.?[0-9]+)\b', response)
                for num_str in number_matches:
                    num = float(num_str)
                    if 0.0 <= num <= 1.0:
                        score = num
                        break
            
            if score is None:
                self.logger.info(f"⚠️ [Ranker] 论文: {paper_title} | 无法从纯文本中提取分数，使用默认值0.5")
                score = 0.5
            
            # 尝试提取推理文本
            reasoning_patterns = [
                r'Reasoning:\s*(.+?)(?:\n\n|\nScore|\nScore:|$)',
                r'reasoning:\s*(.+?)(?:\n\n|\nScore|\nScore:|$)',
                r'推理:\s*(.+?)(?:\n\n|\nScore|\nScore:|$)',
                r'理由:\s*(.+?)(?:\n\n|\nScore|\nScore:|$)',
                r'Explanation:\s*(.+?)(?:\n\n|\nScore|\nScore:|$)',
                r'explanation:\s*(.+?)(?:\n\n|\nScore|\nScore:|$)'
            ]
            
            reasoning = "No reasoning provided in text response"
            for pattern in reasoning_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    # 限制推理长度
                    if len(reasoning) > 500:
                        reasoning = reasoning[:500] + "..."
                    break
            
            # 构造JSON格式的响应
            return {
                "result": [{
                    "query": self.user_query,
                    "reasoning": reasoning,
                    "score": score
                }]
            }
            
        except Exception as e:
            self.logger.exception(f"❌ [Ranker] 论文: {paper_title} | 纯文本解析失败: {e}")
            return None
        
    def batch_rank(self, all_prompts, paper_mapping, filter_num, expand_depth, filter=False):
        import re

        def parse_ranker_output(res_text: str):
            """
            解析 Ranker 输出，支持以下格式：
            Score: 5
            Score：5
            Score: 5/5
            Score: 4 / 5 分
            评分: 5
            最后兜底：全文中找到第一个 1–5 的数字
            """
            if not isinstance(res_text, str):
                return None, ""

            # 优先解析 Score: x
            m = re.search(r"[Ss]core\s*[:：]\s*([1-5])", res_text)
            if not m:
                # 可能是 Score: 5/5
                m = re.search(r"[Ss]core\s*[:：]\s*([1-5])\s*/", res_text)

            if not m:
                # 兜底：全文搜索单独的 1–5
                m = re.search(r"\b([1-5])\b", res_text)

            if not m:
                return None, ""

            score = int(m.group(1))

            # Reasoning (可选)
            m = re.search(r"Reasoning\s*[:：](.*)", res_text, re.S)
            reasoning = m.group(1).strip() if m else ""

            return score, reasoning

        batch_size = 10 * (2 ** expand_depth)

        for idx in range(0, len(all_prompts), batch_size):
            begin = time.time()
            prompts = all_prompts[idx:idx+batch_size]

            responses = self.ranker.batch_infer_safe(
                prompts, batch_size=batch_size, use_cot=False, logprobs=True
            )

            paper_list = []

            # 遍历每篇论文
            for paper, response in zip(paper_mapping[idx:idx+batch_size], responses):
                res, prob = response  # res 是模型输出文本， prob 是 logprob 分数

                score, reasoning = parse_ranker_output(res)

                # 如果解析失败 → score = 0，防止崩
                if score is None:
                    self.logger.error(f"❌ [Ranker] 解析分数失败: {paper.title} | 原文: {res[:60]}...")
                    score = 0
                    reasoning = res

                # 最终 select_score
                paper.select_score = score #+ prob
                paper.reason = reasoning

                # 过滤逻辑
                if filter:
                    if paper.select_score < 3:
                        try:
                            self.papers_queue.remove(paper)
                            filter_num += 1
                        except:
                            pass

                if paper.select_score > 4:
                    paper_list.append(paper)

                self.logger.info(
                    f"🔍 [Ranker] 论文: {paper.title} | Score={score} + prob={prob:.3f} → {paper.select_score:.3f}"
                )

            end = time.time()
            self.logger.info(
                f"🔍 [Ranker] 新增 {len(paper_list)} 篇论文的推理，耗时 {end-begin:.2f} 秒"
            )

    def rank_papers(self, papers,expand_depth,filter=False):
            """Ranker: 根据论文摘要和查询计算与各个关键词的相关性分数"""
            name = "Ranker"
            if filter:
                name = "Selector"
            start_time = time.time()
            
            if not papers:
                return papers

            # 为每篇论文生成与所有关键词的相关性评估prompt
            all_prompts = []
            paper_mapping = []
            for paper in papers:
                if not FREEZE_MODE:
                    if paper.select_score != -1:
                        continue
                prompt_template = self.prompts["ranker"]["keyword_relevance"]
                meta_info = {
                    "authors":paper.authors,
                    "publication_date":paper.year,
                    "citations":paper.citations,
                    "journal":paper.journal,
                    "CCF":paper.ccf
                }
                prompt = prompt_template.format(
                    title=paper.title,
                    abstract=paper.abstract,
                    user_query=self.user_query,
                    meta_info=meta_info
                )
                all_prompts.append(prompt)
                paper_mapping.append(paper)
            filter_num = 0
            if all_prompts:
                self.batch_rank(all_prompts, paper_mapping, filter_num, expand_depth, filter)
            metrics_tracker.add_agent_time(ranker=time.time() - start_time)
            if filter:
                candidates = sorted(papers, key=lambda p: p.select_score, reverse=True)
                high_score_papers = [p for p in candidates if p.select_score >= 4]
                papers = candidates[:min(len(candidates),self.expand_papers)]
                if len(papers) < len(high_score_papers):
                    papers = high_score_papers[:self.expand_papers*2] #最多也不超过2倍扩展数目
                logger.info(f"🔍 [{name}] 总共过滤：{filter_num}篇论文，选取{len(papers)}篇论文,高分论文有：{len(high_score_papers)}篇")
            # 按分数排序
            # 检查所有论文是否都有分数
            for paper in papers:
                if paper.select_score == -1:
                    logger.info(f"❌ [{name}] 论文: {paper.title} | 未找到分数，depth:{paper.depth},跳过")
            return sorted(papers, key=lambda p: p.select_score, reverse=True)
        
            
        
 
    def get_paper_ref(self, papers):
        """获取论文的详细内容（章节信息）"""
        self.logger.info(f"📖 [PROCESS] 3.2 获取论文详细内容...")
        start_time = time.time()
        papers_with_content = []
        
        def fetch_content_worker(paper_list):
            while paper_list:
                with self.lock:
                    if paper_list:
                        paper = paper_list.pop(0)
                    else:
                        break
                
                if paper.sections == "" or paper.sections is None:
                    try:
                        try:
                            # 获取章节引用信息（用于扩展搜索）
                            begin = time.time()
                            paper.sections,hit = search_ref_by_id(paper.arxiv_id)
                            self.hit += hit
                            self.search_times += 1
                            end = time.time()
                            if paper.sections:
                                self.logger.info(f"✅ [FetchContent] 成功获取引用信息:{paper.arxiv_id},{paper.title}，共{len(paper.sections)}个引用，耗时: {end-begin:.2f}秒")
                            else:
                                self.logger.info(f"❌ [FetchContent] 未找到引用信息:{paper.arxiv_id},{paper.title}，耗时: {end-begin:.2f}秒")
                        finally:
                            pass
                        
                    except TimeoutError:
                        self.logger.exception(f"⏰ [FetchContent] 章节获取超时: {paper.title}")
                        paper.sections = None
                    except Exception as e:
                        self.logger.exception(f"❌ [FetchContent] 获取章节失败: {paper.title},{paper.arxiv_id}")
                        paper.sections = None
                
                if paper.sections:
                    self.logger.info(f"\033[92mSuccessfully got content for: {paper.title}\033[0m")
                    with self.lock:
                        papers_with_content.append(paper)
        
        # 并行获取内容
        paper_list = papers.copy()
        PaperAgent.do_parallel(fetch_content_worker, (paper_list,), self.threads_num)
        
        metrics_tracker.add_agent_time(get_content=time.time() - start_time)
        return papers_with_content

########################################
    def fetch_content_sections(self, paper, max_chars=None):
        """用 ar5iv 把论文按章节读出来，写入 paper.content_sections"""
        if max_chars is None:
            max_chars = self.content_max_chars

        try:
            result = search_content_by_arxiv_id(paper.arxiv_id, max_chars=max_chars)
        except Exception as e:
            self.logger.exception(f"❌ [CoreRead] 读取正文失败: {paper.arxiv_id}, {paper.title}, error: {e}")
            return None

        if not result or "sections" not in result or not result["sections"]:
            self.logger.info(f"⚠️ [CoreRead] 未获取到章节内容: {paper.arxiv_id}, {paper.title}")
            return None

        sections = result["sections"]
        # 写到 PaperNode 上（注意 PaperNode.todic 已经支持 content_sections）
        paper.content_sections = sections
        self.logger.info(f"📖 [CoreRead] 成功读取正文章节: {paper.title}，共 {len(sections)} 个章节")
        return sections
    
    def extract_core_knowledge_for_paper(self, paper):
            """
            对单篇论文：
            1) 确保已经有 content_sections（否则先读）
            2) 调 summarizer + prompt 抽核心定性/定量知识
            3) 写入 paper.extra['core_knowledge']
            """
            # 1. 确保有正文内容
            sections = getattr(paper, "content_sections", None)
            if not sections:
                sections = self.fetch_content_sections(paper)
            if not sections:
                return None

            # 2. 把章节拼成一个受控长度的字符串，避免 prompt 爆掉
            max_total_chars = 8000  # 你可以以后再调
            parts = []
            total_len = 0
            for name, text in sections.items():
                snippet = f"Section: {name}\n{text}\n\n"
                if total_len + len(snippet) > max_total_chars:
                    break
                parts.append(snippet)
                total_len += len(snippet)
            sections_str = "".join(parts)

            # 3. 取 summarizer.prompt
            prompt_template = self.prompts.get("summarizer", {}).get("extract_core_knowledge", "")
            if not prompt_template:
                self.logger.info("⚠️ [CoreKnow] 没有配置 summarizer.extract_core_knowledge 提示词，跳过抽核心知识")
                return None

            prompt = prompt_template.format(
                user_query=self.user_query,
                title=paper.title,
                abstract=paper.abstract or "",
                sections=sections_str
            )

            # 4. 调 LLM
            raw_response = self.summarizer.infer(prompt, use_cot=False)
            text = str(raw_response).strip()

            # 5. 尽量从中扒出 JSON
            json_text = text
            # 去掉 ```json 之类的 code fence
            if json_text.startswith("```"):
                first_newline = json_text.find("\n")
                if first_newline != -1:
                    json_text = json_text[first_newline+1:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]

            # 截取第一个 { 到最后一个 }
            l = json_text.find("{")
            r = json_text.rfind("}")
            if l != -1 and r != -1 and r > l:
                json_text = json_text[l:r+1]

            try:
                parsed = json.loads(json_text)
            except Exception as e:
                self.logger.exception(f"❌ [CoreKnow] JSON 解析失败: {paper.title}, 用 raw 文本兜底, error: {e}")
                parsed = {
                    "qualitative": [
                        {
                            "statement": text,
                            "kind": "raw_response",
                            "section": ""
                        }
                    ],
                    "quantitative": []
                }

            if not isinstance(parsed, dict):
                parsed = {"qualitative": [], "quantitative": []}

            qualitative = parsed.get("qualitative") or []
            quantitative = parsed.get("quantitative") or []

            if not isinstance(qualitative, list):
                qualitative = [qualitative]
            if not isinstance(quantitative, list):
                quantitative = [quantitative]

            core = {"qualitative": qualitative, "quantitative": quantitative}

            # 6. 写进 extra 里，PaperNode.todic 会一并保存
            if not isinstance(paper.extra, dict):
                paper.extra = {}
            paper.extra["core_knowledge"] = core

            self.logger.info(
                f"🧠 [CoreKnow] 论文: {paper.title} | 定性条目: {len(qualitative)} | 定量条目: {len(quantitative)}"
            )
            return core   
    
    def extract_core_knowledge_for_papers(self, papers, max_papers=10):
        """
        对一批论文抽核心知识，默认只处理前 max_papers 篇以控制开销。
        """
        if not papers:
            return

        count = 0
        for paper in papers:
            if count >= max_papers:
                break
            try:
                self.extract_core_knowledge_for_paper(paper)
                count += 1
            except Exception as e:
                self.logger.exception(f"❌ [CoreKnow] 处理论文失败: {paper.title}, error: {e}")    


#########################################


    def expand_iteration(self, depth):
        """执行一次扩展迭代 - 已弃用，保留用于向后兼容"""
        self.logger.info(f"\033[93mWarning: expand_iteration is deprecated, use process() instead\033[0m")
        # 保留空实现以避免旧代码出错
        pass
    
    def save_paper_nodes(self, paper_nodes):
        paper_list = []
        for paper_node in paper_nodes:
            paper_list.append(paper_node.todic())
        return paper_list
    
    def read_paper_nodes(self, paper_list):
        paper_nodes = []
        for paper_dic in paper_list:
            paper_node = PaperNode({
                "title": paper_dic["title"],
                "depth": paper_dic["depth"],
                "arxiv_id": paper_dic["arxiv_id"],
                "abstract": paper_dic["abstract"],
                "sections": paper_dic.get("sections", ""),
                "content_sections": paper_dic.get("content_sections", ""),
                "source": paper_dic.get("source", ""),
                "select_score": paper_dic.get("select_score", -1),
                "extra": paper_dic.get("extra", {})})
            paper_nodes.append(paper_node)
        return paper_nodes
    
    
    def top_rank(self):
        """一次性看最好的分数一致的paper，根据论文摘要和查询计算进行排序，赋值相对分数"""
        papers = []
        for i, paper in enumerate(self.papers_queue):
            if i >= 10:
                break
            if paper.select_score > 5.8:
                papers.append(paper)
            else:
                break
        if len(papers) <= 1:
            return
        logger.info(f"🔍 [TopRanker] 重排序前{len(papers)}篇论文")
        sorted_papers = sorted(papers,key=lambda p: p.select_score, reverse=True)
        logger.info(f"origin rank:{[paper.title for paper in sorted_papers]}")
        template = self.prompts["ranker"]["top_ranker"]
        paper_info = {}
        for idx, paper in enumerate(papers):
            paper_info[f"{idx}"] = {"title": paper.title, "abstract": paper.abstract,"authors":paper.authors,"publication_date":paper.year,"citations":paper.citations,"journal":paper.journal,"CCF":paper.ccf}
        prompt = template.format(paper_info=json.dumps(paper_info), user_query=self.user_query)
        response = self.top_ranker.infer(prompt,use_cot=False)
        try:
            json_start = response.find('{')
            json_end = response.rfind('}')
            result = json.loads(response[json_start:json_end+1])
            order = result["order"]
            order_list = [int(i) for i in order.split(",")]
            for idx, paper_id in enumerate(order_list):
                papers[paper_id].select_score += 1/len(papers)*(len(papers)-idx)
            sorted_papers = sorted(papers,key=lambda p: p.select_score, reverse=True)
            logger.info(f"new rank:{[paper.title for paper in sorted_papers]}\nresponse:{response}")
        except Exception as e:
            logger.exception(f"❌ [TopRanker] 论文排序失败: {e},response:{response}")
    
    def process(self):
        """正确的处理流程：Keyminder → DirectSearch → [Locater → Expand] × N轮 → Ranker"""
        start_time = time.time()
        if FREEZE_MODE2 == False:
            if FREEZE_MODE == False:
                self.logger.info(f"\n🚀 =============== [PROCESS] 开始论文搜索流程 ===============")
                self.logger.info(f"🔍 [PROCESS] 用户查询: {self.user_query}")
                self.logger.info(f"📅 [PROCESS] 截止日期: {self.end_date}")
                self.logger.info(f"🔢 [PROCESS] 扩展层数: {self.expand_layers}")
                self.logger.info(f"🧵 [PROCESS] 线程数量: {self.threads_num}")
                begin = time.time()
                self.user_query = self.translate(self.user_query)
                self.search_paper_worker([self.user_query],3)
                end = time.time()
                self.logger.info(f"Init search by google,用时: {end - begin}s,检索query:{self.user_query},检索文献：{[paper.title for paper in self.papers_queue]}")
                
                # Step 1: 提取关键词、权重和查询
                self.logger.info(f"\n🎯 =============== Step 1: Keyminder 阶段 ===============")
                begin = time.time()
                queries = self.extract_keywords_and_queries()
                if not queries:
                    error_msg = "无法从查询中提取有效的查询语句。"
                    self.logger.info(f"❌ [PROCESS] {error_msg}")
                    self.root.output = error_msg
                    return self.root
                end = time.time()
                self.logger.info(f"🔍 \033[92m [Keyminder] 理解用户问题生成查询query用时: {end-begin:.2f} \033[0m 秒")
                # Step 2: 直接搜索初始论文 (depth=0)
                self.logger.info(f"\n🔍 =============== Step 2: DirectSearch 阶段 ===============")
                begin = time.time()
                self.search_papers_with_queries(queries)
                
                if not self.papers_queue:
                    error_msg = "没有找到相关论文。"
                    self.logger.info(f"❌ [PROCESS] {error_msg}")
                    self.root.output = error_msg
                    return self.root
                end = time.time()
                self.logger.info(f"🔍 \033[92m [DirectSearch] 直接搜索用时: {end-begin:.2f} \033[0m 秒")
                
                self.logger.info(f"📊 [PROCESS] 初始论文队列大小: {len(self.papers_queue)}")
                
                # Step 3: 分层扩展循环（严格按深度进行）
                self.logger.info(f"\n🔄 =============== Step 3: 分层扩展阶段 ===============")
                self.logger.info(f"🔢 [PROCESS] 计划迭代轮数: {self.expand_layers}")
                for current_depth in range(self.expand_layers):
                    self.logger.info(f"\n🔄 --------------- 第 {current_depth+1}/{self.expand_layers} 轮迭代 (处理深度 {current_depth}) ---------------")
                    
                    # 检查是否达到限制条件
                    if self.check_limit():
                        self.logger.info(f"⏰ [PROCESS] 达到限制条件，停止扩展,搜到文献：{len(self.papers_queue)}")
                        break
                    # 3.1 获取当前深度的论文（严格按深度筛选）
                    logger.info(f"\n📖 [PROCESS] 开始筛选可扩展文献...")
                    begin = time.time()
                    papers = [p for p in self.papers_queue if p.depth == current_depth]
                    begin = time.time()
                    current_depth_papers = self.rank_papers(papers,current_depth,filter=True)
                    end = time.time()
                    self.logger.info(f"🔍 \033[92m [SELECTOR] 筛选可扩展文献用时: {end-begin:.2f} \033[0m 秒")
                    high_score_papers = [p for p in self.papers_queue if p.select_score > 4]
                    hight_score_papers = sorted(high_score_papers, key=lambda p: p.select_score, reverse=True)
                    title_list = [p.title for p in hight_score_papers]
                    self.logger.info(f"📊 \033[093m 第{current_depth+1}批返回结果：时间：{time.time()-start_time}s,论文数：{len(title_list)}\n\033[0m,{title_list}")
                    end = time.time()
                    self.logger.info(f"[SELECTOR] 过滤论文耗时：{end -begin}")
                    if not current_depth_papers:
                        self.logger.info(f"📭 [PROCESS] 深度 {current_depth} 没有论文需要处理，停止迭代")
                        break
                        
                    self.logger.info(f"📊 [PROCESS] 深度 {current_depth} 的论文数量: {len(current_depth_papers)}")
                    for i, paper in enumerate(current_depth_papers): 
                        self.logger.info(f"  📄 {i+1}. {paper.title[:60]}... (depth={paper.depth})")
                    
                    # 3.2 获取论文详细内容
                    self.logger.info(f"\n📖 [PROCESS] 3.2 获取深度 {current_depth} 论文的详细内容...")
                    begin = time.time()
                    
                    papers_with_content = self.get_paper_ref(current_depth_papers)
                    end = time.time()
                    self.logger.info(f"📖 \033[92m [FetchContent] 获取论文内容用时: {end-begin:.2f} \033[0m 秒")
                    self.logger.info(f"📊 [PROCESS] 成功获取内容的论文内容: {papers_with_content}")
                    
                    # 3.3 Locater: 定位相关章节并扩展到下一深度
                    self.logger.info(f"\n🎯 [PROCESS] 3.3 定位相关章节并扩展到深度 {current_depth + 1}...")
                    old_queue_size = len(self.papers_queue)
                    
                    # 反思检索+ref定位扩展
                    begin = time.time()
                    with ThreadPoolExecutor() as executor:
                        future1 = executor.submit(self.further_search, papers_with_content, target_depth=current_depth + 1)
                        future2 = executor.submit(self.locate_relevant_papers, papers_with_content, target_depth=current_depth + 1)
                        result1 = future1.result()
                        result2 = future2.result()
                    end = time.time()
                    logger.info(f"🎯 \033[92m [FurtherSearch+Locater] 反思检索+ref定位扩展用时: {end-begin:.2f} \033[0m 秒")
                    new_papers_count = len(self.papers_queue) - old_queue_size
                    self.logger.info(f"📊 [PROCESS] 从深度 {current_depth} 扩展到深度 {current_depth + 1}: {new_papers_count} 篇新论文")
                    self.logger.info(f"📊 [PROCESS] 总论文队列大小: {len(self.papers_queue)}")
                    # 显示当前各深度的论文分布
                    # 打印进度摘要
                    self.print_progress_summary(current_depth + 1)
                    
                    # 如果没有扩展出新论文，停止迭代
                    if new_papers_count == 0:
                        self.logger.info(f"📭 [PROCESS] 深度 {current_depth} 没有扩展出新论文，停止迭代")
                        break
                
                # Step 4: 最终统一评分
                self.logger.info(f"\n⭐ =============== Step 4: 最终统一评分阶段 ===============")
                self.logger.info(f"📊 [PROCESS] 对所有 {len(self.papers_queue)} 篇论文进行统一评分...")
                all_papers = list(self.papers_queue)
                begin = time.time()
                file_path = f"{self.output_dir}/record"
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_path = os.path.join(file_path, f"{self.idx}.json")
                save_papers = self.save_paper_nodes(all_papers)
                data = {
                    "all_papers": save_papers,
                }
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                self.logger.info(f"\n🚀 \033[93m=============== Freeze 模式，直接读取记录文件,开始reranker \033[0m ===============")
                self.logger.info(f"🔍 [PROCESS] 用户查询: {self.user_query}")
                begin = time.time()
                file_path = f"{FREEZE_PATH}/record/{self.idx}.json"
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_papers = self.read_paper_nodes(data["all_papers"])
                for paper in all_papers:
                    self.papers_queue.append(paper)
            
            begin = time.time()
            ranked_papers = self.rank_papers(all_papers,self.expand_layers)
            end = time.time()
            self.logger.info(f"🔍 \033[92m [Ranker] 统一评分用时: {end-begin:.2f} \033[0m 秒")
            self.logger.info(f"\033[92m [Ranker] 统一评分完成，开始重排top20 同分的paper \033[0m")
        else:
            self.logger.info(f"\n🚀 \033[93m=============== Freeze 模式，直接读取记录文件,开始top ranker \033[0m ===============")
            self.logger.info(f"🔍 [PROCESS] 用户查询: {self.user_query}")
            path = f"{FREEZE_PATH}/{self.idx}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)["extra"]["all_papers"]
            self.papers_queue = self.read_paper_nodes(data)
        self.papers_queue = sorted(self.papers_queue, key=lambda p: p.select_score, reverse=True)
        begin = time.time()
        self.top_rank()
            
        end = time.time()
        self.logger.info(f"🔍 \033[92m [Ranker] 重排paper用时: {end-begin:.2f} 秒\033[0m ")
        
        high_score_papers = [p for p in self.papers_queue if p.select_score > 4]
        final_selected_papers = sorted(high_score_papers, key=lambda p: p.select_score, reverse=True)

        #######################################
        try:
            self.logger.info(
                f"🧠 [CoreKnow] 开始为前 {min(len(final_selected_papers), 10)} 篇高分论文抽取核心知识..."
            )
            # 这里的 max_papers=10 是“做核心知识抽取”的数量，你可以改大/改小
            self.extract_core_knowledge_for_papers(final_selected_papers, max_papers=10)
        except Exception as e:
            self.logger.exception(f"❌ [CoreKnow] 抽取核心知识整体失败: {e}")
        ########################################

        # ==== 汇总各篇论文的 core_knowledge 到 root.extra["core_knowledge_list"] ====
        core_list = []
        for paper in final_selected_papers:
            ck = None
            if isinstance(paper.extra, dict):
                ck = paper.extra.get("core_knowledge", None)
            if not ck:
                continue

            q_list = ck.get("qualitative") or []
            t_list = ck.get("quantitative") or []

            # 定性条目
            for item in q_list:
                if not isinstance(item, dict):
                    continue
                core_list.append({
                    "type": "qualitative",
                    "paper_title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    **item
                })

            # 定量条目
            for item in t_list:
                if not isinstance(item, dict):
                    continue
                core_list.append({
                    "type": "quantitative",
                    "paper_title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    **item
                })

        self.root.extra["core_knowledge_list"] = core_list
        self.logger.info(f"🧠 [CoreKnow] 汇总核心知识条目数: {len(core_list)}")
        ########################################

        for paper in final_selected_papers:
            if paper.select_score > 4:
                paper_info = {
                    "title":paper.title,
                    "arxiv_id":paper.arxiv_id,
                    "link":f"https://arxiv.org/pdf/{paper.arxiv_id}",
                    "abstract":paper.abstract,
                    "sections":paper.sections,
                    "journal":paper.journal,
                    "authors":paper.authors,
                    "year":paper.year,
                    "h5":paper.h5,
                    "IF":paper.IF,
                    "CCF":paper.ccf,
                    "citations":paper.citations,
                    "score":paper.select_score,
                }
                self.root.extra["recall_papers"].append(paper_info)

        title_list = [p.title for p in final_selected_papers]
        self.logger.info(f"📊 \033[093m 第{self.expand_layers+1}批返回结果：时间：{time.time()-start_time}s,论文数：{len(title_list)}\n\033[0m,{title_list}")
        end = time.time()
        self.logger.info(f"\033[92m [Ranker] 统一评分用时: {end-begin:.2f} \033[0m 秒")
        
        # Step 5: 最终结果整理和输出
        self.logger.info(f"\n📋 =============== Step 5: 最终结果整理 ===============")
        
        # 保存所有搜索到的论文（按分数排序，不筛）
        all_papers_sorted = sorted(list(self.papers_queue), key=lambda p: p.select_score, reverse=True)
        self.root.extra["all_papers"] = [p.todic() for p in all_papers_sorted]
        # 保存所有搜索到的论文标题
        self.root.extra["all_papers_title"] = [p.title for p in all_papers_sorted]
        
        # 保存排序后的论文到sorted_papers字段
        self.root.extra["sorted_papers"] = [p.todic() for p in final_selected_papers]
        # 保存排序后的论文标题到final_selected_papers_title字段
        self.root.extra["final_selected_papers_title"] = [p.title for p in final_selected_papers]
        self.logger.info(f"💾 [PROCESS] 已保存 {len(all_papers_sorted)} 篇所有论文到all_papers字段")
        self.logger.info(f"💾 [PROCESS] 已保存 {len(all_papers_sorted)} 个所有论文标题到all_papers_title字段")
        self.logger.info(f"💾 [PROCESS] 已保存 {len(final_selected_papers)} 篇排序后的论文到sorted_papers字段")
        self.logger.info(f"💾 [PROCESS] 已保存 {len(final_selected_papers)} 个排序后的论文标题到final_selected_papers_title字段")
        
        self.logger.info(f"📊 [PROCESS] 最终统计:")
        self.logger.info(f"  总处理论文数: {len(self.papers_queue)}")
        self.logger.info(f"  高分论文数: {len(final_selected_papers)}")
        self.logger.info(f"  平均分数: {sum(p.select_score for p in final_selected_papers)/len(final_selected_papers):.3f}" if final_selected_papers else "  平均分数: N/A")
        # 按深度统计最终结果
        depth_result_stats = {}
        for paper in final_selected_papers:
            depth_result_stats[paper.depth] = depth_result_stats.get(paper.depth, 0) + 1
        self.logger.info(f"📊 [PROCESS] 最终结果深度分布: {depth_result_stats}")
        
        # 设置输出
        if final_selected_papers:
            output_lines = []
            self.logger.info(f"🏆 [PROCESS] 前10名高分论文:")
            for i, paper in enumerate(final_selected_papers[:10], 1):
                self.logger.info(f"  {i}. {paper.title[:50]}... (Score: {paper.select_score:.3f}, Depth: {paper.depth})")
                output_lines.append(f"{i}. {paper.title} (Score: {paper.select_score:.3f}, Depth: {paper.depth})")
                if hasattr(paper, 'abstract') and paper.abstract:
                    abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
                    output_lines.append(f"   Abstract: {abstract_preview}")
                output_lines.append("")
            
            self.root.output = "\n".join(output_lines)
        else:
            error_msg = "没有找到符合要求的相关论文。"
            self.logger.info(f"❌ [PROCESS] {error_msg}")
            self.root.output = error_msg
        
        total_time = time.time() - start_time
        self.logger.info(f"\n⏱️ [PROCESS] 总处理时间: {total_time:.2f}秒")
        self.logger.info(f"🎯 [PROCESS] 流程完成!")
        
        return self.root

    def run(self):
        """运行新的论文搜索流程 - 向后兼容方法，调用 process()"""
        self.logger.info("\033[92mStarting paper search with new architecture...\033[0m")
        self.logger.info("\033[93mUsing simplified process flow...\033[0m")
        
        # 直接调用新的简化流程
        return self.process()
