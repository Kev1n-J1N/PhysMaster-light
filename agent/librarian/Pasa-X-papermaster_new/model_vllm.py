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
import math
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
import threading,time

from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, wait_random
import openai

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

with open(f"{current_dir}/config/config.json", "r") as f:
    config = json.load(f)

RANKER_USE_STREAM = config['ranker_use_stream']
USE_STREAM = config['use_stream']

class Agent:
    def _call_llm(self, prompt, max_tokens=4096, temperature=0.0,
              top_p=1.0, logprobs=False, **kwargs):
        """
        统一的底层模型调用接口。

        - 对本地模型 (localhost / 127.0.0.1) 才真正请求 logprobs；
        - 对远程模型 (比如 DP 的 glm-4.6)，不传 logprobs 参数，避免 400；
        - 如果调用者传了 logprobs=True，但后端不支持，我们仍然返回 (text, dummy_prob) 结构，
        dummy_prob 取 1.0，这样 ranking 的 tie-break 不会被彻底废掉。
        """
        is_local = ("localhost" in self.url) or ("127.0.0.1" in self.url)
        # 只有本地模型才尝试打开 logprobs
        request_logprobs = bool(logprobs and is_local)

        messages = [{"role": "user", "content": prompt.strip()}]

        if "gpt-5" in self.model_name.lower():
            # GPT-5 系列用 max_completion_tokens
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "top_p": top_p,
            }
            if temperature != 0:
                api_params["temperature"] = temperature
            if request_logprobs:
                api_params["logprobs"] = True
                api_params["top_logprobs"] = 1
        else:
            # 其他模型用 max_tokens
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if request_logprobs:
                api_params["logprobs"] = True
                if RANKER_USE_STREAM:
                    api_params["stream"] = True
            else:
                if USE_STREAM:
                    api_params["stream"] = True

        # 允许上层额外塞参数（一般不会用到）
        api_params.update(kwargs)

        probs = 0.0
        text = ""

        @retry(
            retry=retry_if_exception_type(openai.RateLimitError),
            wait=wait_random(0.5, 16),
            stop=stop_after_attempt(20),
            reraise=True,
        )
        def _call_llm_with_retry():
            t0 = time.time()
            resp = self.client.chat.completions.create(**api_params)
            logger.info(
                f"llm time cost: {time.time() - t0:.3f}s, "
                f"model: {self.model_name}, prompt: {prompt[:80]}"
            )
            return resp

        try:
            response = _call_llm_with_retry()

            # 三种情况：本地流式、本地非流式、不请求 logprobs
            if request_logprobs and RANKER_USE_STREAM:
                final_text = ""
                flag = False
                for event in response:
                    choice = event.choices[0]
                    lp = choice.logprobs
                    if lp and lp.content:
                        tok0 = lp.content[0]
                        if (not flag) and tok0.token.isdigit():
                            probs = math.exp(tok0.logprob)
                            flag = True
                    delta = choice.delta.content or ""
                    final_text += delta
                text = final_text
            elif request_logprobs:
                choice = response.choices[0]
                lp = choice.logprobs
                if lp and lp.content:
                    for t in lp.content:
                        if t.token.isdigit():
                            probs = math.exp(t.logprob)
                            break
                text = choice.message.content or ""
            else:
                # 没有向后端请求 logprobs（远程模型）
                choice = response.choices[0]
                text = choice.message.content or ""
                if logprobs:
                    # 上层希望有一个概率参与排序，这里给一个 dummy 值，
                    # 这样 select_score = score + 1.0，本质上仍然按 score 排序。
                    probs = 1.0

        except openai.RateLimitError as e:
            logger.exception(f"❌ -----> 模型调用失败, rate limit error: {e}")
            raise
        except Exception as e:
            logger.exception("❌ 模型调用失败")
            if is_local:
                logger.info(f"❌ 本地模型调用失败: {e}, url: {self.url}")
                logger.info("💡 请检查本地模型进程 / 端口 / 日志等")
            else:
                logger.info(f"❌ 远程模型调用失败: {e}, model: {self.model_name}")
            raise

        # 统一约定：logprobs=True 时返回 (text, prob)，否则返回原始 response。
        if logprobs:
            return text, probs
        else:
            return response


    def _call_llm_with_cot(self, prompt, max_tokens=4096, temperature=0.0, top_p=1.0,logprobs=False, **kwargs):
        """
        带思维链的模型调用接口，让模型先思考再给出答案
        输出格式：<think>思考过程...</think><answer>最终答案...</answer>
        Args:
            prompt: 输入prompt字符串
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top_p采样
            kwargs: 其他openai参数
        Returns:
            解析后的答案字符串（只返回<answer>标签内的内容）
        """
        # 构建system prompt，要求模型输出思维链
        system_prompt = """You are an intelligent assistant. Please answer using the following format:

1. First, think step by step inside the <think> tag: analyze the problem and your reasoning process.
2. Then, provide the final answer inside the <answer> tag.

If the user requests a specific format (e.g., JSON), output it strictly inside the <answer> tag in that format.

Example output format:
<think>
Let me analyze this problem...
First, I should consider...
Then...
</think>
<answer>
Here is the final answer, following the user's requested format.
</answer>"""

        # 构建完整的消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()}
        ]
        
        # logger.info(f"🤔 正在调用带思维链的模型...")
        # logger.info(f"📝 用户提示: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        try:
            # 调用模型
            # 根据模型类型选择正确的参数名
            #if "gpt-5" in self.model_name.lower():
                # GPT-5模型使用max_completion_tokens，且不支持temperature=0
            #    api_params = {
            ##        "max_completion_tokens": max_tokens,
            #        "top_p": top_p,
            #       logprobs: logprobs,
            #       **kwargs
            #    }
                # 只有当temperature不为0时才添加temperature参数
            #    if temperature != 0:
            #        api_params["temperature"] = temperature
            #else:
            #    # 其他模型使用max_tokens
            #    api_params = {
            #       "messages": messages,
            #        "max_tokens": max_tokens,
            #        "temperature": temperature,
            #        "top_p": top_p,
            #        "logprobs": logprobs,
            #       **kwargs

               # }
            
                # 调用模型
        # 根据模型类型选择正确的参数名
            if "gpt-5" in self.model_name.lower():
                # GPT-5 模型使用 max_completion_tokens
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "top_p": top_p,
                    **kwargs,
                }
                # 只有当 temperature 不为 0 时才添加
                if temperature != 0:
                    api_params["temperature"] = temperature
            else:
                # 其他模型使用 max_tokens
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    **kwargs,
                }


            @retry(
                retry=retry_if_exception_type(openai.RateLimitError),
                wait=wait_random(0.5, 16),
                stop=stop_after_attempt(20),
                reraise=True
            )
            def _call_llm_with_retry():
                t0 = time.time()
                response = self.client.chat.completions.create(**api_params)
                logger.info(f"llm time cost: {time.time() - t0:.3f}s, prompt: {prompt[:100]}")
                return response
            response = _call_llm_with_retry()
            
            # 获取模型原始返回结果
            raw_response = response.choices[0].message.content
            # logger.info(f"🔍 模型原始返回结果:")
            # logger.info(f"   {raw_response}")
            
            # 解析<answer>标签内的内容
            answer = self._extract_answer_from_cot(raw_response)
            
            if answer:
                logger.info(f"✅ 答案解析成功")
                # logger.info(f"   {answer}")
                # 将解析后的答案重新赋值给response对象，保持接口一致性
                response.choices[0].message.content = answer
                return response
            else:
                logger.info(f"⚠️  无法解析答案，返回原始结果")
                return response
                
        except Exception as e:
            logger.exception(f"❌ 思维链模型调用失败")
            # 如果失败，回退到普通调用
            logger.info(f"🔄 回退到普通模型调用...")
            try:
                response = self._call_llm(prompt, max_tokens, temperature, top_p, **kwargs)
                return response.choices[0].message.content
            except Exception as fallback_e:
                logger.exception(f"❌ 回退调用也失败")
                return ""
    
    def _extract_answer_from_cot(self, response_text):
        """
        从思维链响应中提取<answer>标签内的内容
        Args:
            response_text: 模型返回的完整文本
        Returns:
            提取的答案字符串，如果没有找到则返回None
        """
        try:
            # 查找<answer>标签
            start_tag = "<answer>"
            end_tag = "</answer>"
            
            start_idx = response_text.find(start_tag)
            if start_idx == -1:
                logger.info(f"⚠️  未找到开始标签 <answer>")
                return None
                
            start_idx += len(start_tag)
            end_idx = response_text.find(end_tag, start_idx)
            
            if end_idx == -1:
                logger.info(f"⚠️  未找到结束标签 </answer>")
                return None
                
            # 提取答案内容
            answer = response_text[start_idx:end_idx].strip()
            
            if not answer:
                logger.info(f"⚠️  <answer>标签内内容为空")
                return None
                
            return answer
            
        except Exception as e:
            logger.exception(f"❌ 解析答案时出错")
            return None
    
    def __init__(self, model_name=None, url=None, key="EMPTY", config_path="config/config.json"):
        """
        Agent初始化
        Args:
            model_name: 模型名称，如果提供则从配置文件读取URL和key
            url: 模型URL，如果提供model_name则忽略此参数
            key: API密钥，如果提供model_name则忽略此参数
            config_path: 配置文件路径
        """
        # 如果提供了model_name，从配置文件读取配置
        if model_name and not url:
            config = self._load_model_config(config_path)
            model_configs = config.get("model_config", {})
            if model_name not in model_configs:
                available_models = ", ".join(model_configs.keys())
                raise ValueError(f"模型 '{model_name}' 不在配置文件中。可用模型: {available_models}")
            
            model_config = model_configs[model_name]
            self.model_name = model_name
            self.url = model_config.get("base_url")
            self.key = model_config.get("api_key", "EMPTY")
            self.model_config = model_config  # 保存完整配置
            
            if not self.url:
                raise ValueError(f"模型 '{model_name}' 的配置缺少 'base_url' 字段")
            
            logger.info(f"✅ 从配置文件加载模型: {model_name}")
            logger.info(f"   Base URL: {self.url}")
            logger.info(f"   API Key: {'***' + self.key[-4:] if len(self.key) > 4 else '***'}")
            
            # 测试本地模型连接
            if "localhost" in self.url or "127.0.0.1" in self.url:
                self._test_local_model_connection()
        
        # 如果没有提供model_name，使用传统方式（向后兼容）
        else:
            self.model_name = model_name if model_name else "unknown"
            self.url = url
            self.key = key
            self.model_config = None
        
        # 创建OpenAI客户端
        self.client = openai.OpenAI(base_url=self.url, api_key=self.key)
        
        # 初始化统计变量
        self.call_count = 0
        self.total_tokens = 0
        self._usage_lock = threading.Lock()
    
    def _test_local_model_connection(self):
        """测试本地模型连接是否正常"""
        import requests
        import time
        
        logger.info(f"🔍 正在测试本地模型连接: {self.url}")
        
        # 构建测试请求
        test_url = self.url.replace("/v1", "/v1/models")
        test_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        try:
            # 首先测试模型列表接口
            logger.info(f"   📋 测试模型列表接口...")
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"   ✅ 模型列表接口正常")
            else:
                logger.info(f"   ⚠️  模型列表接口返回状态码: {response.status_code}")
            
            # 测试聊天接口
            logger.info(f"   💬 测试聊天接口...")
            chat_url = self.url.replace("/v1", "/v1/chat/completions")
            response = requests.post(chat_url, json=test_data, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"   ✅ 本地模型连接测试成功！")
                logger.info(f"   🚀 模型 {self.model_name} 已准备就绪")
            else:
                logger.info(f"   ⚠️  聊天接口返回状态码: {response.status_code}")
                logger.info(f"   📝 响应内容: {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            logger.exception(f"   ❌ 连接失败: 无法连接到 {self.url}")
            logger.info(f"   💡 请检查:")
            logger.info(f"      1. 模型服务是否已启动")
            logger.info(f"      2. 端口是否正确")
            logger.info(f"      3. 防火墙设置")
            logger.info(f"      4. 使用命令: netstat -tlnp | grep {self.url.split(':')[-1].split('/')[0]}")
        except requests.exceptions.Timeout:
            logger.exception(f"   ⏰ 连接超时: 模型响应时间过长")
        except Exception as e:
            logger.exception(f"   ❌ 连接测试失败: {e}")
        
        logger.info(f"   🔍 本地模型连接测试完成")
    
    def _load_model_config(self, config_path):
        """加载模型配置文件"""
        # 处理相对路径
        if not os.path.isabs(config_path):
            config_path = os.path.join(current_dir, config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"模型配置文件未找到: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"模型配置文件格式错误: {e}")
    
    
    def infer(self, prompt, sample=False, use_cot=False, logprobs=False):
        try:
            caller = self._call_llm_with_cot if use_cot else self._call_llm
            
            response = caller(
                prompt,
                max_tokens=4096,
                temperature=2.0 if sample else 0.0,
                top_p=0.8 if sample else 1.0,
                logprobs=logprobs
            )
            return response.choices[0].message.content if hasattr(response, 'choices') else response
        except Exception as e:
            logger.exception(str(e))
            return ""
    
    def batch_infer(self, prompts, batch_size=20, sample=False, use_cot=False, logprobs=False):
        if len(prompts) == 0:
            return []
        def get_response(prompt):
            caller = self._call_llm_with_cot if use_cot else self._call_llm
            response = caller(
                prompt,
                max_tokens=4096,
                temperature=2.0 if sample else 0.0,
                top_p=0.8 if sample else 1.0,
                logprobs=logprobs
            )
            return response.choices[0].message.content if hasattr(response, 'choices') else response
        try:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                responses = list(executor.map(get_response, prompts))
            return responses
        except Exception as e:
            logger.exception(str(e))
            return []
    
    def batch_infer_safe(self, prompts, batch_size=20, sample=False, use_cot=False,logprobs=False):
        """
        安全的批量推理方法，确保返回结果的顺序与输入prompts的顺序完全一致
        
        Args:
            prompts: 输入的prompt列表
            batch_size: 批处理大小
            sample: 是否使用采样模式
            use_cot: 是否启用思维链调用
            
        Returns:
            responses: 按顺序返回的响应列表
        """
        if len(prompts) == 0:
            return []
        
        logger.info(f"🔒 [SafeBatchInfer] 开始安全批量推理，共 {len(prompts)} 个prompt")
        logger.info(f"🔒 [SafeBatchInfer] 批处理大小: {batch_size}")
        
        # 为每个prompt添加索引标识
        indexed_prompts = [(i, prompt) for i, prompt in enumerate(prompts)]
        
        def get_response_with_index(indexed_item):
            index, prompt = indexed_item
            try:
                caller = self._call_llm_with_cot if use_cot else self._call_llm
                response = caller(
                    prompt,
                    max_tokens=4096,
                    temperature=2.0 if sample else 0.0,
                    top_p=0.8 if sample else 1.0,
                    logprobs=logprobs
                )
                if logprobs:
                    return index, (response[0],response[1])
                return index, (response.choices[0].message.content if hasattr(response, 'choices') else response)
            except Exception as e:
                logger.exception(f"❌ [SafeBatchInfer] prompt {index+1} 处理失败: {e}")
                if logprobs:
                    return index, ("",0)
                return index, ""
        
        try:
            # 使用ThreadPoolExecutor并发处理，但保持索引信息
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                indexed_responses = list(executor.map(get_response_with_index, indexed_prompts))
            
            # 按原始索引重新排序，确保顺序正确
            indexed_responses.sort(key=lambda x: x[0])
            responses = [response for index, response in indexed_responses]
            
            logger.info(f"✅ [SafeBatchInfer] 安全批量推理完成，返回 {len(responses)} 个响应")
            logger.info(f"🔒 [SafeBatchInfer] 已验证响应顺序与输入顺序一致")
            
            return responses
            
        except Exception as e:
            logger.exception(f"❌ [SafeBatchInfer] 批量推理失败: {e}")
            return []
            
    def get_usage_stats(self):
        avg_token = self.total_tokens / self.call_count if self.call_count > 0 else 0
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'avg_token': avg_token
        }
    
if __name__ == "__main__":
    print("🧪 测试Agent模型调用功能")
    
    # 测试使用模型名称创建Agent
    try:
        print("\n=== 测试1: 使用模型名称创建Agent ===")
        agent = Agent(model_name="glm")
        print(f"✅ Agent创建成功: {agent.model_name}")
        print(f"📊 URL: {agent.url}")
        user_query = "Give me papers which show that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets."
        title = "How to Train Data-Efficient LLMs"
        abstract = "  The training of large language models (LLMs) is expensive. In this paper, we\nstudy data-efficient approaches for pre-training LLMs, i.e., techniques that\naim to optimize the Pareto frontier of model quality and training resource/data\nconsumption. We seek to understand the tradeoffs associated with data selection\nroutines based on (i) expensive-to-compute data-quality estimates, and (ii)\nmaximization of coverage and diversity-based measures in the feature space. Our\nfirst technique, Ask-LLM, leverages the zero-shot reasoning capabilities of\ninstruction-tuned LLMs to directly assess the quality of a training example. To\ntarget coverage, we propose Density sampling, which models the data\ndistribution to select a diverse sample. In our comparison of 19 samplers,\ninvolving hundreds of evaluation tasks and pre-training runs, we find that\nAsk-LLM and Density are the best methods in their respective categories.\nCoverage sampling can recover the performance of the full data, while models\ntrained on Ask-LLM data consistently outperform full-data training -- even when\nwe reject 90% of the original dataset, while converging up to 70% faster.\n"
        # 测试简单问题
        simple_questions = [
            f"Evaluate the relevance between the provided paper and the overall user query. Be a nitpicky critic: do not just check if they are related—actively find mismatches and gaps. If the query has multiple constraints, extract them and test each one.\n\nPaper Title: {title}\nPaper Abstract: {abstract}\n\nUser Query Context: {user_query}\n\nLanguage Rule:\n- Only the labels “Score:” and “Reasoning:” may appear in English.\n- From “Reasoning:” onward, produce output exclusively in Simplified Chinese. Do not output any English words, letters, acronyms, or headings after “Reasoning:”. Translate all quoted evidence and section headings into Chinese as well.\n\nInstructions:\n1) Extract constraints from the query as C1, C2, … (e.g., task, domain, dataset, method, objective, population, language, timeframe, metrics, output format, exclusions like \"exclude X\").\n2) For each constraint Ci, judge Status ∈ {{Satisfied, Partially, Not satisfied, Unknown}}. Provide brief evidence from the Title/Abstract and from the Query (short quotes are fine). Treat missing/ambiguous evidence as Unknown (do not assume).\n3) List irrelevancies / red flags (divergent focus, wrong task/domain/dataset, missing constraints).\n4) Scoring rule (strict): 5 = The status of all critical constraints are Satisfied; 4 = The status of majority (>50%) of critical constraints Satisfied; 3 = The status of some critical constraints are Satisfied, but not a majority; 2 = no critical constraints Satisfied, but the paper is in the same domain; 1 = completely unrelated (domain mismatch).\n5) Keep Reasoning concise but specific (≈200 words). Quote only short snippets.\n\nRequired Output Format:\nScore: 1/2/3/4/5\nReasoning: （此处及其后全部用中文，≤100词，概述关键匹配与不匹配原因）\n约束项检查：\n- C1：<名称> — 状态：<满足/部分/不满足/未知>。证据：“<论文>” 对比 “<查询>”\n- C2：...\n不相关性 / 风险点 / 缺口：\n- <要点1>\n- <要点2>\n\nNotes:\n- Use only the provided Title/Abstract and Query for evidence.\n- Prefer precision over length."
        ]
        
        print("\n=== 测试2: 简单问题回答 ===")
        for i, question in enumerate(simple_questions, 1):
            print(f"\n问题{i}: {question}")
            try:
                answer = agent.batch_infer_safe([question],logprobs=True)
                print(f"回答: {answer}")
            except Exception as e:
                print(f"❌ 回答失败: {e}")
        
        print("\n=== 测试3: 批量推理 ===")
        # try:
        #     answers = agent.batch_infer(simple_questions[:2], batch_size=2)
        #     for q, a in zip(simple_questions[:2], answers):
        #         print(f"Q: {q}")
        #         print(f"A: {a[:80]}..." if len(a) > 80 else f"A: {a}")
        #         print()
        # except Exception as e:
        #     print(f"❌ 批量推理失败: {e}")
        
        # # 显示使用统计
        # stats = agent.get_usage_stats()
        # print(f"\n📊 调用统计: 总调用{stats['call_count']}次, 总tokens{stats['total_tokens']}, 平均{stats['avg_token']:.1f}tokens/次")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    