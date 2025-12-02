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
"""
Please note that:
1. You need to first apply for a Google Search API key at https://serpapi.com/,
   and replace the 'your google keys' in utils.py before you can use it.
2. The service for searching arxiv and obtaining paper contents is relatively simple. 
   If there are any bugs or improvement suggestions, you can submit pull requests.
   We would greatly appreciate and look forward to your contributions!!
"""
import argparse
import json
import os
import time
import sys
import shutil
from datetime import datetime


from model_vllm import Agent
from paper_agent import PaperAgent
from search_utils.call_mtr import CallMetricsTracker
current_dir = os.path.dirname(os.path.abspath(__file__))
from log import setup_logging


setup_logging()
class TeeOutput:
    """日志类，读取终端输出保存为日志文件，使用4KB缓冲区优化性能"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8', buffering=4096)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


#########################################
import subprocess
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def auto_transfer():
    transfer_path = os.path.join(current_dir, "inputs&transfer", "transfer.py")
    templete_path = os.path.join(current_dir, "inputs&transfer", "templete.json")
    output_path = os.path.join(current_dir, "inputs&transfer", "my_query.jsonl")

    if not os.path.exists(transfer_path):
        print("❌ transfer.py 不存在！")
        return
    
    if not os.path.exists(templete_path):
        print("❌ templete.json 不存在！")
        return

    print("⚙️ [run] 检测到 templete.json，开始自动运行 transfer.py ...\n")
    cmd = [
        "python", transfer_path,
        "--input_file", templete_path,
        "--output_file", output_path,
        "--published_time", "20250101"
    ]

    subprocess.run(cmd)
    print("\n🎉 [run] transfer.py 处理完毕 → 已生成 my_query.jsonl")
#########################################
auto_transfer()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file',     type=str, default="/home/ma-user/modelarts/work/TianJin/papermaster/PASAX-jt/Pasa-X/data/APAS_bench/questions_with_4_queries.jsonl")
    # parser.add_argument('--dataset_name',   type=str, default="APAS_bench")

    #parser.add_argument('--input_file',     type=str, default="/data/duyuwen/Pasa-X/data/pasamaster_bench.jsonl")

    parser.add_argument('--input_file',     type=str, default="inputs&transfer/my_query.jsonl")
    parser.add_argument('--dataset_name',   type=str, default="pasamaster-bench")
    parser.add_argument('--keyminder_model', type=str, default="qwen3-30b", help="Keyminder使用的模型名称")
    parser.add_argument('--ranker_model',   type=str, default="qwen3-30b", help="Ranker使用的模型名称")
    parser.add_argument('--top_rank_model',    type=str, default="reason_rank", help="Top Ranker使用的模型名称")
    parser.add_argument('--config_path',     type=str, default="config/config.json", help="配置文件路径")
    parser.add_argument('--output_folder',  type=str, default="", help="输出文件夹路径，如果为空则自动生成")
    parser.add_argument('--expand_layers',  type=int, default=2)
    parser.add_argument('--search_queries', type=int, default=5)
    parser.add_argument('--search_papers',  type=int, default=10)
    parser.add_argument('--expand_papers',  type=int, default=20)
    parser.add_argument('--threads_num',    type=int, default=100)
    parser.add_argument('--max_num',   type=int, default=-1)
    parser.add_argument('--time_limit', type=int, default=-1)
    parser.add_argument('--max_queries', type=int, default=10, help="最大处理查询数量，-1表示处理所有数据")
    parser.add_argument('--further_sample_limit', type=int, default=80, help="further_search抽样论文数量上限")
    # 仅保留布尔字符串参数形式：--no_proxy true/false（true=不用代理，false=使用代理）
    parser.add_argument('--no_proxy', type=str, default="true", help="禁用代理设置（true/false），true=不用代理，false=使用代理")
    args = parser.parse_args()

    # 统一解析 no_proxy 参数
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        val = str(value).strip().lower()
        if val in ('true', '1', 'yes', 'y', 't'):
            return True
        if val in ('false', '0', 'no', 'n', 'f'):
            return False
        raise ValueError(f"无效的布尔值: {value}，请使用 true/false")

    # True=不用代理(False=使用代理)
    effective_no_proxy = _parse_bool(args.no_proxy)
    if effective_no_proxy is None:
        effective_no_proxy = False

    # 从配置读取代理并设置环境变量
    if effective_no_proxy:
        print(f"🚫 已禁用代理设置，清除所有代理环境变量")
        # 直接unset所有代理环境变量
        os.system('unset http_proxy')
        os.system('unset https_proxy') 
        os.system('unset HTTP_PROXY')
        os.system('unset HTTPS_PROXY')
        # 同时从Python环境中删除
        for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if proxy_var in os.environ:
                del os.environ[proxy_var]
        print(f"✅ 所有代理环境变量已清除")
    else:
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.json")
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            common_config = loaded_config.get("common_config", {})
            proxy_cfg = common_config.get("proxy", {})
            http_proxy = proxy_cfg.get("http_proxy", "")
            https_proxy = proxy_cfg.get("https_proxy", http_proxy)
            if http_proxy != "" and https_proxy == "":
                os.environ['http_proxy'] = http_proxy
                os.environ['https_proxy'] = https_proxy
                print(f"🌐 已设置代理: http_proxy={os.environ['http_proxy']} https_proxy={os.environ['https_proxy']}")
            else:
                print(f"设置代理失败：http_proxy={http_proxy}, https_proxy = http_proxy")
        except Exception as proxy_e:
            print(f"⚠️ 设置代理失败，原因: {proxy_e}")

    # 模型配置处理
    print(f"\n🔧 模型配置处理...")
    print(f"📄 使用配置文件: {args.config_path}")
    
    # 直接使用模型名称创建Agent
    try:
        print(f"\n🚀 创建Agent实例...")
        keyminder = Agent(model_name=args.keyminder_model, config_path=args.config_path)
        top_ranker = Agent(model_name=args.top_rank_model, config_path=args.config_path)
        ranker = Agent(model_name=args.ranker_model, config_path=args.config_path)
        print(f"✅ 所有Agent创建成功!")
    except Exception as e:
        print(f"❌ 创建Agent失败: {e}")
        sys.exit(1)
    
    print(f"\n📊 Agent配置摘要:")
    print(f"  Keyminder: {keyminder.model_name} @ {keyminder.url}")
    print(f"  TopRanker:   {top_ranker.model_name} @ {top_ranker.url}")
    print(f"  Ranker:    {ranker.model_name} @ {ranker.url}")
    
    # 自动生成输出目录名称
    #if not args.output_folder:
    #    from datetime import datetime
    #    current_date = datetime.now().strftime("%Y%m%d")
    #    result_dir_name = f"result_{args.dataset_name}_{current_date}"
    #    args.output_folder = f"{current_dir}/result/{result_dir_name}"
     #   print(f"\n📁 自动生成输出目录: {args.output_folder}")
    #else:
    #    print(f"\n📁 使用指定输出目录: {args.output_folder}")
    #avg_recall = 0
    #avg_precision = 0
    #recall_list = []
    #precision_list = []
    #total_num = 0

        # 保留原来的结构，只修改输出文件夹路径为 'librarian'
    args.output_folder = "librarian"  # 直接设置输出文件夹为 'librarian' 文件夹

    # 如果 'librarian' 文件夹不存在，则创建它
    if not os.path.exists(os.path.join(args.output_folder)):
        os.makedirs(os.path.join(args.output_folder), exist_ok=True)  # 如果文件夹不存在，则创建它

    print(f"\n📁 使用指定输出目录: {args.output_folder}")




    #if not os.path.exists(os.path.join(args.output_folder)):
    #    os.makedirs(os.path.join(args.output_folder), exist_ok=True)
        
    # 修改stdout为日志类
    log_filename = os.path.join(args.output_folder, "run.log")
    setup_logging(os.path.join(args.output_folder, "detail_run.log"))
    tee_output = TeeOutput(log_filename)
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    # 记录所有启动参数到日志
    try:
        print("\n🧰 [MAIN] 启动参数如下:")
        for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"⚠️ [MAIN] 打印启动参数时出错: {e}")
    
    # 清空paper_cache
    cache_dir = os.path.join('search_utils', 'optimize_utils', 'paper_cache')
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
 
    #metrics_tracker = CallMetricsTracker()
    metrics_tracker = None


    with open(args.input_file) as f:
        lines = f.readlines()
        
    # 控制处理的数据量
    if args.max_queries > 0:
        lines = lines[:args.max_queries]
        print(f"📊 [MAIN] 限制处理前 {args.max_queries} 个查询")
    else:
        print(f"📊 [MAIN] 处理所有 {len(lines)} 个查询")
        
    for idx, line in enumerate(lines):
        print(f"\n🚀 ============= [MAIN] 处理第 {idx+1} 个查询 =============")
        start_time = time.time()
        
        # 检查是否已经处理过
        output_file = os.path.join(args.output_folder, f"{idx}.json")
        if os.path.exists(output_file):
            print(f"⏭️ [MAIN] 跳过已处理的查询 {idx}: {output_file}")
            continue
        
        #total_num += 1
        
        # 解析数据
        print(f"📄 [MAIN] 解析查询数据...")
        data = json.loads(line)
        question = data['question']
        # question = "总结2025年丁洪课题组的所有发表文章（作者含有丁洪的文章），包括其发表时间、发表期刊、第一作者（或所有共同第一作者）、通讯作者、第一单位信息" 
        print(f"❓ [MAIN] 查询内容: {question}")
        
        # 处理时间限制
        end_date = data['source_meta']['published_time']
        # end_date = datetime.now().strftime("%Y%m%d")
        original_end_date = end_date
        end_date = datetime.strptime(end_date, "%Y%m%d")
        end_date = end_date.strftime("%Y%m%d")
        print(f"📅 [MAIN] 时间限制: {original_end_date} -> {end_date}")
        
        # 创建PaperAgent
        print(f"🤖 [MAIN] 创建PaperAgent实例...")
        paper_agent = PaperAgent(
            task_id        = idx,
            user_query     = question, 
            keyminder      = keyminder,
            top_ranker     = top_ranker,
            ranker         = ranker,
            end_date       = end_date,
            expand_layers  = args.expand_layers,
            search_queries = args.search_queries,
            search_papers  = args.search_papers,
            expand_papers  = args.expand_papers,
            threads_num    = args.threads_num,
            max_num        = args.max_num,
            time_limit     = args.time_limit,
            idx            = idx ,
            output_dir     = args.output_folder
        )
        
        print(f"📊 [MAIN] Agent配置:")
        print(f"  扩展层数: {args.expand_layers}")
        print(f"  搜索论文数: {args.search_papers}")
        print(f"  线程数: {args.threads_num}")
        print(f"  最大论文数: {args.max_num}")
        print(f"  时间限制: {args.time_limit}秒")
        
        # 添加答案信息（如果有）
        if "answer" in data:
            paper_agent.root.extra["answer"] = data["answer"]
            print(f"📋 [MAIN] 标准答案: {len(data['answer'])} 篇论文")
        
        # 运行搜索
        print(f"🚀 [MAIN] 开始运行论文搜索...")
        # 记录开始时间
        run_start_time = time.time()
        paper_agent.run()
        # 记录结束时间
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        print(f"⏱️ [MAIN] paper_agent.run() 执行时间: {run_duration:.2f}秒")
        try:
            print(f"数据库命中率统计。命中：{paper_agent.hit},总次数：{paper_agent.search_times},命中率：{paper_agent.hit/paper_agent.search_times:.4f}")
        except:
            pass
        # 评估结果
        recall_papers = paper_agent.root.extra['recall_papers']
        print(f"📊 [MAIN] 搜索结果统计:")
        print(f"  找到论文数: {len(recall_papers)}")
        
        # 统计信息
        batch_time = time.time() - start_time
        
        # 保存结果
        #if args.output_folder != "":
        #    print(f"💾 [MAIN] 保存结果到: {output_file}")
        #    # 获取原始结果数据
        #    result_data = paper_agent.root.todic()
        #    # 添加执行时间信息
        #    result_data['execution_time'] = {
        #        'paper_agent_run_duration': run_duration,
        #        'total_batch_time': batch_time,
        #       'timestamp': datetime.now().isoformat()
        #    }
        #    with open(output_file, "w") as f:
        #        json.dump(result_data, f, ensure_ascii=False,indent=2)
###########################################################################
                # 保存结果
        if args.output_folder != "":
            print(f"💾 [MAIN] 保存结果到: {output_file}")
            # 1. 获取原始结果数据（里面已经包含每篇论文的 extra.core_knowledge）
            result_data = paper_agent.root.todic()

            # 2. 从 all_papers 里收集 core_knowledge，并回填到 recall_papers
            extra = result_data.get("extra", {}) or {}

            # 2.1 从 all_papers 里做一个 arxiv_id -> core_knowledge 的索引
            all_papers = extra.get("all_papers", [])
            core_by_id = {}
            core_list = []  # 额外挂一个汇总列表，方便下游直接用

            for p in all_papers:
                arxiv_id = p.get("arxiv_id")
                p_extra = p.get("extra", {}) or {}
                ck = p_extra.get("core_knowledge")
                if arxiv_id and ck:
                    core_by_id[arxiv_id] = ck
                    core_list.append({
                        "arxiv_id": arxiv_id,
                        "title": p.get("title", ""),
                        "core_knowledge": ck
                    })

            # 2.2 把 core_knowledge 写回到 recall_papers 里
            recall_papers = extra.get("recall_papers", [])
            for p in recall_papers:
                aid = p.get("arxiv_id")
                if aid in core_by_id:
                    # 这里新的字段名你可以自己定，这里叫 core_knowledge
                    p["core_knowledge"] = core_by_id[aid]
            extra["recall_papers"] = recall_papers

            # 2.3 额外挂一个汇总字段，按论文列出核心定性/定量知识
            extra["core_knowledge_list"] = core_list

            result_data["extra"] = extra

            # 3. 添加执行时间信息
            result_data['execution_time'] = {
                'paper_agent_run_duration': run_duration,
                'total_batch_time': batch_time,
                'timestamp': datetime.now().isoformat()
            }

            # 4. 写出到 json
            with open(output_file, "w") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
############################################################################

        print(f"\n📊 [MAIN] ============= 第 {idx+1} 个查询完成 =============")
        print(f"⏱️ [MAIN] 本次查询耗时: {batch_time:.2f}秒")
        print(f"🔧 [MAIN] 工具调用统计:")
        
        #metrics_tracker.update_metrics(idx, batch_time) # 更新并保存指标
        # print(f"  各工具平均用时: {get_times()}")
        # print(f"  Google搜索统计: {get_goo_cnt()}")
        # print(f"  摘要搜索统计: {get_id2ab()}")
        # print(f"  引用搜索统计: {get_id2ref()}")
        # print(f"  标题搜索统计: {get_ti2id()}")
        
        print(f"🤖 [MAIN] 模型调用统计:")
        print(f"  Keyminder: {keyminder.get_usage_stats()}")
        print(f"  Top_ranker: {top_ranker.get_usage_stats()}")
        print(f"  Ranker: {ranker.get_usage_stats()}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n🎉 =============== [MAIN] 所有查询处理完成 ===============")
    print(f"📁 [MAIN] 结果保存在: {args.output_folder}")
    print(f"📝 [MAIN] 日志文件: {log_filename}")
    
    # 关闭call_db.py中的db3_file资源
    # try:
    #     from search_utils.optimize_utils import call_db
    #     call_db.cleanup_resources()
    # except Exception as e:
    #     print(f"[Warning] 关闭call_db资源时出错: {e}")
    
    # 还原stdout
    sys.stdout = original_stdout
    tee_output.close()
    print(f"✅ [MAIN] 程序执行完成!")
