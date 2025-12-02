
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
from datetime import datetime, timedelta

from loguru import logger
import uvicorn
from fastapi import FastAPI
from model_vllm import Agent
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
from paper_agent import PaperAgent
from pydantic import BaseModel
import time

from log import setup_logging
from starlette.concurrency import run_in_threadpool


setup_logging()
parser = argparse.ArgumentParser()
parser.add_argument('--keyminder_model', type=str, default="qwen3-30b", help="Keyminder使用的模型名称")
parser.add_argument('--ranker_model',   type=str, default="qwen3-30b", help="Ranker使用的模型名称")
parser.add_argument('--top_rank_model',    type=str, default="reason_rank", help="Top Ranker使用的模型名称")
parser.add_argument('--config_path',     type=str, default="config/config.json", help="配置文件路径")
parser.add_argument('--output_folder',  type=str, default="", help="输出文件夹路径，如果为空则自动生成")
parser.add_argument('--expand_layers',  type=int, default=2)
parser.add_argument('--search_queries', type=int, default=5)
parser.add_argument('--search_papers',  type=int, default=10)
parser.add_argument('--expand_papers',  type=int, default=100)
parser.add_argument('--threads_num',    type=int, default=20)
parser.add_argument('--max_num',   type=int, default=-1)
parser.add_argument('--time_limit', type=int, default=-1)
parser.add_argument('--max_queries', type=int, default=1, help="最大处理查询数量，-1表示处理所有数据")
parser.add_argument('--further_sample_limit', type=int, default=80, help="further_search抽样论文数量上限")
args = parser.parse_args()

# 从配置读取代理并设置环境变量
try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.json")
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
    common_config = loaded_config.get("common_config", {})
    proxy_cfg = common_config.get("proxy", {})
    http_proxy = proxy_cfg.get("http_proxy", "")
    https_proxy = proxy_cfg.get("https_proxy", http_proxy)
    if http_proxy != "" and https_proxy != "":
        os.environ['http_proxy'] = http_proxy
        os.environ['https_proxy'] = https_proxy
        logger.info(f"🌐 已设置代理: http_proxy={os.environ['http_proxy']} https_proxy={os.environ['https_proxy']}")
    else:
        logger.info(f"🤖 代理配置错误: {http_proxy}, {https_proxy}")
except Exception as proxy_e:
    # 出错时使用默认代理，但不中断流程
    logger.info(f"⚠️ 设置代理失败原因: {proxy_e}")

# 模型配置处理
logger.info(f"\n🔧 模型配置处理...")
logger.info(f"📄 使用配置文件: {args.config_path}")

# 直接使用模型名称创建Agent

try:
    logger.info(f"\n🚀 创建Agent实例...")
    keyminder = Agent(model_name=args.keyminder_model, config_path=args.config_path)
    top_ranker = Agent(model_name=args.top_rank_model, config_path=args.config_path)
    ranker = Agent(model_name=args.ranker_model, config_path=args.config_path)
    logger.info(f"✅ 所有Agent创建成功!")
except Exception as e:
    logger.info(f"❌ 创建Agent失败: {e}")
    sys.exit(1)

logger.info(f"\n📊 Agent配置摘要:")
logger.info(f"  Keyminder: {keyminder.model_name} @ {keyminder.url}")
logger.info(f"  Top Ranker:   {top_ranker.model_name} @ {top_ranker.url}")
logger.info(f"  Ranker:    {ranker.model_name} @ {ranker.url}")

# 记录所有启动参数到日志
try:
    logger.info("\n🧰 [MAIN] 启动参数如下:")
    for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
        logger.info(f"  {key}: {value}")
except Exception as e:
    logger.info(f"⚠️ [MAIN] 打印启动参数时出错: {e}")
    
def run_pasa(question,num=100):
    logger.info(f"❓ [MAIN] 查询内容: {question}")
    end_date = datetime.now().strftime("%Y%m%d")
    idx = 0
    logger.info(f"🤖 [MAIN] 创建PaperAgent实例...")
    task_id = "server"
    paper_agent = PaperAgent(
        task_id        = task_id,
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

    # 记录开始时间
    run_start_time = time.time()
    paper_agent.run()
    # 记录结束时间
    run_end_time = time.time()
    run_duration = run_end_time - run_start_time
    answer_list = paper_agent.root.extra.get("recall_papers", [])
    logger.info(answer_list)
    logger.info(f"⏱️ \033[32m[MAIN] paper_agent.run() 执行时间: {run_duration:.2f}秒\033[0m")
    logger.info(f"  \033[32m[MAIN] 找到论文数: {len(answer_list)}\033[0m")
    # sort by score
    answer_list.sort(key=lambda x: x["score"], reverse=True)
    answer_list = answer_list[:num]
    return answer_list
    
        

# ----------------- FastAPI 部分 -----------------
app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    num:int = 20

@app.post("/pasa")
async def pasa_api(req: QueryRequest):
    result = await run_in_threadpool(run_pasa, req.question, req.num)
    return {"result": result}

# 仅用于本地调试
if __name__ == "__main__":
    import time
    uvicorn.run("pasa_server:app", host="127.0.0.1", port=1235, reload=False)
    # begin = time.time()
    # print(run_pasa("Show me research on the t-j model in high-temperature superconductivity research"))
    # end = time.time()
    # print(f"Time taken: {end - begin} seconds")
