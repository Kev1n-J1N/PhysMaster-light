import json
import uuid
from datetime import datetime

from model_vllm import Agent
from paper_agent import PaperAgent

CONFIG_PATH = "config/config.json"
KB_PATH = "paper_kb.jsonl"   # 先用一个 JSONL 当“知识库”

def build_user_query(keyword: str, query: str) -> str:
    # 你可以根据需要换成其他格式
    if keyword and query:
        return f"{keyword}: {query}"
    return keyword or query

def create_llm_agents():
    """
    按照 config/config.json 里的 model_config 创建 3 个子 Agent：
    - keyminder: 负责翻译 & 关键词 + 查询生成
    - top_ranker: 中间筛选
    - ranker: 最终打分 + reason
    """
    keyminder_model = "gpt-4o"
    top_rank_model  = "gpt-4o"
    ranker_model    = "gpt-4o"


    keyminder = Agent(model_name=keyminder_model, config_path=CONFIG_PATH)
    top_ranker = Agent(model_name=top_rank_model, config_path=CONFIG_PATH)
    ranker = Agent(model_name=ranker_model, config_path=CONFIG_PATH)

    return keyminder, top_ranker, ranker

def run_paper_agent(keyword: str, query: str, max_papers: int = 20):
    user_query = build_user_query(keyword, query)
    task_id = str(uuid.uuid4())

    keyminder, top_ranker, ranker = create_llm_agents()

    agent = PaperAgent(
        task_id        = task_id,
        user_query     = user_query,
        keyminder      = keyminder,
        top_ranker     = top_ranker,
        ranker         = ranker,
        end_date       = datetime.now().strftime("%Y%m%d"),
        expand_layers  = 1,        # 先只扩一层，跑快一些
        search_queries = 5,
        search_papers  = 10,
        expand_papers  = 20,
        threads_num    = 8,
        max_num        = max_papers,
        time_limit     = -1,
        idx            = 0,
        output_dir     = "result"
    )

    root = agent.run()   # 或 agent.process()

    # root.extra["sorted_papers"] 是已经按得分排序后的论文列表（dict）
    sorted_papers = root.extra.get("sorted_papers", [])
    return user_query, sorted_papers

def write_to_kb(user_query: str, papers: list, kb_path: str = KB_PATH):
    """
    把“论文 + 解读”写到一个简单的 JSONL 知识库里。
    后续你要接向 Faiss / Milvus / Chroma，都可以从这个 JSONL 再读。
    """
    with open(kb_path, "a", encoding="utf-8") as f:
        for p in papers:
            record = {
                "id":       p.get("arxiv_id") or str(uuid.uuid4()),
                "title":    p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "reason":   p.get("reason", ""),   # ← 这里就是模型对论文的“解读/推理”
                "depth":    p.get("depth", -1),
                "score":    p.get("select_score", -1),
                "source":   p.get("source", ""),
                "query":    user_query,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 一个简单的 demo：命令行输入 keyword / query
    keyword = input("Keyword（可空）: ").strip()
    query = input("Query: ").strip()

    user_query, papers = run_paper_agent(keyword, query, max_papers=20)
    print(f"\n共获得 {len(papers)} 篇排序后的论文。前 5 篇：\n")
    for i, p in enumerate(papers[:5], 1):
        print(f"{i}. {p.get('title')}")
        print(f"   score = {p.get('select_score')}, depth = {p.get('depth')}")
        print(f"   reason = {p.get('reason', '')[:120]}...\n")

    write_to_kb(user_query, papers)
    print(f"已追加写入知识库文件：{KB_PATH}")
