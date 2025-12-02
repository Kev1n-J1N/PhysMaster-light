import json
import os
import re
import time
import arxiv
from typing import Dict, List, Optional

def clean_filename(topic: str) -> str:
    """清理topic名称作为文件名"""
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', topic)
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned

def fetch_paper_info_from_arxiv(arxiv_id: str) -> Optional[Dict]:
    """
    从arXiv获取论文信息
    
    Args:
        arxiv_id: arXiv论文ID (如 "2009.00236")
    
    Returns:
        论文信息字典，获取失败返回None
    """
    try:
        # 创建arxiv客户端并搜索
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        
        # 获取结果
        results = list(client.results(search))
        
        if not results:
            print(f"[WARNING] No paper found for arXiv ID: {arxiv_id}")
            return None
        
        paper = results[0]
        
        # 提取作者信息
        authors = [author.name for author in paper.authors]
        
        return {
            "id": arxiv_id,
            "title": paper.title,
            "abstract": paper.summary.replace('\n', ' ').strip(),
            "authors": authors,
            "published": paper.published.isoformat()
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch arXiv paper {arxiv_id}: {e}")
        return None

def convert_citation_graph_to_format(graph_data: Dict) -> List[Dict]:
    """
    将引用图数据转换为所需格式，使用arxiv库获取论文信息
    
    Args:
        graph_data: 来自jsonl文件的单行数据
    
    Returns:
        转换后的论文列表
    """
    result = []
    processed_papers = set()
    paper_cache = {}  # 缓存已获取的论文信息
    
    # 收集所有需要获取的论文ID
    paper_ids = set()
    
    # 添加root论文ID
    root_id = graph_data.get("root")
    if root_id:
        paper_ids.add(root_id)
    
    # 添加图中的所有论文ID
    for edge in graph_data.get("graph", []):
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id:
            paper_ids.add(from_id)
        if to_id:
            paper_ids.add(to_id)
    
    # 批量获取论文信息并缓存
    print(f"[INFO] Fetching {len(paper_ids)} papers from arXiv...")
    for paper_id in paper_ids:
        if paper_id not in paper_cache:
            paper_info = fetch_paper_info_from_arxiv(paper_id)
            if paper_info:
                paper_cache[paper_id] = paper_info
            time.sleep(0.5)  # 避免过于频繁的API调用
    
    # 首先添加root论文
    if root_id and root_id in paper_cache:
        root_paper = paper_cache[root_id]
        result.append({
            "id": root_paper["id"],
            "title": root_paper["title"],
            "authors": root_paper["authors"],
            "published": root_paper["published"],
            "cited_by": "",
            "description": root_paper["abstract"]  # root论文用abstract作为description
        })
        processed_papers.add(root_id)
    
    # 处理图中的引用关系
    for edge in graph_data.get("graph", []):
        from_id = edge.get("from")
        to_id = edge.get("to")
        from_paper = paper_cache[from_id]
        to_paper = paper_cache[to_id]
        description = edge.get("description", "")
        
        # 添加from论文（如果还没添加且不是root）
        if from_id and from_id not in processed_papers and from_id in paper_cache:
            result.append({
                "id": from_paper["id"],
                "title": from_paper["title"],
                "authors": from_paper["authors"],
                "published": from_paper["published"],
                "cited_by": "",
                "description": from_paper["abstract"]  # 没有特定描述时使用abstract
            })
            processed_papers.add(from_id)
        
        # 添加或更新to论文
        if to_id and to_id in paper_cache:
            if to_id not in processed_papers:
                # 首次添加论文
                result.append({
                    "id": to_paper["id"],
                    "title": to_paper["title"],
                    "authors": to_paper["authors"],
                    "published": to_paper["published"],
                    "cited_by": from_id,
                    "description": description if description.strip() else to_paper["abstract"]
                })
                processed_papers.add(to_id)
            else:
                # 论文已存在，更新描述
                for paper in result:
                    if paper["id"] == to_id:
                        current_desc = paper["description"]
                        
                        # 如果当前描述是abstract，直接替换为新的description
                        if current_desc == to_paper["abstract"]:
                            paper["description"] = description if description.strip() else to_paper["abstract"]
                            paper["cited_by"] = from_id
                        else:
                            # 如果已有description，则追加新的description
                            if description.strip():
                                paper["description"] = current_desc + "\n" + description
                                paper["cited_by"] += f", {from_id}" if from_id else ""
                        break
    
    return result

def convert_cite_graphs_to_individual_files(
    input_jsonl_path: str, 
    output_dir: str = "citation_graphs"
):
    """
    将cite_graphs.jsonl转换为单独的JSON文件，使用arXiv API获取论文信息
    
    Args:
        input_jsonl_path: 输入的jsonl文件路径
        output_dir: 输出目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    graph_data = json.loads(line.strip())
                    topic = graph_data.get("topic", f"topic_{line_num}")
                    
                    # 检查是否有错误或空图
                    if graph_data.get("error") or not graph_data.get("graph"):
                        print(f"[WARNING] Skipping {topic}: {graph_data.get('error', 'Empty graph')}")
                        continue
                    
                    print(f"[INFO] Processing topic: {topic}")
                    
                    # 转换数据格式（会自动从arXiv获取论文信息）
                    converted_data = convert_citation_graph_to_format(graph_data)
                    
                    if not converted_data:
                        print(f"[WARNING] No valid papers found for topic: {topic}")
                        continue
                    
                    # 创建输出文件名
                    filename = f"{clean_filename(topic)}.json"
                    output_path = os.path.join(output_dir, filename)
                    
                    # 保存转换后的数据
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(converted_data, out_f, ensure_ascii=False, indent=2)
                    
                    print(f"[SUCCESS] Converted '{topic}' -> {filename} ({len(converted_data)} papers)")
                    
                    # 在处理每个topic后稍作停顿
                    time.sleep(1)
                    
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"[ERROR] Processing line {line_num}: {e}")
    
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {input_jsonl_path}")
    except Exception as e:
        print(f"[ERROR] {e}")

# 使用示例
if __name__ == "__main__":
    # 首先需要安装arxiv库：pip install arxiv
    
    print("开始转换cite_graphs.jsonl...")
    print("注意：需要联网从arXiv获取论文信息")
    
    convert_cite_graphs_to_individual_files(
        input_jsonl_path="papers/cite_graphs.jsonl",
        output_dir="papers/APASbech_comp" 
    )
    
    print("转换完成！")