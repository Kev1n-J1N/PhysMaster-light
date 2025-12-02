import json
import os
import re
import sys
import time
import threading
import warnings
from datetime import datetime
import requests
from duckduckgo_search import DDGS
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_utils.call_mtr import CallMetricsTracker
from loguru import logger
import http.client

# 从配置文件读取Google API key和代理设置
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        common_config = config.get("common_config", {})
        google_key = common_config.get("google_api_key", "81abdd41afde8558170fb93f0a2d3766f035e41d")
        proxy_config = common_config.get("proxy", {})
        proxy_url = proxy_config.get("http_proxy", "http://127.0.0.1:7899").replace("http://", "")
        arxiv_search_url = common_config.get("arxiv_search_url", "http://192.168.129.82:30002/search")
        return google_key, proxy_url, arxiv_search_url
    except Exception as e:
        print(f"⚠️ 无法从配置文件加载配置: {e}")
        return "81abdd41afde8558170fb93f0a2d3766f035e41d", "127.0.0.1:7899", "http://192.168.129.82:30002/search"

GOOGLE_KEY, USE_PROXY, ARXIV_SEARCH_URL = load_config()
print(f"🔧 配置加载完成: Google API key: ***{GOOGLE_KEY[-4:]}, 代理: {USE_PROXY}")
def duckduckgo_search_arxiv_id(
    query: str, 
    max_results: int = 10,
    timelimit: str = None,  # 时间范围 d, w, m, y
    region: str = "wt-wt"
) -> list:
    query += " site:arxiv.org"
    try:
        with DDGS(proxy=USE_PROXY, timeout=8) as ddgs:
            arxiv_id_list = []
            kwargs = {
                "keywords": query,
                "region": region,
                "timelimit": timelimit,
                "max_results": max(8, max_results + 5)
            }
            for idx, result in enumerate(ddgs.text(**kwargs)):
                url = result.get("href", "")
                if "arxiv.org" in url:
                    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', url)
                    if match:
                        arxiv_id = match.group(1)
                        arxiv_id_list.append(arxiv_id)
                if len(arxiv_id_list) >= max_results:
                    break
            return list(set(arxiv_id_list))

    except Exception as e:
        print(f"搜索出错: {str(e)}")
        return []

def google_search_arxiv_id(query, num=10, end_date=None):
    query = re.sub(r'\\"|"', '', query)
    metrics_tracker = CallMetricsTracker()
    
    start_time = time.time()
    conn = http.client.HTTPSConnection("google.serper.dev")
    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    
    payload = json.dumps({
    "q": search_query,
    "num": num, 
    "page": 1, 
    })
    headers = {
    'X-API-KEY': GOOGLE_KEY,
    'Content-Type': 'application/json'
    }
    assert headers['X-API-KEY'] != 'your google keys', "add your google search key!!!"
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    try:
        results = json.loads(data.decode("utf-8"))
        arxiv_id_list = []
        for paper in results['organic']:
            if re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]):
                arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]).group(1)
                arxiv_id_list.append(arxiv_id)
        metrics_tracker.add_google(success=True, time_cost=time.time()-start_time)
        return list(dict.fromkeys(arxiv_id_list))
    except:
        warnings.warn(f"google search failed, query: {query}")
    metrics_tracker.add_google(success=False, time_cost=time.time()-start_time)
    return []

def arxiv_search_id(query, num=100, end_date=None,threshold=0.7):
    url = ARXIV_SEARCH_URL
    payload = {
        "questions": query,
        "topk": num
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        results = response.json()["results"][0]
        scores = response.json()["scores"][0]
        result = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)[:num]
        # 保留所有大于阈值的文献
        arxiv_id_list = [r["arxiv_id"] for s, r in result if s > threshold]
        return list(set(arxiv_id_list))
    else:
        print(f"\033[31m {url} 请求失败，状态码: {response.status_code}")
        return []
    
def title_search_rag(query, num=2048, threshold=0.7):
    url = ARXIV_SEARCH_URL
    payload = {
        "questions": query,
        "topk": num
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        scores_list = []
        results_list = []
        response = response.json()
        scores = response["scores"]
        results = response["results"]
        for i in range(len(scores)):
            scores_list.extend(scores[i])
        for i in range(len(results)):
            results_list.extend(results[i])
        result = sorted(zip(scores_list, results_list), key=lambda x: x[0], reverse=True)
        # 保留所有大于阈值的文献
        arxiv_id_list = list(set([r['title'] for s, r in result if s > threshold]))
        logger.info(f"\033[32m 总检索结果: query: {query}, num: {len(arxiv_id_list)}\033[0m")
        return list(set(arxiv_id_list))
    else:
        logger.exception(f"\033[31m {url} 请求失败，状态码: {response.status_code}")
        return []

if __name__ == "__main__":
    query = ["scimaster"]
    num = 5
    end_date = "20251201"
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(arxiv_search_id, query, num) for _ in range(1)]
        for future in as_completed(futures):
            results = future.result()
            print(results)

    exit(1)
    # question_list =  ['data curation AND large language model AND pre-training AND performance', 'data filtering AND data efficiency AND LLM pre-training', 'quality over quantity AND language model pre-training AND dataset size']
    # save_data = []
    # for question in question_list:
    #     results = title_search_rag([question])
    #     save_config = {
    #         "question": question,
    #         "results": results
    #     }
    #     save_data.append(save_config)
    # with open('/data/duyuwen/Pasa-X/result/save_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(save_data, f, ensure_ascii=False, indent=4)