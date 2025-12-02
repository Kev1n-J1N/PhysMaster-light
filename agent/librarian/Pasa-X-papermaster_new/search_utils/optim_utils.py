import os
import re
import bs4
import time
import arxiv
import requests
import urllib.parse
import urllib.request as libreq
import xml.etree.ElementTree as ET
import sys,os
current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from search_utils.optimize_utils.call_db import search_db3_id, search_db3_title, search_db1_id, search_db1_title
from search_utils.optimize_utils.fetch_arxiv import is_similar_title
from search_utils.optimize_utils.parse_utils import parse_tex_folder, fetch_eprint, extract_tar_gz, parse_tex_folder_content
from search_utils.call_mtr import CallMetricsTracker
from func_timeout import func_set_timeout
import json
from loguru import logger
metrics_tracker = CallMetricsTracker()
with open(f"{os.path.dirname(os.path.abspath(__file__))}/../config/config.json", "r") as f:
    config = json.load(f)

TIMEOUT = config["timeout"]

############### ti2id ###############
def fetch_arxiv(title):
    """【不直接调用的工具函数】根据title搜索arXiv，匹配则返回arxiv_id/title/abstract，否则返回None"""
    encoded_title = urllib.parse.quote(title) # 对标题进行 URL 编码
    url = f'http://export.arxiv.org/api/query?search_query=ti:{encoded_title}&max_results=200'
    try:
        with libreq.urlopen(url, timeout=120) as response:
            data = response.read()
        root = ET.fromstring(data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('.//atom:entry', namespace) # 查找所有条目
        best_match = None
        highest_ratio = 0
        for entry in entries:
            entry_title = entry.find('atom:title', namespace).text.lower()
            if is_similar_title(entry_title, title.lower()):
                best_match = entry
                break
        if best_match is not None:
            id_url = best_match.find('atom:id', namespace).text
            arxiv_id = id_url.split('/abs/')[-1]
            summary = best_match.find('atom:summary', namespace).text
            return {
                'arxiv_id': arxiv_id.split('v')[0],
                'title': title,
                'abstract': summary
            }
        else:
            return None
    except Exception as e:
        logger.exception(f"搜索 {title} 时发生错误: {str(e)}")
        return None

def fetch_arxiv1(title):
    """ 原论文做法
    Search arxiv id by title.
    :param title: title of the paper
    :return: arxiv id of the paper
    """
    url = "https://arxiv.org/search/?" + urllib.parse.urlencode({
        'query': title,
        'searchtype': 'title', 
        'abstracts': 'hide', 
        'size': 200, 
    })
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            html_content = response.text
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
            results = []

            if soup.find('meta', charset=True): # paper list
                if soup.find('p', class_="is-size-4 has-text-warning") and "Sorry" in soup.find('p', class_="is-size-4 has-text-warning").text.strip():
                    logger.info(f"Failed to find results by Arxiv Advanced Search: {title}")
                    return None
                
                p_tags = soup.find_all("li", class_="arxiv-result")
                for p_tag in p_tags:
                    title_ = p_tag.find("p", class_="title is-5 mathjax").text.strip()
                    id = p_tag.find('p', class_='list-title is-inline-block').find('a').text.strip('arXiv:')
                    if title_ and id:
                        results.append((title_, id))
            if soup.find('html', xmlns=True): # a single paper
                p_tag = soup.find("head").find("title")
                match = re.match(r'\[(.*?)\]\s*(.*)', soup.title.string)
                if match:
                    id = match.group(1)
                    title_ = match.group(2)
                    if title_ and id:
                        results = [(title_, id)]

            if results:
                for (result, id) in results:
                    if is_similar_title(result, title):
                        return { 'arxiv_id': id }
                return None
            
            logger.info(f"Failed to parse the html: {url}")
            return None
        else:
            logger.info(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        logger.exception(f"An error occurred while search_arxiv_id_by_title: {e}")
        return None    

def search_id_by_title(title):
    """根据title搜索本地数据库db3、db1和arxiv，匹配则返回arxiv_id/title/abstract，否则返回None"""
    start_time = time.time()
    res = search_db3_title(title)
    if res is None:
        db1_time = time.time()
        res = search_db1_title(title)
        metrics_tracker.add_ti2id(
            success=res is not None, db1_call=True, db1_success=res is not None, 
            db3_time=db1_time-start_time, db1_time=time.time()-db1_time, time_cost=time.time() - start_time
        )
    else:
        metrics_tracker.add_ti2id(success=True, db3_success=True, db3_time=time.time() - start_time, time_cost=time.time() - start_time)
    # res包括id/title/abstract，为对接仅返回arxiv_id
    if res is not None:
        res = res['arxiv_id']
    return res


############### id2abs ###############
arxiv_client = arxiv.Client(delay_seconds = 0.05)
def search_axv_id(arxiv_id):
    """根据aid，调用arxiv搜索paper，返回aid/title/abstract，或者None"""
    search = arxiv.Search(
        query = "",
        id_list = [arxiv_id],
        max_results = 10,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending,
    )

    try:
        results = list(arxiv_client.results(search, offset=0))
    except:
        logger.exception(f"Failed to search arxiv id: {arxiv_id}")
        return None

    res = None
    for arxiv_id_result in results:
        entry_id = arxiv_id_result.entry_id.split("/")[-1]
        entry_id = entry_id.split('v')[0]
        if entry_id == arxiv_id:
            res = {
                "arxiv_id": arxiv_id,
                "title": arxiv_id_result.title.replace("\n", " "),
                "abstract": arxiv_id_result.summary.replace("\n", " "),
                "sections": "",
                "source": 'SearchFrom:arxiv',
                "journal": "",
                "authors": "",
                "year": ""
            }
            break
    return res

def search_abs_by_id(arxiv_id):
    """根据aid搜索paper，返回aid/title/abstract，或者None"""
    start_time = time.time()
    db3_start = start_time
    res = search_db3_id(arxiv_id)
    if res is None:
        db1_start = time.time()
        res = search_db1_id(arxiv_id)
        if res is None:
            axv_start = time.time()
            res = search_axv_id(arxiv_id)
            if res is not None:
                metrics_tracker.add_id2ab(
                    success=True, db1_call=True, axv_call=True, axv_success=True,
                    db3_time=db1_start-db3_start, db1_time=axv_start-db1_start, axv_time=time.time()-axv_start,
                    time_cost=time.time() - start_time
                )
            else:
                metrics_tracker.add_id2ab(
                    success=False, db1_call=True, axv_call=True, axv_success=False,
                    db3_time=db1_start-db3_start, db1_time=axv_start-db1_start, axv_time=time.time()-axv_start,
                    time_cost=time.time() - start_time
                )
        else:
            metrics_tracker.add_id2ab(
                success=True, db1_call=True, db1_success=True,
                db3_time=db1_start-db3_start, db1_time=time.time()-db1_start,
                time_cost=time.time() - start_time
            )
    else:
        metrics_tracker.add_id2ab(
            success=True, db3_success=True,
            db3_time=time.time() - db3_start, time_cost=time.time() - start_time
        )
    return res


############### id2ref ###############
def search_ref_by_ar5iv(arxiv_id):
    from utils import search_section_by_arxiv_id
    res,hit = search_section_by_arxiv_id(arxiv_id)
    if res is not None:
        res = {
            "arxiv_id": arxiv_id,
            "sections": res,
        }
    if res is None:
        logger.info(f"ar5iv内容解析失败: {arxiv_id}")
        return None,hit
    logger.info(f"\033[94m use ar5iv html解析成功：{arxiv_id} \033[0m")
    return res,hit

def search_ref_by_tex(arxiv_id,base_dir):
    tar_path,hit = fetch_eprint(arxiv_id, base_dir)
    if tar_path is None:
        return None,hit
    paper_dir = os.path.join(base_dir, f'{arxiv_id}')
    extract_tar_gz(tar_path, paper_dir)
    result = parse_tex_folder(paper_dir)
    if result == [] or result == "" or result == {}:
        return None,hit
    return {
        'arxiv_id': arxiv_id,
        'sections': result,
    },hit


@func_set_timeout(TIMEOUT)
def fetch_ref(arxiv_id):
    """【不直接调用的工具函数】根据arxiv_id搜索tex和ar5iv，匹配则返回arxiv_id/sections，否则返回None"""
    if config["save_data"] or config["use_local_db"]:
        base_dir = f"{config['arxiv_database_path']}/arxiv/tex/{arxiv_id.split('.')[0]}"
    else:
        base_dir = 'search_utils/optimize_utils/paper_cache'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    paper_dir = os.path.join(base_dir, f'{arxiv_id}')
    begin = time.time()
    result,hit = search_ref_by_ar5iv(arxiv_id)
    metrics_tracker.add_id2ref(
        success=True, ar5iv_call=True, ar5iv_success=True,
        ar5iv_time=time.time()-begin, time_cost=time.time() - begin
    )
    return result,hit
   
    
def search_ref_by_id(arxiv_id):
    """根据arxiv_id搜索tex和ar5iv，匹配则返回arxiv_id/sections，否则返回None"""
    begin = time.time()
    try:
        res,hit = fetch_ref(arxiv_id.split('v')[0])
    except Exception as e:
        logger.exception(f"\033[91m 获取章节引用失败: {arxiv_id}, 错误: {e}\033[0m")
        res,hit = None,0
    end = time.time()
    if end-begin > TIMEOUT:
        logger.info(f"\033[91m 获取章节引用超时，耗时：{end-begin}，论文为：{arxiv_id}\033[0m")
   
    # res包括id/sections原方法仅返回sections
    if res is not None:
        res = res['sections']
    return res,hit


def search_info_by_id(arxiv_id):
    if not config["use_local_pdf"]:
        ref,hit = search_ref_by_id(arxiv_id)
        refs = {}
        if ref == None:
            return [],0
        idx = 1
        for key,values in ref.items():
            for value in values:
                refs[str(idx)] = value
                idx += 1
    else:
        from search_utils.optimize_utils.arxiv_database import search_info_by_local_db
        refs = search_info_by_local_db(arxiv_id)
        hit = 0
    return refs,hit


def search_content_by_tex(arxiv_id,base_dir):
    """
    根据arxiv_id搜索tex文件，返回{arxiv_id: id, sections: {章节名: 前max_chars字内容}}，否则返回None
    """
    tar_path,hit = fetch_eprint(arxiv_id, base_dir)
    if tar_path is None:
        return None
    paper_dir = os.path.join(base_dir, f'{arxiv_id}')
    extract_tar_gz(tar_path, paper_dir)
    result = parse_tex_folder_content(paper_dir)
    if result == [] or result == "" or result == {}:
        return None
    return {
        'arxiv_id': arxiv_id,
        'sections': result,
    }
    
def search_content_by_ar5iv(arxiv_id):
    """
    根据arxiv_id搜索ar5iv，返回{arxiv_id: id, sections: {章节名: 前max_chars字内容}}，否则返回None
    """
    from utils import search_content_by_arxiv_id
    res = search_content_by_arxiv_id(arxiv_id)
    if res is not None:
        res = {
            "arxiv_id": arxiv_id,
            "sections": res,
        }
    return res


############### id2cnt ###############
def fetch_content(arxiv_id, max_chars=300):
    """
    获取章节内容
    根据arxiv_id搜索tex和ar5iv，返回{arxiv_id: id, sections: {章节名: 前max_chars字内容}}，否则返回None
    """
    from utils import search_content_by_arxiv_id
    start_time = time.time()
    if config["save_data"] or config["use_local_db"]:
        base_dir = f"{config['arxiv_database_path']}/arxiv/tex/{arxiv_id.split('.')[0]}"
    else:
        base_dir = 'search_utils/optimize_utils/paper_cache'
    try:
        res = search_content_by_ar5iv(arxiv_id)
        if res is None:
            raise Exception("ar5iv内容解析失败")
        metrics_tracker.add_id2cnt(
            success=True, ar5iv_call=True, ar5iv_success=True,
            ar5iv_time=time.time()-start_time, time_cost=time.time() - start_time
        )
        return res
    except Exception as e:
        begin = time.time()
        logger.exception(f"ar5iv内容解析失败: {arxiv_id}, 使用tex解析，错误: {e}")
        res = search_content_by_tex(arxiv_id,base_dir)
        metrics_tracker.add_id2cnt(
            success=True, tex_call=True, tex_success=res is not None, ar5iv_call=True, ar5iv_success=False, tex_time=time.time()-begin, time_cost=time.time() - start_time
        )
        if res is None:
            logger.error(f"\033[91m tex内容解析失败: {arxiv_id}\033[0m")
            return None
        logger.info(f"\033[94m tex内容解析成功：{arxiv_id} \033[0m")
        return res
        

def search_content_by_id(arxiv_id, max_chars=300):
    """根据arxiv_id搜索章节内容，返回{章节名: 前max_chars字内容}的字典，否则返回None"""
    res = fetch_content(arxiv_id.split('v')[0], max_chars)
    if res is not None:
        res = res['sections']
    return res


if __name__ == '__main__':
    print(search_info_by_id('2402.09668'))
    # print(search_abs_by_id('2402.09668'))
    # print(search_id_by_title('CEREALS – Cost-Effective REgion-based Active Learning for Semantic Segmentation'))
