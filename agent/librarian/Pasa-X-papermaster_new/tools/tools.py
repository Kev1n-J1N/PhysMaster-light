from urllib.parse import quote_plus

import requests


def search_arxiv_by_title_2(title):
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=ti:{quote_plus(title)}&start=0&max_results=1"  # 搜索标题
    response = requests.get(base_url + query)

    if response.status_code == 200:
        # 解析XML数据
        from xml.etree import ElementTree

        root = ElementTree.fromstring(response.content)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')

        if entry is not None:
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
            return arxiv_id
        else:
            return None
    else:
        return None

# 示例用法
import concurrent.futures


def search_arxiv_by_title_wrapper(title):
    return search_arxiv_by_title_2(title)

times = 100
count = 0
# with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
#     futures = [executor.submit(search_arxiv_by_title_wrapper, "The power of scale for parameter-efficient prompt tuning") for _ in range(times)]
#     for future in concurrent.futures.as_completed(futures):
#         arxiv_id = future.result()
#         if arxiv_id is not None:
#             count += 1
# print(f"Found {count} ArXiv IDs")
print(search_arxiv_by_title_2("Palm: Scaling language modeling with pathways."))