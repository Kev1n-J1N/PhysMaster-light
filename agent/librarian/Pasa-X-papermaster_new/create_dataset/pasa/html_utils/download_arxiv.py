import requests
import xml.etree.ElementTree as ET
import json
import time
from tqdm import tqdm

BASE = 'http://export.arxiv.org/oai2'


def fetch_ids(num, paper_type):
    PARAMS = {
        'verb': 'ListIdentifiers',
        'set': paper_type,
        'metadataPrefix': 'arXiv',
        'from': '2020-01-01',
        'until': '2025-06-23',
    }
    ids = []
    params = PARAMS.copy()

    # 初始化 tqdm，预设目标为 num，动态显示 progress
    pbar = tqdm(total=num, desc=f"Fetching {paper_type}", unit="id")
    try:
        while True:
            resp = requests.get(BASE, params=params)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {'oai': 'http://www.openarchives.org/OAI/2.0/'}

            # 遍历每次响应的 identifier
            for header in root.findall('.//oai:header', ns):
                ident = header.find('oai:identifier', ns)
                if ident is not None:
                    arxiv_id = ident.text.strip().split(':')[-1]
                    ids.append(arxiv_id)
                    pbar.update(1)
                    pbar.set_postfix_str(f"last={arxiv_id}")
                    if len(ids) >= num:
                        break

            # 判断是否达到目标数量
            if len(ids) >= num:
                break

            token = root.find('.//oai:resumptionToken', ns)
            if token is None or token.text is None:
                break

            params = {
                'verb': 'ListIdentifiers',
                'resumptionToken': token.text
            }
            time.sleep(1)  # 限速，避免请求过快

    finally:
        pbar.close()

    return ids

def filter_ids(ids):
    filtered_ids = []
    for id in ids:
        try:
            if int(id[:2]) >= 0:
                filtered_ids.append(id)
        except:
            continue
    return filtered_ids

def main():
    num = 100000
    math_ids = fetch_ids(num,'math')
    math_ids = filter_ids(math_ids)
    print(f"共获取到 {len(math_ids)} 条数学领域 arXiv ID")
    with open('result/math_ids.json', 'w') as f:
        json.dump(math_ids, f, indent=2)
    physics_ids = fetch_ids(num,'physics')
    physics_ids = filter_ids(physics_ids)
    print(f"共获取到 {len(physics_ids)} 条物理领域 arXiv ID")
    with open('result/physics_ids.json', 'w') as f:
        json.dump(physics_ids, f, indent=2)

if __name__ == '__main__':
    main()
