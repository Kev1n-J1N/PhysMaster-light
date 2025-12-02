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
   and replace the 'your google keys' below before you can use it.
2. The service for searching arxiv and obtaining paper contents is relatively simple. 
   If there are any bugs or improvement suggestions, you can submit pull requests.
   We would greatly appreciate and look forward to your contributions!!
"""
import re
import bs4
import json
import arxiv
import urllib
import zipfile
import warnings
import requests
from datetime   import datetime
warnings.simplefilter("always")
import threading
import time

GOOGLE_KEY   = 'a5d20c46dc0c1a7a926cb8491ca9e459610d4a02'
arxiv_client = arxiv.Client(delay_seconds = 0.05)


def search_section_ref_by_arxiv_id(entry_id, cite_pattern):
    assert re.match(r'^\d+\.\d+$', entry_id)
    url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'

    try:
        response = requests.get(url)
        if response.status_code != 200:
            warnings.warn(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
        
        html_content = response.text
        # 保存HTML到当前目录
        with open(f"{entry_id}.html", "w", encoding="utf-8") as f:
            f.write(html_content)


    except requests.RequestException as e:
        warnings.warn(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    print(search_section_ref_by_arxiv_id("2001.00046", r"~\\cite\{(.*?)\}"))#2404.03447
    #print(search_paper_by_arxiv_id("2501.10120"))
    #print(search_paper_by_title("A hybrid approach to CMB lensing reconstruction on all-sky intensity maps"))

    ### 提取某章节的原文


    ##### 提取各章节的引用（标题、作者、期刊等）信息
    # max_count=100
    # input_path="../create_dataset/result/dataset/math_ids.json"
    # output_path=f"../create_dataset/result/dataset/math_ids_metadata_{max_count}.json"
    # process_arxiv_ids(input_path, output_path,max_count)
    # process_arxiv_ids(input_path, output_path)