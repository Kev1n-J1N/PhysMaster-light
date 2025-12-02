import time
import re
import sys,os
import secrets
from typing import *
import fitz
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,".."))
from create_dataset.paper_download import paper_downloader


def is_title(text: str):
    title_keywords = ["Abstract", "Introduction", "Conclusion", "Methodology", "Results", "Discussion"]
    heading_pattern = re.compile(
        r'^(\s*)((?:\d+\.)*\d+\.?)(\s*[A-Z][a-zA-Z\s\-]*)',  # 匹配数字编号标题
        flags=re.MULTILINE
    )
    stripped_text = text.strip()
    parts = stripped_text.split()
    # 不超过 5 个单词、首字母大写、包含特定关键词
    if len(parts) < 5 and stripped_text and stripped_text[0].isupper():
        for key in title_keywords:
            if key in stripped_text:
                return True, "0.", stripped_text
    else:
        match = heading_pattern.match(text)
        if match:
            _, number, title = match.groups()
            if number.count('.') == 0:
                number += '.'
            return True, number, title
    return False, "", ""

def pdf_reader(pdf_path: str) -> Dict[str, str]:
    """
    解析PDF返回结构化字典
    :param pdf_path: 文件路径
    :return: {章节标题: 内容}
    """
    doc = fitz.open(pdf_path)
    sections = {}
    current_hierarchy = {}     # 当前层级
    current_content = []       # 当前内容
    current_key = "Paper Information"

    for page in doc:
        blocks = page.get_text("blocks",sort=True)
        for block in blocks:
            # 取块内容
            x0, y0, x1, y1, text, _, block_type = block
            text = text.strip()
            if not text or y0 < 50 or y1 > page.rect.height - 50 or bool(block_type):
                continue

            match, number, title = is_title(text)
            if match:
                # 提取层级
                level = number.count('.')  # 点判断
                full_title = f"{number} {title}".strip().replace("\n", "")
                current_hierarchy[level] = full_title
                # 清除低层
                for l in list(current_hierarchy.keys()):
                    if l > level:
                        del current_hierarchy[l]
                # 构建标题
                sorted_levels = sorted(current_hierarchy.items())
                compound_key = ' '.join([v for _, v in sorted_levels])
                # 保存前节
                if current_key and current_content:
                    sections[current_key] = "\n".join(current_content)
                # 初始化节
                current_key = compound_key
                current_content = []
                continue

            if current_key:
                cleaned_text = re.sub(r'(\w-) ?\n(\w)', r'\1\2', text)
                current_content.append(cleaned_text)

    if current_key and current_content:
        sections[current_key] = '\n'.join(current_content)

    return sections



if __name__ == "__main__":
    t1 = time.time()

    url = "https://arxiv.org/abs/2501.04519"
    pdf_path = "./test.pdf"
    pdf_path = paper_downloader(url,f'{current_dir}/pdf_temp/')
    
    text = ""
    extracted_text = pdf_reader(pdf_path)
    print(extracted_text)