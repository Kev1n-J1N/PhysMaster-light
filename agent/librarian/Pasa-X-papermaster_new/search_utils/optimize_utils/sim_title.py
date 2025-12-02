import re
import unicodedata


def normalize_title(text):
    """将标题转换为纯字符"""
    text = unicodedata.normalize('NFKC', text) # Unicode标准化
    text = text.lower() # 转小写
    text = re.sub(r'[^a-z0-9]', '', text) # 只保留字母和数字
    return text

def get_main_title(title):
    """按冒号或破折号分割，取主标题"""
    return re.split(r'[:\-]', title, 1)[0].strip()

def jaccard_similarity(a, b):
    """jaccard相似度"""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0
    return len(set_a & set_b) / len(set_a | set_b)


if __name__ == "__main__":
    pass