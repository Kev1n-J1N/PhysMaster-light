import os
import json
import requests
from typing import List, Dict, Optional, Any
import time
import re
from rapidfuzz import process, fuzz

class JournalClient:
    def __init__(self, mode="scholar"):
        start_time = time.time()
        
        self.db = []
        with open("journal_metainfo_1029.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.db.append(json.loads(line))
        
        # 构建 choices: 仅字符串；同时维护一个索引到 item 的映射
        self.choices = []  # 用于匹配的别名列表（标准化后的字符串）
        self.choice_to_item = {}  # 映射: 标准化别名 -> 对应的 item
        for item in self.db:
            aliases = item.get("journal_alias", [])
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    continue
                norm_alias = self.preprocess_journal_name(alias)
                if norm_alias not in self.choice_to_item:
                    self.choices.append(norm_alias)
                    self.choice_to_item[norm_alias] = item
        
        print(f"✅ 加载 {len(self.db)} 条期刊信息，构建 {len(self.choices)} 个别名匹配项")

    def preprocess_journal_name(self, journal_name: str) -> str:
        journal_name = journal_name.lower()
        # 移除括号及其中的内容
        journal_name = re.sub(r"\([^)]*\)", "", journal_name)   # 圆括号
        # 移除四位数字年份
        journal_name = re.sub(r"\b\d{4}\b", "", journal_name)
        # 移除标点符号：. , - 等
        journal_name = journal_name.replace(".", "").replace("-", "").replace(",", "")
        # 清理多余空格并去除首尾空格
        journal_name = " ".join(journal_name.split()).strip()
        return journal_name
        
    def pair_journal_info(self, journal_name: str):
        if not journal_name or not self.db:
            return self._create_empty_journal_info()

        # 辅助函数
        letter_len = lambda s: sum(c.isalpha() for c in s)
        last_letter = lambda s: (
            (p[-1].upper() if (p := s.strip().split()) and len(p[-1]) == 1 and p[-1].isalpha() else None)
        )

        journal_name = self.preprocess_journal_name(journal_name)
        jl = letter_len(journal_name)
        suffix = last_letter(journal_name) if jl > 6 else None

        # 一次匹配：fuzz.ratio
        candidates = process.extract(
            journal_name, self.choices,
            scorer=fuzz.ratio, score_cutoff=92, limit=10
        )
        if not candidates:
            return self._create_empty_journal_info()

        # 过滤：短简称 & 系列后缀不一致
        filtered = []
        for text, score, _ in candidates:
            if jl >= 7 and letter_len(text) < 7:
                continue
            if suffix and last_letter(text) != suffix:
                continue
            filtered.append((text, score))
        if not filtered:
            return self._create_empty_journal_info()

        # 取最高分匹配
        best_text, best_score = max(filtered, key=lambda x: x[1])
        return self._extract_journal_info(self.choice_to_item[best_text], best_score)

    def _extract_journal_info(self, item, score):
        """
        从数据库项中提取需要的期刊信息
        """
        return {
            "paired_score": score,
            "paired_journal": item.get('journal_full'),
            "journal_alias": item.get("journal_alias", []),
            # "categories": item.get("categories"),
            # "domain": item.get("domain"),
            "h5_index": item.get("h5_index"),
            "h5_median": item.get("h5_median"),
            "h5_rank": item.get("h5_rank"),
            "IF": item.get("IF"),
            "CiteScore": item.get("CiteScore"),
            "CCF": item.get("CCF"),
            "CORE": item.get("CORE"),
            "JCR": item.get("JCR"),
            "CAS": item.get("CAS"),
        }

    def _create_empty_journal_info(self):
        """
        创建空期刊信息字典
        """
        return {
            "paired_score": 0,
            "paired_journal": None,
            "journal_alias": [],
            # "categories": None,
            # "domain": None,
            "h5_index": None,
            "h5_median": None,
            "h5_rank": None,
            "IF": None,
            "CiteScore": None,
            "CCF": None,
            "CORE": None,
            "JCR": None,
            "CAS": None,
        }
    
    def add_level(self, result):
        if result['paired_journal'] is None:
            return -1
        
        if result['paired_journal'] in ['Nature', 'Science', 'Cell', 'The New England Journal of Medicine', 'The Lancet']:
            return 1
        
        if result['h5_rank'] is not None:
            if result['h5_rank'] <= 3:
                return 2
            
        if result['h5_rank'] is not None:
            if result['h5_rank'] <= 5:
                return 3
        if result['CCF']=='A':
            return 3
        if result['CORE']=='A*' or result['CORE']=='A':
            return 3
        if result['IF'] is not None:
            if int(result['IF'])>=20:
                return 3
        if result['CiteScore']is not None:
            if result['CiteScore']>=20:
                return 3
        
        if result['CCF']=='B':
            return 4
        if result['CORE']=='B':
            return 4
        if result['CAS']=='1':
            return 4
        if result['h5_rank'] is not None:
            if result['h5_rank'] <= 10:
                return 4

        return 5