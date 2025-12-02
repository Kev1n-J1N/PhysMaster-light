import json
import time
from typing import List, Optional, Dict, Any

class MetaInfoClient:
    def __init__(self,data_path):
        self.data_path = data_path
        self.dataset = self.load_dataset()

    def load_dataset(self) -> Dict[str, Dict[str, Any]]:
        """加载整个数据集到内存，并构建 小写title -> record 的映射"""
        title_to_record = {}
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                title = record.get("title")
                if title is None:
                    continue  # 跳过没有 title 的记录
                title_lower = title.lower()
                title_to_record[title_lower] = record
        return title_to_record

    def clear_title(self, title: str) -> str:
        """清除标题中的前导/尾随空格和换行符"""
        title = title.replace("\n ", "")
        return title.strip()

    def get_paper_by_title(self, titles: List[str]) -> List[Dict[str, Any]]:
        """根据标题列表（大小写不敏感）查询论文记录。
        
        对每个标题：
          - 若找到，返回原记录 + {"in_dataset": True}
          - 若未找到，返回 {"in_dataset": False}
        """
        results = []
        for title in titles:
            key = self.clear_title(title).lower()
            record = self.dataset.get(key)
            if record is not None:
                result_record = dict(record)
                result_record["title"] = title
                result_record["in_dataset"] = True
                results.append(result_record)
            else:
                results.append({"in_dataset": False})
        return results

if __name__ == "__main__":
    client = MetaInfoClient()
    t = time.time()
    records = client.get_paper_by_title(['Attention Is All You Need','Attention Is All You Need','Attention Is All You Need','Attention Is All You Need','Attention Is All You Need'])
    print("Query time:", time.time() - t)
    print(records[0] if records else "Not found")