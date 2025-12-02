import json
import time
from typing import List, Optional, Dict, Any
import os
current_path = os.path.dirname(os.path.realpath(__file__))
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn,time
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/config.json")
with open(config_path, "r", encoding="utf-8") as f:
    data_path = json.load(f)["meta_info_path"]

class Request(BaseModel):
    titles: list[str]

app = FastAPI()

    

class MetaInfoClient:
    def __init__(self):
        self.data_path = data_path
        self.dataset = self.load_dataset()

    def load_dataset(self) -> Dict[str, Dict[str, Any]]:
        """加载整个数据集到内存，并构建 小写title -> record 的映射"""
        title_to_record = {}
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line. strip()
                if not line:
                    continue
                record = json.loads(line)
                title = record.get("title")
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
    
client = MetaInfoClient()

@app.post("/search_meta_info")
async def search_meta_info(request: Request):
    begin = time.time()
    records = client.get_paper_by_title(request.titles)
    end = time.time()
    return {"records": records, "time": end-begin}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1111)