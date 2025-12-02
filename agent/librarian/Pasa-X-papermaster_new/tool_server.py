from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn



# 创建 FastAPI 实例
app = FastAPI()

# 定义请求参数的 Pydantic 模型（如果需要）
class ContentRequest(BaseModel):
    url: str

# 路由：POST 请求
@app.post("/get-section")
async def get_content_by_body(request: ContentRequest):
    from search_utils.optim_utils import search_content_by_id
    arxiv_id = request.url.split("/")[-1]
    content = search_content_by_id(arxiv_id)
    return {"arxiv_id": arxiv_id, "content": content}

# main 函数，应用部署
def main():
    # 使用 uvicorn 启动 FastAPI 应用
    uvicorn.run(app, host="0.0.0.0", port=1237)

# 如果这个脚本是作为主程序运行，执行 main 函数
if __name__ == "__main__":
    main()
