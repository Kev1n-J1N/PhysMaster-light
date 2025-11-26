import requests
import os
import json
import aiohttp
import httpx
from uuid import uuid4
import asyncio


class BaseToolManager:
    def __init__(self, url:str):
        self.server_url = url
        self.headers = {
            "Content-Type": "application/json"
        }

    def execute_tool(self, tool_call:str):
        
        # os.environ["http_proxy"] = ""
        
        tool_call = "from tools import *\n" + tool_call
        payload = {
            "code":tool_call
        }
        # print("execution...")
        resp = requests.post(
            f"{self.server_url}/execute",
            headers=self.headers,
            json=payload
        )
        
        # os.environ["http_proxy"] = "http://127.0.0.1:7890"
        # print(resp)
        return resp.json()


class StreamToolManager(BaseToolManager):
    def __init__(self, url, session_id:str = None, timeout:int=180):
        super().__init__(url)
        self.session_id = str(uuid4()) if not session_id else session_id
        self.headers['session_id'] = session_id
        # self.session_id = str("test_id2")
        self.timeout = timeout

    async def submit_task(self, code:str):
        submit_url = f"{self.server_url}/submit"

        payload = {
            "code":code,
            "session_id":self.session_id,
            "timeout": self.timeout
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    submit_url,
                    headers=self.headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        return {"status": "fail", "status_code": resp.status}
            except Exception as e:
                return {"status": "fail", "error":f"{e}"}


    async def recieve_task_process(self, ):
        recieve_url = f"{self.server_url}/get_mcp_result/{self.session_id}"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", recieve_url, headers=self.headers) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    # print("Received:", data)
                    # print(data['content'], end="", flush=True)
                    # print(data['content'], data['stream_state'])


                    if (not data.get("sub_stream_type")) and (data.get("stream_state") == "end"):
                        yield data
                        break    
                    else:
                        yield data
                        
    async def execute_code_async_stream(self, tool_call: str,):
        submit_status = await self.submit_task(tool_call)
        if submit_status["status"] == "fail":
            yield {"output":""}
            return
        
        # await self.recieve_task_process()
        # print('start recieve')
        async for item in self.recieve_task_process():
            yield item

    async def execute_code_async_resonly(self, tool_call:str):
        submit_status = await self.submit_task(tool_call)
        if submit_status["status"] == "fail":
            return {"output":"code submit fail"}
        
        # await self.recieve_task_process()
        async for item in self.recieve_task_process():
            if item["main_stream_type"] == "code_result":
                return_value = {"output":item['content']}
            
             
        return return_value

    async def close_session(self):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/del_session",
                params={"session_id": self.session_id},
                headers=self.headers
            )
            return resp.json()


# class mySearchToolManager(BaseToolManager):
#     def __init__(self, url:str, session_id:str):
#         super().__init__(url)
#         self.session_id = session_id

#     def execute_tool(self, tool_call:str):
#         # print(f"session_id: {self.session_id}")
#         tool_call = "from tools import *\n" + tool_call
#         payload = {
#             "code":tool_call,
#             "session_id":self.session_id,
#             "timeout":600
#         }
#         # print("execution...")
#         resp = requests.post(
#             f"{self.server_url}/execute",
#             headers=self.headers,
#             json=payload
#         )

#         return resp.json()

#     def close_session(self):
#         resp = requests.post(
#             f"{self.server_url}/del_session",
#             params={"session_id":self.session_id}
#         )
#         return resp.json()


def extract_from_stream(stream_data):
    """从流式数据中提取信息
    
    Args:
        stream_data (dict): 流式数据字典
        
    Returns:
        tuple: (是否是代码结果, 工具调用信息)
            - 如果是代码结果，返回(True, 打印内容)
            - 如果是工具调用，返回(False, (工具名, 参数))
    """
    if stream_data.get("main_stream_type") == "code_result":
        # 处理代码打印的结果
        return True, stream_data.get("content", "")
    
    elif stream_data.get("main_stream_type") == "tool_result":
        # 处理工具调用信息
        other_info = stream_data.get("other_info", {})
        if "tool_name" in other_info:
            # 工具开始调用
            return False, None
        elif "call_tool" in other_info:
            # 工具调用的具体参数
            return False, (other_info["call_tool"], other_info.get("call_args", {}))
    
    return False, None

async def execute_code_and_analyze(code: str, tool_manager: StreamToolManager):
    """执行代码并分析结果（异步版本）
    
    Args:
        code (str): 要执行的代码
        tool_manager (StreamToolManager): 工具管理器实例
        
    Returns:
        tuple: (代码输出字符串, 工具调用统计字典)
    """
    tool_calls = {}  # 记录工具调用次数
    code_outputs = []  # 记录代码输出
    
    async for item in tool_manager.execute_code_async_stream(code):
        is_code_result, info = extract_from_stream(item)
        
        if is_code_result:
            code_outputs.append(info)
        elif info is not None:
            tool_name, args = info
            if tool_name not in tool_calls:
                tool_calls[tool_name] = {}
                tool_calls[tool_name]['count'] = 1
                tool_calls[tool_name]['args'] = [args]
            else:
                tool_calls[tool_name]['count'] += 1
                tool_calls[tool_name]['args'].append(args)
    
    return '\n'.join(code_outputs), tool_calls

def execute_code(code: str, tool_manager: StreamToolManager):
    """执行代码并分析结果（同步版本）
    
    Args:
        code (str): 要执行的代码
        tool_manager (StreamToolManager): 工具管理器实例
        
    Returns:
        tuple: (代码输出字符串, 工具调用统计字典)
    """
    async def _run():
        return await execute_code_and_analyze(code, tool_manager)
    
    return asyncio.run(_run())

if __name__ == '__main__':
    # 测试同步版本
    tool_manager = StreamToolManager(url="http://10.200.0.53:30019", session_id="bbbbb", timeout=1800)
    test_code_0 = """
println("Hello, Julia!")
1 + 2 * 3
"""
    outputs, tool_stats = execute_code(test_code_0, tool_manager)
    
    print("\n=== output ===")
    print(outputs)
        
    # print("\n=== tool call stats ===")
    # for tool_name, tool_info in tool_stats.items():
    #     print(f"{tool_name}: called {tool_info['count']} times")
    #     print(f"args: {tool_info['args']}")
