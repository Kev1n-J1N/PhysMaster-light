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
        self.headers = {"Content-Type": "application/json"}

    def execute_tool(self, code:str, lang:str="python"):
        # 只有 Python 代码才注入 tools
        payload_code = code if lang != "python" else ("from tools import *\n" + code)

        payload = {
            "code": payload_code,
            "lang": lang
        }
        resp = requests.post(
            f"{self.server_url}/execute",
            headers=self.headers,
            json=payload
        )
        return resp.json()

class StreamToolManager(BaseToolManager):
    def __init__(self, url, session_id:str = None, timeout:int=180):
        super().__init__(url)
        self.session_id = str(uuid4()) if not session_id else session_id
        self.headers['session_id'] = self.session_id  # 修正：确保一定有值
        self.timeout = timeout

    async def submit_task(self, code:str, lang:str="python"):
        submit_url = f"{self.server_url}/submit"
        payload_code = code if lang != "python" else ("from tools import *\n" + code)
        payload = {
            "code": payload_code,
            "lang": lang,
            "session_id": self.session_id,
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

    async def recieve_task_process(self):
        recieve_url = f"{self.server_url}/get_mcp_result/{self.session_id}"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", recieve_url, headers=self.headers) as response:
                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    data = json.loads(line)
                    yield data
                    if (not data.get("sub_stream_type")) and (data.get("stream_state") == "end"):
                        break    

    async def execute_code_async_stream(self, code: str, lang:str="python"):
        submit_status = await self.submit_task(code, lang=lang)
        if submit_status.get("status") == "fail":
            yield {"output": "", "error": submit_status}
            return
        async for item in self.recieve_task_process():
            yield item

    async def execute_code_async_resonly(self, code:str, lang:str="python"):
        submit_status = await self.submit_task(code, lang=lang)
        if submit_status.get("status") == "fail":
            return {"output":"code submit fail", "error": submit_status}
        return_value = {"output":""}
        async for item in self.recieve_task_process():
            if item.get("main_stream_type") == "code_result":
                return_value = {"output": item.get('content', '')}
        return return_value

def extract_from_stream(stream_data):
    if stream_data.get("main_stream_type") == "code_result":
        return True, stream_data.get("content", "")
    elif stream_data.get("main_stream_type") == "tool_result":
        other_info = stream_data.get("other_info", {})
        if "tool_name" in other_info:
            return False, None
        elif "call_tool" in other_info:
            return False, (other_info["call_tool"], other_info.get("call_args", {}))
    return False, None

async def execute_code_and_analyze(code: str, tool_manager: StreamToolManager, lang:str="python"):
    tool_calls = {}
    code_outputs = []
    async for item in tool_manager.execute_code_async_stream(code, lang=lang):
        is_code_result, info = extract_from_stream(item)
        if is_code_result:
            code_outputs.append(info)
        elif info is not None:
            tool_name, args = info
            if tool_name not in tool_calls:
                tool_calls[tool_name] = {'count': 1, 'args': [args]}
            else:
                tool_calls[tool_name]['count'] += 1
                tool_calls[tool_name]['args'].append(args)
    return '\n'.join(code_outputs), tool_calls

def execute_code(code: str, tool_manager: StreamToolManager, lang:str="python"):
    async def _run():
        return await execute_code_and_analyze(code, tool_manager, lang=lang)
    return asyncio.run(_run())

if __name__ == '__main__':
    tool_manager = StreamToolManager(url="http://10.200.0.53:30019", session_id="bbbbb", timeout=1800)

    # 跑 JULIA：注意 lang="julia"，并且不再拼 Python 的 import
    julia_code = """
println("Hello, Julia!")
1 + 2 * 3
"""
    outputs, tool_stats = execute_code(julia_code, tool_manager, lang="julia")
    print("\n=== output (julia) ===")
    print(outputs)
