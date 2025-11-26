from uuid import uuid4
from typing import Dict, Callable
from openai import OpenAI
import json
import requests
import time
import traceback
from utils.llm_agent.tools.tool_manager import StreamToolManager, execute_code
from utils.save_utils import MarkdownWriter

# OpenAI API配置
base_url = "http://123.129.219.111:3000/v1"
api_key = "sk-hnHVnUfqxBsPjGmC7EYAu5TWp9m31YqUJlPga0zulaqWRPTA"


test_tools = [
    {
        "type": "function",
        "function": {
            "name": "Python_code_interpreter",
            "description": "execute python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "python code to execute",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]

def execute_python_code(code: str) -> str:
    """在沙盒中执行Python代码"""
    tool_manager = StreamToolManager(url="http://10.200.0.53:30019", session_id=str(uuid4()), timeout=1800)
    outputs, tool_stats = execute_code(code, tool_manager)
    return outputs


tool_functions = {
    "Python_code_interpreter":execute_python_code
}


def call_model_wo_tools(
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gpt-5",
    markdown_writer: MarkdownWriter = None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    final_content = completion.choices[0].message.content.strip()
    if markdown_writer:
        markdown_writer.write_to_markdown(final_content, 'agent_result')
    return final_content


def call_model(
    system_prompt: str,
    user_prompt: str,
    tools: list = [],
    tool_functions: Dict[str, Callable] = {},
    model_name: str = "gpt-5",
    max_tool_calls: int = 20,
    markdown_writer: MarkdownWriter = None,
) -> str:
    """调用GPT模型并处理工具调用
    
    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        tools: 可用的工具列表
        tool_functions: 工具名称到实现函数的映射
        model_name: 模型名称
        max_tool_calls: 最大工具调用次数
        
    Returns:
        str: 模型的最终回答
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for _ in range(max_tool_calls):
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
        )
        
        messages.append(completion.choices[0].message)
        
        # save completion content
        if markdown_writer and completion.choices[0].message.content:
            markdown_writer.write_to_markdown(completion.choices[0].message.content, 'agent_result')
        
        if completion.choices[0].finish_reason == 'stop':
            break
            
        tool_call_list = completion.choices[0].message.tool_calls
        
        # save tool call arguments
        if markdown_writer and completion.choices[0].message.tool_calls:
            assert len(completion.choices[0].message.tool_calls) == 1, "Only one tool call is supported"
            
            if len(tools) == 1: 
                call_args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
                call_code = call_args.get("code")
                markdown_writer.write_to_markdown(call_code, 'julia_call')
            
            elif len(tools) == 2: 
                function_name = completion.choices[0].message.tool_calls[0].function.name
                call_args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
                query = call_args.get("query")
                markdown_writer.write_to_markdown(query, function_name)
            else:
                raise ValueError(f"Unsupported tools length: {len(tools)}")
            

        for tool_call in tool_call_list:
            try:
                call_args = json.loads(tool_call.function.arguments)
                result = tool_functions[tool_call.function.name](**call_args)

                if markdown_writer:
                                        
                    if len(tools) == 1: 
                        markdown_writer.write_to_markdown(result, 'julia_result')
                    elif len(tools) == 2: 
                        pass
                    else:
                        raise ValueError(f"Unsupported tools length: {len(tools)}")
                    
            except Exception as e:
                result = traceback.format_exc()

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    final_content = messages[-1].content.strip()
    
    if markdown_writer:
        if len(tools) == 1: # experimentalist
            pass
        elif len(tools) == 2: # supervisor
            markdown_writer.write_to_markdown(final_content, 'supervisor_text')
        else:
            raise ValueError(f"Unsupported tools length: {len(tools)}")
    
    return final_content


if __name__ == '__main__':
    
    system_prompt = "You are a helpful assistant."
    user_prompt = "calculate 89089*89089*0.678 with the help of python"
    
    result = call_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=test_tools,
        tool_functions=tool_functions,
        model_name="gpt-5",
        max_tool_calls=20,
    )
    print(result)
        