# openai调用模型回答
import openai

model_config = {
    "gpt-4o": {
        "base_url": "http://123.129.219.111:3000/v1",
        "api_key": "sk-rzDP0cqefadoHtHcNKm5waIKbxkxucIcW70Pqwl4fmdStJaB",
    },
    "qwen-72b": {
        "base_url": "http://192.168.215.44:30020/v1",
        "api_key": "EMPTY",
    },
    "qwen-turbo": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-f657a94c48944fb2a3cd35f60d94c668",
    }
}

def llm_call(query: str, model_name: str = "qwen-72b"):
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = "你好，世界"
    print(llm_call(query, model_name="qwen-turbo"))