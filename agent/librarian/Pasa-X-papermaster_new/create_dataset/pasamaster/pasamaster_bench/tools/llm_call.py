# openai调用模型回答
import json

import openai

model_config = {
    "gpt-4o": {
        "base_url": "http://123.129.219.111:3000/v1",
        "api_key": "sk-Scr7YbNcHRq35qW1HpRe7DnF8jVuUguF8PPMcKsbhGGrRCjA",
    },
    "qwen-72b": {
        "base_url": "http://198.18.148.100:30020/v1",
        "api_key": "EMPTY",
    },
    "qwen3-30b": {
        "base_url": "http://127.0.0.1:30010/v1",
        "api_key": "EMPTY",
    },
    "deepseek-r1":{
            "base_url": "http://127.0.0.1:8888/v1",
            "api_key": "EMPTY"
        },
}

def llm_call(query: str, model_name: str = "qwen-72b"):
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        timeout=60
    )
    return response.choices[0].message.content


if __name__ == "__main__":
   
    user_query = "Our team’s code helper looks great when we test it on tiny, self‑contained snippets with simple inputs, but it struggles when we ask it to close real tickets that touch multiple files and need the project to be run to confirm the fix. We want a fair way to judge whether it’s actually improving at these day‑to‑day fixes, not just the toy checks. What kind of test setup should we build so it can try changes, run checks safely, and report results, and how do we compare its progress on these tougher tasks against the easier ones we currently use?"
    query = f"You are an academic expert, and you need to carefully analyze and understand the user's question. Upon seeing this question, identify the core keywords for academic paper retrieval, providing 3-7 professional terms along with their respective weights. Ensure the keywords are as much as possible **academic, conceptual terms** rather than general words. The more closely the keyword is related to the user’s query, the higher the weight, which should be between 0 and 1. Based on this, generate search queries for literature that will help solve the user's problem, ensuring the queries are concise and focused. **Research Question**: {user_query}\n\nReturn in strict JSON format: \n{{\n  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"],\n  \"weights\": [0.9, 0.7, 0.5],\n  \"queries\": [\"search query 1\", \"search query 2\", \"search query 3\"]\n}}.\nNote that the order of keywords and weights should correspond one-to-one."
    print(llm_call(query, model_name="qwen-72b"))
   