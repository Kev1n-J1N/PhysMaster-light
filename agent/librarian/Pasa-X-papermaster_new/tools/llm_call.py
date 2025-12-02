# openai调用模型回答
import json

import openai

model_config = {
    "gpt-4o": {
        "base_url": "http://123.129.219.111:3000/v1",
        "api_key": "sk-Scr7YbNcHRq35qW1HpRe7DnF8jVuUguF8PPMcKsbhGGrRCjA",
    },
    "qwen-72b": {
        "base_url": "http://127.0.0.1:30020/v1",
        "api_key": "EMPTY",
    },
        "qwen3-30b": {
        "base_url": "http://9.3.37.142:30010/v1",
        "api_key": "EMPTY",
    },

    "deepseek-r1":{
            "base_url": "http://127.0.0.1:8888/v1",
            "api_key": "EMPTY"
        },
    "glm":{
        "base_url": "http://127.0.0.1:8889/v1",
        "api_key": "EMPTY"
    }
}

def llm_call(query: str, model_name: str = "qwen3-30b"):
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        timeout=60
    )
    return response.choices[0].message.content

def get_title_from_reference(reference: str):
    query = f"Get the title from the following reference: {reference}, and return the title in a json format:{{'title': 'title'}}.For example, if the reference is '1. [1] Smith, J. (2021). A study on the effects of AI on society. Journal of AI Research, 10(2), 1-10.', you should return {{'title': 'A study on the effects of AI on society'}}."
    # try:
    #     response = llm_call(query, model_name="gpt-4.1-nano-2025-04-14")
    # except:
    #     response = llm_call(query, model_name="qwen-max")
    response = llm_call(query, model_name="qwen-72b")
    json_begin = response.find("{")
    json_end = response.rfind("}") + 1
    json_str = response[json_begin:json_end].replace("'", '"')
    try:
        result = json.loads(json_str)
        return result["title"]
    except:
        return ""

def conclude_abstract(abstract: str,question: str):
    query = f"Please conclude the abstract of the following paper according to the question: {question}, and return the conclusion in a json format:{{'conclusion': 'conclusion'}}. Your answer should be concise and to the point, and should not be too long. You can conclude it in different aspects, such as the problem, the solution, the result, the limitation, etc. Abstract: {abstract}."
    
    try:
        response = llm_call(query, model_name="qwen-72b")
    except:
        response = llm_call(query, model_name="gpt-4.1-nano-2025-04-14")
    json_begin = response.find("{")
    json_end = response.rfind("}") + 1
    json_str = response[json_begin:json_end].replace("'", '"')
    try:
        result = json.loads(json_str)
        return result["conclusion"]
    except:
        return ""

if __name__ == "__main__":
    # import time
    # import concurrent.futures
    # query = "你好，世界"
    # begin = time.time()
    # num = 500
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num) as executor:
    #     futures = [executor.submit(llm_call, query, model_name="gpt-4.1-nano-2025-04-14") for _ in range(num)]
    #     results = [future.result() for future in concurrent.futures.as_completed(futures)]
    # end = time.time()
    # print(f"Time taken: {end - begin} seconds")
    user_query = "Our team’s code helper looks great when we test it on tiny, self‑contained snippets with simple inputs, but it struggles when we ask it to close real tickets that touch multiple files and need the project to be run to confirm the fix. We want a fair way to judge whether it’s actually improving at these day‑to‑day fixes, not just the toy checks. What kind of test setup should we build so it can try changes, run checks safely, and report results, and how do we compare its progress on these tougher tasks against the easier ones we currently use?"
    query = f"You are an academic expert, and you need to carefully analyze and understand the user's question. Upon seeing this question, identify the core keywords for academic paper retrieval, providing 3-7 professional terms along with their respective weights. Ensure the keywords are as much as possible **academic, conceptual terms** rather than general words. The more closely the keyword is related to the user’s query, the higher the weight, which should be between 0 and 1. Based on this, generate search queries for literature that will help solve the user's problem, ensuring the queries are concise and focused. **Research Question**: {user_query}\n\nReturn in strict JSON format: \n{{\n  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"],\n  \"weights\": [0.9, 0.7, 0.5],\n  \"queries\": [\"search query 1\", \"search query 2\", \"search query 3\"]\n}}.\nNote that the order of keywords and weights should correspond one-to-one."
    print(llm_call(query, model_name="deepseek-r1"))
    # abstract = """ 'The demand for synthetic data in mathematical reasoning has increased due to its potential to enhance the mathematical capabilities of large language models (LLMs). However, ensuring the validity of intermediate reasoning steps remains a significant challenge, affecting data quality. While formal verification via theorem provers effectively validates LLM reasoning, the autoformalisation of mathematical proofs remains error-prone. In response, we introduce iterative autoformalisation, an approach that iteratively refines theorem prover formalisation to mitigate errors, thereby increasing the execution rate on the Lean prover from 60% to 87%. Building upon that, we introduce Theorem Prover as a Judge (TP-as-a-Judge), a method that employs theorem prover formalisation to rigorously assess LLM intermediate reasoning, effectively integrating autoformalisation with synthetic data generation. Finally, we present Reinforcement Learning from Theorem Prover Feedback (RLTPF), a framework that replaces human annotation with theorem prover feedback in Reinforcement Learning from Human Feedback (RLHF). Across multiple LLMs, applying TP-as-a-Judge and RLTPF improves benchmarks with only 3,508 samples, achieving 5.56% accuracy gain on Mistral-7B for MultiArith, 6.00% on Llama-2-7B for SVAMP, and 3.55% on Llama-3.1-8B for AQUA.'"""
    # query = "LLM-as-judge synthetic data scoring"
    # print(conclude_abstract(abstract, query))
