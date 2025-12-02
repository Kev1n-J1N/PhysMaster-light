import time

import requests

url = "http://192.168.201.34:30018/pasa"


headers = {
    "Content-Type": "application/json"
}
begin_time = time.time()
response = requests.post(url, headers=headers, json={"question": "LLM Agents, reinforcement learning"})
end_time = time.time()
print(f"Time taken: {end_time - begin_time} seconds")
print(response.json()["result"])