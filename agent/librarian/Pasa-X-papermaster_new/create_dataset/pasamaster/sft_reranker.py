import os,sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, '../..'))
from tools.llm_call import *
from search_utils.optim_utils import search_abs_by_id
from tqdm import tqdm

with open(f'{current_path}/../../papermaster_prompt.json', 'r') as f:
    prompt_json = json.load(f)
    
ranker_prompt = prompt_json['ranker']['keyword_relevance']
num = 20
threads = 10
base_path = "/data/duyuwen/Pasa-X/result/result_APASbech_comp_train_20250905/record"
record_files = sorted(os.listdir(base_path), key=lambda x: int(x.split('.')[0]))
query_path = "/data/duyuwen/Pasa-X/data/APAS_bench/APASbech_comp_train_converted.jsonl"
with open(query_path, 'r') as f:
    queries = [json.loads(line) for line in f]

save_path = f"{current_path}/ranker_result/sft_ranker.jsonl"
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

for (record_file,query) in tqdm(zip(record_files,queries)):
    with open(os.path.join(base_path, record_file), 'r') as f:
        papers = json.load(f)['all_papers']
    prompt_list = []
    gt_papers = query['answer']
    gt_arxiv_id = query["answer_arxiv_id"]
    ######################################## gt paper #######################################
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(search_abs_by_id, gt_arxiv_id))
    results = [result for result in results if result is not None]
    for result in results:
        try:
            title = result['title']
            abstract = result['abstract']
            user_query = query['question']
            prompt = ranker_prompt.format(user_query=user_query, title=title, abstract=abstract)
            prompt_list.append(prompt)
        except:
            import pdb;pdb.set_trace()
    ######################################## expanded papers #####################################
    for paper in papers:
        try:
            title = paper['title']
            abstract = paper['abstract']
            user_query = query['question']
            prompt = ranker_prompt.format(user_query=user_query, title=title, abstract=abstract)
            prompt_list.append(prompt)
        except:
            import pdb;pdb.set_trace()
            pass
    import concurrent.futures
    print(f"record_file: {record_file}, prompt_list length: {len(prompt_list)}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        results = list(tqdm(
            executor.map(llm_call, prompt_list),
            total=len(prompt_list),
            desc="Processing Prompts"
        ))
    import pdb;pdb.set_trace()
    results = [result for result in results if result is not None]
    final_results = [[],[],[]]
    for result,prompt in zip(results,prompt_list):
        score = int(result.split('Reason')[0].split(':')[-1].strip())
        res = {
            "instruction":prompt,
            "input":"",
            "output":result
        }
        final_results[score-1].append(res)
    score_1_len,score_2_len,score_3_len = len(final_results[0]),len(final_results[1]),len(final_results[2])
    print(f"record_file: {record_file}, score_1: {score_1_len}, score_2: {score_2_len}, score_3: {score_3_len}")
    import pdb;pdb.set_trace()
    final_len = min(score_1_len,score_2_len,score_3_len)
    final_len = min(final_len,num)  
    final_results = [final_results[0][:final_len],final_results[1][:final_len],final_results[2][:final_len]]
    for result in final_results:
        for res in result:
            with open(save_path, 'a') as f:
                f.write(json.dumps(res) + '\n')
    
