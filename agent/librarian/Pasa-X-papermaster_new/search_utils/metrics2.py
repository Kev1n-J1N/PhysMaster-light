import json
import glob
from tqdm import tqdm

def get_metric(d, key):
    # 安全获取指标
    if key in d:
        return d[key].get('call', 0), d[key].get('success', 0)
    return 0, 0

output_folder = 'results/pasa_crawer_pasa_selector_20250702'
pred_files = [f for f in glob.glob(output_folder + "/mtr_*.json")]

batch_time_list = []
keyminder_time_list = []
search_time_list = []
locater_time_list = []
ranker_time_list = []
get_content_time_list = []

sum_metrics = {
    'ti2id': {'call': 0, 'success': 0},
    'id2ab': {'call': 0, 'success': 0},
    'id2ref': {'call': 0, 'success': 0},
}

for pred_file in tqdm(pred_files):
    with open(pred_file) as f:
        data = json.load(f)
    batch_time = data.get('batch_time', 0)
    batch_time_list.append(batch_time)
    keyminder_time = data.get('keyminder', {}).get('avg_time', 0)
    keyminder_time_list.append(keyminder_time)
    search_time = data.get('search', {}).get('avg_time', 0)
    search_time_list.append(search_time)
    locater_time = data.get('locater', {}).get('avg_time', 0)
    locater_time_list.append(locater_time)
    ranker_time = data.get('ranker', {}).get('avg_time', 0)
    ranker_time_list.append(ranker_time)
    get_content_time = data.get('get_content', {}).get('avg_time', 0)
    get_content_time_list.append(get_content_time)

    new = data.get('new', {})
    for k in sum_metrics.keys():
        call, success = get_metric(new, k)
        sum_metrics[k]['call'] += call
        sum_metrics[k]['success'] += success

num_files = len(batch_time_list)
batch_time_avg = sum(batch_time_list) / num_files if num_files > 0 else 0
keyminder_time_avg = sum(keyminder_time_list) / num_files if num_files > 0 else 0
search_time_avg = sum(search_time_list) / num_files if num_files > 0 else 0
locater_time_avg = sum(locater_time_list) / num_files if num_files > 0 else 0
ranker_time_avg = sum(ranker_time_list) / num_files if num_files > 0 else 0
get_content_time_avg = sum(get_content_time_list) / num_files if num_files > 0 else 0

print(f"=== batch_time 统计 ===")
print(f"avg={batch_time_avg:.4f}, num_files={num_files}")
print(f"=== keyminder_time 统计 ===")
print(f"avg={keyminder_time_avg:.4f}, num_files={num_files}")
print(f"=== search_time 统计 ===")
print(f"avg={search_time_avg:.4f}, num_files={num_files}")
print(f"=== locater_time 统计 ===")
print(f"avg={locater_time_avg:.4f}, num_files={num_files}")
print(f"=== ranker_time 统计 ===")
print(f"avg={ranker_time_avg:.4f}, num_files={num_files}")
print(f"=== get_content_time 统计 ===")
print(f"avg={get_content_time_avg:.4f}, num_files={num_files}")

print("=== ti2id/id2ab/id2ref 成功率统计 ===")
for k, v in sum_metrics.items():
    call = v['call']
    success = v['success']
    rate = success / call if call > 0 else 0
    avg_call = call / num_files if num_files > 0 else 0
    avg_success = success / num_files if num_files > 0 else 0
    print(f"{k}: call_sum={call}, success_sum={success}, rate={rate:.4f}, avg_call={avg_call:.2f}, avg_success={avg_success:.2f}")
