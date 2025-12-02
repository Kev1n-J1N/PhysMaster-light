import json
from typing import List, Dict


def load_graph_first_line(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline().strip())
    return first["graph"]


def test_graph_size(path) -> Dict[str, List[str]]:
    graph = load_graph_first_line(path)
    id_list = set()
    for e in graph:
        to_id = e["to"]
        id_list.add(to_id)
        
    return len(id_list)

if __name__ == "__main__":
    path = "/data/tingchia/Pasa-X/create_dataset/pasamaster/pasamaster_bench/output/20251009/cite_graphs.jsonl"
    size = test_graph_size(path)
    print(size)