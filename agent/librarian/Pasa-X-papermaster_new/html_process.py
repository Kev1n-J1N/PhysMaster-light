import time
import sys
import json
from typing import List, Generator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger

from utils import parse_html


def gen_file_list(data_path: str, file_list_path: str, max_depth: int = 2):
    root = Path(data_path)
    root_depth = len(root.parts)
    with open(file_list_path, "w") as f:
        for dirpath, dirnames, filenames in os.walk(root):
            cur_path = Path(dirpath)
            depth = len(cur_path.parts) - root_depth  # 相对 root 的深度

            # 超过 max_depth 就不再往下递归
            if depth >= max_depth:
                dirnames[:] = []  # 清空 dirnames，os.walk 就不会再往下了

            for name in filenames:
                if not name.endswith(".html"):
                    continue
                p = cur_path / name
                cnt += 1
                f.write(str(p) + "\n")
                if cnt % 10000 == 0:
                    print("------------->", cnt)
        print("total:", cnt)


def gen_file_list_from_file(
    file_list_path: str, parts: int = 3, parts_num: int = 0, batch_size: int = 10000
) -> Generator[List[str]]:
    with open(file_list_path, "r") as f:
        file_list = f.readlines()
    # 将file_list分成parts份， 取第parts_num份
    number_each_part = len(file_list) // parts
    start_index = number_each_part * parts_num
    end_index = start_index + number_each_part
    file_list = file_list[start_index:end_index]
    for i in range(0, len(file_list), batch_size):
        yield file_list[i : i + batch_size]


def single_file_process(html_files: List[str], target_path: str) -> bool:
    t0 = time.time()
    success_num = 0
    for html_file in html_files:
        try:
            html_file = Path(html_file)
            file_name = html_file.stem
            parent_name = html_file.parent.name
            with open(html_file, "r") as f:
                html_content = f.read()
            try:
                data = parse_html(html_content)
            except Exception as e:
                logger.exception(f"Error parsing {html_file}: {e}")
                continue
            save_path = Path(f"{target_path}/{parent_name}")
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / f"{file_name}.json", "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            success_num += 1
        except Exception as e:
            logger.exception(f"Error parsing {html_file}: {e}")
    time_cost = time.time() - t0
    logger.info(
        f"Time cost: {time_cost:.2f}s, Success num: {success_num}, Ave speed: {success_num / time_cost:.2f} files/s"
    )
    return success_num / time_cost


def prcess(
    html_data_path: str,
    target_data_path: str,
    file_list_path: str,
    parts: int,
    parts_num: int,
    batch_size: int = 20,
):
    cnt = 0
    futures = []
    ave_total = 0
    total = 0
    with ProcessPoolExecutor(max_workers=20) as executor:
        file_generator = gen_file_list_from_file(file_list_path, parts, parts_num, batch_size)
        try:
            while True:
                data = next(file_generator)
                futures.append(
                    executor.submit(single_file_process, data, target_data_path)
                )
                cnt += 1
                if cnt > 40:
                    break
        except StopIteration:
            pass

        for future in as_completed(futures):
            ret = future.result()
            if isinstance(ret, float):
                ave_total += ret
                total += 1
    logger.info(f"Total num: {total}, Ave total: {ave_total / total:.2f} files/s")


if __name__ == "__main__":
    parts = int(sys.argv[1])
    parts_num = int(sys.argv[2])
    data_path = "/dataset/tos/yaosikai/arxiv_database/arxiv/ar5iv_html/ar5iv_html/"
    target_path = "/dataset/tos/yaosikai/arxiv_database/arxiv/ar5iv_html/ar5iv_json/"
    file_list_path = "/dataset/tos/yaosikai/arxiv_database/file_list.txt"
    prcess(data_path, target_path)

    # single_file_process([f"{data_path}/0704/0704.2576.html"], target_path)
