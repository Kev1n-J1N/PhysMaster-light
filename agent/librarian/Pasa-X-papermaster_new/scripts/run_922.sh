#!/bin/bash

# 定义运行脚本路径
SCRIPT_PATH="/data/Tianjin/Pasa-X/run.py"

# 第一次运行
python $SCRIPT_PATH \
    --dataset_name RealScholarQuery \
    --input_file data/RealScholarQuery/test.jsonl \
    --ranker_model qwen-72b \
    --output_folder /data/Tianjin/Pasa-X/result/qwen72b-922 \
    --threads_num 100