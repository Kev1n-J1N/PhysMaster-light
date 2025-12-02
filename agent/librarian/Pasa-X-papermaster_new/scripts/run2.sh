#!/bin/bash

# 定义运行脚本路径
SCRIPT_PATH="/mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/run.py"

# 第一次运行
python $SCRIPT_PATH \
    --ranker_model qwen2.5-7b-instruct-sft \
    --output_folder /mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/result/qwen2.5-7b-instruct-sft

# 第二次运行
python $SCRIPT_PATH \
    --ranker_model DeepSeek-R1-Distill-Qwen-7B \
    --output_folder /mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/result/DeepSeek-R1-Distill-Qwen-7B

