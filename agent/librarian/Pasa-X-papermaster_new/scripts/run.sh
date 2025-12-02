#!/bin/bash

# 定义运行脚本路径
SCRIPT_PATH="/mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/run.py"

# 第一次运行
python $SCRIPT_PATH \
    --ranker_model qwen2.5-7b \
    --output_folder /mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/result/qwen2.5-7b

# 第二次运行
python $SCRIPT_PATH \
    --ranker_model qwen2.5-7b-sft \
    --output_folder /mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/result/qwen2.5-7b-sft

# 第三次运行
python $SCRIPT_PATH \
    --ranker_model qwen-72b \
    --output_folder /mnt/chensiheng/Tianjin/papermaster/PASAX-jt/Pasa-X/result/qwen-72b
