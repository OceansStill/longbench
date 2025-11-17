#!/bin/bash
# 测试脚本：使用改进的pred_improved.py

echo "开始测试改进版本的pred_improved.py"
echo "使用单GPU运行Llamba-1B模型"

python pred_improved.py \
    --model "Llamba-1B" \
    --model_path "/home/liyijia/LinearAttaetion/downloaded_models/Llamba-1B" \
    --max_length 15000 \
    --num_gpus 1

echo "测试完成"

