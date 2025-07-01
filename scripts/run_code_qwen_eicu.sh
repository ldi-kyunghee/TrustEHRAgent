#!/bin/bash

# eICU Dataset Execution Script
# Run EhrAgent on eICU dataset

echo "Starting EhrAgent execution on eICU dataset..."

python3 ehragent/main.py \
    --llm "Qwen/Qwen2.5-7B-Instruct-Turbo" \
    --dataset eicu \
    --data_path data/eicu/valid_preprocessed.json \
    --logs_path logs_qwen_eicu \
    --num_questions -1 \
    --turn_step_wise_confidence \
    --seed 0

echo "eICU execution completed!"