#!/bin/bash

# eICU Dataset Execution Script
# Run EhrAgent on eICU dataset

echo "Starting EhrAgent execution on eICU dataset..."

python3 ehragent/main.py \
    --llm "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" \
    --dataset eicu \
    --data_path data/eicu/valid_preprocessed.json \
    --logs_path logs_llama_eicu \
    --num_questions -1 \
    --turn_step_wise_confidence \
    --seed 0

echo "eICU execution completed!"