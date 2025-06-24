#!/bin/bash

# eICU Dataset Execution Script
# Run EhrAgent on eICU dataset

echo "Starting EhrAgent execution on eICU dataset..."

python3 ehragent/main.py \
    --llm "gpt-4.1-mini" \
    --dataset eicu \
    --data_path data/eicu/valid_preprocessed.json \
    --logs_path logs_eicu \
    --num_questions -1 \
    --seed 0

echo "eICU execution completed!"