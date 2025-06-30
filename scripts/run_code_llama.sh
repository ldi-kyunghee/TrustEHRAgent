#!/bin/bash

# MIMIC-III Dataset Execution Script
# Run EhrAgent on MIMIC-III dataset

echo "Starting EhrAgent execution on MIMIC-III dataset..."

python3 ehragent/main.py \
    --llm "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" \
    --dataset mimic_iii \
    --data_path data/mimic_iii/valid_preprocessed.json \
    --logs_path logs_llama_mimic \
    --num_questions -1 \
    --seed 0

echo "MIMIC-III execution completed!"