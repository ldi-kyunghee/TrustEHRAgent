#!/bin/bash

# MIMIC-III Dataset Execution Script (Baseline - No Confidence Rating)
# Run EhrAgent on MIMIC-III dataset without confidence rating prompts

echo "Starting EhrAgent execution on MIMIC-III dataset (Baseline)..."

python3 ehragent/main.py \
    --llm "gemini-2.5-flash" \
    --dataset mimic_iii \
    --data_path data/mimic_iii/valid_preprocessed.json \
    --logs_path logs_gemini_mimic_baseline \
    --num_questions -1 \
    --seed 0

echo "MIMIC-III baseline execution completed!"