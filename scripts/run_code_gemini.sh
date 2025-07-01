#!/bin/bash

# MIMIC-III Dataset Execution Script
# Run EhrAgent on MIMIC-III dataset

echo "Starting EhrAgent execution on MIMIC-III dataset..."

python3 ehragent/main.py \
    --llm "gemini-2.5-flash" \
    --dataset mimic_iii \
    --data_path data/mimic_iii/valid_preprocessed.json \
    --logs_path logs_gemini_mimic \
    --num_questions -1 \
    --seed 0 \
    --turn_step_wise_confidence

echo "MIMIC-III execution completed!"