#!/bin/bash

# Evaluation Script for EhrAgent
# Set your configuration variables here
LOGS_PATH="logs_test/4"
ANSWER_FILE="data/mimic_iii/valid_preprocessed.json"
OUTPUT_DIR="results/mimic_qwen"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Starting evaluation of EhrAgent results..."
echo "Logs path: $LOGS_PATH"
echo "Answer file: $ANSWER_FILE"
echo "Output directory: $OUTPUT_DIR"

# Run the evaluation
python3 ehragent/evaluate.py \
    --logs_path $LOGS_PATH \
    --answer_file $ANSWER_FILE \
    --output_json $OUTPUT_DIR/evaluation_results.json \
    --verbose

echo "Evaluation completed! Results are saved in $OUTPUT_DIR"