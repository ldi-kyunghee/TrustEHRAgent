#!/bin/bash

# Post-Evaluation Analysis Script
# Analyze and visualize results from confidence evaluation

RESULTS_DIR="results"
OUTPUT_DIR="analysis_output"

echo "Starting post-evaluation analysis..."

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run post-evaluation analysis
python ehragent/post_evaluate.py \
    --results_dir $RESULTS_DIR \
    --output_dir $OUTPUT_DIR

echo "Post-evaluation analysis completed! Results are saved in $OUTPUT_DIR"