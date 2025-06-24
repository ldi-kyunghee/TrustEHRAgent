#!/bin/bash

# Enhanced Confidence Analysis and HcACC Evaluation Script
# Set your configuration variables here
# LOGS_PATH="logs_mimic/4/"
LOGS_PATH="logs_mimic/4"
ANSWER_FILE="data/mimic_iii/valid_preprocessed.json"
OUTPUT_DIR="results/mimic_base"
K_VALUES="0,50,80,90,95,100"
COMBINING_METHODS="unweighted,position,length,pl,last_step"

# Set to true if you want to use external confidence evaluation
USE_EXTERNAL_CONFIDENCE=true
EXTERNAL_MODEL="gpt-4.1"  # Only used if USE_EXTERNAL_CONFIDENCE is true

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the HcACC calculator
if [ "$USE_EXTERNAL_CONFIDENCE" = true ]; then
    # With external confidence
    python ehragent/main.py \
        --logs_path $LOGS_PATH \
        --answer_file $ANSWER_FILE \
        --output_dir $OUTPUT_DIR \
        --k_values $K_VALUES \
        --combining_methods $COMBINING_METHODS \
        --use_external_confidence \
        --external_confidence_model $EXTERNAL_MODEL \
        --analyze_confidence

else
    # Without external confidence
    python ehragent/main.py \
        --logs_path $LOGS_PATH \
        --answer_file $ANSWER_FILE \
        --output_dir $OUTPUT_DIR \
        --k_values $K_VALUES \
        --combining_methods $COMBINING_METHODS
fi

echo "HcACC calculation completed! Results are saved in $OUTPUT_DIR"