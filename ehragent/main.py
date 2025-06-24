#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage example:

python run_confidence_analysis.py \
    --logs_path ./logs \
    --answer_file ./answers.json \
    --output_dir ./results \
    --combining_methods unweighted,position,length,pl,last_step \
    --confidence_checker_types standard \
    --use_external_confidence
"""

import argparse
import os
from confidence_calculation import (
    calculate_basic_stats, 
    process_logs_and_calculate_hcacc,
    get_raw_results_for_analysis
)
from enhanced_confidence_analysis import extend_main_process

def main():
    parser = argparse.ArgumentParser(description='Analyze confidence scores in log files')
    parser.add_argument('--logs_path', type=str, required=True, help='Path to log files')
    parser.add_argument('--answer_file', type=str, required=True, help='Path to answer file')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--combining_methods', type=str, default='unweighted,position,length,pl,last_step', 
                        help='Comma-separated list of combining methods')
    parser.add_argument('--confidence_checker_types', type=str, default='standard', 
                        help='Comma-separated list of confidence checker types (standard, binary, integer)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    parser.add_argument('--use_external_confidence', action='store_true', help='Calculate external confidence scores')
    parser.add_argument('--external_confidence_model', type=str, default='gpt-4.1', 
                        help='Model to use for external confidence evaluation')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.logs_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse combining methods and checker types
    combining_methods = args.combining_methods.split(',')
    confidence_checker_types = args.confidence_checker_types.split(',')
    
    # Set section inclusion - all sections included by default
    include_knowledge = True
    include_solution = True
    include_examples = True
    
    # Calculate basic stats
    stats = calculate_basic_stats(args.logs_path, args.answer_file)
    
    print("Basic Stats:")
    print(f"Total questions: {stats['total_num']}")
    print(f"Correct: {stats['correct']} ({stats['correct']/stats['total_num']*100:.2f}%)")
    print(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total_num']*100:.2f}%)")
    print(f"Unfinished: {stats['unfinished']} ({stats['unfinished']/stats['total_num']*100:.2f}%)")
    
    # Add 'external' to combining methods if using external confidence
    if args.use_external_confidence and 'external' not in combining_methods:
        combining_methods.append('external')
        print("\nAdded 'external' method for confidence analysis.")
    
    # If using external confidence, run the full HcACC calculation first
    # This will generate the external confidence results files
    if args.use_external_confidence:
        print("\nCalculating external confidence scores...")
        # We only need k=0 for this purpose
        k_values = [0]
        
        process_logs_and_calculate_hcacc(
            args.logs_path,
            args.answer_file,
            k_values,
            combining_methods,
            args.use_external_confidence,
            confidence_checker_types,
            args.external_confidence_model,
            output_dir,
            args.verbose,
            include_knowledge,
            include_solution,
            include_examples
        )
        print("External confidence scores calculated and saved.")
    
    # Perform enhanced confidence analysis
    print("\nPerforming confidence analysis...")
    
    # Start with empty result structure and extract data directly from log files
    analysis_results = extend_main_process(
        {}, # Start with empty result structure
        output_dir, 
        args.logs_path,
        args.answer_file,
        combining_methods,
        confidence_checker_types,
        args.use_external_confidence  # Pass the flag to consider external confidence
    )
    
    print("Confidence analysis completed.")
    print(f"Analysis results can be found in: {os.path.join(output_dir, 'confidence_analysis')}")
    
    return analysis_results

if __name__ == "__main__":
    main()