#!/usr/bin/env python3
"""
Run confidence calculation based on existing evaluation results.
This script loads evaluation results and runs confidence analysis on them.
"""

import json
import argparse
from pathlib import Path
from confidence_calculation import process_logs_and_calculate_hcacc

def load_evaluation_results(evaluation_json_path):
    """Load evaluation results from JSON file."""
    with open(evaluation_json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Run confidence calculation from evaluation results"
    )
    parser.add_argument(
        "--evaluation_json",
        type=str,
        required=True,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        help="Path to logs directory (optional, extracted from evaluation results if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for confidence results (default: same as evaluation JSON directory)"
    )
    parser.add_argument(
        "--no_external_confidence",
        action="store_true",
        help="Disable external confidence calculation using LLM"
    )
    parser.add_argument(
        "--confidence_checker_types",
        nargs="+",
        default=["standard"],
        choices=["standard", "binary", "integer"],
        help="Types of confidence checkers to use"
    )
    parser.add_argument(
        "--combining_methods",
        nargs="+",
        default=["unweighted", "position", "length", "pl"],
        help="Methods to combine confidence scores"
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[0, 50, 80, 90, 95],
        help="K values for HcACC@k% calculation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Load evaluation results
    evaluation_results = load_evaluation_results(args.evaluation_json)
    
    # Extract logs_path from evaluation results if not provided
    if args.logs_path is None:
        # Try to extract from the first detailed result
        detailed_results = evaluation_results.get('detailed_results', [])
        if detailed_results:
            first_log_path = detailed_results[0]['log_file_path']
            args.logs_path = str(Path(first_log_path).parent)
        else:
            raise ValueError("No logs_path provided and cannot extract from evaluation results")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.evaluation_json).parent)
    
    print(f"Loading evaluation results from: {args.evaluation_json}")
    print(f"Processing {len(evaluation_results.get('detailed_results', []))} samples")
    print(f"Output directory: {args.output_dir}")
    
    # Run confidence calculation
    use_external_confidence = not args.no_external_confidence
    confidence_results = process_logs_and_calculate_hcacc(
        logs_path=args.logs_path,
        answer_file=None,  # Not needed when using evaluation results
        k_values=args.k_values,
        combining_methods=args.combining_methods,
        use_external_confidence=use_external_confidence,
        confidence_checker_types=args.confidence_checker_types,
        output_dir=args.output_dir,
        verbose=args.verbose,
        evaluation_results=evaluation_results  # Pass evaluation results
    )
    
    # Save confidence analysis results
    confidence_output_file = Path(args.output_dir) / "confidence_analysis_results.json"
    with open(confidence_output_file, 'w') as f:
        # Convert pandas DataFrames to dict for JSON serialization
        serializable_results = {}
        for checker_type, methods in confidence_results['confidence_analysis'].items():
            serializable_results[checker_type] = {}
            for method, df in methods.items():
                serializable_results[method] = df.to_dict('records')
        
        output_data = {
            'confidence_analysis': serializable_results,
            'confidence_results_file': confidence_results['confidence_results_file'],
            'based_on_evaluation': confidence_results.get('based_on_evaluation', False),
            'evaluation_summary': confidence_results.get('evaluation_summary', {}),
            'parameters': {
                'k_values': args.k_values,
                'combining_methods': args.combining_methods,
                'confidence_checker_types': args.confidence_checker_types,
                'use_external_confidence': use_external_confidence
            }
        }
        
        json.dump(output_data, f, indent=2)
    
    print(f"Confidence analysis results saved to: {confidence_output_file}")
    print(f"Raw confidence data saved to: {confidence_results['confidence_results_file']}")
    
    # Print summary
    print("\nConfidence Analysis Summary:")
    print(f"- Evaluation accuracy: {confidence_results['evaluation_summary'].get('accuracy', 'N/A'):.2f}%")
    print(f"- Completion rate: {confidence_results['evaluation_summary'].get('completion_rate', 'N/A'):.2f}%")
    print(f"- Total samples: {confidence_results['evaluation_summary'].get('total_samples', 'N/A')}")

if __name__ == "__main__":
    main()