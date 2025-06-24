import os
import json
import re
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from evaluate import judge  # Import judge function from evaluate.py

# judge function is now imported from evaluate.py

def extract_confidence_scores(log_content):
    """
    Extract confidence scores and solution lengths from the log content.
    
    Args:
        log_content (str): The content of the log file
        
    Returns:
        list: List of dictionaries with confidence scores and solution length
    """
    # Find all occurrences of "# Confidence: X" pattern
    confidence_pattern = r"# Confidence:\s*(\d+(?:\.\d+)?)"
    confidence_matches = re.findall(confidence_pattern, log_content)
    
    # Convert to float values
    confidence_scores = [float(score) / 10.0 for score in confidence_matches]  # Normalize to 0-1 range
    print(confidence_scores)
    # Get the overall solution length
    # Split by separator lines
    solution_parts = log_content.split('----------------------------------------------------------')
    # Find the parts that contain actual solution (usually after first few parts)
    solution_text = ""
    for part in solution_parts:
        if '# Confidence:' in part and not part.startswith('Question:'):
            solution_text = part.strip()
            break
    
    # Calculate solution length
    solution_length = len(solution_text) if solution_text else 0
    
    # Create list of dictionaries with confidence and solution_length
    result = []
    for i, confidence in enumerate(confidence_scores):
        result.append({
            "confidence": confidence,
            "query_length": solution_length,  # All steps get the same solution length
            "step_index": i + 1
        })
    
    return result

def combine_confidence_scores(confidence_data, method="unweighted"):
    """
    Combine multiple confidence scores into a single score using the specified method.
    
    Args:
        confidence_data (list): List of dictionaries with confidence scores and metadata
        method (str): Method to combine scores ('unweighted', 'position', 'length', 'pl', or 'last_step')
        
    Returns:
        float: Combined confidence score
    """
    if not confidence_data:
        return None
    
    # Extract confidence scores from the data
    scores = [item["confidence"] for item in confidence_data]
    
    if method == "unweighted":
        # Simple average
        return sum(scores) / len(scores)
    
    elif method == "position":
        # Weighted average based on position (later steps have higher weight)
        weights = [item["step_index"] for item in confidence_data]
        weighted_sum = sum(item["confidence"] * item["step_index"] for item in confidence_data)
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else None
    
    elif method == "length":
        # Weighted average based on solution length
        weights = [max(item["query_length"], 1) for item in confidence_data]  # Ensure minimum weight of 1
        weighted_sum = sum(item["confidence"] * max(item["query_length"], 1) for item in confidence_data)
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else None
    
    elif method == "pl":  # Position and Length combined
        # Weights are position multiplied by query length (with minimum length of 1)
        weights = [item["step_index"] * max(item["query_length"], 1) for item in confidence_data]
        weighted_sum = sum(item["confidence"] * (item["step_index"] * max(item["query_length"], 1)) for item in confidence_data)
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else None
    
    elif method == "last_step":
        # Only use the last step's confidence score
        return confidence_data[-1]["confidence"] if confidence_data else None
    
    else:
        # Default to unweighted if method is not recognized
        return sum(scores) / len(scores)

def calculate_hcacc(results, k_percent=90):
    """
    Calculate HcACC@k%, where k% is the minimum conditional accuracy requirement.
    
    Args:
        results (list): List of dictionaries with 'confidence' and 'is_correct' keys
        k_percent (float): The k value for HcACC@k%
        
    Returns:
        tuple: (best_oa, best_metrics) - the best overall accuracy and associated metrics
    """
    if not results:
        return 0, None
    
    # Get unique confidence values to try as thresholds
    thresholds = sorted(set(r["confidence"] for r in results))
    
    best_oa = 0
    best_metrics = None
    total_questions = len(results)
    
    # Try each threshold
    for threshold in thresholds:
        # Filter results with confidence >= threshold
        filtered = [r for r in results if r["confidence"] >= threshold]
        
        if not filtered:
            continue
        
        # Calculate metrics
        questions_attempted = len(filtered)
        correct_answers = sum(1 for r in filtered if r["is_correct"])
        
        ca = correct_answers / questions_attempted if questions_attempted > 0 else 0
        rr = questions_attempted / total_questions
        oa = correct_answers / total_questions
        hr = 1 - ca
        
        # Check if conditional accuracy is at least k%
        if ca >= k_percent / 100:
            if oa > best_oa:
                best_oa = oa
                best_metrics = {
                    "threshold": threshold,
                    "conditional_accuracy": ca,
                    "response_rate": rr,
                    "overall_accuracy": oa,
                    "hallucination_rate": hr,
                    "questions_attempted": questions_attempted,
                    "total_questions": total_questions
                }
    
    return best_oa, best_metrics



def visualize_hcacc_comparison(results, output_dir):
    """
    Visualize and compare HcACC performance by k for each method.
    
    Args:
        results (dict): Result dictionary with combining_method as key and DataFrame as value
        output_dir (str): Directory path to save result images
    """
    # 모든 방법의 결과를 병합
    combined_data = []
    for method, df in results.items():
        df_copy = df.copy()
        df_copy['Method'] = method
        combined_data.append(df_copy)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 각 방법별로 색상과 스타일 정의
    colors = {
        'unweighted': '#999999',
        'position': '#C6A300',
        'length': '#008B8B',
        'pl': '#D95F02',
        'last_step': '#6495ED'
    }
    
    line_styles = {
        'unweighted': '-',
        'position': '--',
        'length': '-.',
        'pl': '-',
        'last_step': '--'
    }
    
    # 각 방법별로 k값에 따른 HcACC 그래프 그리기
    for method in results.keys():
        method_df = combined_df[combined_df['Method'] == method]
        plt.plot(
            method_df['K%'], 
            method_df['HcACC'], 
            label=method,
            color=colors.get(method, 'black'),
            linestyle=line_styles.get(method, '-'),
            linewidth=2,
            marker='o' if method == 'pl' else None,
            markevery=10
        )
    
    # HR 참조선 추가
    for hr_level in [10, 20, 30]:
        ca_level = 100 - hr_level
        plt.axvline(
            x=ca_level, 
            color='#555555',
            linestyle=(0, (5, 5)),
            alpha=0.5, 
            linewidth=0.9
        )
        plt.text(
            ca_level+1, 
            0.05,
            f"HR<{hr_level}%", 
            rotation=90, 
            color='#555555', 
            alpha=0.8, 
            fontsize=8
        )
    
    # 그래프 레이블, 제목 등 설정
    plt.xlabel('k%')
    plt.ylabel('HcACC@k%')
    plt.title('HcACC Performance Comparison Across Different Methods')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 결과 저장
    output_path = os.path.join(output_dir, 'hcacc_methods_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualization saved to {output_path}")

def load_answers(answer_file: str) -> dict:
    """Load answers from JSON file."""
    if not os.path.exists(answer_file):
        raise FileNotFoundError(f"Answer file not found: {answer_file}")
        
    with open(answer_file, 'r') as f:
        contents = json.load(f)
    
    answers = {}
    for item in contents:
        answers[item['id']] = item['answer']
    
    return answers

def process_logs_and_calculate_hcacc(logs_path, answer_file, k_values=[0, 50, 80, 90, 95], combining_methods=["unweighted", "position", "length", "pl"]):
    """
    Process log files and calculate HcACC metrics.
    
    Args:
        logs_path (str): Path to the log files directory
        answer_file (str): Path to the answer file
        k_values (list): List of k values for HcACC@k%
        combining_methods (list): List of methods to combine confidence scores
        
    Returns:
        dict: Dictionary with results for each combining method and k value
    """
    # Load answers
    answers = load_answers(answer_file)
    
    files = os.listdir(logs_path)
    
    # Initialize results
    results_by_method = {method: [] for method in combining_methods}
    
    # Process each file
    for file in tqdm(files, desc="Processing files"):
        if not file.split('.')[0] in answers.keys():
            continue
        
        with open(os.path.join(logs_path, file), 'r') as f:
            logs = f.read()
        
        split_logs = logs.split('\n----------------------------------------------------------\n')
        question = split_logs[0]
        answer = answers[file.split('.')[0]]
        if type(answer) == list:
            answer = ', '.join(answer)
        
        # Check if the prediction is correct
        if "TERMINATE" in logs:
            if '"cell": "' in logs:
                last_code_start = logs.rfind('"cell": "')
                last_code_end = logs.rfind('"\n}')
                last_code = logs[last_code_start+9:last_code_end]
            else:
                last_code_end = logs.rfind('Solution:')
            prediction_end = logs.rfind('TERMINATE')
            prediction = logs[last_code_end:prediction_end]
            logs_content = logs.split('TERMINATE')[0]
            is_correct = judge(prediction, answer)
        else:
            # Unfinished logs
            logs_content = logs
            is_correct = False
        
        # Extract confidence scores and query metadata
        confidence_data = extract_confidence_scores(logs_content)
        
        # Skip if no confidence scores found
        if not confidence_data:
            continue
        
        # Combine confidence scores using different methods
        for method in combining_methods:
            combined_confidence = combine_confidence_scores(confidence_data, method)
            
            if combined_confidence is not None:
                results_by_method[method].append({
                    "file": file,
                    "confidence": combined_confidence,
                    "is_correct": is_correct
                })
    
    # Calculate HcACC for each method and k value
    final_results = {}
    
    for method in combining_methods:
        method_results = []
        
        for k in k_values:
            hcacc, metrics = calculate_hcacc(results_by_method[method], k_percent=k)
            
            if metrics:
                method_results.append({
                    "K%": k,
                    "HcACC": hcacc,
                    "Threshold": metrics["threshold"],
                    "CA": metrics["conditional_accuracy"],
                    "RR": metrics["response_rate"],
                    "HR": metrics["hallucination_rate"],
                    "Attempted": metrics["questions_attempted"],
                    "Total": metrics["total_questions"]
                })
            else:
                method_results.append({
                    "K%": k,
                    "HcACC": 0,
                    "Threshold": "N/A",
                    "CA": 0,
                    "RR": 0,
                    "HR": 0,
                    "Attempted": 0,
                    "Total": len(results_by_method[method])
                })
        
        final_results[method] = pd.DataFrame(method_results)
    
    return final_results


def calculate_basic_stats(logs_path: str, answer_file: str) -> dict:
    """Calculate basic evaluation statistics."""
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Logs directory not found: {logs_path}")
        
    files = os.listdir(logs_path)
    answers = load_answers(answer_file)
    
    stats = {"total_num": 0, "correct": 0, "unfinished": 0, "incorrect": 0}
    
    for file in files:
        if not file.split('.')[0] in answers.keys():
            continue
            
        try:
            with open(os.path.join(logs_path, file), 'r') as f:
                logs = f.read()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        
        answer = answers[file.split('.')[0]]
        if isinstance(answer, list):
            answer = ', '.join(answer)
        
        stats["total_num"] += 1
        if "TERMINATE" not in logs:
            stats["unfinished"] += 1
        else:
            try:
                if '"cell": "' in logs:
                    last_code_start = logs.rfind('"cell": "')
                    last_code_end = logs.rfind('"\n}')
                else:
                    last_code_end = logs.rfind('Solution:')
                prediction_end = logs.rfind('TERMINATE')
                prediction = logs[last_code_end:prediction_end]
                
                result = judge(prediction, answer)
                if result:
                    stats["correct"] += 1
                else:
                    stats["incorrect"] += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")
                stats["incorrect"] += 1
    
    return stats

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-evaluate log files with confidence score analysis and HcACC calculation"
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        default="logs_mimic/4/",
        help="Path to directory containing log files (default: logs_mimic/4/)"
    )
    parser.add_argument(
        "--answer_file",
        type=str,
        default="data/mimic_iii/valid_preprocessed.json",
        help="Path to JSON file with ground truth answers (default: data/mimic_iii/valid_preprocessed.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results (default: same as logs_path)"
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="0,50,80,90,95",
        help="Comma-separated k values for HcACC@k% (default: 0,50,80,90,95)"
    )
    parser.add_argument(
        "--combining_methods",
        type=str,
        default="unweighted,position,length,pl,last_step",
        help="Comma-separated combining methods (default: unweighted,position,length,pl,last_step)"
    )
    parser.add_argument(
        "--full_range",
        action="store_true",
        help="Use full range k values (0-100) instead of specified k_values"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.logs_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse k values and combining methods
    if args.full_range:
        k_values = list(range(0, 101, 1))  # k values from 0% to 100%
    else:
        k_values = [int(k.strip()) for k in args.k_values.split(',')]
    
    combining_methods = [method.strip() for method in args.combining_methods.split(',')]
    
    try:
        # Calculate basic stats first
        print("Calculating basic statistics...")
        stats = calculate_basic_stats(args.logs_path, args.answer_file)
        
        print("Basic Stats:")
        print(f"Total questions: {stats['total_num']}")
        print(f"Correct: {stats['correct']} ({stats['correct']/stats['total_num']*100:.2f}%)")
        print(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total_num']*100:.2f}%)")
        print(f"Unfinished: {stats['unfinished']} ({stats['unfinished']/stats['total_num']*100:.2f}%)")
        
        # Process logs and calculate HcACC
        print("\nCalculating HcACC metrics...")
        results = process_logs_and_calculate_hcacc(
            args.logs_path, args.answer_file, k_values, combining_methods
        )
        
        # Save results to CSV files
        for method, df in results.items():
            output_file = os.path.join(output_dir, f"{method}_hcacc_results.csv")
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        # Create visualization
        print("\nCreating visualization...")
        visualize_hcacc_comparison(results, output_dir)
        
        # Print results
        if args.verbose:
            print("\n==== HcACC Results ====")
            for method in combining_methods:
                if method in results:
                    print(f"\nMethod: {method}")
                    print(results[method].to_string(index=False))
                    
    except Exception as e:
        print(f"Error during post-evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()