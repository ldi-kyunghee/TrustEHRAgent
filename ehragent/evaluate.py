import os
import json
import argparse
from pathlib import Path

def judge(pred, ans):
    old_flag = True
    if not ans in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if ", " in ans:
        ans = ans.split(', ')
    if ans[-2:] == ".0":
        ans = ans[:-2]
    if not type(ans) == list:
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if not ans[i] in pred:
            new_flag = False
            break
    return old_flag or new_flag

def load_answers(answer_file: str) -> dict:
    """Load answers from JSON file.
    
    Args:
        answer_file: Path to the answer file
        
    Returns:
        Dictionary mapping question IDs to answers
    """
    if not os.path.exists(answer_file):
        raise FileNotFoundError(f"Answer file not found: {answer_file}")
        
    with open(answer_file, 'r') as f:
        contents = json.load(f)
    
    answers = {}
    for item in contents:
        answers[item['id']] = item['answer']
    
    return answers

def evaluate_logs(logs_path: str, answer_file: str) -> tuple[dict, list]:
    """Evaluate log files against ground truth answers.
    
    Args:
        logs_path: Path to directory containing log files
        answer_file: Path to JSON file with ground truth answers
        
    Returns:
        Tuple of (statistics dictionary, detailed results list)
    """
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Logs directory not found: {logs_path}")
    
    # Load answers
    answers = load_answers(answer_file)
    
    # Get log files
    files = os.listdir(logs_path)
    
    stats = {"total_num": 0, "correct": 0, "unfinished": 0, "incorrect": 0}
    detailed_results = []
    
    for file in files:
        file_id = file.split('.')[0]
        if file_id not in answers:
            continue
            
        log_file_path = os.path.join(logs_path, file)
        try:
            with open(log_file_path, 'r') as f:
                logs = f.read()
        except Exception as e:
            print(f"Error reading {log_file_path}: {e}")
            continue
            
        answer = answers[file_id]
        original_answer = answer  # Keep original format for JSON output
        if isinstance(answer, list):
            answer = ', '.join(answer)
            
        stats["total_num"] += 1
        
        # Extract query from logs
        split_logs = logs.split('\n----------------------------------------------------------\n')
        query = split_logs[0].strip() if split_logs else "Query not found"
        
        # Initialize sample result
        sample_result = {
            "sample_id": file_id,
            "log_file_path": log_file_path,
            "query": query,
            "ground_truth_answer": original_answer,
            "status": None,
            "is_correct": None,
            "prediction": None,
            "error_message": None
        }
        
        if "TERMINATE" not in logs:
            stats["unfinished"] += 1
            sample_result["status"] = "unfinished"
            sample_result["is_correct"] = False
        else:
            # Extract prediction from logs
            try:
                if '"cell": "' in logs:
                    last_code_start = logs.rfind('"cell": "')
                    last_code_end = logs.rfind('"\n}')
                    last_code_end = last_code_end if last_code_end != -1 else len(logs)
                else:
                    last_code_end = logs.rfind('Solution:')
                    last_code_end = last_code_end if last_code_end != -1 else 0
                    
                prediction_end = logs.rfind('TERMINATE')
                prediction = logs[last_code_end:prediction_end].strip()
                
                sample_result["prediction"] = prediction
                
                result = judge(prediction, answer)
                sample_result["is_correct"] = result
                
                if result:
                    stats["correct"] += 1
                    sample_result["status"] = "correct"
                else:
                    stats["incorrect"] += 1
                    sample_result["status"] = "incorrect"
                    
            except Exception as e:
                sample_result["error_message"] = str(e)
                sample_result["status"] = "error"
                sample_result["is_correct"] = False
                stats["incorrect"] += 1
        
        detailed_results.append(sample_result)
    
    return stats, detailed_results

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate log files against ground truth answers"
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
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Path to save detailed results as JSON file"
    )
    return parser.parse_args()

def save_detailed_results(results: list, stats: dict, output_path: str) -> None:
    """Save detailed evaluation results to JSON file.
    
    Args:
        results: List of detailed sample results
        stats: Overall statistics
        output_path: Path to save JSON file
    """
    output_data = {
        "evaluation_summary": {
            "total_samples": stats["total_num"],
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "unfinished": stats["unfinished"],
            "accuracy": stats["correct"] / stats["total_num"] * 100 if stats["total_num"] > 0 else 0,
            "completion_rate": (stats["correct"] + stats["incorrect"]) / stats["total_num"] * 100 if stats["total_num"] > 0 else 0
        },
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: {output_path}")

def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    try:
        stats, detailed_results = evaluate_logs(args.logs_path, args.answer_file)
        
        # Print results
        print("Evaluation Results:")
        print(f"Total questions: {stats['total_num']}")
        print(f"Correct: {stats['correct']} ({stats['correct']/stats['total_num']*100:.2f}%)")
        print(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total_num']*100:.2f}%)")
        print(f"Unfinished: {stats['unfinished']} ({stats['unfinished']/stats['total_num']*100:.2f}%)")
        
        if args.verbose:
            print(f"\nDetailed stats: {stats}")
            print(f"\nSample results:")
            for result in detailed_results[:5]:  # Show first 5 samples
                print(f"  Sample {result['sample_id']}: {result['status']} - {result['query'][:50]}...")
        
        # Save detailed results to JSON if requested
        if args.output_json:
            save_detailed_results(detailed_results, stats, args.output_json)
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
