import re
import os
import json
import pandas as pd
from tqdm import tqdm
from confidence_checker import (
    ConfidenceChecker, 
    BinaryConfidenceChecker, 
    IntegerConfidenceChecker, 
    format_log_to_history
)

def judge(pred, ans):
    """
    Judge whether the prediction is correct based on the answer.
    
    Args:
        pred (str): The prediction string
        ans (str or list): The answer(s)
        
    Returns:
        bool: True if the prediction matches the answer, False otherwise
    """
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
    return (old_flag or new_flag)

def extract_confidence_scores(log_content, verbose=False):
    """
    Extract confidence scores and solution lengths from the log content.
    
    Args:
        log_content (str): The content of the log file
        verbose (bool): Whether to print confidence scores
        
    Returns:
        list: List of dictionaries with confidence scores and solution length
    """
    # Find all occurrences of "# Confidence: X" pattern
    confidence_pattern = r"# Confidence:\s*(\d+(?:\.\d+)?)"
    confidence_matches = re.findall(confidence_pattern, log_content)
    
    # Convert to float values
    confidence_scores = [float(score) / 10.0 for score in confidence_matches]  # Normalize to 0-1 range
    if verbose:
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

def calculate_basic_stats(logs_path, answer_file):
    """
    Calculate basic statistics for the log files.
    
    Args:
        logs_path (str): Path to the log files directory
        answer_file (str): Path to the answer file
        
    Returns:
        dict: Dictionary with basic statistics
    """
    # Load answers
    with open(answer_file, 'r') as f:
        contents = json.load(f)
    answers = {}
    for i in range(len(contents)):
        answers[contents[i]['id']] = contents[i]['answer']
    
    files = os.listdir(logs_path)
    
    stats = {"total_num": 0, "correct": 0, "unfinished": 0, "incorrect": 0}
    
    for file in files:
        if not file.endswith('.txt') and not file.endswith('.log'):
            continue
            
        file_id = file.split('.')[0]
        if file_id not in answers:
            continue
            
        with open(os.path.join(logs_path, file), 'r') as f:
            logs = f.read()
        
        answer = answers[file_id]
        if type(answer) == list:
            answer = ', '.join(answer)
        
        stats["total_num"] += 1
        if not "TERMINATE" in logs:
            stats["unfinished"] += 1
        else:
            if '"cell": "' in logs:
                last_code_start = logs.rfind('"cell": "')
                last_code_end = logs.rfind('"\n}')
                last_code = logs[last_code_start+9:last_code_end]
            else:
                last_code_end = logs.rfind('Solution:')
            prediction_end = logs.rfind('TERMINATE')
            prediction = logs[last_code_end:prediction_end]
            logs = logs.split('TERMINATE')[0]
            result = judge(prediction, answer)
            if result:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
    
    return stats

def process_logs_and_calculate_hcacc(
    logs_path, 
    answer_file, 
    k_values=[0, 50, 80, 90, 95], 
    combining_methods=["unweighted", "position", "length", "pl"],
    use_external_confidence=False,
    confidence_checker_types=["standard"],  # New parameter for confidence checker types
    external_confidence_model="gpt-4.1",
    output_dir=None,
    verbose=False,
    include_knowledge=True,
    include_solution=True,
    include_examples=True,
    evaluation_results=None  # Add evaluation results from evaluate.py
):
    """
    Process log files and calculate HcACC metrics.
    
    Args:
        logs_path (str): Path to the log files directory
        answer_file (str): Path to the answer file
        k_values (list): List of k values for HcACC@k%
        combining_methods (list): List of methods to combine confidence scores
        use_external_confidence (bool): Whether to calculate external confidence using LLM
        confidence_checker_types (list): List of confidence checker types to use 
                                        ('standard', 'binary', 'integer')
        external_confidence_model (str): Model to use for external confidence evaluation
        output_dir (str): Output directory for results (default: logs_path)
        verbose (bool): Whether to print detailed information
        include_knowledge (bool): Whether to include knowledge section in external confidence evaluation
        include_solution (bool): Whether to include solution section in external confidence evaluation
        include_examples (bool): Whether to include example code sections in external confidence evaluation
        
    Returns:
        dict: Dictionary with results for each confidence checker type, combining method and k value
    """
    if output_dir is None:
        output_dir = logs_path
        
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use evaluation results if provided, otherwise load from files
    if evaluation_results is not None:
        # Extract file information from evaluation results
        detailed_results = evaluation_results.get('detailed_results', [])
        files_info = {}
        for result in detailed_results:
            file_id = result['sample_id']
            files_info[file_id] = {
                'log_file_path': result['log_file_path'],
                'ground_truth_answer': result['ground_truth_answer'],
                'is_correct': result['is_correct'],
                'query': result['query'],
                'prediction': result.get('prediction', '')
            }
    else:
        # Fallback to original file loading method
        with open(answer_file, 'r') as f:
            contents = json.load(f)
        answers = {}
        for i in range(len(contents)):
            answers[contents[i]['id']] = contents[i]['answer']
        
        files = os.listdir(logs_path)
        files_info = None
    
    # Initialize external confidence checkers based on the requested types
    external_checkers = {}
    if use_external_confidence:
        if "external" not in combining_methods:
            combining_methods.append("external")
            
        for checker_type in confidence_checker_types:
            if checker_type == "standard":
                external_checkers["standard"] = ConfidenceChecker(model=external_confidence_model)
            elif checker_type == "binary":
                external_checkers["binary"] = BinaryConfidenceChecker(model=external_confidence_model)
            elif checker_type == "integer":
                external_checkers["integer"] = IntegerConfidenceChecker(model=external_confidence_model)
    
    # Initialize results, now including checker type dimension
    results_by_checker_and_method = {}
    for checker_type in confidence_checker_types:
        results_by_checker_and_method[checker_type] = {method: [] for method in combining_methods}
    
    # Use single JSON file for confidence results
    sections_suffix = f"_k{int(include_knowledge)}_s{int(include_solution)}_e{int(include_examples)}"
    confidence_results_file = os.path.join(output_dir, f"confidence_results{sections_suffix}.json")
    
    # Load existing confidence results from single file
    if os.path.exists(confidence_results_file):
        with open(confidence_results_file, 'r') as f:
            all_confidence_results = json.load(f)
    else:
        all_confidence_results = {}
    
    # Initialize external confidence results for each checker type
    external_confidence_results = {}
    for checker_type in confidence_checker_types:
        external_confidence_results[checker_type] = all_confidence_results.get(f'external_confidence_{checker_type}', {})
    
    # Process files based on evaluation results or file system
    if files_info is not None:
        # Use evaluation results
        items_to_process = list(files_info.items())
        desc = "Processing files from evaluation results"
    else:
        # Use file system
        items_to_process = [(file.split('.')[0], file) for file in files if file.endswith('.txt') or file.endswith('.log')]
        desc = "Processing files"
    
    for item in tqdm(items_to_process, desc=desc):
        if files_info is not None:
            # Using evaluation results
            file_id, file_info = item
            question = file_info['query']
            answer = file_info['ground_truth_answer']
            is_correct = file_info['is_correct']
            prediction = file_info['prediction']
            
            # Read the log file to get logs_content
            with open(file_info['log_file_path'], 'r') as f:
                logs = f.read()
            logs_content = logs.split('TERMINATE')[0] if 'TERMINATE' in logs else logs
        else:
            # Using file system (original method)
            file_id, file = item
            if file_id not in answers:
                if verbose:
                    print(f"Skipping {file_id}: no answer found")
                continue
            
            # Read the file to get question and prediction
            with open(os.path.join(logs_path, file), 'r') as f:
                logs = f.read()
            
            split_logs = logs.split('\n----------------------------------------------------------\n')
            question = split_logs[0]
            answer = answers[file_id]
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
                prediction = ""
        
        # Process external confidence for each checker type if requested
        if use_external_confidence:
            # Format log for confidence checker with section options (do this once)
            formatted_question, formatted_prediction, formatted_history = format_log_to_history(
                logs_content, 
                include_knowledge=include_knowledge,
                include_solution=include_solution,
                include_examples=include_examples
            )
            
            for checker_type in confidence_checker_types:
                # Check if we already have external confidence for this file and checker type
                if file_id in external_confidence_results[checker_type]:
                    # Use existing external confidence
                    external_conf_result = external_confidence_results[checker_type][file_id]
                    external_conf = external_conf_result.get("confidence_score", 0.5)
                    if verbose:
                        print(f"Using existing {checker_type} external confidence for {file_id}")
                elif checker_type in external_checkers:
                    # Calculate new external confidence with the appropriate checker
                    try:
                        external_conf_result = external_checkers[checker_type].check_confidence(
                            question=formatted_question,
                            answer=formatted_prediction,
                            history=formatted_history,
                            mode="all"
                        )
                        
                        # Add to results dictionary with question
                        external_confidence_results[checker_type][file_id] = {
                            "id": file_id,
                            "question": question,
                            "score": external_conf_result.get("score", 0),
                            "score_distribution": external_conf_result.get("score_distribution", []),
                            "weighted_score": external_conf_result.get("weighted_score", 0),
                            "confidence_score": external_conf_result.get("confidence_score", 0.5)
                        }
                        
                        # Save after each new calculation to avoid losing data
                        all_confidence_results[f'external_confidence_{checker_type}'] = external_confidence_results[checker_type]
                        with open(confidence_results_file, 'w') as f:
                            json.dump(all_confidence_results, f, indent=2)
                        
                        external_conf = external_conf_result.get("confidence_score", 0.5)
                        if verbose:
                            print(f"{checker_type} external confidence for {file_id} calculated and saved")
                    
                    except Exception as e:
                        print(f"Error calculating {checker_type} external confidence for {file}: {e}")
                        # Use a default value if calculation fails
                        external_conf = 0.5
                else:
                    # No checker, use default
                    external_conf = 0.5
                
                # Add external confidence result  
                if files_info is not None:
                    file_name = file_info['log_file_path']
                else:
                    file_name = file
                results_by_checker_and_method[checker_type]["external"].append({
                    "file": file_name,
                    "file_id": file_id,
                    "confidence": external_conf,
                    "is_correct": is_correct
                })
        
        # Extract confidence scores and query metadata for internal methods
        confidence_data = extract_confidence_scores(logs_content, verbose=verbose)
        
        # Skip file for internal methods if no confidence scores found
        if not confidence_data and all(m != "external" for m in combining_methods):
            if verbose:
                print(f"Skipping {file_id} for internal methods: no confidence scores found")
            continue
        
        # Process internal confidence methods for each checker type
        for checker_type in confidence_checker_types:
            for method in [m for m in combining_methods if m != "external"]:
                combined_confidence = combine_confidence_scores(confidence_data, method)
                
                if combined_confidence is not None:
                    if files_info is not None:
                        file_name = file_info['log_file_path']
                    else:
                        file_name = file
                    results_by_checker_and_method[checker_type][method].append({
                        "file": file_name,
                        "file_id": file_id,
                        "confidence": combined_confidence,
                        "is_correct": is_correct
                    })
    
    # Save full confidence results to single file
    if use_external_confidence:
        for checker_type in confidence_checker_types:
            all_confidence_results[f'external_confidence_{checker_type}'] = external_confidence_results[checker_type]
        
        with open(confidence_results_file, 'w') as f:
            json.dump(all_confidence_results, f, indent=2)
    
    # Calculate HcACC for each checker type, method, and k value
    final_results = {}
    
    for checker_type in confidence_checker_types:
        final_results[checker_type] = {}
        
        for method in combining_methods:
            if method not in results_by_checker_and_method[checker_type] or not results_by_checker_and_method[checker_type][method]:
                print(f"No results for {checker_type} checker with method: {method}")
                continue
                
            method_results = []
            
            for k in k_values:
                hcacc, metrics = calculate_hcacc(results_by_checker_and_method[checker_type][method], k_percent=k)
                
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
                        "Total": len(results_by_checker_and_method[checker_type][method])
                    })
            
            final_results[checker_type][method] = pd.DataFrame(method_results)
    
    # Add confidence results to the final output
    output_data = {
        'confidence_analysis': final_results,
        'confidence_results_file': confidence_results_file
    }
    
    if evaluation_results is not None:
        output_data['based_on_evaluation'] = True
        output_data['evaluation_summary'] = evaluation_results.get('evaluation_summary', {})
    
    return output_data

def get_raw_results_for_analysis(logs_path, answer_file, combining_methods, confidence_checker_types, **kwargs):
    """
    Extract raw results data for enhanced confidence analysis.
    This function processes logs and returns results_by_checker_and_method structure.
    
    Args:
        logs_path (str): Path to the log files directory
        answer_file (str): Path to the answer file
        combining_methods (list): List of methods to combine confidence scores
        confidence_checker_types (list): List of confidence checker types
        **kwargs: Additional parameters for process_logs_and_calculate_hcacc
        
    Returns:
        dict: Raw results organized by checker_type and method
    """
    # Load answers
    with open(answer_file, 'r') as f:
        contents = json.load(f)
    answers = {}
    for i in range(len(contents)):
        answers[contents[i]['id']] = contents[i]['answer']
    
    files = os.listdir(logs_path)
    
    # Initialize results structure
    results_by_checker_and_method = {}
    for checker_type in confidence_checker_types:
        results_by_checker_and_method[checker_type] = {method: [] for method in combining_methods}
    
    # Process each file
    for file in tqdm(files, desc="Extracting raw results for analysis"):
        if not file.endswith('.txt') and not file.endswith('.log'):
            continue
            
        file_id = file.split('.')[0]
        if file_id not in answers:
            continue
        
        # Read the file to get question and prediction
        with open(os.path.join(logs_path, file), 'r') as f:
            logs = f.read()
        
        split_logs = logs.split('\n----------------------------------------------------------\n')
        question = split_logs[0]
        answer = answers[file_id]
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
            prediction = ""
        
        # Extract confidence scores
        confidence_data = extract_confidence_scores(logs_content)
        
        # Skip file if no confidence scores found
        if not confidence_data:
            continue
        
        # Process data for each checker type and method
        for checker_type in confidence_checker_types:
            for method in combining_methods:
                if method == "external":
                    # Skip external method as it requires a separate process
                    continue
                    
                combined_confidence = combine_confidence_scores(confidence_data, method)
                
                if combined_confidence is not None:
                    results_by_checker_and_method[checker_type][method].append({
                        "file": file,
                        "file_id": file_id,
                        "confidence": combined_confidence,
                        "is_correct": is_correct,
                        "question": question,
                        "predicted_answer": prediction.strip(),
                        "correct_answer": answer
                    })
    
    return results_by_checker_and_method