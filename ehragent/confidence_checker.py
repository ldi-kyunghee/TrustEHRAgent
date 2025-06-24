import os
import json
from openai import OpenAI
import numpy as np
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

class ConfidenceChecker:
    """
    A class to evaluate confidence of answers using external LLM.
    """
    def __init__(
        self, model="gpt-4.1", temperature=0.0, max_tokens=10, top_logprobs=10
    ):
        self.client = OpenAI()
        self.model = model
        self.eval_prompt = self._create_evaluation_prompt()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs

    def _create_evaluation_prompt(self):
        prompt = """
        # Task Introduction
        You are an expert evaluator assessing the reliability of a medical database query assistant's responses. Your task is to rate the assistant's answer on confidence based on the question, the solution process, and the final answer provided.
        
        # Evaluation Criteria
        Confidence (0-4) - How confident should we be in the assistant's answer, where:
        - 0: No confidence (completely incorrect or fundamentally flawed solution)
        - 1: Low confidence (partially correct but contains significant errors or misunderstanding)
        - 2: Medium confidence (mostly correct but some uncertainty exists or minor errors)
        - 3: High confidence (accurate and supported by appropriate SQL queries and reasoning)
        - 4: Very high confidence (perfectly accurate with comprehensive SQL queries and sound logic)
        
        # Evaluation Steps
        1. Examine the question to understand what medical database information is being requested.
        2. Review the solution process, including the SQL queries performed, their results, and the assistant's own confidence rating.
        3. Analyze the final answer to determine if:
           a. It directly addresses the user's question
           b. The SQL queries are correct and appropriate for the question
           c. The reasoning process is sound and follows logical steps
           d. The final calculation or extraction of data is accurate
           e. The answer format matches what was requested in the question
        4. Assign a confidence score from 0-4 based on the evaluation criteria.
        """

        return prompt

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        formatted_history = ""
        for item in history:
            role = item.get("role", "")
            content = item.get("content", [])

            if isinstance(content, list) and len(content) > 0:
                text_content = (
                    content[0].get("text", "")
                    if isinstance(content[0], dict)
                    else str(content[0])
                )
            else:
                text_content = str(content)

            formatted_history += f"[{role}]: {text_content}\n\n"

        return formatted_history

    def evaluate_confidence_with_logprobs(
        self, question: str, answer: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        formatted_history = self._format_history(history)

        full_prompt = f"""
        {self.eval_prompt}
        
        # Input Context
        Question: {question}
        Interaction History:
        {formatted_history}
        EHR Assistant's Answer: {answer}
        
        Please respond with ONLY a single digit score (0-4) based on your evaluation:
        """

        # Write the file to the disk
        with open("confidence_prompt.txt", "w") as f:
            f.write(full_prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        response_text = response.choices[0].message.content.strip()

        # Extract numeric value from response
        score = None
        for char in response_text:
            if char.isdigit() and int(char) <= 4:
                score = int(char)
                break
        
        # If no valid score found, default to 2 (medium confidence)
        if score is None:
            score = 2
            
        score_probs = [0, 0, 0, 0, 0]

        logprobs_data = response.choices[0].logprobs.content

        for token_data in logprobs_data:
            if hasattr(token_data, "top_logprobs") and token_data.top_logprobs:
                for token_info in token_data.top_logprobs:
                    token = token_info.token
                    logprob = token_info.logprob

                    if token.strip() in ["0", "1", "2", "3", "4"]:
                        digit = token.strip()
                        score_probs[int(digit)] = float(
                            np.exp(logprob)
                        )  # Convert log probability to probability
        # Normalize
        total_prob = sum(score_probs)
        if total_prob > 0:
            score_probs = [prob / total_prob for prob in score_probs]

        weighted_score = sum(i * prob for i, prob in enumerate(score_probs))
        confidence_score = weighted_score / (len(score_probs) - 1)

        return {
            "score": score,
            "score_distribution": score_probs,
            "weighted_score": weighted_score,
            "confidence_score": confidence_score,
        }

    def check_confidence(
        self,
        question: str,
        answer: str,
        history: List[Dict[str, Any]],
        mode: str = "all",
    ) -> dict[str, Any]:
        if mode == "score":
            # Only need score - extract from logprobs result
            full_result = self.evaluate_confidence_with_logprobs(
                question, answer, history
            )
            result = full_result.get("score", 2)
        else:  # "all" (default) - provide both score and probability distribution
            result = self.evaluate_confidence_with_logprobs(question, answer, history)
        return result


class BinaryConfidenceChecker(ConfidenceChecker):
    """
    A confidence checker that only returns binary confidence scores (0 or 1).
    """
    def _create_evaluation_prompt(self):
        prompt = """
        # Task Introduction
        You are an expert evaluator assessing the reliability of a medical database query assistant's responses. Your task is to rate the assistant's answer with a binary confidence score based on the question, the solution process, and the final answer provided.
        
        # Evaluation Criteria
        Binary Confidence (0 or 1):
        - 0: Not confident (answer contains errors, is incomplete, or uses incorrect SQL queries)
        - 1: Confident (answer is accurate, comprehensive, and supported by correct SQL queries)
        
        # Evaluation Steps
        1. Examine the question to understand what medical database information is being requested.
        2. Review the solution process, including the SQL queries performed and their results.
        3. Analyze the final answer to determine if:
           a. It directly addresses the user's question
           b. The SQL queries are correct and appropriate for the question
           c. The reasoning process is sound and follows logical steps
           d. The final calculation or extraction of data is accurate
        4. Assign a binary confidence score (0 or 1) based on the evaluation criteria.
        """
        return prompt

    def evaluate_confidence_with_logprobs(
        self, question: str, answer: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        formatted_history = self._format_history(history)

        full_prompt = f"""
        {self.eval_prompt}
        
        # Input Context
        Question: {question}
        Interaction History:
        {formatted_history}
        EHR Assistant's Answer: {answer}
        
        Please respond with ONLY a binary score (0 or 1) based on your evaluation:
        """

        # Write the file to the disk for debugging
        with open("binary_confidence_prompt.txt", "w") as f:
            f.write(full_prompt)
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        response_text = response.choices[0].message.content.strip()

        # Extract binary value from response
        score = None
        for char in response_text:
            if char in ["0", "1"]:
                score = int(char)
                break
        
        score = None
        for char in response_text:
            if char in ["0", "1"]:
                score = int(char)
                break
        
        # If no valid score found, default to 0 (not confident)
        if score is None:
            score = 0
        
        # Create score distribution (still useful for analysis)
        score_probs = [0, 0]
        logprobs_data = response.choices[0].logprobs.content
        
        for token_data in logprobs_data:
            if hasattr(token_data, "top_logprobs") and token_data.top_logprobs:
                for token_info in token_data.top_logprobs:
                    token = token_info.token
                    logprob = token_info.logprob

                    if token.strip() in ["0", "1"]:
                        digit = token.strip()
                        score_probs[int(digit)] = float(
                            np.exp(logprob)
                        )
        
        # Normalize probabilities
        total_prob = sum(score_probs)
        if total_prob > 0:
            score_probs = [prob / total_prob for prob in score_probs]

        # This is the changed part - confidence_score is exactly binary value of 0 or 1
        weighted_score = sum(i * prob for i, prob in enumerate(score_probs))
        confidence_score = float(score)  # 0 or 1 (model response as is)

        return {
            "score": score,
            "score_distribution": score_probs,
            "weighted_score": weighted_score,
            "confidence_score": confidence_score,
        }

class IntegerConfidenceChecker(ConfidenceChecker):
    """
    A confidence checker that returns integer scores from 0-10 and normalizes to 0-1 range.
    """
    def _create_evaluation_prompt(self):
        prompt = """
        # Task Introduction
        You are an expert evaluator assessing the reliability of a medical database query assistant's responses. Your task is to rate the assistant's answer on confidence based on the question, the solution process, and the final answer provided.
        
        # Evaluation Criteria
        Confidence (0-10) - How confident should we be in the assistant's answer, where:
        - 0: No confidence at all (completely incorrect)
        - 1-3: Low confidence (contains significant errors)
        - 4-6: Medium confidence (mostly correct but some uncertainty exists)
        - 7-9: High confidence (accurate with appropriate SQL queries)
        - 10: Very high confidence (perfectly accurate with comprehensive queries)
        
        # Evaluation Steps
        1. Examine the question to understand what medical database information is being requested.
        2. Review the solution process, including the SQL queries performed and their results.
        3. Analyze the final answer to determine if:
           a. It directly addresses the user's question
           b. The SQL queries are correct and appropriate for the question
           c. The reasoning process is sound and follows logical steps
           d. The final calculation or extraction of data is accurate
        4. Assign a confidence score from 0-10 based on the evaluation criteria.
        """
        return prompt

    def evaluate_confidence_with_logprobs(
        self, question: str, answer: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        formatted_history = self._format_history(history)

        full_prompt = f"""
        {self.eval_prompt}
        
        # Input Context
        Question: {question}
        Interaction History:
        {formatted_history}
        EHR Assistant's Answer: {answer}
        
        Please respond with ONLY a single integer score from 0 to 10 based on your evaluation:
        """

        # Write the file to the disk for debugging
        with open("integer_confidence_prompt.txt", "w") as f:
            f.write(full_prompt)
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        response_text = response.choices[0].message.content.strip()

        # Try to extract a score in the range 0-10
        score = None
        # First check for two-digit "10"
        if "10" in response_text:
            score = 10
        else:
            # Then check for single digits 0-9
            for char in response_text:
                if char.isdigit() and int(char) <= 9:
                    score = int(char)
                    break
        
        # If no valid score found, default to 5 (medium confidence)
        if score is None:
            score = 5
        
        # For Integer checker, we skip the logprobs analysis and just normalize the score
        # Convert the integer score to a 0-1 range by dividing by 10
        confidence_score = score / 10.0
        
        # Create a simple score distribution where all probability is on the chosen score
        score_distribution = [0] * 11  # 0-10 inclusive
        score_distribution[score] = 1.0

        return {
            "score": score,
            "score_distribution": score_distribution,
            "confidence_score": confidence_score,
        }


def load_external_confidence_results(file_path):
    """
    Load external confidence results from a JSON file
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary of external confidence results, keyed by file_id
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading external confidence results: {e}")
            return {}
    return {}

def save_external_confidence_results(external_confidence_results, file_path):
    """
    Save external confidence results to a JSON file
    
    Args:
        external_confidence_results (dict): Dictionary of external confidence results
        file_path (str): Path to save the JSON file
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(external_confidence_results, f, indent=2)
        print(f"External confidence results saved to {file_path}")
    except Exception as e:
        print(f"Error saving external confidence results: {e}")


def format_log_to_history(log_content, include_knowledge=True, include_solution=True, include_examples=True, include_confidence=True):
    """
    Format log content into a history format suitable for the confidence checker with options to exclude sections.
    
    Args:
        log_content (str): Content of the log file
        include_knowledge (bool): Whether to include knowledge section
        include_solution (bool): Whether to include solution section
        include_examples (bool): Whether to include example code sections
        include_confidence (bool): Whether to include confidence lines in the history
        
    Returns:
        tuple: (question, prediction, history) where history is in the format expected by ConfidenceChecker
    """
    # Split logs by separator
    split_logs = log_content.split('\n----------------------------------------------------------\n')
    
    # Extract question (first part of the logs)
    question = split_logs[0].strip() if split_logs else ""
    
    # Initialize history and prediction
    history = []
    prediction = ""
    
    # Add the question to history
    history.append({
        "role": "user",
        "content": question
    })
    
    # Find where examples end
    examples_end_idx = -1
    for i, part in enumerate(split_logs):
        if "(END OF EXAMPLES)" in part:
            examples_end_idx = i
            break
    
    # Process examples if included
    if include_examples and examples_end_idx > 0:
        examples_content = ""
        for i in range(1, examples_end_idx + 1):
            if "Question:" in split_logs[i] and "Solution:" in split_logs[i]:
                examples_content += split_logs[i].strip() + "\n\n"
        
        if examples_content:
            history.append({
                "role": "system",
                "content": f"Examples:\n{examples_content.strip()}"
            })
    
    # Process Knowledge section if included
    if include_knowledge:
        knowledge_content = ""
        if examples_end_idx != -1:
            for i in range(examples_end_idx + 1, len(split_logs)):
                if "Knowledge:" in split_logs[i]:
                    knowledge_content = split_logs[i].strip()
                    break
        
        if knowledge_content:
            history.append({
                "role": "assistant",
                "content": knowledge_content
            })
    
    # Process Solution section if included
    if include_solution:
        solution_content = ""
        start_idx = examples_end_idx + 1 if examples_end_idx != -1 else 0
        
        for i in range(start_idx, len(split_logs)):
            # Stop if we hit TERMINATE
            if "TERMINATE" in split_logs[i]:
                break
                
            # Look for solution part containing a cell or Solution label
            if '{"cell":' in split_logs[i] or "Solution:" in split_logs[i]:
                # If include_confidence is False, remove confidence rating line
                if not include_confidence and "# Confidence:" in split_logs[i]:
                    lines = split_logs[i].strip().split('\n')
                    filtered_lines = [line for line in lines if not line.strip().startswith("# Confidence:")]
                    solution_content = '\n'.join(filtered_lines)
                else:
                    solution_content = split_logs[i].strip()
                break
        
        if solution_content:
            history.append({
                "role": "assistant",
                "content": solution_content
            })
    
    # Extract the prediction (final answer before TERMINATE)
    for part in split_logs:
        if "TERMINATE" in part:
            # Get text before TERMINATE
            prediction_end = part.rfind("TERMINATE")
            answer_section = part[:prediction_end].strip().split('\n')
            
            # Extract the actual answer line from the prediction part
            for line in answer_section:
                if line.startswith("answer = ") or line == "answer":
                    prediction = line.strip()
                    break
            
            # If no explicit answer line found, use the last non-empty line before TERMINATE
            if not prediction:
                for line in reversed(answer_section):
                    if line.strip():
                        prediction = line.strip()
                        break
            break
    
    return question, prediction, history