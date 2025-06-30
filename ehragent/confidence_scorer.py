import json
import math
import os
import time
from typing import Dict, Any, Optional, List, Union
import openai
from openai import OpenAI


class AgentStepConfidenceScorer:
    """Confidence scorer for EhrAgent step responses using LogU approach"""
    
    def __init__(self, conversation_id: str = None, confidence_log_dir: str = "confidence_logs"):
        self.step_scores = []
        self.conversation_id = conversation_id or f"conv_{int(time.time())}"
        self.confidence_log_dir = confidence_log_dir
        self.ensure_log_dir()
        
    def calculate_confidence_scores(self, func_call: Optional[Dict] = None, message_content: Optional[str] = None, 
                                  openai_response: Optional[Any] = None) -> Dict[str, Any]:
        """Calculate confidence scores from function calls, message content, and OpenAI response"""
        result = {}

        # Extract information from function call (for code execution steps)
        if func_call:
            func_name = func_call.get("name", "")
            arguments = func_call.get("arguments", "")
            
            result["function_name"] = func_name
            result["step_type"] = "function_execution"
            
            # Try to extract code from arguments
            try:
                if isinstance(arguments, dict):
                    code = arguments.get("cell", "")
                elif isinstance(arguments, str):
                    args_dict = json.loads(arguments)
                    code = args_dict.get("cell", "")
                else:
                    code = str(arguments)
                    
                result["code"] = code
                
                # Extract confidence from code comments
                if code:
                    confidence_score = self._extract_embedded_confidence(code)
                    result.update(confidence_score)
                    
            except (json.JSONDecodeError, Exception):
                pass

        # Extract information from message content (for agent responses)
        if message_content:
            result["message_content"] = message_content
            if not func_call:
                result["step_type"] = "agent_response"
            
            # Extract confidence from message
            confidence_score = self._extract_embedded_confidence(message_content)
            result.update(confidence_score)
            
        # Calculate logprob-based confidence if OpenAI response available
        if openai_response:
            logprob_confidence = self._calculate_logprob_confidence(openai_response, message_content)
            if logprob_confidence:  # Only add if we got actual logprob data
                result.update(logprob_confidence)
            
        return result
    
    def _calculate_logprob_confidence(self, response: Any, target_text: Optional[str] = None) -> Dict[str, Any]:
        """Calculate logprob-based confidence using original LogU approach"""
        result = {}
        
        # Handle different response formats from different backends
        backend = getattr(response, 'backend', 'unknown')
        result["response_backend"] = backend
        
        # OpenAI format
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            
            if hasattr(choice, "logprobs") and choice.logprobs:
                if hasattr(choice.logprobs, "content"):
                    logprobs_data = choice.logprobs.content
                    if logprobs_data:
                        return self._process_openai_logprobs(logprobs_data, target_text)
        
        # Dictionary format (from our model client)
        elif isinstance(response, dict):
            raw_response = response.get("raw_response")
            backend = response.get("backend", "unknown")
            result["response_backend"] = backend
            
            # Try OpenAI format in raw_response
            if hasattr(raw_response, 'choices') and len(raw_response.choices) > 0:
                choice = raw_response.choices[0]
                if hasattr(choice, "logprobs") and choice.logprobs:
                    if hasattr(choice.logprobs, "content"):
                        logprobs_data = choice.logprobs.content
                        if logprobs_data:
                            return self._process_openai_logprobs(logprobs_data, target_text)
            
            # Together AI format (similar to OpenAI)
            elif backend == "together" and raw_response is not None:
                # Together AI uses OpenAI-compatible format
                if hasattr(raw_response, 'choices') and len(raw_response.choices) > 0:
                    choice = raw_response.choices[0]
                    if hasattr(choice, "logprobs") and choice.logprobs:
                        if hasattr(choice.logprobs, "content"):
                            logprobs_data = choice.logprobs.content
                            if logprobs_data:
                                return self._process_openai_logprobs(logprobs_data, target_text)
        
        return result
    
    def _process_openai_logprobs(self, logprobs_data: List, target_text: Optional[str] = None) -> Dict[str, Any]:
        """Process OpenAI format logprobs"""
        result = {}
        
        # Use all tokens if no specific target text
        target_tokens = logprobs_data
        
        # If target text provided, try to find relevant tokens
        if target_text:
            relevant_tokens = []
            current_text = ""
            
            for token_info in logprobs_data:
                # Try different possible attribute names for token text
                token_text = ""
                if hasattr(token_info, "token"):
                    token_text = token_info.token
                elif hasattr(token_info, "text"):
                    token_text = token_info.text
                elif hasattr(token_info, "bytes"):
                    token_text = token_info.bytes.decode('utf-8', errors='ignore')
                    
                if token_text:
                    current_text += token_text
                    if target_text in current_text:
                        relevant_tokens.append(token_info)
                        
            if relevant_tokens:
                target_tokens = relevant_tokens

        # Basic confidence metrics using target tokens
        token_logprobs = [token_info.logprob for token_info in target_tokens if hasattr(token_info, 'logprob')]
        if token_logprobs:
            result.update({
                "first_token_prob": math.exp(token_logprobs[0]),
                "average_token_prob": math.exp(sum(token_logprobs) / len(token_logprobs)),
                "perplexity": math.exp(-sum(token_logprobs) / len(token_logprobs)),
                "total_tokens": len(token_logprobs)
            })

        # LogU implementation
        logu_scores = []
        for token_info in target_tokens:
            if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
                try:
                    # Extract logprob values from TopLogprob objects
                    top_logits = []
                    for top_logprob in token_info.top_logprobs:
                        if hasattr(top_logprob, "logprob"):
                            top_logits.append(math.exp(top_logprob.logprob))
                        elif hasattr(top_logprob, "value"):
                            top_logits.append(math.exp(top_logprob.value))
                        elif hasattr(top_logprob, "score"):
                            top_logits.append(math.exp(top_logprob.score))
                            
                    if not top_logits:
                        continue

                    # Calculate Aleatoric Uncertainty (AU)
                    total = sum(top_logits)
                    if total <= 0:
                        continue

                    probabilities = [logit / total for logit in top_logits]
                    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probabilities)
                    max_entropy = math.log(len(probabilities))
                    AU = entropy / max_entropy if max_entropy > 0 else 0

                    # Calculate Epistemic Uncertainty (EU)
                    evidence_strength = sum(top_logits)
                    EU = 1.0 / (1.0 + evidence_strength)

                    # Calculate LogU score
                    logu_score = AU * EU
                    logu_scores.append((AU, EU, logu_score))
                    
                except Exception:
                    continue

        if logu_scores:
            # Get top-k most uncertain tokens
            k = min(5, len(logu_scores))
            sorted_scores = sorted(logu_scores, key=lambda x: x[2], reverse=True)[:k]

            # Calculate average LogU for most uncertain tokens
            avg_AU = sum(score[0] for score in sorted_scores) / k
            avg_EU = sum(score[1] for score in sorted_scores) / k
            avg_logu = sum(score[2] for score in sorted_scores) / k

            # Map to confidence score
            confidence = 1.0 - avg_logu

            result.update({
                "logu_confidence": confidence,
                "aleatoric_uncertainty": avg_AU,
                "epistemic_uncertainty": avg_EU,
                "uncertain_tokens_count": len(logu_scores)
            })

        return result
    
    
    def score_agent_step(self, step_type: str, func_call: Optional[Dict] = None, 
                        message_content: Optional[str] = None, execution_result: Optional[str] = None,
                        openai_response: Optional[Any] = None) -> Dict[str, Any]:
        """Score a single agent step"""
        
        step_score = {
            "step_type": step_type,
            "timestamp": self._get_timestamp(),
            "step_number": len(self.step_scores) + 1,
            "confidence_scores": {}
        }
        
        # Calculate confidence scores (both embedded and logprob-based)
        confidence_scores = self.calculate_confidence_scores(func_call, message_content, openai_response)
        step_score["confidence_scores"] = confidence_scores
        
        # Add execution result if available
        if execution_result:
            step_score["execution_result"] = execution_result
            
        # Add raw data for debugging (but exclude large response objects)
        if func_call:
            step_score["func_call"] = func_call
        if message_content:
            step_score["message_content"] = message_content[:500] + "..." if len(message_content) > 500 else message_content
            
        self.step_scores.append(step_score)
        
        # Save to separate confidence log immediately
        self.save_step_confidence(step_score)
        
        return step_score
    
    def _extract_embedded_confidence(self, content: str) -> Dict[str, Any]:
        """Extract confidence from embedded comments like # Confidence: X"""
        result = {}
        
        # Look for confidence patterns
        confidence_patterns = [
            r"# Confidence: (\d+(?:\.\d+)?)",
            r"Confidence: (\d+(?:\.\d+)?)",
            r"confidence:\s*(\d+(?:\.\d+)?)"
        ]
        
        import re
        for pattern in confidence_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                confidence_value = float(match.group(1))
                # Normalize to 0-1 scale if needed
                if confidence_value > 1:
                    confidence_value = confidence_value / 10.0
                result["verbal_confidence"] = confidence_value
                break
                
        return result
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return str(time.time())
    
    def ensure_log_dir(self):
        """Ensure confidence log directory exists"""
        if not os.path.exists(self.confidence_log_dir):
            os.makedirs(self.confidence_log_dir)
    
    def save_step_confidence(self, step_score: Dict[str, Any]):
        """Save individual step confidence to separate file"""
        step_file = os.path.join(self.confidence_log_dir, f"{self.conversation_id}_steps.jsonl")
        
        # Convert to JSON line format
        with open(step_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(step_score, ensure_ascii=False) + '\n')
    
    def save_conversation_summary(self, question: str = None, answer: str = None, 
                                conversation_logs: List[str] = None):
        """Save final conversation confidence summary"""
        summary_file = os.path.join(self.confidence_log_dir, f"{self.conversation_id}_summary.json")
        
        summary = {
            "conversation_id": self.conversation_id,
            "timestamp": self._get_timestamp(),
            "question": question,
            "ground_truth_answer": answer,
            "total_steps": len(self.step_scores),
            "confidence_summary": self.get_conversation_confidence_summary(),
            "step_scores": self.step_scores
        }
        
        # Add conversation logs if provided
        if conversation_logs:
            summary["conversation_logs"] = conversation_logs
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def get_confidence_log_path(self) -> str:
        """Get path for confidence logs"""
        return os.path.join(self.confidence_log_dir, f"{self.conversation_id}_steps.jsonl")
    
    def get_step_scores(self) -> List[Dict[str, Any]]:
        """Get all recorded step scores"""
        return self.step_scores
    
    def get_conversation_confidence_summary(self) -> Dict[str, Any]:
        """Get overall confidence summary for the conversation"""
        if not self.step_scores:
            return {}
            
        # Aggregate confidence scores
        all_confidences = []
        step_types = set()
        
        for step in self.step_scores:
            step_types.add(step["step_type"])
            conf_scores = step["confidence_scores"]
            
            if "logu_confidence" in conf_scores:
                all_confidences.append(conf_scores["logu_confidence"])
            elif "verbal_confidence" in conf_scores:
                all_confidences.append(conf_scores["verbal_confidence"])
        
        summary = {
            "total_steps": len(self.step_scores),
            "step_types": list(step_types),
        }
        
        if all_confidences:
            summary.update({
                "avg_confidence": sum(all_confidences) / len(all_confidences),
                "min_confidence": min(all_confidences),
                "max_confidence": max(all_confidences),
                "confidence_std": self._calculate_std(all_confidences)
            })
            
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)