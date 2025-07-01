import time
from typing import Any, Dict, List, Optional, Union, Callable, Literal
import logging
import asyncio
import json
import os
import sys
# Add the parent directory to the Python path so tools can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autogen import Agent, UserProxyAgent, ConversableAgent
from termcolor import colored
import Levenshtein
from ehragent.confidence_scorer import AgentStepConfidenceScorer

logger = logging.getLogger(__name__)

class MedAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        config_list: Optional[List[Dict]] = None,
        turn_step_wise_confidence: Optional[bool] = True,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.config_list = config_list
        self.question = ''
        self.code = ''
        self.knowledge = ''
        self.conversation_id = None
        self.confidence_scorer = None
        self.last_response = None  # Store last OpenAI response for logprob analysis
        self.turn_step_wise_confidence = turn_step_wise_confidence
        
    def retrieve_knowledge(self, config, query):
        # import prompt
        if self.dataset == 'mimic_iii':
            from ehragent.prompts_mimic import RetrKnowledge
        else:
            from ehragent.prompts_eicu import RetrKnowledge
        
        from ehragent.model_client import create_model_client
        
        # Returns the related information to the given query.
        patience = 2
        sleep_time = 30
        
        query_message = RetrKnowledge.format(question=query)
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                    {"role":"user","content": query_message}]
        
        # Use our model client instead of direct OpenAI client
        client = create_model_client(config)
        
        while patience > 0:
            patience -= 1
            try:
                response = client.chat_completion(
                    messages=messages,
                    temperature=0,
                    max_tokens=2000,
                    top_p=0.95
                )
                content = response.get("content")
                if content is not None:
                    prediction = content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to retrieve related knowledge, please try again later."

    def retrieve_examples(self, query):
        levenshtein_dist = {}
        for i in range(len(self.memory)):
            question = self.memory[i]["question"]
            levenshtein_dist[i] = Levenshtein.distance(query, question)
        levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
        selected_indexes = [levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))]
        examples = []
        for i in selected_indexes:
            template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(self.memory[i]["question"], self.memory[i]["knowledge"], self.memory[i]["code"])
            examples.append(template)
        examples = '\n'.join(examples)
        return examples

    def generate_init_message(self, **context):
        # import prompt
        if self.dataset == 'mimic_iii':
            from ehragent.prompts_mimic import EHRAgent_Message_Prompt
        else:
            from ehragent.prompts_eicu import EHRAgent_Message_Prompt
        self.question = context["message"]
        knowledge = self.retrieve_knowledge(self.config_list[0], context["message"])
        self.knowledge = knowledge

        examples = self.retrieve_examples(context["message"])

        # Modify prompt based on confidence setting
        if self.turn_step_wise_confidence:
            init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=context["message"])
        else:
            # Use prompt without confidence rating instruction
            prompt_without_confidence = EHRAgent_Message_Prompt.replace(
                "IMPORTANT: Begin your code solution with a confidence rating comment in the format '# Confidence: X' where X is a number from 0-10.\n", ""
            )
            init_message = prompt_without_confidence.format(examples=examples, knowledge=knowledge, question=context["message"])
        return init_message
    
    def send(self, message: Union[Dict, str], recipient: Agent, request_reply: Optional[bool]=None, silent: Optional[bool]=False):
        valid = self._append_oai_message(message, "assistant", recipient, True)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def initiate_chat(self, recipient: "ConversableAgent", clear_history: Optional[bool]=True, silent: Optional[bool]=False, **context,):
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
            
        self._process_received_message(message, sender, silent)
        
        # Score the received message
        if self.confidence_scorer:
            message_content = message if isinstance(message, str) else message.get("content", "")
            if message_content:
                self.confidence_scorer.score_agent_step(
                    step_type="message_received",
                    message_content=message_content,
                    openai_response=self.last_response
                )
        
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    def error_debugger(self, config, code, error_info):
        # import prompt
        if self.dataset == 'mimic_iii':
            from ehragent.prompts_mimic import CodeDebugger
        else:
            from ehragent.prompts_eicu import CodeDebugger
        
        from ehragent.model_client import create_model_client
        
        # Returns the related information to the given query.
        patience = 2
        sleep_time = 30
        
        query_message = CodeDebugger.format(question=self.question, code=code, error_info=error_info)
        messages = [{"role":"system","content":"You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors."},
                    {"role":"user","content": query_message}]
        
        # Use our model client instead of direct OpenAI client
        client = create_model_client(config)
        
        while patience > 0:
            patience -= 1
            try:
                response = client.chat_completion(
                    messages=messages,
                    temperature=0,
                    max_tokens=2000,
                    top_p=0.95
                )
                content = response.get("content")
                if content is not None:
                    prediction = content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to diagnose the reasons to the errors."

    def execute_function(self, func_call):
        """Execute a function call and return the result.

        Override this function to modify the way to execute a function call.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" with keys "name" and "arguments".

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".
        """
        
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                arguments_string = func_call["arguments"].split(': "')[-1]
                arguments_string = arguments_string.split('", ')[0]
                arguments = {"cell": arguments_string}
                # content = f"Error: {e}\n You argument should follow json format."
                content = f"Error: {e}\n There might be compilation errors in the code. Please check the code and try again."

            # Try to execute the function
            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                self.code = arguments["cell"]
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."
        if "error" in content or "Error" in content:
            reasons = self.error_debugger(self.config_list[0], self.code, content)
            content = content + '\nPotential Reasons: ' + reasons

        # Score the function execution step
        if self.confidence_scorer:
            self.confidence_scorer.score_agent_step(
                step_type="function_execution",
                func_call=func_call,
                execution_result=str(content)
            )

        # Check backend and use appropriate role
        backend = "openai"  # default
        if hasattr(self, 'config_list') and self.config_list:
            backend = self.config_list[0].get('backend', 'openai')
        
        # Together AI and Gemini use 'tool' role instead of 'function'
        role = "tool" if backend in ["together", "google", "gemini"] else "function"
        
        return is_exec_success, {
            "name": func_name,
            "role": role,
            "content": str(content),
        }
    
    def update_memory(self, num_shots, memory):
        self.num_shots = num_shots
        self.memory = memory

    def register_dataset(self, dataset):
        self.dataset = dataset
    
    def initialize_confidence_scorer(self, conversation_id: str, confidence_log_dir: str = "confidence_logs"):
        """Initialize confidence scorer with conversation ID"""
        self.conversation_id = conversation_id
        self.confidence_scorer = AgentStepConfidenceScorer(conversation_id, confidence_log_dir)
    
    def get_confidence_scores(self) -> List[Dict[str, Any]]:
        """Get all confidence scores for this conversation"""
        if self.confidence_scorer:
            return self.confidence_scorer.get_step_scores()
        return []
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get confidence summary for this conversation"""
        if self.confidence_scorer:
            return self.confidence_scorer.get_conversation_confidence_summary()
        return {}
    
    def save_final_confidence_summary(self, question: str = None, answer: str = None, 
                                    conversation_logs: List[str] = None):
        """Save final confidence summary"""
        if self.confidence_scorer:
            self.confidence_scorer.save_conversation_summary(question, answer, conversation_logs)
    
