import sys
import openai
import autogen
import time
import os
# Add the parent directory to the Python path so tools can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ehragent.config import openai_config
from openai import AzureOpenAI
import traceback

def run_code(cell):
    """
    Returns the path to the python interpreter.
    """
    print(f"\n=== FUNCTION CALLED: run_code ===")
    print(f"Input cell: {cell[:100]}..." if len(cell) > 100 else f"Input cell: {cell}")
    
    # Additional cleanup for any remaining artifacts
    if cell.startswith("```python"):
        cell = cell.replace("```python", "").replace("```", "").strip()
    elif cell.startswith("```"):
        cell = cell.replace("```", "").strip()
    if "TERMINATE" in cell:
        cell = cell.replace("TERMINATE", "").strip()
    
    # import prompts
    from ehragent.prompts_mimic import CodeHeader
    try:
        global_var = {"answer": 0}
        exec(CodeHeader+cell, global_var)
        cell = "\n".join([line for line in cell.split("\n") if line.strip() and not line.strip().startswith("#")])
        cell_lines = cell.split('\n')
        if not cell_lines or not 'answer' in cell_lines[-1]:
            return "Please save the answer to the question in the variable 'answer'."
        return str(global_var['answer'])
    except Exception as e:
        error_info = traceback.format_exc()
        code = CodeHeader + cell
        if "SyntaxError" in str(repr(e)):
            error_line = str(repr(e))
            
            error_type = error_line.split('(')[0]
            # then parse out the error message
            error_message = error_line.split(',')[0].split('(')[1]
            # then parse out the error line
            error_line = error_line.split('"')[1]
        elif "KeyError" in str(repr(e)):
            code = code.split('\n')
            key = str(repr(e)).split("'")[1]
            error_type = str(repr(e)).split('(')[0]
            for i in range(len(code)):
                if key in code[i]:
                    error_line = code[i]
            error_message = str(repr(e))
        elif "TypeError" in str(repr(e)):
            error_type = str(repr(e)).split('(')[0]
            error_message = str(e)
            function_mapping_dict = {"get_value": "GetValue", "data_filter": "FilterDB", "db_loader": "LoadDB", "sql_interpreter": "SQLInterpreter", "date_calculator": "Calendar"}
            error_key = ""
            for key in function_mapping_dict.keys():
                if key in error_message:
                    error_message = error_message.replace(key, function_mapping_dict[key])
                    error_key = function_mapping_dict[key]
            code = code.split('\n')
            error_line = ""
            for i in range(len(code)):
                if error_key in code[i]:
                    error_line = code[i]
        else:
            error_type = ""
            error_message = str(repr(e)).split("('")[-1].split("')")[0]
            error_line = ""
        # use one sentence to introduce the previous parsed error information
        if error_type != "" and error_line != "":
            error_info = f'{error_type}: {error_message}. The error messages occur in the code line "{error_line}".'
        else:
            error_info = f'Error: {error_message}.'
        error_info += '\nPlease make modifications accordingly and make sure the rest code works well with the modification.'

        return error_info


def terminate(message="Task completed"):
    """
    Terminate the conversation when the task is completed.
    
    Args:
        message (str): Optional message to include with termination
    
    Returns:
        str: Termination message
    """
    print(f"\n=== FUNCTION CALLED: terminate ===")
    print(f"Termination message: {message}")
    return "TERMINATE"


def llm_agent(config_list):
    llm_config = {
        "functions": [
            {
                "name": "python",
                "description": "run cell in ipython and return the execution result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python cell to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "request_timeout": 120,
    }
    chatbot = autogen.AssistantAgent(
        name="chatbot",
        system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
        llm_config=llm_config,
    )
    return chatbot
