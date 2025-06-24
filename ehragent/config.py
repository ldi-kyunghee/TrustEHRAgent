import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_var(key: str, default: str = None, required: bool = True) -> str:
    """Get environment variable with error handling.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable '{key}' not found")
    return value

def validate_api_config(config: dict) -> bool:
    """Validate API configuration.
    
    Args:
        config: API configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['model', 'api_key', 'api_type']
    return all(key in config and config[key] for key in required_keys)

def openai_config(model: str) -> dict:
    """Get OpenAI configuration from environment variables.
    
    Args:
        model: Model name to use
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If required environment variables are not set
    """
    if model == 'gpt-4.1-mini':
        config = {
            "model": "gpt-4.1-mini",
            "api_key": get_env_var("OPENAI_API_KEY"),
            "api_type": "open_ai"
        }
    elif model.startswith('gpt-'):
        # Generic GPT model configuration
        config = {
            "model": model,
            "api_key": get_env_var("OPENAI_API_KEY"),
            "api_type": "open_ai"
        }
    else:
        # Azure or custom model configuration
        config = {
            "model": model,
            "api_key": get_env_var("AZURE_OPENAI_API_KEY", required=False) or get_env_var("OPENAI_API_KEY"),
            "base_url": get_env_var("AZURE_OPENAI_ENDPOINT", required=False),
            "api_version": get_env_var("AZURE_OPENAI_API_VERSION", "2023-12-01-preview", required=False),
            "api_type": get_env_var("API_TYPE", "open_ai", required=False)
        }
        
    if not validate_api_config(config):
        raise ValueError(f"Invalid API configuration for model '{model}'")
        
    return config

def llm_config_list(seed: int, config_list: list) -> dict:
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list