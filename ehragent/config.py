import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

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

def get_model_config(model: str) -> dict:
    """Get model configuration based on model name and type.
    
    Args:
        model: Model name to use (e.g., 'gpt-3.5-turbo', 'llama-2-7b', 'qwen-14b')
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If required environment variables are not set
    """
    model_lower = model.lower()
    
    # OpenAI models
    if model_lower.startswith('gpt-') or model_lower in ['gpt-4.1-mini']:
        if model == 'gpt-4.1-mini':
            return {
                "model": "gpt-4.1-mini",
                "api_key": get_env_var("OPENAI_API_KEY"),
                "api_type": "openai",
                "backend": "openai"
            }
        elif model.startswith('gpt-'):
            return {
                "model": model,
                "api_key": get_env_var("OPENAI_API_KEY"),
                "api_type": "openai",
                "backend": "openai"
            }
    
    # Claude models (if using Anthropic)
    elif 'claude' in model_lower:
        return anthropic_config(model)
    
    # Gemini models (if using Google)
    elif 'gemini' in model_lower:
        return gemini_config(model)
    
    # Default to Together AI for all other models
    else:
        return together_config(model)


def together_config(model: str) -> dict:
    """Get Together AI model configuration."""
    return {
        "model": model,
        "api_key": get_env_var("TOGETHER_API_KEY"),
        "api_type": "together",
        "backend": "together",
        "base_url": get_env_var("TOGETHER_BASE_URL", "https://api.together.xyz", required=False)
    }

def anthropic_config(model: str) -> dict:
    """Get Anthropic Claude configuration."""
    return {
        "model": model,
        "api_key": get_env_var("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
        "backend": "anthropic"
    }

def gemini_config(model: str) -> dict:
    """Get Google Gemini configuration."""
    return {
        "model": model,
        "api_key": get_env_var("GOOGLE_API_KEY"),
        "api_type": "google",
        "backend": "google",
        "base_url": get_env_var("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/", required=False)
    }


# Keep the old function for backward compatibility
def openai_config(model: str) -> dict:
    """Backward compatibility wrapper for OpenAI configs only."""
    if model == 'gpt-4.1-mini':
        config = {
            "model": "gpt-4.1-mini",
            "api_key": get_env_var("OPENAI_API_KEY"),
            "api_type": "openai",
            "backend": "openai"
        }
    elif model.startswith('gpt-'):
        config = {
            "model": model,
            "api_key": get_env_var("OPENAI_API_KEY"),
            "api_type": "openai",
            "backend": "openai"
        }
    else:
        # Azure OpenAI
        config = {
            "model": model,
            "api_key": get_env_var("AZURE_OPENAI_API_KEY", required=False) or get_env_var("OPENAI_API_KEY"),
            "base_url": get_env_var("AZURE_OPENAI_ENDPOINT", required=False),
            "api_version": get_env_var("AZURE_OPENAI_API_VERSION", "2023-12-01-preview", required=False),
            "api_type": "azure",
            "backend": "openai"
        }
        
    return config

def llm_config_list(seed: int, config_list: list) -> dict:
    # Clean config_list to only include OpenAI-compatible fields
    clean_config_list = []
    for config in config_list:
        clean_config = {}
        
        # Only include OpenAI-compatible fields
        openai_fields = ['model', 'api_key', 'base_url', 'api_version', 'api_type']
        for field in openai_fields:
            if field in config:
                clean_config[field] = config[field]
        
        # Handle Together AI: convert to OpenAI-compatible format
        if config.get('backend') == 'together':
            clean_config['base_url'] = config.get('base_url', 'https://api.together.xyz/v1')
            # Together AI uses OpenAI-compatible endpoint
            if not clean_config['base_url'].endswith('/v1'):
                clean_config['base_url'] += '/v1'
        
        clean_config_list.append(clean_config)
    
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
        "config_list": clean_config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
        "logprobs": True,  # Enable logprobs for confidence calculation
        "top_logprobs": 5,  # Get top 5 alternative tokens for uncertainty analysis
    }
    return llm_config_list