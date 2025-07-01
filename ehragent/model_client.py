"""
Multi-backend model client for supporting OpenAI, Llama, Qwen, and other models
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod


class ModelClient(ABC):
    """Abstract base class for model clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config["model"]
        
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model backend is available"""
        pass


class OpenAIClient(ModelClient):
    """OpenAI API client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config["api_key"],
                base_url=config.get("base_url"),
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        # Convert to standard format
        message = response.choices[0].message
        result = {
            "role": "assistant",
            "raw_response": response,
            "backend": "openai"
        }
        
        # Check for function calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Convert tool_calls to function_call format for compatibility
            tool_call = message.tool_calls[0]
            result["function_call"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
            result["tool_calls"] = message.tool_calls
            result["content"] = message.content  # May be None for function calls
        elif hasattr(message, 'function_call') and message.function_call:
            # Handle legacy function_call format
            result["function_call"] = {
                "name": message.function_call.name,
                "arguments": message.function_call.arguments
            }
            result["content"] = message.content
        else:
            result["content"] = message.content
            # Ensure function_call is None when there are no function calls
            result["function_call"] = None
            
        return result
    
    def is_available(self) -> bool:
        try:
            # Test API key by making a simple request
            self.client.models.list()
            return True
        except Exception:
            return False




class TogetherAIClient(ModelClient):
    """Together AI API client using OpenAI client with custom base_url"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from openai import OpenAI
            base_url = config.get("base_url", "https://api.together.xyz")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            
            self.client = OpenAI(
                api_key=config["api_key"],
                base_url=base_url
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Together AI API via OpenAI client"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        # Convert to standard format (same as OpenAI)
        message = response.choices[0].message
        result = {
            "role": "assistant",
            "raw_response": response,
            "backend": "together"
        }
        
        # Check for function calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Convert tool_calls to function_call format for compatibility
            tool_call = message.tool_calls[0]
            result["function_call"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
            result["tool_calls"] = message.tool_calls
            result["content"] = message.content  # May be None for function calls
        elif hasattr(message, 'function_call') and message.function_call:
            # Handle legacy function_call format
            result["function_call"] = {
                "name": message.function_call.name,
                "arguments": message.function_call.arguments
            }
            result["content"] = message.content
        else:
            result["content"] = message.content
            # Ensure function_call is None when there are no function calls
            result["function_call"] = None
            
        return result
    
    def is_available(self) -> bool:
        try:
            # Test API key by making a simple request
            self.client.models.list()
            return True
        except Exception:
            return False


class GeminiClient(ModelClient):
    """Google Gemini API client using OpenAI client with custom base_url (OpenAI compatible)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from openai import OpenAI
            base_url = config.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
            if not base_url.endswith("/"):
                base_url += "/"
            
            self.client = OpenAI(
                api_key=config["api_key"],
                base_url=base_url
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Gemini API via OpenAI client"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        # Handle function calling properly
        message = response.choices[0].message
        result = {
            "role": "assistant",
            "raw_response": response,
            "backend": "gemini"
        }
        # Check for function calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Convert tool_calls to function_call format for compatibility
            tool_call = message.tool_calls[0]
            result["function_call"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
            result["tool_calls"] = message.tool_calls
            result["content"] = message.content  # May be None for function calls
        else:
            result["content"] = message.content
            # Ensure function_call is None when there are no function calls
            result["function_call"] = None
            
        return result
    
    def is_available(self) -> bool:
        try:
            # Test API key by making a simple request
            self.client.models.list()
            return True
        except Exception:
            return False


def create_model_client(config: Dict[str, Any]) -> ModelClient:
    """Factory function to create appropriate model client"""
    backend = config.get("backend", "openai")
    
    if backend == "openai":
        return OpenAIClient(config)
    elif backend == "together":
        return TogetherAIClient(config)
    elif backend == "google" or backend == "gemini":
        return GeminiClient(config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")