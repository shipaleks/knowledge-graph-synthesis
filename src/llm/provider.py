"""
LLM Provider Interface

This module defines the abstract base class for all LLM providers.
Each provider implementation must inherit from LLMProvider and implement
all required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import json

from src.utils.result import Result


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM providers must implement.
    It provides methods for text generation, text embeddings, and other LLM-related
    functionality.
    """
    
    @abstractmethod
    def initialize(self) -> Result[bool]:
        """
        Initialize the LLM provider with necessary setup.
        
        Returns:
            Result[bool]: Success or failure with error details
        """
        pass
    
    @abstractmethod
    def generate_text(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     temperature: float = 0.7,
                     stop_sequences: Optional[List[str]] = None,
                     top_p: Optional[float] = None,
                     top_k: Optional[int] = None) -> Result[str]:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The main user prompt
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0-1.0)
            stop_sequences: List of sequences that will stop generation
            top_p: Controls diversity via nucleus sampling
            top_k: Controls diversity via top-k sampling
            
        Returns:
            Result[str]: The generated text or error details
        """
        pass
    
    @abstractmethod
    def generate_json(self, 
                     prompt: str,
                     json_schema: Dict[str, Any],
                     system_prompt: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     temperature: float = 0.7) -> Result[Dict[str, Any]]:
        """
        Generate structured JSON output based on the given prompt and schema.
        
        Args:
            prompt: The main user prompt
            json_schema: JSON schema defining the expected output structure
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0-1.0)
            
        Returns:
            Result[Dict[str, Any]]: The generated JSON data or error details
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> Result[List[List[float]]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Result[List[List[float]]]: List of embedding vectors or error details
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> Result[int]:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Result[int]: Number of tokens or error details
        """
        pass
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """
        Get the maximum context length supported by this provider.
        
        Returns:
            int: Maximum context length in tokens
        """
        pass
    
    @property
    @abstractmethod
    def max_output_length(self) -> int:
        """
        Get the maximum output length supported by this provider.
        
        Returns:
            int: Maximum output length in tokens
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            str: Model name
        """
        pass


def get_provider(provider_name: str, model_name: Optional[str] = None) -> Result[LLMProvider]:
    """
    Factory function to get an instance of the specified LLM provider.
    
    Args:
        provider_name: Name of the provider (e.g., "claude", "openai")
        model_name: Optional specific model to use
        
    Returns:
        Result[LLMProvider]: Provider instance or error details
    """
    from src.utils.result import Result
    from src.config.llm_config import LLMConfig
    
    # Get default model if not specified
    if not model_name:
        config = LLMConfig()
        default_provider_config = config.get_provider_config(provider_name)
        if not default_provider_config.success:
            return Result.fail(f"Failed to get default provider config: {default_provider_config.error}")
        model_name = default_provider_config.value.get("default_model")
    
    # Initialize the appropriate provider
    if provider_name.lower() == "claude":
        from src.llm.claude_provider import ClaudeProvider
        return Result.ok(ClaudeProvider(model_name))
    
    # Add more providers as they are implemented
    # elif provider_name.lower() == "openai":
    #     from src.llm.openai_provider import OpenAIProvider
    #     return Result.ok(OpenAIProvider(model_name))
    
    return Result.fail(f"Unknown provider: {provider_name}") 