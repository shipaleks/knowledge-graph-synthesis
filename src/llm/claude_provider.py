"""
Claude AI Provider Implementation

This module implements the LLMProvider interface for Anthropic's Claude models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.result import Result
from src.llm.provider import LLMProvider
from src.config.llm_config import LLMConfig
from src.utils.logger import get_logger

# Configure logger
logger = get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """
    Implementation of LLMProvider interface for Anthropic's Claude models.
    
    This provider supports text generation, JSON structured output, and 
    embeddings using Claude API.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Claude provider.
        
        Args:
            model_name: Name of the Claude model to use. Defaults to the latest Claude model.
        """
        # Default to the claude-3-7-sonnet-latest if not specified
        self._model_name = model_name or "claude-3-7-sonnet-latest"
        self._client = None
        self._initialized = False
        
        # Model specific parameters
        self._model_params = {
            "claude-3-7-sonnet-latest": {
                "max_context_length": 200000,  # Approximate token limit
                "max_output_length": 4096,     # Default max output
            }
        }
        
        # If model is not in our parameters list, set default limits
        if self._model_name not in self._model_params:
            self._model_params[self._model_name] = {
                "max_context_length": 100000,  # Conservative default
                "max_output_length": 4096,     # Default max output
            }
    
    def initialize(self) -> Result[bool]:
        """
        Initialize the Claude provider client.
        
        Returns:
            Result[bool]: Success or failure with error details
        """
        try:
            # Get API key from environment or config
            config = LLMConfig()
            api_key_result = config.get_api_key("claude") 
            
            if api_key_result.is_error:
                return Result.failure(f"Failed to get Claude API key: {api_key_result.error}")
            
            api_key = api_key_result.value
            
            # Initialize Anthropic client
            self._client = Anthropic(api_key=api_key)
            self._initialized = True
            
            logger.info(f"Claude provider initialized with model: {self._model_name}")
            return Result.success(True)
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude provider: {str(e)}")
            return Result.failure(f"Failed to initialize Claude provider: {str(e)}")
    
    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_text(self, 
                     prompt: str, 
                     system_prompt: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     temperature: float = 0.7,
                     stop_sequences: Optional[List[str]] = None,
                     top_p: Optional[float] = None,
                     top_k: Optional[int] = None) -> Result[str]:
        """
        Generate text using Claude model.
        
        Args:
            prompt: User prompt to send to the model
            system_prompt: Optional system prompt for additional context and instructions
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0-1.0)
            stop_sequences: List of sequences that will stop generation
            top_p: Controls diversity via nucleus sampling
            top_k: Controls diversity via top-k sampling
            
        Returns:
            Result[str]: Generated text or error details
        """
        if not self._initialized:
            init_result = self.initialize()
            if init_result.is_error:
                return Result.failure(f"Provider not initialized: {init_result.error}")
        
        try:
            # Set default max tokens if not provided
            max_tokens = max_tokens or self.max_output_length
            
            # Create message parameters
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Configure request parameters
            params = {
                "model": self._model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            
            # Add optional parameters if provided
            if system_prompt:
                params["system"] = system_prompt
                
            if stop_sequences:
                params["stop_sequences"] = stop_sequences
                
            if top_p is not None:
                params["top_p"] = top_p
                
            if top_k is not None:
                params["top_k"] = top_k
            
            # Make the API call
            logger.debug(f"Sending request to Claude API with model {self._model_name}")
            response = self._client.messages.create(**params)
            
            # Extract the response text
            generated_text = response.content[0].text
            
            logger.debug(f"Received response from Claude API: {len(generated_text)} characters")
            return Result.success(generated_text)
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            return Result.failure(f"Claude API error: {str(e)}")
            
        except anthropic.RateLimitError as e:
            logger.error(f"Claude API rate limit exceeded: {str(e)}")
            return Result.failure(f"Rate limit exceeded: {str(e)}")
            
        except anthropic.APIConnectionError as e:
            logger.error(f"Claude API connection error: {str(e)}")
            return Result.failure(f"Connection error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error during Claude text generation: {str(e)}")
            return Result.failure(f"Text generation failed: {str(e)}")
    
    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
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
        if not self._initialized:
            init_result = self.initialize()
            if init_result.is_error:
                return Result.failure(f"Provider not initialized: {init_result.error}")
        
        try:
            # Create a system prompt that includes the JSON schema if not provided
            enhanced_system_prompt = system_prompt or ""
            if enhanced_system_prompt:
                enhanced_system_prompt += "\n\n"
                
            enhanced_system_prompt += (
                "Please provide your response in JSON format according to the following schema:\n"
                f"{json.dumps(json_schema, indent=2)}\n\n"
                "Your response should be valid JSON that conforms to this schema. "
                "Do not include any explanations, markdown formatting, or text outside of the JSON structure."
            )
            
            # Set response format to JSON
            response_format = {"type": "json_object"}
            
            # Create message parameters
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Configure request parameters
            params = {
                "model": self._model_name,
                "max_tokens": max_tokens or self.max_output_length,
                "temperature": temperature,
                "messages": messages,
                "system": enhanced_system_prompt,
                "response_format": response_format
            }
            
            # Make the API call
            logger.debug(f"Sending JSON request to Claude API with model {self._model_name}")
            response = self._client.messages.create(**params)
            
            # Extract the response text
            json_text = response.content[0].text
            
            # Parse the JSON response
            try:
                json_data = json.loads(json_text)
                logger.debug(f"Successfully parsed JSON response: {len(json_text)} characters")
                return Result.success(json_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                return Result.failure(f"Failed to parse JSON response: {str(e)}, Response text: {json_text[:100]}...")
            
        except anthropic.BadRequestError as e:
            logger.error(f"Claude API bad request: {str(e)}")
            return Result.failure(f"Bad request: {str(e)}")
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            return Result.failure(f"Claude API error: {str(e)}")
            
        except anthropic.RateLimitError as e:
            logger.error(f"Claude API rate limit exceeded: {str(e)}")
            return Result.failure(f"Rate limit exceeded: {str(e)}")
            
        except anthropic.APIConnectionError as e:
            logger.error(f"Claude API connection error: {str(e)}")
            return Result.failure(f"Connection error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error during Claude JSON generation: {str(e)}")
            return Result.failure(f"JSON generation failed: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> Result[List[List[float]]]:
        """
        Generate embeddings for a list of texts.
        
        Note: As of now, Claude may not have a dedicated embeddings API.
        This is a placeholder implementation until Anthropic provides 
        an official embeddings endpoint.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Result[List[List[float]]]: List of embedding vectors or error details
        """
        # As of the current implementation, Claude doesn't have a dedicated embeddings API
        # This method will return an error until the API is available
        
        logger.warning("Claude embeddings API is not yet implemented")
        return Result.failure("Claude embeddings API is not yet implemented or available")
    
    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def count_tokens(self, text: str) -> Result[int]:
        """
        Count the number of tokens in the given text using Claude's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Result[int]: Number of tokens or error details
        """
        if not self._initialized:
            init_result = self.initialize()
            if init_result.is_error:
                return Result.failure(f"Provider not initialized: {init_result.error}")
        
        try:
            # Use Anthropic's token counting method
            token_count = self._client.count_tokens(text)
            return Result.success(token_count)
            
        except Exception as e:
            logger.error(f"Failed to count tokens: {str(e)}")
            return Result.failure(f"Token counting failed: {str(e)}")
    
    @property
    def max_context_length(self) -> int:
        """
        Get the maximum context length supported by this Claude model.
        
        Returns:
            int: Maximum context length in tokens
        """
        return self._model_params[self._model_name]["max_context_length"]
    
    @property
    def max_output_length(self) -> int:
        """
        Get the maximum output length supported by this Claude model.
        
        Returns:
            int: Maximum output length in tokens
        """
        return self._model_params[self._model_name]["max_output_length"]
    
    @property
    def provider_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "claude"
    
    @property
    def model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            str: Model name
        """
        return self._model_name 