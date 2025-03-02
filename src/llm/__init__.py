"""
LLM (Large Language Model) Interface Module

This module provides interfaces and implementations for working
with various large language model providers.
"""

from src.llm.provider import LLMProvider, get_provider
from src.llm.claude_provider import ClaudeProvider

__all__ = [
    'LLMProvider', 
    'get_provider',
    'ClaudeProvider',
]
