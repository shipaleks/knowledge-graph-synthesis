"""
Text Processing Module

This module provides functionality for processing text, including
loading, segmentation, and summarization.
"""

from src.text_processing.text_loader import TextLoader, load_text

__all__ = [
    'TextLoader',
    'load_text',
]
