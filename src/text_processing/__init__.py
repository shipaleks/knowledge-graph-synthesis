"""
Text Processing Module

This module provides functionality for processing text, including
loading, segmentation, and summarization.
"""

from src.text_processing.text_loader import TextLoader, load_text
from src.text_processing.segment import Segment, SegmentationResult
from src.text_processing.text_segmenter import TextSegmenter, segment_text

__all__ = [
    # Text loading
    'TextLoader',
    'load_text',
    
    # Text segmentation
    'Segment',
    'SegmentationResult',
    'TextSegmenter',
    'segment_text',
]
