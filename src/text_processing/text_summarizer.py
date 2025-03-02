"""
Text Summarization Module

This module provides functionality for summarizing text segments 
using LLM-based approaches.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.language import get_prompt_for_language
from src.config.app_config import AppConfig
from src.config.llm_config import LLMConfig
from src.llm import get_provider, LLMProvider
from src.text_processing.segment import Segment, SegmentationResult

# Configure logger
logger = get_logger(__name__)

# Cache directory for summaries
CACHE_DIR = Path("./cache/summaries")

# Summarization prompts for different languages
SUMMARIZATION_PROMPTS = {
    "en": """
Create a contextual summarization for the following text segment.

Text: {text}

Segment type: {segment_type}
Hierarchical position: {hierarchical_id}

Please provide:

1. Brief summary (1-2 sentences)
2. Key points (3-5 bullets)
3. Role in document structure (e.g., "introduction", "argument", "example", "conclusion")
4. Keywords or concepts (3-5 terms)

Your summary should be concise but capture the essential information and context.
""",
    "ru": """
Создайте контекстуальную суммаризацию для следующего сегмента текста.

Текст: {text}

Тип сегмента: {segment_type}
Иерархическая позиция: {hierarchical_id}

Пожалуйста, предоставьте:

1. Краткое резюме (1-2 предложения)
2. Ключевые тезисы (3-5 пунктов)
3. Роль в структуре документа (например, "введение", "аргумент", "пример", "вывод")
4. Ключевые слова или концепции (3-5 терминов)

Ваше резюме должно быть лаконичным, но при этом охватывать существенную информацию и контекст.
"""
}

# Simplified schema for JSON response validation
SUMMARIZATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "role": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["summary", "key_points", "role", "keywords"]
}


class SummaryCache:
    """Class for caching summaries to reduce LLM API calls."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the summary cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        logger.debug(f"Summary cache initialized at {self.cache_dir}")
    
    def get_cache_key(self, text: str, segment_type: str, language: str) -> str:
        """
        Generate a cache key for a segment.
        
        Args:
            text: Text content
            segment_type: Type of segment
            language: Language code
            
        Returns:
            str: Cache key
        """
        # Create a hash of the input parameters
        data = f"{text}|{segment_type}|{language}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_cache_path(self, key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path: Cache file path
        """
        return self.cache_dir / f"{key}.json"
    
    def get(self, text: str, segment_type: str, language: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached summary if available.
        
        Args:
            text: Text content
            segment_type: Type of segment
            language: Language code
            
        Returns:
            Optional[Dict[str, Any]]: Cached summary or None
        """
        key = self.get_cache_key(text, segment_type, language)
        
        # First check memory cache
        if key in self.memory_cache:
            logger.debug(f"Summary cache hit (memory) for key {key[:8]}...")
            return self.memory_cache[key]
        
        # Then check file cache
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Store in memory cache for faster access next time
                    self.memory_cache[key] = cached_data
                    logger.debug(f"Summary cache hit (file) for key {key[:8]}...")
                    return cached_data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read cache file {cache_path}: {str(e)}")
        
        return None
    
    def put(self, text: str, segment_type: str, language: str, summary: Dict[str, Any]) -> None:
        """
        Store a summary in the cache.
        
        Args:
            text: Text content
            segment_type: Type of segment
            language: Language code
            summary: Summary to cache
        """
        key = self.get_cache_key(text, segment_type, language)
        
        # Store in memory cache
        self.memory_cache[key] = summary
        
        # Store in file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.debug(f"Summary cached with key {key[:8]}...")
        except IOError as e:
            logger.warning(f"Failed to write cache file {cache_path}: {str(e)}")


class TextSummarizer:
    """
    Class for summarizing text segments.
    
    Provides methods for LLM-based summarization with caching.
    """
    
    def __init__(self, config: Optional[AppConfig] = None, cache: Optional[SummaryCache] = None):
        """
        Initialize the TextSummarizer.
        
        Args:
            config: Optional application configuration
            cache: Optional summary cache
        """
        self.config = config or AppConfig()
        self.cache = cache or SummaryCache()
        self.llm_provider = None
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    def summarize_segment(self, 
                         segment: Segment, 
                         language: Optional[str] = None,
                         use_cache: Optional[bool] = None) -> Result[Dict[str, Any]]:
        """
        Summarize a text segment.
        
        Args:
            segment: Segment to summarize
            language: Language code (auto-detected if None)
            use_cache: Whether to use caching (defaults to global cache setting)
            
        Returns:
            Result[Dict[str, Any]]: Summarization result or error
        """
        # Set default language if not provided
        language = language or self.config.language
        
        # Determine whether to use cache
        use_cache = self.cache_enabled if use_cache is None else use_cache
        
        # Check if summary is already cached
        if use_cache:
            cached_summary = self.cache.get(segment.text, segment.segment_type, language)
            if cached_summary:
                return Result.ok(cached_summary)
        
        # Generate summary using LLM
        summary_result = self._summarize_with_llm(segment, language)
        
        # Cache the result if successful and caching is enabled
        if summary_result.success and use_cache:
            self.cache.put(segment.text, segment.segment_type, language, summary_result.value)
        
        return summary_result
    
    def summarize_segments(self, 
                          segmentation_result: SegmentationResult,
                          language: Optional[str] = None,
                          use_cache: Optional[bool] = None) -> Result[Dict[str, Dict[str, Any]]]:
        """
        Summarize all segments in a segmentation result.
        
        Args:
            segmentation_result: Result of text segmentation
            language: Language code (auto-detected if None)
            use_cache: Whether to use caching (defaults to global cache setting)
            
        Returns:
            Result[Dict[str, Dict[str, Any]]]: Dictionary mapping segment IDs to summaries
        """
        # Set default language if not provided
        language = language or self.config.language
        
        # Dictionary to store summaries
        summaries = {}
        
        # Process each segment
        for segment in segmentation_result.segments:
            # Skip very short segments
            if len(segment.text.strip()) < 10:
                continue
                
            # Summarize segment
            result = self.summarize_segment(segment, language, use_cache)
            
            if result.success:
                summaries[segment.id] = result.value
            else:
                logger.warning(f"Failed to summarize segment {segment.id}: {result.error}")
        
        if not summaries:
            return Result.fail("Failed to generate any summaries")
        
        return Result.ok(summaries)
    
    def _summarize_with_llm(self, segment: Segment, language: str) -> Result[Dict[str, Any]]:
        """
        Summarize a segment using an LLM.
        
        Args:
            segment: Segment to summarize
            language: Language code
            
        Returns:
            Result[Dict[str, Any]]: Summarization result or error
        """
        # Initialize LLM provider if needed
        if not self.llm_provider:
            # Get LLM provider from LLMConfig
            llm_config = LLMConfig()
            provider_name = llm_config.provider
            model_name = llm_config.model
            
            provider_result = get_provider(provider_name, model_name)
            if not provider_result.success:
                return Result.fail(f"Failed to initialize LLM provider: {provider_result.error}")
            self.llm_provider = provider_result.value
        
        # Get appropriate prompt for language
        prompt_template = SUMMARIZATION_PROMPTS.get(language, SUMMARIZATION_PROMPTS["en"])
        
        # Format prompt with segment information
        prompt = prompt_template.format(
            text=segment.text,
            segment_type=segment.segment_type,
            hierarchical_id=segment.get_hierarchical_id()
        )
        
        # Call LLM to get summarization
        try:
            logger.info(f"Calling LLM for text summarization (language: {language}, segment type: {segment.segment_type})")
            
            # Generate structured JSON response
            response = self.llm_provider.generate_json(
                prompt=prompt,
                json_schema=SUMMARIZATION_RESPONSE_SCHEMA,
                temperature=0.3
            )
            
            if not response.success:
                logger.error(f"LLM summarization failed: {response.error}")
                return Result.fail(f"Failed to summarize text: {response.error}")
            
            summary_data = response.value
            
            # Add metadata to summary
            summary_data["segment_id"] = segment.id
            summary_data["segment_type"] = segment.segment_type
            summary_data["language"] = language
            summary_data["timestamp"] = time.time()
            
            logger.info(f"Summarization complete for segment {segment.id}")
            return Result.ok(summary_data)
            
        except Exception as e:
            logger.error(f"Error during LLM summarization: {str(e)}")
            return Result.fail(f"Failed to summarize text: {str(e)}")


def summarize_text(segmentation_result: SegmentationResult,
                  language: Optional[str] = None,
                  use_cache: bool = True,
                  config: Optional[AppConfig] = None) -> Result[Dict[str, Dict[str, Any]]]:
    """
    Convenience function to summarize text segments.
    
    Args:
        segmentation_result: Result of text segmentation
        language: Language code (auto-detected if None)
        use_cache: Whether to use caching
        config: Optional application configuration
        
    Returns:
        Result[Dict[str, Dict[str, Any]]]: Summarization results or error
    """
    summarizer = TextSummarizer(config)
    return summarizer.summarize_segments(segmentation_result, language, use_cache) 