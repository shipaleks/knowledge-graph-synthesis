"""
Text Processing Pipeline Module

This module provides a complete text processing pipeline that integrates
text loading, segmentation, and summarization into a cohesive workflow.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.timer import Timer
from src.config.app_config import AppConfig
from src.text_processing.text_loader import load_text, detect_language
from src.text_processing.text_segmenter import segment_text, TextSegmenter
from src.text_processing.text_summarizer import summarize_text, TextSummarizer
from src.text_processing.segment import Segment, SegmentationResult

# Configure logger
logger = get_logger(__name__)


class TextProcessingPipeline:
    """
    A complete pipeline for processing text documents.
    
    Includes text loading, segmentation, and summarization with configuration options.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the text processing pipeline.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or AppConfig()
        self.segmenter = TextSegmenter(config)
        self.summarizer = TextSummarizer(config)
        self.output_dir = Path(os.getenv("APP_OUTPUT_DIR", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing options
        self.use_llm_segmentation = True
        self.use_cache = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.save_intermediates = True
        
        logger.debug("Text processing pipeline initialized")
    
    def process(self, 
               input_text: Union[str, Dict[str, Any], Path], 
               language: Optional[str] = None,
               output_prefix: Optional[str] = None) -> Result[Dict[str, Any]]:
        """
        Process text through the complete pipeline.
        
        Args:
            input_text: Text to process (string, dict, or file path)
            language: Language code (auto-detected if None)
            output_prefix: Prefix for output files
            
        Returns:
            Result[Dict[str, Any]]: Processing results or error
        """
        # Create timer to measure performance
        timer = Timer()
        
        # Start processing
        logger.info("Starting text processing pipeline")
        overall_timer = timer.start("overall")
        
        # Step 1: Load and preprocess text
        load_timer = timer.start("text_loading")
        text_result = load_text(input_text)
        timer.stop(load_timer)
        
        if not text_result.success:
            return Result.fail(f"Failed to load text: {text_result.error}")
        
        text_data = text_result.value
        text_content = text_data["text"]
        detected_language = text_data.get("language")
        
        # Use provided language or detected language
        language = language or detected_language or self.config.language
        
        logger.info(f"Text loaded: {len(text_content)} chars, language: {language}")
        
        # Step 2: Segment text
        segment_timer = timer.start("segmentation")
        segmentation_result = segment_text(
            text_content, 
            language=language, 
            use_llm=self.use_llm_segmentation,
            config=self.config
        )
        timer.stop(segment_timer)
        
        if not segmentation_result.success:
            return Result.fail(f"Failed to segment text: {segmentation_result.error}")
        
        segments = segmentation_result.value
        logger.info(f"Text segmented into {len(segments.segments)} segments")
        
        # Save segmentation results if enabled
        if self.save_intermediates:
            self._save_segments(segments, output_prefix, language)
        
        # Step 3: Summarize text segments
        summarize_timer = timer.start("summarization")
        summary_result = summarize_text(
            segments, 
            language=language,
            use_cache=self.use_cache,
            config=self.config
        )
        timer.stop(summarize_timer)
        
        if not summary_result.success:
            return Result.fail(f"Failed to summarize text: {summary_result.error}")
        
        summaries = summary_result.value
        logger.info(f"Generated {len(summaries)} summaries")
        
        # Save summarization results if enabled
        if self.save_intermediates:
            self._save_summaries(summaries, output_prefix, language)
        
        # Stop overall timer
        timer.stop(overall_timer)
        
        # Compile results
        timing_info = timer.get_all_timings()
        
        results = {
            "segments": segments.build_hierarchy(),
            "summaries": summaries,
            "metadata": {
                "language": language,
                "text_length": len(text_content),
                "segment_count": len(segments.segments),
                "summary_count": len(summaries),
                "timing": timing_info,
                "process_date": time.time()
            }
        }
        
        # Save complete results if enabled
        if self.save_intermediates:
            self._save_results(results, output_prefix, language)
        
        logger.info(f"Text processing complete in {timing_info['overall']:.2f} seconds")
        return Result.ok(results)
    
    def _save_segments(self, 
                      segments: SegmentationResult, 
                      prefix: Optional[str] = None, 
                      language: str = "en") -> None:
        """
        Save segmentation results to a file.
        
        Args:
            segments: Segmentation results
            prefix: Output file prefix
            language: Language code
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            base_name = f"{prefix}_" if prefix else ""
            filename = f"{base_name}segments_{language}.json"
            filepath = self.output_dir / filename
            
            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(segments.to_json(pretty=True))
            
            logger.debug(f"Saved segments to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save segments to file: {str(e)}")
    
    def _save_summaries(self, 
                       summaries: Dict[str, Dict[str, Any]], 
                       prefix: Optional[str] = None, 
                       language: str = "en") -> None:
        """
        Save summarization results to a file.
        
        Args:
            summaries: Summarization results
            prefix: Output file prefix
            language: Language code
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            base_name = f"{prefix}_" if prefix else ""
            filename = f"{base_name}summaries_{language}.json"
            filepath = self.output_dir / filename
            
            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved summaries to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save summaries to file: {str(e)}")
    
    def _save_results(self, 
                     results: Dict[str, Any], 
                     prefix: Optional[str] = None, 
                     language: str = "en") -> None:
        """
        Save complete processing results to a file.
        
        Args:
            results: Processing results
            prefix: Output file prefix
            language: Language code
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            base_name = f"{prefix}_" if prefix else ""
            filename = f"{base_name}processed_{language}.json"
            filepath = self.output_dir / filename
            
            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved complete results to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save complete results to file: {str(e)}")


def process_text(
    input_text: Union[str, Dict[str, Any], Path], 
    language: Optional[str] = None,
    use_llm_segmentation: bool = True,
    use_cache: bool = True,
    save_intermediates: bool = True,
    output_prefix: Optional[str] = None,
    config: Optional[AppConfig] = None
) -> Result[Dict[str, Any]]:
    """
    Process text through the complete pipeline (convenience function).
    
    Args:
        input_text: Text to process (string, dict, or file path)
        language: Language code (auto-detected if None)
        use_llm_segmentation: Whether to use LLM for segmentation
        use_cache: Whether to use caching
        save_intermediates: Whether to save intermediate results
        output_prefix: Prefix for output files
        config: Optional application configuration
        
    Returns:
        Result[Dict[str, Any]]: Processing results or error
    """
    pipeline = TextProcessingPipeline(config)
    pipeline.use_llm_segmentation = use_llm_segmentation
    pipeline.use_cache = use_cache
    pipeline.save_intermediates = save_intermediates
    
    return pipeline.process(input_text, language, output_prefix) 