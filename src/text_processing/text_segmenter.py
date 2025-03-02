"""
Text Segmentation Module

This module provides functionality for segmenting text into a hierarchical structure
using LLM-based segmentation or rule-based approaches.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import uuid
import logging
from pathlib import Path

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.language import get_prompt_for_language
from src.config.app_config import AppConfig
from src.llm import get_provider, LLMProvider
from src.text_processing.segment import Segment, SegmentationResult
from src.text_processing.text_loader import load_text

# Configure logger
logger = get_logger(__name__)

# Segmentation prompts for different languages
# These are templates for Claude to segment text
SEGMENTATION_PROMPTS = {
    "en": """
Divide the following text into hierarchically organized segments.

Text: {text}

Create a structure with the following levels:
1. Main sections (if any)
2. Subsections (if any)
3. Logical blocks within subsections
4. Individual paragraphs or groups of related sentences

For each segment, specify:
- Unique ID (e.g., "1.2.3" for the 3rd block of the 2nd subsection of the 1st section)
- Original text
- Position in document (start-end in characters)
- Segment type (e.g., "section", "subsection", "block", "paragraph")

Present the result in JSON format with a nested structure reflecting the hierarchy, following this schema:
{{
  "segments": [
    {{
      "id": "string",
      "text": "string",
      "position": {{"start": number, "end": number}},
      "segment_type": "string",
      "children": [
        // Child segments with the same structure
      ]
    }}
  ]
}}
""",
    "ru": """
Разделите следующий текст на иерархически организованные сегменты.

Текст: {text}

Создайте структуру с следующими уровнями:
1. Основные разделы (если есть)
2. Подразделы (если есть)
3. Логические блоки в пределах подразделов
4. Отдельные абзацы или группы связанных предложений

Для каждого сегмента укажите:
- Уникальный ID (например, "1.2.3" для 3-го блока 2-го подраздела 1-го раздела)
- Исходный текст
- Позицию в документе (начало-конец в символах)
- Тип сегмента (например, "section", "subsection", "block", "paragraph")

Представьте результат в JSON-формате с вложенной структурой, отражающей иерархию, по следующей схеме:
{{
  "segments": [
    {{
      "id": "string",
      "text": "string",
      "position": {{"start": number, "end": number}},
      "segment_type": "string",
      "children": [
        // Дочерние сегменты с такой же структурой
      ]
    }}
  ]
}}
"""
}

# Simplified schema for JSON response validation
SEGMENTATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "position": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"}
                        },
                        "required": ["start", "end"]
                    },
                    "segment_type": {"type": "string"},
                    "children": {"type": "array"}
                },
                "required": ["id", "text", "position", "segment_type"]
            }
        }
    },
    "required": ["segments"]
}


def parse_segment_json(json_data: Dict[str, Any]) -> Result[List[Dict[str, Any]]]:
    """
    Parse segmentation JSON data into a list of segment dictionaries.
    
    Args:
        json_data: Raw JSON data from LLM
        
    Returns:
        Result[List[Dict[str, Any]]]: List of segment dictionaries or error
    """
    try:
        segments = json_data.get("segments", [])
        if not segments:
            return Result.fail("No segments found in JSON response")
        
        return Result.ok(segments)
    except Exception as e:
        logger.error(f"Failed to parse segment JSON: {str(e)}")
        return Result.fail(f"Failed to parse segment JSON: {str(e)}")


def build_segment_hierarchy(segments_data: List[Dict[str, Any]]) -> Result[Segment]:
    """
    Build a segment hierarchy from flat segment data.
    
    Args:
        segments_data: List of segment dictionaries
        
    Returns:
        Result[Segment]: Root segment or error
    """
    try:
        if not segments_data:
            return Result.fail("No segments provided")
        
        # Create root segment
        root_data = segments_data[0].copy()
        children_data = root_data.pop("children", [])
        
        root_segment = Segment(
            text=root_data["text"],
            segment_type=root_data["segment_type"],
            id=root_data.get("id", str(uuid.uuid4())),
            position=root_data["position"],
            level=0
        )
        
        # Process function to recursively build the hierarchy
        def process_children(parent_segment: Segment, children: List[Dict[str, Any]]) -> None:
            for child_data in children:
                child_children = child_data.pop("children", [])
                
                child_segment = Segment(
                    text=child_data["text"],
                    segment_type=child_data["segment_type"],
                    id=child_data.get("id", str(uuid.uuid4())),
                    position=child_data["position"],
                )
                
                parent_segment.add_child(child_segment)
                
                # Process child's children
                process_children(child_segment, child_children)
        
        # Process root's children
        process_children(root_segment, children_data)
        
        return Result.ok(root_segment)
    except Exception as e:
        logger.error(f"Failed to build segment hierarchy: {str(e)}")
        return Result.fail(f"Failed to build segment hierarchy: {str(e)}")


class TextSegmenter:
    """
    Class for segmenting text into hierarchical structures.
    
    Provides methods for LLM-based segmentation and rule-based fallbacks.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the TextSegmenter.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or AppConfig()
        self.llm_provider = None
    
    def segment_text(self, 
                     text_data: Union[str, Dict[str, Any]], 
                     language: Optional[str] = None,
                     use_llm: bool = True) -> Result[SegmentationResult]:
        """
        Segment text into a hierarchical structure.
        
        Args:
            text_data: Text to segment or result from TextLoader
            language: Language code (auto-detected if None)
            use_llm: Whether to use LLM for segmentation (vs. rule-based)
            
        Returns:
            Result[SegmentationResult]: Segmentation result or error
        """
        # Handle text data input
        if isinstance(text_data, str):
            # Load text if string is provided
            text_result = load_text(text_data)
            if not text_result.success:
                return Result.fail(f"Failed to load text: {text_result.error}")
            text_info = text_result.value
        else:
            # Assume it's already a processed text dictionary
            text_info = text_data
        
        # Get text and language
        text = text_info["text"]
        if not language:
            language = text_info.get("language", "en")
        
        # Choose segmentation method
        if use_llm and len(text) > 0:
            return self._segment_with_llm(text, language)
        else:
            return self._segment_with_rules(text, language)
    
    def _segment_with_llm(self, text: str, language: str) -> Result[SegmentationResult]:
        """
        Segment text using an LLM.
        
        Args:
            text: Text to segment
            language: Language code
            
        Returns:
            Result[SegmentationResult]: Segmentation result or error
        """
        # Initialize LLM provider if needed
        if not self.llm_provider:
            # Get LLM provider from LLMConfig
            from src.config.llm_config import LLMConfig
            llm_config = LLMConfig()
            provider_name = llm_config.provider
            model_name = llm_config.model
            
            provider_result = get_provider(provider_name, model_name)
            if not provider_result.success:
                return Result.fail(f"Failed to initialize LLM provider: {provider_result.error}")
            self.llm_provider = provider_result.value
        
        # Get appropriate prompt for language
        prompt_template = SEGMENTATION_PROMPTS.get(language, SEGMENTATION_PROMPTS["en"])
        
        # For very long texts, we would need chunking, but for simplicity, we'll assume text fits in context
        prompt = prompt_template.format(text=text)
        
        # Call LLM to get segmentation
        try:
            logger.info(f"Calling LLM for text segmentation (language: {language}, text length: {len(text)})")
            
            # Generate structured JSON response
            response = self.llm_provider.generate_json(
                prompt=prompt,
                json_schema=SEGMENTATION_RESPONSE_SCHEMA,
                temperature=0.2
            )
            
            if not response.success:
                logger.error(f"LLM segmentation failed: {response.error}")
                # Fall back to rule-based segmentation
                logger.info("Falling back to rule-based segmentation")
                return self._segment_with_rules(text, language)
            
            json_data = response.value
            
            # Parse segment data
            segments_result = parse_segment_json(json_data)
            if not segments_result.success:
                return Result.fail(segments_result.error)
            
            segments_data = segments_result.value
            
            # Build segment hierarchy
            root_result = build_segment_hierarchy(segments_data)
            if not root_result.success:
                return Result.fail(root_result.error)
            
            root_segment = root_result.value
            
            # Create segmentation result
            segmentation_result = SegmentationResult(
                root=root_segment,
                metadata={
                    "method": "llm",
                    "language": language,
                    "text_length": len(text),
                    "provider": self.llm_provider.provider_name,
                    "model": self.llm_provider.model_name
                }
            )
            
            logger.info(f"Segmentation complete: {len(segmentation_result.segments)} segments created")
            return Result.ok(segmentation_result)
            
        except Exception as e:
            logger.error(f"Error during LLM segmentation: {str(e)}")
            # Fall back to rule-based segmentation
            logger.info("Falling back to rule-based segmentation due to error")
            return self._segment_with_rules(text, language)
    
    def _segment_with_rules(self, text: str, language: str) -> Result[SegmentationResult]:
        """
        Segment text using rule-based approaches.
        
        This is a fallback method when LLM segmentation is not available or fails.
        It uses simple heuristics to split text into paragraphs and blocks.
        
        Args:
            text: Text to segment
            language: Language code
            
        Returns:
            Result[SegmentationResult]: Segmentation result or error
        """
        try:
            logger.info(f"Using rule-based segmentation (language: {language}, text length: {len(text)})")
            
            # Create root segment for the entire text
            root_segment = Segment(
                text=text,
                segment_type="document",
                position={"start": 0, "end": len(text)}
            )
            
            # Split into paragraphs using blank lines
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Current position tracker
            current_pos = 0
            
            for idx, para_text in enumerate(paragraphs):
                if not para_text.strip():
                    # Skip empty paragraphs
                    current_pos += len(para_text) + 2  # +2 for the newlines
                    continue
                
                # Find the start position in original text
                # This is simplified and might not be accurate if there are multiple identical paragraphs
                start_pos = text.find(para_text, current_pos)
                if start_pos == -1:
                    # Fallback if exact match not found
                    start_pos = current_pos
                
                end_pos = start_pos + len(para_text)
                current_pos = end_pos + 2  # +2 for the newlines
                
                # Create paragraph segment
                para_segment = Segment(
                    text=para_text,
                    segment_type="paragraph",
                    position={"start": start_pos, "end": end_pos}
                )
                
                # Add paragraph as child to root
                root_segment.add_child(para_segment)
                
                # For longer paragraphs, split into sentences
                if len(para_text) > 200:
                    # Simple sentence splitting (not perfect but a good approximation)
                    # For better sentence splitting, use a more sophisticated NLP library
                    sentences = re.split(r'(?<=[.!?])\s+', para_text)
                    
                    # Current position tracker within paragraph
                    para_pos = 0
                    
                    for sent_idx, sent_text in enumerate(sentences):
                        if not sent_text.strip():
                            # Skip empty sentences
                            para_pos += len(sent_text) + 1  # +1 for the space
                            continue
                        
                        # Find the start position in paragraph text
                        sent_start = para_text.find(sent_text, para_pos)
                        if sent_start == -1:
                            # Fallback if exact match not found
                            sent_start = para_pos
                        
                        sent_end = sent_start + len(sent_text)
                        para_pos = sent_end + 1  # +1 for the space
                        
                        # Create sentence segment
                        sent_segment = Segment(
                            text=sent_text,
                            segment_type="sentence",
                            position={"start": start_pos + sent_start, "end": start_pos + sent_end}
                        )
                        
                        # Add sentence as child to paragraph
                        para_segment.add_child(sent_segment)
            
            # Create segmentation result
            segmentation_result = SegmentationResult(
                root=root_segment,
                metadata={
                    "method": "rule-based",
                    "language": language,
                    "text_length": len(text)
                }
            )
            
            logger.info(f"Rule-based segmentation complete: {len(segmentation_result.segments)} segments created")
            return Result.ok(segmentation_result)
            
        except Exception as e:
            logger.error(f"Error during rule-based segmentation: {str(e)}")
            return Result.fail(f"Failed to segment text with rules: {str(e)}")


def segment_text(text_data: Union[str, Dict[str, Any]], 
                language: Optional[str] = None,
                use_llm: bool = True,
                config: Optional[AppConfig] = None) -> Result[SegmentationResult]:
    """
    Convenience function to segment text.
    
    Args:
        text_data: Text to segment or result from TextLoader
        language: Language code (auto-detected if None)
        use_llm: Whether to use LLM for segmentation
        config: Optional application configuration
        
    Returns:
        Result[SegmentationResult]: Segmentation result or error
    """
    segmenter = TextSegmenter(config)
    return segmenter.segment_text(text_data, language, use_llm) 