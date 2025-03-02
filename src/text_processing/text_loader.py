"""
Text Loader Module

This module provides functionality for loading text from various sources,
detecting language, and performing basic normalization.
"""

import os
import re
import chardet
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from src.utils.result import Result
from src.utils.language import detect_language, is_supported_language
from src.utils.logger import get_logger
from src.config.app_config import AppConfig

# Configure logger
logger = get_logger(__name__)


class TextLoader:
    """
    Class for loading text from various sources, detecting language,
    and performing basic normalization.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the TextLoader.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or AppConfig()
        self.default_encoding = "utf-8"
        
        # Regular expressions for text cleaning
        self.cleanup_patterns = [
            # Remove extra whitespace
            (r'\s+', ' '),
            # Normalize line breaks
            (r'\r\n', '\n'),
            # Remove zero-width spaces and similar characters
            (r'[\u200B-\u200D\uFEFF]', ''),
        ]
    
    def load_from_file(self, file_path: Union[str, Path], encoding: Optional[str] = None) -> Result[Dict[str, Any]]:
        """
        Load text from a file.
        
        Args:
            file_path: Path to the file
            encoding: Optional encoding to use
            
        Returns:
            Result[Dict[str, Any]]: Dictionary with loaded text and metadata
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return Result.fail(f"File not found: {file_path}")
                
            if not file_path.is_file():
                return Result.fail(f"Path is not a file: {file_path}")
            
            # Determine encoding if not provided
            if not encoding:
                encoding_result = self._detect_encoding(file_path)
                if not encoding_result.success:
                    return Result.fail(f"Failed to detect encoding: {encoding_result.error}")
                encoding = encoding_result.value
            
            # Read file content
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Process the loaded text
            return self.process_text(
                content, 
                metadata={
                    "source": str(file_path),
                    "source_type": "file",
                    "encoding": encoding,
                    "file_size": file_path.stat().st_size,
                    "file_modified": file_path.stat().st_mtime
                }
            )
            
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error when reading {file_path}: {str(e)}")
            return Result.fail(f"Unicode decode error: {str(e)}. Try specifying the correct encoding.")
            
        except IOError as e:
            logger.error(f"IO error when reading {file_path}: {str(e)}")
            return Result.fail(f"Failed to read file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error when loading {file_path}: {str(e)}")
            return Result.fail(f"Failed to load file: {str(e)}")
    
    def load_from_text(self, text: str) -> Result[Dict[str, Any]]:
        """
        Load text from a string.
        
        Args:
            text: Text string to load
            
        Returns:
            Result[Dict[str, Any]]: Dictionary with loaded text and metadata
        """
        if not text:
            return Result.fail("Empty text provided")
            
        return self.process_text(
            text,
            metadata={
                "source": "direct_input",
                "source_type": "text",
                "text_length": len(text)
            }
        )
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any]]:
        """
        Process loaded text: normalize, detect language, and prepare metadata.
        
        Args:
            text: Text to process
            metadata: Optional metadata about the text
            
        Returns:
            Result[Dict[str, Any]]: Dictionary with processed text and metadata
        """
        try:
            # Initialize metadata if not provided
            metadata = metadata or {}
            
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Detect language
            lang_result = detect_language(normalized_text)
            if not lang_result.success:
                return Result.fail(f"Failed to detect language: {lang_result.error}")
            
            language = lang_result.value
            
            # Check if language is supported
            is_supported = is_supported_language(language)
            
            # Create result dictionary
            result = {
                "text": normalized_text,
                "language": language,
                "is_supported_language": is_supported,
                "length": len(normalized_text),
                "metadata": metadata
            }
            
            logger.info(f"Processed text: {len(normalized_text)} chars, language: {language}, supported: {is_supported}")
            return Result.ok(result)
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return Result.fail(f"Failed to process text: {str(e)}")
    
    def _detect_encoding(self, file_path: Path) -> Result[str]:
        """
        Detect encoding of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Result[str]: Detected encoding
        """
        try:
            # Read a chunk of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(min(file_path.stat().st_size, 10000))
            
            if not raw_data:
                return Result.ok(self.default_encoding)
            
            # Use chardet to detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or self.default_encoding
            confidence = result.get('confidence', 0)
            
            logger.debug(f"Detected encoding: {encoding} with confidence: {confidence}")
            
            # If confidence is low, fall back to default encoding
            if confidence < 0.7:
                logger.warning(f"Low confidence in encoding detection ({confidence}), falling back to {self.default_encoding}")
                return Result.ok(self.default_encoding)
            
            return Result.ok(encoding)
            
        except Exception as e:
            logger.error(f"Error detecting encoding: {str(e)}")
            return Result.fail(f"Failed to detect encoding: {str(e)}")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing redundant whitespace, normalizing line breaks,
        and applying other basic cleanup.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Apply cleanup patterns
        normalized = text
        for pattern, replacement in self.cleanup_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Trim whitespace
        normalized = normalized.strip()
        
        return normalized


def load_text(source: Union[str, Path], 
             encoding: Optional[str] = None, 
             config: Optional[AppConfig] = None) -> Result[Dict[str, Any]]:
    """
    Convenience function to load text from a file or string.
    
    Args:
        source: File path or text string
        encoding: Optional encoding for file reading
        config: Optional application configuration
        
    Returns:
        Result[Dict[str, Any]]: Dictionary with loaded text and metadata
    """
    loader = TextLoader(config)
    
    # If source is a string path to a file
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and path.is_file():
            return loader.load_from_file(path, encoding)
        elif str(path).endswith(('.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm')):
            # Source looks like a file path but file doesn't exist
            return Result.fail(f"File not found: {path}")
    
    # Otherwise, treat as direct text input
    return loader.load_from_text(str(source)) 