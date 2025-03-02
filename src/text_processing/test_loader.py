"""
Test script for TextLoader functionality.

This script provides simple tests for text loading, language detection,
and text normalization.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.text_processing import TextLoader, load_text
from src.utils import configure_logging


def test_load_from_text():
    """Test loading text directly from a string."""
    
    # Russian text
    russian_text = """
    Графовое представление знаний позволяет эффективно моделировать 
    сложные взаимосвязи между концепциями в тексте. Это особенно
    полезно при анализе научных текстов или художественной литературы.
    """
    
    # English text
    english_text = """
    Knowledge graph representation allows for effective modeling of
    complex relationships between concepts in text. This is especially
    useful when analyzing scientific texts or fiction.
    """
    
    # Create loader
    loader = TextLoader()
    
    # Test Russian text
    print("\nTesting Russian text loading and processing...")
    result = loader.load_from_text(russian_text)
    
    if not result.success:
        print(f"Error loading Russian text: {result.error}")
    else:
        print("Russian text processing results:")
        print(f"Language: {result.value['language']}")
        print(f"Supported: {result.value['is_supported_language']}")
        print(f"Length: {result.value['length']} characters")
        print(f"Normalized text: {result.value['text'][:100]}...")
    
    # Test English text
    print("\nTesting English text loading and processing...")
    result = loader.load_from_text(english_text)
    
    if not result.success:
        print(f"Error loading English text: {result.error}")
    else:
        print("English text processing results:")
        print(f"Language: {result.value['language']}")
        print(f"Supported: {result.value['is_supported_language']}")
        print(f"Length: {result.value['length']} characters")
        print(f"Normalized text: {result.value['text'][:100]}...")


def test_load_from_file():
    """Test loading text from files."""
    
    # Create temporary files with test content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as ru_file:
        ru_file.write("""
        Графовое представление знаний позволяет эффективно моделировать 
        сложные взаимосвязи между концепциями в тексте. Это особенно
        полезно при анализе научных текстов или художественной литературы.
        """)
        ru_file_path = ru_file.name
        
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as en_file:
        en_file.write("""
        Knowledge graph representation allows for effective modeling of
        complex relationships between concepts in text. This is especially
        useful when analyzing scientific texts or fiction.
        """)
        en_file_path = en_file.name
    
    try:
        # Test file loading
        print("\nTesting file loading...")
        
        # Test Russian file
        print(f"\nLoading Russian text from file: {ru_file_path}")
        result = load_text(ru_file_path)
        
        if not result.success:
            print(f"Error loading Russian file: {result.error}")
        else:
            print("Russian file loading results:")
            print(f"Language: {result.value['language']}")
            print(f"Supported: {result.value['is_supported_language']}")
            print(f"Length: {result.value['length']} characters")
            print(f"Source: {result.value['metadata']['source']}")
            print(f"Encoding: {result.value['metadata']['encoding']}")
            print(f"Normalized text: {result.value['text'][:100]}...")
        
        # Test English file
        print(f"\nLoading English text from file: {en_file_path}")
        result = load_text(en_file_path)
        
        if not result.success:
            print(f"Error loading English file: {result.error}")
        else:
            print("English file loading results:")
            print(f"Language: {result.value['language']}")
            print(f"Supported: {result.value['is_supported_language']}")
            print(f"Length: {result.value['length']} characters")
            print(f"Source: {result.value['metadata']['source']}")
            print(f"Encoding: {result.value['metadata']['encoding']}")
            print(f"Normalized text: {result.value['text'][:100]}...")
            
        # Test non-existent file
        print("\nTesting loading from non-existent file...")
        result = load_text("non_existent_file.txt")
        if not result.success:
            print(f"Expected error received: {result.error}")
        else:
            print("Unexpectedly loaded non-existent file!")
            
    finally:
        # Clean up temporary files
        try:
            os.unlink(ru_file_path)
            os.unlink(en_file_path)
        except Exception as e:
            print(f"Error removing temporary files: {str(e)}")


def test_text_normalization():
    """Test text normalization functionality."""
    
    # Test text with various normalization needs
    test_text = """
    This   text  has    extra   spaces
    and multiple
    line    breaks.
    
    It also has some   weird\u200Bzero-width\u200Bspaces.
    """
    
    print("\nTesting text normalization...")
    print("Original text:")
    print("-" * 60)
    print(test_text)
    print("-" * 60)
    
    # Create loader and normalize text
    loader = TextLoader()
    normalized = loader._normalize_text(test_text)
    
    print("\nNormalized text:")
    print("-" * 60)
    print(normalized)
    print("-" * 60)


def main():
    """Main entry point for testing TextLoader."""
    
    parser = argparse.ArgumentParser(description="Test TextLoader functionality")
    parser.add_argument("--test", default="all", choices=["text", "file", "normalize", "all"], 
                        help="Test type: text, file, normalize, or all (default: all)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    configure_logging(log_level="INFO")
    
    # Run tests
    if args.test in ["text", "all"]:
        test_load_from_text()
        
    if args.test in ["file", "all"]:
        test_load_from_file()
        
    if args.test in ["normalize", "all"]:
        test_text_normalization()


if __name__ == "__main__":
    main() 