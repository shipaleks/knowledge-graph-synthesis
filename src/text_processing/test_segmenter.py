"""
Test script for TextSegmenter functionality.

This script provides simple tests for text segmentation,
both rule-based and LLM-based.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.text_processing import TextSegmenter, segment_text
from src.text_processing.text_loader import load_text
from src.utils import configure_logging


def print_segment_tree(segment, indent=0):
    """Print a segment and its children in a tree-like format."""
    indent_str = "  " * indent
    print(f"{indent_str}|-{segment.segment_type}: {segment.id} - {segment.text[:50]}...")
    for child in segment.children:
        print_segment_tree(child, indent + 1)


def test_rule_based_segmentation():
    """Test rule-based text segmentation."""
    
    # Sample text with clear paragraph structure
    sample_text = """
    This is the first paragraph with some content.
    It has multiple sentences for testing.
    
    This is the second paragraph. It also has multiple sentences.
    This will help test our segmentation.
    
    The third paragraph is here.
    It contains information that should be segmented properly.
    """
    
    print("\nTesting rule-based segmentation...")
    
    # First load the text properly
    load_result = load_text(sample_text)
    if not load_result.success:
        print(f"Error loading text: {load_result.error}")
        return
    
    # Create segmenter and process text
    segmenter = TextSegmenter()
    result = segmenter.segment_text(load_result.value, use_llm=False)
    
    if not result.success:
        print(f"Error in rule-based segmentation: {result.error}")
        return
    
    seg_result = result.value
    
    # Print results
    print(f"\nSegmentation completed successfully!")
    print(f"Total segments: {len(seg_result.segments)}")
    print(f"Method used: {seg_result.metadata.get('method', 'unknown')}")
    
    print("\nSegment Tree:")
    print_segment_tree(seg_result.root)
    
    # Print segment information
    print("\nFirst-level segments:")
    for segment in seg_result.root.children:
        print(f"Type: {segment.segment_type}")
        print(f"Text: {segment.text[:100]}...")
        print(f"Position: {segment.position}")
        print(f"Children: {len(segment.children)}")
        print("-" * 50)


def test_llm_segmentation():
    """Test LLM-based text segmentation."""
    
    # More complex text that would benefit from LLM segmentation
    sample_text = """
    # Introduction to Knowledge Graphs
    
    Knowledge graphs represent a powerful way to organize information. They consist of entities, relationships, and attributes.
    
    ## Key Components
    
    The main components of knowledge graphs include:
    
    1. Entities (nodes): Representing concepts, objects, or things.
    2. Relationships (edges): Connecting entities and showing how they relate.
    3. Attributes: Properties that describe entities.
    
    ## Applications
    
    Knowledge graphs are used in various domains:
    
    * Search engines for semantic understanding
    * Recommendation systems for personalization
    * Scientific research for data integration
    """
    
    print("\nTesting LLM-based segmentation...")
    
    # Load text first to get language detection
    load_result = load_text(sample_text)
    if not load_result.success:
        print(f"Error loading text: {load_result.error}")
        return
    
    # Create segmenter and process text
    segmenter = TextSegmenter()
    result = segmenter.segment_text(load_result.value, use_llm=True)
    
    if not result.success:
        print(f"Error in LLM-based segmentation: {result.error}")
        # Test fallback to rule-based
        print("\nTesting fallback to rule-based segmentation...")
        result = segmenter.segment_text(load_result.value, use_llm=False)
        if not result.success:
            print(f"Error in fallback segmentation: {result.error}")
            return
    
    seg_result = result.value
    
    # Print results
    print(f"\nSegmentation completed successfully!")
    print(f"Total segments: {len(seg_result.segments)}")
    print(f"Method used: {seg_result.metadata.get('method', 'unknown')}")
    if seg_result.metadata.get('method') == 'llm':
        print(f"LLM provider: {seg_result.metadata.get('provider')}")
        print(f"LLM model: {seg_result.metadata.get('model')}")
    
    print("\nSegment Tree:")
    print_segment_tree(seg_result.root)
    
    # Export segment hierarchy to JSON for inspection
    segments_json = json.dumps(seg_result.to_dict(), indent=2, ensure_ascii=False)
    print(f"\nSegment hierarchy exported to JSON ({len(segments_json)} chars)")
    
    # Write JSON to file for easier inspection (optional)
    try:
        with open("segments.json", "w", encoding="utf-8") as f:
            f.write(segments_json)
        print("JSON written to segments.json")
    except Exception as e:
        print(f"Could not write JSON to file: {str(e)}")


def main():
    """Main entry point for testing TextSegmenter."""
    
    parser = argparse.ArgumentParser(description="Test TextSegmenter functionality")
    parser.add_argument("--test", default="all", choices=["rule", "llm", "all"], 
                        help="Test type: rule-based, llm-based, or all (default: all)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    configure_logging(log_level="INFO")
    
    # Run tests
    if args.test in ["rule", "all"]:
        test_rule_based_segmentation()
        
    if args.test in ["llm", "all"]:
        test_llm_segmentation()


if __name__ == "__main__":
    main() 