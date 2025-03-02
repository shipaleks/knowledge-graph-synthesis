"""
Test script for LLM providers.

This script provides simple tests for LLM providers to verify
their functionality.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from pprint import pprint

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import get_provider
from src.utils.logger import configure_logging


def test_text_generation(provider_name, model_name=None):
    """
    Test text generation with the specified provider.
    
    Args:
        provider_name: Name of the provider to test
        model_name: Optional specific model to use
    """
    # Get provider instance
    provider_result = get_provider(provider_name, model_name)
    
    if not provider_result.success:
        print(f"Error: {provider_result.error}")
        return
        
    provider = provider_result.value
    
    # Initialize provider
    init_result = provider.initialize()
    
    if not init_result.success:
        print(f"Error initializing provider: {init_result.error}")
        return
        
    print(f"Successfully initialized {provider.provider_name} provider with model {provider.model_name}")
    print(f"Model limits: context={provider.max_context_length}, output={provider.max_output_length}")
    
    # Test text generation
    prompt = "Опиши три основных преимущества графового представления знаний для анализа текстов."
    system_prompt = "Ты эксперт в области обработки естественного языка и представления знаний."
    
    print("\nGenerating text...")
    result = provider.generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=1024
    )
    
    if not result.success:
        print(f"Error generating text: {result.error}")
        return
        
    print("\nGenerated Text:")
    print("=" * 80)
    print(result.value)
    print("=" * 80)
    
    # Test token counting
    token_result = provider.count_tokens(prompt)
    
    if not token_result.success:
        print(f"Error counting tokens: {token_result.error}")
    else:
        print(f"\nPrompt token count: {token_result.value}")


def test_json_generation(provider_name, model_name=None):
    """
    Test JSON generation with the specified provider.
    
    Args:
        provider_name: Name of the provider to test
        model_name: Optional specific model to use
    """
    # Get provider instance
    provider_result = get_provider(provider_name, model_name)
    
    if not provider_result.success:
        print(f"Error: {provider_result.error}")
        return
        
    provider = provider_result.value
    
    # Initialize provider
    init_result = provider.initialize()
    
    if not init_result.success:
        print(f"Error initializing provider: {init_result.error}")
        return
    
    # Define JSON schema
    json_schema = {
        "type": "object",
        "properties": {
            "advantages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "use_cases": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["title", "description", "use_cases"]
                }
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["title", "description"]
                }
            }
        },
        "required": ["advantages", "limitations"]
    }
    
    # Test JSON generation
    prompt = "Приведи три основных преимущества и два ограничения графового представления знаний для анализа текстов."
    system_prompt = "Ты эксперт в области обработки естественного языка и представления знаний."
    
    print("\nGenerating JSON...")
    result = provider.generate_json(
        prompt=prompt,
        json_schema=json_schema,
        system_prompt=system_prompt,
        temperature=0.7
    )
    
    if not result.success:
        print(f"Error generating JSON: {result.error}")
        return
        
    print("\nGenerated JSON:")
    print("=" * 80)
    pprint(result.value)
    print("=" * 80)


def main():
    """Main entry point for testing LLM providers."""
    
    parser = argparse.ArgumentParser(description="Test LLM providers")
    parser.add_argument("--provider", default="claude", help="Provider name (default: claude)")
    parser.add_argument("--model", default=None, help="Model name (default: provider's default)")
    parser.add_argument("--test", default="text", choices=["text", "json", "all"], 
                        help="Test type: text, json, or all (default: text)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    configure_logging(log_level="INFO")
    
    print(f"Testing {args.provider} provider" + (f" with model {args.model}" if args.model else ""))
    
    # Run tests
    if args.test in ["text", "all"]:
        test_text_generation(args.provider, args.model)
        
    if args.test in ["json", "all"]:
        test_json_generation(args.provider, args.model)


if __name__ == "__main__":
    main() 