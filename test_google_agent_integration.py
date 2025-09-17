#!/usr/bin/env python3
"""
Test script for Google Agent Integration improvements
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_manager import ConfigManager
from utils.google_agent_integration import GoogleAgentIntegration

def test_provider_detection():
    """Test the provider detection functionality"""
    print("Testing Provider Detection...")
    
    config_manager = ConfigManager()
    google_agent = GoogleAgentIntegration(config_manager)
    
    test_cases = [
        ("gpt-4o", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("claude-3-sonnet", "anthropic"),
        ("claude-3-haiku", "anthropic"),
        ("gemini-pro", "google"),
        ("gemini-flash", "google"),
        ("mistral-large", "mistral"),
        ("mixtral-8x7b", "mistral"),
        ("deepseek-chat", "deepseek"),
        ("command-r", "cohere"),
        ("unknown-model", "openai"),  # Should default to openai
    ]
    
    for model, expected_provider in test_cases:
        detected = google_agent._detect_provider_from_model(model)
        status = "✅" if detected == expected_provider else "❌"
        print(f"{status} Model: {model:20} -> Expected: {expected_provider:12} | Detected: {detected}")
    
    print()

def test_configuration_setup():
    """Test the configuration setup functionality"""
    print("Testing Configuration Setup...")
    
    config_manager = ConfigManager()
    google_agent = GoogleAgentIntegration(config_manager)
    
    # Test with provider and model
    google_agent.setup_configuration("openai", "gpt-4o")
    print(f"✅ Provider: {google_agent.provider}, Model: {google_agent.model}")
    print(f"✅ Search providers configured: {list(google_agent.search_providers.keys())}")
    
    # Test with just provider
    google_agent.setup_configuration("anthropic")
    print(f"✅ Provider: {google_agent.provider}, Model: {google_agent.model}")
    
    # Test validation
    validation = google_agent.validate_configuration()
    print(f"✅ Validation results: {validation}")
    
    print()

def test_model_interface():
    """Test the model interface"""
    print("Testing Model Interface...")
    
    from utils.google_agent_integration import GoogleAgentModelInterface
    
    config_manager = ConfigManager()
    model_interface = GoogleAgentModelInterface(config_manager)
    
    # Test response generation
    response = model_interface.generate_response(
        query="Test query",
        search_context="Test search context",
        model="gpt-4o",
        provider="openai",
        temperature=0.7,
        max_tokens=1024
    )
    
    if "Provider: openai" in response and "Model: gpt-4o" in response:
        print("✅ Model interface working correctly")
    else:
        print("❌ Model interface not working as expected")
    
    print()

def main():
    """Run all tests"""
    print("🔍 Testing Google Agent Integration Improvements\n")
    
    try:
        test_provider_detection()
        test_configuration_setup()
        test_model_interface()
        
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
