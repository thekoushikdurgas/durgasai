"""
Test script for DurgasAI application.
Simple integration test to verify components work together.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        print("Testing imports...")
        
        # Test utils imports
        from utils.config import Config, ModelConfig, ModelProvider
        from utils.model_manager import ModelManager, ModelResponse
        from utils.ui_helpers import UIHelpers, SessionManager
        from utils.error_handler import ErrorHandler, DurgasAIError
        
        print("✅ All utils modules imported successfully")
        
        # Test pages imports
        from pages.aiagent import AIAgentPage
        print("✅ Pages modules imported successfully")
        
        # Test components imports
        from pages.component.chat_components import enhanced_chat_input
        print("✅ Component modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


def test_config():
    """Test configuration module."""
    try:
        print("\nTesting configuration...")
        
        # Test config access
        assert Config.APP_TITLE == "🤖 DurgasAI - Advanced AI Agent"
        assert len(Config.AVAILABLE_MODELS) > 0
        
        # Test model config
        model_names = Config.get_model_names()
        assert len(model_names) > 0
        
        first_model = Config.get_model_config(model_names[0])
        assert isinstance(first_model, ModelConfig)
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_model_manager():
    """Test model manager initialization."""
    try:
        print("\nTesting model manager...")
        
        from utils.model_manager import ModelManager
        
        # Test initialization
        manager = ModelManager()
        assert manager is not None
        
        # Test chat history
        manager.clear_chat_history()
        history = manager.get_chat_history()
        assert isinstance(history, list)
        assert len(history) == 0
        
        print("✅ Model manager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return False


def test_error_handler():
    """Test error handling functionality."""
    try:
        print("\nTesting error handler...")
        
        from utils.error_handler import ErrorHandler, ModelError, APIError
        
        # Test custom exceptions
        try:
            raise ModelError("Test model error", model_name="test-model")
        except ModelError as e:
            assert e.error_code == "MODEL_ERROR"
            assert e.details["model_name"] == "test-model"
        
        try:
            raise APIError("Test API error", api_endpoint="test-endpoint", status_code=404)
        except APIError as e:
            assert e.error_code == "API_ERROR"
            assert e.details["status_code"] == 404
        
        print("✅ Error handler tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handler test failed: {e}")
        return False


def test_validators():
    """Test validation functions."""
    try:
        print("\nTesting validators...")
        
        from utils.error_handler import Validators
        
        # Test API token validation
        assert Validators.is_valid_api_token("hf_1234567890123456789012345") == True
        assert Validators.is_valid_api_token("invalid_token") == False
        assert Validators.is_valid_api_token("") == False
        
        # Test model name validation
        assert Validators.is_valid_model_name("microsoft/DialoGPT-medium") == True
        assert Validators.is_valid_model_name("invalid-model") == False
        
        # Test temperature validation
        assert Validators.is_valid_temperature(0.7) == True
        assert Validators.is_valid_temperature(-0.1) == False
        assert Validators.is_valid_temperature(2.5) == False
        
        # Test max tokens validation
        assert Validators.is_valid_max_tokens(512) == True
        assert Validators.is_valid_max_tokens(0) == False
        assert Validators.is_valid_max_tokens(5000) == False
        
        # Test safe input validation
        assert Validators.is_safe_input("Hello, world!") == True
        assert Validators.is_safe_input("<script>alert('xss')</script>") == False
        
        print("✅ Validators tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Validators test failed: {e}")
        return False


def test_ui_helpers():
    """Test UI helper functions."""
    try:
        print("\nTesting UI helpers...")
        
        from utils.ui_helpers import SessionManager
        
        # Test session defaults
        defaults = {
            "messages": [],
            "model_loaded": False,
            "current_model": None,
            "api_token": "",
        }
        
        # This would normally initialize session state, but we can't test it without Streamlit
        # Just verify the class exists and has the right methods
        assert hasattr(SessionManager, 'initialize_session_state')
        assert hasattr(SessionManager, 'add_message')
        assert hasattr(SessionManager, 'clear_messages')
        assert hasattr(SessionManager, 'get_conversation_metrics')
        
        print("✅ UI helpers tests passed")
        return True
        
    except Exception as e:
        print(f"❌ UI helpers test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    try:
        print("\nTesting file structure...")
        
        required_files = [
            "app.py",
            "pages/aiagent.py",
            "utils/__init__.py",
            "utils/config.py",
            "utils/model_manager.py",
            "utils/ui_helpers.py",
            "utils/error_handler.py",
            "pages/css/styles.css",
            "pages/js/app.js",
            "pages/component/__init__.py",
            "pages/component/chat_components.py",
            "requirements.txt",
            "README.md"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                print(f"❌ Missing file: {file_path}")
                return False
        
        # Test directory structure
        required_dirs = [
            "utils",
            "pages",
            "pages/css",
            "pages/js", 
            "pages/component",
            "logs"
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                print(f"❌ Missing directory: {dir_path}")
                return False
        
        print("✅ File structure tests passed")
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🤖 DurgasAI Integration Tests")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_model_manager,
        test_error_handler,
        test_validators,
        test_ui_helpers,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
