"""
Configuration module for DurgasAI application.
Contains model configurations, API settings, and application constants.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, continue without it
    pass

# Set TensorFlow environment variables early
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF2_BEHAVIOR', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

# Additional suppression for warnings
import warnings
import logging

# Suppress all TensorFlow/Keras warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')
warnings.filterwarnings('ignore', message='.*tf\..*')

# Suppress logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)


class ModelProvider(Enum):
    """Enum for different model providers."""
    HUGGINGFACE_API = "huggingface_api"
    HUGGINGFACE_LOCAL = "huggingface_local"
    OPENAI = "openai"


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    name: str
    model_id: str
    provider: ModelProvider
    description: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class Config:
    """Main configuration class for the application."""
    
    # Application settings
    APP_TITLE = "🤖 DurgasAI - Advanced AI Agent"
    APP_ICON = "🤖"
    LAYOUT = "wide"
    
    # API Keys (from environment variables)
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Model configurations
    AVAILABLE_MODELS: Dict[str, ModelConfig] = {
        "mistral_7b": ModelConfig(
            name="Mistral 7B Instruct",
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            provider=ModelProvider.HUGGINGFACE_API,
            description="Powerful instruction-following model",
            max_tokens=512,
            temperature=0.7
        ),
        "zephyr_7b": ModelConfig(
            name="Zephyr 7B Beta",
            model_id="HuggingFaceH4/zephyr-7b-beta",
            provider=ModelProvider.HUGGINGFACE_API,
            description="Great conversational AI model",
            max_tokens=512,
            temperature=0.7
        ),
        "flan_t5": ModelConfig(
            name="Flan T5 Large",
            model_id="google/flan-t5-large",
            provider=ModelProvider.HUGGINGFACE_API,
            description="Google's instruction-tuned T5 model",
            max_tokens=256,
            temperature=0.3
        ),
        "dialogpt_medium": ModelConfig(
            name="DialoGPT Medium",
            model_id="microsoft/DialoGPT-medium",
            provider=ModelProvider.HUGGINGFACE_LOCAL,
            description="Microsoft's conversational AI model",
            max_tokens=150,
            temperature=0.8
        ),
        "blenderbot": ModelConfig(
            name="BlenderBot 400M",
            model_id="facebook/blenderbot-400M-distill",
            provider=ModelProvider.HUGGINGFACE_API,
            description="Facebook's open-domain chatbot",
            max_tokens=200,
            temperature=0.7
        )
    }
    
    # Default system prompts
    DEFAULT_SYSTEM_PROMPTS: Dict[str, str] = {
        "helpful_assistant": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
        "creative_writer": "You are a creative writing assistant. Help users with storytelling, poetry, and creative content.",
        "code_expert": "You are a programming expert. Help users with coding questions, debugging, and best practices.",
        "research_assistant": "You are a research assistant. Help users find information, analyze data, and summarize findings.",
        "tutor": "You are a friendly tutor. Explain concepts clearly and help users learn new topics step by step."
    }
    
    # UI Configuration
    SIDEBAR_WIDTH = 300
    CHAT_HEIGHT = 600
    MAX_MESSAGE_HISTORY = 50
    
    # API Configuration
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # File paths
    ASSETS_DIR = "assets"
    CSS_DIR = "pages/css"
    JS_DIR = "pages/js"
    COMPONENTS_DIR = "pages/component"
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get model configuration by name."""
        return cls.AVAILABLE_MODELS.get(model_name)
    
    @classmethod
    def get_api_models(cls) -> Dict[str, ModelConfig]:
        """Get models that use API."""
        return {
            name: config for name, config in cls.AVAILABLE_MODELS.items()
            if config.provider == ModelProvider.HUGGINGFACE_API
        }
    
    @classmethod
    def get_local_models(cls) -> Dict[str, ModelConfig]:
        """Get models that run locally."""
        return {
            name: config for name, config in cls.AVAILABLE_MODELS.items()
            if config.provider == ModelProvider.HUGGINGFACE_LOCAL
        }
