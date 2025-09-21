"""
Configuration module for DurgasAI application.
Contains model configurations, API settings, and application constants.

This module provides:
- Model configuration management for different AI providers
- Application settings and constants
- Environment variable handling
- TensorFlow warning suppression
- Configuration validation and logging

Key Classes:
- ModelProvider: Enum for different AI model providers
- ModelConfig: Configuration data class for individual models
- Config: Main configuration class with all application settings
"""

import os
import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

if TYPE_CHECKING:
    from services import ConfigService

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    # python-dotenv not available, continue without it
    print("ℹ️ python-dotenv not available, using system environment variables")

# Environment variables will be set dynamically from config
# See setup_environment_variables() function below

# Create cache directories if they don't exist
def _create_cache_directories():
    """
    Create HuggingFace cache directories if they don't exist.
    
    This function ensures all required cache directories are available for:
    - HuggingFace Hub downloads and model caching
    - Transformers library model and tokenizer caching  
    - PyTorch model weights and optimization caching
    - Model offloading for large models
    
    The directories are created with proper permissions and parent directory
    structure to support the AI model caching system.
    """
    # Import logger after it's available
    try:
        from .logger import debug, info
    except ImportError:
        # Fallback to print statements if logger not available
        def debug(msg, component="config", **kwargs): print(f"[DEBUG] {msg}")
        def info(msg, component="config", **kwargs): print(f"[INFO] {msg}")
    
    # Define cache directory structure for different ML components
    cache_dirs = [
        './output/cache/huggingface',    # HuggingFace Hub cache
        './output/cache/transformers',   # Transformers library cache
        './output/cache/torch',          # PyTorch model cache
        './output/cache/offload'         # Model offloading cache
    ]
    
    debug("Starting cache directory creation process", "config", 
          cache_dirs=cache_dirs, total_dirs=len(cache_dirs))
    
    print("🔧 Creating HuggingFace cache directories...")
    
    created_dirs = []
    existing_dirs = []
    
    # Process each cache directory
    for cache_dir in cache_dirs:
        debug(f"Processing cache directory: {cache_dir}", "config")
        
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            try:
                # Create directory with all parent directories
                cache_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(cache_dir)
                debug(f"Successfully created cache directory: {cache_dir}", "config")
                print(f"  📁 Created: {cache_dir}")
            except Exception as e:
                debug(f"Failed to create cache directory {cache_dir}: {e}", "config")
                print(f"  ❌ Failed to create: {cache_dir}")
        else:
            existing_dirs.append(cache_dir)
            debug(f"Cache directory already exists: {cache_dir}", "config")
            print(f"  ✓ Exists: {cache_dir}")
    
    # Log summary of directory creation
    if created_dirs:
        info(f"Created {len(created_dirs)} new cache directories", "config",
             created_directories=created_dirs)
        print(f"✅ Created {len(created_dirs)} new cache directories")
    if existing_dirs:
        info(f"Found {len(existing_dirs)} existing cache directories", "config",
             existing_directories=existing_dirs)
        print(f"ℹ️ Found {len(existing_dirs)} existing cache directories")
    
    debug("Cache directory creation process completed", "config",
          total_processed=len(cache_dirs),
          created_count=len(created_dirs),
          existing_count=len(existing_dirs))
    
    print("✅ HuggingFace cache directories configured successfully")

# Initialize cache directories
_create_cache_directories()

# Environment variables will be initialized after Config class is defined
# See initialization at the end of this file

# Additional suppression for ML library warnings
import warnings
import logging

# Suppress all TensorFlow/Keras/ML warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')
warnings.filterwarnings('ignore', message=r'.*tf\..*')

# Suppress logging from ML libraries
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

print("✅ TensorFlow warning suppression configured")


class ModelProvider(Enum):
    """Enum for different model providers."""
    HUGGINGFACE_API = "huggingface_api"
    HUGGINGFACE_LOCAL = "huggingface_local"
    OPENAI = "openai"


@dataclass
class ModelConfig:
    """Configuration for AI models with modern Transformers features."""
    name: str
    model_id: str
    provider: ModelProvider
    description: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    top_k: int = 50
    do_sample: bool = True
    early_stopping: bool = True
    num_return_sequences: int = 1
    use_cache: bool = True
    # Modern Transformers features
    device_map: str = "auto"
    dtype: str = "auto"
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    attn_implementation: str = "flash_attention_2"  # or None for default


class Config:
    """
    Main configuration class for the DurgasAI application.
    
    This class centralizes all application configuration including:
    - Application metadata and UI settings
    - API credentials and authentication
    - Model configurations and provider settings
    - Default system prompts and templates
    - Performance and caching parameters
    
    The configuration system supports:
    - Environment variable integration
    - Dynamic configuration loading
    - HuggingFace-specific optimizations
    - Multi-provider model support
    - Extensible configuration patterns
    
    Configuration Sources:
    1. Environment variables (highest priority)
    2. Configuration files (config/*.json)
    3. Default values (fallback)
    
    Usage:
    ```python
    # Get model configuration
    model_config = Config.get_model_config("mistral_7b")
    
    # Load HuggingFace settings
    hf_config = Config.load_huggingface_config()
    
    # Get available models
    models = Config.get_api_models()
    ```
    """
    
    # Application settings and metadata
    # These define the basic application appearance and behavior
    APP_TITLE = "🤖 DurgasAI - Advanced AI Agent"
    APP_ICON = "🤖"
    LAYOUT = "wide"
    
    # API Keys from environment variables
    # These are loaded from environment or .env file for security
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    @classmethod
    def load_huggingface_config(cls):
        """
        Load HuggingFace configuration from distributed config files.
        
        This method performs comprehensive HuggingFace configuration loading:
        1. Uses ConfigService to load configuration from config/ directory
        2. Extracts HuggingFace and model path settings
        3. Updates environment variables for proper caching
        4. Creates necessary cache directories
        5. Validates configuration completeness
        
        Environment Variables Set:
        - HUGGINGFACE_HUB_CACHE: Main HuggingFace cache directory
        - HF_HOME: HuggingFace home directory
        - TRANSFORMERS_CACHE: Transformers library cache
        - TORCH_HOME: PyTorch cache directory
        
        Returns:
            Dict[str, Any]: Loaded HuggingFace configuration or empty dict on error
            
        Note:
            This method is called during application initialization to ensure
            proper caching behavior for all HuggingFace operations.
        """
        print("🔧 Loading HuggingFace configuration from distributed config files...")
        
        try:
            # Use ConfigService to load configuration
            from services import ConfigService
            config_service = ConfigService()
            
            print("✓ ConfigService initialized, loading HuggingFace config...")
            
            # Get HuggingFace configuration from API config
            hf_config = config_service.get_huggingface_config()
            
            if hf_config:
                print(f"✓ HuggingFace configuration loaded: {len(hf_config)} sections")
                
                # Extract model paths from environment variables
                env_vars = hf_config.get('environment_variables', {})
                model_paths = env_vars.get('model_paths', {})
                
                print(f"📋 HuggingFace config sections: {list(hf_config.keys())}")
                print(f"📋 Model path configs: {list(model_paths.keys())}")
                
                # Update environment variables with config values
                env_updates = []
                
                if hf_config.get('cache_dir'):
                    os.environ['HUGGINGFACE_HUB_CACHE'] = hf_config['cache_dir']
                    os.environ['HF_HOME'] = hf_config['cache_dir']
                    env_updates.append(f"HUGGINGFACE_HUB_CACHE={hf_config['cache_dir']}")
                
                if model_paths.get('transformers_cache'):
                    os.environ['TRANSFORMERS_CACHE'] = model_paths['transformers_cache']
                    env_updates.append(f"TRANSFORMERS_CACHE={model_paths['transformers_cache']}")
                
                if model_paths.get('torch_home'):
                    os.environ['TORCH_HOME'] = model_paths['torch_home']
                    env_updates.append(f"TORCH_HOME={model_paths['torch_home']}")
                
                print(f"🔧 Updated {len(env_updates)} environment variables")
                for update in env_updates:
                    print(f"  ✓ {update}")
                
                # Create cache directories
                cache_dirs = [
                    hf_config.get('cache_dir', './output/cache/huggingface'),
                    model_paths.get('transformers_cache', './output/cache/transformers'),
                    model_paths.get('torch_home', './output/cache/torch'),
                    hf_config.get('offload_folder', './output/cache/offload')
                ]
                
                print("📁 Creating cache directories...")
                created_count = 0
                
                for cache_dir in cache_dirs:
                    if cache_dir:
                        cache_path = Path(cache_dir)
                        if not cache_path.exists():
                            cache_path.mkdir(parents=True, exist_ok=True)
                            created_count += 1
                            print(f"  📁 Created: {cache_dir}")
                        else:
                            print(f"  ✓ Exists: {cache_dir}")
                
                print(f"✅ HuggingFace configuration loaded successfully")
                print(f"📊 Cache directory: {hf_config.get('cache_dir', 'default')}")
                print(f"📊 Created {created_count} new directories")
                
                return hf_config
            else:
                print("⚠️ HuggingFace configuration not found in config files, using defaults")
                return {}
                
        except ImportError as e:
            print(f"⚠️ Could not import ConfigService: {e}")
            print("🔍 Falling back to default configuration")
            return {}
        except Exception as e:
            print(f"⚠️ Could not load HuggingFace config: {e}")
            print(f"🔍 Error details: {str(e)}")
            return {}
    
    @classmethod
    def get_huggingface_download_kwargs(cls):
        """Get HuggingFace download arguments from configuration."""
        config = cls.load_huggingface_config()
        
        kwargs = {
            'cache_dir': config.get('cache_dir', './output/cache/huggingface'),
            'local_files_only': config.get('local_files_only', False),
            'use_auth_token': config.get('use_auth_token', True),
            'force_download': config.get('force_download', False),
            'resume_download': config.get('resume_download', True),
            'use_fast_tokenizer': config.get('use_fast_tokenizer', True),
            'trust_remote_code': config.get('trust_remote_code', False),
            'revision': config.get('revision', 'main'),
            'torch_dtype': config.get('torch_dtype', 'auto'),
            'device_map': config.get('device_map', 'auto'),
            'low_cpu_mem_usage': config.get('low_cpu_mem_usage', True),
            'offload_folder': config.get('offload_folder', './output/cache/offload'),
            'max_memory': config.get('max_memory', None)
        }
        
        # Remove None values
        return {k: v for k, v in kwargs.items() if v is not None}
    
    @classmethod
    def setup_environment_variables(cls, force_update: bool = False):
        """
        Set up environment variables from configuration files.
        
        Args:
            force_update (bool): Whether to force update existing environment variables
        """
        print("🔧 Setting up environment variables from configuration...")
        
        try:
            # Use ConfigService to load environment variable configuration
            from services import ConfigService
            config_service = ConfigService()
            
            # Get environment variables from API config
            api_config = config_service.get_api_config()
            env_vars = api_config.get('environment_variables', {})
            
            # Environment variable mapping
            env_mapping = {
                # TensorFlow variables
                'tensorflow': {
                    'cpp_min_log_level': 'TF_CPP_MIN_LOG_LEVEL',
                    'enable_onednn_opts': 'TF_ENABLE_ONEDNN_OPTS',
                    'tf2_behavior': 'TF2_BEHAVIOR',
                    'cuda_visible_devices': 'CUDA_VISIBLE_DEVICES'
                },
                # PyTorch variables
                'pytorch': {
                    'cuda_alloc_conf': 'PYTORCH_CUDA_ALLOC_CONF',
                    'tokenizers_parallelism': 'TOKENIZERS_PARALLELISM'
                },
                # Python variables
                'python': {
                    'warnings': 'PYTHONWARNINGS'
                },
                # HuggingFace variables
                'huggingface': {
                    'hub_cache': 'HUGGINGFACE_HUB_CACHE',
                    'home': 'HF_HOME',
                    'transformers_cache': 'TRANSFORMERS_CACHE'
                },
                # Model paths
                'model_paths': {
                    'torch_home': 'TORCH_HOME'
                }
            }
            
            set_count = 0
            updated_count = 0
            
            # Process each category of environment variables
            for category, mappings in env_mapping.items():
                category_config = env_vars.get(category, {})
                
                for config_key, env_var_name in mappings.items():
                    if config_key in category_config:
                        config_value = str(category_config[config_key])
                        
                        # Check if environment variable already exists
                        current_value = os.environ.get(env_var_name)
                        
                        if current_value is None:
                            # Set new environment variable
                            os.environ[env_var_name] = config_value
                            set_count += 1
                            print(f"  ✓ Set {env_var_name} = {config_value}")
                        elif force_update or current_value != config_value:
                            # Update existing environment variable
                            os.environ[env_var_name] = config_value
                            updated_count += 1
                            print(f"  🔄 Updated {env_var_name} = {config_value}")
                        else:
                            print(f"  ℹ️ Kept {env_var_name} = {current_value}")
            
            print(f"✅ Environment variables configured: {set_count} new, {updated_count} updated")
            return True
            
        except ImportError as e:
            print(f"⚠️ Could not import ConfigService: {e}")
            print("🔍 Falling back to default environment variable setup")
            cls._setup_default_environment_variables()
            return False
        except Exception as e:
            print(f"⚠️ Could not load environment configuration: {e}")
            print(f"🔍 Error details: {str(e)}")
            cls._setup_default_environment_variables()
            return False
    
    @classmethod
    def _setup_default_environment_variables(cls):
        """Set up default environment variables as fallback."""
        default_env_vars = {
            'TF_ENABLE_ONEDNN_OPTS': '0',
            'TF_CPP_MIN_LOG_LEVEL': '2',
            'TF2_BEHAVIOR': '1',
            'CUDA_VISIBLE_DEVICES': '',
            'PYTHONWARNINGS': 'ignore',
            'HUGGINGFACE_HUB_CACHE': './output/cache/huggingface',
            'HF_HOME': './output/cache/huggingface',
            'TRANSFORMERS_CACHE': './output/cache/transformers',
            'TORCH_HOME': './output/cache/torch',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'TOKENIZERS_PARALLELISM': 'false'
        }
        
        set_count = 0
        for env_var, default_value in default_env_vars.items():
            if env_var not in os.environ:
                os.environ[env_var] = default_value
                set_count += 1
        
        print(f"🔧 Set {set_count} default environment variables")
    
    @classmethod
    def get_environment_variables(cls) -> Dict[str, str]:
        """
        Get current environment variables relevant to the application.
        
        Returns:
            Dict[str, str]: Dictionary of environment variables and their values
        """
        relevant_env_vars = [
            'TF_ENABLE_ONEDNN_OPTS', 'TF_CPP_MIN_LOG_LEVEL', 'TF2_BEHAVIOR', 'CUDA_VISIBLE_DEVICES',
            'PYTORCH_CUDA_ALLOC_CONF', 'TOKENIZERS_PARALLELISM', 'PYTHONWARNINGS',
            'HUGGINGFACE_HUB_CACHE', 'HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME'
        ]
        
        return {var: os.environ.get(var, '') for var in relevant_env_vars}
    
    @classmethod
    def update_environment_variable(cls, var_name: str, value: str, save_to_config: bool = True) -> bool:
        """
        Update a single environment variable and optionally save to config.
        
        Args:
            var_name (str): Environment variable name
            value (str): New value
            save_to_config (bool): Whether to save the change to configuration
            
        Returns:
            bool: True if successfully updated
        """
        try:
            # Update environment variable
            old_value = os.environ.get(var_name, '')
            os.environ[var_name] = value
            
            if save_to_config:
                # Save to configuration file
                # This would require updating the config file structure
                print(f"🔄 Updated {var_name}: '{old_value}' → '{value}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update {var_name}: {e}")
            return False
    
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


class ConfigManager:
    """
    Advanced configuration manager for DurgasAI with validation and logging.
    
    This class handles:
    - Loading configuration from JSON files
    - API key management and validation
    - Configuration validation and error handling
    - Environment variable integration
    - Configuration change logging
    """
    
    def __init__(self, config_service: Optional['ConfigService'] = None):
        """
        Initialize the configuration manager with ConfigService.
        
        Args:
            config_service: Optional ConfigService instance. If None, creates new one.
        """
        if config_service is None:
            from services import ConfigService
            self.config_service = ConfigService()
        else:
            self.config_service = config_service
            
        self.config_data = {}
        self.load_configuration()
    
    def load_configuration(self):
        """
        Load configuration from JSON file with error handling.
        
        This method:
        1. Checks if configuration file exists
        2. Loads and validates JSON configuration
        3. Provides fallback defaults if file is missing
        4. Logs configuration loading status
        """
        # Import logger after it's available
        try:
            from .logger import debug, info, warning, error
        except ImportError:
            # Fallback to print statements if logger not available
            def debug(msg, component="config", **kwargs): print(f"[DEBUG] {msg}")
            def info(msg, component="config", **kwargs): print(f"[INFO] {msg}")
            def warning(msg, component="config", **kwargs): print(f"[WARNING] {msg}")
            def error(msg, component="config", **kwargs): print(f"[ERROR] {msg}")
        
        debug(f"Starting configuration load from: {self.config_file}", "config")
        
        try:
            # Load all configurations through ConfigService
            app_config = self.config_service.get_app_config()
            api_config = self.config_service.get_api_config()
            comprehensive_config = self.config_service.get_comprehensive_config()
            
            # Consolidate configurations for backward compatibility
            self.config_data = {
                **comprehensive_config,  # Features, tools, workflows, etc.
                **api_config,  # API keys and endpoints
                'application': app_config.get('application', {}),
                'ui': app_config.get('ui', {}),
                'chat': {**app_config.get('chat', {}), **comprehensive_config.get('chat', {})},
                'logging': app_config.get('logging', {}),
                'performance': app_config.get('performance', {})
            }
            
            if self.config_data:
                debug(f"Configuration loaded from distributed files", "config",
                      sections_loaded=len(self.config_data))
                
                info(f"Configuration loaded successfully from distributed config files", "config",
                     sections_loaded=len(self.config_data),
                     config_sections=list(self.config_data.keys()))
                
                print(f"✅ Configuration loaded from distributed config files")
                print(f"   Loaded {len(self.config_data)} configuration sections")
                
                # Validate critical sections
                debug("Starting configuration validation", "config")
                self._validate_configuration()
                debug("Configuration validation completed", "config")
                
            else:
                warning(f"Configuration file not found: {self.config_file}", "config")
                print(f"⚠️ Configuration file not found: {self.config_file}")
                print("   Using default configuration")
                
                debug("Loading default configuration", "config")
                self.config_data = self._get_default_configuration()
                info("Default configuration loaded", "config",
                     sections_count=len(self.config_data))
                
        except json.JSONDecodeError as e:
            error(f"JSON parsing error in configuration file: {e}", "config", e)
            print(f"❌ Error parsing configuration JSON: {e}")
            print("   Using default configuration")
            
            debug("Loading default configuration due to JSON error", "config")
            self.config_data = self._get_default_configuration()
            
        except Exception as e:
            error(f"Unexpected error loading configuration: {e}", "config", e)
            print(f"❌ Error loading configuration: {e}")
            print("   Using default configuration")
            
            debug("Loading default configuration due to unexpected error", "config")
            self.config_data = self._get_default_configuration()
    
    def _validate_configuration(self):
        """Validate configuration structure and required fields."""
        # Import logger after it's available
        try:
            from .logger import debug, warning
        except ImportError:
            # Fallback to print statements if logger not available
            def debug(msg, component="config", **kwargs): print(f"[DEBUG] {msg}")
            def warning(msg, component="config", **kwargs): print(f"[WARNING] {msg}")
        
        # Define required configuration sections for proper application functioning
        required_sections = ["huggingface", "chat", "ui"]
        
        debug("Starting configuration validation", "config",
              required_sections=required_sections,
              current_sections=list(self.config_data.keys()))
        
        missing_sections = []
        
        for section in required_sections:
            debug(f"Validating configuration section: {section}", "config")
            
            if section not in self.config_data:
                warning(f"Missing required configuration section: {section}", "config",
                       section_name=section)
                print(f"⚠️ Missing configuration section: {section}")
                
                # Create empty section as fallback
                self.config_data[section] = {}
                missing_sections.append(section)
                debug(f"Created empty section for: {section}", "config")
            else:
                debug(f"Configuration section validated: {section}", "config",
                      section_keys=list(self.config_data[section].keys()) if isinstance(self.config_data[section], dict) else "non-dict")
        
        if missing_sections:
            warning(f"Created {len(missing_sections)} missing configuration sections", "config",
                   missing_sections=missing_sections)
        else:
            debug("All required configuration sections are present", "config")
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration when file is not available."""
        return {
            "huggingface": {"api_keys": ""},
            "chat": {
                "default_max_tokens": 2048,
                "default_temperature": 0.7,
                "enable_chat_history": True,
                "max_history_length": 50
            },
            "ui": {
                "theme": "light",
                "page_title": "DurgasAI",
                "enable_debug_mode": False
            },
            "logging": {
                "log_level": "INFO",
                "enable_debug_logging": False
            }
        }
    
    def get_api_key(self, provider: str) -> str:
        """
        Get API key for a specific provider.
        
        Args:
            provider (str): Provider name (e.g., 'huggingface', 'openai')
            
        Returns:
            str: API key or empty string if not found
        """
        provider_config = self.config_data.get(provider, {})
        api_key = provider_config.get("api_keys", "")
        
        # Don't log the actual key, just its presence
        key_status = "present" if api_key and api_key.strip() else "missing"
        print(f"🔑 API key for {provider}: {key_status}")
        
        return api_key if api_key else ""
    
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Set API key for a provider and save to configuration.
        
        Args:
            provider (str): Provider name
            api_key (str): API key to set
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            if provider not in self.config_data:
                self.config_data[provider] = {}
            
            self.config_data[provider]["api_keys"] = api_key
            
            # Save configuration
            return self.save_configuration()
            
        except Exception as e:
            print(f"❌ Error setting API key for {provider}: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section (str): Section name
            
        Returns:
            Dict[str, Any]: Configuration section or empty dict
        """
        return self.config_data.get(section, {})
    
    def update_config_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update a configuration section.
        
        Args:
            section (str): Section name
            updates (Dict[str, Any]): Updates to apply
            
        Returns:
            bool: True if successfully updated and saved
        """
        try:
            if section not in self.config_data:
                self.config_data[section] = {}
            
            self.config_data[section].update(updates)
            return self.save_configuration()
            
        except Exception as e:
            print(f"❌ Error updating configuration section {section}: {e}")
            return False


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Initialize environment variables from configuration after Config class is defined
try:
    Config.setup_environment_variables()
except Exception as e:
    print(f"⚠️ Could not setup environment variables from config: {e}")
    print("🔧 Using default environment variable setup")
    Config._setup_default_environment_variables()
