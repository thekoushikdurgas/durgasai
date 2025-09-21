"""
Configuration Service for DurgasAI.

Handles configuration loading, validation, and management across
multiple configuration files.

This service provides centralized configuration management for DurgasAI:
- Loads and validates JSON configuration files
- Provides type-safe access to configuration data
- Handles missing or corrupted configuration files gracefully
- Supports saving and updating configurations
- Comprehensive logging for debugging configuration issues

The service supports multiple configuration types:
- app_config.json: Application-wide settings
- model_config.json: AI model configurations
- api_config.json: API keys and endpoint configurations

Architecture:
- Singleton pattern for global configuration access
- Lazy loading with caching for performance
- Error recovery with fallback defaults
- Comprehensive logging for troubleshooting
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, warning, error


class ConfigService:
    """Service for managing application configuration."""
    
    def __init__(self):
        """Initialize the configuration service."""
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.config_files = {
            'app': 'config/app_config.json',
            'models': 'config/model_config.json', 
            'api': 'config/api_config.json',
            'comprehensive': 'config/comprehensive_config.json',
            'model_catalog': 'config/comprehensive_model_catalog.json'
        }
        
        debug("ConfigService initialized", "config_service")
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        for config_name, config_path in self.config_files.items():
            self.configs[config_name] = self._load_config_file(config_path, config_name)
        
        info(f"Loaded {len(self.configs)} configuration files", "config_service")
    
    def _load_config_file(self, file_path: str, config_name: str) -> Dict[str, Any]:
        """Load a single configuration file."""
        config_path = Path(file_path)
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                debug(f"Configuration loaded: {config_name}", "config_service", 
                      file_path=str(config_path),
                      sections=len(config_data))
                
                return config_data
            else:
                warning(f"Configuration file not found: {config_path}", "config_service")
                return {}
                
        except json.JSONDecodeError as e:
            error(f"Invalid JSON in config file: {config_path}", "config_service", e)
            return {}
        except Exception as e:
            error(f"Error loading config file: {config_path}", "config_service", e)
            return {}
    
    def get_config(self, config_name: str, section: str = None) -> Dict[str, Any]:
        """
        Get configuration data with detailed logging and validation.
        
        This method provides safe access to configuration data with:
        - Validation of configuration name
        - Optional section-specific access
        - Fallback to empty dict for missing configs
        - Detailed logging for debugging
        
        Args:
            config_name (str): Name of the config file ('app', 'models', 'api')
            section (str, optional): Optional section within the config
            
        Returns:
            Dict[str, Any]: Configuration dictionary (empty if not found)
        """
        debug(f"Retrieving configuration: {config_name}", "config_service",
              section=section, available_configs=list(self.configs.keys()))
        
        # Validate config name
        if config_name not in self.configs:
            warning(f"Configuration not found: {config_name}", "config_service",
                   available_configs=list(self.configs.keys()))
            return {}
        
        config_data = self.configs.get(config_name, {})
        
        if section:
            # Access specific section with validation
            if section not in config_data:
                warning(f"Section not found in {config_name}: {section}", "config_service",
                       available_sections=list(config_data.keys()))
                return {}
            
            result = config_data.get(section, {})
            debug(f"Retrieved config section: {config_name}.{section}", "config_service",
                  section_keys=list(result.keys()) if isinstance(result, dict) else None)
            return result
        
        debug(f"Retrieved full configuration: {config_name}", "config_service",
              config_keys=list(config_data.keys()) if isinstance(config_data, dict) else None)
        return config_data
    
    def get_app_config(self) -> Dict[str, Any]:
        """
        Get application configuration with logging.
        
        Returns:
            Dict[str, Any]: Application configuration data
        """
        debug("Retrieving application configuration", "config_service")
        config = self.get_config('app')
        debug("Application configuration retrieved", "config_service",
              config_sections=list(config.keys()) if config else [])
        return config
    
    def get_model_config(self, model_id: str = None) -> Dict[str, Any]:
        """
        Get model configuration with optional model-specific filtering.
        
        Args:
            model_id (str, optional): Specific model ID to retrieve config for
            
        Returns:
            Dict[str, Any]: Model configuration data
        """
        debug("Retrieving model configuration", "config_service", model_id=model_id)
        models_config = self.get_config('models', 'models')
        
        if model_id:
            # Get specific model configuration
            model_config = models_config.get(model_id, {})
            if model_config:
                debug(f"Model configuration found for: {model_id}", "config_service")
            else:
                warning(f"No configuration found for model: {model_id}", "config_service",
                       available_models=list(models_config.keys()))
            return model_config
        
        debug("Retrieved all model configurations", "config_service",
              models_count=len(models_config))
        return models_config
    
    def get_api_config(self, provider: str = None) -> Dict[str, Any]:
        """
        Get API configuration with optional provider filtering.
        
        Args:
            provider (str, optional): Specific API provider to get config for
            
        Returns:
            Dict[str, Any]: API configuration data
        """
        debug("Retrieving API configuration", "config_service", provider=provider)
        
        if provider:
            # Get specific provider configuration
            provider_config = self.get_config('api', provider)
            if provider_config:
                debug(f"API configuration found for provider: {provider}", "config_service")
            else:
                warning(f"No API configuration found for provider: {provider}", "config_service")
            return provider_config
        
        api_config = self.get_config('api')
        debug("Retrieved all API configurations", "config_service",
              providers_count=len(api_config))
        return api_config
    
    def get_comprehensive_config(self, section: str = None) -> Dict[str, Any]:
        """
        Get comprehensive configuration (features, tools, workflows, etc.).
        
        Args:
            section (str, optional): Specific section to retrieve
            
        Returns:
            Dict[str, Any]: Comprehensive configuration data
        """
        debug("Retrieving comprehensive configuration", "config_service", section=section)
        
        comprehensive_config = self.get_config('comprehensive', section)
        if comprehensive_config:
            debug("Comprehensive configuration retrieved", "config_service",
                  sections=list(comprehensive_config.keys()) if isinstance(comprehensive_config, dict) else None)
        return comprehensive_config
    
    def get_model_catalog(self) -> Dict[str, Any]:
        """
        Get model catalog configuration.
        
        Returns:
            Dict[str, Any]: Model catalog data
        """
        debug("Retrieving model catalog", "config_service")
        catalog = self.get_config('model_catalog')
        debug("Model catalog retrieved", "config_service",
              categories=list(catalog.get('categories', {}).keys()) if catalog else [])
        return catalog
    
    def get_chat_config(self) -> Dict[str, Any]:
        """Get chat-specific configuration."""
        return self.get_comprehensive_config('chat')
    
    def get_tools_config(self) -> Dict[str, Any]:
        """Get tools configuration."""
        return self.get_comprehensive_config('tools')
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration from app config."""
        return self.get_app_config().get('ui', {})
    
    def get_analytics_config(self) -> Dict[str, Any]:
        """Get analytics configuration."""
        return self.get_comprehensive_config('analytics')
    
    def get_huggingface_config(self) -> Dict[str, Any]:
        """
        Get HuggingFace configuration from API config.
        
        Returns:
            Dict[str, Any]: HuggingFace configuration including API keys and settings
        """
        debug("Retrieving HuggingFace configuration", "config_service")
        api_config = self.get_api_config()
        hf_config = api_config.get('huggingface', {})
        
        # Also get environment variables for model paths
        env_vars = api_config.get('environment_variables', {})
        if env_vars:
            hf_config = {**hf_config, 'environment_variables': env_vars}
        
        debug("HuggingFace configuration retrieved", "config_service",
              has_api_key=bool(hf_config.get('api_keys')),
              cache_dir=hf_config.get('cache_dir'))
        return hf_config
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_name: Name of the config file
            config_data: Configuration data to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        config_path = Path(self.config_files.get(config_name, ''))
        
        if not config_path:
            error(f"Unknown config name: {config_name}", "config_service")
            return False
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # Update in-memory config
            self.configs[config_name] = config_data
            
            info(f"Configuration saved: {config_name}", "config_service")
            return True
            
        except Exception as e:
            error(f"Error saving config: {config_name}", "config_service", e)
            return False
