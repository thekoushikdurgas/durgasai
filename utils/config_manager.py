"""
Configuration Manager for DurgasAI
Handles loading and saving of application configuration
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from .logging_utils import log_info, log_error, log_warning, log_debug
from .security_utils import APISecurityManager

# Constants
DEFAULT_OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
DEFAULT_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229", 
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]
DEFAULT_GOOGLE_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash", 
    "gemini-pro"
]
DEFAULT_GROQ_MODELS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
DEFAULT_LLAMA_MODEL = "llama3.1"
DEFAULT_ALPHANUMERIC_DESC = "20+ alphanumeric characters"

class ConfigManager:
    """Manages DurgasAI application configuration"""
    
    def __init__(self, config_path: str = "output/config/config.json"):
        self.config_path = Path(config_path)
        self._config = None
        self.security_manager = APISecurityManager()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                log_info(f"Configuration loaded from {self.config_path}", "config")
            else:
                log_warning(f"Config file not found at {self.config_path}, creating default", "config")
                self._config = self._get_default_config()
                self._save_config()
        except Exception as e:
            log_error(f"Error loading config from {self.config_path}", "config", e)
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "api_keys": {
                "openai": "",
                "groq": "",
                "tavily": "",
                "huggingface": "",
                "google": "",
                "deepseek": "",
                "mistral": "",
                "perplexity": "",
                "anthropic": "",
                "web_search": "",
                "exa": "",
                "serpapi": "",
                "github": "",
                "openrouter": "",
                "nvidia": "",
                "cohere": "",
                "jina": ""
            },
            "ollama": {
                "host": "http://localhost:11434",
                "model": DEFAULT_LLAMA_MODEL,
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 30,
                "enable_streaming": True,
                "enable_embeddings": True,
                "enable_function_calling": True,
                "max_retries": 3,
                "retry_delay": 1.0,
                "connection_pool_size": 5,
                "enable_model_auto_switch": True,
                "fallback_models": [
                    DEFAULT_LLAMA_MODEL,
                    "gemma2",
                    "qwen2.5",
                    "mistralai/Mistral-7B-Instruct-v0.3"
                ],
                "model_priority": [
                    DEFAULT_LLAMA_MODEL,
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "deepseek-r1",
                    "gpt-oss:20b"
                ]
            },
            "gpt_oss": {
                "enable_gpt_oss": True,
                "default_model": "gpt-oss:20b",
                "enable_browser_tools": True,
                "enable_thinking_mode": True,
                "max_thinking_iterations": 10,
                "browser_timeout": 30,
                "enable_web_search": True,
                "enable_content_analysis": True,
                "max_search_results": 5,
                "enable_streaming_thinking": True
            },
            "langgraph": {
                "enable_langgraph": True,
                "default_workflow": "chatbot_with_tools",
                "max_concurrent_sessions": 10,
                "session_timeout_minutes": 30,
                "enable_tool_calling": True,
                "enable_memory": True,
                "memory_type": "conversation_buffer",
                "max_memory_size": 50,
                "enable_streaming": True,
                "enable_debug_mode": False,
                "max_iterations": 20,
                "recursion_limit": 100,
                "enable_parallel_execution": True,
                "max_parallel_tools": 3,
                "tool_timeout_seconds": 30,
                "enable_tool_validation": True,
                "enable_conversation_memory": True,
                "conversation_memory_size": 100,
                "enable_workflow_persistence": False,
                "workflow_cache_size": 50
            },
            "vector_db": {
                "db_path": "./vector_db",
                "collection_name": "documents",
                "embedding_model": "all-MiniLM-L6-v2",
                "max_results": 5,
                "similarity_threshold": 0.7,
                "enable_persistent_storage": True,
                "enable_metadata_indexing": True,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "enable_auto_cleanup": True,
                "cleanup_interval_days": 30
            },
            "vector_db_advanced": {
                "enable_hybrid_search": True,
                "enable_contextual_search": True,
                "enable_smart_chunking": True,
                "default_chunking_strategy": "semantic",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "context_window": 3,
                "cache_ttl": 300,
                "max_cache_size": 1000,
                "enable_document_insights": True,
                "enable_summarization": True,
                "max_summary_length": 500,
                "enable_reranking": True,
                "reranking_model": "ms-marco-MiniLM-L-12-v2",
                "enable_clustering": False,
                "clustering_algorithm": "kmeans"
            },
            "system_monitor": {
                "refresh_rate": 5,
                "max_history": 100,
                "show_network": True,
                "show_disk": True,
                "alert_cpu_threshold": 80.0,
                "alert_memory_threshold": 85.0,
                "alert_disk_threshold": 90.0,
                "alert_process_threshold": 300,
                "enable_alerts": True,
                "max_alerts_history": 100,
                "enable_email_alerts": False,
                "email_recipients": [],
                "enable_desktop_notifications": True,
                "alert_cooldown_minutes": 5,
                "enable_performance_profiling": False
            },
            "chat": {
                "enable_tools": True,
                "enable_vector_search": True,
                "enable_multimodal": True,
                "enable_structured_outputs": True,
                "enable_thinking_mode": True,
                "enable_async_processing": True,
                "enable_langchain_endpoint": True,
                "default_max_tokens": 2048,
                "default_temperature": 0.7,
                "default_top_p": 0.9,
                "default_top_k": 40,
                "stream_responses": True,
                "enable_avatars": True,
                "max_image_size": 10,
                "supported_image_formats": [
                    "png",
                    "jpg",
                    "jpeg",
                    "gif",
                    "bmp",
                    "webp"
                ],
                "enable_chat_history": True,
                "max_history_length": 50,
                "enable_export_import": True,
                "enable_voice_input": False,
                "enable_voice_output": False,
                "enable_auto_save": True,
                "auto_save_interval": 30,
                "enable_context_awareness": True,
                "context_window_size": 10
            },
            "tools": {
                "enable_tool_management": True,
                "tools_directory": "./output/tools",
                "max_tool_execution_time": 60,
                "enable_tool_caching": True,
                "tool_cache_ttl": 300,
                "enable_tool_logging": True,
                "enable_tool_validation": True,
                "allowed_tool_categories": [
                    "utility",
                    "data",
                    "analysis",
                    "automation",
                    "research"
                ],
                "blocked_tool_functions": [],
                "enable_tool_sandboxing": False,
                "max_concurrent_tool_executions": 5,
                "enable_tool_performance_monitoring": True
            },
            "templates": {
                "enable_template_management": True,
                "templates_directory": "./output/templates",
                "enable_template_sharing": False,
                "enable_template_validation": True,
                "max_template_size_kb": 100,
                "enable_template_caching": True,
                "template_cache_ttl": 600,
                "default_template_categories": [
                    "assistant",
                    "creative",
                    "analytical",
                    "technical"
                ],
                "enable_template_import_export": True
            },
            "workflows": {
                "enable_workflow_management": True,
                "workflows_directory": "./output/workflows",
                "max_workflow_steps": 20,
                "enable_workflow_scheduling": False,
                "enable_workflow_monitoring": True,
                "max_concurrent_workflows": 3,
                "enable_workflow_logging": True,
                "workflow_timeout_minutes": 30,
                "enable_workflow_validation": True,
                "enable_workflow_templates": True
            },
            "sessions": {
                "enable_session_management": True,
                "sessions_directory": "./output/sessions",
                "session_retention_days": 30,
                "max_sessions": 100,
                "enable_session_backup": True,
                "enable_session_compression": False,
                "session_auto_save_interval": 60,
                "enable_session_sharing": False,
                "max_session_size_mb": 10,
                "enable_session_analytics": True,
                "session_cleanup_enabled": True
            },
            "logging": {
                "log_level": "INFO",
                "log_file_path": "./output/logs/genai_agent.log",
                "max_log_file_size_mb": 50,
                "max_log_files": 5,
                "enable_log_rotation": True,
                "enable_log_compression": True,
                "enable_console_logging": True,
                "enable_session_logging": True,
                "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "enable_performance_logging": True,
                "enable_error_tracking": True,
                "log_retention_days": 30,
                "enable_log_analysis": True
            },
            "async_config": {
                "enable_async_chat": True,
                "enable_async_vector_search": True,
                "enable_async_document_processing": True,
                "max_concurrent_requests": 5,
                "timeout_seconds": 30,
                "thread_pool_size": 4,
                "enable_async_tools": True,
                "enable_async_templates": True,
                "enable_async_workflows": True,
                "async_queue_size": 100,
                "enable_async_monitoring": True
            },
            "ui": {
                "theme": "light",
                "page_title": "DurgasAI",
                "page_icon": "🤖",
                "enable_sidebar_collapse": True,
                "enable_auto_refresh": True,
                "auto_refresh_interval": 30,
                "enable_animations": True,
                "enable_sound_effects": False,
                "enable_keyboard_shortcuts": True,
                "default_page": "Tools",
                "enable_page_caching": True,
                "max_cached_pages": 10
            },
            "security": {
                "enable_api_authentication": False,
                "api_key_required": False,
                "enable_rate_limiting": False,
                "max_requests_per_minute": 60,
                "enable_cors": True,
                "allowed_origins": ["*"],
                "enable_input_sanitization": True,
                "enable_output_filtering": False,
                "blocked_keywords": [],
                "enable_session_security": True,
                "session_timeout_minutes": 480,
                "enable_audit_logging": False
            },
            "performance": {
                "enable_caching": True,
                "cache_size_mb": 100,
                "cache_ttl_seconds": 3600,
                "enable_memory_optimization": True,
                "max_memory_usage_mb": 2048,
                "enable_cpu_optimization": True,
                "max_cpu_usage_percent": 80,
                "enable_disk_optimization": True,
                "max_disk_usage_mb": 10240,
                "enable_lazy_loading": True,
                "enable_connection_pooling": True,
                "connection_pool_size": 10,
                "enable_performance_monitoring": True
            },
            "integrations": {
                "enable_web_search": True,
                "web_search_provider": "exa",
                "web_search_api_key": "",
                "enable_file_processing": True,
                "max_file_size_mb": 50,
                "supported_file_types": [
                    "txt",
                    "pdf",
                    "docx",
                    "md",
                    "json",
                    "csv",
                    "xlsx"
                ],
                "enable_email_integration": False,
                "email_provider": "smtp",
                "enable_calendar_integration": False,
                "enable_database_integration": False,
                "enable_api_integrations": False
            },
            "development": {
                "enable_debug_mode": False,
                "enable_hot_reload": False,
                "enable_profiling": False,
                "enable_testing_mode": False,
                "enable_development_logging": False,
                "enable_feature_flags": False,
                "feature_flags": {},
                "enable_experimental_features": False,
                "experimental_features": []
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        return self._config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section"""
        return self._config.get(section, {})
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value"""
        return self._config.get(section, {}).get(key, default)
    
    def set_value(self, section: str, key: str, value: Any):
        """Set a configuration value"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def update_section(self, section: str, values: Dict[str, Any]):
        """Update a configuration section"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section].update(values)
    
    def save_config(self):
        """Save configuration to file"""
        self._save_config()
    
    def _save_config(self):
        """Internal method to save configuration"""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            log_info(f"Configuration saved to {self.config_path}", "config")
        except Exception as e:
            log_error(f"Error saving config to {self.config_path}", "config", e)
    
    def reset_to_default(self):
        """Reset configuration to defaults"""
        self._config = self._get_default_config()
        self._save_config()
    
    def export_config(self, export_path: str):
        """Export configuration to a file"""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error exporting config: {e}")
    
    def import_config(self, import_path: str):
        """Import configuration from a file"""
        try:
            import_path = Path(import_path)
            if import_path.exists():
                with open(import_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                self._save_config()
        except Exception as e:
            print(f"Error importing config: {e}")
    
    # API Key Management Methods
    def get_api_key(self, provider: str, decrypt: bool = False) -> str:
        """Get API key for a specific provider"""
        encrypted_key = self._config.get("api_keys", {}).get(provider, "")
        if decrypt and encrypted_key:
            return self.security_manager.unsecure_api_keys({provider: encrypted_key})[provider]
        return encrypted_key
    
    def set_api_key(self, provider: str, api_key: str, encrypt: bool = True) -> bool:
        """Set API key for a specific provider"""
        try:
            if "api_keys" not in self._config:
                self._config["api_keys"] = {}
            
            # Secure the API key
            secured_key = self.security_manager.secure_api_keys({provider: api_key}, encrypt)[provider]
            self._config["api_keys"][provider] = secured_key
            
            self._save_config()
            log_info(f"API key updated for {provider} (encrypted: {encrypt})", "config")
            return True
        except Exception as e:
            log_error(f"Error setting API key for {provider}", "config", e)
            return False
    
    def delete_api_key(self, provider: str) -> bool:
        """Delete API key for a specific provider"""
        try:
            if "api_keys" in self._config and provider in self._config["api_keys"]:
                del self._config["api_keys"][provider]
                self._save_config()
                log_info(f"API key deleted for {provider}", "config")
                return True
            return False
        except Exception as e:
            log_error(f"Error deleting API key for {provider}", "config", e)
            return False
    
    def get_all_api_keys(self, decrypt: bool = False) -> Dict[str, str]:
        """Get all API keys"""
        encrypted_keys = self._config.get("api_keys", {})
        # print(f'encrypted_keys: {encrypted_keys}')
        if decrypt:
            return self.security_manager.unsecure_api_keys(encrypted_keys)
        # print(f'encrypted_keys: {encrypted_keys}')
        return encrypted_keys
    
    def get_masked_api_keys(self) -> Dict[str, str]:
        """Get masked API keys for display purposes"""
        encrypted_keys = self._config.get("api_keys", {})
        unencrypted_keys = self.security_manager.unsecure_api_keys(encrypted_keys)
        return self.security_manager.mask_api_keys_for_display(unencrypted_keys)
    
    def update_api_keys(self, api_keys: Dict[str, str], encrypt: bool = True) -> bool:
        """Update multiple API keys at once"""
        try:
            if "api_keys" not in self._config:
                self._config["api_keys"] = {}
            
            # Secure all API keys
            secured_keys = self.security_manager.secure_api_keys(api_keys, encrypt)
            self._config["api_keys"].update(secured_keys)
            
            self._save_config()
            log_info("Updated {} API keys (encrypted: {})".format(len(api_keys), encrypt), "config")
            return True
        except Exception as e:
            log_error("Error updating API keys", "config", e)
            return False
    
    def validate_api_key_format(self, provider: str, api_key: str) -> tuple[bool, str]:
        """Validate API key format for a specific provider"""
        if not api_key or not api_key.strip():
            return False, "API key cannot be empty"
        
        api_key = api_key.strip()
        
        # Provider-specific validation patterns
        validation_patterns = {
            "openai": r"^sk-proj-[a-zA-Z0-9_-]{100,}$",
            "anthropic": r"^sk-ant-api03-[a-zA-Z0-9_-]{50,}$",
            "groq": r"^gsk_[a-zA-Z0-9]{20,}$",
            "tavily": r"^tvly-dev-[a-zA-Z0-9]{20,}$",
            "huggingface": r"^hf_[a-zA-Z0-9]{20,}$",
            "google": r"^AIzaSy[a-zA-Z0-9_-]{33}$",
            "deepseek": r"^sk-[a-zA-Z0-9]{32}$",
            "mistral": r"^[a-zA-Z0-9]{20,}$",
            "perplexity": r"^pplx-[a-zA-Z0-9]{40,}$",
            "web_search": r"^[a-zA-Z0-9-_]{10,}$",
            "exa": r"^[a-zA-Z0-9-_]{20,}$",
            "serpapi": r"^[a-zA-Z0-9]{64}$",
            "jina": r"^jina_[a-zA-Z0-9]{60,70}$",
            "github": r"^github_pat_[a-zA-Z0-9]{22}[a-zA-Z0-9_]{20,}$",
            "openrouter": r"^sk-or-v1-[a-zA-Z0-9]{64}$",
            "nvidia": r"^nvapi-[a-zA-Z0-9-_]{65,75}$",
            "cohere": r"^[a-zA-Z0-9]{32,50}$"
        }
        
        # Expected formats for better error messages
        expected_formats = {
            "openai": "sk-proj- followed by 100+ alphanumeric characters, underscores, and hyphens",
            "anthropic": "sk-ant-api03- followed by 50+ alphanumeric characters, underscores, and hyphens",
            "groq": "gsk_ followed by 20+ alphanumeric characters",
            "tavily": "tvly-dev- followed by 20+ alphanumeric characters",
            "huggingface": "hf_ followed by 20+ alphanumeric characters",
            "google": "AIzaSy followed by 33 alphanumeric characters, underscores, and hyphens",
            "deepseek": "sk- followed by exactly 32 alphanumeric characters",
            "mistral": DEFAULT_ALPHANUMERIC_DESC,
            "perplexity": "pplx- followed by 40+ alphanumeric characters",
            "web_search": "10+ alphanumeric characters",
            "exa": DEFAULT_ALPHANUMERIC_DESC,
            "serpapi": "exactly 64 alphanumeric characters",
            "jina": "jina_ followed by 60-70 alphanumeric characters",
            "github": "github_pat_ followed by 22+ alphanumeric characters, then 20+ alphanumeric characters or underscores",
            "openrouter": "sk-or-v1- followed by exactly 64 alphanumeric characters",
            "nvidia": "nvapi- followed by 65-75 alphanumeric characters, hyphens, and underscores",
            "cohere": "32-50 alphanumeric characters"
        }
        
        import re
        pattern = validation_patterns.get(provider)
        
        if not pattern:
            # Generic validation for unknown providers
            if len(api_key) < 10:
                return False, "API key too short (minimum 10 characters, got {})".format(len(api_key))
            return True, "Valid API key format"
        
        if not re.match(pattern, api_key):
            expected = expected_formats.get(provider, "valid format")
            return False, "Invalid format. Expected: {}".format(expected)
        
        return True, "Valid API key format"
    
    def validate_api_key_security(self, _provider: str, api_key: str) -> Dict[str, Any]:
        """Validate API key security characteristics"""
        return self.security_manager.validate_api_key_security(api_key)
    
    def get_api_key_security_report(self) -> Dict[str, Dict[str, Any]]:
        """Get security report for all API keys"""
        api_keys = self.get_all_api_keys()
        security_report = {}
        
        for provider, key in api_keys.items():
            if key and key.strip():
                security_report[provider] = self.validate_api_key_security(provider, key)
            else:
                security_report[provider] = {
                    "is_valid": False,
                    "length": 0,
                    "has_uppercase": False,
                    "has_lowercase": False,
                    "has_digits": False,
                    "has_special_chars": False,
                    "security_score": 0,
                    "status": "empty"
                }
        
        return security_report
    
    def fetch_openai_models(self) -> List[str]:
        """Fetch available OpenAI models from API"""
        try:
            api_key = self.get_api_key("openai")
            if not api_key:
                log_warning("No OpenAI API key found, returning default models", "config")
                return DEFAULT_OPENAI_MODELS
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            # Filter for chat-completion models (exclude embedding, image, audio models)
            chat_models = []
            for model in models:
                if any(prefix in model.lower() for prefix in ["gpt", "o1", "o3", "o4"]):
                    # Exclude specific non-chat models
                    if not any(exclude in model.lower() for exclude in [
                        "embedding", "dall-e", "tts", "whisper", "moderation", "transcribe", "audio"
                    ]):
                        chat_models.append(model)
            
            # Sort models by preference (newer models first)
            model_priority = [
                "gpt-5", "gpt-4.1", "o3", "o1", "gpt-4o", "gpt-4", "gpt-3.5-turbo"
            ]
            
            def sort_key(model):
                for i, priority in enumerate(model_priority):
                    if model.startswith(priority):
                        return (i, model)
                return (len(model_priority), model)
            
            chat_models.sort(key=sort_key)
            
            log_info(f"Fetched {len(chat_models)} OpenAI models from API", "config")
            return chat_models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching OpenAI models: {e}", "config", e)
            return ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
        except Exception as e:
            log_error(f"Unexpected error fetching OpenAI models: {e}", "config", e)
            return ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    
    def fetch_anthropic_models(self) -> List[str]:
        """Fetch available Anthropic models"""
        # Anthropic doesn't have a public models endpoint, so we return known models
        return DEFAULT_ANTHROPIC_MODELS
    
    def fetch_google_models(self) -> List[str]:
        """Fetch available Google models from API"""
        try:
            api_key = self.get_api_key("google")
            if not api_key:
                log_warning("No Google API key found, returning default models", "config")
                return DEFAULT_GOOGLE_MODELS
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Extract model names from the Google API response
            for model in data.get("models", []):
                if isinstance(model, dict) and "name" in model:
                    # Extract the model ID from the full name (e.g., "models/gemini-pro" -> "gemini-pro")
                    model_name = model["name"]
                    if model_name.startswith("models/"):
                        model_id = model_name.replace("models/", "")
                        
                        # Filter for generative models (exclude embedding and other specialized models)
                        if any(keyword in model_id.lower() for keyword in ["gemini", "palm", "bison"]):
                            # Exclude non-generative models
                            if not any(exclude in model_id.lower() for exclude in [
                                "embedding", "aqa", "text-bison-001"
                            ]):
                                models.append(model_id)
            
            # If no models found, return default list
            if not models:
                log_warning("No models found in Google API response, using defaults", "config")
                return DEFAULT_GOOGLE_MODELS
            
            # Sort models by preference (newer/better models first)
            model_priority = [
                "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision",
                "palm", "text-bison", "chat-bison"
            ]
            
            def sort_key(model):
                for i, priority in enumerate(model_priority):
                    if priority in model.lower():
                        return (i, model)
                return (len(model_priority), model)
            
            models.sort(key=sort_key)
            
            log_info(f"Fetched {len(models)} Google models from API", "config")
            return models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching Google models: {e}", "config", e)
            return DEFAULT_GOOGLE_MODELS
        except Exception as e:
            log_error(f"Unexpected error fetching Google models: {e}", "config", e)
        return DEFAULT_GOOGLE_MODELS
    
    def fetch_groq_models(self) -> List[str]:
        """Fetch available Groq models from API"""
        try:
            api_key = self.get_api_key("groq")
            if not api_key:
                log_warning("No Groq API key found, returning default models", "config")
                return DEFAULT_GROQ_MODELS
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            # Filter for chat-completion models (exclude TTS, whisper, and specialized models)
            chat_models = []
            for model in models:
                # Include common chat model patterns
                if any(pattern in model.lower() for pattern in [
                    "llama", "mixtral", "gemma", "qwen", "compound", "gpt-oss", 
                    "kimi", "deepseek", "meta-llama"
                ]):
                    # Exclude specific non-chat models
                    if not any(exclude in model.lower() for exclude in [
                        "whisper", "tts", "guard", "prompt-guard", "scout", "maverick"
                    ]):
                        chat_models.append(model)
            
            # Sort models by preference (popular models first)
            model_priority = [
                "llama-3.3", "llama-3.1", "mixtral", "gemma", "compound", 
                "gpt-oss", "qwen", "kimi", "deepseek"
            ]
            
            def sort_key(model):
                for i, priority in enumerate(model_priority):
                    if priority in model.lower():
                        return (i, model)
                return (len(model_priority), model)
            
            chat_models.sort(key=sort_key)
            
            log_info(f"Fetched {len(chat_models)} Groq models from API", "config")
            return chat_models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching Groq models: {e}", "config", e)
            return ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        except Exception as e:
            log_error(f"Unexpected error fetching Groq models: {e}", "config", e)
            return ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
    
    def fetch_github_models(self) -> List[str]:
        """Fetch available GitHub models from API"""
        try:
            api_key = self.get_api_key("github")
            if not api_key:
                log_warning("No GitHub API key found, returning default models", "config")
                return [
                    "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini",
                    "claude-3-5-sonnet", "claude-3-haiku",
                    "llama-3.1-405b-instruct", "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
                    "phi-3.5-mini-instruct", "mistral-large", "mistral-nemo", "cohere-command-r-plus"
                ]
            
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {api_key}",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            
            response = requests.get("https://models.github.ai/catalog/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Extract model IDs from the GitHub Models API response
            for model in data:
                if isinstance(model, dict) and "id" in model:
                    models.append(model["id"])
                elif isinstance(model, dict) and "name" in model:
                    models.append(model["name"])
            
            # If no models found, return default list
            if not models:
                log_warning("No models found in GitHub API response, using defaults", "config")
                return [
                    "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini",
                    "claude-3-5-sonnet", "claude-3-haiku",
                    "llama-3.1-405b-instruct", "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
                    "phi-3.5-mini-instruct", "mistral-large", "mistral-nemo", "cohere-command-r-plus"
                ]
            
            # Sort models by preference
            model_priority = [
                "gpt-4o", "o1", "claude-3-5", "llama-3.1-405b", "llama-3.1-70b",
                "mistral-large", "cohere-command-r-plus", "phi-3.5", "llama-3.1-8b",
                "claude-3-haiku", "mistral-nemo", "gpt-4o-mini"
            ]
            
            def sort_key(model):
                for i, priority in enumerate(model_priority):
                    if priority in model.lower():
                        return (i, model)
                return (len(model_priority), model)
            
            models.sort(key=sort_key)
            
            log_info(f"Fetched {len(models)} GitHub models from API", "config")
            return models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching GitHub models: {e}", "config", e)
            return [
                "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini",
                "claude-3-5-sonnet", "claude-3-haiku",
                "llama-3.1-405b-instruct", "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
                "phi-3.5-mini-instruct", "mistral-large", "mistral-nemo", "cohere-command-r-plus"
            ]
        except Exception as e:
            log_error(f"Unexpected error fetching GitHub models: {e}", "config", e)
            return [
                "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini",
                "claude-3-5-sonnet", "claude-3-haiku",
                "llama-3.1-405b-instruct", "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
                "phi-3.5-mini-instruct", "mistral-large", "mistral-nemo", "cohere-command-r-plus"
            ]
    
    def fetch_openrouter_models(self) -> List[str]:
        """Fetch available OpenRouter models from API"""
        try:
            api_key = self.get_api_key("openrouter")
            if not api_key:
                log_warning("No OpenRouter API key found, returning default models", "config")
                return [
                    "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1-preview", "openai/o1-mini",
                    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku",
                    "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-70b-instruct",
                    "meta-llama/llama-3.1-8b-instruct", "google/gemini-pro-1.5", "google/gemini-flash-1.5",
                    "mistralai/mistral-large", "cohere/command-r-plus", "perplexity/llama-3.1-sonar-large-128k-online",
                    "deepseek/deepseek-r1", "qwen/qwen-2.5-72b-instruct"
                ]
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/DurgasAI/DurgasAI",
                "X-Title": "DurgasAI"
            }
            
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Extract model IDs from the OpenRouter API response
            for model in data.get("data", []):
                if isinstance(model, dict) and "id" in model:
                    model_id = model["id"]
                    
                    # Filter for popular and well-supported models
                    if any(provider in model_id.lower() for provider in [
                        "openai", "anthropic", "meta-llama", "google", "mistralai", 
                        "cohere", "perplexity", "deepseek", "qwen", "microsoft"
                    ]):
                        # Exclude specific models that might not work well
                        if not any(exclude in model_id.lower() for exclude in [
                            "free", "beta", "preview", "instruct-beta", "moderation"
                        ]):
                            models.append(model_id)
            
            # If no models found, return default list
            if not models:
                log_warning("No models found in OpenRouter API response, using defaults", "config")
                return [
                    "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1-preview", "openai/o1-mini",
                    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku",
                    "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-70b-instruct",
                    "meta-llama/llama-3.1-8b-instruct", "google/gemini-pro-1.5", "google/gemini-flash-1.5",
                    "mistralai/mistral-large", "cohere/command-r-plus", "perplexity/llama-3.1-sonar-large-128k-online",
                    "deepseek/deepseek-r1", "qwen/qwen-2.5-72b-instruct"
                ]
            
            # Sort models by preference and provider
            model_priority = [
                "openai/gpt-4o", "openai/o1", "anthropic/claude-3.5", "meta-llama/llama-3.1-405b",
                "google/gemini-pro", "mistralai/mistral-large", "cohere/command-r-plus",
                "perplexity/llama", "deepseek/deepseek", "qwen/qwen", "microsoft"
            ]
            
            def sort_key(model):
                for i, priority in enumerate(model_priority):
                    if priority in model.lower():
                        return (i, model)
                return (len(model_priority), model)
            
            models.sort(key=sort_key)
            
            # Limit to top 50 models to avoid overwhelming the UI
            models = models[:50]
            
            log_info(f"Fetched {len(models)} OpenRouter models from API", "config")
            return models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching OpenRouter models: {e}", "config", e)
            return [
                "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1-preview", "openai/o1-mini",
                "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku",
                "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct", "google/gemini-pro-1.5", "google/gemini-flash-1.5",
                "mistralai/mistral-large", "cohere/command-r-plus", "perplexity/llama-3.1-sonar-large-128k-online",
                "deepseek/deepseek-r1", "qwen/qwen-2.5-72b-instruct"
            ]
        except Exception as e:
            log_error(f"Unexpected error fetching OpenRouter models: {e}", "config", e)
            return [
                "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1-preview", "openai/o1-mini",
                "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku",
                "meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct", "google/gemini-pro-1.5", "google/gemini-flash-1.5",
                "mistralai/mistral-large", "cohere/command-r-plus", "perplexity/llama-3.1-sonar-large-128k-online",
                "deepseek/deepseek-r1", "qwen/qwen-2.5-72b-instruct"
            ]
    
    def fetch_nvidia_models(self) -> List[str]:
        """Fetch available NVIDIA models from API"""
        try:
            api_key = self.get_api_key("nvidia")
            if not api_key:
                log_warning("No NVIDIA API key found, returning default models", "config")
                return [
                    "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-nemotron-51b-instruct",
                    "meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
                    "microsoft/phi-3-medium-128k-instruct", "microsoft/phi-3-mini-128k-instruct",
                    "mistralai/mixtral-8x7b-instruct-v0.1", "mistralai/mistral-7b-instruct-v0.3",
                    "google/gemma-2-9b-it", "google/gemma-2-2b-it", "nvidia/nemotron-4-340b-instruct"
                ]
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Extract model IDs from the NVIDIA API response
            for model in data.get("data", []):
                if isinstance(model, dict) and "id" in model:
                    model_id = model["id"]
                    
                    # Filter for chat/instruct models (exclude embedding, completion-only, etc.)
                    if any(keyword in model_id.lower() for keyword in [
                        "instruct", "chat", "nemotron", "llama", "phi", "mixtral", 
                        "mistral", "gemma", "qwen", "yi"
                    ]):
                        # Exclude specific model types that might not work well for chat
                        if not any(exclude in model_id.lower() for exclude in [
                            "embedding", "rerank", "completion", "base", "rag", "nv-embed"
                        ]):
                            models.append(model_id)
            
            # If no models found, return default list
            if not models:
                log_warning("No models found in NVIDIA API response, using defaults", "config")
                return [
                    "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-nemotron-51b-instruct",
                    "meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
                    "microsoft/phi-3-medium-128k-instruct", "microsoft/phi-3-mini-128k-instruct",
                    "mistralai/mixtral-8x7b-instruct-v0.1", "mistralai/mistral-7b-instruct-v0.3",
                    "google/gemma-2-9b-it", "google/gemma-2-2b-it", "nvidia/nemotron-4-340b-instruct"
                ]
            
            # Sort models by preference (NVIDIA models first, then by size/capability)
            model_priority = [
                "nvidia/nemotron-4-340b", "nvidia/llama-3.1-nemotron-70b", "nvidia/llama-3.1-nemotron-51b",
                "meta/llama-3.1-405b", "meta/llama-3.1-70b", "microsoft/phi-3-medium",
                "mistralai/mixtral-8x7b", "google/gemma-2-9b", "meta/llama-3.1-8b",
                "mistralai/mistral-7b", "microsoft/phi-3-mini", "google/gemma-2-2b"
            ]
            
            def sort_key(model):
                for i, priority in enumerate(model_priority):
                    if priority in model.lower():
                        return (i, model)
                return (len(model_priority), model)
            
            models.sort(key=sort_key)
            
            # Limit to top 30 models to avoid overwhelming the UI
            models = models[:30]
            
            log_info(f"Fetched {len(models)} NVIDIA models from API", "config")
            return models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching NVIDIA models: {e}", "config", e)
            return [
                "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-nemotron-51b-instruct",
                "meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
                "microsoft/phi-3-medium-128k-instruct", "microsoft/phi-3-mini-128k-instruct",
                "mistralai/mixtral-8x7b-instruct-v0.1", "mistralai/mistral-7b-instruct-v0.3",
                "google/gemma-2-9b-it", "google/gemma-2-2b-it", "nvidia/nemotron-4-340b-instruct"
            ]
        except Exception as e:
            log_error(f"Unexpected error fetching NVIDIA models: {e}", "config", e)
            return [
                "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-nemotron-51b-instruct",
                "meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
                "microsoft/phi-3-medium-128k-instruct", "microsoft/phi-3-mini-128k-instruct",
                "mistralai/mixtral-8x7b-instruct-v0.1", "mistralai/mistral-7b-instruct-v0.3",
                "google/gemma-2-9b-it", "google/gemma-2-2b-it", "nvidia/nemotron-4-340b-instruct"
            ]
    
    def fetch_huggingface_models(self) -> List[str]:
        """Fetch available HuggingFace models from router API"""
        try:
            api_key = self.get_api_key("huggingface")
            if not api_key:
                log_warning("No HuggingFace API key found, returning default models", "config")
                return [
                    "meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large",
                    "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-small", "google/flan-t5-large",
                    "google/flan-t5-xl", "bigscience/bloom-560m", "EleutherAI/gpt-neo-2.7B",
                    "EleutherAI/gpt-j-6B", "huggingface/CodeBERTa-small-v1"
                ]
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://router.huggingface.co/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Extract model IDs from the HuggingFace router API response
            for model in data.get("data", []):
                if isinstance(model, dict) and "id" in model:
                    model_id = model["id"]
                    
                    # Filter for conversational/chat models
                    if any(keyword in model_id.lower() for keyword in [
                        "chat", "instruct", "dialog", "conversation", "blenderbot", 
                        "dialo", "llama", "mistral", "phi", "gemma", "qwen", "flan"
                    ]):
                        # Exclude specific model types that might not work well for chat
                        if not any(exclude in model_id.lower() for exclude in [
                            "embedding", "feature-extraction", "fill-mask", "token-classification",
                            "question-answering", "summarization", "translation", "text2text-generation",
                            "base", "tokenizer", "pytorch_model"
                        ]):
                            models.append(model_id)
                    
                    # Also include popular general models that work for chat
                    elif any(popular in model_id.lower() for popular in [
                        "gpt-neo", "gpt-j", "bloom", "opt", "t5", "bart", "pegasus"
                    ]):
                        if "chat" in model_id.lower() or "instruct" in model_id.lower():
                            models.append(model_id)
            
            # If no models found, return default list
            if not models:
                log_warning("No models found in HuggingFace API response, using defaults", "config")
                return [
                    "meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large",
                    "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-small", "google/flan-t5-large",
                    "google/flan-t5-xl", "bigscience/bloom-560m", "EleutherAI/gpt-neo-2.7B",
                    "EleutherAI/gpt-j-6B", "huggingface/CodeBERTa-small-v1"
                ]
            
            # Sort models by preference and popularity
            model_priority = [
                "meta-llama/llama-2", "microsoft/dialogpt", "facebook/blenderbot", "google/flan-t5",
                "mistralai/mistral", "microsoft/phi", "google/gemma", "qwen/qwen",
                "bigscience/bloom", "eleutherai/gpt", "huggingface/"
            ]
            
            def sort_key(model):
                model_lower = model.lower()
                for i, priority in enumerate(model_priority):
                    if priority in model_lower:
                        return (i, model)
                return (len(model_priority), model)
            
            models.sort(key=sort_key)
            
            # Limit to top 25 models to avoid overwhelming the UI
            models = models[:25]
            
            log_info(f"Fetched {len(models)} HuggingFace models from router API", "config")
            return models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching HuggingFace models: {e}", "config", e)
            return [
                "meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-small", "google/flan-t5-large",
                "google/flan-t5-xl", "bigscience/bloom-560m", "EleutherAI/gpt-neo-2.7B",
                "EleutherAI/gpt-j-6B", "huggingface/CodeBERTa-small-v1"
            ]
        except Exception as e:
            log_error(f"Unexpected error fetching HuggingFace models: {e}", "config", e)
            return [
                "meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-small", "google/flan-t5-large",
                "google/flan-t5-xl", "bigscience/bloom-560m", "EleutherAI/gpt-neo-2.7B",
                "EleutherAI/gpt-j-6B", "huggingface/CodeBERTa-small-v1"
            ]
    
    def fetch_cohere_models(self) -> List[str]:
        """Fetch available Cohere models from API"""
        try:
            api_key = self.get_api_key("cohere")
            if not api_key:
                log_warning("No Cohere API key found, returning default models", "config")
                return [
                    "command-r-plus", "command-r", "command", "command-nightly",
                    "command-light", "command-light-nightly", "c4ai-aya-23-35b", "c4ai-aya-23-8b"
                ]
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get("https://api.cohere.ai/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            # Extract model names from the Cohere API response
            for model in data.get("models", []):
                if isinstance(model, dict) and "name" in model:
                    model_name = model["name"]
                    
                    # Filter for generative/chat models (exclude embedding and reranking models)
                    if any(keyword in model_name.lower() for keyword in [
                        "command", "aya", "chat", "instruct", "generate"
                    ]):
                        # Exclude embedding and reranking models
                        if not any(exclude in model_name.lower() for exclude in [
                            "embed", "rerank", "classify", "detect"
                        ]):
                            models.append(model_name)
            
            # If no models found, return default list
            if not models:
                log_warning("No models found in Cohere API response, using defaults", "config")
                return [
                    "command-r-plus", "command-r", "command", "command-nightly",
                    "command-light", "command-light-nightly", "c4ai-aya-23-35b", "c4ai-aya-23-8b"
                ]
            
            # Sort models by preference (Command R models first, then others)
            model_priority = [
                "command-r-plus", "command-r", "c4ai-aya-23-35b", "command",
                "c4ai-aya-23-8b", "command-light", "command-nightly", "command-light-nightly"
            ]
            
            def sort_key(model):
                model_lower = model.lower()
                for i, priority in enumerate(model_priority):
                    if priority in model_lower:
                        return (i, model)
                return (len(model_priority), model)
            
            models.sort(key=sort_key)
            
            # Limit to top 15 models to avoid overwhelming the UI
            models = models[:15]
            
            log_info(f"Fetched {len(models)} Cohere models from API", "config")
            return models
            
        except requests.exceptions.RequestException as e:
            log_error(f"Error fetching Cohere models: {e}", "config", e)
            return [
                "command-r-plus", "command-r", "command", "command-nightly",
                "command-light", "command-light-nightly", "c4ai-aya-23-35b", "c4ai-aya-23-8b"
            ]
        except Exception as e:
            log_error(f"Unexpected error fetching Cohere models: {e}", "config", e)
            return [
                "command-r-plus", "command-r", "command", "command-nightly",
                "command-light", "command-light-nightly", "c4ai-aya-23-35b", "c4ai-aya-23-8b"
            ]
    
    def get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider == "openai":
            return self.fetch_openai_models()
        elif provider == "anthropic":
            return self.fetch_anthropic_models()
        elif provider == "google":
            return self.fetch_google_models()
        elif provider == "deepseek":
            return ["deepseek-chat", "deepseek-coder"]
        elif provider == "groq":
            return self.fetch_groq_models()
        elif provider == "github":
            return self.fetch_github_models()
        elif provider == "openrouter":
            return self.fetch_openrouter_models()
        elif provider == "nvidia":
            return self.fetch_nvidia_models()
        elif provider == "cohere":
            return self.fetch_cohere_models()
        elif provider == "mistral":
            return ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]
        elif provider == "perplexity":
            return ["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"]
        elif provider == "huggingface":
            return self.fetch_huggingface_models()
        elif provider in ["tavily", "web_search", "exa", "serpapi"]:
            return ["search"]
        else:
            return ["default"]
    
    def cache_provider_models(self, provider: str, models: List[str]):
        """Cache models for a provider in config"""
        if "provider_models" not in self._config:
            self._config["provider_models"] = {}
        
        self._config["provider_models"][provider] = {
            "models": models,
            "cached_at": str(Path().cwd()),  # Simple timestamp placeholder
        }
        self._save_config()
    
    def get_cached_provider_models(self, provider: str) -> Optional[List[str]]:
        """Get cached models for a provider"""
        cached = self._config.get("provider_models", {}).get(provider)
        if cached and "models" in cached:
            return cached["models"]
        return None
