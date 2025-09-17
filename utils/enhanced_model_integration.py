"""
Enhanced Model Integration for DurgasAI

Provides unified interface for multiple AI model providers including
OpenAI, Anthropic, Google, local models, and llama-cpp-agent integration.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Generator
from datetime import datetime, timezone
from enum import Enum

from utils.logging_utils import log_info, log_error, log_warning
from utils.config_manager import ConfigManager

# Setup logging
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Enumeration of supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"
    COHERE = "cohere"
    NVIDIA = "nvidia"
    LLAMA_CPP = "llama_cpp"
    LOCAL = "local"

class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    WEB_SEARCH = "web_search"
    CODE_GENERATION = "code_generation"
    VISION = "vision"
    AUDIO = "audio"

class EnhancedModelIntegration:
    """Enhanced model integration supporting multiple providers"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.provider_clients = {}
        self.model_configs = self._load_model_configurations()
        self._initialize_providers()
    
    def _load_model_configurations(self) -> Dict[str, Any]:
        """Load model configurations for different providers"""
        return {
            ModelProvider.OPENAI.value: {
                "models": {
                    "gpt-4o": {
                        "name": "GPT-4o",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.FUNCTION_CALLING.value,
                            ModelCapability.STREAMING.value,
                            ModelCapability.VISION.value
                        ],
                        "context_length": 128000,
                        "cost_per_1k_tokens": {"input": 0.005, "output": 0.015}
                    },
                    "gpt-4o-mini": {
                        "name": "GPT-4o Mini",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.FUNCTION_CALLING.value,
                            ModelCapability.STREAMING.value
                        ],
                        "context_length": 128000,
                        "cost_per_1k_tokens": {"input": 0.00015, "output": 0.0006}
                    },
                    "gpt-3.5-turbo": {
                        "name": "GPT-3.5 Turbo",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.FUNCTION_CALLING.value,
                            ModelCapability.STREAMING.value
                        ],
                        "context_length": 16384,
                        "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015}
                    }
                }
            },
            ModelProvider.ANTHROPIC.value: {
                "models": {
                    "claude-3-sonnet": {
                        "name": "Claude 3 Sonnet",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.STREAMING.value,
                            ModelCapability.VISION.value
                        ],
                        "context_length": 200000,
                        "cost_per_1k_tokens": {"input": 0.003, "output": 0.015}
                    },
                    "claude-3-haiku": {
                        "name": "Claude 3 Haiku",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.STREAMING.value
                        ],
                        "context_length": 200000,
                        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125}
                    }
                }
            },
            ModelProvider.GOOGLE.value: {
                "models": {
                    "gemini-pro": {
                        "name": "Gemini Pro",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.FUNCTION_CALLING.value,
                            ModelCapability.STREAMING.value,
                            ModelCapability.VISION.value
                        ],
                        "context_length": 32768,
                        "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015}
                    },
                    "gemini-flash": {
                        "name": "Gemini Flash",
                        "max_tokens": 8192,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.STREAMING.value
                        ],
                        "context_length": 1048576,
                        "cost_per_1k_tokens": {"input": 0.000075, "output": 0.0003}
                    }
                }
            },
            ModelProvider.LLAMA_CPP.value: {
                "models": {
                    "Mistral-7B-Instruct-v0.3-Q6_K.gguf": {
                        "name": "Mistral 7B Instruct",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.STREAMING.value,
                            ModelCapability.WEB_SEARCH.value
                        ],
                        "context_length": 32768,
                        "cost_per_1k_tokens": {"input": 0.0, "output": 0.0}
                    },
                    "cognitivecomputations_Dolphin3.0-Mistral-24B-Q8_0.gguf": {
                        "name": "Dolphin 3.0 Mistral 24B",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.STREAMING.value,
                            ModelCapability.WEB_SEARCH.value,
                            ModelCapability.CODE_GENERATION.value
                        ],
                        "context_length": 8192,
                        "cost_per_1k_tokens": {"input": 0.0, "output": 0.0}
                    },
                    "gemma-2-27b-it-Q8_0.gguf": {
                        "name": "Gemma 2 27B Instruct",
                        "max_tokens": 4096,
                        "capabilities": [
                            ModelCapability.TEXT_GENERATION.value,
                            ModelCapability.CHAT.value,
                            ModelCapability.STREAMING.value
                        ],
                        "context_length": 8192,
                        "cost_per_1k_tokens": {"input": 0.0, "output": 0.0}
                    }
                }
            }
        }
    
    def _initialize_providers(self):
        """Initialize available providers based on API keys"""
        api_keys = self.config_manager.get_all_api_keys()
        
        for provider in ModelProvider:
            api_key = api_keys.get(provider.value)
            if api_key and api_key.strip():
                try:
                    self._initialize_provider_client(provider, api_key)
                except Exception as e:
                    log_warning(f"Failed to initialize {provider.value} client", "model_integration", e)
    
    def _initialize_provider_client(self, provider: ModelProvider, api_key: str):
        """Initialize a specific provider client"""
        try:
            if provider == ModelProvider.OPENAI:
                import openai
                self.provider_clients[provider.value] = openai.OpenAI(api_key=api_key)
                log_info(f"Initialized {provider.value} client", "model_integration")
            
            elif provider == ModelProvider.ANTHROPIC:
                import anthropic
                self.provider_clients[provider.value] = anthropic.Anthropic(api_key=api_key)
                log_info(f"Initialized {provider.value} client", "model_integration")
            
            elif provider == ModelProvider.GOOGLE:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.provider_clients[provider.value] = genai
                log_info(f"Initialized {provider.value} client", "model_integration")
            
            # Add more provider initializations as needed
            
        except ImportError as e:
            log_warning(f"Required package not installed for {provider.value}: {e}", "model_integration")
        except Exception as e:
            log_error(f"Failed to initialize {provider.value} client", "model_integration", e)
    
    def get_available_models(self, provider: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get available models, optionally filtered by provider"""
        available_models = {}
        
        for provider_name, config in self.model_configs.items():
            if provider and provider != provider_name:
                continue
            
            # Check if provider is available
            if provider_name in self.provider_clients or provider_name == ModelProvider.LLAMA_CPP.value:
                available_models[provider_name] = []
                
                for model_id, model_config in config["models"].items():
                    model_info = {
                        "id": model_id,
                        "name": model_config["name"],
                        "provider": provider_name,
                        "capabilities": model_config["capabilities"],
                        "max_tokens": model_config["max_tokens"],
                        "context_length": model_config["context_length"],
                        "cost": model_config.get("cost_per_1k_tokens", {"input": 0.0, "output": 0.0})
                    }
                    available_models[provider_name].append(model_info)
        
        return available_models
    
    def get_model_info(self, model_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        provider_config = self.model_configs.get(provider, {})
        model_config = provider_config.get("models", {}).get(model_id)
        
        if model_config:
            return {
                "id": model_id,
                "name": model_config["name"],
                "provider": provider,
                "capabilities": model_config["capabilities"],
                "max_tokens": model_config["max_tokens"],
                "context_length": model_config["context_length"],
                "cost": model_config.get("cost_per_1k_tokens", {"input": 0.0, "output": 0.0}),
                "available": provider in self.provider_clients or provider == ModelProvider.LLAMA_CPP.value
            }
        
        return None
    
    def generate_response(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response using specified model and provider
        """
        try:
            log_info(f"Generating response with {provider}/{model_id}", "model_integration")
            
            if provider == ModelProvider.OPENAI.value:
                return self._generate_openai_response(
                    prompt, model_id, temperature, max_tokens, system_prompt, streaming, **kwargs
                )
            elif provider == ModelProvider.ANTHROPIC.value:
                return self._generate_anthropic_response(
                    prompt, model_id, temperature, max_tokens, system_prompt, streaming, **kwargs
                )
            elif provider == ModelProvider.GOOGLE.value:
                return self._generate_google_response(
                    prompt, model_id, temperature, max_tokens, system_prompt, streaming, **kwargs
                )
            elif provider == ModelProvider.LLAMA_CPP.value:
                return self._generate_llama_cpp_response(
                    prompt, model_id, temperature, max_tokens, system_prompt, streaming, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        except Exception as e:
            log_error(f"Failed to generate response with {provider}/{model_id}", "model_integration", e)
            return f"Error generating response: {str(e)}"
    
    def _generate_openai_response(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        streaming: bool,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response using OpenAI API"""
        try:
            client = self.provider_clients.get(ModelProvider.OPENAI.value)
            if not client:
                raise ValueError("OpenAI client not initialized")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if streaming:
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
                
                def stream_generator():
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                
                return stream_generator()
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
        
        except Exception as e:
            log_error("OpenAI response generation failed", "model_integration", e)
            return f"OpenAI error: {str(e)}"
    
    def _generate_anthropic_response(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        streaming: bool,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response using Anthropic API"""
        try:
            client = self.provider_clients.get(ModelProvider.ANTHROPIC.value)
            if not client:
                raise ValueError("Anthropic client not initialized")
            
            if streaming:
                stream = client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **kwargs
                )
                
                def stream_generator():
                    for event in stream:
                        if event.type == "content_block_delta":
                            yield event.delta.text
                
                return stream_generator()
            else:
                response = client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.content[0].text
        
        except Exception as e:
            log_error("Anthropic response generation failed", "model_integration", e)
            return f"Anthropic error: {str(e)}"
    
    def _generate_google_response(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        streaming: bool,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response using Google Gemini API"""
        try:
            genai = self.provider_clients.get(ModelProvider.GOOGLE.value)
            if not genai:
                raise ValueError("Google client not initialized")
            
            model = genai.GenerativeModel(model_id)
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            if streaming:
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    stream=True
                )
                
                def stream_generator():
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                
                return stream_generator()
            else:
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                return response.text
        
        except Exception as e:
            log_error("Google response generation failed", "model_integration", e)
            return f"Google error: {str(e)}"
    
    def _generate_llama_cpp_response(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        streaming: bool,
        **_kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response using llama-cpp-agent"""
        try:
            # Use the llama-cpp-agent integration
            # from utils.llama_cpp_agent_integration import LlamaCppAgentIntegration
            # llama_integration = LlamaCppAgentIntegration(self.config_manager)
            
            # For now, return a placeholder response
            # In full implementation, this would use the llama-cpp-agent
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = f"""Response from {model_id}:

{full_prompt}

**Note:** This is a placeholder response. Full llama-cpp-agent integration requires:
1. Downloaded GGUF models
2. Proper model initialization
3. GPU/CPU configuration

Model: {model_id}
Temperature: {temperature}
Max Tokens: {max_tokens}
Timestamp: {datetime.now().isoformat()}"""
            
            if streaming:
                def stream_generator():
                    words = response.split()
                    for word in words:
                        yield word + " "
                
                return stream_generator()
            else:
                return response
        
        except Exception as e:
            log_error("LLaMA-CPP response generation failed", "model_integration", e)
            return f"LLaMA-CPP error: {str(e)}"
    
    def estimate_cost(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        response_length: Optional[int] = None
    ) -> Dict[str, float]:
        """Estimate cost for a request"""
        model_info = self.get_model_info(model_id, provider)
        if not model_info:
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        
        # Simple token estimation (rough approximation)
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        output_tokens = response_length or 500  # Default estimate
        
        cost_config = model_info.get("cost", {"input": 0.0, "output": 0.0})
        
        input_cost = (input_tokens / 1000) * cost_config["input"]
        output_cost = (output_tokens / 1000) * cost_config["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens)
        }
    
    def validate_model_availability(self, model_id: str, provider: str) -> Dict[str, Any]:
        """Validate if a model is available and properly configured"""
        validation = {
            "available": False,
            "provider_configured": False,
            "model_exists": False,
            "error": None
        }
        
        try:
            # Check if provider is configured
            validation["provider_configured"] = provider in self.provider_clients or provider == ModelProvider.LLAMA_CPP.value
            
            # Check if model exists in configuration
            model_info = self.get_model_info(model_id, provider)
            validation["model_exists"] = model_info is not None
            
            # Overall availability
            validation["available"] = validation["provider_configured"] and validation["model_exists"]
            
            if not validation["available"]:
                if not validation["provider_configured"]:
                    validation["error"] = f"Provider {provider} not configured or API key missing"
                elif not validation["model_exists"]:
                    validation["error"] = f"Model {model_id} not found in {provider} configuration"
        
        except Exception as e:
            validation["error"] = str(e)
            log_error(f"Model validation failed for {provider}/{model_id}", "model_integration", e)
        
        return validation
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        
        for provider in ModelProvider:
            provider_name = provider.value
            status[provider_name] = {
                "available": provider_name in self.provider_clients or provider_name == ModelProvider.LLAMA_CPP.value,
                "models_count": len(self.model_configs.get(provider_name, {}).get("models", {})),
                "capabilities": []
            }
            
            # Aggregate capabilities
            if provider_name in self.model_configs:
                all_capabilities = set()
                for model_config in self.model_configs[provider_name]["models"].values():
                    all_capabilities.update(model_config.get("capabilities", []))
                status[provider_name]["capabilities"] = list(all_capabilities)
        
        return status
