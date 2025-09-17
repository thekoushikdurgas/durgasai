"""
LLaMA-CPP-Agent Integration for DurgasAI

This module provides integration with llama-cpp-agent for advanced AI capabilities
including web search, content analysis, and structured output generation.
Based on the Google-Go implementation with DurgasAI enhancements.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timezone

from utils.logging_utils import log_info, log_error, log_warning
from utils.config_manager import ConfigManager

# Setup logging
logger = logging.getLogger(__name__)

class LlamaCppAgentSettings:
    """Settings management for LLaMA-CPP-Agent based on Google-Go implementation"""
    
    @staticmethod
    def get_context_by_model(model_name: str) -> int:
        """Get context length for different models"""
        model_context_limits = {
            "Mistral-7B-Instruct-v0.3-Q6_K.gguf": 32768,
            "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf": 32768,
            "Meta-Llama-3-8B-Instruct-Q6_K.gguf": 8192,
            "gemma-2-9b-it-Q8_0.gguf": 8192,
            "cognitivecomputations_Dolphin3.0-Mistral-24B-Q8_0.gguf": 8192,
            "gemma-2-27b-it-Q8_0.gguf": 8192,
            # Add more models as needed
            "default": 4096
        }
        return model_context_limits.get(model_name, model_context_limits["default"])
    
    @staticmethod
    def get_messages_formatter_type(model_name: str):
        """Get appropriate message formatter for the model"""
        try:
            from llama_cpp_agent import MessagesFormatterType
            
            model_name_lower = model_name.lower()
            
            if any(keyword in model_name_lower for keyword in ["meta", "llama", "aya"]):
                return MessagesFormatterType.LLAMA_3
            elif any(keyword in model_name_lower for keyword in ["mistral", "mixtral"]):
                return MessagesFormatterType.MISTRAL
            elif any(keyword in model_name_lower for keyword in ["einstein", "dolphin", "cognitivecomputations"]):
                return MessagesFormatterType.CHATML
            elif any(keyword in model_name_lower for keyword in ["gemma"]):
                return MessagesFormatterType.GEMMA_2
            elif "phi" in model_name_lower:
                return MessagesFormatterType.PHI_3
            else:
                return MessagesFormatterType.CHATML
                
        except ImportError:
            log_warning("llama-cpp-agent not available, using default formatter", "llama_cpp_integration")
            return "chatml"  # fallback

class LlamaCppAgentIntegration:
    """Main integration class for LLaMA-CPP-Agent functionality"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.settings = LlamaCppAgentSettings()
        self.llm = None
        self.llm_model = None
        self.provider = None
        self.models_directory = Path("models")
        self.models_directory.mkdir(exist_ok=True)
        
        # Check if llama-cpp-agent is available
        self.llama_cpp_available = self._check_llama_cpp_availability()
        
        if self.llama_cpp_available:
            log_info("LLaMA-CPP-Agent integration initialized", "llama_cpp_integration")
        else:
            log_warning("LLaMA-CPP-Agent not available, using fallback mode", "llama_cpp_integration")
    
    def _check_llama_cpp_availability(self) -> bool:
        """Check if llama-cpp-agent dependencies are available"""
        try:
            from llama_cpp import Llama
            from llama_cpp_agent.providers import LlamaCppPythonProvider
            from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
            from llama_cpp_agent.chat_history import BasicChatHistory
            from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
            from llama_cpp_agent.tools import WebSearchTool, GoogleWebSearchProvider
            return True
        except ImportError as e:
            log_warning(f"LLaMA-CPP-Agent dependencies not available: {e}", "llama_cpp_integration")
            return False
    
    def download_model(self, model_config: Dict[str, str]) -> bool:
        """Download a model from HuggingFace Hub"""
        try:
            from huggingface_hub import hf_hub_download
            
            repo_id = model_config.get("repo_id")
            filename = model_config.get("filename")
            
            if not repo_id or not filename:
                log_error("Invalid model configuration", "llama_cpp_integration")
                return False
            
            log_info(f"Downloading model {filename} from {repo_id}", "llama_cpp_integration")
            
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_directory)
            )
            
            log_info(f"Successfully downloaded {filename}", "llama_cpp_integration")
            return True
            
        except Exception as e:
            log_error(f"Failed to download model {model_config.get('filename', 'unknown')}", "llama_cpp_integration", e)
            return False
    
    def download_default_models(self) -> bool:
        """Download default models based on Google-Go configuration"""
        default_models = [
            {
                "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                "filename": "Mistral-7B-Instruct-v0.3-Q6_K.gguf"
            },
            {
                "repo_id": "bartowski/cognitivecomputations_Dolphin3.0-Mistral-24B-GGUF",
                "filename": "cognitivecomputations_Dolphin3.0-Mistral-24B-Q8_0.gguf"
            },
            {
                "repo_id": "bartowski/gemma-2-27b-it-GGUF",
                "filename": "gemma-2-27b-it-Q8_0.gguf"
            }
        ]
        
        success_count = 0
        for model_config in default_models:
            if self.download_model(model_config):
                success_count += 1
        
        log_info(f"Downloaded {success_count}/{len(default_models)} models", "llama_cpp_integration")
        return success_count > 0
    
    def initialize_llm(self, model_name: str) -> bool:
        """Initialize the LLaMA model"""
        if not self.llama_cpp_available:
            log_warning("Cannot initialize LLM: llama-cpp-agent not available", "llama_cpp_integration")
            return False
        
        try:
            from llama_cpp import Llama
            from llama_cpp_agent.providers import LlamaCppPythonProvider
            
            model_path = self.models_directory / model_name
            
            if not model_path.exists():
                log_error(f"Model file not found: {model_path}", "llama_cpp_integration")
                return False
            
            # Initialize LLaMA model
            if self.llm is None or self.llm_model != model_name:
                log_info(f"Initializing LLaMA model: {model_name}", "llama_cpp_integration")
                
                self.llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=81,  # Use GPU if available
                    n_batch=1024,
                    n_ctx=self.settings.get_context_by_model(model_name),
                    flash_attn=True
                )
                
                self.llm_model = model_name
                self.provider = LlamaCppPythonProvider(self.llm)
                
                log_info(f"Successfully initialized model: {model_name}", "llama_cpp_integration")
                return True
            
            return True
            
        except Exception as e:
            log_error(f"Failed to initialize LLM model {model_name}", "llama_cpp_integration", e)
            return False
    
    def create_web_search_agent(self, model_name: str, system_prompt: str = None) -> Optional[Any]:
        """Create a web search agent"""
        if not self.llama_cpp_available:
            return None
        
        try:
            from llama_cpp_agent import LlamaCppAgent
            from llama_cpp_agent.tools import WebSearchTool, GoogleWebSearchProvider
            from llama_cpp_agent.prompt_templates import web_search_system_prompt
            
            if not self.initialize_llm(model_name):
                return None
            
            chat_template = self.settings.get_messages_formatter_type(model_name)
            
            # Create web search tool
            search_tool = WebSearchTool(
                llm_provider=self.provider,
                web_search_provider=GoogleWebSearchProvider(),
                message_formatter_type=chat_template,
                max_tokens_search_results=12000,
                max_tokens_per_summary=2048,
            )
            
            # Create web search agent
            web_search_agent = LlamaCppAgent(
                self.provider,
                system_prompt=system_prompt or web_search_system_prompt,
                predefined_messages_formatter_type=chat_template,
                debug_output=True,
            )
            
            return {
                "agent": web_search_agent,
                "search_tool": search_tool,
                "chat_template": chat_template
            }
            
        except Exception as e:
            log_error("Failed to create web search agent", "llama_cpp_integration", e)
            return None
    
    def create_research_agent(self, model_name: str, system_prompt: str = None) -> Optional[Any]:
        """Create a research/answer agent"""
        if not self.llama_cpp_available:
            return None
        
        try:
            from llama_cpp_agent import LlamaCppAgent
            from llama_cpp_agent.prompt_templates import research_system_prompt
            
            if not self.initialize_llm(model_name):
                return None
            
            chat_template = self.settings.get_messages_formatter_type(model_name)
            
            # Create research agent
            research_agent = LlamaCppAgent(
                self.provider,
                system_prompt=system_prompt or research_system_prompt,
                predefined_messages_formatter_type=chat_template,
                debug_output=True,
            )
            
            return {
                "agent": research_agent,
                "chat_template": chat_template
            }
            
        except Exception as e:
            log_error("Failed to create research agent", "llama_cpp_integration", e)
            return None
    
    def perform_web_search_with_analysis(
        self,
        query: str,
        model_name: str = None,
        temperature: float = 0.45,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """
        Perform web search with AI analysis using llama-cpp-agent
        """
        # Default model if none specified
        if model_name is None:
            model_name = "Mistral-7B-Instruct-v0.3-Q6_K.gguf"
        try:
            if not self.llama_cpp_available:
                return self._fallback_search_analysis(query)
            
            from llama_cpp_agent.chat_history import BasicChatHistory
            from llama_cpp_agent.chat_history.messages import Roles
            from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
            
            # Create web search agent
            search_components = self.create_web_search_agent(model_name)
            if not search_components:
                return self._fallback_search_analysis(query)
            
            web_search_agent = search_components["agent"]
            search_tool = search_components["search_tool"]
            
            # Create research agent
            research_components = self.create_research_agent(model_name)
            if not research_components:
                return self._fallback_search_analysis(query)
            
            answer_agent = research_components["agent"]
            
            # Configure LLM settings
            settings = self.provider.get_provider_default_settings()
            settings.stream = False
            settings.temperature = temperature
            settings.top_k = top_k
            settings.top_p = top_p
            settings.max_tokens = max_tokens
            settings.repeat_penalty = repeat_penalty
            
            # Configure structured output for search
            output_settings = LlmStructuredOutputSettings.from_functions(
                [search_tool.get_tool()]
            )
            
            # Initialize chat history
            messages = BasicChatHistory()
            
            # Perform web search
            log_info(f"Performing web search for: {query}", "llama_cpp_integration")
            
            search_result = web_search_agent.get_chat_response(
                query,
                llm_sampling_settings=settings,
                structured_output_settings=output_settings,
                add_message_to_chat_history=False,
                add_response_to_chat_history=False,
                print_output=False,
            )
            
            # Generate comprehensive response
            settings.stream = False  # For now, disable streaming for simplicity
            
            research_prompt = f"""Write a detailed and complete research document that fulfills the following user request: '{query}', based on the information from the web below.

{search_result[0]["return_value"]}

Please provide a comprehensive analysis with proper citations and structured information."""
            
            response = answer_agent.get_chat_response(
                research_prompt,
                role=Roles.tool,
                llm_sampling_settings=settings,
                chat_history=messages,
                returns_streaming_generator=False,
                print_output=False,
            )
            
            # Extract citations (placeholder for now)
            citations = self._extract_citations_from_search_result(search_result)
            
            result = {
                "query": query,
                "response": response,
                "search_results": search_result,
                "citations": citations,
                "model_used": model_name,
                "settings": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success"
            }
            
            log_info(f"Successfully completed web search and analysis for: {query}", "llama_cpp_integration")
            return result
            
        except Exception as e:
            log_error(f"Failed to perform web search with analysis for: {query}", "llama_cpp_integration", e)
            return self._fallback_search_analysis(query, error=str(e))
    
    def _extract_citations_from_search_result(self, search_result: List[Dict]) -> List[str]:
        """Extract citations from search results"""
        citations = []
        try:
            if search_result and len(search_result) > 0:
                result_data = search_result[0].get("return_value", "")
                # Simple extraction - in real implementation, this would be more sophisticated
                import re
                urls = re.findall(r'https?://[^\s<>"{}|\\^`[\]]+', result_data)
                citations = list(set(urls))  # Remove duplicates
        except Exception as e:
            log_warning(f"Failed to extract citations: {e}", "llama_cpp_integration")
        
        return citations[:10]  # Limit to top 10 citations
    
    def _fallback_search_analysis(self, query: str, error: str = None) -> Dict[str, Any]:
        """Fallback method when llama-cpp-agent is not available"""
        log_info(f"Using fallback search analysis for: {query}", "llama_cpp_integration")
        
        fallback_response = f"""# Research Analysis: {query}

## Summary
This is a fallback response generated when the full llama-cpp-agent integration is not available.

## Key Points
- Query: {query}
- Status: Fallback mode
- Reason: {"Error: " + error if error else "LLaMA-CPP-Agent not available"}

## Recommendations
1. Install llama-cpp-agent dependencies
2. Download required models
3. Configure API keys for web search
4. Retry the query

## Note
For full functionality, please ensure all dependencies are installed and configured properly.
"""
        
        return {
            "query": query,
            "response": fallback_response,
            "search_results": [],
            "citations": [],
            "model_used": "fallback",
            "settings": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "fallback",
            "error": error
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models in the models directory"""
        if not self.models_directory.exists():
            return []
        
        model_files = []
        for file_path in self.models_directory.glob("*.gguf"):
            model_files.append(file_path.name)
        
        return sorted(model_files)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the llama-cpp-agent configuration"""
        validation = {
            "llama_cpp_available": self.llama_cpp_available,
            "models_directory_exists": self.models_directory.exists(),
            "available_models": self.get_available_models(),
            "huggingface_hub_available": False,
            "web_search_providers": {
                "google": False,
                "tavily": False,
                "serpapi": False
            }
        }
        
        # Check HuggingFace Hub
        try:
            import huggingface_hub
            validation["huggingface_hub_available"] = True
        except ImportError:
            pass
        
        # Check web search provider APIs
        api_keys = self.config_manager.get_all_api_keys()
        validation["web_search_providers"]["google"] = bool(api_keys.get("google"))
        validation["web_search_providers"]["tavily"] = bool(api_keys.get("tavily"))
        validation["web_search_providers"]["serpapi"] = bool(api_keys.get("serpapi"))
        
        return validation

# Utility functions for Google-Go style content extraction
def get_server_time() -> str:
    """Get current server time in UTC"""
    utc_time = datetime.now(timezone.utc)
    return utc_time.strftime("%Y-%m-%d %H:%M:%S")

def get_website_content_from_url(url: str) -> str:
    """
    Get website content from a URL using trafilatura
    (Google-Go style implementation)
    """
    try:
        from trafilatura import fetch_url, extract
        
        downloaded = fetch_url(url)
        if not downloaded:
            return f"Failed to download content from {url}"
        
        result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)
        
        if result:
            result_data = json.loads(result)
            return f'=========== Website Title: {result_data.get("title", "Unknown")} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result_data.get("raw_text", "No content extracted")}\n\n=========== Website Content End ===========\n\n'
        else:
            return f"No content could be extracted from {url}"
            
    except ImportError:
        log_warning("trafilatura not available for content extraction", "llama_cpp_integration")
        return f"Content extraction not available for {url}"
    except Exception as e:
        log_error(f"Error extracting content from {url}", "llama_cpp_integration", e)
        return f"An error occurred: {str(e)}"

# Citation sources model (Google-Go style)
class CitingSources:
    """Model for citing sources used in responses"""
    
    def __init__(self, sources: List[str]):
        self.sources = sources
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {"sources": self.sources}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
