"""
Google Agent Integration Module
Provides integration utilities for the Google Agent functionality
"""

import json
import logging
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from trafilatura import fetch_url, extract
from utils.logging_utils import log_info, log_error, log_warning
from utils.config_manager import ConfigManager

class GoogleAgentIntegration:
    """Integration class for Google Agent functionality"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.model: str = ""
        self.provider: str = ""
        self.logger = logging.getLogger(__name__)
    
    def _detect_provider_from_model(self, model_name: str) -> str:
        """
        Detect the provider from a model name
        Args:
            model_name (str): The model name (e.g., "gpt-4o", "claude-3-sonnet")
        Returns:
            str: The detected provider name
        """
        model_lower = model_name.lower()
        
        # OpenAI models
        if any(keyword in model_lower for keyword in ["gpt", "o1", "davinci", "curie", "babbage", "ada"]):
            return "openai"
        
        # Anthropic models
        elif any(keyword in model_lower for keyword in ["claude", "haiku", "sonnet", "opus"]):
            return "anthropic"
        
        # Google models
        elif any(keyword in model_lower for keyword in ["gemini", "palm", "bard"]):
            return "google"
        
        # Groq models
        elif any(keyword in model_lower for keyword in ["llama", "mixtral", "gemma"]) and "groq" in model_lower:
            return "groq"
        
        # Mistral models
        elif any(keyword in model_lower for keyword in ["mistral", "mixtral"]) and "groq" not in model_lower:
            return "mistral"
        
        # DeepSeek models
        elif "deepseek" in model_lower:
            return "deepseek"
        
        # Cohere models
        elif any(keyword in model_lower for keyword in ["command", "aya"]):
            return "cohere"
        
        # Perplexity models
        elif any(keyword in model_lower for keyword in ["pplx", "perplexity"]):
            return "perplexity"
        
        # Default fallback
        else:
            log_warning(f"Could not detect provider for model: {model_name}, defaulting to openai", "google_agent")
            return "openai"
    
    def setup_configuration(self, provider: str, model: str = None):
        """
        Setup configuration for Google Agent
        Args:
            provider (str): The LLM provider (e.g., "openai", "anthropic")
            model (str, optional): The specific model name. If not provided, provider is used as fallback
        """
        try:
            # Get API keys from config
            self.api_keys = self.config_manager.get_all_api_keys()
            
            # Set provider and model
            self.provider = provider
            self.model = model or provider  # Use model if provided, otherwise use provider as fallback
            
            # If we have a model name, try to detect the actual provider
            if model and model != provider:
                detected_provider = self._detect_provider_from_model(model)
                if detected_provider != provider:
                    log_warning(f"Model {model} suggests provider {detected_provider} but {provider} was specified", "google_agent")
                    # Use the explicitly specified provider, but log the discrepancy
            
            # Setup search providers with proper API key mapping
            self.search_providers = {
                "serpapi": self.api_keys.get("serpapi", ""),
                "tavily": self.api_keys.get("tavily", ""),
                "exa": self.api_keys.get("exa", ""),
                "ai_provider": self.api_keys.get(self.provider, "")  # Use provider for API key
            }
            
            log_info(f"Google Agent configured - Provider: {self.provider}, Model: {self.model}", "google_agent")
            
        except Exception as e:
            log_error("Failed to setup Google Agent configuration", "google_agent", e)
    
    def get_website_content_from_url(self, url: str) -> str:
        """
        Extract website content from a URL using trafilatura
        Args:
            url (str): URL to extract content from
        Returns:
            str: Extracted content with metadata
        """
        try:
            
            log_info(f"Extracting content from URL: {url}", "google_agent")
            
            # Download the webpage
            downloaded = fetch_url(url)
            if not downloaded:
                return f"Failed to download content from {url}"
            
            # Extract content
            result = extract(
                downloaded, 
                include_formatting=True, 
                include_links=True, 
                output_format='json', 
                url=url
            )
            
            if result:
                result_data = json.loads(result)
                formatted_content = f"""
=========== Website Title: {result_data.get("title", "Unknown")} ===========

=========== Website URL: {url} ===========

=========== Website Content ===========

{result_data.get("raw_text", "No content extracted")}

=========== Website Content End ===========
"""
                return formatted_content
            else:
                return f"No content could be extracted from {url}"
                
        except ImportError:
            log_warning("trafilatura not available, using basic content extraction", "google_agent")
            return self._basic_content_extraction(url)
        except Exception as e:
            log_error(f"Error extracting content from {url}", "google_agent", e)
            return f"An error occurred while extracting content: {str(e)}"
    
    def _basic_content_extraction(self, url: str) -> str:
        """
        Basic content extraction fallback when trafilatura is not available
        """
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Basic HTML parsing (very simple fallback)
            content = response.text
            return f"""
=========== Website URL: {url} ===========

=========== Basic Content ===========

{content[:2000]}... (truncated)

=========== Content End ===========
"""
        except Exception as e:
            log_error(f"Basic content extraction failed for {url}", "google_agent", e)
            return f"Failed to extract content from {url}: {str(e)}"
    
    def search_web_serpapi(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search web using SerpAPI
        """
        try:
            serpapi_key = self.search_providers.get("serpapi")
            if not serpapi_key:
                log_warning("SerpAPI key not configured", "google_agent")
                return []
            
            import serpapi
            
            search = serpapi.GoogleSearch({
                "q": query,
                "api_key": serpapi_key,
                "num": num_results
            })
            
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            
            formatted_results = []
            for result in organic_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": result.get("displayed_link", ""),
                    "position": result.get("position", 0)
                })
            
            log_info(f"SerpAPI search returned {len(formatted_results)} results", "google_agent")
            return formatted_results
            
        except ImportError:
            log_warning("serpapi package not available", "google_agent")
            return []
        except Exception as e:
            log_error("SerpAPI search failed", "google_agent", e)
            return []
    
    def search_web_tavily(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search web using Tavily API
        """
        try:
            tavily_key = self.search_providers.get("tavily")
            if not tavily_key:
                log_warning("Tavily API key not configured", "google_agent")
                return []
            
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=tavily_key)
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results
            )
            
            formatted_results = []
            for i, result in enumerate(results.get("results", [])):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "source": result.get("url", "").split("/")[2] if result.get("url") else "",
                    "position": i + 1,
                    "score": result.get("score", 0)
                })
            
            log_info(f"Tavily search returned {len(formatted_results)} results", "google_agent")
            return formatted_results
            
        except ImportError:
            log_warning("tavily package not available", "google_agent")
            return []
        except Exception as e:
            log_error("Tavily search failed", "google_agent", e)
            return []
    
    def search_web_fallback(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fallback search method when APIs are not available
        """
        log_info(f"Using fallback search for query: {query}", "google_agent")
        
        # Generate sample results for demonstration
        sample_results = []
        for i in range(min(num_results, 5)):
            sample_results.append({
                "title": f"Sample Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a sample search result #{i+1} for the query '{query}'. In a real implementation with API keys configured, this would be actual web search results.",
                "source": "example.com",
                "position": i + 1
            })
        
        return sample_results
    
    def search_web(self, query: str, num_results: int = 10, provider: str = "auto") -> List[Dict[str, Any]]:
        """
        Main web search method that tries different providers
        """
        log_info(f"Searching web for: {query} (provider: {provider})", "google_agent")
        
        results = []
        
        if provider == "auto" or provider == "tavily":
            results = self.search_web_tavily(query, num_results)
            if results:
                return results
        
        if provider == "auto" or provider == "serpapi":
            results = self.search_web_serpapi(query, num_results)
            if results:
                return results
        
        # Fallback to sample results
        log_info("Using fallback search method", "google_agent")
        return self.search_web_fallback(query, num_results)
    
    def format_search_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for use as context in AI responses
        """
        if not results:
            return "No search results found."
        
        formatted_context = "## Web Search Results:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_context += f"### Result {i}: {result.get('title', 'Unknown Title')}\n"
            formatted_context += f"**URL:** {result.get('url', 'Unknown URL')}\n"
            formatted_context += f"**Source:** {result.get('source', 'Unknown Source')}\n"
            formatted_context += f"**Content:** {result.get('snippet', 'No content available')}\n\n"
        
        return formatted_context
    
    def extract_citations_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract citation URLs from search results
        """
        citations = []
        for result in results:
            url = result.get('url')
            if url and url not in citations:
                citations.append(url)
        
        return citations
    
    def get_server_time(self) -> str:
        """Get current server time in UTC"""
        utc_time = datetime.now(timezone.utc)
        return utc_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate Google Agent configuration
        """
        validation_results = {
            "trafilatura_available": False,
            "serpapi_configured": False,
            "tavily_configured": False,
            "ai_provider_configured": False,
            "basic_functionality": True
        }
        
        # Check trafilatura
        try:
            validation_results["trafilatura_available"] = True
        except ImportError:
            pass
        
        # Check API keys
        validation_results["serpapi_configured"] = bool(self.search_providers.get("serpapi"))
        validation_results["tavily_configured"] = bool(self.search_providers.get("tavily"))
        validation_results["ai_provider_configured"] = bool(self.search_providers.get("ai_provider"))
        
        log_info(f"Configuration validation: {validation_results}", "google_agent")
        return validation_results

class GoogleAgentModelInterface:
    """Interface for integrating with different language models"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def generate_response(
        self, 
        query: str, 
        search_context: str, 
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate response using the specified model
        This is a placeholder for actual model integration
        """
        try:
            # This would integrate with your existing model service
            # from utils.model_service import get_model_service
            # model_service = get_model_service()
            
            # This would use your existing model service to generate the response
            # For now, return a formatted response
            response = f"""Based on the web search results for "{query}", here's what I found:

{search_context}

**Summary:**
This is a demonstration of the Google Agent integration. In the full implementation, this would:
1. Use the selected language model ({model}) to generate a comprehensive response
2. Analyze and synthesize information from multiple web sources
3. Provide accurate citations and source attributions
4. Generate contextually relevant and helpful answers

**Model Configuration:**
- Provider: {provider}
- Model: {model}
- Temperature: {temperature}
- Max Tokens: {max_tokens}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Note:** This is a framework demonstration. Full functionality requires proper API configuration and model integration."""

            return response
            
        except Exception as e:
            log_error("Error generating model response", "google_agent", e)
            return f"An error occurred while generating the response: {str(e)}"
