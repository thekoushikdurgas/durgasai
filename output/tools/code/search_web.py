"""
Search the web for information using Tavily API.

This tool provides web search capabilities with comprehensive logging and error handling:
- Real-time web search using Tavily API
- Fallback to mock results when API is unavailable
- Comprehensive error handling and recovery
- Performance monitoring and logging
- Input validation and sanitization
- Structured result formatting

Features:
- Configurable number of results (1-10)
- API key management from config
- Markdown formatted output
- JSON structured results
- Error recovery with mock data
- Performance timing and logging

Dependencies:
- requests: For HTTP API calls
- Tavily API key: For real web search (optional)
"""

import json
import os
import sys
from typing import Optional
import requests
from datetime import datetime
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# Import logging utilities
try:
    from utils.logger import debug, info, warning, error, log_tool_execution
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    
    # Fallback logging functions
    def debug(msg, component="search_web", **kwargs): print(f"DEBUG: {msg}")
    def info(msg, component="search_web", **kwargs): print(f"INFO: {msg}")
    def warning(msg, component="search_web", **kwargs): print(f"WARNING: {msg}")
    def error(msg, component="search_web", **kwargs): print(f"ERROR: {msg}")
    def log_tool_execution(tool_name, inputs, outputs=None, error=None): pass

def get_tavily_api_key() -> Optional[str]:
    """Get Tavily API key from environment or config"""
    # Try environment variable first
    api_key = os.getenv('TAVILY_API_KEY')
    if api_key:
        return api_key
    
    # Try to get from config file
    try:
        # Add the parent directory to the path to import utils
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(parent_dir)
        
        from utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        api_key = config_manager.get_api_key('tavily')
        
        if api_key and api_key.strip():
            return api_key.strip()
    except Exception as e:
        print(f"Warning: Could not load Tavily API key from config: {e}")
    
    return None

def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web for information using Tavily API with comprehensive logging.
    
    This function provides intelligent web search capabilities:
    1. Validates and sanitizes input parameters
    2. Attempts real API search with Tavily
    3. Falls back to mock results if API unavailable
    4. Logs all operations for debugging
    5. Returns structured results in multiple formats
    
    Args:
        query (str): Search query string
        num_results (int): Number of results to return (1-10)
    
    Returns:
        str: JSON string containing:
            - search_results (array[object]): List of search result objects with title, URL, and snippet
            - formatted_results (string): Human-readable formatted search results (markdown format)
            - query_processed (string): The processed search query (plain_text format)
            - result_count (integer): Number of results returned
    """
    start_time = time.time()
    
    info(f"Starting web search", "tools",
        query=query[:100] + "..." if len(query) > 100 else query,
        num_results=num_results,
        tool="search_web")
    
    try:
        # Step 1: Input validation and sanitization
        debug("Validating search inputs", "tools")
        
        if not query or not query.strip():
            error_msg = "Search query cannot be empty"
            warning(error_msg, "tools")
            raise ValueError(error_msg)
        
        processed_query = query.strip()
        num_results = max(1, min(10, int(num_results)))  # Clamp between 1-10
        
        debug(f"Input validation completed", "tools",
            original_query_length=len(query),
            processed_query_length=len(processed_query),
            requested_results=num_results)
        
        # Step 2: Attempt to get Tavily API key
        debug("Checking for Tavily API key", "tools")
        api_key = get_tavily_api_key()
        
        if not api_key:
            warning("No Tavily API key found, using mock search", "tools")
            log_tool_execution("search_web", {"query": query, "num_results": num_results}, 
                             {"status": "fallback_mock", "reason": "no_api_key"})
            return _mock_search(query, num_results)
        
        info("Tavily API key found, proceeding with real search", "tools")
        
        # Prepare request payload
        payload = {
            "api_key": api_key,
            "query": processed_query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": num_results
        }
        
        # Make API request
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # Check response status
        if response.status_code != 200:
            error_msg = f"Tavily API error: {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f" - {error_data['error']}"
            except:
                error_msg += f" - {response.text}"
            raise Exception(error_msg)
        
        # Parse response
        data = response.json()
        
        # Extract results
        search_results = data.get('results', [])
        answer = data.get('answer', '')
        
        # Format results for backward compatibility
        formatted_results = f"# 🔍 Search Results for '{processed_query}'\n\n"
        
        if answer:
            formatted_results += f"## 🤖 AI Answer\n\n{answer}\n\n---\n\n"
        
        if search_results:
            formatted_results += f"## 📋 Search Results ({len(search_results)} found)\n\n"
            
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No Title')
                url = result.get('url', '')
                content = result.get('content', '')
                
                formatted_results += f"### {i}. {title}\n\n"
                formatted_results += f"**🔗 URL:** {url}\n\n"
                
                if content:
                    # Truncate content if too long
                    if len(content) > 300:
                        content = content[:300] + "..."
                    formatted_results += f"**📝 Content:** {content}\n\n"
                
                formatted_results += "---\n\n"
        else:
            formatted_results += "No results found for your query.\n\n"
        
        # Convert to backward compatible format
        legacy_results = []
        for result in search_results:
            legacy_results.append({
                "title": result.get('title', 'No Title'),
                "url": result.get('url', ''),
                "snippet": result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
            })
        
        result = {
            "search_results": legacy_results,
            "formatted_results": formatted_results,
            "query_processed": processed_query,
            "result_count": len(legacy_results)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        # Fallback to mock implementation on error
        return _mock_search(query, num_results, str(e))

def _mock_search(query: str, num_results: int, error_msg: str = None) -> str:
    """Fallback mock search implementation"""
    try:
        processed_query = query.strip()
        
        mock_results = [
            {
                "title": f"Result about {processed_query}",
                "url": f"https://example.com/search?q={processed_query.replace(' ', '+')}",
                "snippet": f"This is a mock search result for '{processed_query}'. Real implementation would use Tavily API."
            },
            {
                "title": f"More information on {processed_query}",
                "url": f"https://wikipedia.org/search?q={processed_query.replace(' ', '+')}",
                "snippet": f"Additional mock information about '{processed_query}'. Configure Tavily API key for real search results."
            },
            {
                "title": f"{processed_query} - Latest Updates",
                "url": f"https://news.example.com/{processed_query.replace(' ', '-')}",
                "snippet": f"Latest news and updates about '{processed_query}'. Enable Tavily API for real-time information."
            }
        ]
        
        num_results = min(max(1, num_results), len(mock_results))
        search_results = mock_results[:num_results]
        
        # Generate markdown formatted results
        formatted_results = f"# Search Results for '{processed_query}'\n\n"
        
        if error_msg:
            formatted_results += f"⚠️ **Note:** {error_msg}\n\n"
            formatted_results += "Showing mock results below. Configure Tavily API key in settings for real search.\n\n"
        
        for i, result in enumerate(search_results, 1):
            formatted_results += f"## {i}. {result['title']}\n\n"
            formatted_results += f"**URL:** {result['url']}\n\n"
            formatted_results += f"{result['snippet']}\n\n"
            formatted_results += "---\n\n"
        
        formatted_results += "*Note: These are mock search results. Configure Tavily API key for real web search.*"
        
        result = {
            "search_results": search_results,
            "formatted_results": formatted_results,
            "query_processed": processed_query,
            "result_count": len(search_results)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "search_results": [],
            "formatted_results": f"# Search Error\n\nError searching for '{query}': {str(e)}",
            "query_processed": query,
            "result_count": 0
        }
        return json.dumps(error_result, indent=2)
