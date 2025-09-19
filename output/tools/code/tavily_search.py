"""
Tavily Web Search Tool
Real implementation using Tavily API for web search functionality.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

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

def search_web_tavily(query: str, num_results: int = 5, search_depth: str = "basic", include_answer: bool = True, include_images: bool = False, include_raw_content: bool = False) -> str:
    """
    Search the web using Tavily API for real-time information.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-20, default: 5)
        search_depth: Search depth - "basic" or "advanced" (default: "basic")
        include_answer: Whether to include AI-generated answer (default: True)
        include_images: Whether to include images in results (default: False)
        include_raw_content: Whether to include raw content (default: False)
    
    Returns:
        search_results (array[object]): List of search result objects with title, URL, content, and score
        formatted_results (string): Human-readable formatted search results (markdown format)
        query_processed (string): The processed search query
        result_count (integer): Number of results returned
        answer (string): AI-generated answer (if include_answer is True)
        search_metadata (object): Search metadata including timestamp and API info
    """
    try:
        # Get API key
        api_key = get_tavily_api_key()
        if not api_key:
            raise ValueError("Tavily API key not found. Please configure it in settings.")
        
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        processed_query = query.strip()
        num_results = max(1, min(20, int(num_results)))  # Clamp between 1-20
        
        # Prepare request payload
        payload = {
            "api_key": api_key,
            "query": processed_query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_images": include_images,
            "include_raw_content": include_raw_content,
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
        answer = data.get('answer', '') if include_answer else ''
        
        # Format results
        formatted_results = f"# 🔍 Search Results for '{processed_query}'\n\n"
        
        if answer and include_answer:
            formatted_results += f"## 🤖 AI Answer\n\n{answer}\n\n---\n\n"
        
        if search_results:
            formatted_results += f"## 📋 Search Results ({len(search_results)} found)\n\n"
            
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No Title')
                url = result.get('url', '')
                content = result.get('content', '')
                score = result.get('score', 0)
                
                formatted_results += f"### {i}. {title}\n\n"
                formatted_results += f"**🔗 URL:** {url}\n\n"
                formatted_results += f"**📊 Relevance Score:** {score:.2f}\n\n"
                
                if content:
                    # Truncate content if too long
                    if len(content) > 300:
                        content = content[:300] + "..."
                    formatted_results += f"**📝 Content:** {content}\n\n"
                
                formatted_results += "---\n\n"
        else:
            formatted_results += "No results found for your query.\n\n"
        
        # Add search metadata
        search_metadata = {
            "timestamp": datetime.now().isoformat(),
            "api_provider": "Tavily",
            "search_depth": search_depth,
            "query_processed": processed_query,
            "results_requested": num_results,
            "results_returned": len(search_results)
        }
        
        # Prepare final result
        result = {
            "search_results": search_results,
            "formatted_results": formatted_results,
            "query_processed": processed_query,
            "result_count": len(search_results),
            "answer": answer if include_answer else "",
            "search_metadata": search_metadata
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "search_results": [],
            "formatted_results": f"# ❌ Search Error\n\nError searching for '{query}': {str(e)}\n\n*Please check your Tavily API key configuration in settings.*",
            "query_processed": query,
            "result_count": 0,
            "answer": "",
            "search_metadata": {
                "timestamp": datetime.now().isoformat(),
                "api_provider": "Tavily",
                "error": str(e)
            }
        }
        return json.dumps(error_result, indent=2)

def search_web(query: str, num_results: int = 5) -> str:
    """
    Simplified search function for backward compatibility.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
    
    Returns:
        JSON string with search results
    """
    return search_web_tavily(query, num_results, search_depth="basic", include_answer=True)

# For testing
if __name__ == "__main__":
    # Test the search function
    test_query = "latest AI developments 2024"
    result = search_web(test_query, 3)
    print(result)
