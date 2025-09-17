"""
Search the web for information using Tavily API.
"""

import json
import os
import sys
from typing import Optional
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

def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web for information using Tavily API.
    
    Args:
        query: Search query
        num_results: Number of results to return (1-10)
    
    Returns:
        search_results (array[object]): List of search result objects with title, URL, and snippet (json format)
        formatted_results (string): Human-readable formatted search results (markdown format)
        query_processed (string): The processed search query (plain_text format)
        result_count (integer): Number of results returned
    """
    try:
        # Get Tavily API key
        api_key = get_tavily_api_key()
        if not api_key:
            # Fallback to mock implementation if no API key
            return _mock_search(query, num_results)
        
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        processed_query = query.strip()
        num_results = max(1, min(10, int(num_results)))  # Clamp between 1-10
        
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
