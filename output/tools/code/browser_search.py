"""
Browser Search Tool for DurgasAI

This tool provides web search capabilities using GPT-OSS browser tools.
It integrates with the ExaBackend to perform advanced web research.
"""

import json
import asyncio
from typing import Optional

# Try to import GPT-OSS browser tools
try:
    from gpt_oss.tools.simple_browser import ExaBackend, SimpleBrowserTool
    from openai_harmony import Author, Role, TextContent
    from openai_harmony import Message as HarmonyMessage
    GPT_OSS_AVAILABLE = True
except ImportError:
    GPT_OSS_AVAILABLE = False

# Initialize browser tools if available
if GPT_OSS_AVAILABLE:
    try:
        _backend = ExaBackend(source='web')
        _browser_tool = SimpleBrowserTool(backend=_backend)
    except Exception:
        GPT_OSS_AVAILABLE = False


async def _browser_search_async(query: str, topn: int = 10, source: Optional[str] = None) -> str:
    """Async browser search function"""
    if not GPT_OSS_AVAILABLE:
        return json.dumps({
            "status": "error",
            "message": "Browser tools not available. Please install gpt-oss with browser dependencies.",
            "results": []
        }, indent=2)
    
    try:
        # Map to Harmony format
        harmony_message = HarmonyMessage(
            author=Author(role=Role.USER),
            content=[TextContent(text=json.dumps({'query': query, 'topn': topn}))],
            recipient='browser.search',
        )
        
        result_text = ''
        async for response in _browser_tool._process(harmony_message):
            if response.content:
                for content in response.content:
                    if isinstance(content, TextContent):
                        result_text += content.text
        
        if result_text:
            return json.dumps({
                "status": "success",
                "query": query,
                "results": result_text,
                "topn": topn
            }, indent=2)
        else:
            return json.dumps({
                "status": "no_results",
                "message": f"No results found for query: {query}",
                "query": query,
                "results": []
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Browser search error: {str(e)}",
            "query": query,
            "error_details": str(e)
        }, indent=2)


def browser_search(query: str, topn: int = 10, source: Optional[str] = None) -> str:
    """
    Search the web for information using advanced research capabilities.
    
    This tool leverages GPT-OSS browser tools to find relevant web content
    and return structured results with URLs, titles, and snippets.
    
    Args:
        query (str): Search query to find information on the web
        topn (int): Number of top results to return (default: 10)
        source (str): Specific source to search (optional)
    
    Returns:
        results (str): Search results with URLs, titles, and snippets in JSON format
    """
    try:
        # Run async search
        result = asyncio.run(_browser_search_async(query=query, topn=topn, source=source))
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}",
            "query": query,
            "error_details": str(e)
        }
        return json.dumps(error_result, indent=2)


# For direct execution testing
if __name__ == "__main__":
    # Test the browser search tool
    test_query = "What is artificial intelligence?"
    print("Testing Browser Search Tool:")
    print("Query:", test_query)
    print("\nResults:")
    result = browser_search(test_query, topn=5)
    print(result)
