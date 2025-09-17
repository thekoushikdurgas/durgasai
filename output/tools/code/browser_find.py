"""
Browser Find Tool for DurgasAI

This tool finds specific text patterns or content within opened documents.
It helps locate specific information within web pages or documents using GPT-OSS browser tools.
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


async def _browser_find_async(pattern: str, cursor: int = -1) -> str:
    """Async browser find function"""
    if not GPT_OSS_AVAILABLE:
        return json.dumps({
            "status": "error",
            "message": "Browser tools not available. Please install gpt-oss with browser dependencies.",
            "matches": []
        }, indent=2)
    
    try:
        payload = {'pattern': pattern, 'cursor': cursor}
        
        harmony_message = HarmonyMessage(
            author=Author(role=Role.USER),
            content=[TextContent(text=json.dumps(payload))],
            recipient='browser.find',
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
                "pattern": pattern,
                "cursor": cursor,
                "matches": result_text
            }, indent=2)
        else:
            return json.dumps({
                "status": "not_found",
                "message": f"Pattern not found: {pattern}",
                "pattern": pattern,
                "matches": []
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Browser find error: {str(e)}",
            "pattern": pattern,
            "error_details": str(e)
        }, indent=2)


def browser_find(pattern: str, cursor: int = -1) -> str:
    """
    Find specific text patterns or content within opened documents.
    
    This tool helps locate specific information within web pages or documents
    that have been previously opened with the browser_open tool.
    
    Args:
        pattern (str): Text pattern or keyword to find in the document
        cursor (int): Starting search position (default: -1 for beginning)
    
    Returns:
        matches (str): Found matches with context and locations in JSON format
    """
    try:
        # Run async find
        result = asyncio.run(_browser_find_async(pattern=pattern, cursor=cursor))
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}",
            "pattern": pattern,
            "error_details": str(e)
        }
        return json.dumps(error_result, indent=2)


# For direct execution testing
if __name__ == "__main__":
    # Test the browser find tool
    test_pattern = "artificial intelligence"
    print("Testing Browser Find Tool:")
    print("Pattern:", test_pattern)
    print("\nMatches:")
    result = browser_find(test_pattern)
    print(result)
