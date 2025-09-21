"""
Browser Open Tool for DurgasAI

This tool opens and reads content from web pages or documents using GPT-OSS browser tools.
It can access specific URLs or document IDs to extract detailed information.
"""

import json
import asyncio
from typing import Union, Optional

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


async def _browser_open_async(id: Union[int, str] = -1, cursor: int = -1, 
                             loc: int = -1, num_lines: int = -1, 
                             view_source: bool = False) -> str:
    """Async browser open function"""
    if not GPT_OSS_AVAILABLE:
        return json.dumps({
            "status": "error",
            "message": "Browser tools not available. Please install gpt-oss with browser dependencies.",
            "content": ""
        }, indent=2)
    
    try:
        payload = {
            'id': id, 'cursor': cursor, 'loc': loc, 
            'num_lines': num_lines, 'view_source': view_source
        }
        
        harmony_message = HarmonyMessage(
            author=Author(role=Role.USER),
            content=[TextContent(text=json.dumps(payload))],
            recipient='browser.open',
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
                "id": str(id),
                "content": result_text,
                "view_source": view_source,
                "cursor": cursor,
                "loc": loc,
                "num_lines": num_lines
            }, indent=2)
        else:
            return json.dumps({
                "status": "no_content",
                "message": f"Could not open or no content found: {id}",
                "id": str(id),
                "content": ""
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Browser open error: {str(e)}",
            "id": str(id),
            "error_details": str(e)
        }, indent=2)


def browser_open(id: str, cursor: int = -1, loc: int = -1, 
                num_lines: int = -1, view_source: bool = False) -> str:
    """
    Open and read content from a web page or document.
    
    This tool can access specific URLs or document IDs to extract detailed 
    information. It integrates with GPT-OSS browser capabilities to provide
    comprehensive content access.
    
    Args:
        id (str): Document ID or URL to open (use "-1" for latest)
        cursor (int): Starting position in the document (default: -1 for beginning)
        loc (int): Specific location to jump to (default: -1)
        num_lines (int): Number of lines to read (default: -1 for all)
        view_source (bool): Whether to view the HTML source code (default: false)
    
    Returns:
        content (str): Content of the opened page or document in JSON format
    """
    try:
        # Convert string id to appropriate type
        if id == "-1":
            doc_id = -1
        else:
            try:
                doc_id = int(id)
            except ValueError:
                doc_id = id  # Keep as string for URLs
        
        # Run async open
        result = asyncio.run(_browser_open_async(
            id=doc_id, 
            cursor=cursor, 
            loc=loc, 
            num_lines=num_lines, 
            view_source=view_source
        ))
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}",
            "id": str(id),
            "error_details": str(e)
        }
        return json.dumps(error_result, indent=2)


# For direct execution testing
if __name__ == "__main__":
    # Test the browser open tool
    test_id = "-1"  # Latest document
    print("Testing Browser Open Tool:")
    print("Document ID:", test_id)
    print("\nContent:")
    result = browser_open(test_id, num_lines=10)
    print(result)
