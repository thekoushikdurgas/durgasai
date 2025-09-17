from typing import Dict, Any
import json

"""
Analyze uploaded files to extract information and content.
"""

def file_analyzer(uploaded_file: Dict[str, Any], analysis_type: str = None):
    """
    Analyze uploaded files to extract information and content.
    
    Args:
        uploaded_file: File to analyze (supports text, JSON, CSV, and image files) (uploaded file object with name, content, type, size)
        analysis_type: Type of analysis: basic, content, metadata
    
    Returns:
        file_info (object): Basic file information (name, size, type) (json format)
        content_preview (string): Preview of file content (first 500 characters) (plain_text format)
        analysis_report (string): Detailed analysis report based on file type (markdown format)
        file_stats (object): File statistics (line count, word count, etc.) (json format)
    """
    try:
        # Handle case when no file is uploaded
        if not uploaded_file:
            result = {
                "file_info": {"error": "No file uploaded"},
                "content_preview": "No file content available",
                "analysis_report": "# File Analysis Report\n\nNo file was uploaded for analysis.",
                "file_stats": {"error": "No file to analyze"}
            }
            return json.dumps(result, indent=2)
        
        # Process uploaded_file file
        if uploaded_file:
            file_name = uploaded_file.get('name', 'unknown')
            file_size = uploaded_file.get('size', 0)
            file_type = uploaded_file.get('type', 'unknown')
            file_content = uploaded_file.get('content', b'')
        
        # Extract basic file info
        file_info = {
            "name": file_name,
            "size": file_size,
            "type": file_type,
            "extension": file_name.split('.')[-1].lower() if '.' in file_name else "no_extension"
        }
        
        # Determine if file is text-based
        text_extensions = ['txt', 'csv', 'json', 'xml', 'html', 'py', 'js', 'css', 'md', 'yaml', 'yml']
        is_text_file = file_info["extension"] in text_extensions
        
        # Process content preview
        content_preview = ""
        file_stats = {}
        
        if is_text_file:
            try:
                # Try to decode as text
                text_content = file_content.decode('utf-8', errors='ignore')
                content_preview = text_content[:500]
                if len(text_content) > 500:
                    content_preview += "..."
                
                # Calculate text statistics
                lines = text_content.split('\n')
                words = text_content.split()
                file_stats = {
                    "line_count": len(lines),
                    "word_count": len(words),
                    "character_count": len(text_content),
                    "is_text_file": True
                }
                
            except Exception as e:
                content_preview = f"Error reading text content: {str(e)}"
                file_stats = {"error": "Could not process as text file", "is_text_file": False}
        else:
            content_preview = f"Binary file - cannot preview content. File type: {file_type}"
            file_stats = {
                "is_text_file": False,
                "binary_size": file_size
            }
        
        # Generate analysis report
        analysis_report = f"# File Analysis Report\n\n"
        analysis_report += f"## File Information\n"
        analysis_report += f"- **Name:** {file_name}\n"
        analysis_report += f"- **Size:** {file_size} bytes ({file_size/1024:.1f} KB)\n"
        analysis_report += f"- **Type:** {file_type}\n"
        analysis_report += f"- **Extension:** {file_info['extension']}\n\n"
        
        if analysis_type and analysis_type.lower() == 'content' and is_text_file:
            analysis_report += f"## Content Analysis\n"
            analysis_report += f"- **Lines:** {file_stats.get('line_count', 0)}\n"
            analysis_report += f"- **Words:** {file_stats.get('word_count', 0)}\n"
            analysis_report += f"- **Characters:** {file_stats.get('character_count', 0)}\n\n"
            analysis_report += f"### Content Preview\n```\n{content_preview}\n```\n"
        
        elif analysis_type and analysis_type.lower() == 'metadata':
            analysis_report += f"## Metadata Analysis\n"
            analysis_report += f"- **File Type:** {'Text-based' if is_text_file else 'Binary'}\n"
            analysis_report += f"- **Encoding:** UTF-8 compatible: {is_text_file}\n"
            
            # Special handling for specific file types
            if file_info["extension"] == 'json' and is_text_file:
                try:
                    text_content = file_content.decode('utf-8', errors='ignore')
                    json_data = json.loads(text_content)
                    analysis_report += f"- **JSON Structure:** Valid JSON with {len(json_data)} top-level items\n"
                except:
                    analysis_report += f"- **JSON Structure:** Invalid JSON format\n"
            
            elif file_info["extension"] == 'csv' and is_text_file:
                try:
                    text_content = file_content.decode('utf-8', errors='ignore')
                    lines = text_content.split('\n')
                    first_line = lines[0] if lines else ""
                    comma_count = first_line.count(',')
                    analysis_report += f"- **CSV Structure:** Estimated {comma_count + 1} columns\n"
                except:
                    analysis_report += f"- **CSV Structure:** Could not analyze CSV structure\n"
        
        else:
            analysis_report += f"## Basic Analysis\n"
            analysis_report += f"File successfully uploaded and analyzed. "
            if is_text_file:
                analysis_report += f"Text file with {file_stats.get('line_count', 0)} lines."
            else:
                analysis_report += f"Binary file - use specialized tools for detailed analysis."
        
        # Return structured JSON output
        result = {
            "file_info": file_info,
            "content_preview": content_preview,
            "analysis_report": analysis_report,
            "file_stats": file_stats
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        # Return error in expected format
        error_result = {
            "file_info": {"error": str(e)},
            "content_preview": f"Error analyzing file: {str(e)}",
            "analysis_report": f"# File Analysis Error\n\nError occurred during analysis: {str(e)}",
            "file_stats": {"error": str(e)}
        }
        return json.dumps(error_result, indent=2)
