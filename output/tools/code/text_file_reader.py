from typing import Dict, Any
import json

"""
Read and process text files.
"""

def text_file_reader(text_file: Dict[str, Any]):
    """
    Read and process text files.
    
    Args:
        text_file: Text file to read and process (uploaded file object with name, content, type, size)
    
    Returns:
        file_content (string): Content of the uploaded text file (plain_text format)
        file_name (string): Name of the uploaded file (plain_text format)
        file_size (integer): Size of the file in bytes
    """
    try:
        # Process text_file file
        if text_file:
            file_name = text_file.get('name', 'unknown')
            file_size = text_file.get('size', 0)
            file_type = text_file.get('type', 'unknown')
            file_content = text_file.get('content', b'')
            
            # Try to decode the file content as text
            try:
                decoded_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try other encodings
                try:
                    decoded_content = file_content.decode('latin-1')
                except:
                    decoded_content = "Error: Could not decode file as text"
            
            result = {
                "file_content": decoded_content,
                "file_name": file_name,
                "file_size": file_size
            }
        else:
            result = {
                "file_content": "No file uploaded",
                "file_name": "No file",
                "file_size": 0
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        # Return error in expected format
        error_result = {
            "file_content": f"Error reading file: {str(e)}",
            "file_name": "Error",
            "file_size": 0
        }
        return json.dumps(error_result, indent=2)
