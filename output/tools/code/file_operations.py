"""
Perform basic file operations (read, write, list).
"""

import os
import json
from pathlib import Path

def file_operations(operation: str, filename: str, content: str = "") -> str:
    """
    Perform basic file operations (read, write, list).
    
    Args:
        operation: Operation type ("read", "write", "list", "exists")
        filename: File path (relative to output directory)
        content: Content to write (for write operation)
    
    Returns:
        operation_result (string): Result of the file operation (plain_text format)
        file_content (string): Content of the file (for read operations) (plain_text format)
        file_list (array[string]): List of files (for list operations) (plain_text format)
        operation_status (string): Success or error status of the operation (plain_text format)
    """
    try:
        base_dir = Path("output/temp")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = base_dir / filename
        
        # Security check - ensure file is within base directory
        if not str(file_path.resolve()).startswith(str(base_dir.resolve())):
            error_result = {
                "operation_result": "Error: File path not allowed for security reasons",
                "file_content": "",
                "file_list": [],
                "operation_status": "error - security violation"
            }
            return json.dumps(error_result, indent=2)
        
        # Initialize result structure
        result = {
            "operation_result": "",
            "file_content": "",
            "file_list": [],
            "operation_status": "success"
        }
        
        if operation == "read":
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                result["operation_result"] = f"Successfully read {filename}"
                result["file_content"] = file_content
                result["operation_status"] = "success"
            else:
                result["operation_result"] = f"Error: File {filename} does not exist"
                result["operation_status"] = "error - file not found"
                
        elif operation == "write":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            result["operation_result"] = f"Successfully wrote {len(content)} characters to {filename}"
            result["operation_status"] = "success"
            
        elif operation == "list":
            files = [f.name for f in base_dir.iterdir() if f.is_file()]
            result["operation_result"] = f"Found {len(files)} files in directory"
            result["file_list"] = files
            result["operation_status"] = "success"
            
        elif operation == "exists":
            exists = file_path.exists()
            result["operation_result"] = f"File {filename} {'exists' if exists else 'does not exist'}"
            result["operation_status"] = "success"
            
        else:
            result["operation_result"] = f"Error: Unknown operation '{operation}'. Use: read, write, list, exists"
            result["operation_status"] = "error - invalid operation"
        
        return json.dumps(result, indent=2)
            
    except PermissionError:
        error_result = {
            "operation_result": f"Permission denied for {operation} operation on {filename}",
            "file_content": "",
            "file_list": [],
            "operation_status": "error - permission denied"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "operation_result": f"File operation error: {str(e)}",
            "file_content": "",
            "file_list": [],
            "operation_status": f"error - {str(e)}"
        }
        return json.dumps(error_result, indent=2)
