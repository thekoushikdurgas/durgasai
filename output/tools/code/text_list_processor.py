from typing import List
import json

"""
Process a list of text strings with various operations.
"""

def text_list_processor(text_list: List[str], operation: str):
    """
    Process a list of text strings with various operations.
    
    Args:
        text_list: List of text strings to process (list of string items)
        operation: Operation to perform: uppercase, lowercase, reverse, sort, unique
    
    Returns:
        processed_items (array[string]): List of processed text items (plain_text format)
        operation_summary (string): Summary of the operation performed (plain_text format)
        item_count (integer): Number of items processed
    """
    try:
        # Initialize default values
        processed_items = []
        operation_summary = ""
        item_count = len(text_list) if text_list else 0
        
        # Process text_list
        if text_list:
            if operation.lower() == "uppercase":
                processed_items = [item.strip().upper() for item in text_list]
                operation_summary = f"Converted {len(text_list)} items to uppercase"
            elif operation.lower() == "lowercase":
                processed_items = [item.strip().lower() for item in text_list]
                operation_summary = f"Converted {len(text_list)} items to lowercase"
            elif operation.lower() == "reverse":
                processed_items = [item.strip()[::-1] for item in text_list]
                operation_summary = f"Reversed {len(text_list)} text items"
            elif operation.lower() == "sort":
                processed_items = sorted([item.strip() for item in text_list])
                operation_summary = f"Sorted {len(text_list)} items alphabetically"
            elif operation.lower() == "unique":
                processed_items = list(set([item.strip() for item in text_list]))
                operation_summary = f"Filtered to {len(processed_items)} unique items from {len(text_list)} total items"
                item_count = len(processed_items)
            else:
                processed_items = text_list
                operation_summary = f"Unknown operation '{operation}'. Available: uppercase, lowercase, reverse, sort, unique"
        else:
            operation_summary = "No text items provided for processing"
        
        # Return structured JSON output
        result = {
            "processed_items": processed_items,
            "operation_summary": operation_summary,
            "item_count": item_count
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        # Return error in expected format
        error_result = {
            "processed_items": [],
            "operation_summary": f"Error processing text list: {str(e)}",
            "item_count": 0
        }
        return json.dumps(error_result, indent=2)
