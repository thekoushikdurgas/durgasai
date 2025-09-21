"""
Generate a random number within a specified range.
"""

import random
import json

def generate_random_number(min_value: int = 1, max_value: int = 100) -> str:
    """
    Generate a random number within a specified range.
    
    Args:
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
    
    Returns:
        random_number (integer): The generated random number
        range_info (string): Information about the range used (plain_text format)
        generation_status (string): Status of the number generation (plain_text format)
    """
    try:
        # Convert to integers if they're not already
        min_val = int(min_value)
        max_val = int(max_value)
        
        if min_val > max_val:
            error_result = {
                "random_number": None,
                "range_info": f"Invalid range: {min_val} to {max_val}",
                "generation_status": "error - minimum value greater than maximum value"
            }
            return json.dumps(error_result, indent=2)
        
        random_num = random.randint(min_val, max_val)
        
        result = {
            "random_number": random_num,
            "range_info": f"Generated number in range {min_val} to {max_val} (inclusive)",
            "generation_status": "success"
        }
        
        return json.dumps(result, indent=2)
        
    except ValueError as e:
        error_result = {
            "random_number": None,
            "range_info": f"Invalid input values: {min_value}, {max_value}",
            "generation_status": f"error - invalid input: {str(e)}"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "random_number": None,
            "range_info": f"Range: {min_value} to {max_value}",
            "generation_status": f"error - {str(e)}"
        }
        return json.dumps(error_result, indent=2)
