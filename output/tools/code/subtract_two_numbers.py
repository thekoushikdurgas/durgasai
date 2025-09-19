"""
Subtract two numbers (tools.py pattern)
"""

def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    
    Args:
        a: The first number
        b: The second number
    
    Returns:
        result (integer): The difference of the two numbers
    """
    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)
