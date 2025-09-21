"""
Add two numbers together (tools.py pattern)
"""

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers together
    
    Args:
        a: The first number to add
        b: The second number to add
    
    Returns:
        result (integer): The sum of the two numbers
    """
    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
    return int(a) + int(b)
