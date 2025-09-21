"""
Safely evaluate mathematical expressions.
"""

import math
import json

def calculate_math(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        result (number): The calculated result of the expression
        expression_validated (string): The validated expression that was evaluated (plain_text format)
        calculation_status (string): Status of the calculation (success/error) (plain_text format)
    """
    try:
        # Clean and validate the expression
        cleaned_expression = expression.strip()
        
        # Whitelist of allowed characters and functions
        allowed_chars = set("0123456789+-*/.() ")
        allowed_functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt,
            'abs': abs, 'round': round, 'pow': pow,
            'pi': math.pi, 'e': math.e
        }
        
        # Check for allowed characters
        if not all(c in allowed_chars or c.isalpha() for c in cleaned_expression):
            result_data = {
                "result": None,
                "expression_validated": cleaned_expression,
                "calculation_status": "error - expression contains disallowed characters"
            }
            return json.dumps(result_data, indent=2)
        
        # Prepare safe namespace
        safe_dict = {"__builtins__": {}}
        safe_dict.update(allowed_functions)
        
        # Evaluate expression
        calculation_result = eval(cleaned_expression, safe_dict)
        
        # Return successful result
        result_data = {
            "result": float(calculation_result) if isinstance(calculation_result, (int, float)) else calculation_result,
            "expression_validated": cleaned_expression,
            "calculation_status": "success"
        }
        
        return json.dumps(result_data, indent=2)
        
    except ZeroDivisionError:
        result_data = {
            "result": None,
            "expression_validated": expression.strip(),
            "calculation_status": "error - division by zero"
        }
        return json.dumps(result_data, indent=2)
    except SyntaxError as e:
        result_data = {
            "result": None,
            "expression_validated": expression.strip(),
            "calculation_status": f"error - invalid syntax: {str(e)}"
        }
        return json.dumps(result_data, indent=2)
    except Exception as e:
        result_data = {
            "result": None,
            "expression_validated": expression.strip(),
            "calculation_status": f"error - {str(e)}"
        }
        return json.dumps(result_data, indent=2)
