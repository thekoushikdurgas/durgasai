"""
Common Utilities for AI Agent Dashboard

This module provides common utility functions used across the application,
including input sanitization, validation, and helper functions.

Features:
- Input sanitization and validation
- String manipulation utilities
- Data conversion helpers
- Security utilities
"""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


def sanitize_input(value: Any, max_length: int = 1000) -> Any:
    """
    Sanitize input data to prevent injection attacks and ensure data integrity.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length for string values
        
    Returns:
        Sanitized value
    """
    if value is None:
        return None
    
    if isinstance(value, str):
        # Remove potentially dangerous characters
        sanitized = html.escape(value)
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized
    
    elif isinstance(value, (int, float, bool)):
        return value
    
    elif isinstance(value, dict):
        return {sanitize_input(k): sanitize_input(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [sanitize_input(item) for item in value]
    
    else:
        # Convert to string and sanitize
        return sanitize_input(str(value), max_length)


def validate_json_string(json_str: str) -> bool:
    """
    Validate if a string is valid JSON.
    
    Args:
        json_str: String to validate
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback to default value.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_filename(filename: str) -> str:
    """
    Clean filename to remove dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for filesystem use
    """
    # Remove or replace dangerous characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    cleaned = cleaned.strip('. ')
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "unnamed_file"
    
    # Limit length
    if len(cleaned) > 255:
        name, ext = cleaned.rsplit('.', 1) if '.' in cleaned else (cleaned, '')
        cleaned = name[:255 - len(ext) - 1] + ('.' + ext if ext else '')
    
    return cleaned


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL format, False otherwise
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def generate_safe_id(prefix: str = "id", length: int = 8) -> str:
    """
    Generate a safe identifier.
    
    Args:
        prefix: Prefix for the ID
        length: Length of random part
        
    Returns:
        Safe identifier
    """
    import uuid
    import random
    import string
    
    # Generate random string
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    # Clean prefix
    safe_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
    
    return f"{safe_prefix}_{random_part}"


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def is_safe_code_identifier(identifier: str) -> bool:
    """
    Check if identifier is safe for code use.
    
    Args:
        identifier: Identifier to check
        
    Returns:
        True if safe, False otherwise
    """
    # Must start with letter or underscore
    if not re.match(r'^[a-zA-Z_]', identifier):
        return False
    
    # Can only contain letters, digits, and underscores
    if not re.match(r'^[a-zA-Z0-9_]+$', identifier):
        return False
    
    # Cannot be a Python keyword
    import keyword
    if keyword.iskeyword(identifier):
        return False
    
    return True


def extract_code_metrics(code: str) -> Dict[str, int]:
    """
    Extract basic code metrics from Python code.
    
    Args:
        code: Python code string
        
    Returns:
        Dictionary with code metrics
    """
    try:
        import ast
        
        tree = ast.parse(code)
        
        metrics = {
            'lines': len(code.splitlines()),
            'functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            'classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
            'imports': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
            'statements': len([node for node in ast.walk(tree) if isinstance(node, ast.stmt)])
        }
        
        return metrics
        
    except SyntaxError:
        return {
            'lines': len(code.splitlines()),
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'statements': 0
        }


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def escape_sql_like_pattern(pattern: str) -> str:
    """
    Escape SQL LIKE pattern special characters.
    
    Args:
        pattern: Pattern to escape
        
    Returns:
        Escaped pattern
    """
    # Escape SQL LIKE special characters
    escaped = pattern.replace('\\', '\\\\')
    escaped = escaped.replace('%', '\\%')
    escaped = escaped.replace('_', '\\_')
    
    return escaped


def create_hash(data: str, algorithm: str = 'sha256') -> str:
    """
    Create hash of data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hash string
    """
    import hashlib
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data.encode('utf-8'))
    return hash_obj.hexdigest()


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that required fields are present in data.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        List of missing field names
    """
    missing = []
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            missing.append(field)
    
    return missing
