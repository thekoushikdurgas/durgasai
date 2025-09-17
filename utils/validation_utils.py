"""
Validation and Error Handling Utilities for DurgasAI

This module provides comprehensive validation and error handling utilities
for working with the enhanced model classes throughout the application.

Features:
- Model validation with detailed error reporting
- Data sanitization and security validation
- Error handling and recovery mechanisms
- Validation caching and performance optimization
- Custom validation rules and constraints
"""

import re
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError, Field
from models import (
    ChatMessage, ChatResponse, ChatSession, ChatHistory,
    LogEntry, LogLevel, LogAnalysisResult, LogStatistics,
    APIResponse, UserProfile, FileMetadata, NotificationData,
    ToolDefinition, ToolExecution, ToolSecurityReport,
    WorkflowDefinition, WorkflowExecution, WorkflowStep
)

T = TypeVar('T', bound=BaseModel)


class ValidationError(Exception):
    """Custom validation error with detailed information"""
    def __init__(self, message: str, field: str = None, value: Any = None, 
                 error_type: str = "validation_error", suggestions: List[str] = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.error_type = error_type
        self.suggestions = suggestions or []


class SecurityValidationError(ValidationError):
    """Security-related validation error"""
    def __init__(self, message: str, field: str = None, value: Any = None, 
                 security_level: str = "medium", recommendations: List[str] = None):
        super().__init__(message, field, value, "security_error", recommendations)
        self.security_level = security_level


class DataSanitizationError(ValidationError):
    """Data sanitization error"""
    def __init__(self, message: str, field: str = None, value: Any = None, 
                 sanitized_value: Any = None):
        super().__init__(message, field, value, "sanitization_error")
        self.sanitized_value = sanitized_value


class ValidationResult:
    """Result of a validation operation"""
    def __init__(self, is_valid: bool, errors: List[ValidationError] = None, 
                 warnings: List[str] = None, sanitized_data: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.sanitized_data = sanitized_data or {}
    
    def add_error(self, error: ValidationError = None):
        """Add a validation error"""
        if error is not None:
            self.errors.append(error)
            self.is_valid = False
    
    def add_warning(self, warning: str = None):
        """Add a validation warning"""
        if warning is not None:
            self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors"""
        return {
            "total_errors": len(self.errors),
            "error_types": list(set(error.error_type for error in self.errors)),
            "fields_with_errors": list(set(error.field for error in self.errors if error.field)),
            "errors": [
                {
                    "field": error.field,
                    "message": str(error),
                    "type": error.error_type,
                    "value": error.value,
                    "suggestions": error.suggestions
                }
                for error in self.errors
            ]
        }


class ValidationRule:
    """Base class for validation rules"""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def validate(self, value: Any, field: str = None, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate a value against this rule"""
        raise NotImplementedError


class RequiredFieldRule(ValidationRule):
    """Rule for required fields"""
    def __init__(self):
        super().__init__("required", "Field is required")
    
    def validate(self, value: Any, field: str = None, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)
        
        if value is None or (isinstance(value, str) and not value.strip()):
            result.add_error(ValidationError(
                f"Field '{field}' is required",
                field=field,
                value=value,
                suggestions=[f"Provide a value for {field}"]
            ))
        
        return result


class StringLengthRule(ValidationRule):
    """Rule for string length validation"""
    def __init__(self, min_length: int = 0, max_length: int = None):
        super().__init__("string_length", f"String length between {min_length} and {max_length or 'unlimited'}")
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any, field: str = None, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            return result
        
        if len(value) < self.min_length:
            result.add_error(ValidationError(
                f"Field '{field}' must be at least {self.min_length} characters long",
                field=field,
                value=value,
                suggestions=[f"Add at least {self.min_length - len(value)} more characters"]
            ))
        
        if self.max_length and len(value) > self.max_length:
            result.add_error(ValidationError(
                f"Field '{field}' must be no more than {self.max_length} characters long",
                field=field,
                value=value,
                suggestions=[f"Remove {len(value) - self.max_length} characters"]
            ))
        
        return result


class EmailFormatRule(ValidationRule):
    """Rule for email format validation"""
    def __init__(self):
        super().__init__("email_format", "Valid email format")
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def validate(self, value: Any, field: str = None, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            return result
        
        if not self.email_pattern.match(value):
            result.add_error(ValidationError(
                f"Field '{field}' must be a valid email address",
                field=field,
                value=value,
                suggestions=["Use format: user@example.com"]
            ))
        
        return result


class SecurityPatternRule(ValidationRule):
    """Rule for security pattern validation"""
    def __init__(self):
        super().__init__("security_pattern", "No dangerous patterns")
        self.dangerous_patterns = [
            (r'<script.*?</script>', "Script tags"),
            (r'javascript:', "JavaScript protocol"),
            (r'on\w+\s*=', "Event handlers"),
            (r'eval\s*\(', "Eval function"),
            (r'exec\s*\(', "Exec function"),
            (r'__import__\s*\(', "Dynamic imports")
        ]
    
    def validate(self, value: Any, field: str = None, context: Dict[str, Any] = None) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            return result
        
        for pattern, description in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                result.add_error(SecurityValidationError(
                    f"Field '{field}' contains potentially dangerous content: {description}",
                    field=field,
                    value=value,
                    security_level="high",
                    recommendations=["Remove or sanitize dangerous content"]
                ))
        
        return result


class ValidationEngine:
    """Main validation engine for DurgasAI models"""
    
    def __init__(self):
        self.rules = {}
        self.cache = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules"""
        self.rules = {
            "required": RequiredFieldRule(),
            "string_length": StringLengthRule(),
            "email_format": EmailFormatRule(),
            "security_pattern": SecurityPatternRule()
        }
    
    def register_rule(self, name: str, rule: ValidationRule):
        """Register a custom validation rule"""
        self.rules[name] = rule
    
    def validate_model(self, model_class: Type[T], data: Dict[str, Any], 
                      rules: Dict[str, List[str]] = None) -> ValidationResult:
        """Validate data against a model class with custom rules"""
        try:
            # Check cache first
            cache_key = f"{model_class.__name__}_{hash(str(data))}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Validate using Pydantic first
            try:
                instance = model_class(**data)
            except ValidationError as e:
                result = ValidationResult(False)
                for error in e.errors():
                    result.add_error(ValidationError(
                        error["msg"],
                        field=".".join(str(x) for x in error["loc"]),
                        value=error.get("input"),
                        error_type="pydantic_validation"
                    ))
                return result
            
            # Apply custom rules if provided
            if rules:
                result = self._apply_custom_rules(instance, rules)
                self.cache[cache_key] = result
                return result
            
            # Default validation
            result = ValidationResult(True)
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            result = ValidationResult(False)
            result.add_error(ValidationError(
                f"Unexpected validation error: {str(e)}",
                error_type="system_error"
            ))
            return result
    
    def _apply_custom_rules(self, instance: BaseModel, rules: Dict[str, List[str]]) -> ValidationResult:
        """Apply custom validation rules to a model instance"""
        result = ValidationResult(True)
        
        for field_name, rule_names in rules.items():
            if hasattr(instance, field_name):
                value = getattr(instance, field_name)
                
                for rule_name in rule_names:
                    if rule_name in self.rules:
                        rule_result = self.rules[rule_name].validate(value, field_name)
                        if not rule_result.is_valid:
                            result.errors.extend(rule_result.errors)
                            result.is_valid = False
                        result.warnings.extend(rule_result.warnings)
        
        return result
    
    def sanitize_data(self, data: Dict[str, Any], sanitization_rules: Dict[str, List[str]] = None) -> ValidationResult:
        """Sanitize data according to rules"""
        result = ValidationResult(True)
        sanitized_data = data.copy()
        
        for field_name, value in data.items():
            if isinstance(value, str):
                # Basic sanitization
                sanitized_value = self._sanitize_string(value)
                if sanitized_value != value:
                    sanitized_data[field_name] = sanitized_value
                    result.warnings.append(f"Field '{field_name}' was sanitized")
        
        result.sanitized_data = sanitized_data
        return result
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string value"""
        # Remove dangerous HTML tags
        value = re.sub(r'<script.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r'<iframe.*?</iframe>', '', value, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous protocols
        value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
        value = re.sub(r'data:', '', value, flags=re.IGNORECASE)
        
        # Remove event handlers
        value = re.sub(r'on\w+\s*=', '', value, flags=re.IGNORECASE)
        
        # Limit length
        if len(value) > 10000:
            value = value[:10000] + "..."
        
        return value.strip()
    
    def validate_security(self, data: Dict[str, Any], security_level: str = "medium") -> ValidationResult:
        """Validate data for security issues"""
        result = ValidationResult(True)
        
        # Check for dangerous patterns
        for field_name, value in data.items():
            if isinstance(value, str):
                security_rule = SecurityPatternRule()
                rule_result = security_rule.validate(value, field_name)
                if not rule_result.is_valid:
                    result.errors.extend(rule_result.errors)
                    result.is_valid = False
        
        # Check for suspicious file paths
        if "file_path" in data:
            file_path = data["file_path"]
            if isinstance(file_path, str):
                if ".." in file_path or file_path.startswith("/"):
                    result.add_error(SecurityValidationError(
                        "File path contains suspicious patterns",
                        field="file_path",
                        value=file_path,
                        security_level="high",
                        recommendations=["Use relative paths and avoid directory traversal"]
                    ))
        
        return result
    
    def validate_chat_message(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate chat message data"""
        rules = {
            "content": ["required", "string_length", "security_pattern"],
            "role": ["required", "string_length"]
        }
        
        result = self.validate_model(ChatMessage, data, rules)
        
        # Additional chat-specific validation
        if "role" in data:
            valid_roles = ["user", "assistant", "system", "tool"]
            if data["role"] not in valid_roles:
                result.add_error(ValidationError(
                    f"Invalid role '{data['role']}'. Must be one of: {valid_roles}",
                    field="role",
                    value=data["role"]
                ))
        
        return result
    
    def validate_tool_definition(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate tool definition data"""
        rules = {
            "name": ["required", "string_length", "security_pattern"],
            "description": ["required", "string_length"],
            "code": ["required", "string_length", "security_pattern"]
        }
        
        result = self.validate_model(ToolDefinition, data, rules)
        
        # Additional tool-specific validation
        if "code" in data:
            code = data["code"]
            if isinstance(code, str):
                # Check for dangerous imports
                dangerous_imports = ["os", "subprocess", "sys", "shutil", "glob"]
                for import_name in dangerous_imports:
                    if f"import {import_name}" in code or f"from {import_name}" in code:
                        result.add_warning(f"Tool code imports potentially dangerous module: {import_name}")
        
        return result
    
    def validate_workflow_definition(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate workflow definition data"""
        rules = {
            "name": ["required", "string_length", "security_pattern"],
            "description": ["required", "string_length"]
        }
        
        result = self.validate_model(WorkflowDefinition, data, rules)
        
        # Additional workflow-specific validation
        if "steps" in data and isinstance(data["steps"], list):
            for i, step in enumerate(data["steps"]):
                if not isinstance(step, dict):
                    result.add_error(ValidationError(
                        f"Step {i} must be a dictionary",
                        field=f"steps[{i}]",
                        value=step
                    ))
                elif "name" not in step:
                    result.add_error(ValidationError(
                        f"Step {i} must have a name",
                        field=f"steps[{i}].name",
                        value=step
                    ))
        
        return result


class ErrorHandler:
    """Error handling and recovery utilities"""
    
    def __init__(self):
        self.error_log = []
        self.recovery_strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies"""
        self.recovery_strategies = {
            "validation_error": self._handle_validation_error,
            "security_error": self._handle_security_error,
            "sanitization_error": self._handle_sanitization_error,
            "system_error": self._handle_system_error
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle an error and attempt recovery"""
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        self.error_log.append(error_info)
        
        # Attempt recovery
        if isinstance(error, ValidationError):
            return self._handle_validation_error(error, context)
        elif isinstance(error, SecurityValidationError):
            return self._handle_security_error(error, context)
        elif isinstance(error, DataSanitizationError):
            return self._handle_sanitization_error(error, context)
        else:
            return self._handle_system_error(error, context)
    
    def _handle_validation_error(self, error: ValidationError, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle validation errors"""
        return {
            "recovered": False,
            "action": "validation_failed",
            "message": "Data validation failed",
            "suggestions": error.suggestions,
            "field": error.field
        }
    
    def _handle_security_error(self, error: SecurityValidationError, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle security errors"""
        return {
            "recovered": False,
            "action": "security_violation",
            "message": "Security validation failed",
            "security_level": error.security_level,
            "recommendations": error.suggestions
        }
    
    def _handle_sanitization_error(self, error: DataSanitizationError, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle sanitization errors"""
        return {
            "recovered": True,
            "action": "data_sanitized",
            "message": "Data was sanitized",
            "sanitized_value": error.sanitized_value
        }
    
    def _handle_system_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle system errors"""
        return {
            "recovered": False,
            "action": "system_error",
            "message": "System error occurred",
            "error_details": str(error)
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors"""
        if not self.error_log:
            return {"total_errors": 0, "error_types": {}}
        
        error_types = {}
        for error in self.error_log:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "recent_errors": self.error_log[-10:]  # Last 10 errors
        }


# Global instances
validation_engine = ValidationEngine()
error_handler = ErrorHandler()


def validate_model_data(model_class: Type[T], data: Dict[str, Any], 
                       rules: Dict[str, List[str]] = None) -> ValidationResult:
    """Validate data against a model class"""
    return validation_engine.validate_model(model_class, data, rules)


def sanitize_data(data: Dict[str, Any], sanitization_rules: Dict[str, List[str]] = None) -> ValidationResult:
    """Sanitize data according to rules"""
    return validation_engine.sanitize_data(data, sanitization_rules)


def validate_security(data: Dict[str, Any], security_level: str = "medium") -> ValidationResult:
    """Validate data for security issues"""
    return validation_engine.validate_security(data, security_level)


def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle an error and attempt recovery"""
    return error_handler.handle_error(error, context)


def get_validation_summary() -> Dict[str, Any]:
    """Get validation and error summary"""
    return {
        "validation_cache_size": len(validation_engine.cache),
        "error_summary": error_handler.get_error_summary()
    }
