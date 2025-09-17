"""
Logging Utilities for DurgasAI
Provides convenient functions and decorators for logging throughout the application.
"""

import functools
import time
from typing import Any, Callable, Optional, Dict
from datetime import datetime
import traceback
import inspect

from .logger_service import get_logger, logger_service


def setup_logging_from_config(config_manager):
    """
    Setup logging from configuration manager
    
    Args:
        config_manager: Configuration manager instance
    """
    try:
        config = config_manager.get_config()
        logging_config = config.get('logging', {})
        
        from .logger_service import configure_logging
        configure_logging(logging_config)
        
        logger = get_logger('setup')
        logger.info("Logging service configured successfully")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")


def log_function_entry_exit(logger_name: str = None, log_args: bool = False, log_result: bool = False):
    """
    Decorator to log function entry and exit
    
    Args:
        logger_name: Name of the logger to use
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger name from function module if not provided
            if logger_name is None:
                module_name = func.__module__
                if module_name:
                    logger_name_actual = module_name.split('.')[-1]
                else:
                    logger_name_actual = 'default'
            else:
                logger_name_actual = logger_name
            
            logger = get_logger(logger_name_actual)
            
            # Log function entry
            entry_msg = f"Entering {func.__name__}"
            if log_args:
                entry_msg += f" with args={args}, kwargs={kwargs}"
            logger.debug(entry_msg)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log function exit
                duration = time.time() - start_time
                exit_msg = f"Exiting {func.__name__} (took {duration:.4f}s)"
                if log_result:
                    exit_msg += f", result={result}"
                logger.debug(exit_msg)
                
                # Log performance
                logger_service.log_performance(func.__name__, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Exception in {func.__name__} after {duration:.4f}s: {e}")
                logger_service.log_error_with_context(e, {
                    'function': func.__name__,
                    'duration': duration,
                    'module': func.__module__
                })
                raise
                
        return wrapper
    return decorator


def log_performance(threshold_seconds: float = 1.0):
    """
    Decorator to log slow function performance
    
    Args:
        threshold_seconds: Only log if function takes longer than this threshold
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > threshold_seconds:
                    logger = get_logger('performance')
                    logger.warning(f"Slow function {func.__name__} took {duration:.4f}s")
                    logger_service.log_performance(func.__name__, duration, 
                                                 threshold=threshold_seconds, 
                                                 slow=True)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger = get_logger('performance')
                logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
                raise
                
        return wrapper
    return decorator


def log_user_action(action_name: str):
    """
    Decorator to log user actions
    
    Args:
        action_name: Name of the action being performed
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('user_actions')
            logger.info(f"User action: {action_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"User action {action_name} completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"User action {action_name} failed: {e}")
                raise
                
        return wrapper
    return decorator


def log_api_call(endpoint: str = None):
    """
    Decorator to log API calls
    
    Args:
        endpoint: API endpoint being called
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            api_endpoint = endpoint or func.__name__
            logger = get_logger('api_calls')
            
            logger.info(f"API call to {api_endpoint}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"API call to {api_endpoint} completed in {duration:.4f}s")
                logger_service.log_performance(f"api_{api_endpoint}", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"API call to {api_endpoint} failed after {duration:.4f}s: {e}")
                logger_service.log_error_with_context(e, {
                    'endpoint': api_endpoint,
                    'duration': duration,
                    'function': func.__name__
                })
                raise
                
        return wrapper
    return decorator


def log_database_operation(operation_type: str):
    """
    Decorator to log database operations
    
    Args:
        operation_type: Type of database operation (SELECT, INSERT, UPDATE, DELETE)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('database')
            
            logger.debug(f"Database {operation_type} operation: {func.__name__}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.debug(f"Database {operation_type} operation completed in {duration:.4f}s")
                logger_service.log_performance(f"db_{operation_type.lower()}", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Database {operation_type} operation failed after {duration:.4f}s: {e}")
                logger_service.log_error_with_context(e, {
                    'operation_type': operation_type,
                    'duration': duration,
                    'function': func.__name__
                })
                raise
                
        return wrapper
    return decorator


def log_file_operation(operation_type: str):
    """
    Decorator to log file operations
    
    Args:
        operation_type: Type of file operation (READ, WRITE, DELETE, etc.)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('file_operations')
            
            # Try to extract file path from arguments
            file_path = None
            if args:
                file_path = str(args[0])
            
            logger.info(f"File {operation_type}: {file_path or func.__name__}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"File {operation_type} completed in {duration:.4f}s")
                logger_service.log_performance(f"file_{operation_type.lower()}", duration, 
                                             file_path=file_path)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"File {operation_type} failed after {duration:.4f}s: {e}")
                logger_service.log_error_with_context(e, {
                    'operation_type': operation_type,
                    'file_path': file_path,
                    'duration': duration,
                    'function': func.__name__
                })
                raise
                
        return wrapper
    return decorator


def log_streamlit_event(event_type: str):
    """
    Decorator to log Streamlit UI events
    
    Args:
        event_type: Type of Streamlit event (button_click, page_change, etc.)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('streamlit_events')
            
            logger.info(f"Streamlit event: {event_type} - {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Streamlit event {event_type} completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"Streamlit event {event_type} failed: {e}")
                logger_service.log_error_with_context(e, {
                    'event_type': event_type,
                    'function': func.__name__
                })
                raise
                
        return wrapper
    return decorator


def log_chat_interaction():
    """
    Decorator to log chat interactions
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger('chat')
            
            # Try to extract user message
            user_message = None
            if args:
                user_message = str(args[0])[:100] + "..." if len(str(args[0])) > 100 else str(args[0])
            
            logger.info(f"Chat interaction: {user_message or 'No message'}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"Chat interaction completed in {duration:.4f}s")
                logger_service.log_performance("chat_interaction", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Chat interaction failed after {duration:.4f}s: {e}")
                logger_service.log_error_with_context(e, {
                    'user_message': user_message,
                    'duration': duration,
                    'function': func.__name__
                })
                raise
                
        return wrapper
    return decorator


def log_tool_execution(tool_name: str = None):
    """
    Decorator to log tool executions
    
    Args:
        tool_name: Name of the tool being executed
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool = tool_name or func.__name__
            logger = get_logger('tools')
            
            logger.info(f"Executing tool: {tool}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"Tool {tool} executed successfully in {duration:.4f}s")
                logger_service.log_performance(f"tool_{tool}", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Tool {tool} execution failed after {duration:.4f}s: {e}")
                logger_service.log_error_with_context(e, {
                    'tool_name': tool,
                    'duration': duration,
                    'function': func.__name__
                })
                raise
                
        return wrapper
    return decorator


# Convenience functions for common logging scenarios
def log_info(message: str, component: str = 'main', **kwargs):
    """Log info message"""
    logger = get_logger(component)
    logger.info(message, extra=kwargs)


def log_warning(message: str, component: str = 'main', **kwargs):
    """Log warning message"""
    logger = get_logger(component)
    logger.warning(message, extra=kwargs)


def log_error(message: str, component: str = 'main', exception: Exception = None, **kwargs):
    """Log error message with optional exception"""
    logger = get_logger(component)
    logger.error(message, extra=kwargs)
    
    if exception:
        logger_service.log_error_with_context(exception, kwargs)


def log_debug(message: str, component: str = 'main', **kwargs):
    """Log debug message"""
    logger = get_logger(component)
    logger.debug(message, extra=kwargs)


def log_success(message: str, component: str = 'main', **kwargs):
    """Log success message"""
    logger = get_logger(component)
    logger.info(f"SUCCESS: {message}", extra=kwargs)


def log_startup(component_name: str, version: str = None):
    """Log component startup"""
    logger = get_logger('startup')
    version_info = f" v{version}" if version else ""
    logger.info(f"Starting {component_name}{version_info}")


def log_shutdown(component_name: str):
    """Log component shutdown"""
    logger = get_logger('shutdown')
    logger.info(f"Shutting down {component_name}")


def log_configuration_change(component: str, setting: str, old_value: Any, new_value: Any):
    """Log configuration changes"""
    logger = get_logger('configuration')
    logger.info(f"Configuration change in {component}: {setting} changed from {old_value} to {new_value}")
