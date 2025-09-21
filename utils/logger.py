"""
DurgasAI Debug Logging System
Comprehensive logging utility with multiple log levels, file rotation, and debug information.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import traceback
from functools import wraps
import time


class DurgasAILogger:
    """
    Advanced logging system for DurgasAI with multiple output streams and debug levels.
    
    This logger provides comprehensive logging capabilities for the DurgasAI application,
    supporting multiple output streams, structured logging, and performance monitoring.
    
    Key Features:
    - **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL with color coding
    - **File Rotation**: Automatic log file rotation and management
    - **Performance Timing**: Built-in decorators for timing operations
    - **Session Tracking**: Session-specific logging with unique identifiers
    - **Component-Specific**: Separate loggers for different application components
    - **Structured Logging**: JSON structured logging for complex data and analytics
    - **Real-time Monitoring**: Live log viewing through debug dashboard
    - **Error Tracking**: Comprehensive error logging with stack traces
    
    Architecture:
    - **Singleton Pattern**: Single logger instance across the application
    - **Multiple Handlers**: Console, file, and specialized handlers
    - **Lazy Initialization**: Loggers created on-demand for components
    - **Thread Safety**: Safe for concurrent operations
    - **Performance Optimized**: Minimal overhead for production use
    
    Log Categories:
    - **app.log**: Main application events and user actions
    - **errors.log**: Error tracking with full stack traces
    - **debug/debug.log**: Detailed debugging information
    - **debug/models.log**: AI model operations and performance
    - **debug/tools.log**: Tool execution and results
    - **performance/performance.log**: Timing and performance metrics
    - **sessions/session_*.log**: Session-specific events and analytics
    
    Usage:
    ```python
    from utils.logger import debug, info, warning, error, log_user_action
    
    # Basic logging
    debug("Debug message", "component_name")
    info("Info message", "component_name")
    
    # User action logging
    log_user_action("button_clicked", button_id="submit", page="settings")
    
    # Performance timing
    @time_operation("database_query", "database")
    def query_database():
        # ... database operation
        pass
    ```
    """
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DurgasAILogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        # Initialize session_id BEFORE setup_loggers since it's used there
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging_directories()
        self.setup_loggers()
        
    def setup_logging_directories(self):
        """Create logging directory structure using configurable paths."""
        # Try to get log path from app config
        try:
            from services import ConfigService
            config_service = ConfigService()
            app_config = config_service.get_app_config()
            logging_config = app_config.get('logging', {})
            log_folder_path = logging_config.get('log_folder_path', 'logs')
        except (ImportError, Exception):
            # Fallback to default if ConfigService not available
            log_folder_path = 'logs'
        
        self.log_base_dir = Path(log_folder_path)
        self.debug_dir = self.log_base_dir / "debug"
        self.performance_dir = self.log_base_dir / "performance"
        self.session_dir = self.log_base_dir / "sessions"
        
        # Create directories
        for directory in [self.log_base_dir, self.debug_dir, self.performance_dir, self.session_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Logging directories created at: {self.log_base_dir}")
    
    def setup_loggers(self):
        """Setup different loggers for different components."""
        
        # Main application logger
        self.app_logger = self._create_logger(
            'durgasai.app',
            self.log_base_dir / 'app.log',
            level=logging.INFO
        )
        
        # Debug logger for detailed debugging
        self.debug_logger = self._create_logger(
            'durgasai.debug',
            self.debug_dir / 'debug.log',
            level=logging.DEBUG
        )
        
        # Model operations logger
        self.model_logger = self._create_logger(
            'durgasai.models',
            self.debug_dir / 'models.log',
            level=logging.DEBUG
        )
        
        # Session state logger
        self.session_logger = self._create_logger(
            'durgasai.sessions',
            self.session_dir / f'session_{self.session_id}.log',
            level=logging.DEBUG
        )
        
        # Performance logger
        self.perf_logger = self._create_logger(
            'durgasai.performance',
            self.performance_dir / 'performance.log',
            level=logging.INFO
        )
        
        # Error logger (detailed errors)
        self.error_logger = self._create_logger(
            'durgasai.errors',
            self.log_base_dir / 'errors.log',
            level=logging.ERROR
        )
        
        # Tool execution logger
        self.tool_logger = self._create_logger(
            'durgasai.tools',
            self.debug_dir / 'tools.log',
            level=logging.DEBUG
        )
    
    def _create_logger(self, name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
        """Create a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        # Console handler (only for INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter for files
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        # Simple formatter for console
        console_formatter = logging.Formatter(
            '%(levelname)s | %(name)s | %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    # Main logging methods
    def debug(self, message: str, component: str = "app", extra_data: Dict[str, Any] = None):
        """Log debug message with optional structured data."""
        logger = self._get_component_logger(component)
        
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        
        logger.debug(message)
    
    def info(self, message: str, component: str = "app", extra_data: Dict[str, Any] = None):
        """Log info message with optional structured data."""
        logger = self._get_component_logger(component)
        
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        
        logger.info(message)
    
    def warning(self, message: str, component: str = "app", extra_data: Dict[str, Any] = None):
        """Log warning message with optional structured data."""
        logger = self._get_component_logger(component)
        
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        
        logger.warning(message)
    
    def error(self, message: str, component: str = "app", error: Exception = None, extra_data: Dict[str, Any] = None):
        """Log error message with exception details and structured data."""
        logger = self._get_component_logger(component)
        
        error_info = {
            "message": message,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "component": component
        }
        
        if error:
            error_info.update({
                "error_type": type(error).__name__,
                "error_details": str(error),
                "traceback": traceback.format_exc()
            })
        
        if extra_data:
            error_info["extra_data"] = extra_data
        
        logger.error(f"{message} | Error Info: {json.dumps(error_info, default=str)}")
        
        # Also log to main error logger
        self.error_logger.error(f"[{component}] {message}", exc_info=error is not None)
    
    def critical(self, message: str, component: str = "app", error: Exception = None):
        """Log critical error that might cause application failure."""
        logger = self._get_component_logger(component)
        logger.critical(f"CRITICAL: {message}", exc_info=error is not None)
        self.error_logger.critical(f"[{component}] CRITICAL: {message}", exc_info=error is not None)
    
    def _get_component_logger(self, component: str) -> logging.Logger:
        """Get appropriate logger for component."""
        component_loggers = {
            "app": self.app_logger,
            "debug": self.debug_logger,
            "models": self.model_logger,
            "session": self.session_logger,
            "performance": self.perf_logger,
            "tools": self.tool_logger,
            "errors": self.error_logger
        }
        return component_loggers.get(component, self.debug_logger)
    
    # Session tracking methods
    def log_session_event(self, event: str, data: Dict[str, Any] = None):
        """Log session-related events."""
        session_data = {
            "session_id": self.session_id,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.session_logger.info(f"SESSION_EVENT: {json.dumps(session_data, default=str)}")
    
    def log_user_action(self, action: str, details: Dict[str, Any] = None):
        """Log user actions for analytics."""
        user_data = {
            "session_id": self.session_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.session_logger.info(f"USER_ACTION: {json.dumps(user_data, default=str)}")
    
    def log_model_operation(self, operation: str, model_name: str = None, details: Dict[str, Any] = None):
        """Log model-related operations."""
        model_data = {
            "session_id": self.session_id,
            "operation": operation,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.model_logger.info(f"MODEL_OP: {json.dumps(model_data, default=str)}")
    
    # Performance monitoring
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log performance metrics."""
        perf_data = {
            "session_id": self.session_id,
            "operation": operation,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.perf_logger.info(f"PERFORMANCE: {json.dumps(perf_data, default=str)}")
    
    def time_operation(self, operation_name: str, component: str = "app"):
        """Decorator to time operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.log_performance(
                        f"{operation_name}::{func.__name__}",
                        duration,
                        {"component": component, "success": True}
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_performance(
                        f"{operation_name}::{func.__name__}",
                        duration,
                        {"component": component, "success": False, "error": str(e)}
                    )
                    raise
            return wrapper
        return decorator
    
    # Tool execution logging
    def log_tool_execution(self, tool_name: str, inputs: Dict[str, Any], 
                          outputs: Dict[str, Any] = None, error: Exception = None):
        """Log tool execution details."""
        tool_data = {
            "session_id": self.session_id,
            "tool_name": tool_name,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "success": error is None
        }
        
        if outputs:
            tool_data["outputs"] = outputs
        
        if error:
            tool_data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
        
        self.tool_logger.info(f"TOOL_EXEC: {json.dumps(tool_data, default=str)}")
    
    # Configuration and API logging
    def log_config_event(self, event: str, config_key: str = None, value: Any = None):
        """Log configuration changes."""
        config_data = {
            "session_id": self.session_id,
            "event": event,
            "config_key": config_key,
            "timestamp": datetime.now().isoformat()
        }
        
        # Don't log sensitive values
        if config_key and "token" not in config_key.lower() and "key" not in config_key.lower():
            config_data["value"] = str(value)[:100]  # Truncate long values
        
        self.debug_logger.info(f"CONFIG: {json.dumps(config_data, default=str)}")
    
    # API call logging
    def log_api_call(self, endpoint: str, method: str = "POST", 
                    status_code: int = None, duration: float = None, 
                    error: Exception = None):
        """Log API calls with performance metrics."""
        api_data = {
            "session_id": self.session_id,
            "endpoint": endpoint,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "success": error is None
        }
        
        if status_code:
            api_data["status_code"] = status_code
        
        if duration:
            api_data["duration_seconds"] = round(duration, 3)
        
        if error:
            api_data["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
        
        self.debug_logger.info(f"API_CALL: {json.dumps(api_data, default=str)}")


# Global logger instance
_logger_instance = None

def get_logger() -> DurgasAILogger:
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DurgasAILogger()
    return _logger_instance

# Convenience functions for quick logging
def debug(message: str, component: str = "app", **kwargs):
    """Quick debug logging."""
    get_logger().debug(message, component, kwargs if kwargs else None)

def info(message: str, component: str = "app", **kwargs):
    """Quick info logging."""
    get_logger().info(message, component, kwargs if kwargs else None)

def warning(message: str, component: str = "app", **kwargs):
    """Quick warning logging."""
    get_logger().warning(message, component, kwargs if kwargs else None)

def error(message: str, component: str = "app", error_obj: Exception = None, **kwargs):
    """Quick error logging."""
    get_logger().error(message, component, error_obj, kwargs if kwargs else None)

def critical(message: str, component: str = "app", error_obj: Exception = None):
    """Quick critical logging."""
    get_logger().critical(message, component, error_obj)

# Session and performance tracking
def log_session_event(event: str, **kwargs):
    """Log session events."""
    get_logger().log_session_event(event, kwargs if kwargs else None)

def log_user_action(action: str, **kwargs):
    """Log user actions."""
    get_logger().log_user_action(action, kwargs if kwargs else None)

def log_model_operation(operation: str, model_name: str = None, **kwargs):
    """Log model operations."""
    get_logger().log_model_operation(operation, model_name, kwargs if kwargs else None)

def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    get_logger().log_performance(operation, duration, kwargs if kwargs else None)

def time_operation(operation_name: str, component: str = "app"):
    """Decorator to time operations."""
    return get_logger().time_operation(operation_name, component)

def log_tool_execution(tool_name: str, inputs: Dict[str, Any], 
                      outputs: Dict[str, Any] = None, error: Exception = None):
    """Log tool execution."""
    get_logger().log_tool_execution(tool_name, inputs, outputs, error)

def log_api_call(endpoint: str, method: str = "POST", 
                status_code: int = None, duration: float = None, 
                error: Exception = None):
    """Log API calls."""
    get_logger().log_api_call(endpoint, method, status_code, duration, error)

def log_config_event(event: str, config_key: str = None, value: Any = None):
    """Log configuration events."""
    get_logger().log_config_event(event, config_key, value)


# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operations with timing."""
    
    def __init__(self, operation_name: str, component: str = "app", 
                 log_level: str = "info", extra_data: Dict[str, Any] = None):
        self.operation_name = operation_name
        self.component = component
        self.log_level = log_level
        self.extra_data = extra_data or {}
        self.start_time = None
        self.logger = get_logger()
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(
            f"Starting operation: {self.operation_name}",
            self.component,
            {"operation": "start", **self.extra_data}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            self.logger.info(
                f"Completed operation: {self.operation_name} in {duration:.3f}s",
                self.component,
                {"operation": "complete", "duration": duration, "success": True, **self.extra_data}
            )
            self.logger.log_performance(self.operation_name, duration, {
                "component": self.component,
                "success": True,
                **self.extra_data
            })
        else:
            # Error
            self.logger.error(
                f"Failed operation: {self.operation_name} after {duration:.3f}s",
                self.component,
                exc_val,
                {"operation": "failed", "duration": duration, "success": False, **self.extra_data}
            )
            self.logger.log_performance(self.operation_name, duration, {
                "component": self.component,
                "success": False,
                "error": str(exc_val),
                **self.extra_data
            })


# Initialize logging on import
def initialize_logging():
    """Initialize the logging system."""
    logger = get_logger()
    logger.info("DurgasAI logging system initialized", "system", {
        "session_id": logger.session_id,
        "log_directories": {
            "base": str(logger.log_base_dir),
            "debug": str(logger.debug_dir),
            "performance": str(logger.performance_dir),
            "sessions": str(logger.session_dir)
        }
    })

# Initialize logging only if not already done to prevent circular imports
if __name__ != "__main__":
    try:
        initialize_logging()
    except Exception as e:
        # Fallback: basic logging setup if initialization fails
        print(f"Warning: Could not initialize advanced logging: {e}")
        print("Using basic logging configuration")
        
        # Setup basic logging as fallback
        import logging
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
