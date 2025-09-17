"""
Advanced Logging Service for DurgasAI
Provides comprehensive logging functionality with date-based file naming,
rotation, compression, and structured logging capabilities.
"""

import logging
import logging.handlers
import os
import gzip
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union
import threading
from functools import wraps
import traceback
import sys


class AdvancedLogger:
    """
    Advanced logging service with date-based file naming and rotation
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AdvancedLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logging service"""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.loggers = {}
            self.config = {}
            self.log_directory = "./output/logs"
            self._setup_log_directory()
    
    def _setup_log_directory(self):
        """Create log directory if it doesn't exist"""
        Path(self.log_directory).mkdir(parents=True, exist_ok=True)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the logging service
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.log_directory = config.get('log_directory', './output/logs')
        self._setup_log_directory()
        
        # Configure root logger
        self._configure_root_logger()
        
        # Configure specific loggers
        self._configure_component_loggers()
    
    def _configure_root_logger(self) -> None:
        """Configure the root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler if enabled
        if self.config.get('enable_console_logging', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
            console_handler.setFormatter(self._get_formatter())
            root_logger.addHandler(console_handler)
        
        # Add file handler with date-based naming
        if self.config.get('enable_file_logging', True):
            file_handler = self._create_file_handler('main')
            root_logger.addHandler(file_handler)
    
    def _configure_component_loggers(self) -> None:
        """Configure component-specific loggers"""
        components = [
            'chat', 'tools', 'templates', 'workflows', 'sessions',
            'vector_db', 'system_monitor', 'security', 'performance',
            'integrations', 'development', 'ui', 'api'
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
            
            # Add file handler for component
            if self.config.get('enable_component_logging', True):
                file_handler = self._create_file_handler(component)
                logger.addHandler(file_handler)
    
    def _create_file_handler(self, component: str) -> logging.Handler:
        """
        Create a file handler with date-based naming
        
        Args:
            component: Component name for log file
            
        Returns:
            Configured file handler
        """
        # Create date-based filename
        date_str = datetime.now().strftime('%Y%m%d')
        log_filename = f"{component}_{date_str}.log"
        log_path = os.path.join(self.log_directory, log_filename)
        
        # Create rotating file handler
        max_bytes = self.config.get('max_log_file_size_mb', 50) * 1024 * 1024
        backup_count = self.config.get('max_log_files', 5)
        
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        handler.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        handler.setFormatter(self._get_formatter())
        
        # Add compression to rotated files if enabled
        if self.config.get('enable_log_compression', True):
            handler.namer = self._compress_rotated_file
        
        return handler
    
    def _compress_rotated_file(self, name: str) -> str:
        """
        Compress rotated log files
        
        Args:
            name: Original filename
            
        Returns:
            Compressed filename
        """
        if name.endswith('.log'):
            compressed_name = f"{name}.gz"
            with open(name, 'rb') as f_in:
                with gzip.open(compressed_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(name)
            return compressed_name
        return name
    
    def _get_formatter(self) -> logging.Formatter:
        """Get the logging formatter"""
        format_string = self.config.get(
            'log_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.Formatter(format_string)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific component
        
        Args:
            name: Logger name (usually module or component name)
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_performance(self, func_name: str, duration: float, **kwargs) -> None:
        """
        Log performance metrics
        
        Args:
            func_name: Function name
            duration: Execution duration in seconds
            **kwargs: Additional performance metrics
        """
        if self.config.get('enable_performance_logging', False):
            perf_logger = self.get_logger('performance')
            perf_data = {
                'function': func_name,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            perf_logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Log error with additional context
        
        Args:
            error: Exception instance
            context: Additional context information
        """
        error_logger = self.get_logger('error')
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        error_logger.error(f"ERROR: {json.dumps(error_data)}")
    
    def log_user_action(self, action: str, user_id: str = None, **kwargs) -> None:
        """
        Log user actions for audit trail
        
        Args:
            action: User action description
            user_id: User identifier
            **kwargs: Additional action data
        """
        if self.config.get('enable_audit_logging', False):
            audit_logger = self.get_logger('audit')
            action_data = {
                'action': action,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            audit_logger.info(f"AUDIT: {json.dumps(action_data)}")
    
    def cleanup_old_logs(self) -> None:
        """Clean up old log files based on retention policy"""
        if not self.config.get('enable_log_cleanup', True):
            return
        
        retention_days = self.config.get('log_retention_days', 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            log_path = Path(self.log_directory)
            for log_file in log_path.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.get_logger('cleanup').info(f"Deleted old log file: {log_file}")
        except Exception as e:
            self.get_logger('cleanup').error(f"Error cleaning up logs: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dictionary with logging statistics
        """
        try:
            log_path = Path(self.log_directory)
            log_files = list(log_path.glob('*.log*'))
            
            total_size = sum(f.stat().st_size for f in log_files)
            
            return {
                'total_files': len(log_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'log_directory': str(log_path.absolute()),
                'files': [
                    {
                        'name': f.name,
                        'size_mb': round(f.stat().st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in log_files
                ]
            }
        except Exception as e:
            return {'error': str(e)}


# Decorators for easy logging
def log_function_calls(logger_name: str = 'function_calls'):
    """
    Decorator to log function calls with execution time
    
    Args:
        logger_name: Name of the logger to use
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = AdvancedLogger().get_logger(logger_name)
            start_time = datetime.now()
            
            logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Function {func.__name__} completed in {duration:.4f}s")
                
                # Log performance if enabled
                AdvancedLogger().log_performance(func.__name__, duration)
                
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
                AdvancedLogger().log_error_with_context(e, {
                    'function': func.__name__,
                    'duration': duration,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
                raise
        return wrapper
    return decorator


def log_errors(logger_name: str = 'errors'):
    """
    Decorator to log errors with context
    
    Args:
        logger_name: Name of the logger to use
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                AdvancedLogger().log_error_with_context(e, {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
                raise
        return wrapper
    return decorator


# Global logger instance
logger_service = AdvancedLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logger_service.get_logger(name)


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure the global logging service
    
    Args:
        config: Logging configuration
    """
    logger_service.configure(config)


def cleanup_logs() -> None:
    """Clean up old log files"""
    logger_service.cleanup_old_logs()


def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics"""
    return logger_service.get_log_stats()
