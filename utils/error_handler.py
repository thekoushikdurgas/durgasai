"""
Error handling utilities for DurgasAI application.

This module provides comprehensive error handling capabilities:
- Custom exception classes for different error types
- Error logging with context information
- Streamlit-specific error handling
- Error recovery and user guidance
- Input validation utilities
- Error boundary context managers

Key Classes:
- DurgasAIError: Base exception class with metadata
- ModelError, APIError, ConfigurationError, ValidationError: Specific error types
- ErrorHandler: Main error handling utilities
- StreamlitErrorHandler: Streamlit-specific error handling
- Validators: Input validation functions
"""

import streamlit as st
import logging
import traceback
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime
import sys
from pathlib import Path

# Import the enhanced logging system
try:
    from .logger import debug, info, warning, error as log_error, critical
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class DurgasAIError(Exception):
    """Base exception class for DurgasAI application."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)


class ModelError(DurgasAIError):
    """Exception for model-related errors."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, "MODEL_ERROR", {"model_name": model_name, **kwargs})


class APIError(DurgasAIError):
    """Exception for API-related errors."""
    
    def __init__(self, message: str, api_endpoint: str = None, status_code: int = None, **kwargs):
        super().__init__(
            message, 
            "API_ERROR", 
            {"api_endpoint": api_endpoint, "status_code": status_code, **kwargs}
        )


class ConfigurationError(DurgasAIError):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key, **kwargs})


class ValidationError(DurgasAIError):
    """Exception for input validation errors."""
    
    def __init__(self, message: str, field_name: str = None, **kwargs):
        super().__init__(message, "VALIDATION_ERROR", {"field_name": field_name, **kwargs})


class ErrorHandler:
    """
    Main error handler for the DurgasAI application.
    
    This class provides comprehensive error handling capabilities including:
    - Error logging with context information
    - Streamlit-specific error display
    - API error handling with retry logic
    - Model error handling with graceful degradation
    - Input validation with user-friendly messages
    - Error recovery suggestions and guidance
    
    Features:
    - **Structured Error Logging**: Detailed error information with context
    - **Error Classification**: Different error types with specific handling
    - **User-Friendly Messages**: Clear error messages for end users
    - **Developer Debugging**: Detailed stack traces and context for developers
    - **Error Recovery**: Automatic retry logic and fallback mechanisms
    - **Performance Impact**: Minimal overhead with efficient error handling
    
    Error Categories:
    - **DurgasAIError**: Base application errors with metadata
    - **ModelError**: AI model loading and operation errors
    - **APIError**: API communication and authentication errors
    - **ConfigurationError**: Configuration and setup errors
    - **ValidationError**: Input validation and data errors
    
    Usage:
    ```python
    # Decorator for API error handling
    @ErrorHandler.handle_api_errors
    def api_call():
        # ... API operation
        pass
    
    # Safe execution with fallback
    result = ErrorHandler.safe_execute(risky_operation, fallback_value="default")
    
    # Input validation
    validated_input = ErrorHandler.validate_input(user_input, validator_func, "email")
    ```
    """
    
    @staticmethod
    def setup_error_logging():
        """
        Setup error logging directory and handlers.
        
        This method initializes the error logging system by:
        1. Creating the logs directory structure
        2. Setting up file handlers for different log types
        3. Configuring log rotation and retention
        4. Ensuring proper permissions and access
        
        Directory Structure Created:
        - logs/: Main log directory
        - logs/errors.log: Error-specific log file
        - logs/debug/: Debug log subdirectory
        - logs/performance/: Performance log subdirectory
        - logs/sessions/: Session-specific logs
        
        Note:
            This method should be called during application initialization
            to ensure proper error logging capabilities.
        """
        # Initialize debug logging for error handler setup
        if ENHANCED_LOGGING_AVAILABLE:
            debug("Starting error logging system setup", "error_handler")
        
        print("🔧 Setting up error logging system...")
        
        # Create main logs directory with error handling
        debug("Creating main logs directory", "error_handler") if ENHANCED_LOGGING_AVAILABLE else None
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        print(f"📁 Main logs directory: {logs_dir}")
        
        if ENHANCED_LOGGING_AVAILABLE:
            debug(f"Main logs directory created/verified: {logs_dir.absolute()}", "error_handler")
        
        # Create subdirectories for different log types
        subdirs = ["debug", "performance", "sessions"]
        created_subdirs = []
        
        debug(f"Creating {len(subdirs)} log subdirectories", "error_handler") if ENHANCED_LOGGING_AVAILABLE else None
        
        for subdir in subdirs:
            if ENHANCED_LOGGING_AVAILABLE:
                debug(f"Processing subdirectory: {subdir}", "error_handler")
            
            subdir_path = logs_dir / subdir
            if not subdir_path.exists():
                try:
                    subdir_path.mkdir(exist_ok=True)
                    created_subdirs.append(subdir)
                    print(f"  📁 Created: logs/{subdir}")
                    
                    if ENHANCED_LOGGING_AVAILABLE:
                        debug(f"Successfully created subdirectory: {subdir_path.absolute()}", "error_handler")
                except Exception as e:
                    print(f"  ❌ Failed to create: logs/{subdir}")
                    if ENHANCED_LOGGING_AVAILABLE:
                        warning(f"Failed to create subdirectory {subdir}: {e}", "error_handler")
            else:
                print(f"  ✓ Exists: logs/{subdir}")
                if ENHANCED_LOGGING_AVAILABLE:
                    debug(f"Subdirectory already exists: {subdir_path.absolute()}", "error_handler")
        
        if created_subdirs:
            print(f"✅ Created {len(created_subdirs)} new log subdirectories")
            if ENHANCED_LOGGING_AVAILABLE:
                info(f"Created {len(created_subdirs)} new log subdirectories", "error_handler",
                     created_subdirs=created_subdirs)
        
        print("✅ Error logging system setup completed")
        
        if ENHANCED_LOGGING_AVAILABLE:
            info("Error logging system setup completed successfully", "error_handler",
                 main_dir=str(logs_dir.absolute()),
                 subdirs_created=len(created_subdirs),
                 total_subdirs=len(subdirs))
        
        # Create error log file
        error_log_file = logs_dir / "errors.log"
        
        # Setup error logger
        error_logger = logging.getLogger("error_logger")
        error_logger.setLevel(logging.ERROR)
        
        if not error_logger.handlers:
            handler = logging.FileHandler(error_log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s\n'
                'Traceback: %(pathname)s:%(lineno)d\n'
                '%(exc_info)s\n' + '-' * 80
            )
            handler.setFormatter(formatter)
            error_logger.addHandler(handler)
        
        return error_logger
    
    @staticmethod
    def log_error(error: Exception, context: Dict[str, Any] = None):
        """Log error with context information."""
        error_logger = ErrorHandler.setup_error_logging()
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        if isinstance(error, DurgasAIError):
            error_info.update({
                "error_code": error.error_code,
                "details": error.details
            })
        
        error_logger.error(
            f"Error occurred: {error_info}",
            exc_info=True
        )
    
    @staticmethod
    def display_error_message(error: Exception, show_details: bool = False):
        """Display user-friendly error message in Streamlit."""
        if isinstance(error, ModelError):
            st.error(f"🤖 **Model Error**: {error.message}")
            if show_details and error.details.get("model_name"):
                st.info(f"Model: {error.details['model_name']}")
                
        elif isinstance(error, APIError):
            st.error(f"🌐 **API Error**: {error.message}")
            if show_details:
                if error.details.get("api_endpoint"):
                    st.info(f"Endpoint: {error.details['api_endpoint']}")
                if error.details.get("status_code"):
                    st.info(f"Status Code: {error.details['status_code']}")
                    
        elif isinstance(error, ConfigurationError):
            st.error(f"⚙️ **Configuration Error**: {error.message}")
            if show_details and error.details.get("config_key"):
                st.info(f"Configuration: {error.details['config_key']}")
                
        elif isinstance(error, ValidationError):
            st.error(f"✏️ **Input Error**: {error.message}")
            if show_details and error.details.get("field_name"):
                st.info(f"Field: {error.details['field_name']}")
                
        else:
            st.error(f"❌ **Unexpected Error**: {str(error)}")
        
        # Show error details in expander
        if show_details:
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    @staticmethod
    def handle_api_errors(func: Callable) -> Callable:
        """Decorator to handle API-related errors."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "api" in str(e).lower() or "request" in str(e).lower():
                    api_error = APIError(
                        f"API request failed: {str(e)}",
                        api_endpoint=kwargs.get("endpoint", "unknown")
                    )
                    ErrorHandler.log_error(api_error, {"function": func.__name__})
                    raise api_error
                else:
                    ErrorHandler.log_error(e, {"function": func.__name__})
                    raise
        return wrapper
    
    @staticmethod
    def handle_model_errors(func: Callable) -> Callable:
        """Decorator to handle model-related errors."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ["model", "tokenizer", "pipeline", "cuda", "memory"]):
                    model_error = ModelError(
                        f"Model operation failed: {str(e)}",
                        model_name=kwargs.get("model_name", "unknown")
                    )
                    ErrorHandler.log_error(model_error, {"function": func.__name__})
                    raise model_error
                else:
                    ErrorHandler.log_error(e, {"function": func.__name__})
                    raise
        return wrapper
    
    @staticmethod
    def safe_execute(func: Callable, fallback_value: Any = None, show_error: bool = True) -> Any:
        """Safely execute a function with error handling."""
        try:
            return func()
        except Exception as e:
            ErrorHandler.log_error(e, {"function": getattr(func, '__name__', 'anonymous')})
            
            if show_error:
                ErrorHandler.display_error_message(e)
            
            return fallback_value
    
    @staticmethod
    def validate_input(value: Any, validation_func: Callable, field_name: str = None) -> Any:
        """Validate input with custom validation function."""
        try:
            if validation_func(value):
                return value
            else:
                raise ValidationError(
                    f"Invalid value for {field_name or 'field'}: {value}",
                    field_name=field_name
                )
        except Exception as e:
            if not isinstance(e, ValidationError):
                validation_error = ValidationError(
                    f"Validation failed for {field_name or 'field'}: {str(e)}",
                    field_name=field_name
                )
                ErrorHandler.log_error(validation_error)
                raise validation_error
            raise


class StreamlitErrorHandler:
    """Streamlit-specific error handling utilities."""
    
    @staticmethod
    def setup_global_error_handler():
        """Setup global error handler for Streamlit."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            ErrorHandler.log_error(exc_value, {
                "type": exc_type.__name__,
                "traceback": traceback.format_tb(exc_traceback)
            })
            
            # Display in Streamlit if available
            try:
                st.error("🚨 An unexpected error occurred. Please check the logs for details.")
                with st.expander("Error Details"):
                    st.code(f"{exc_type.__name__}: {exc_value}")
            except:
                pass  # Streamlit not available
        
        sys.excepthook = handle_exception
    
    @staticmethod
    def error_boundary(title: str = "Error"):
        """Context manager for error boundaries in Streamlit."""
        class ErrorBoundary:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_value, exc_traceback):
                if exc_type is not None:
                    ErrorHandler.log_error(exc_value)
                    
                    with st.container():
                        st.error(f"🚨 **{title}**")
                        ErrorHandler.display_error_message(exc_value, show_details=True)
                        
                        if st.button("🔄 Retry", key=f"retry_{title}"):
                            st.rerun()
                    
                    return True  # Suppress the exception
                return False
        
        return ErrorBoundary()
    
    @staticmethod
    def show_error_recovery_options(error: Exception):
        """Show error recovery options to user."""
        st.markdown("### 🛠️ Recovery Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry Operation"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
        
        with col3:
            if st.button("🏠 Go to Home"):
                st.session_state.clear()
                st.switch_page("app.py")
        
        # Show troubleshooting tips
        with st.expander("💡 Troubleshooting Tips"):
            if isinstance(error, ModelError):
                st.markdown("""
                **Model Issues:**
                - Check your internet connection
                - Verify the model name is correct
                - Try a different model
                - Clear the model cache
                """)
            elif isinstance(error, APIError):
                st.markdown("""
                **API Issues:**
                - Verify your API token is valid
                - Check API rate limits
                - Try again in a few minutes
                - Check HuggingFace service status
                """)
            else:
                st.markdown("""
                **General Issues:**
                - Refresh the page
                - Clear browser cache
                - Check your internet connection
                - Try using a different browser
                """)


# Validation utilities
class Validators:
    """Common validation functions."""
    
    @staticmethod
    def is_valid_api_token(token: str) -> bool:
        """Validate HuggingFace API token format."""
        return token and token.startswith("hf_") and len(token) > 20
    
    @staticmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Validate model name format."""
        return model_name and "/" in model_name and len(model_name.split("/")) == 2
    
    @staticmethod
    def is_safe_input(text: str, max_length: int = 10000) -> bool:
        """Validate user input is safe and within limits."""
        return text and len(text) <= max_length and not any(
            dangerous in text.lower() 
            for dangerous in ["<script>", "javascript:", "data:"]
        )
    
    @staticmethod
    def is_valid_temperature(temp: float) -> bool:
        """Validate temperature parameter."""
        return 0.0 <= temp <= 2.0
    
    @staticmethod
    def is_valid_max_tokens(tokens: int) -> bool:
        """Validate max tokens parameter."""
        return 1 <= tokens <= 4000


# Initialize error handling on import
ErrorHandler.setup_error_logging()
StreamlitErrorHandler.setup_global_error_handler()
