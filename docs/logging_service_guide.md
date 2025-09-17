# DurgasAI Advanced Logging Service

## Overview

The DurgasAI logging service provides comprehensive logging functionality with date-based file naming, rotation, compression, and structured logging capabilities. It replaces the simple log file configuration with a robust, enterprise-grade logging solution.

## Features

### 🗂️ Date-Based File Naming
- **Format**: `{component}_{YYYYMMDD}.log`
- **Examples**: 
  - `main_20250915.log`
  - `chat_20250915.log`
  - `tools_20250915.log`

### 🔄 Automatic Log Rotation
- **Size-based rotation**: Configurable maximum file size (default: 50MB)
- **Backup files**: Maintains configurable number of backup files (default: 5)
- **Compression**: Automatic compression of rotated files (.gz)

### 📊 Component-Based Logging
- **Separate loggers** for different application components
- **Components**: main, chat, tools, templates, workflows, sessions, vector_db, system_monitor, security, performance, integrations, development, ui, api, error, audit, performance, function_calls

### 🎯 Advanced Features
- **Performance logging**: Track function execution times
- **Error tracking**: Structured error logging with context
- **Audit logging**: User action tracking
- **Configuration change logging**: Track settings modifications
- **Log statistics**: Real-time log file statistics

## Configuration

### Logging Configuration Structure

```json
{
  "logging": {
    "log_level": "INFO",
    "log_directory": "./output/logs",
    "enable_file_logging": true,
    "enable_console_logging": true,
    "enable_component_logging": true,
    "enable_performance_logging": true,
    "enable_audit_logging": true,
    "enable_error_tracking": true,
    "max_log_file_size_mb": 50,
    "max_log_files": 5,
    "enable_log_rotation": true,
    "enable_log_compression": true,
    "enable_log_cleanup": true,
    "log_retention_days": 30,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y%m%d",
    "log_file_naming": "date_based"
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `log_level` | string | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `log_directory` | string | "./output/logs" | Directory for log files |
| `enable_file_logging` | boolean | true | Enable file-based logging |
| `enable_console_logging` | boolean | true | Enable console logging |
| `enable_component_logging` | boolean | true | Enable component-specific loggers |
| `enable_performance_logging` | boolean | true | Enable performance tracking |
| `enable_audit_logging` | boolean | true | Enable audit trail logging |
| `max_log_file_size_mb` | integer | 50 | Maximum log file size in MB |
| `max_log_files` | integer | 5 | Maximum number of backup files |
| `log_retention_days` | integer | 30 | Days to retain log files |

## Usage

### Basic Logging

```python
from utils.logging_utils import log_info, log_warning, log_error, log_debug

# Basic logging
log_info("Application started", "main")
log_warning("Low disk space", "system_monitor")
log_error("Database connection failed", "database")
log_debug("Processing request", "api")
```

### Component-Specific Logging

```python
from utils.logging_utils import get_logger

# Get component logger
chat_logger = get_logger('chat')
tools_logger = get_logger('tools')

# Log to specific component
chat_logger.info("User sent message")
tools_logger.info("Tool executed successfully")
```

### Decorators

#### Function Entry/Exit Logging
```python
from utils.logging_utils import log_function_entry_exit

@log_function_entry_exit("api", log_args=True, log_result=True)
def process_request(data):
    return {"status": "success"}
```

#### Performance Logging
```python
from utils.logging_utils import log_performance

@log_performance(threshold_seconds=1.0)
def slow_operation():
    time.sleep(2)  # Will be logged as slow
    return "done"
```

#### User Action Logging
```python
from utils.logging_utils import log_user_action

@log_user_action("file_upload")
def upload_file(file_path):
    return "uploaded"
```

#### API Call Logging
```python
from utils.logging_utils import log_api_call

@log_api_call("chat_endpoint")
def chat_api(message):
    return "response"
```

#### Database Operation Logging
```python
from utils.logging_utils import log_database_operation

@log_database_operation("SELECT")
def get_user_data(user_id):
    return user_data
```

#### File Operation Logging
```python
from utils.logging_utils import log_file_operation

@log_file_operation("READ")
def read_config_file(path):
    return config_data
```

#### Chat Interaction Logging
```python
from utils.logging_utils import log_chat_interaction

@log_chat_interaction()
def process_chat_message(message):
    return response
```

#### Tool Execution Logging
```python
from utils.logging_utils import log_tool_execution

@log_tool_execution("data_analyzer")
def analyze_data(data):
    return analysis_result
```

### Advanced Features

#### Performance Tracking
```python
from utils.logger_service import logger_service

# Log performance metrics
logger_service.log_performance("data_processing", 2.5, 
                              records_processed=1000,
                              memory_used="512MB")
```

#### Error Context Logging
```python
from utils.logger_service import logger_service

try:
    risky_operation()
except Exception as e:
    logger_service.log_error_with_context(e, {
        'user_id': 'user123',
        'operation': 'data_export',
        'file_path': '/tmp/export.csv'
    })
```

#### User Action Audit Trail
```python
from utils.logger_service import logger_service

logger_service.log_user_action("settings_change", user_id="user123",
                              setting="log_level", 
                              old_value="INFO", 
                              new_value="DEBUG")
```

#### Configuration Changes
```python
from utils.logging_utils import log_configuration_change

log_configuration_change("logging", "log_level", "INFO", "DEBUG")
```

#### Startup/Shutdown Logging
```python
from utils.logging_utils import log_startup, log_shutdown

log_startup("DurgasAI Application", "1.0.0")
# ... application logic ...
log_shutdown("DurgasAI Application")
```

### Log Statistics

```python
from utils.logger_service import get_log_stats

stats = get_log_stats()
print(f"Total log files: {stats['total_files']}")
print(f"Total size: {stats['total_size_mb']} MB")
```

### Log Cleanup

```python
from utils.logger_service import cleanup_logs

# Clean up old log files based on retention policy
cleanup_logs()
```

## Integration

### Application Initialization

```python
from utils.config_manager import ConfigManager
from utils.logging_utils import setup_logging_from_config

# Initialize configuration
config_manager = ConfigManager()

# Setup logging from configuration
setup_logging_from_config(config_manager)
```

### Settings UI Integration

The logging configuration is integrated into the DurgasAI settings page under "Security & Logging" tab. Users can:

- Change log directory
- Adjust log levels
- Configure file rotation settings
- Enable/disable specific logging features
- View log statistics
- Test logging configuration

## File Structure

```
output/logs/
├── main_20250915.log          # Main application logs
├── chat_20250915.log          # Chat interactions
├── tools_20250915.log         # Tool executions
├── api_20250915.log           # API calls
├── error_20250915.log         # Error tracking
├── audit_20250915.log         # Audit trail
├── performance_20250915.log   # Performance metrics
└── main_20250915.log.1.gz     # Compressed rotated files
```

## Best Practices

1. **Use appropriate log levels**:
   - DEBUG: Detailed information for debugging
   - INFO: General information about application flow
   - WARNING: Something unexpected happened
   - ERROR: An error occurred but the application can continue
   - CRITICAL: A serious error occurred

2. **Include context in log messages**:
   ```python
   log_info(f"User {user_id} uploaded file {filename}", "chat")
   ```

3. **Use decorators for consistent logging**:
   ```python
   @log_function_entry_exit("api")
   def api_endpoint():
       pass
   ```

4. **Monitor log file sizes** and adjust rotation settings as needed

5. **Regular cleanup** of old log files to manage disk space

6. **Use component-specific loggers** for better organization

## Troubleshooting

### Common Issues

1. **Log files not created**:
   - Check log directory permissions
   - Verify logging configuration
   - Ensure `enable_file_logging` is true

2. **High disk usage**:
   - Adjust `max_log_file_size_mb`
   - Reduce `log_retention_days`
   - Enable log compression

3. **Missing log messages**:
   - Check log level configuration
   - Verify component logger setup
   - Check for logging exceptions

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
# In config.json
{
  "logging": {
    "log_level": "DEBUG",
    "enable_console_logging": true
  }
}
```

## Migration from Old Logging

The new logging service is backward compatible. Existing code using simple print statements can be gradually migrated:

### Before (Old)
```python
print("Application started")
```

### After (New)
```python
from utils.logging_utils import log_info
log_info("Application started", "main")
```

## Performance Impact

- **Minimal overhead**: Logging operations are optimized for performance
- **Asynchronous writing**: File I/O doesn't block application flow
- **Configurable levels**: Debug logging can be disabled in production
- **Efficient rotation**: Automatic cleanup prevents disk space issues

## Security Considerations

- **Sensitive data**: Never log passwords, API keys, or personal information
- **Log access**: Secure log directory with appropriate permissions
- **Audit trail**: User actions are logged for security monitoring
- **Error sanitization**: Sanitize error messages before logging
