# 🛠️ DurgasAI Development Guide

## Overview

This guide provides comprehensive information for developers working on DurgasAI, including debugging, logging, performance monitoring, and best practices.

## 🔧 Enhanced Debugging System

### Logging Architecture

DurgasAI now includes a comprehensive logging system with multiple categories and output streams:

#### Log Categories

| Category | File | Purpose | Level |
|----------|------|---------|-------|
| **Application** | `logs/app.log` | Main application events and lifecycle | INFO |
| **Debug** | `logs/debug/debug.log` | Detailed debugging information | DEBUG |
| **Models** | `logs/debug/models.log` | AI model operations and performance | DEBUG |
| **Sessions** | `logs/sessions/session_*.log` | User interactions and session events | DEBUG |
| **Performance** | `logs/performance/performance.log` | Timing and performance metrics | INFO |
| **Tools** | `logs/debug/tools.log` | Tool execution and monitoring | DEBUG |
| **Errors** | `logs/errors.log` | Error details and stack traces | ERROR |

#### Using the Logging System

```python
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation

# Basic logging
debug("Detailed debugging information", "component_name")
info("General information", "component_name")
warning("Warning message", "component_name")
error("Error message", "component_name", exception_object)

# User action tracking
log_user_action("action_name", {"key": "value"})

# Performance timing with context manager
with LoggedOperation("operation_name", "component"):
    # Your code here
    result = some_operation()

# Performance timing with decorator
@time_operation("operation_name", "component")
def my_function():
    # Your code here
    pass
```

### Debug Dashboard

Access the Debug Dashboard via "🔧 Debug" in the navigation:

#### System Information Tab

- Python environment details
- System resource usage (CPU, memory, disk)
- Package versions and dependencies
- Platform and hardware information

#### Session State Tab

- Current session state variables (sensitive data masked)
- Session metrics and analytics
- Message history analysis
- Performance tracking

#### Logs Tab

- Real-time log file status and sizes
- Recent log entries by category
- Log file analysis and filtering
- Error pattern identification

#### Model Status Tab

- Current model configuration
- Available models and settings
- Model performance metrics
- Loading and response times

#### Performance Tab

- Response time trends
- Tool execution statistics
- System resource monitoring
- Performance bottleneck identification

#### Tools Tab

- Available tools and their status
- Configuration file validation
- API key status (without revealing keys)
- Workflow definitions and execution

## 📊 Session State Management

### Session State Keys

DurgasAI tracks the following session state variables:

| Key | Type | Purpose | Default | Logged |
|-----|------|---------|---------|---------|
| `messages` | List[Dict] | Conversation history | `[]` | ✅ |
| `model_loaded` | bool | Model availability status | `False` | ✅ |
| `current_model` | str | Active model name | `None` | ✅ |
| `api_token` | str | HuggingFace API token | `""` | ✅ (masked) |
| `system_prompt` | str | AI behavior prompt | Default | ✅ |
| `chat_session_id` | str | Session identifier | `"default"` | ✅ |
| `total_messages` | int | Message counter | `0` | ✅ |
| `session_start_time` | datetime | Session start time | Current | ✅ |
| `current_page` | str | Active page | `"home"` | ✅ |
| `debug_mode` | bool | Debug mode toggle | `False` | ✅ |
| `session_id` | str | Unique session ID | Generated | ✅ |

### Session State Best Practices

1. **Always validate session state**: Check if keys exist before accessing
2. **Use SessionManager methods**: Don't modify session state directly
3. **Log state changes**: All modifications are automatically logged
4. **Handle missing keys gracefully**: Provide defaults for missing values
5. **Monitor session size**: Large session states can impact performance

## 🤖 Model Management

### Model Lifecycle

1. **Configuration**: Load model config from `Config.AVAILABLE_MODELS`
2. **Validation**: Validate API tokens and model availability
3. **Initialization**: Load model (API or local) with LangChain integration
4. **Chain Creation**: Create conversation chain with memory
5. **Response Generation**: Process user inputs and generate responses
6. **Performance Monitoring**: Track response times and success rates

### Adding New Models

```python
# Add to utils/config.py
AVAILABLE_MODELS = {
    "new_model": ModelConfig(
        name="New Model Name",
        model_id="provider/model-name",
        provider=ModelProvider.HUGGINGFACE_API,
        description="Model description",
        max_tokens=512,
        temperature=0.7
    )
}
```

### Model Performance Monitoring

- Response generation times are automatically logged
- Success/failure rates tracked per model
- Memory usage monitoring for local models
- API call performance and error rates

## 🛠️ Tool System

### Tool Architecture

Tools consist of:
1. **JSON Configuration** (`output/tools/*.json`): Metadata and parameters
2. **Python Implementation** (`output/tools/code/*.py`): Tool logic
3. **Tool Manager** (`utils/tool_manager.py`): Execution and monitoring

### Creating New Tools

#### 1. Create JSON Configuration

```json
{
  "name": "my_tool",
  "description": "Tool description",
  "function_name": "my_tool_function",
  "input_parameters": [
    {
      "name": "param1",
      "type": "string",
      "description": "Parameter description",
      "required": true
    }
  ],
  "output_parameters": [
    {
      "name": "result",
      "type": "string",
      "description": "Result description"
    }
  ],
  "code": "my_tool.py",
  "category": "utility",
  "status": "enabled",
  "security_level": "low"
}
```

#### 2. Implement Python Function

```python
import json
from utils.logger import debug, info, error, log_tool_execution

def my_tool_function(param1: str) -> str:
    """
    Tool function with comprehensive logging.
    
    Args:
        param1 (str): Input parameter
        
    Returns:
        str: JSON result string
    """
    start_time = time.time()
    
    try:
        debug(f"Executing my_tool with param1: {param1}", "tools")
        
        # Your tool logic here
        result = {"output": f"Processed: {param1}"}
        
        execution_time = time.time() - start_time
        info(f"Tool executed successfully in {execution_time:.3f}s", "tools")
        
        log_tool_execution("my_tool", {"param1": param1}, result)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        execution_time = time.time() - start_time
        error(f"Tool execution failed: {e}", "tools", e)
        log_tool_execution("my_tool", {"param1": param1}, error=e)
        
        return json.dumps({"error": str(e)}, indent=2)
```

## 📈 Performance Monitoring

### Automatic Performance Tracking

All major operations are automatically timed and logged:

- **Model Operations**: Loading, initialization, response generation
- **Tool Execution**: Individual tool performance
- **User Interactions**: Page navigation, message sending
- **API Calls**: HuggingFace API request timing
- **Session Operations**: State updates and management

### Performance Optimization

#### Response Time Optimization

```python
# Use the timing decorator for critical functions
@time_operation("critical_operation", "component")
def critical_function():
    # Your code here
    pass

# Use context manager for detailed timing
with LoggedOperation("complex_operation", "component", extra_data={"param": "value"}):
    # Your complex operation here
    pass
```

#### Memory Management

- Monitor session state size
- Clear chat history periodically
- Use appropriate model sizes
- Monitor tool execution memory usage

### Performance Metrics

Access performance data via:
1. **Debug Dashboard**: Real-time performance monitoring
2. **Log Files**: `logs/performance/performance.log`
3. **Session State**: `last_model_response_time`, `last_interaction_time`

## 🚨 Error Handling

### Error Categories

- **DurgasAIError**: Base exception with metadata
- **ModelError**: AI model related errors
- **APIError**: API communication errors
- **ConfigurationError**: Configuration and setup errors
- **ValidationError**: Input validation errors

### Error Handling Best Practices

```python
from utils.error_handler import ErrorHandler, ModelError

# Use error decorators
@ErrorHandler.handle_model_errors
def model_operation():
    # Your model code here
    pass

# Use error boundaries
with StreamlitErrorHandler.error_boundary("Model Loading"):
    # Code that might fail
    load_model()

# Manual error handling
try:
    risky_operation()
except Exception as e:
    ErrorHandler.log_error(e, {"context": "additional_info"})
    ErrorHandler.display_error_message(e, show_details=True)
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Run all tests
python -m pytest test/

# Run specific test file
python test_app.py

# Run with coverage
python -m pytest test/ --cov=utils --cov=page
```

### Validation Functions

```python
from utils.error_handler import Validators

# Validate API tokens
if Validators.is_valid_api_token(token):
    # Token is valid
    pass

# Validate user input
if Validators.is_safe_input(user_input):
    # Input is safe to process
    pass
```

## 📝 Code Standards

### Documentation Requirements

1. **Module Docstrings**: Comprehensive module description
2. **Class Docstrings**: Class purpose, attributes, and usage
3. **Method Docstrings**: Parameters, returns, and behavior
4. **Inline Comments**: Explain complex logic and decisions
5. **Type Hints**: All function parameters and returns

### Logging Requirements

1. **Debug Logs**: Detailed operation information
2. **Info Logs**: General application flow
3. **Warning Logs**: Non-critical issues
4. **Error Logs**: Error details with context
5. **Performance Logs**: Timing for critical operations

### Example Function Template

```python
from utils.logger import debug, info, warning, error, time_operation, LoggedOperation

@time_operation("function_name", "component")
def example_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Example function with comprehensive logging and documentation.
    
    This function demonstrates best practices for:
    - Parameter validation and type hints
    - Comprehensive logging at each step
    - Error handling with context
    - Performance monitoring
    - Structured return values
    
    Args:
        param1 (str): Description of parameter 1
        param2 (int, optional): Description of parameter 2. Defaults to 10.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - result: The main result
            - metadata: Additional information
            - success: Boolean success indicator
            
    Raises:
        ValueError: If param1 is invalid
        RuntimeError: If operation fails
    """
    debug(f"Starting example_function", "component", {
        "param1_length": len(param1),
        "param2": param2
    })
    
    try:
        with LoggedOperation("example_operation", "component", extra_data={"param1": param1}):
            
            # Step 1: Validate inputs
            if not param1 or not param1.strip():
                raise ValueError("param1 cannot be empty")
            
            # Step 2: Process inputs
            result = f"Processed: {param1} with {param2}"
            
            # Step 3: Return structured result
            info("Example function completed successfully", "component", {
                "result_length": len(result)
            })
            
            return {
                "result": result,
                "metadata": {"param2_used": param2},
                "success": True
            }
            
    except Exception as e:
        error("Example function failed", "component", e, {
            "param1": param1,
            "param2": param2
        })
        raise
```

## 🔍 Debugging Workflows

### Common Debugging Scenarios

#### 1. Model Loading Issues

```bash
# Check model logs
tail -f logs/debug/models.log

# Check API connectivity
curl -H "Authorization: Bearer YOUR_TOKEN" https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2
```

#### 2. Session State Issues

- Use Debug Dashboard → Session State tab
- Check `logs/sessions/session_*.log`
- Verify session initialization in `logs/debug/debug.log`

#### 3. Performance Issues

- Monitor Debug Dashboard → Performance tab
- Check `logs/performance/performance.log`
- Use browser dev tools for frontend performance

#### 4. Tool Execution Issues

- Check `logs/debug/tools.log`
- Verify tool configurations in `output/tools/`
- Test tools individually with debug output

### Debugging Commands

```bash
# Monitor all logs in real-time
tail -f logs/*.log logs/debug/*.log logs/performance/*.log

# Search for specific errors
grep -r "ERROR" logs/

# Monitor performance
grep "PERFORMANCE" logs/performance/performance.log | tail -20

# Check session events
grep "SESSION_EVENT" logs/sessions/session_*.log | tail -10
```

## 🚀 Performance Optimization

### Monitoring Performance

1. **Response Times**: Monitor AI model response generation
2. **Memory Usage**: Track session state and model memory
3. **Tool Execution**: Monitor individual tool performance
4. **System Resources**: CPU, memory, disk usage

### Optimization Strategies

1. **Model Selection**: Choose appropriate models for tasks
2. **Caching**: Leverage Streamlit caching for expensive operations
3. **Session Management**: Limit message history and clean up unused data
4. **Resource Monitoring**: Use Debug Dashboard to identify bottlenecks

## 🧪 Testing

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **Performance Tests**: Response time and resource usage
4. **Error Handling Tests**: Error recovery and user experience

### Running Tests

```bash
# Run all tests with logging
python test_app.py

# Run specific test categories
python -c "from test.test_app import test_model_manager; test_model_manager()"

# Test tool execution
python output/tools/code/search_web.py
```

## 📋 Contribution Guidelines

### Before Contributing

1. **Review Architecture**: Read `ARCHITECTURE.md`
2. **Understand Logging**: Familiarize yourself with the logging system
3. **Check Debug Dashboard**: Understand current system state
4. **Run Tests**: Ensure existing functionality works

### Code Review Checklist

- [ ] Comprehensive docstrings and comments
- [ ] Appropriate logging at all levels
- [ ] Error handling with context
- [ ] Performance monitoring for critical operations
- [ ] Type hints for all parameters
- [ ] Input validation and sanitization
- [ ] Test coverage for new functionality
- [ ] Documentation updates

### Pull Request Requirements

1. **Logging**: All new code must include appropriate logging
2. **Documentation**: Update relevant documentation files
3. **Testing**: Include tests for new functionality
4. **Performance**: Monitor impact on system performance
5. **Error Handling**: Implement comprehensive error handling

## 🔮 Future Enhancements

### Planned Debugging Features

1. **Real-time Monitoring Dashboard**: Live system metrics
2. **Advanced Analytics**: User behavior analysis
3. **Automated Error Recovery**: Self-healing capabilities
4. **Performance Profiling**: Detailed performance analysis
5. **Log Analysis Tools**: Automated log pattern recognition

### Development Roadmap

1. **Enhanced Tool System**: More sophisticated tool chaining
2. **Advanced Model Management**: Multi-model conversations
3. **Improved Performance**: Async operations and caching
4. **Better Error Recovery**: Automated error resolution
5. **Advanced Analytics**: ML-powered usage analysis

---

This development guide should be updated as new features and debugging capabilities are added to the system.
