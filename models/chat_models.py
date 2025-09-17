"""
Chat Models and Data Structures for AI Agent Dashboard

This module provides Pydantic models and data structures for the chat interface,
including structured output schemas, built-in tool functions, and message types.

Features:
- Structured output models for various use cases
- Built-in tool function definitions
- Message and response data structures
- Type definitions for chat interactions
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
import psutil
import re
import uuid


# ==============================================================================
# STRUCTURED OUTPUT MODELS
# ==============================================================================

class CodeResponse(BaseModel):
    """Structured response for code generation"""
    code: str = Field(description="The generated code")
    language: str = Field(description="Programming language")
    explanation: str = Field(description="Explanation of the code")
    dependencies: List[str] = Field(default=[], description="Required dependencies")


class AnalysisResponse(BaseModel):
    """Structured response for data analysis"""
    summary: str = Field(description="Summary of the analysis")
    key_findings: List[str] = Field(description="Key findings from the analysis")
    recommendations: List[str] = Field(description="Recommendations based on analysis")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)


class TaskResponse(BaseModel):
    """Structured response for task breakdown"""
    task_name: str = Field(description="Name of the task")
    steps: List[str] = Field(description="Step-by-step breakdown")
    estimated_time: str = Field(description="Estimated time to complete")
    difficulty: str = Field(description="Difficulty level (Easy/Medium/Hard)")
    resources: List[str] = Field(default=[], description="Required resources")


# ==============================================================================
# MESSAGE AND RESPONSE DATA STRUCTURES
# ==============================================================================

class ChatMessage(BaseModel):
    """Standard chat message structure"""
    role: Literal["user", "assistant", "system", "tool"] = Field(description="Message role")
    content: str = Field(description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    images: Optional[List[str]] = Field(default=None, description="Base64 encoded images")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique message ID")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid timestamp format")
        return v


class ToolCall(BaseModel):
    """Tool call structure"""
    function: Dict[str, Any] = Field(description="Function call details")
    tool_name: Optional[str] = Field(default=None, description="Tool name for display")
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique call ID")
    status: Literal["pending", "running", "completed", "failed"] = Field(default="pending", description="Call status")
    result: Optional[Any] = Field(default=None, description="Call result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    
    @validator('function')
    def validate_function(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Function must be a dictionary")
        if 'name' not in v:
            raise ValueError("Function must have a 'name' field")
        return v


class ChatResponse(BaseModel):
    """Complete chat response structure"""
    content: str = Field(description="Main response content")
    thinking: Optional[str] = Field(default=None, description="Thinking process if enabled")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Tool calls made")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured output data")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Response metadata")
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique response ID")
    model_used: Optional[str] = Field(default=None, description="Model used for generation")
    tokens_used: Optional[int] = Field(default=None, description="Number of tokens used")
    generation_time: Optional[float] = Field(default=None, description="Generation time in seconds")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Response content cannot be empty")
        return v.strip()


class InteractionMetrics(BaseModel):
    """Performance and interaction metrics"""
    interaction_id: str = Field(description="Unique interaction identifier")
    start_time: float = Field(description="Interaction start timestamp")
    end_time: Optional[float] = Field(default=None, description="Interaction end timestamp")
    duration_ms: Optional[float] = Field(default=None, description="Total duration in milliseconds")
    stage_timings: Optional[Dict[str, float]] = Field(default=None, description="Stage-specific timings")
    features_used: Optional[Dict[str, bool]] = Field(default=None, description="Features enabled")
    context_info: Optional[Dict[str, Any]] = Field(default=None, description="Context information")


# ==============================================================================
# BUILT-IN TOOL FUNCTIONS
# ==============================================================================

def get_current_time() -> str:
    """Get the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_math(expression: str) -> str:
    """
    Calculate a mathematical expression safely
    
    Args:
        expression (str): Mathematical expression to evaluate
    
    Returns:
        str: Result of the calculation or error message
    """
    try:
        # Simple safety check - only allow basic math operations
        allowed_chars = set('0123456789+-*/().= ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web for information (placeholder implementation)
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
    
    Returns:
        str: Search results summary
    """
    return f"Web search for '{query}' - This is a placeholder. In a real implementation, this would use a web search API."


def system_info() -> Dict[str, Any]:
    """
    Get current system information
    
    Returns:
        Dict[str, Any]: System information
    """
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "process_count": len(psutil.pids())
    }


# ==============================================================================
# TOOL REGISTRY
# ==============================================================================

# Built-in tools registry
BUILT_IN_TOOLS = {
    'get_current_time': {
        'function': get_current_time,
        'description': 'Get the current date and time',
        'parameters': {}
    },
    'calculate_math': {
        'function': calculate_math,
        'description': 'Calculate a mathematical expression safely',
        'parameters': {
            'expression': {'type': 'string', 'description': 'Mathematical expression to evaluate'}
        }
    },
    'search_web': {
        'function': search_web,
        'description': 'Search the web for information',
        'parameters': {
            'query': {'type': 'string', 'description': 'Search query'},
            'max_results': {'type': 'integer', 'description': 'Maximum number of results', 'default': 3}
        }
    },
    'system_info': {
        'function': system_info,
        'description': 'Get current system information',
        'parameters': {}
    }
}


# ==============================================================================
# CONFIGURATION MODELS
# ==============================================================================

class ChatConfig(BaseModel):
    """Chat interface configuration"""
    enable_tools: bool = Field(default=True, description="Enable tool execution")
    enable_vector_search: bool = Field(default=True, description="Enable vector database search")
    enable_multimodal: bool = Field(default=True, description="Enable image input support")
    enable_structured_outputs: bool = Field(default=True, description="Enable structured outputs")
    enable_thinking_mode: bool = Field(default=True, description="Enable thinking mode")
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    default_max_tokens: int = Field(default=2048, description="Default maximum tokens")
    default_temperature: float = Field(default=0.7, description="Default temperature")
    stream_responses: bool = Field(default=True, description="Enable response streaming")
    max_image_size: int = Field(default=10, description="Maximum image size in MB")
    supported_image_formats: List[str] = Field(
        default=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        description="Supported image formats"
    )


class ChatSession(BaseModel):
    """Chat session configuration"""
    session_id: str = Field(description="Unique session identifier")
    model: str = Field(description="Selected model")
    temperature: float = Field(description="Temperature setting")
    max_tokens: int = Field(description="Maximum tokens setting")
    features: ChatConfig = Field(description="Enabled features")
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_available_structured_models() -> Dict[str, BaseModel]:
    """Get available structured output models"""
    return {
        'Code Generation': CodeResponse,
        'Data Analysis': AnalysisResponse,
        'Task Breakdown': TaskResponse,
    }


def create_tool_schema(tool_name: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create Ollama-compatible tool schema"""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_info['description'],
            "parameters": {
                "type": "object",
                "properties": tool_info.get('parameters', {}),
                "required": list(tool_info.get('parameters', {}).keys())
            }
        }
    }


def validate_message_structure(message: Dict[str, Any]) -> bool:
    """Validate chat message structure"""
    required_fields = ['role', 'content']
    return all(field in message for field in required_fields)


def create_error_message(error: str, context: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """Create standardized error message"""
    return ChatMessage(
        role="assistant",
        content=f"❌ Error: {error}",
        metadata={
            "error": True,
            "error_type": "system_error",
            "context": context or {}
        }
    )


# ==============================================================================
# ENHANCED CHAT MODELS
# ==============================================================================

class ChatSession(BaseModel):
    """Enhanced chat session with better tracking"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    model: str = Field(description="Selected model")
    temperature: float = Field(ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: int = Field(ge=1, le=32000, description="Maximum tokens setting")
    features: ChatConfig = Field(description="Enabled features")
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = Field(default=0, description="Total messages in session")
    total_tokens: int = Field(default=0, description="Total tokens used")
    is_active: bool = Field(default=True, description="Whether session is active")
    
    @validator('model')
    def validate_model(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ChatHistory(BaseModel):
    """Chat history container with enhanced functionality"""
    messages: List[ChatMessage] = Field(default=[], description="Chat messages")
    session_id: str = Field(description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    total_messages: int = Field(default=0, description="Total message count")
    user_messages: int = Field(default=0, description="User message count")
    assistant_messages: int = Field(default=0, description="Assistant message count")
    
    def add_message(self, message: ChatMessage = None):
        """Add message to history"""
        if message is None:
            return
        
        self.messages.append(message)
        self.total_messages += 1
        self.last_updated = datetime.now()
        
        if message.role == "user":
            self.user_messages += 1
        elif message.role == "assistant":
            self.assistant_messages += 1
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages"""
        return self.messages[-limit:] if self.messages else []
    
    def clear_history(self):
        """Clear chat history"""
        self.messages = []
        self.total_messages = 0
        self.user_messages = 0
        self.assistant_messages = 0
        self.last_updated = datetime.now()


class ModelCapabilities(BaseModel):
    """Model capabilities and limitations"""
    model_name: str = Field(description="Model name")
    max_tokens: int = Field(description="Maximum tokens supported")
    supports_tools: bool = Field(default=False, description="Supports tool calling")
    supports_vision: bool = Field(default=False, description="Supports image input")
    supports_streaming: bool = Field(default=False, description="Supports streaming responses")
    supports_structured_output: bool = Field(default=False, description="Supports structured output")
    supported_languages: List[str] = Field(default=[], description="Supported languages")
    cost_per_token: Optional[float] = Field(default=None, description="Cost per token")
    context_window: int = Field(description="Context window size")


class ChatAnalytics(BaseModel):
    """Chat analytics and metrics"""
    session_id: str = Field(description="Session identifier")
    total_interactions: int = Field(default=0, description="Total interactions")
    average_response_time: float = Field(default=0.0, description="Average response time")
    tool_usage_count: int = Field(default=0, description="Number of tool calls")
    error_count: int = Field(default=0, description="Number of errors")
    user_satisfaction: Optional[float] = Field(default=None, description="User satisfaction score")
    most_used_tools: List[str] = Field(default=[], description="Most frequently used tools")
    conversation_topics: List[str] = Field(default=[], description="Main conversation topics")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)


# ==============================================================================
# ENHANCED UTILITY FUNCTIONS
# ==============================================================================

def create_chat_session(model: str, temperature: float = 0.7, max_tokens: int = 2048,
                       user_id: str = None) -> ChatSession:
    """Create a new chat session"""
    config = ChatConfig()
    return ChatSession(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        features=config,
        user_id=user_id
    )


def create_chat_history(session_id: str) -> ChatHistory:
    """Create a new chat history"""
    return ChatHistory(session_id=session_id)


def validate_chat_message(message_data: Dict[str, Any]) -> bool:
    """Validate chat message data"""
    try:
        ChatMessage(**message_data)
        return True
    except Exception:
        return False


def create_success_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """Create standardized success message"""
    return ChatMessage(
        role="assistant",
        content=content,
        metadata={
            "success": True,
            "message_type": "success",
            **(metadata or {})
        }
    )


def create_info_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """Create standardized info message"""
    return ChatMessage(
        role="assistant",
        content=f"ℹ️ {content}",
        metadata={
            "info": True,
            "message_type": "info",
            **(metadata or {})
        }
    )


def create_warning_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """Create standardized warning message"""
    return ChatMessage(
        role="assistant",
        content=f"⚠️ {content}",
        metadata={
            "warning": True,
            "message_type": "warning",
            **(metadata or {})
        }
    )


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text"""
    mention_pattern = r'@(\w+)'
    return re.findall(mention_pattern, text)


def extract_hashtags(text: str) -> List[str]:
    """Extract #hashtags from text"""
    hashtag_pattern = r'#(\w+)'
    return re.findall(hashtag_pattern, text)


def sanitize_message_content(content: str) -> str:
    """Sanitize message content for security"""
    # Remove potential XSS attempts
    content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
    content = re.sub(r'on\w+\s*=', '', content, flags=re.IGNORECASE)
    
    # Limit length
    if len(content) > 10000:
        content = content[:10000] + "..."
    
    return content.strip()
