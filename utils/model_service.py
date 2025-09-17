"""
Model Service Layer for DurgasAI

This module provides high-level services for working with the enhanced model classes,
including validation, serialization, and integration with the application.

Features:
- Unified model validation and error handling
- Model serialization and deserialization
- Integration with existing services
- Performance monitoring and analytics
- Security validation and reporting
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Type, TypeVar
from pathlib import Path

from pydantic import BaseModel, ValidationError
from models import (
    # Chat models
    ChatMessage, ChatResponse, ChatSession, ChatHistory, ChatAnalytics,
    create_chat_session, create_chat_history, validate_chat_message,
    create_success_message, create_info_message, create_warning_message, create_error_message,
    
    # Logger models
    LogEntry, LogLevel, LogAnalysisResult, LogStatistics,
    create_log_entry, parse_log_file, get_log_statistics, export_logs_to_json,
    
    # Structured models
    APIResponse, UserProfile, FileMetadata, NotificationData, SearchResult, SearchResponse,
    PerformanceMetrics, SystemHealth, ConfigurationSchema, AuditLog,
    create_api_response, create_success_response, create_error_response,
    validate_structured_output, convert_to_structured_output, get_model_by_name,
    
    # Tool models
    ToolDefinition, ToolExecution, ToolSecurityReport, ToolConfiguration,
    create_tool_definition, validate_tool_code, generate_security_report,
    
    # Workflow models
    WorkflowDefinition, WorkflowExecution, WorkflowTemplate, WorkflowSchedule,
    WorkflowTrigger, WorkflowMetrics, WorkflowDependencyGraph,
    create_workflow_definition, create_workflow_template, create_workflow_schedule,
    create_workflow_trigger, build_dependency_graph, validate_workflow_dependencies
)

T = TypeVar('T', bound=BaseModel)


class ModelValidationError(Exception):
    """Custom exception for model validation errors"""
    def __init__(self, message: str, errors: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.errors = errors or []


class ModelService:
    """High-level service for working with DurgasAI models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_cache = {}
        self.serialization_cache = {}
        
    def validate_model(self, model_class: Type[T], data: Dict[str, Any]) -> T:
        """Validate data against a model class with enhanced error handling"""
        try:
            # Check cache first
            cache_key = f"{model_class.__name__}_{hash(str(data))}"
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
            
            # Validate the data
            instance = model_class(**data)
            
            # Cache the result
            self.validation_cache[cache_key] = instance
            return instance
            
        except ValidationError as e:
            errors = []
            for error in e.errors():
                errors.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input")
                })
            raise ModelValidationError(f"Validation failed for {model_class.__name__}", errors)
        except Exception as e:
            raise ModelValidationError(f"Unexpected error validating {model_class.__name__}: {str(e)}")
    
    def serialize_model(self, instance: BaseModel, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Serialize a model instance to various formats"""
        try:
            if format == "json":
                return instance.model_dump_json(indent=2)
            elif format == "dict":
                return instance.model_dump()
            elif format == "dict_exclude_none":
                return instance.model_dump(exclude_none=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            raise ModelValidationError(f"Serialization failed: {str(e)}")
    
    def deserialize_model(self, model_class: Type[T], data: Union[str, Dict[str, Any]]) -> T:
        """Deserialize data to a model instance"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            return self.validate_model(model_class, data)
        except json.JSONDecodeError as e:
            raise ModelValidationError(f"JSON decode error: {str(e)}")
        except Exception as e:
            raise ModelValidationError(f"Deserialization failed: {str(e)}")
    
    def create_model_from_template(self, model_class: Type[T], template: str = "default") -> T:
        """Create a model instance from a template"""
        try:
            if hasattr(model_class, 'model_fields'):
                # Create instance with default values
                field_values = {}
                for field_name, field_info in model_class.model_fields.items():
                    if hasattr(field_info, 'default') and field_info.default is not None:
                        field_values[field_name] = field_info.default
                    elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                        field_values[field_name] = field_info.default_factory()
                
                return model_class(**field_values)
            else:
                return model_class()
        except Exception as e:
            raise ModelValidationError(f"Template creation failed: {str(e)}")


class ChatModelService:
    """Service for chat-related models"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.active_sessions = {}
        self.chat_histories = {}
    
    def create_session(self, model: str, user_id: str = None, **kwargs) -> ChatSession:
        """Create a new chat session"""
        session = create_chat_session(model, user_id=user_id, **kwargs)
        self.active_sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an active chat session"""
        return self.active_sessions.get(session_id)
    
    def create_message(self, role: str, content: str, **kwargs) -> ChatMessage:
        """Create a new chat message"""
        return ChatMessage(role=role, content=content, **kwargs)
    
    def create_response(self, content: str, **kwargs) -> ChatResponse:
        """Create a chat response"""
        return ChatResponse(content=content, **kwargs)
    
    def add_message_to_history(self, session_id: str, message: ChatMessage = None):
        """Add a message to chat history"""
        if message is None:
            return
        
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = create_chat_history(session_id)
        
        self.chat_histories[session_id].add_message(message)
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """Get chat history for a session"""
        if session_id in self.chat_histories:
            return self.chat_histories[session_id].get_recent_messages(limit)
        return []
    
    def create_standard_messages(self, content: str, message_type: str = "info") -> ChatMessage:
        """Create standardized messages"""
        if message_type == "success":
            return create_success_message(content)
        elif message_type == "info":
            return create_info_message(content)
        elif message_type == "warning":
            return create_warning_message(content)
        elif message_type == "error":
            return create_error_message(content)
        else:
            return self.create_message("assistant", content)


class LoggerModelService:
    """Service for logger-related models"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.log_entries = []
    
    def create_log_entry(self, level: LogLevel, message: str, logger: str = "durgasai", **kwargs) -> LogEntry:
        """Create a new log entry"""
        entry = create_log_entry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            logger=logger,
            **kwargs
        )
        self.log_entries.append(entry)
        return entry
    
    def parse_log_file(self, file_path: str, max_lines: int = 1000) -> List[LogEntry]:
        """Parse a log file"""
        return parse_log_file(file_path, max_lines)
    
    def get_log_statistics(self, entries: List[LogEntry] = None) -> LogStatistics:
        """Get log statistics"""
        if entries is None:
            entries = self.log_entries
        
        stats_data = get_log_statistics(entries)
        return LogStatistics(**stats_data)
    
    def export_logs(self, entries: List[LogEntry], file_path: str, format: str = "json") -> bool:
        """Export logs to file"""
        if format == "json":
            return export_logs_to_json(entries, file_path)
        else:
            return False


class StructuredModelService:
    """Service for structured output models"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
    
    def create_api_response(self, success: bool, message: str, data: Any = None, **kwargs) -> APIResponse:
        """Create an API response"""
        return create_api_response(success, message, data, **kwargs)
    
    def create_success_response(self, message: str, data: Any = None, **kwargs) -> APIResponse:
        """Create a success response"""
        return create_success_response(message, data, **kwargs)
    
    def create_error_response(self, message: str, error_code: str = None, **kwargs) -> APIResponse:
        """Create an error response"""
        return create_error_response(message, error_code, **kwargs)
    
    def validate_structured_data(self, data: Dict[str, Any], model_name: str) -> bool:
        """Validate data against a structured model"""
        return validate_structured_output(data, model_name)
    
    def convert_to_structured(self, data: Dict[str, Any], model_name: str) -> Optional[BaseModel]:
        """Convert data to structured output"""
        return convert_to_structured_output(data, model_name)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        try:
            return get_model_by_name(model_name).model_json_schema()
        except Exception:
            return {}


class ToolModelService:
    """Service for tool-related models"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.tool_definitions = {}
        self.tool_executions = []
    
    def create_tool_definition(self, name: str, description: str, code: str, **kwargs) -> ToolDefinition:
        """Create a new tool definition"""
        tool = create_tool_definition(name, description, code, **kwargs)
        self.tool_definitions[tool.id] = tool
        return tool
    
    def validate_tool_code(self, code: str) -> Dict[str, Any]:
        """Validate tool code for security and syntax"""
        return validate_tool_code(code)
    
    def generate_security_report(self, tool_id: str, code: str) -> ToolSecurityReport:
        """Generate security report for a tool"""
        return generate_security_report(tool_id, code)
    
    def create_tool_execution(self, tool_id: str, tool_name: str, inputs: Dict[str, Any], **kwargs) -> ToolExecution:
        """Create a tool execution record"""
        execution = ToolExecution(
            tool_id=tool_id,
            tool_name=tool_name,
            inputs=inputs,
            **kwargs
        )
        self.tool_executions.append(execution)
        return execution


class WorkflowModelService:
    """Service for workflow-related models"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.workflow_definitions = {}
        self.workflow_executions = []
    
    def create_workflow_definition(self, name: str, description: str, steps: List[Dict[str, Any]], **kwargs) -> WorkflowDefinition:
        """Create a new workflow definition"""
        workflow = create_workflow_definition(name, description, steps, **kwargs)
        self.workflow_definitions[workflow.id] = workflow
        return workflow
    
    def create_workflow_template(self, name: str, description: str, workflow_definition: WorkflowDefinition, **kwargs) -> WorkflowTemplate:
        """Create a workflow template"""
        return create_workflow_template(name, description, workflow_definition, **kwargs)
    
    def create_workflow_schedule(self, workflow_id: str, name: str, cron_expression: str, **kwargs) -> WorkflowSchedule:
        """Create a workflow schedule"""
        return create_workflow_schedule(workflow_id, name, cron_expression, **kwargs)
    
    def create_workflow_trigger(self, workflow_id: str, trigger_type: str, name: str, configuration: Dict[str, Any], **kwargs) -> WorkflowTrigger:
        """Create a workflow trigger"""
        return create_workflow_trigger(workflow_id, trigger_type, name, configuration, **kwargs)
    
    def build_dependency_graph(self, workflow: WorkflowDefinition) -> WorkflowDependencyGraph:
        """Build dependency graph for a workflow"""
        return build_dependency_graph(workflow)
    
    def validate_workflow_dependencies(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow dependencies"""
        return validate_workflow_dependencies(workflow)
    
    def create_workflow_execution(self, workflow_id: str, workflow_name: str, inputs: Dict[str, Any], **kwargs) -> WorkflowExecution:
        """Create a workflow execution record"""
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            inputs=inputs,
            **kwargs
        )
        self.workflow_executions.append(execution)
        return execution


class ModelServiceManager:
    """Main service manager for all model services"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_service = ModelService(config)
        
        # Initialize specialized services
        self.chat_service = ChatModelService(self.model_service)
        self.logger_service = LoggerModelService(self.model_service)
        self.structured_service = StructuredModelService(self.model_service)
        self.tool_service = ToolModelService(self.model_service)
        self.workflow_service = WorkflowModelService(self.model_service)
    
    def get_service(self, service_type: str):
        """Get a specific service by type"""
        services = {
            "chat": self.chat_service,
            "logger": self.logger_service,
            "structured": self.structured_service,
            "tool": self.tool_service,
            "workflow": self.workflow_service
        }
        return services.get(service_type)
    
    def validate_model(self, model_class: Type[T], data: Dict[str, Any]) -> T:
        """Validate data against a model class"""
        return self.model_service.validate_model(model_class, data)
    
    def serialize_model(self, instance: BaseModel, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Serialize a model instance"""
        return self.model_service.serialize_model(instance, format)
    
    def deserialize_model(self, model_class: Type[T], data: Union[str, Dict[str, Any]]) -> T:
        """Deserialize data to a model instance"""
        return self.model_service.deserialize_model(model_class, data)
    
    def create_model_from_template(self, model_class: Type[T], template: str = "default") -> T:
        """Create a model instance from a template"""
        return self.model_service.create_model_from_template(model_class, template)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        return self.structured_service.get_model_info(model_name)
    
    def create_api_response(self, success: bool, message: str, data: Any = None, **kwargs) -> APIResponse:
        """Create an API response"""
        return self.structured_service.create_api_response(success, message, data, **kwargs)
    
    def create_success_response(self, message: str, data: Any = None, **kwargs) -> APIResponse:
        """Create a success response"""
        return self.structured_service.create_success_response(message, data, **kwargs)
    
    def create_error_response(self, message: str, error_code: str = None, **kwargs) -> APIResponse:
        """Create an error response"""
        return self.structured_service.create_error_response(message, error_code, **kwargs)


# Global service manager instance
model_service_manager = ModelServiceManager()


def get_model_service() -> ModelServiceManager:
    """Get the global model service manager"""
    return model_service_manager


def get_chat_service() -> ChatModelService:
    """Get the chat service"""
    return model_service_manager.chat_service


def get_logger_service() -> LoggerModelService:
    """Get the logger service"""
    return model_service_manager.logger_service


def get_structured_service() -> StructuredModelService:
    """Get the structured service"""
    return model_service_manager.structured_service


def get_tool_service() -> ToolModelService:
    """Get the tool service"""
    return model_service_manager.tool_service


def get_workflow_service() -> WorkflowModelService:
    """Get the workflow service"""
    return model_service_manager.workflow_service
