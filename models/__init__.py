"""
DurgasAI Models Package

This package provides comprehensive Pydantic models for the DurgasAI application,
including chat interfaces, logging, structured outputs, tools, and workflows.

Enhanced Features:
- Comprehensive validation and error handling
- Security analysis and validation
- Performance metrics and analytics
- Advanced workflow management
- Structured output generation
- Log analysis and filtering
"""

# Import all model classes for easy access
from .chat_models import (
    # Basic models
    ChatMessage, ToolCall, ChatResponse, InteractionMetrics,
    ChatConfig, ChatSession, ChatHistory,
    
    # Enhanced models
    ModelCapabilities, ChatAnalytics,
    
    # Utility functions
    create_chat_session, create_chat_history, validate_chat_message,
    create_success_message, create_info_message, create_warning_message,
    create_error_message, extract_mentions, extract_hashtags, sanitize_message_content
)

from .logger_models import (
    # Basic models
    LogLevel, LogEntry, LogAnalysisResult, LogExportData, LogConfiguration,
    LogStatistics, LogFilter, LogArchiveInfo,
    
    # Utility functions
    create_log_entry, create_log_filter, create_default_log_config,
    validate_log_entry, parse_log_line, parse_log_file,
    filter_logs_by_level, filter_logs_by_time_range, filter_logs_by_logger,
    filter_logs_by_message_pattern, get_log_statistics,
    export_logs_to_json, export_logs_to_csv
)

from .structured_models import (
    # Basic models
    PersonInfo, FriendInfo, FriendList,
    DetectedObject, ImageDescription, VisualContentAnalysis,
    DocumentImageAnalysis, ChartDataExtraction, MedicalImageAnalysis,
    SentimentAnalysis, CodeAnalysis, DocumentSummary, QuestionAnswer,
    TranslationResult, DataExtractionResult, ProductReview, MeetingNotes,
    ComprehensiveAnalysis, ConversationSummary, ValidationResult, ComparisonResult,
    
    # Enhanced models
    APIResponse, UserProfile, FileMetadata, NotificationData,
    SearchResult, SearchResponse, PerformanceMetrics, SystemHealth,
    ConfigurationSchema, AuditLog,
    
    # Registry and utilities
    MODEL_REGISTRY, MODEL_CATEGORIES,
    get_model_by_name, list_available_models, get_model_info,
    get_models_by_category, get_all_categories,
    create_api_response, create_success_response, create_error_response,
    validate_structured_output, convert_to_structured_output,
    extract_structured_fields, get_model_field_info, generate_model_example
)

from .tool_models import (
    # Basic models
    ToolStatus, ExecutionStatus, SecurityLevel, ToolParameter, ToolDefinitionJson,
    ToolDefinition, ToolExecution, ToolExecutionHistory, ToolSecurityReport,
    ToolConfiguration, ToolRegistry, ToolExecutionResult,
    
    # Utility functions
    convert_json_to_tool_definition, create_tool_definition, create_execution_record,
    validate_tool_code, generate_security_report, get_default_tool_configuration
)

from .workflow_models import (
    # Basic models
    WorkflowStatus, ExecutionStatus as WorkflowExecutionStatus, StepInputType,
    WorkflowStep, WorkflowDefinition, WorkflowExecution, WorkflowExecutionHistory,
    StepDependency, WorkflowConfiguration, WorkflowRegistry, WorkflowExecutionResult,
    StepExecutionResult,
    
    # Enhanced models
    WorkflowTemplate, WorkflowSchedule, WorkflowTrigger, WorkflowMetrics,
    WorkflowDependencyGraph, WorkflowExecutionPlan,
    
    # Utility functions
    create_workflow_definition, create_workflow_step, create_execution_record as create_workflow_execution_record,
    validate_workflow_structure, get_default_workflow_configuration, create_workflow_registry,
    create_workflow_template, create_workflow_schedule, create_workflow_trigger,
    build_dependency_graph, topological_sort, find_parallel_groups, calculate_critical_path,
    validate_workflow_dependencies, optimize_workflow_execution
)

# Version information
__version__ = "1.0.0"
__author__ = "DurgasAI Team"

# Export all public classes and functions
__all__ = [
    # Chat models
    "ChatMessage", "ToolCall", "ChatResponse", "InteractionMetrics",
    "ChatConfig", "ChatSession", "ChatHistory", "ModelCapabilities", "ChatAnalytics",
    
    # Logger models
    "LogLevel", "LogEntry", "LogAnalysisResult", "LogExportData", "LogConfiguration",
    "LogStatistics", "LogFilter", "LogArchiveInfo",
    
    # Structured models
    "PersonInfo", "FriendInfo", "FriendList", "DetectedObject", "ImageDescription",
    "VisualContentAnalysis", "DocumentImageAnalysis", "ChartDataExtraction", "MedicalImageAnalysis",
    "SentimentAnalysis", "CodeAnalysis", "DocumentSummary", "QuestionAnswer",
    "TranslationResult", "DataExtractionResult", "ProductReview", "MeetingNotes",
    "ComprehensiveAnalysis", "ConversationSummary", "ValidationResult", "ComparisonResult",
    "APIResponse", "UserProfile", "FileMetadata", "NotificationData",
    "SearchResult", "SearchResponse", "PerformanceMetrics", "SystemHealth",
    "ConfigurationSchema", "AuditLog",
    
    # Tool models
    "ToolStatus", "ExecutionStatus", "SecurityLevel", "ToolParameter", "ToolDefinitionJson",
    "ToolDefinition", "ToolExecution", "ToolExecutionHistory", "ToolSecurityReport",
    "ToolConfiguration", "ToolRegistry", "ToolExecutionResult",
    
    # Workflow models
    "WorkflowStatus", "WorkflowExecutionStatus", "StepInputType", "WorkflowStep",
    "WorkflowDefinition", "WorkflowExecution", "WorkflowExecutionHistory", "StepDependency",
    "WorkflowConfiguration", "WorkflowRegistry", "WorkflowExecutionResult", "StepExecutionResult",
    "WorkflowTemplate", "WorkflowSchedule", "WorkflowTrigger", "WorkflowMetrics",
    "WorkflowDependencyGraph", "WorkflowExecutionPlan",
    
    # Registry and utilities
    "MODEL_REGISTRY", "MODEL_CATEGORIES",
    
    # Utility functions
    "create_chat_session", "create_chat_history", "validate_chat_message",
    "create_success_message", "create_info_message", "create_warning_message",
    "create_error_message", "extract_mentions", "extract_hashtags", "sanitize_message_content",
    "create_log_entry", "create_log_filter", "create_default_log_config",
    "validate_log_entry", "parse_log_line", "parse_log_file",
    "filter_logs_by_level", "filter_logs_by_time_range", "filter_logs_by_logger",
    "filter_logs_by_message_pattern", "get_log_statistics",
    "export_logs_to_json", "export_logs_to_csv",
    "get_model_by_name", "list_available_models", "get_model_info",
    "get_models_by_category", "get_all_categories",
    "create_api_response", "create_success_response", "create_error_response",
    "validate_structured_output", "convert_to_structured_output",
    "extract_structured_fields", "get_model_field_info", "generate_model_example",
    "convert_json_to_tool_definition", "create_tool_definition", "create_execution_record",
    "validate_tool_code", "generate_security_report", "get_default_tool_configuration",
    "create_workflow_definition", "create_workflow_step", "create_workflow_execution_record",
    "validate_workflow_structure", "get_default_workflow_configuration", "create_workflow_registry",
    "create_workflow_template", "create_workflow_schedule", "create_workflow_trigger",
    "build_dependency_graph", "topological_sort", "find_parallel_groups", "calculate_critical_path",
    "validate_workflow_dependencies", "optimize_workflow_execution"
]
