"""
Business Logic Services for DurgasAI.

This package contains service modules that handle business logic
separate from UI and infrastructure concerns.

Services Architecture:
- ConfigService: Centralized configuration management
- ModelService: AI model management and operations
- ChatService: Chat functionality and conversation management  
- AnalyticsService: Analytics and metrics tracking

Each service follows the same architectural patterns:
- Comprehensive logging and error handling
- Configuration-driven behavior
- Clean separation of concerns
- Async support where appropriate
- Performance monitoring integration
"""

from .config_service import ConfigService
from .model_service import ModelService, ModelStatus, ConversationContext
from .chat_service import ChatService, ChatMessage, ChatSession, ExportOptions
from .analytics_service import AnalyticsService, UserInteraction, PerformanceMetric, ErrorEvent, AnalyticsReport
from .service_manager import ServiceManager

__all__ = [
    'ConfigService',
    'ModelService', 'ModelStatus', 'ConversationContext',
    'ChatService', 'ChatMessage', 'ChatSession', 'ExportOptions', 
    'AnalyticsService', 'UserInteraction', 'PerformanceMetric', 'ErrorEvent', 'AnalyticsReport',
    'ServiceManager'
]
