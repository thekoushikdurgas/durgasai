"""
Model Service for DurgasAI.

Provides centralized model management and AI operations abstraction.
This service encapsulates all model-related functionality including:
- Model loading and initialization
- Response generation and conversation management
- Model configuration and parameter management
- Performance monitoring and error handling
- Provider abstraction (HuggingFace API, Local models)

Architecture:
- Service layer abstraction over utils.model_manager
- Centralized configuration management via ConfigService
- Comprehensive logging and error handling
- Performance monitoring and analytics integration
- Support for multiple model providers and types

Key Features:
- Model lifecycle management (load, unload, switch)
- Conversation context management
- Response generation with error handling
- Model status and health monitoring
- Configuration-driven model parameters
- Async operations support for better UX
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import sys
from pathlib import Path
import time

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_manager import ModelManager, ModelResponse, ModelConfig
from utils.logger import debug, info, warning, error, log_user_action, log_model_operation, LoggedOperation
from .config_service import ConfigService


@dataclass
class ModelStatus:
    """Model status information."""
    model_id: str
    status: str  # 'loading', 'ready', 'error', 'unloaded'
    provider: str
    loaded_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None


@dataclass
class ConversationContext:
    """Conversation context management."""
    session_id: str
    messages: List[Dict[str, Any]]
    system_prompt: Optional[str] = None
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelService:
    """
    Service for managing AI models and conversations.
    
    This service provides a high-level interface for all model operations,
    abstracting the complexity of model management and providing a clean API
    for the application layers.
    
    Key Responsibilities:
    - Model lifecycle management
    - Conversation context management
    - Response generation coordination
    - Performance monitoring
    - Error handling and recovery
    - Configuration management
    """
    
    def __init__(self, config_service: ConfigService):
        """
        Initialize the model service.
        
        Args:
            config_service: Configuration service instance for model configs
        """
        debug("Initializing ModelService", "model_service")
        
        self.config_service = config_service
        self.model_manager = ModelManager()
        self.current_model_status: Optional[ModelStatus] = None
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.model_cache: Dict[str, ModelConfig] = {}
        
        # Load model configurations
        self._load_model_configurations()
        
        info("ModelService initialized successfully", "model_service")
    
    def _load_model_configurations(self) -> None:
        """Load model configurations from config service."""
        debug("Loading model configurations", "model_service")
        
        try:
            model_configs = self.config_service.get_model_config()
            
            for model_id, config_data in model_configs.items():
                try:
                    model_config = ModelConfig(
                        name=config_data.get('name', model_id),
                        model_id=model_id,
                        provider=config_data.get('provider', 'huggingface_api'),
                        description=config_data.get('description', ''),
                        max_tokens=config_data.get('max_tokens', 512),
                        temperature=config_data.get('temperature', 0.7),
                        top_p=config_data.get('top_p', 0.9),
                        repetition_penalty=config_data.get('repetition_penalty', 1.0)
                    )
                    self.model_cache[model_id] = model_config
                    debug(f"Loaded model configuration: {model_id}", "model_service")
                    
                except Exception as e:
                    warning(f"Failed to load model config for {model_id}", "model_service", error_obj=e)
            
            info(f"Loaded {len(self.model_cache)} model configurations", "model_service")
            
        except Exception as e:
            error("Failed to load model configurations", "model_service", error_obj=e)
            self.model_cache = {}
    
    async def load_model(self, model_id: str, api_token: Optional[str] = None) -> ModelStatus:
        """
        Load a model asynchronously.
        
        Args:
            model_id: ID of the model to load
            api_token: Optional API token for cloud models
            
        Returns:
            ModelStatus: Status of the loaded model
        """
        debug(f"Loading model: {model_id}", "model_service")
        
        with LoggedOperation("model_loading", "model_service"):
            try:
                # Get model configuration
                if model_id not in self.model_cache:
                    error(f"Model configuration not found: {model_id}", "model_service")
                    return ModelStatus(
                        model_id=model_id,
                        status='error',
                        provider='unknown',
                        error_message=f"Configuration not found for model: {model_id}"
                    )
                
                model_config = self.model_cache[model_id]
                
                # Update current status
                self.current_model_status = ModelStatus(
                    model_id=model_id,
                    status='loading',
                    provider=model_config.provider.value,
                    loaded_at=datetime.now()
                )
                
                # Load the model using model manager
                success = await asyncio.to_thread(
                    self.model_manager.setup_model,
                    model_config,
                    api_token
                )
                
                if success:
                    self.current_model_status.status = 'ready'
                    self.current_model_status.last_used = datetime.now()
                    
                    log_model_operation("model_loaded", model=model_id)
                    info(f"Model loaded successfully: {model_id}", "model_service")
                else:
                    self.current_model_status.status = 'error'
                    self.current_model_status.error_message = "Failed to load model"
                    
                    error(f"Failed to load model: {model_id}", "model_service")
                
                return self.current_model_status
                
            except Exception as e:
                error(f"Exception during model loading: {model_id}", "model_service", error_obj=e)
                
                return ModelStatus(
                    model_id=model_id,
                    status='error',
                    provider='unknown',
                    error_message=str(e)
                )
    
    def get_model_status(self) -> Optional[ModelStatus]:
        """Get current model status."""
        debug("Retrieving model status", "model_service")
        return self.current_model_status
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        debug("Retrieving available models", "model_service")
        
        models = []
        for model_id, config in self.model_cache.items():
            models.append({
                'id': model_id,
                'name': config.name,
                'provider': config.provider.value,
                'description': config.description,
                'parameters': {
                    'max_tokens': config.max_tokens,
                    'temperature': config.temperature,
                    'top_p': config.top_p,
                    'repetition_penalty': config.repetition_penalty
                }
            })
        
        debug(f"Retrieved {len(models)} available models", "model_service")
        return models
    
    async def generate_response(
        self,
        user_input: str,
        session_id: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate AI response with conversation context.
        
        Args:
            user_input: User's input message
            session_id: Session identifier for conversation context
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters for generation
            
        Returns:
            ModelResponse: Generated response with metadata
        """
        debug(f"Generating response for session: {session_id}", "model_service")
        
        with LoggedOperation("response_generation", "model_service"):
            try:
                # Check if model is loaded
                if not self.current_model_status or self.current_model_status.status != 'ready':
                    warning("No model loaded for response generation", "model_service")
                    return ModelResponse(
                        content="",
                        success=False,
                        error="No model is currently loaded. Please load a model first.",
                        metadata={'session_id': session_id}
                    )
                
                # Update conversation context
                self._update_conversation_context(session_id, user_input, system_prompt)
                
                # Generate response using model manager
                start_time = time.time()
                response = await asyncio.to_thread(
                    self.model_manager.generate_response,
                    user_input
                )
                response_time = time.time() - start_time
                
                # Update model status
                self.current_model_status.last_used = datetime.now()
                if not self.current_model_status.performance_metrics:
                    self.current_model_status.performance_metrics = {}
                
                self.current_model_status.performance_metrics['last_response_time'] = response_time
                
                # Update conversation context with response
                if response.success:
                    self._add_assistant_message(session_id, response.content)
                
                # Add performance metadata
                if not response.metadata:
                    response.metadata = {}
                response.metadata.update({
                    'session_id': session_id,
                    'model_id': self.current_model_status.model_id,
                    'response_time': response_time,
                    'timestamp': datetime.now().isoformat()
                })
                
                log_user_action("ai_response_generated",
                    session_id=session_id,
                    model_id=self.current_model_status.model_id,
                    success=response.success,
                    response_time=response_time
                )
                
                debug(f"Response generated successfully in {response_time:.2f}s", "model_service")
                return response
                
            except Exception as e:
                error(f"Exception during response generation", "model_service", error_obj=e)
                
                return ModelResponse(
                    content="",
                    success=False,
                    error=f"Error generating response: {str(e)}",
                    metadata={'session_id': session_id}
                )
    
    def _update_conversation_context(
        self,
        session_id: str,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> None:
        """Update conversation context with user input."""
        debug(f"Updating conversation context: {session_id}", "model_service")
        
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = ConversationContext(
                session_id=session_id,
                messages=[],
                system_prompt=system_prompt,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
        
        context = self.conversation_contexts[session_id]
        context.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        context.last_activity = datetime.now()
        
        if system_prompt:
            context.system_prompt = system_prompt
    
    def _add_assistant_message(self, session_id: str, content: str) -> None:
        """Add assistant message to conversation context."""
        if session_id in self.conversation_contexts:
            context = self.conversation_contexts[session_id]
            context.messages.append({
                'role': 'assistant',
                'content': content,
                'timestamp': datetime.now().isoformat()
            })
            context.last_activity = datetime.now()
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        debug(f"Retrieving conversation context: {session_id}", "model_service")
        return self.conversation_contexts.get(session_id)
    
    def clear_conversation_context(self, session_id: str) -> bool:
        """Clear conversation context for a session."""
        debug(f"Clearing conversation context: {session_id}", "model_service")
        
        if session_id in self.conversation_contexts:
            del self.conversation_contexts[session_id]
            log_user_action("conversation_cleared", session_id=session_id)
            info(f"Conversation context cleared: {session_id}", "model_service")
            return True
        
        return False
    
    def unload_model(self) -> bool:
        """Unload the current model."""
        debug("Unloading current model", "model_service")
        
        try:
            if self.current_model_status:
                model_id = self.current_model_status.model_id
                
                # Update status
                self.current_model_status.status = 'unloaded'
                
                # Clear model from manager (if supported)
                # Note: ModelManager doesn't have explicit unload, but we track status
                
                log_model_operation("model_unloaded", model=model_id)
                info(f"Model unloaded: {model_id}", "model_service")
                
                self.current_model_status = None
                return True
            
            return False
            
        except Exception as e:
            error("Failed to unload model", "model_service", error_obj=e)
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        debug("Retrieving performance metrics", "model_service")
        
        metrics = {
            'total_conversations': len(self.conversation_contexts),
            'available_models': len(self.model_cache),
            'current_model': None,
            'session_stats': {}
        }
        
        if self.current_model_status:
            metrics['current_model'] = {
                'id': self.current_model_status.model_id,
                'status': self.current_model_status.status,
                'provider': self.current_model_status.provider,
                'loaded_at': self.current_model_status.loaded_at.isoformat() if self.current_model_status.loaded_at else None,
                'last_used': self.current_model_status.last_used.isoformat() if self.current_model_status.last_used else None,
                'performance_metrics': self.current_model_status.performance_metrics
            }
        
        # Session statistics
        for session_id, context in self.conversation_contexts.items():
            metrics['session_stats'][session_id] = {
                'message_count': len(context.messages),
                'created_at': context.created_at.isoformat() if context.created_at else None,
                'last_activity': context.last_activity.isoformat() if context.last_activity else None
            }
        
        return metrics
