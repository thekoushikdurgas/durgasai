"""
Chat Service for DurgasAI.

Provides centralized chat functionality and conversation management.
This service orchestrates all chat-related operations including:
- Message processing and validation
- Conversation flow management
- Chat history persistence and retrieval
- User interaction tracking
- Export/import functionality
- Real-time chat enhancements

Architecture:
- Service layer abstraction over chat components
- Integration with ModelService for AI responses
- Centralized conversation state management
- Comprehensive logging and analytics
- Support for multiple conversation sessions
- Export/import capabilities

Key Features:
- Message lifecycle management
- Conversation persistence and retrieval
- Real-time typing indicators and enhancements
- Export functionality (JSON, Markdown, etc.)
- Message validation and filtering
- User interaction analytics
- Session management and cleanup
"""

import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import uuid
import asyncio

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation
from .config_service import ConfigService
from .model_service import ModelService, ModelResponse


@dataclass
class ChatMessage:
    """Structured chat message."""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    session_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatSession:
    """Chat session information."""
    session_id: str
    title: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExportOptions:
    """Export configuration options."""
    format: str  # 'json', 'markdown', 'txt', 'csv'
    include_metadata: bool = True
    include_system_messages: bool = True
    date_range: Optional[tuple] = None
    session_filter: Optional[List[str]] = None


class ChatService:
    """
    Service for managing chat functionality and conversations.
    
    This service provides a high-level interface for all chat operations,
    managing conversation state, message processing, and user interactions.
    
    Key Responsibilities:
    - Message processing and validation
    - Conversation session management
    - Chat history persistence
    - Export/import functionality
    - User interaction analytics
    - Real-time enhancements coordination
    """
    
    def __init__(self, config_service: ConfigService, model_service: ModelService):
        """
        Initialize the chat service.
        
        Args:
            config_service: Configuration service instance
            model_service: Model service instance for AI responses
        """
        debug("Initializing ChatService", "chat_service")
        
        self.config_service = config_service
        self.model_service = model_service
        
        # Chat state management
        self.active_sessions: Dict[str, ChatSession] = {}
        self.message_history: Dict[str, List[ChatMessage]] = {}
        self.current_session_id: Optional[str] = None
        
        # Configuration
        self.max_message_length = 4000
        self.max_history_length = 1000
        self.auto_save_interval = 300  # 5 minutes
        
        # Load configuration
        self._load_chat_configuration()
        
        info("ChatService initialized successfully", "chat_service")
    
    def _load_chat_configuration(self) -> None:
        """Load chat configuration from config service."""
        debug("Loading chat configuration", "chat_service")
        
        try:
            app_config = self.config_service.get_app_config()
            chat_config = app_config.get('chat', {})
            
            self.max_message_length = chat_config.get('max_message_length', 4000)
            self.max_history_length = chat_config.get('max_history_length', 1000)
            self.auto_save_interval = chat_config.get('auto_save_interval', 300)
            
            debug("Chat configuration loaded", "chat_service",
                  max_message_length=self.max_message_length,
                  max_history_length=self.max_history_length)
            
        except Exception as e:
            warning("Failed to load chat configuration, using defaults", "chat_service", error_obj=e)
    
    def create_session(
        self,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            title: Optional session title
            system_prompt: Optional system prompt
            model_id: Optional model ID to use
            
        Returns:
            str: Session ID
        """
        debug("Creating new chat session", "chat_service")
        
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        session = ChatSession(
            session_id=session_id,
            title=title or f"Chat Session {current_time.strftime('%Y-%m-%d %H:%M')}",
            created_at=current_time,
            last_activity=current_time,
            message_count=0,
            model_id=model_id,
            system_prompt=system_prompt
        )
        
        self.active_sessions[session_id] = session
        self.message_history[session_id] = []
        self.current_session_id = session_id
        
        log_user_action("chat_session_created",
            session_id=session_id,
            title=session.title,
            model_id=model_id
        )
        
        info(f"Chat session created: {session_id}", "chat_service")
        return session_id
    
    def get_current_session_id(self) -> Optional[str]:
        """Get current active session ID."""
        return self.current_session_id
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session."""
        debug(f"Setting current session: {session_id}", "chat_service")
        
        if session_id in self.active_sessions:
            self.current_session_id = session_id
            
            log_user_action("chat_session_switched", session_id=session_id)
            info(f"Current session set to: {session_id}", "chat_service")
            return True
        
        warning(f"Session not found: {session_id}", "chat_service")
        return False
    
    def add_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Add a message to the conversation.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            session_id: Optional session ID (uses current if not provided)
            metadata: Optional message metadata
            
        Returns:
            ChatMessage: Created message object
        """
        debug(f"Adding message: {role}", "chat_service")
        
        # Use current session if not specified
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.active_sessions:
            error("No valid session for adding message", "chat_service")
            raise ValueError("No active session available")
        
        # Validate message content
        if len(content) > self.max_message_length:
            warning(f"Message truncated: {len(content)} > {self.max_message_length}", "chat_service")
            content = content[:self.max_message_length] + "... [truncated]"
        
        # Create message
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata=metadata
        )
        
        # Add to history
        if session_id not in self.message_history:
            self.message_history[session_id] = []
        
        self.message_history[session_id].append(message)
        
        # Update session info
        session = self.active_sessions[session_id]
        session.last_activity = message.timestamp
        session.message_count += 1
        
        # Trim history if too long
        if len(self.message_history[session_id]) > self.max_history_length:
            removed_count = len(self.message_history[session_id]) - self.max_history_length
            self.message_history[session_id] = self.message_history[session_id][-self.max_history_length:]
            debug(f"Trimmed {removed_count} old messages from history", "chat_service")
        
        log_user_action("message_added",
            session_id=session_id,
            role=role,
            content_length=len(content),
            message_id=message.id
        )
        
        debug(f"Message added successfully: {message.id}", "chat_service")
        return message
    
    async def process_user_message(
        self,
        content: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Process user message and generate AI response.
        
        Args:
            content: User message content
            session_id: Optional session ID
            system_prompt: Optional system prompt override
            
        Returns:
            ModelResponse: AI response
        """
        debug("Processing user message", "chat_service")
        
        with LoggedOperation("user_message_processing", "chat_service"):
            try:
                # Use current session if not specified
                if not session_id:
                    session_id = self.current_session_id
                
                if not session_id:
                    # Create new session if none exists
                    session_id = self.create_session()
                
                # Add user message
                user_message = self.add_message('user', content, session_id)
                
                # Generate AI response
                response = await self.model_service.generate_response(
                    content,
                    session_id,
                    system_prompt
                )
                
                # Add assistant response if successful
                if response.success:
                    self.add_message('assistant', response.content, session_id, {
                        'model_response_metadata': response.metadata
                    })
                
                log_user_action("user_message_processed",
                    session_id=session_id,
                    user_message_id=user_message.id,
                    response_success=response.success,
                    content_length=len(content)
                )
                
                info("User message processed successfully", "chat_service")
                return response
                
            except Exception as e:
                error("Failed to process user message", "chat_service", error_obj=e)
                
                return ModelResponse(
                    content="",
                    success=False,
                    error=f"Error processing message: {str(e)}",
                    metadata={'session_id': session_id}
                )
    
    def get_message_history(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        role_filter: Optional[List[str]] = None
    ) -> List[ChatMessage]:
        """
        Get message history for a session.
        
        Args:
            session_id: Optional session ID (uses current if not provided)
            limit: Optional limit on number of messages
            role_filter: Optional filter by message roles
            
        Returns:
            List[ChatMessage]: Message history
        """
        debug("Retrieving message history", "chat_service")
        
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.message_history:
            debug("No message history found", "chat_service")
            return []
        
        messages = self.message_history[session_id]
        
        # Apply role filter
        if role_filter:
            messages = [msg for msg in messages if msg.role in role_filter]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]
        
        debug(f"Retrieved {len(messages)} messages", "chat_service")
        return messages
    
    def get_session_info(self, session_id: Optional[str] = None) -> Optional[ChatSession]:
        """Get session information."""
        debug("Retrieving session info", "chat_service")
        
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.active_sessions:
            return None
        
        return self.active_sessions[session_id]
    
    def get_all_sessions(self) -> List[ChatSession]:
        """Get all active sessions."""
        debug("Retrieving all sessions", "chat_service")
        return list(self.active_sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        debug(f"Deleting session: {session_id}", "chat_service")
        
        if session_id not in self.active_sessions:
            warning(f"Session not found for deletion: {session_id}", "chat_service")
            return False
        
        try:
            # Remove session and messages
            del self.active_sessions[session_id]
            if session_id in self.message_history:
                del self.message_history[session_id]
            
            # Update current session if needed
            if self.current_session_id == session_id:
                self.current_session_id = None
                if self.active_sessions:
                    self.current_session_id = list(self.active_sessions.keys())[0]
            
            log_user_action("chat_session_deleted", session_id=session_id)
            info(f"Session deleted: {session_id}", "chat_service")
            return True
            
        except Exception as e:
            error(f"Failed to delete session: {session_id}", "chat_service", error_obj=e)
            return False
    
    def clear_session_history(self, session_id: Optional[str] = None) -> bool:
        """Clear message history for a session."""
        debug("Clearing session history", "chat_service")
        
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.message_history:
            return False
        
        try:
            self.message_history[session_id] = []
            
            # Update session info
            if session_id in self.active_sessions:
                self.active_sessions[session_id].message_count = 0
                self.active_sessions[session_id].last_activity = datetime.now()
            
            log_user_action("session_history_cleared", session_id=session_id)
            info(f"Session history cleared: {session_id}", "chat_service")
            return True
            
        except Exception as e:
            error("Failed to clear session history", "chat_service", error_obj=e)
            return False
    
    def export_conversation(
        self,
        session_id: Optional[str] = None,
        options: Optional[ExportOptions] = None
    ) -> Dict[str, Any]:
        """
        Export conversation data.
        
        Args:
            session_id: Optional session ID (uses current if not provided)
            options: Export options
            
        Returns:
            Dict containing exported data
        """
        debug("Exporting conversation", "chat_service")
        
        if not session_id:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.active_sessions:
            error("No valid session for export", "chat_service")
            return {}
        
        if not options:
            options = ExportOptions(format='json')
        
        try:
            session = self.active_sessions[session_id]
            messages = self.get_message_history(session_id)
            
            # Filter messages by date range if specified
            if options.date_range:
                start_date, end_date = options.date_range
                messages = [
                    msg for msg in messages
                    if start_date <= msg.timestamp <= end_date
                ]
            
            # Filter system messages if requested
            if not options.include_system_messages:
                messages = [msg for msg in messages if msg.role != 'system']
            
            export_data = {
                'session_info': asdict(session),
                'messages': [],
                'export_metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'format': options.format,
                    'message_count': len(messages),
                    'include_metadata': options.include_metadata
                }
            }
            
            # Convert messages
            for message in messages:
                message_data = {
                    'id': message.id,
                    'role': message.role,
                    'content': message.content,
                    'timestamp': message.timestamp.isoformat()
                }
                
                if options.include_metadata and message.metadata:
                    message_data['metadata'] = message.metadata
                
                export_data['messages'].append(message_data)
            
            log_user_action("conversation_exported",
                session_id=session_id,
                format=options.format,
                message_count=len(messages)
            )
            
            info(f"Conversation exported: {session_id}", "chat_service")
            return export_data
            
        except Exception as e:
            error("Failed to export conversation", "chat_service", error_obj=e)
            return {}
    
    def import_conversation(self, import_data: Dict[str, Any]) -> Optional[str]:
        """
        Import conversation data.
        
        Args:
            import_data: Imported conversation data
            
        Returns:
            Optional session ID if successful
        """
        debug("Importing conversation", "chat_service")
        
        try:
            session_info = import_data.get('session_info', {})
            messages = import_data.get('messages', [])
            
            # Create new session
            session_id = self.create_session(
                title=session_info.get('title', 'Imported Session'),
                system_prompt=session_info.get('system_prompt'),
                model_id=session_info.get('model_id')
            )
            
            # Import messages
            for message_data in messages:
                self.add_message(
                    role=message_data['role'],
                    content=message_data['content'],
                    session_id=session_id,
                    metadata=message_data.get('metadata')
                )
            
            log_user_action("conversation_imported",
                session_id=session_id,
                message_count=len(messages)
            )
            
            info(f"Conversation imported: {session_id}", "chat_service")
            return session_id
            
        except Exception as e:
            error("Failed to import conversation", "chat_service", error_obj=e)
            return None
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get chat usage statistics."""
        debug("Retrieving chat statistics", "chat_service")
        
        total_messages = sum(len(messages) for messages in self.message_history.values())
        total_sessions = len(self.active_sessions)
        
        # Calculate average messages per session
        avg_messages = total_messages / total_sessions if total_sessions > 0 else 0
        
        # Find most active session
        most_active_session = None
        max_messages = 0
        for session_id, messages in self.message_history.items():
            if len(messages) > max_messages:
                max_messages = len(messages)
                most_active_session = session_id
        
        # Recent activity (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_sessions = [
            session for session in self.active_sessions.values()
            if session.last_activity >= recent_threshold
        ]
        
        statistics = {
            'total_sessions': total_sessions,
            'total_messages': total_messages,
            'average_messages_per_session': round(avg_messages, 2),
            'most_active_session': {
                'session_id': most_active_session,
                'message_count': max_messages
            } if most_active_session else None,
            'recent_activity': {
                'active_sessions_24h': len(recent_sessions),
                'threshold': recent_threshold.isoformat()
            },
            'current_session': self.current_session_id
        }
        
        debug("Chat statistics calculated", "chat_service", statistics=statistics)
        return statistics
