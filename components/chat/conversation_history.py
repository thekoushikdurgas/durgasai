"""
Conversation History Component.

Manages the display and organization of chat message history.
"""

import streamlit as st
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.application_state import ApplicationState
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation
from .message_bubble import MessageBubble


class ConversationHistory:
    """Component for managing and displaying conversation history."""
    
    def __init__(self):
        """Initialize the conversation history component."""
        self.message_bubble = MessageBubble()
        debug("ConversationHistory initialized", "conversation_history")
    
    def render(self) -> None:
        """
        Render the conversation history with comprehensive logging and error handling.
        
        This method displays the complete conversation history, including all previous
        messages between the user and AI assistant. It handles empty states and
        provides detailed logging for debugging and analytics.
        
        Features:
        - Complete conversation history display
        - Empty state handling with examples
        - Message validation and error handling
        - Performance logging for large conversations
        - User interaction tracking
        
        Logging:
        - Message count and validation
        - Rendering performance
        - Error handling for malformed messages
        - User interaction analytics
        """
        debug("Starting conversation history rendering", "conversation_history")
        
        try:
            with LoggedOperation("conversation_history_rendering", "conversation_history"):
                # Get messages from application state
                messages = ApplicationState.get('messages', [])
                message_count = len(messages)
                
                debug("Retrieved messages from application state", "conversation_history",
                      message_count=message_count,
                      total_messages=ApplicationState.get('total_messages', 0))
                
                # Handle empty conversation state
                if not messages:
                    debug("No messages found, rendering empty state", "conversation_history")
                    self._render_empty_state()
                    debug("Empty state rendering completed", "conversation_history")
                    return
                
                # Log conversation statistics for analytics
                log_user_action("conversation_history_displayed",
                              message_count=message_count,
                              conversation_length=self._calculate_conversation_length(messages))
                
                # Display all messages with validation and error handling
                debug(f"Rendering {message_count} messages", "conversation_history")
                
                for i, message in enumerate(messages):
                    try:
                        # Validate message structure
                        if not self._validate_message(message):
                            warning(f"Skipping invalid message at index {i}", "conversation_history",
                                  message_keys=list(message.keys()) if isinstance(message, dict) else "not_dict")
                            continue
                        
                        # Extract message components
                        role = message.get('role', 'user')
                        content = message.get('content', '')
                        timestamp = message.get('timestamp')
                        metadata = message.get('metadata')
                        
                        debug(f"Rendering message {i+1}/{message_count}", "conversation_history",
                              role=role,
                              content_length=len(content),
                              has_timestamp=bool(timestamp),
                              has_metadata=bool(metadata))
                        
                        # Render individual message
                        self.message_bubble.render(
                            role=role,
                            content=content,
                            timestamp=timestamp,
                            metadata=metadata
                        )
                        
                        debug(f"Message {i+1} rendered successfully", "conversation_history")
                        
                    except Exception as e:
                        error(f"Error rendering message {i+1}", "conversation_history", e,
                              message_role=message.get('role', 'unknown'),
                              message_content_length=len(str(message.get('content', ''))))
                        
                        # Continue with other messages even if one fails
                        continue
                
                debug(f"All {message_count} messages rendered successfully", "conversation_history")
                
        except Exception as e:
            error("Critical error in conversation history rendering", "conversation_history", e)
            # Show user-friendly error message
            st.error("🚨 An error occurred while loading conversation history. Please refresh the page.")
            raise
        
        debug("Conversation history rendering completed", "conversation_history")
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate message structure for rendering.
        
        Args:
            message: Message dictionary to validate
            
        Returns:
            bool: True if message is valid, False otherwise
        """
        if not isinstance(message, dict):
            return False
        
        required_fields = ['role', 'content']
        return all(field in message for field in required_fields)
    
    def _calculate_conversation_length(self, messages: List[Dict[str, Any]]) -> int:
        """
        Calculate total conversation length in characters.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            int: Total character count of all messages
        """
        total_length = 0
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                total_length += len(content)
        return total_length
    
    def _render_empty_state(self) -> None:
        """Render empty state when no messages exist."""
        st.info("👋 Welcome! Start a conversation by typing a message below.")
        
        # Show conversation examples
        self._render_conversation_examples()
    
    def _render_conversation_examples(self) -> None:
        """Render conversation starter examples."""
        with st.expander("💬 Conversation Starters", expanded=True):
            examples = [
                {
                    "title": "📚 Learning",
                    "prompt": "Explain quantum computing in simple terms",
                    "description": "Get clear explanations of complex topics"
                },
                {
                    "title": "💻 Coding", 
                    "prompt": "How do I create a REST API in Python?",
                    "description": "Get programming help and code examples"
                },
                {
                    "title": "✍️ Creative",
                    "prompt": "Write a short story about a robot learning to paint",
                    "description": "Generate creative content and stories"
                },
                {
                    "title": "🔍 Research",
                    "prompt": "What are the benefits of renewable energy?",
                    "description": "Get research help and information summaries"
                }
            ]
            
            for example in examples:
                st.markdown(f"**{example['title']}**")
                st.write(f"*{example['description']}*")
                
                if st.button(f"Try: {example['prompt'][:50]}...", key=f"example_{example['title']}"):
                    if ApplicationState.is_model_loaded():
                        # Set the example prompt for processing
                        st.session_state.example_prompt = example['prompt']
                    else:
                        st.warning("Please load a model first!")
                
                st.markdown("---")
