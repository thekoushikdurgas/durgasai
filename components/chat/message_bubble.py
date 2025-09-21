"""
Message Bubble Component.

Handles the display of individual chat messages with proper styling,
metadata, and comprehensive logging for debugging and analytics.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add utils to path for proper module resolution
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation


class MessageBubble:
    """Component for rendering individual chat messages."""
    
    def render(self, role: str, content: str, timestamp: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Render a message bubble with comprehensive logging and error handling.
        
        This method displays individual chat messages with proper styling, timestamps,
        and optional metadata. It includes detailed logging for debugging and analytics.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content to display
            timestamp: Optional timestamp string for the message
            metadata: Optional metadata dictionary containing additional information
            
        Features:
        - Message rendering with proper role-based styling
        - Timestamp display with fallback to current time
        - Debug metadata display when debug mode is enabled
        - Comprehensive error handling and logging
        - User interaction tracking for analytics
        
        Logging:
        - Message rendering details
        - Content validation and processing
        - Timestamp handling
        - Metadata display status
        - Error handling for malformed messages
        """
        debug("Starting message bubble rendering", "message_bubble",
              role=role,
              content_length=len(content) if content else 0,
              has_timestamp=bool(timestamp),
              has_metadata=bool(metadata),
              debug_mode=st.session_state.get("debug_mode", False))
        
        try:
            with LoggedOperation("message_bubble_rendering", "message_bubble"):
                # Validate input parameters
                if not role or role not in ['user', 'assistant']:
                    warning("Invalid message role provided", "message_bubble",
                           role=role,
                           valid_roles=['user', 'assistant'])
                    role = 'user'  # Default fallback
                
                if not content:
                    warning("Empty message content provided", "message_bubble",
                           role=role)
                    content = "[Empty message]"
                
                debug("Message parameters validated", "message_bubble",
                      role=role,
                      content_length=len(content))
                
                # Render message with proper role-based styling
                with st.chat_message(role):
                    # Display message content with error handling
                    try:
                        st.markdown(content)
                        debug("Message content displayed successfully", "message_bubble",
                              role=role,
                              content_length=len(content))
                    except Exception as e:
                        error("Error displaying message content", "message_bubble", e,
                              role=role,
                              content_preview=content[:100] if content else "empty")
                        # Fallback: display raw content
                        st.text(content)
                        warning("Message content displayed as raw text due to rendering error", "message_bubble")
                    
                    # Handle timestamp display
                    display_timestamp = timestamp or datetime.now().strftime('%H:%M:%S')
                    try:
                        st.caption(f"🕒 {display_timestamp}")
                        debug("Message timestamp displayed", "message_bubble",
                              timestamp=display_timestamp,
                              was_provided=bool(timestamp))
                    except Exception as e:
                        error("Error displaying message timestamp", "message_bubble", e,
                              timestamp=display_timestamp)
                    
                    # Display metadata in debug mode
                    if metadata and st.session_state.get("debug_mode", False):
                        try:
                            with st.expander("🔍 Message Metadata", expanded=False):
                                st.json(metadata)
                            debug("Message metadata displayed in debug mode", "message_bubble",
                                  metadata_keys=list(metadata.keys()) if isinstance(metadata, dict) else "not_dict")
                        except Exception as e:
                            error("Error displaying message metadata", "message_bubble", e,
                                  metadata_type=type(metadata).__name__)
                
                # Log successful message rendering for analytics
                log_user_action("message_rendered",
                              role=role,
                              content_length=len(content),
                              has_timestamp=bool(timestamp),
                              has_metadata=bool(metadata),
                              debug_mode=st.session_state.get("debug_mode", False))
                
                debug("Message bubble rendered successfully", "message_bubble",
                      role=role,
                      content_length=len(content))
                
        except Exception as e:
            error("Critical error in message bubble rendering", "message_bubble", e,
                  role=role,
                  content_length=len(content) if content else 0,
                  has_timestamp=bool(timestamp),
                  has_metadata=bool(metadata))
            
            # Fallback: render basic message without styling
            try:
                st.write(f"**{role.title()}:** {content}")
                warning("Message rendered using fallback method due to critical error", "message_bubble")
            except Exception as fallback_error:
                error("Fallback message rendering also failed", "message_bubble", fallback_error)
                st.error("🚨 Error displaying message")
            
            raise
        
        debug("Message bubble rendering completed", "message_bubble")
