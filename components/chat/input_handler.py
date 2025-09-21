"""
Input Handler Component.

This module handles user input processing, AI response generation, and conversation
flow management for the DurgasAI chat interface.

Key Features:
- User input validation and processing
- AI response generation coordination
- Conversation flow management
- Message persistence and state management
- Performance tracking and logging
- Error handling and recovery

Architecture:
- Integrates with ModelManager for AI response generation
- Uses ApplicationState for conversation persistence
- Implements comprehensive logging for debugging
- Provides error handling with user-friendly messages
- Tracks performance metrics for optimization

Responsibilities:
- Process and validate user input
- Coordinate AI response generation
- Manage conversation state updates
- Handle errors gracefully
- Track interaction metrics

Dependencies:
- utils.model_manager: AI model operations and response generation
- utils.logger: Centralized logging and performance tracking
- core.application_state: Global state and conversation persistence
- message_bubble: Message display component
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import sys
import time

# Add utils to path for proper module resolution
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.model_manager import ModelManager
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation
from core.application_state import ApplicationState
from .message_bubble import MessageBubble


class InputHandler:
    """Handles user input processing and AI response generation."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the input handler with model manager integration.
        
        This constructor sets up the input handler with the necessary components
        for processing user input and generating AI responses.
        
        Args:
            model_manager: ModelManager instance for AI operations
            
        Components Initialized:
        - Model manager reference for AI response generation
        - Message bubble for displaying messages
        - Interaction counter for analytics
        
        Note:
            The interaction counter is used for performance tracking
            and debugging purposes.
        """
        debug("Starting InputHandler initialization", "input_handler")
        
        # Store reference to model manager for AI operations
        self.model_manager = model_manager
        debug("Model manager reference established", "input_handler")
        
        # Initialize message bubble for message display
        self.message_bubble = MessageBubble()
        debug("Message bubble component initialized", "input_handler")
        
        # Initialize interaction counter for analytics
        self.interaction_count = 0
        debug("Interaction counter initialized to 0", "input_handler")
        
        info("InputHandler initialization completed successfully", "input_handler")
    
    def process_input(self, user_input: str, config: Dict[str, Any]) -> None:
        """
        Process user input and generate AI response.
        
        Args:
            user_input: The user's message
            config: Configuration dictionary
        """
        start_time = time.time()
        self.interaction_count += 1
        
        info(f"Processing user input #{self.interaction_count}", "input_handler",
            input_length=len(user_input),
            model_loaded=ApplicationState.is_model_loaded())
        
        try:
            # Add user message to session
            self._add_user_message(user_input)
            
            # Display user message
            self.message_bubble.render("user", user_input)
            
            # Generate AI response
            self._generate_ai_response(user_input, config)
            
            # Log successful interaction
            total_time = time.time() - start_time
            info(f"User interaction completed in {total_time:.2f}s", "input_handler",
                total_time=total_time,
                interaction_number=self.interaction_count)
            
            # Update interaction metrics
            ApplicationState.set('last_interaction_time', total_time)
            
        except Exception as e:
            error("Critical error in input processing", "input_handler", e,
                user_input_length=len(user_input),
                interaction_number=self.interaction_count)
            
            st.error("🚨 An error occurred while processing your message. Please try again.")
    
    def _add_user_message(self, content: str) -> None:
        """Add user message to session state."""
        messages = ApplicationState.get('messages', [])
        
        message = {
            "role": "user",
            "content": content,
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "full_timestamp": datetime.now().isoformat(),
            "message_id": f"user_{int(time.time() * 1000)}"
        }
        
        messages.append(message)
        ApplicationState.update({
            'messages': messages,
            'total_messages': ApplicationState.get('total_messages', 0) + 1
        })
        
        debug("User message added to session", "input_handler")
    
    def _generate_ai_response(self, user_input: str, config: Dict[str, Any]) -> None:
        """
        Generate and display AI response with comprehensive logging and error handling.
        
        This method coordinates the AI response generation process, including model
        interaction, response processing, error handling, and user feedback.
        
        Args:
            user_input: The user's input message
            config: Configuration dictionary containing model parameters
            
        Features:
        - AI model response generation
        - Response validation and error handling
        - Performance timing and logging
        - User feedback and error messages
        - Session state management
        - Comprehensive error recovery
        
        Logging:
        - Response generation timing
        - Model interaction details
        - Success/failure status
        - Error details and recovery
        - Performance metrics
        """
        debug("Starting AI response generation", "input_handler",
              input_length=len(user_input),
              interaction_count=self.interaction_count)
        
        # Log model interaction start
        log_user_action("ai_response_generation_started",
                       input_length=len(user_input),
                       interaction_count=self.interaction_count)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    with LoggedOperation("ai_response_generation", "input_handler"):
                        # Generate response with detailed timing
                        response_start = time.time()
                        debug("Calling model manager for response generation", "input_handler",
                              model_loaded=ApplicationState.is_model_loaded(),
                              current_model=ApplicationState.get_current_model())
                        
                        response = self.model_manager.generate_response(user_input)
                        response_time = time.time() - response_start
                        
                        debug("Model response received", "input_handler",
                              response_success=response.success,
                              response_time=response_time,
                              response_length=len(response.content) if hasattr(response, 'content') else 0)
                        
                        if response.success:
                            # Log successful response details
                            debug("Processing successful AI response", "input_handler",
                                  response_length=len(response.content),
                                  response_time=response_time)
                            
                            # Display successful response
                            st.markdown(response.content)
                            debug("AI response displayed to user", "input_handler")
                            
                            # Add to session state with detailed logging
                            self._add_assistant_message(response.content, response_time)
                            
                            # Log successful response for analytics
                            log_user_action("ai_response_received",
                                          response_length=len(response.content),
                                          response_time=response_time,
                                          interaction_count=self.interaction_count,
                                          success=True)
                            
                            debug("Successful AI response processing completed", "input_handler")
                            
                        else:
                            # Handle error response with detailed logging
                            error_msg = response.error or "Unknown error occurred"
                            debug("AI response generation failed", "input_handler",
                                  error_message=error_msg,
                                  response_time=response_time)
                            
                            st.error(f"❌ {error_msg}")
                            debug("Error message displayed to user", "input_handler")
                            
                            # Add error message to session with metadata
                            self._add_assistant_message(
                                "I apologize, but I encountered an error. Please try again.",
                                response_time,
                                {"error": error_msg, "error_type": "model_error"}
                            )
                            
                            # Log error response for analytics
                            log_user_action("ai_response_error", 
                                          error=error_msg,
                                          response_time=response_time,
                                          interaction_count=self.interaction_count,
                                          success=False)
                            
                            debug("Error response processing completed", "input_handler")
                
                except Exception as e:
                    # Handle unexpected errors with comprehensive logging
                    error_msg = f"Unexpected error: {str(e)}"
                    debug("Unexpected error during AI response generation", "input_handler",
                          error_type=type(e).__name__,
                          error_message=str(e),
                          interaction_count=self.interaction_count)
                    
                    st.error(f"❌ {error_msg}")
                    debug("Unexpected error message displayed to user", "input_handler")
                    
                    # Log unexpected error for debugging
                    error("Unexpected error during response generation", "input_handler", e,
                          user_input_length=len(user_input),
                          interaction_count=self.interaction_count)
                    
                    # Add error message to session with detailed metadata
                    self._add_assistant_message(
                        "I apologize, but I encountered an unexpected error. Please try again.",
                        0,
                        {"error": error_msg, "error_type": "unexpected", "exception_type": type(e).__name__}
                    )
                    
                    # Log unexpected error for analytics
                    log_user_action("ai_response_unexpected_error",
                                  error=error_msg,
                                  error_type=type(e).__name__,
                                  interaction_count=self.interaction_count,
                                  success=False)
                
                # Display timestamp with logging
                timestamp = datetime.now().strftime('%H:%M:%S')
                st.caption(f"🕒 {timestamp}")
                debug("Response timestamp displayed", "input_handler", timestamp=timestamp)
        
        debug("AI response generation process completed", "input_handler",
              interaction_count=self.interaction_count)
    
    def _add_assistant_message(self, content: str, response_time: float, metadata: Dict[str, Any] = None) -> None:
        """Add assistant message to session state."""
        messages = ApplicationState.get('messages', [])
        
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "full_timestamp": datetime.now().isoformat(),
            "message_id": f"assistant_{int(time.time() * 1000)}",
            "response_time": response_time,
            "metadata": metadata or {}
        }
        
        messages.append(message)
        ApplicationState.update({
            'messages': messages,
            'total_messages': ApplicationState.get('total_messages', 0) + 1
        })
        
        debug("Assistant message added to session", "input_handler")
