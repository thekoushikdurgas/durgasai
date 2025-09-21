"""
Main Chat Interface Component.

This module provides the primary chat interface for DurgasAI, orchestrating
the complete chat experience including message display, input processing,
and response generation.

Key Features:
- Unified chat interface orchestration
- Message display and history management
- Input handling and processing coordination
- Quick action buttons for common prompts
- Real-time conversation flow management
- Integration with model manager for AI responses

Architecture:
- Uses composition pattern with specialized components
- Integrates MessageBubble for individual message rendering
- Uses InputHandler for processing user input
- Manages ConversationHistory for message persistence
- Coordinates with ApplicationState for session management

Components:
- MessageBubble: Renders individual chat messages
- InputHandler: Processes user input and generates responses
- ConversationHistory: Manages conversation display and persistence

Dependencies:
- utils.model_manager: AI model management and response generation
- utils.logger: Centralized logging system
- core.application_state: Global state management
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
from .input_handler import InputHandler
from .conversation_history import ConversationHistory


class ChatInterface:
    """
    Main chat interface component.
    
    Orchestrates the chat experience by coordinating message display,
    input handling, and response generation.
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the chat interface with required components.
        
        This constructor sets up the chat interface by initializing all
        required sub-components and establishing the connection to the
        model manager for AI response generation.
        
        Args:
            model_manager: ModelManager instance for AI model operations
            
        Components Initialized:
        - MessageBubble: For rendering individual chat messages
        - InputHandler: For processing user input and generating responses
        - ConversationHistory: For managing conversation display and persistence
        
        Note:
            The model_manager is shared with InputHandler to ensure consistent
            model access across the chat interface.
        """
        debug("Starting ChatInterface initialization", "chat_interface")
        
        # Store reference to model manager for AI operations
        self.model_manager = model_manager
        debug("Model manager reference stored", "chat_interface")
        
        # Initialize sub-components for chat functionality
        debug("Initializing chat sub-components", "chat_interface")
        
        # MessageBubble handles individual message rendering
        self.message_bubble = MessageBubble()
        debug("MessageBubble component initialized", "chat_interface")
        
        # InputHandler processes user input and generates AI responses
        self.input_handler = InputHandler(model_manager)
        debug("InputHandler component initialized with model manager", "chat_interface")
        
        # ConversationHistory manages conversation display and persistence
        self.conversation_history = ConversationHistory()
        debug("ConversationHistory component initialized", "chat_interface")
        
        info("ChatInterface initialization completed successfully", "chat_interface")
    
    def render(self, config: Dict[str, Any]) -> None:
        """
        Render the complete chat interface.
        
        This method orchestrates the rendering of all chat interface components
        in the correct order to provide a cohesive user experience.
        
        Rendering Order:
        1. Conversation history (past messages)
        2. Chat input field (for new messages)
        3. Quick action buttons (for common prompts)
        
        Args:
            config: Configuration dictionary containing:
                - Model settings and parameters
                - UI configuration options
                - Chat behavior settings
                
        Note:
            The rendering order is important for proper UI layout and
            user interaction flow.
        """
        debug("Starting complete chat interface rendering", "chat_interface")
        
        # Step 1: Render conversation history
        # This shows all previous messages in the conversation
        debug("Rendering conversation history component", "chat_interface")
        self.conversation_history.render()
        debug("Conversation history rendered successfully", "chat_interface")
        
        # Step 2: Render chat input field
        # This provides the main input mechanism for users
        debug("Rendering chat input component", "chat_interface")
        self._render_chat_input(config)
        debug("Chat input rendered successfully", "chat_interface")
        
        # Step 3: Render quick action buttons
        # These provide shortcuts for common user actions
        debug("Rendering quick actions component", "chat_interface")
        self._render_quick_actions(config)
        debug("Quick actions rendered successfully", "chat_interface")
        
        debug("Complete chat interface rendering finished", "chat_interface")
    
    def _render_chat_input(self, config: Dict[str, Any]) -> None:
        """
        Render the chat input area with comprehensive logging.
        
        This method creates the main chat input interface where users can type
        their messages. It includes model status checking and input validation.
        
        Args:
            config: Configuration dictionary containing model settings and parameters
            
        Features:
        - Model availability checking before enabling input
        - User input validation and processing
        - Comprehensive logging for debugging and analytics
        - Integration with InputHandler for message processing
        
        Logging:
        - Input field rendering status
        - Model availability checks
        - User input attempts and validation
        - Error handling for input processing
        """
        debug("Starting chat input area rendering", "chat_interface")
        
        # Check model availability with detailed logging
        model_loaded = ApplicationState.is_model_loaded()
        current_model = ApplicationState.get_current_model()
        
        debug("Model status checked for chat input", "chat_interface", 
              model_loaded=model_loaded,
              current_model=current_model)
        
        # Log input field state
        input_disabled = not model_loaded
        debug("Chat input field configuration", "chat_interface",
              input_disabled=input_disabled,
              placeholder_text="Type your message here...")
        
        # Render chat input field with comprehensive error handling
        try:
            with LoggedOperation("chat_input_rendering", "chat_interface"):
                user_input = st.chat_input(
                    "Type your message here...", 
                    disabled=input_disabled,
                    key="main_chat_input"
                )
                
                debug("Chat input field rendered successfully", "chat_interface",
                      user_input_provided=user_input is not None,
                      input_length=len(user_input) if user_input else 0)
                
                # Process user input if provided
                if user_input:
                    debug("User input received, starting processing", "chat_interface",
                          input_length=len(user_input),
                          input_preview=user_input[:50] + "..." if len(user_input) > 50 else user_input)
                    
                    # Log user action for analytics
                    log_user_action("chat_input_submitted", 
                                  input_length=len(user_input),
                                  model_loaded=model_loaded,
                                  current_model=current_model)
                    
                    # Process the input through the input handler
                    with LoggedOperation("user_input_processing", "chat_interface"):
                        self.input_handler.process_input(user_input, config)
                    
                    debug("User input processing completed successfully", "chat_interface")
                else:
                    debug("No user input provided in this render cycle", "chat_interface")
                    
        except Exception as e:
            error("Error in chat input rendering", "chat_interface", e,
                  model_loaded=model_loaded,
                  input_disabled=input_disabled)
            raise
        
        debug("Chat input area rendering completed", "chat_interface")
    
    def _render_quick_actions(self, config: Dict[str, Any]) -> None:
        """
        Render quick action buttons with comprehensive logging and error handling.
        
        This method creates quick action buttons that provide shortcuts for common
        user interactions. It includes model status checking and user action tracking.
        
        Args:
            config: Configuration dictionary containing model settings and parameters
            
        Features:
        - Quick prompt buttons for common actions
        - Model availability checking
        - User action logging for analytics
        - Error handling and user feedback
        - Session state management for quick prompts
        
        Logging:
        - Quick action rendering status
        - Button click events
        - Model availability checks
        - User interaction tracking
        - Error handling for quick actions
        """
        debug("Starting quick actions rendering", "chat_interface")
        
        try:
            with LoggedOperation("quick_actions_rendering", "chat_interface"):
                # Render quick actions header
                st.markdown("### ⚡ Quick Actions")
                debug("Quick actions header rendered", "chat_interface")
                
                # Create column layout for buttons
                col1, col2, col3, col4 = st.columns(4)
                debug("Quick actions columns created", "chat_interface")
                
                # Define quick prompt templates with metadata
                quick_prompts = [
                    ("💡 Explain", "Can you explain", "explanation_request"),
                    ("📝 Summarize", "Please summarize", "summarization_request"),
                    ("🔍 Analyze", "Can you analyze", "analysis_request"),
                    ("💭 Brainstorm", "Help me brainstorm ideas about", "brainstorming_request")
                ]
                
                debug(f"Quick prompts defined: {len(quick_prompts)} prompts", "chat_interface",
                      prompt_types=[prompt[2] for prompt in quick_prompts])
                
                # Check model availability for button state
                model_loaded = ApplicationState.is_model_loaded()
                current_model = ApplicationState.get_current_model()
                
                debug("Model status checked for quick actions", "chat_interface",
                      model_loaded=model_loaded,
                      current_model=current_model)
                
                # Render quick action buttons
                for i, (label, prompt, action_type) in enumerate(quick_prompts):
                    column = [col1, col2, col3, col4][i]
                    
                    with column:
                        button_key = f"quick_action_{action_type}_{i}"
                        
                        if st.button(label, use_container_width=True, key=button_key):
                            debug(f"Quick action button clicked: {label}", "chat_interface",
                                  action_type=action_type,
                                  prompt_template=prompt,
                                  model_loaded=model_loaded)
                            
                            if model_loaded:
                                # Set quick prompt for completion
                                st.session_state.quick_prompt = prompt
                                st.session_state.quick_prompt_type = action_type
                                
                                debug("Quick prompt set in session state", "chat_interface",
                                      prompt_template=prompt,
                                      action_type=action_type)
                                
                                # Log user action for analytics
                                log_user_action("quick_action_clicked",
                                              action_type=action_type,
                                              button_label=label,
                                              prompt_template=prompt,
                                              model_loaded=model_loaded,
                                              current_model=current_model)
                            else:
                                # Show warning for missing model
                                st.warning("Please load a model first!")
                                debug("Quick action clicked but model not loaded", "chat_interface",
                                      action_type=action_type,
                                      button_label=label)
                                
                                # Log user action with warning
                                log_user_action("quick_action_clicked_no_model",
                                              action_type=action_type,
                                              button_label=label,
                                              prompt_template=prompt)
                
                debug("Quick action buttons rendered successfully", "chat_interface")
                
                # Handle quick prompt completion
                if hasattr(st.session_state, 'quick_prompt'):
                    debug("Quick prompt completion interface activated", "chat_interface",
                          prompt_type=getattr(st.session_state, 'quick_prompt_type', 'unknown'))
                    
                    completed_prompt = st.text_input(
                        "Complete your question:",
                        value=st.session_state.quick_prompt + " ",
                        key="quick_input_completion"
                    )
                    
                    debug("Quick prompt completion input rendered", "chat_interface",
                          prompt_length=len(completed_prompt) if completed_prompt else 0)
                    
                    if st.button("Send Quick Prompt", key="send_quick_prompt"):
                        debug("Send quick prompt button clicked", "chat_interface",
                              prompt_completed=bool(completed_prompt),
                              model_loaded=model_loaded)
                        
                        if completed_prompt and model_loaded:
                            # Log the completed prompt action
                            action_type = getattr(st.session_state, 'quick_prompt_type', 'unknown')
                            log_user_action("quick_prompt_sent",
                                          action_type=action_type,
                                          prompt_length=len(completed_prompt),
                                          model_loaded=model_loaded,
                                          current_model=current_model)
                            
                            # Process the completed prompt
                            with LoggedOperation("quick_prompt_processing", "chat_interface"):
                                self.input_handler.process_input(completed_prompt, config)
                            
                            debug("Quick prompt processed successfully", "chat_interface")
                            
                            # Clear quick prompt state
                            if hasattr(st.session_state, 'quick_prompt'):
                                delattr(st.session_state, 'quick_prompt')
                            if hasattr(st.session_state, 'quick_prompt_type'):
                                delattr(st.session_state, 'quick_prompt_type')
                            
                            debug("Quick prompt state cleared", "chat_interface")
                        else:
                            if not completed_prompt:
                                st.warning("Please complete your question!")
                                debug("Quick prompt send attempted with empty input", "chat_interface")
                            if not model_loaded:
                                st.warning("Please load a model first!")
                                debug("Quick prompt send attempted without loaded model", "chat_interface")
                
                debug("Quick actions rendering completed successfully", "chat_interface")
                
        except Exception as e:
            error("Error in quick actions rendering", "chat_interface", e,
                  model_loaded=model_loaded,
                  current_model=current_model)
            # Show user-friendly error message
            st.error("🚨 An error occurred while rendering quick actions. Please try again.")
            raise
        
        debug("Quick actions rendering finished", "chat_interface")
