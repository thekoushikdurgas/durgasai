"""
UI Helper functions for the Streamlit application.

This module provides utility functions for UI components and session management:
- UIHelpers: Static methods for UI component creation and styling
- SessionManager: Session state management and conversation tracking

Key Features:
- Page configuration and CSS loading
- Model information display
- Chat message formatting
- Session state initialization and management
- Conversation metrics and analytics
- Export/import functionality
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import time

from .config import Config

# Import logging with fallback
try:
    from .logger import debug, info, warning, error, log_session_event, log_user_action
except ImportError:
    # Fallback logging functions
    def debug(msg, component="ui_helpers", **kwargs): pass
    def info(msg, component="ui_helpers", **kwargs): pass
    def warning(msg, component="ui_helpers", **kwargs): pass
    def error(msg, component="ui_helpers", **kwargs): pass
    def log_session_event(event, **kwargs): pass
    def log_user_action(action, **kwargs): pass


class UIHelpers:
    """
    Helper functions for UI components and application setup.
    
    This class provides static utility methods for common UI operations including:
    - Page configuration and setup
    - Custom CSS styling and themes
    - Model information display
    - Chat message formatting
    - Metrics and analytics display
    - Navigation and sidebar management
    
    All methods are static and can be called without instantiation.
    The class integrates with the logging system for debugging and monitoring.
    """
    
    @staticmethod
    def setup_page_config():
        """
        Setup Streamlit page configuration with DurgasAI branding.
        
        This method configures the Streamlit page with:
        - Application title and icon from Config
        - Wide layout for better component display
        - Expanded sidebar for navigation
        - Proper meta tags and settings
        
        Note:
            This should be called once at application startup before
            any other Streamlit components are rendered.
        """
        debug("Setting up Streamlit page configuration", "ui_helpers")
        
        st.set_page_config(
            page_title=Config.APP_TITLE,
            page_icon=Config.APP_ICON,
            layout=Config.LAYOUT,
            initial_sidebar_state="expanded"
        )
        
        debug("Page configuration completed", "ui_helpers",
              title=Config.APP_TITLE,
              icon=Config.APP_ICON,
              layout=Config.LAYOUT)
    
    @staticmethod
    def load_custom_css():
        """Load custom CSS styling."""
        css = """
        <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        .user-message {
            background-color: #f0f2f6;
            margin-left: 20%;
        }
        
        .assistant-message {
            background-color: #e8f4fd;
            margin-right: 20%;
        }
        
        .sidebar-content {
            padding: 1rem;
        }
        
        .model-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        }
        
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .timestamp {
            font-size: 0.8rem;
            color: #6c757d;
            font-style: italic;
        }
        
        .metrics-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .footer {
            text-align: center;
            color: #6c757d;
            margin-top: 2rem;
            padding: 1rem;
            border-top: 1px solid #dee2e6;
        }
        
        .stButton > button {
            width: 100%;
        }
        
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 1rem;
            border-top: 1px solid #dee2e6;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    @staticmethod
    def display_model_info(model_config, status: str = "ready"):
        """Display model information card."""
        status_class = {
            "ready": "status-success",
            "loading": "status-warning",
            "error": "status-error"
        }.get(status, "status-success")
        
        st.markdown(f"""
        <div class="model-card">
            <h4>{model_config.name}</h4>
            <p><strong>Model ID:</strong> {model_config.model_id}</p>
            <p><strong>Provider:</strong> {model_config.provider.value}</p>
            <p><strong>Description:</strong> {model_config.description}</p>
            <p class="{status_class}"><strong>Status:</strong> {status.title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_chat_message(role: str, content: str, timestamp: str = None):
        """Display a chat message with proper styling."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        message_class = "user-message" if role == "user" else "assistant-message"
        icon = "👤" if role == "user" else "🤖"
        
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{icon} {role.title()}:</strong><br>
            {content}<br>
            <span class="timestamp">🕒 {timestamp}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_metrics(metrics: Dict[str, Any]):
        """Display application metrics."""
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(key.replace('_', ' ').title(), value)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_sidebar_navigation():
        """Create sidebar navigation using enhanced sidebar component."""
        try:
            # Import enhanced sidebar component
            from components.common.sidebar import create_durgasai_sidebar
            
            # Create and render enhanced sidebar
            sidebar = create_durgasai_sidebar()
            navigation_config = sidebar.create_navigation_config()
            selected_page = sidebar.render(navigation_config)
            
            return selected_page
            
        except ImportError as e:
            # Fallback to simple selectbox navigation
            debug(f"Enhanced sidebar not available, using fallback: {e}", "ui_helpers")
            return UIHelpers._create_fallback_navigation()
    
    @staticmethod
    def _create_fallback_navigation():
        """Fallback navigation method using simple selectbox."""
        with st.sidebar:
            st.title("🤖 DurgasAI")
            st.markdown("---")
            
            # Navigation pages with debug dashboard
            pages = {
                "🏠 Home": "home",
                "🤖 AI Agent": "ai_agent",
                "🤗 HuggingFace Chat": "huggingface",
                "📚 Model Catalog": "model_catalog",
                "🎭 Facial AI": "facial_ai",
                "⚙️ Settings": "settings",
                "📊 Analytics": "analytics",
                "❓ Help": "help",
                "🔧 Debug": "debug"
            }
            
            selected_page = st.selectbox(
                "Navigate to:",
                list(pages.keys()),
                index=1  # Default to AI Agent
            )
            
            return pages[selected_page]
    
    @staticmethod
    def export_chat_history(messages: List[Dict[str, str]], filename: str = None):
        """Export chat history as JSON."""
        if filename is None:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        chat_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": messages
        }
        
        json_data = json.dumps(chat_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 Download Chat History",
            data=json_data,
            file_name=filename,
            mime="application/json",
            help="Download the current chat session as a JSON file"
        )
    
    @staticmethod
    def display_usage_tips():
        """Display usage tips and help information."""
        with st.expander("💡 Usage Tips & Help", expanded=False):
            st.markdown("""
            ### Getting Started
            1. **API Token**: Get your free HuggingFace API token from [here](https://huggingface.co/settings/tokens)
            2. **Model Selection**: Choose from various AI models based on your needs
            3. **System Prompt**: Customize the AI's behavior and personality
            4. **Start Chatting**: Type your message and press Enter!
            
            ### Model Recommendations
            - **🎯 Mistral 7B**: Best for instruction-following and reasoning
            - **💬 Zephyr 7B**: Excellent for conversations and general queries  
            - **📚 Flan T5**: Great for educational and factual questions
            - **🗣️ DialoGPT**: Optimized for dialogue and chat interactions
            - **🌟 BlenderBot**: Good for open-domain conversations
            
            ### Tips for Better Results
            - Be specific and clear in your questions
            - Use system prompts to guide the AI's responses
            - Adjust temperature for creativity vs consistency
            - Try different models for different types of tasks
            
            ### Troubleshooting
            - **API Errors**: Check your API token and internet connection
            - **Slow Responses**: Try a smaller model or reduce max tokens
            - **Memory Issues**: Clear chat history or restart the app
            - **Model Loading**: Wait for the model to fully initialize
            """)
    
    # @staticmethod
    # def display_footer():
    #     """Display application footer."""
    #     st.markdown("""
    #     <div class="footer">
    #         <p>
    #             Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong>, and <strong>🤗 HuggingFace</strong><br>
    #             <a href="https://huggingface.co/models" target="_blank">Explore More Models</a> | 
    #             <a href="https://docs.streamlit.io/" target="_blank">Streamlit Docs</a> | 
    #             <a href="https://python.langchain.com/" target="_blank">LangChain Docs</a>
    #         </p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    @staticmethod
    def show_loading_animation(message: str = "Processing..."):
        """Show loading animation with custom message."""
        return st.spinner(f"🔄 {message}")
    
    @staticmethod
    def display_error_message(error: str, details: str = None):
        """Display formatted error message."""
        st.error(f"❌ **Error**: {error}")
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def display_success_message(message: str):
        """Display formatted success message."""
        st.success(f"✅ {message}")
    
    @staticmethod
    def display_warning_message(message: str):
        """Display formatted warning message."""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def display_info_message(message: str):
        """Display formatted info message."""
        st.info(f"ℹ️ {message}")


class SessionManager:
    """
    Manage Streamlit session state with comprehensive logging and validation.
    
    This class handles all session state operations including:
    - Initialization of default session variables
    - Message management with history limits
    - Conversation metrics and analytics
    - Session state validation and debugging
    
    Session State Keys:
    - messages: List of conversation messages
    - model_loaded: Boolean indicating if a model is loaded
    - current_model: Name of the currently loaded model
    - api_token: HuggingFace API token (encrypted in session)
    - system_prompt: Current system prompt for AI behavior
    - chat_session_id: Unique identifier for the chat session
    - total_messages: Total messages sent in this session
    - session_start_time: When the session was started
    """
    
    @staticmethod
    def initialize_session_state():
        """
        Initialize session state variables with default values.
        
        This method sets up the initial session state for a new user session.
        It only initializes keys that don't already exist, preserving any
        existing session data across page reloads.
        
        The initialization is logged for debugging and analytics purposes.
        """
        debug("Initializing session state", "session")
        
        # Define default values for all session state variables
        defaults = {
            "messages": [],                    # Conversation message history
            "model_loaded": False,             # Whether an AI model is currently loaded
            "current_model": None,             # Name of the currently active model
            "api_token": "",                   # HuggingFace API token
            "system_prompt": Config.DEFAULT_SYSTEM_PROMPTS["helpful_assistant"],  # AI behavior prompt
            "chat_session_id": "default",      # Session identifier for conversation tracking
            "total_messages": 0,               # Total messages sent in this session
            "session_start_time": datetime.now(),  # Session start timestamp
            "current_page": "home",            # Currently active page
            "debug_mode": False,               # Debug mode toggle
            "last_model_response_time": None,  # Performance tracking
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Unique session ID
        }
        
        # Track which keys are newly initialized vs existing
        new_keys = []
        existing_keys = []
        
        # Initialize only missing keys to preserve existing session data
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                new_keys.append(key)
                debug(f"Initialized session state key: {key} = {value}", "session")
            else:
                existing_keys.append(key)
                debug(f"Preserving existing session state key: {key}", "session")
        
        # Log session initialization summary
        info("Session state initialization completed", "session",
            new_keys=new_keys,
            existing_keys=existing_keys,
            total_keys=len(defaults),
            session_id=st.session_state.get("session_id", "unknown"))
        
        # Log session event for analytics
        log_session_event("session_initialized",
            new_keys_count=len(new_keys),
            existing_keys_count=len(existing_keys),
            session_id=st.session_state.get("session_id"),
            start_time=st.session_state.get("session_start_time", datetime.now()).isoformat())
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add message to session state with comprehensive logging and validation.
        
        This method manages the conversation history by:
        1. Creating a structured message object with timestamp and metadata
        2. Adding the message to the session state
        3. Updating message counters for analytics
        4. Enforcing message history limits to prevent memory issues
        5. Logging the message addition for debugging
        
        Args:
            role (str): Message role ('user' or 'assistant')
            content (str): Message content text
            metadata (Dict[str, Any], optional): Additional message metadata
        """
        debug(f"Adding {role} message to session", "session",
            content_length=len(content),
            has_metadata=bool(metadata),
            current_message_count=len(st.session_state.get("messages", [])))
        
        # Validate input parameters
        if not role or role not in ["user", "assistant"]:
            warning(f"Invalid message role: {role}", "session")
            role = "user"  # Default to user role
        
        if not content or not content.strip():
            warning("Empty message content", "session")
            content = "[Empty message]"
        
        # Create structured message object
        message = {
            "role": role,
            "content": content.strip(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "full_timestamp": datetime.now().isoformat(),
            "message_id": f"{role}_{int(time.time() * 1000)}",  # Unique message ID
            "session_id": st.session_state.get("session_id", "unknown"),
            "metadata": metadata or {}
        }
        
        # Add message to session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.append(message)
        st.session_state.total_messages = st.session_state.get("total_messages", 0) + 1
        
        # Enforce message history limits to prevent memory issues
        if len(st.session_state.messages) > Config.MAX_MESSAGE_HISTORY:
            removed_count = len(st.session_state.messages) - Config.MAX_MESSAGE_HISTORY
            st.session_state.messages = st.session_state.messages[-Config.MAX_MESSAGE_HISTORY:]
            
            debug(f"Trimmed message history: removed {removed_count} old messages", "session")
        
        # Log message addition for analytics and debugging
        info(f"Message added: {role}", "session",
            content_length=len(content),
            total_messages=st.session_state.total_messages,
            current_history_size=len(st.session_state.messages),
            message_id=message["message_id"],
            has_metadata=bool(metadata))
        
        # Log user action for analytics
        log_user_action("message_sent",
            role=role,
            content_length=len(content),
            message_count=st.session_state.total_messages)
    
    @staticmethod
    def clear_messages():
        """Clear all messages from session state."""
        st.session_state.messages = []
        st.session_state.total_messages = 0
    
    @staticmethod
    def get_conversation_metrics():
        """Get conversation metrics."""
        session_duration = datetime.now() - st.session_state.session_start_time
        
        return {
            "total_messages": st.session_state.total_messages,
            "current_messages": len(st.session_state.messages),
            "session_duration": str(session_duration).split('.')[0],  # Remove microseconds
            "current_model": st.session_state.current_model or "None"
        }
