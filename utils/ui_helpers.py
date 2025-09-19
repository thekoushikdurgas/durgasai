"""
UI Helper functions for the Streamlit application.
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from .config import Config


class UIHelpers:
    """Helper functions for UI components."""
    
    @staticmethod
    def setup_page_config():
        """Setup page configuration."""
        st.set_page_config(
            page_title=Config.APP_TITLE,
            page_icon=Config.APP_ICON,
            layout=Config.LAYOUT,
            initial_sidebar_state="expanded"
        )
    
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
        """Create sidebar navigation."""
        with st.sidebar:
            st.title("🤖 DurgasAI")
            st.markdown("---")
            
            # Navigation
            pages = {
                "🏠 Home": "home",
                "🤖 AI Agent": "ai_agent",
                "⚙️ Settings": "settings",
                "📊 Analytics": "analytics",
                "❓ Help": "help"
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
    
    @staticmethod
    def display_footer():
        """Display application footer."""
        st.markdown("""
        <div class="footer">
            <p>
                Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong>, and <strong>🤗 HuggingFace</strong><br>
                <a href="https://huggingface.co/models" target="_blank">Explore More Models</a> | 
                <a href="https://docs.streamlit.io/" target="_blank">Streamlit Docs</a> | 
                <a href="https://python.langchain.com/" target="_blank">LangChain Docs</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
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
    """Manage Streamlit session state."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        defaults = {
            "messages": [],
            "model_loaded": False,
            "current_model": None,
            "api_token": "",
            "system_prompt": Config.DEFAULT_SYSTEM_PROMPTS["helpful_assistant"],
            "chat_session_id": "default",
            "total_messages": 0,
            "session_start_time": datetime.now()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to session state."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "metadata": metadata or {}
        }
        
        st.session_state.messages.append(message)
        st.session_state.total_messages += 1
        
        # Limit message history
        if len(st.session_state.messages) > Config.MAX_MESSAGE_HISTORY:
            st.session_state.messages = st.session_state.messages[-Config.MAX_MESSAGE_HISTORY:]
    
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
