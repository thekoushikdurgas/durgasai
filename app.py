"""
DurgasAI - Advanced AI Agent Application
Main Streamlit application with navigation and model management.
"""

# Import warning suppression module FIRST
import suppress_warnings

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.config import Config
from utils.ui_helpers import UIHelpers, SessionManager
from utils.model_manager import ModelManager

# Import pages
from page.aiagent import AIAgentPage


class DurgasAIApp:
    """Main application class."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        
    def setup_application(self):
        """Setup the application."""
        # Page configuration
        UIHelpers.setup_page_config()
        
        # Load custom CSS
        UIHelpers.load_custom_css()
        
        # Initialize session state
        SessionManager.initialize_session_state()
    
    def render_home_page(self):
        """Render the home page."""
        st.markdown('<h1 class="main-header">🤖 Welcome to DurgasAI</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Advanced AI Agent Platform
        
        Welcome to **DurgasAI**, your comprehensive AI assistant powered by cutting-edge language models 
        from HuggingFace and enhanced with LangChain for superior conversational capabilities.
        
        ### 🌟 Key Features
        
        - **Multiple AI Models**: Choose from various state-of-the-art models
        - **LangChain Integration**: Advanced conversation memory and context management
        - **Customizable Prompts**: Tailor the AI's behavior to your needs
        - **Real-time Chat**: Smooth, responsive conversational interface
        - **Export Conversations**: Save and share your chat sessions
        - **Model Analytics**: Track usage and performance metrics
        
        ### 🚀 Available Models
        """)
        
        # Display available models
        cols = st.columns(2)
        models = Config.AVAILABLE_MODELS
        
        for i, (key, model_config) in enumerate(models.items()):
            with cols[i % 2]:
                UIHelpers.display_model_info(model_config)
        
        st.markdown("""
        ### 🎯 Use Cases
        
        - **💼 Business**: Customer support, content creation, data analysis
        - **🎓 Education**: Tutoring, research assistance, learning support  
        - **💻 Development**: Code review, documentation, debugging help
        - **🎨 Creative**: Writing, brainstorming, creative projects
        - **🔬 Research**: Information gathering, summarization, analysis
        
        ### 🚀 Getting Started
        
        1. Navigate to the **AI Agent** page
        2. Enter your HuggingFace API token (get one [here](https://huggingface.co/settings/tokens))
        3. Select your preferred AI model
        4. Customize the system prompt (optional)
        5. Start chatting!
        
        ---
        
        Ready to begin? Head over to the **🤖 AI Agent** page to start your conversation!
        """)
        
        # Display usage tips
        UIHelpers.display_usage_tips()
    
    def render_settings_page(self):
        """Render the settings page."""
        st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
        
        st.markdown("### 🔑 API Configuration")
        
        # API Token setting
        api_token = st.text_input(
            "HuggingFace API Token:",
            value=st.session_state.get("api_token", ""),
            type="password",
            help="Your HuggingFace API token for accessing models"
        )
        
        if api_token != st.session_state.get("api_token", ""):
            st.session_state.api_token = api_token
            st.success("API token updated!")
        
        st.markdown("### 🎨 UI Preferences")
        
        # Theme settings
        theme = st.selectbox(
            "Theme:",
            ["Auto", "Light", "Dark"],
            index=0
        )
        
        # Chat settings
        st.markdown("### 💬 Chat Configuration")
        
        max_messages = st.slider(
            "Maximum chat history:",
            min_value=10,
            max_value=100,
            value=Config.MAX_MESSAGE_HISTORY,
            help="Maximum number of messages to keep in chat history"
        )
        
        # Model defaults
        st.markdown("### 🤖 Model Defaults")
        
        default_temperature = st.slider(
            "Default Temperature:",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses"
        )
        
        default_max_tokens = st.slider(
            "Default Max Tokens:",
            min_value=50,
            max_value=1000,
            value=512,
            help="Maximum length of generated responses"
        )
        
        # Export/Import settings
        st.markdown("### 📁 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Data"):
                SessionManager.clear_messages()
                st.success("All data cleared!")
                st.rerun()
        
        with col2:
            if st.session_state.messages:
                UIHelpers.export_chat_history(st.session_state.messages)
    
    def render_analytics_page(self):
        """Render the analytics page."""
        st.markdown('<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
        
        # Get metrics
        metrics = SessionManager.get_conversation_metrics()
        
        st.markdown("### 📈 Session Statistics")
        UIHelpers.display_metrics(metrics)
        
        st.markdown("### 💬 Message Distribution")
        
        if st.session_state.messages:
            # Message type distribution
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("User Messages", user_messages)
            with col2:
                st.metric("Assistant Messages", assistant_messages)
            
            # Recent activity
            st.markdown("### 🕐 Recent Activity")
            
            recent_messages = st.session_state.messages[-10:] if st.session_state.messages else []
            
            for msg in recent_messages:
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                st.write(f"{role_icon} **{msg['role'].title()}** ({msg['timestamp']}): {msg['content'][:100]}...")
        
        else:
            st.info("No conversation data available. Start chatting to see analytics!")
        
        # Model usage
        st.markdown("### 🤖 Model Usage")
        if st.session_state.current_model:
            st.write(f"**Current Model**: {st.session_state.current_model}")
        else:
            st.write("**Current Model**: None selected")
    
    def render_help_page(self):
        """Render the help page."""
        st.markdown('<h1 class="main-header">❓ Help & Documentation</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 📚 User Guide
        
        ### Getting Started
        
        1. **Setup API Token**
           - Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
           - Create a new token (read access is sufficient)
           - Enter the token in the AI Agent page sidebar
        
        2. **Choose a Model**
           - Select from available models based on your needs
           - API models require internet connection
           - Local models run on your device (requires more resources)
        
        3. **Customize System Prompt**
           - Use predefined prompts or create your own
           - System prompts guide the AI's behavior and responses
           - Try different prompts for different use cases
        
        ### 🔧 Troubleshooting
        
        **Common Issues:**
        
        - **"No model loaded"**: Make sure to initialize a model first
        - **API errors**: Check your internet connection and API token
        - **Slow responses**: Try a smaller model or reduce max tokens
        - **Memory issues**: Clear chat history or restart the application
        
        **Performance Tips:**
        
        - Use API models for faster responses
        - Adjust temperature for desired creativity level
        - Keep conversations focused for better context
        - Clear chat history periodically
        
        ### 🤖 Model Guide
        
        **Mistral 7B Instruct**
        - Best for: Instructions, reasoning, coding
        - Strengths: Following complex instructions, logical reasoning
        - Use when: You need precise, instruction-following responses
        
        **Zephyr 7B Beta**
        - Best for: General conversation, Q&A
        - Strengths: Natural dialogue, helpfulness
        - Use when: You want natural, conversational interactions
        
        **Flan T5 Large**
        - Best for: Educational content, factual questions
        - Strengths: Factual accuracy, educational responses
        - Use when: You need reliable, educational information
        
        **DialoGPT Medium**
        - Best for: Casual chat, dialogue
        - Strengths: Conversational flow, personality
        - Use when: You want engaging, casual conversations
        
        **BlenderBot 400M**
        - Best for: Open-domain chat, creative conversations
        - Strengths: Creativity, diverse topics
        - Use when: You want creative, diverse conversations
        
        ### 📋 Best Practices
        
        1. **Be Specific**: Clear, specific questions get better answers
        2. **Use Context**: Reference previous messages for continuity
        3. **Experiment**: Try different models and settings
        4. **Save Important Chats**: Export conversations you want to keep
        5. **Provide Feedback**: Note what works well for future reference
        
        ### 🆘 Support
        
        If you encounter issues:
        1. Check the troubleshooting section above
        2. Try clearing your browser cache
        3. Restart the application
        4. Check your internet connection
        5. Verify your API token is valid
        
        ### 📖 Additional Resources
        
        - [HuggingFace Documentation](https://huggingface.co/docs)
        - [LangChain Documentation](https://python.langchain.com/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        """)
        
        UIHelpers.display_usage_tips()
    
    def run(self):
        """Run the main application."""
        self.setup_application()
        
        # Sidebar navigation
        selected_page = UIHelpers.create_sidebar_navigation()
        
        # Main content area
        if selected_page == "home":
            self.render_home_page()
        elif selected_page == "ai_agent":
            ai_agent_page = AIAgentPage(self.model_manager)
            ai_agent_page.render()
        elif selected_page == "settings":
            self.render_settings_page()
        elif selected_page == "analytics":
            self.render_analytics_page()
        elif selected_page == "help":
            self.render_help_page()
        
        # Footer
        UIHelpers.display_footer()


def main():
    """Main function to run the application."""
    app = DurgasAIApp()
    app.run()


if __name__ == "__main__":
    main()
