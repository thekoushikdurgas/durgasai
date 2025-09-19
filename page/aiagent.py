"""
AI Agent Page - Main chat interface with HuggingFace models and LangChain.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config, ModelProvider
from utils.ui_helpers import UIHelpers, SessionManager
from utils.model_manager import ModelManager, ModelResponse, APIModelManager


class AIAgentPage:
    """AI Agent chat page."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with model configuration."""
        with st.sidebar:
            st.markdown("### 🤖 AI Model Configuration")
            
            # API Token input
            api_token = st.text_input(
                "🔑 HuggingFace API Token:",
                type="password",
                value=st.session_state.get("api_token", ""),
                help="Get your token from https://huggingface.co/settings/tokens"
            )
            
            if api_token:
                st.session_state.api_token = api_token
                st.success("✅ API token configured")
            else:
                st.warning("⚠️ API token required for most models")
            
            st.markdown("---")
            
            # Model selection
            st.markdown("### 🎯 Model Selection")
            
            model_names = list(Config.AVAILABLE_MODELS.keys())
            model_display_names = [Config.AVAILABLE_MODELS[name].name for name in model_names]
            
            selected_model_display = st.selectbox(
                "Choose AI Model:",
                model_display_names,
                index=0,
                help="Select the AI model for conversation"
            )
            
            # Get the actual model key
            selected_model = model_names[model_display_names.index(selected_model_display)]
            model_config = Config.get_model_config(selected_model)
            
            # Display model info
            if model_config:
                status = "ready" if st.session_state.get("model_loaded", False) else "not_loaded"
                UIHelpers.display_model_info(model_config, status)
            
            st.markdown("---")
            
            # Model parameters
            st.markdown("### ⚙️ Generation Parameters")
            
            temperature = st.slider(
                "🌡️ Temperature:",
                min_value=0.1,
                max_value=2.0,
                value=model_config.temperature if model_config else 0.7,
                step=0.1,
                help="Controls randomness (lower = more focused, higher = more creative)"
            )
            
            max_tokens = st.slider(
                "📝 Max Tokens:",
                min_value=50,
                max_value=1000,
                value=model_config.max_tokens if model_config else 512,
                help="Maximum length of generated response"
            )
            
            top_p = st.slider(
                "🎯 Top-p:",
                min_value=0.1,
                max_value=1.0,
                value=model_config.top_p if model_config else 0.9,
                step=0.1,
                help="Controls diversity (lower = more focused)"
            )
            
            repetition_penalty = st.slider(
                "🔄 Repetition Penalty:",
                min_value=1.0,
                max_value=2.0,
                value=model_config.repetition_penalty if model_config else 1.1,
                step=0.1,
                help="Penalty for repeating text"
            )
            
            st.markdown("---")
            
            # System prompt configuration
            st.markdown("### 📝 System Prompt")
            
            prompt_options = list(Config.DEFAULT_SYSTEM_PROMPTS.keys())
            prompt_display_names = [name.replace('_', ' ').title() for name in prompt_options]
            
            selected_prompt_display = st.selectbox(
                "Choose prompt style:",
                prompt_display_names + ["Custom"],
                index=0
            )
            
            if selected_prompt_display == "Custom":
                system_prompt = st.text_area(
                    "Custom system prompt:",
                    value=st.session_state.get("custom_system_prompt", ""),
                    height=100,
                    help="Define how the AI should behave"
                )
                st.session_state.custom_system_prompt = system_prompt
            else:
                prompt_key = prompt_options[prompt_display_names.index(selected_prompt_display)]
                system_prompt = Config.DEFAULT_SYSTEM_PROMPTS[prompt_key]
                
                with st.expander("📖 View System Prompt"):
                    st.write(system_prompt)
            
            st.markdown("---")
            
            # Model initialization
            st.markdown("### 🚀 Model Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Load Model", type="primary", use_container_width=True):
                    if not api_token and model_config.provider == ModelProvider.HUGGINGFACE_API:
                        st.error("❌ API token required for this model")
                    else:
                        with st.spinner("Loading model..."):
                            # Update model config with current parameters
                            model_config.temperature = temperature
                            model_config.max_tokens = max_tokens
                            model_config.top_p = top_p
                            model_config.repetition_penalty = repetition_penalty
                            
                            success = self.model_manager.initialize_model(
                                selected_model, api_token, system_prompt
                            )
                            
                            if success:
                                st.session_state.model_loaded = True
                                st.session_state.current_model = model_config.name
                                st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    SessionManager.clear_messages()
                    self.model_manager.clear_chat_history()
                    st.success("Chat cleared!")
                    st.rerun()
            
            # Chat controls
            st.markdown("### 💬 Chat Options")
            
            # Export chat
            if st.session_state.messages:
                UIHelpers.export_chat_history(st.session_state.messages)
            
            # Display metrics
            metrics = SessionManager.get_conversation_metrics()
            st.markdown("### 📊 Session Stats")
            st.write(f"**Messages**: {metrics['current_messages']}")
            st.write(f"**Duration**: {metrics['session_duration']}")
            st.write(f"**Model**: {metrics['current_model']}")
            
            return {
                "selected_model": selected_model,
                "model_config": model_config,
                "api_token": api_token,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            }
    
    def render_chat_interface(self, config: Dict[str, Any]):
        """Render the main chat interface."""
        st.markdown('<h1 class="main-header">🤖 AI Agent Chat</h1>', unsafe_allow_html=True)
        
        # Display model status
        if st.session_state.get("model_loaded", False):
            st.success(f"✅ **{st.session_state.current_model}** is ready for conversation!")
        else:
            st.warning("⚠️ Please load a model from the sidebar to start chatting.")
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(f"🕒 {message['timestamp']}")
        
        # Chat input
        if user_input := st.chat_input(
            "Type your message here...", 
            disabled=not st.session_state.get("model_loaded", False)
        ):
            self.process_user_input(user_input, config)
    
    def process_user_input(self, user_input: str, config: Dict[str, Any]):
        """Process user input and generate response."""
        # Add user message
        SessionManager.add_message("user", user_input)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
            st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Generate response using model manager
                    response = self.model_manager.generate_response(user_input)
                    
                    if response.success:
                        st.markdown(response.content)
                        
                        # Add assistant message to session
                        SessionManager.add_message(
                            "assistant", 
                            response.content,
                            response.metadata
                        )
                    else:
                        error_msg = response.error or "Unknown error occurred"
                        st.error(f"❌ {error_msg}")
                        
                        # Add error message to session
                        SessionManager.add_message(
                            "assistant",
                            "I apologize, but I encountered an error. Please try again.",
                            {"error": error_msg}
                        )
                
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    
                    # Add error message to session
                    SessionManager.add_message(
                        "assistant",
                        "I apologize, but I encountered an unexpected error. Please try again.",
                        {"error": error_msg}
                    )
                
                # Add timestamp
                st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
        
        # Rerun to update the interface
        st.rerun()
    
    def render_quick_actions(self):
        """Render quick action buttons."""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        quick_prompts = [
            ("💡 Explain", "Can you explain"),
            ("📝 Summarize", "Please summarize"),
            ("🔍 Analyze", "Can you analyze"),
            ("💭 Brainstorm", "Help me brainstorm ideas about")
        ]
        
        for i, (label, prompt) in enumerate(quick_prompts):
            with [col1, col2, col3, col4][i]:
                if st.button(label, use_container_width=True):
                    st.session_state.quick_prompt = prompt
                    st.rerun()
        
        # Handle quick prompt selection
        if hasattr(st.session_state, 'quick_prompt'):
            st.text_input(
                "Complete your question:",
                value=st.session_state.quick_prompt + " ",
                key="quick_input",
                on_change=self.handle_quick_input
            )
    
    def handle_quick_input(self):
        """Handle quick input submission."""
        if st.session_state.quick_input and st.session_state.get("model_loaded", False):
            # Process the quick input
            config = {
                "selected_model": st.session_state.get("current_model", ""),
                "api_token": st.session_state.get("api_token", "")
            }
            self.process_user_input(st.session_state.quick_input, config)
            
            # Clear the quick prompt
            if hasattr(st.session_state, 'quick_prompt'):
                delattr(st.session_state, 'quick_prompt')
    
    def render_conversation_examples(self):
        """Render conversation examples."""
        with st.expander("💬 Conversation Examples", expanded=False):
            examples = [
                {
                    "title": "📚 Learning Assistant",
                    "prompt": "Explain quantum computing in simple terms",
                    "description": "Get clear explanations of complex topics"
                },
                {
                    "title": "💻 Coding Helper", 
                    "prompt": "How do I create a REST API in Python?",
                    "description": "Get programming help and code examples"
                },
                {
                    "title": "✍️ Creative Writing",
                    "prompt": "Write a short story about a robot learning to paint",
                    "description": "Generate creative content and stories"
                },
                {
                    "title": "🔍 Research Assistant",
                    "prompt": "What are the benefits of renewable energy?",
                    "description": "Get research help and information summaries"
                }
            ]
            
            for example in examples:
                st.markdown(f"**{example['title']}**")
                st.write(f"*{example['description']}*")
                
                if st.button(f"Try: {example['prompt'][:50]}...", key=f"example_{example['title']}"):
                    if st.session_state.get("model_loaded", False):
                        config = {
                            "api_token": st.session_state.get("api_token", "")
                        }
                        self.process_user_input(example['prompt'], config)
                    else:
                        st.warning("Please load a model first!")
                
                st.markdown("---")
    
    def render(self):
        """Render the complete AI Agent page."""
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Main chat interface
            self.render_chat_interface(config)
        
        with col2:
            # Quick actions
            self.render_quick_actions()
            
            # Conversation examples
            self.render_conversation_examples()
        
        # Usage tips at the bottom
        UIHelpers.display_usage_tips()
