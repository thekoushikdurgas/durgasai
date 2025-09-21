"""
HuggingFace Model Selection and Chat Page.

This page allows users to:
1. Input their HuggingFace API key
2. Fetch and select from available HuggingFace models
3. Chat with the selected model
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_manager import ModelManager, ModelResponse
from utils.logger import debug, info, warning, error, log_session_event, log_user_action
from utils.ui_helpers import UIHelpers, SessionManager
from components.chat.chat_interface import ChatInterface


class HuggingFacePage:
    """
    HuggingFace model selection and chat page.
    
    This class handles:
    - HuggingFace API authentication
    - Model fetching and selection
    - Chat interface with selected models
    - Session management for HuggingFace models
    """
    
    def __init__(self):
        """Initialize the HuggingFace page."""
        self.model_manager = ModelManager()
        self.available_models = []
        self.model_cache = {}
        debug("HuggingFacePage initialized", "hf_page")
    
    def fetch_huggingface_models(self, api_token: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch available models from HuggingFace API.
        
        Args:
            api_token (str): HuggingFace API token
            limit (int): Maximum number of models to fetch
            
        Returns:
            List[Dict[str, Any]]: List of available models
        """
        if not api_token or not api_token.startswith('hf_'):
            return []
        
        try:
            headers = {"Authorization": f"Bearer {api_token}"}
            
            # Fetch models from HuggingFace API
            url = "https://huggingface.co/api/models"
            params = {
                "pipeline_tag": "text-generation",  # Focus on text generation models
                "limit": limit,
                "sort": "downloads",  # Sort by popularity
                "direction": -1
            }
            
            with st.spinner("🔄 Fetching available models from HuggingFace..."):
                response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                models_data = response.json()
                
                # Filter and format models
                filtered_models = []
                for model in models_data:
                    if self._is_valid_model(model):
                        filtered_models.append({
                            "id": model.get("id", ""),
                            "name": model.get("modelId", ""),
                            "pipeline_tag": model.get("pipeline_tag", ""),
                            "downloads": model.get("downloads", 0),
                            "tags": model.get("tags", []),
                            "created_at": model.get("created_at", ""),
                            "last_modified": model.get("last_modified", "")
                        })
                
                info(f"Fetched {len(filtered_models)} valid models from HuggingFace", "hf_page")
                log_session_event("models_fetched", 
                    api_used="huggingface",
                    models_count=len(filtered_models),
                    limit_requested=limit)
                
                return filtered_models
            
            else:
                error_msg = f"Failed to fetch models: {response.status_code} - {response.text}"
                error(error_msg, "hf_page")
                st.error(f"❌ {error_msg}")
                return []
                
        except requests.exceptions.Timeout:
            error("Timeout while fetching models from HuggingFace API", "hf_page")
            st.error("⏱️ Request timed out. Please try again.")
            return []
        except Exception as e:
            error_msg = f"Error fetching models: {str(e)}"
            error(error_msg, "hf_page", e)
            st.error(f"❌ {error_msg}")
            return []
    
    def _is_valid_model(self, model: Dict[str, Any]) -> bool:
        """
        Check if a model is valid for chat/text generation.
        
        Args:
            model (Dict[str, Any]): Model data from API
            
        Returns:
            bool: True if model is valid
        """
        # Check if model has required fields
        if not model.get("id") or not model.get("modelId"):
            return False
        
        # Check pipeline tag
        pipeline_tag = model.get("pipeline_tag", "")
        valid_tags = ["text-generation", "conversational", "text2text-generation"]
        
        if pipeline_tag not in valid_tags:
            return False
        
        # Check if model is not too old (basic filter)
        downloads = model.get("downloads", 0)
        if downloads < 100:  # Filter out very unpopular models
            return False
        
        # Check tags for exclusions
        tags = model.get("tags", [])
        excluded_tags = ["deprecated", "archived", "gated"]
        
        if any(tag in tags for tag in excluded_tags):
            return False
        
        return True
    
    def render_api_key_section(self) -> str:
        """
        Render API key input section in the sidebar.
        
        Returns:
            str: The entered API key
        """
        with st.sidebar:
            st.markdown("### 🔑 HuggingFace API Configuration")
            
            # Load API key from config.json first, then session state
            api_key_from_config = self._load_api_key_from_config()
            current_api_key = api_key_from_config or st.session_state.get("hf_api_token", "")
            
            # API key input
            api_key = st.text_input(
                "Enter your HuggingFace API Token:",
                value=current_api_key,
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens",
                key="hf_api_key_input"
            )
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save", key="save_hf_api_key"):
                    if api_key:
                        self._save_api_key_to_config(api_key)
                        st.session_state.hf_api_token = api_key
                        st.success("✅ API key saved!")
                        st.rerun()
                    else:
                        st.error("❌ Please enter a valid API key")
            
            with col2:
                if st.button("🧪 Test", key="test_hf_api_key"):
                    if api_key:
                        self._test_api_key(api_key)
                    else:
                        st.error("❌ Please enter an API key first")
            
            # Show current status
            if api_key:
                st.success("✅ API key configured")
                if api_key.startswith('hf_'):
                    st.success("✅ Token format looks valid")
                else:
                    st.warning("⚠️ Token should start with 'hf_'")
            else:
                st.info("ℹ️ Enter your HuggingFace API token to access models")
            
            st.markdown("---")
            
            # Model status and info
            if api_key:
                st.markdown("#### 📊 Current Status")
                if hasattr(st.session_state, 'hf_selected_model') and st.session_state.hf_selected_model:
                    st.success(f"✅ Model: {st.session_state.hf_selected_model}")
                else:
                    st.info("ℹ️ No model selected")
                
                # Show chat message count
                if hasattr(st.session_state, 'hf_chat_messages'):
                    message_count = len(st.session_state.hf_chat_messages)
                    st.metric("Messages", message_count)
                
                # Show available models count
                if hasattr(st.session_state, 'hf_models_cache') and st.session_state.hf_models_cache:
                    models_count = len(st.session_state.hf_models_cache)
                    st.metric("Available Models", models_count)
            
            st.markdown("---")
            
            # Quick links
            st.markdown("#### 🔗 Quick Links")
            st.markdown("""
            - [Get API Token](https://huggingface.co/settings/tokens)
            - [Model Hub](https://huggingface.co/models)
            - [Documentation](https://huggingface.co/docs)
            - [Inference API](https://huggingface.co/docs/api/inference)
            """)
            
            return api_key
    
    def _load_api_key_from_config(self) -> Optional[str]:
        """Load HuggingFace API key from distributed config files."""
        try:
            from services import ConfigService
            config_service = ConfigService()
            
            hf_config = config_service.get_huggingface_config()
            return hf_config.get('api_keys', '')
        except Exception as e:
            debug(f"Error loading API key from config: {e}", "hf_page")
        
        return None
    
    def _save_api_key_to_config(self, api_key: str) -> None:
        """Save HuggingFace API key to distributed config files."""
        try:
            from services import ConfigService
            config_service = ConfigService()
            
            # Load existing API config
            api_config = config_service.get_api_config()
            
            # Ensure huggingface section exists
            if 'huggingface' not in api_config:
                api_config['huggingface'] = {}
            
            # Update API key
            api_config['huggingface']['api_keys'] = api_key
            
            # Save updated config
            success = config_service.save_config('api', api_config)
            
            if success:
                info("HuggingFace API key saved to config/api_config.json", "hf_page")
            else:
                raise Exception("Failed to save API configuration")
            
        except Exception as e:
            error(f"Error saving API key to config: {e}", "hf_page")
            st.error(f"❌ Error saving API key: {e}")
    
    def _test_api_key(self, api_key: str):
        """Test the API key with a simple request."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = "https://huggingface.co/api/whoami"
            
            with st.spinner("Testing API key..."):
                response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                user_info = response.json()
                st.success(f"✅ API key valid! Welcome, {user_info.get('name', 'User')}")
                log_user_action("api_key_tested", success=True)
            else:
                st.error(f"❌ API key test failed: {response.status_code}")
                log_user_action("api_key_tested", success=False)
                
        except Exception as e:
            st.error(f"❌ Error testing API key: {str(e)}")
            log_user_action("api_key_tested", success=False, error=str(e))
    
    def render_model_selection(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Render model selection section in the sidebar.
        
        Args:
            api_key (str): HuggingFace API key
            
        Returns:
            Optional[Dict[str, Any]]: Selected model configuration
        """
        if not api_key:
            return None
        
        with st.sidebar:
            st.markdown("### 🤖 Model Selection")
            
            # Fetch models button
            if st.button("🔄 Fetch Models", type="primary", use_container_width=True):
                # Clear cache and fetch new models
                if 'hf_models_cache' in st.session_state:
                    del st.session_state.hf_models_cache
                st.rerun()
            
            # Check if we have cached models
            if 'hf_models_cache' not in st.session_state:
                with st.spinner("Fetching models from HuggingFace API..."):
                    models = self.fetch_huggingface_models(api_key)
                    if models:
                        st.session_state.hf_models_cache = models
                        st.success(f"✅ Loaded {len(models)} models")
                    else:
                        st.error("❌ Failed to load models")
                        return None
            
            models = st.session_state.get('hf_models_cache', [])
            
            if not models:
                st.warning("No models available. Please try fetching again.")
                return None
            
            # Model selection dropdown
            model_options = []
            for model in models:
                display_name = f"{model['name']} ({model['downloads']:,} downloads)"
                model_options.append((display_name, model))
            
            if not model_options:
                st.warning("No valid models found")
                return None
            
            selected_display = st.selectbox(
                "Choose a model:",
                [option[0] for option in model_options],
                help="Select a HuggingFace model for chat"
            )
            
            # Get selected model
            selected_model = None
            for display_name, model in model_options:
                if display_name == selected_display:
                    selected_model = model
                    break
            
            if selected_model:
                # Store selected model in session state
                st.session_state.hf_selected_model = selected_model['name']
                
                # Display model information in expander
                with st.expander("📋 Model Info", expanded=False):
                    st.write(f"**Model:** {selected_model['name']}")
                    st.write(f"**Pipeline:** {selected_model['pipeline_tag']}")
                    st.write(f"**Downloads:** {selected_model['downloads']:,}")
                    
                    if selected_model.get('tags'):
                        st.write("**Tags:**")
                        tags_text = ", ".join(selected_model['tags'][:8])
                        st.write(tags_text)
                
                return selected_model
            
            return None
    
    def render_chat_interface(self, selected_model: Dict[str, Any], api_key: str):
        """
        Render chat interface for the selected model.
        
        Args:
            selected_model (Dict[str, Any]): Selected model configuration
            api_key (str): HuggingFace API key
        """
        # Initialize chat messages if not exists
        if 'hf_chat_messages' not in st.session_state:
            st.session_state.hf_chat_messages = []
        
        # Initialize model configuration if not exists
        if 'hf_model_config' not in st.session_state:
            st.session_state.hf_model_config = {
                'temperature': 0.7,
                'max_tokens': 512,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'do_sample': True,
                'early_stopping': True,
                'num_return_sequences': 1
            }
        
        # Main container with model details and configuration
        with st.container():
            # Model Header
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
                <h1 style="color: white; margin: 0;">🤗 {selected_model['name']}</h1>
                <p style="color: #f0f0f0; margin: 5px 0;">{selected_model['pipeline_tag']} • {selected_model['downloads']:,} downloads</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Model Details", "⚙️ Configuration", "💬 Chat Interface", "🚀 Modern Features", "📤 Model Sharing", "📈 Usage Stats"])
            
            with tab1:
                self._render_model_details(selected_model)
            
            with tab2:
                self._render_model_configuration()
            
            with tab3:
                self._render_chat_interface_tab(selected_model, api_key)
            
            with tab4:
                self._render_modern_features(selected_model, api_key)
            
            with tab5:
                self._render_model_sharing(selected_model, api_key)
            
            with tab6:
                self._render_usage_statistics()
    
    def _render_model_details(self, selected_model: Dict[str, Any]):
        """Render comprehensive model details."""
        st.markdown("### 📊 Model Information")
        
        # Basic information in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Downloads", f"{selected_model['downloads']:,}")
        
        with col2:
            st.metric("Pipeline", selected_model['pipeline_tag'])
        
        with col3:
            if selected_model.get('created_at'):
                created_date = selected_model['created_at'][:10]
                st.metric("Created", created_date)
        
        # Detailed model information
        st.markdown("#### 🔍 Detailed Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("**Model Specifications:**")
            st.write(f"• **Model ID:** `{selected_model['name']}`")
            st.write(f"• **Pipeline Tag:** `{selected_model['pipeline_tag']}`")
            st.write(f"• **Downloads:** `{selected_model['downloads']:,}`")
            
            if selected_model.get('last_modified'):
                st.write(f"• **Last Modified:** `{selected_model['last_modified'][:10]}`")
        
        with info_col2:
            st.markdown("**Model Tags:**")
            if selected_model.get('tags'):
                # Display tags in a more organized way
                tag_cols = st.columns(3)
                for i, tag in enumerate(selected_model['tags'][:12]):
                    with tag_cols[i % 3]:
                        st.code(tag, language=None)
                
                if len(selected_model['tags']) > 12:
                    st.write(f"... and {len(selected_model['tags']) - 12} more tags")
        
        # Model description and links
        st.markdown("#### 🔗 Quick Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🌐 View on HuggingFace", use_container_width=True):
                st.markdown(f"[Open Model Page](https://huggingface.co/{selected_model['name']})")
        
        with action_col2:
            if st.button("📚 View Documentation", use_container_width=True):
                st.markdown(f"[Model Documentation](https://huggingface.co/{selected_model['name']}#model-card)")
        
        with action_col3:
            if st.button("💻 Copy Model ID", use_container_width=True):
                st.code(selected_model['name'], language=None)
    
    def _render_model_configuration(self):
        """Render model configuration parameters."""
        st.markdown("### ⚙️ Model Configuration")
        st.markdown("Configure the parameters for text generation with this model.")
        
        config = st.session_state.hf_model_config
        
        # Configuration in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎛️ Generation Parameters")
            
            config['temperature'] = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=config['temperature'],
                step=0.1,
                help="Controls randomness. Lower values make output more deterministic."
            )
            
            config['max_tokens'] = st.slider(
                "Max Tokens",
                min_value=10,
                max_value=2048,
                value=config['max_tokens'],
                step=10,
                help="Maximum number of tokens to generate."
            )
            
            config['top_p'] = st.slider(
                "Top P (Nucleus Sampling)",
                min_value=0.1,
                max_value=1.0,
                value=config['top_p'],
                step=0.05,
                help="Controls diversity via nucleus sampling."
            )
            
            config['top_k'] = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=config['top_k'],
                help="Limits vocabulary to top K most likely tokens."
            )
        
        with col2:
            st.markdown("#### 🔧 Advanced Parameters")
            
            config['repetition_penalty'] = st.slider(
                "Repetition Penalty",
                min_value=1.0,
                max_value=2.0,
                value=config['repetition_penalty'],
                step=0.1,
                help="Penalizes repetition of tokens."
            )
            
            config['do_sample'] = st.checkbox(
                "Enable Sampling",
                value=config['do_sample'],
                help="Enable sampling for generation."
            )
            
            config['early_stopping'] = st.checkbox(
                "Early Stopping",
                value=config['early_stopping'],
                help="Stop generation when EOS token is encountered."
            )
            
            config['num_return_sequences'] = st.selectbox(
                "Number of Sequences",
                options=[1, 2, 3, 4, 5],
                index=config['num_return_sequences'] - 1,
                help="Number of sequences to generate."
            )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            st.session_state.hf_model_config = config
            st.success("✅ Configuration saved!")
        
        # Reset to defaults
        if st.button("🔄 Reset to Defaults"):
            st.session_state.hf_model_config = {
                'temperature': 0.7,
                'max_tokens': 512,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'do_sample': True,
                'early_stopping': True,
                'num_return_sequences': 1
            }
            st.success("✅ Configuration reset to defaults!")
            st.rerun()
        
        # Display current configuration
        with st.expander("📋 Current Configuration", expanded=False):
            st.json(config)
    
    def _render_chat_interface_tab(self, selected_model: Dict[str, Any], api_key: str):
        """Render the chat interface in a dedicated tab."""
        st.markdown("### 💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.hf_chat_messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message.get("timestamp"):
                        st.caption(f"At {message['timestamp'][:19]}")
        
        # Chat input with current configuration info
        config = st.session_state.hf_model_config
        prompt = st.chat_input(
            f"Chat with {selected_model['name']} (temp: {config['temperature']}, max_tokens: {config['max_tokens']})..."
        )
        
        if prompt:
            # Add user message
            st.session_state.hf_chat_messages.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response with current configuration
            with st.chat_message("assistant"):
                with st.spinner("🤖 Generating response..."):
                    response = self._generate_huggingface_response(
                        selected_model, api_key, prompt
                    )
                
                if response.success:
                    st.write(response.content)
                    
                    # Add assistant message
                    st.session_state.hf_chat_messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "timestamp": datetime.now().isoformat(),
                        "model": selected_model['name'],
                        "config": config.copy()
                    })
                else:
                    st.error(f"❌ {response.error}")
                    
                    # Add error message
                    st.session_state.hf_chat_messages.append({
                        "role": "assistant",
                        "content": f"Error: {response.error}",
                        "timestamp": datetime.now().isoformat(),
                        "error": True
                    })
            
            st.rerun()
    
    def _render_modern_features(self, selected_model: Dict[str, Any], api_key: str):
        """Render modern Transformers features demonstration."""
        st.markdown("### 🚀 Modern Transformers Features")
        st.markdown("This section demonstrates the latest features from Hugging Face Transformers based on the official documentation.")
        
        # Feature 1: Device Detection
        st.markdown("#### 🔍 Device Detection")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Detect Optimal Device", use_container_width=True):
                try:
                    from transformers import infer_device
                    device = infer_device()
                    st.success(f"✅ Optimal device detected: {device}")
                    
                    # Show device capabilities
                    if device.startswith('cuda'):
                        import torch
                        if torch.cuda.is_available():
                            gpu_name = torch.cuda.get_device_name(0)
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            st.info(f"🎮 GPU: {gpu_name}")
                            st.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                    else:
                        st.info("💻 Using CPU for inference")
                        
                except Exception as e:
                    st.error(f"❌ Error detecting device: {e}")
        
        with col2:
            if st.button("📊 Show Model Info", use_container_width=True):
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(selected_model['name'])
                    
                    info_data = {
                        "Model Type": config.model_type if hasattr(config, 'model_type') else "Unknown",
                        "Architecture": config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "Unknown",
                        "Hidden Size": config.hidden_size if hasattr(config, 'hidden_size') else "Unknown",
                        "Attention Heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else "Unknown",
                        "Layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else "Unknown"
                    }
                    
                    for key, value in info_data.items():
                        st.write(f"**{key}:** {value}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading model info: {e}")
        
        # Feature 2: Pipeline Comparison
        st.markdown("#### ⚖️ Pipeline Comparison")
        
        if st.button("🔄 Compare Pipeline Types"):
            try:
                from transformers import pipeline, infer_device
                device = infer_device()
                
                # Test different pipeline types
                pipelines_to_test = [
                    ("text-generation", "Text Generation"),
                    ("sentiment-analysis", "Sentiment Analysis"),
                    ("feature-extraction", "Feature Extraction")
                ]
                
                test_text = "Hugging Face Transformers is amazing!"
                
                results = []
                for task, name in pipelines_to_test:
                    try:
                        pipe = pipeline(task, model=selected_model['name'], device=device)
                        result = pipe(test_text)
                        
                        if task == "text-generation":
                            generated_text = result[0]['generated_text'][:100] + "..."
                            results.append(f"**{name}:** {generated_text}")
                        elif task == "sentiment-analysis":
                            label = result[0]['label']
                            score = result[0]['score']
                            results.append(f"**{name}:** {label} (confidence: {score:.3f})")
                        else:
                            results.append(f"**{name}:** ✅ Features extracted successfully")
                            
                    except Exception as e:
                        results.append(f"**{name}:** ❌ {str(e)[:50]}...")
                
                st.markdown("**Pipeline Test Results:**")
                for result in results:
                    st.write(result)
                    
            except Exception as e:
                st.error(f"❌ Error testing pipelines: {e}")
        
        # Feature 3: Advanced Configuration
        st.markdown("#### ⚙️ Advanced Configuration")
        
        with st.expander("🔧 Model Loading Options", expanded=False):
            st.markdown("**Modern Loading Parameters:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.code("""
# Optimal model loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatic device allocation
    dtype="auto",       # Optimal data type
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
                """, language="python")
            
            with col2:
                st.code("""
# Modern pipeline creation
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
                """, language="python")
        
        # Feature 4: Batch Processing Demo
        st.markdown("#### 📦 Batch Processing Demo")
        
        if st.button("🚀 Test Batch Processing"):
            try:
                from transformers import pipeline, infer_device
                device = infer_device()
                
                # Create a simple pipeline for batch testing
                pipe = pipeline("sentiment-analysis", device=device)
                
                # Sample texts for batch processing
                batch_texts = [
                    "I love this product!",
                    "This is terrible.",
                    "It's okay, nothing special.",
                    "Absolutely amazing!",
                    "I hate this so much."
                ]
                
                st.markdown("**Processing batch of texts:**")
                
                # Process in batch
                results = pipe(batch_texts)
                
                for i, (text, result) in enumerate(zip(batch_texts, results)):
                    sentiment = result['label']
                    confidence = result['score']
                    st.write(f"{i+1}. **'{text}'** → {sentiment} ({confidence:.3f})")
                
                st.success(f"✅ Successfully processed {len(batch_texts)} texts in batch!")
                
            except Exception as e:
                st.error(f"❌ Error in batch processing: {e}")
        
        # Feature 5: Model Quantization Info
        st.markdown("#### 💾 Memory Optimization")
        
        with st.expander("🔧 Quantization Options", expanded=False):
            st.markdown("""
            **Memory Optimization Techniques:**
            
            - **8-bit Quantization**: Reduce model size by ~50%
            - **4-bit Quantization**: Reduce model size by ~75%
            - **Gradient Checkpointing**: Reduce memory during training
            - **Flash Attention**: Faster attention computation
            
            **Example Usage:**
            ```python
            # 8-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
            ```
            """)
            
            if st.button("💾 Check Memory Usage"):
                try:
                    import torch
                    import psutil
                    
                    # Get system memory info
                    memory = psutil.virtual_memory()
                    gpu_memory = None
                    
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        gpu_used = torch.cuda.memory_allocated(0) / 1024**3
                        gpu_free = gpu_memory - gpu_used
                        
                        st.metric("GPU Memory Total", f"{gpu_memory:.1f} GB")
                        st.metric("GPU Memory Used", f"{gpu_used:.1f} GB")
                        st.metric("GPU Memory Free", f"{gpu_free:.1f} GB")
                    else:
                        st.info("No GPU available for memory monitoring")
                    
                    st.metric("RAM Total", f"{memory.total / 1024**3:.1f} GB")
                    st.metric("RAM Available", f"{memory.available / 1024**3:.1f} GB")
                    st.metric("RAM Used", f"{memory.used / 1024**3:.1f} GB")
                    
                except Exception as e:
                    st.error(f"❌ Error checking memory: {e}")
    
    def _render_model_sharing(self, selected_model: Dict[str, Any], api_key: str):
        """Render model sharing interface."""
        st.markdown("### 📤 Model Sharing")
        st.markdown("Share your trained models or fine-tuned versions to the HuggingFace Hub.")
        
        # Initialize sharing status
        if 'sharing_initialized' not in st.session_state:
            st.session_state.sharing_initialized = False
        
        # Sharing configuration section
        with st.expander("🔧 Sharing Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Repository Settings")
                
                # Get username from API key
                username = st.text_input(
                    "HuggingFace Username:",
                    value=st.session_state.get('hf_username', ''),
                    help="Your HuggingFace username"
                )
                
                model_name = st.text_input(
                    "Model Name:",
                    value=st.session_state.get('model_name', selected_model['name'].split('/')[-1]),
                    help="Name for your shared model"
                )
                
                repo_id = st.text_input(
                    "Repository ID:",
                    value=f"{username}/{model_name}" if username and model_name else "",
                    help="Full repository ID (username/model-name)",
                    disabled=True
                )
                
            with col2:
                st.markdown("#### ⚙️ Sharing Options")
                
                is_private = st.checkbox(
                    "Private Repository",
                    value=False,
                    help="Make the repository private"
                )
                
                is_gated = st.checkbox(
                    "Gated Model",
                    value=False,
                    help="Require approval to access the model"
                )
                
                license_type = st.selectbox(
                    "License:",
                    options=["apache-2.0", "mit", "gpl-3.0", "bsd-3-clause", "other"],
                    index=0,
                    help="Choose a license for your model"
                )
        
        # Model card creation section
        with st.expander("📝 Model Card", expanded=False):
            st.markdown("Create a comprehensive model card for your shared model.")
            
            model_description = st.text_area(
                "Model Description:",
                value=f"This is a fine-tuned version of {selected_model['name']} for improved performance.",
                height=100,
                help="Describe what your model does and how it's different"
            )
            
            # Performance metrics
            st.markdown("#### 📊 Performance Metrics")
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                accuracy = st.number_input("Accuracy:", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
                f1_score = st.number_input("F1 Score:", value=0.94, min_value=0.0, max_value=1.0, step=0.01)
            
            with perf_col2:
                inference_time = st.number_input("Inference Time (ms):", value=2.3, min_value=0.0, step=0.1)
                memory_usage = st.number_input("Memory Usage (GB):", value=4.2, min_value=0.0, step=0.1)
            
            # Usage example
            usage_example = st.text_area(
                "Usage Example:",
                value=f"""from transformers import pipeline

generator = pipeline('text-generation', model='{repo_id}')
result = generator("Hello, I am")""",
                height=100,
                help="Python code example showing how to use your model"
            )
            
            limitations = st.text_area(
                "Limitations and Bias:",
                value="- The model may exhibit bias present in the training data\n- Performance may vary across different domains\n- Limited to English language processing",
                height=80,
                help="Describe any limitations or biases of your model"
            )
        
        # Initialize sharing manager
        if st.button("🔐 Initialize Sharing Manager", type="primary"):
            if api_key:
                with st.spinner("Initializing HuggingFace sharing manager..."):
                    success = self.model_manager.initialize_sharing_manager(api_key)
                    
                    if success:
                        st.session_state.sharing_initialized = True
                        st.session_state.hf_username = username
                        st.session_state.model_name = model_name
                        st.success("✅ Sharing manager initialized successfully!")
                        
                        # Show sharing status
                        status = self.model_manager.get_sharing_status()
                        if status.get('current_user'):
                            st.info(f"📝 Authenticated as: {status['current_user']}")
                    else:
                        st.error("❌ Failed to initialize sharing manager. Please check your API key.")
            else:
                st.error("❌ Please configure your HuggingFace API key first.")
        
        # Model sharing section
        if st.session_state.get('sharing_initialized', False):
            st.markdown("---")
            st.markdown("### 🚀 Share Your Model")
            
            # Show current configuration
            st.markdown("#### 📋 Current Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.write(f"**Repository:** `{repo_id}`")
                st.write(f"**Private:** {is_private}")
                st.write(f"**Gated:** {is_gated}")
            
            with config_col2:
                st.write(f"**License:** {license_type}")
                st.write(f"**Base Model:** {selected_model['name']}")
                st.write(f"**Pipeline:** {selected_model['pipeline_tag']}")
            
            # Share model button
            if st.button("📤 Share Model to HuggingFace Hub", type="primary", use_container_width=True):
                if repo_id and username and model_name:
                    # Create model card
                    performance_metrics = {
                        "Accuracy": f"{accuracy:.3f}",
                        "F1 Score": f"{f1_score:.3f}",
                        "Inference Time": f"{inference_time}ms",
                        "Memory Usage": f"{memory_usage}GB"
                    }
                    
                    with st.spinner("Creating model card and sharing model..."):
                        # Create model card
                        model_card = self.model_manager.create_model_card(
                            model_name=model_name,
                            description=model_description,
                            performance_metrics=performance_metrics,
                            usage_example=usage_example,
                            limitations=limitations,
                            license=license_type
                        )
                        
                        # For demo purposes, we'll simulate sharing
                        # In a real implementation, you would:
                        # 1. Save model files to a temporary directory
                        # 2. Create the model card file
                        # 3. Call model_manager.share_model()
                        
                        st.success("🎉 Model sharing simulation complete!")
                        st.info("📝 In a real implementation, your model would be uploaded to the HuggingFace Hub.")
                        
                        # Show the generated model card
                        with st.expander("📄 Generated Model Card", expanded=False):
                            st.markdown(model_card)
                        
                        # Show sharing results
                        hub_url = f"https://huggingface.co/{repo_id}"
                        st.markdown(f"🌐 **Model Hub URL:** [{hub_url}]({hub_url})")
                        
                        # Log the sharing event
                        log_user_action("model_sharing_attempted", 
                                      repo_id=repo_id,
                                      private=is_private,
                                      gated=is_gated,
                                      license=license_type)
                else:
                    st.error("❌ Please fill in all required fields (username, model name).")
        
        # Sharing guide section
        with st.expander("📚 Sharing Guide", expanded=False):
            st.markdown("""
            ### 🤗 How to Share Models
            
            **Step 1: Prepare Your Model**
            - Train or fine-tune your model
            - Save model files (pytorch_model.bin, config.json, etc.)
            - Test your model thoroughly
            
            **Step 2: Create Repository**
            - Enter your HuggingFace username
            - Choose a descriptive model name
            - Set privacy and gating options
            
            **Step 3: Create Model Card**
            - Write a clear description
            - Add performance metrics
            - Include usage examples
            - Document limitations and biases
            
            **Step 4: Share**
            - Click "Share Model to HuggingFace Hub"
            - Your model will be uploaded and publicly available
            - Others can discover and use your model
            
            ### 🎯 Best Practices
            
            - **Clear Naming:** Use descriptive names for your models
            - **Comprehensive Documentation:** Include detailed model cards
            - **Performance Metrics:** Share benchmarks and evaluation results
            - **Usage Examples:** Provide clear code examples
            - **Ethical Considerations:** Document biases and limitations
            - **License:** Choose appropriate licenses for your use case
            
            ### 🔗 Resources
            
            - [HuggingFace Hub Documentation](https://huggingface.co/docs/hub)
            - [Model Card Guide](https://huggingface.co/docs/hub/model-cards)
            - [Sharing Best Practices](https://huggingface.co/docs/hub/models-adding-libraries)
            """)
    
    def _render_usage_statistics(self):
        """Render usage statistics and analytics."""
        st.markdown("### 📈 Usage Statistics")
        
        messages = st.session_state.get('hf_chat_messages', [])
        
        if not messages:
            st.info("No chat history available. Start chatting to see statistics!")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_messages = len(messages)
            st.metric("Total Messages", total_messages)
        
        with col2:
            user_messages = len([m for m in messages if m.get("role") == "user"])
            st.metric("User Messages", user_messages)
        
        with col3:
            assistant_messages = len([m for m in messages if m.get("role") == "assistant"])
            st.metric("Assistant Messages", assistant_messages)
        
        with col4:
            error_messages = len([m for m in messages if m.get("error")])
            st.metric("Errors", error_messages)
        
        # Message length statistics
        st.markdown("#### 📊 Message Analysis")
        
        if messages:
            user_lengths = [len(m["content"]) for m in messages if m.get("role") == "user"]
            assistant_lengths = [len(m["content"]) for m in messages if m.get("role") == "assistant"]
            
            if user_lengths:
                avg_user_length = sum(user_lengths) / len(user_lengths)
                st.metric("Avg User Message Length", f"{avg_user_length:.1f} chars")
            
            if assistant_lengths:
                avg_assistant_length = sum(assistant_lengths) / len(assistant_lengths)
                st.metric("Avg Assistant Message Length", f"{avg_assistant_length:.1f} chars")
        
        # Recent activity
        st.markdown("#### 🕐 Recent Activity")
        recent_messages = messages[-5:] if messages else []
        
        for msg in recent_messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            timestamp = msg.get("timestamp", "")[:19] if msg.get("timestamp") else "Unknown"
            
            st.write(f"{role_icon} **{msg['role'].title()}** ({timestamp}): {content_preview}")
        
        # Export chat history
        if st.button("📥 Export Chat History"):
            import json
            chat_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_messages": len(messages),
                "messages": messages
            }
            
            json_data = json.dumps(chat_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Chat History",
                data=json_data,
                file_name=f"huggingface_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _generate_huggingface_response(self, model: Dict[str, Any], api_key: str, 
                                     prompt: str) -> ModelResponse:
        """
        Generate response using HuggingFace Inference API.
        
        Args:
            model (Dict[str, Any]): Model configuration
            api_key (str): API key
            prompt (str): User prompt
            
        Returns:
            ModelResponse: Generated response
        """
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"https://api-inference.huggingface.co/models/{model['name']}"
            
            # Get current configuration from session state
            config = st.session_state.get('hf_model_config', {
                'max_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True,
                'repetition_penalty': 1.1
            })
            
            # Prepare payload with current configuration
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": config.get('max_tokens', 512),
                    "temperature": config.get('temperature', 0.7),
                    "top_p": config.get('top_p', 0.9),
                    "repetition_penalty": config.get('repetition_penalty', 1.1),
                    "do_sample": config.get('do_sample', True),
                    "top_k": config.get('top_k', 50),
                    "early_stopping": config.get('early_stopping', True),
                    "num_return_sequences": config.get('num_return_sequences', 1),
                    "return_full_text": False
                }
            }
            
            start_time = datetime.now()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "").strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    content = result["generated_text"].strip()
                else:
                    content = str(result).strip()
                
                return ModelResponse(
                    content=content,
                    success=True,
                    metadata={
                        "model": model['name'],
                        "response_time": response_time,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return ModelResponse(
                    content="",
                    success=False,
                    error=error_msg,
                    metadata={
                        "model": model['name'],
                        "response_time": response_time
                    }
                )
                
        except requests.exceptions.Timeout:
            return ModelResponse(
                content="",
                success=False,
                error="Request timed out. Please try again.",
                metadata={"model": model['name']}
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error=f"Error: {str(e)}",
                metadata={"model": model['name']}
            )
    
    def render_clear_chat_button(self):
        """Render clear chat button in the sidebar."""
        with st.sidebar:
            st.markdown("---")
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.hf_chat_messages = []
                st.success("Chat cleared!")
                st.rerun()
    
    def render_installation_guide(self):
        """Render the Transformers installation guide section."""
        with st.expander("📚 Transformers Installation Guide", expanded=False):
            st.markdown("""
            ### 🤗 Hugging Face Transformers Installation
            
            **Hugging Face Transformers** is a state-of-the-art machine learning library that provides thousands of pre-trained models for NLP, Computer Vision, Audio, and Multimodal tasks.
            
            #### 🚀 Quick Installation
            
            **1. Create Virtual Environment (Recommended)**
            ```bash
            python -m venv transformers_env
            # Windows:
            transformers_env\\Scripts\\activate
            # Linux/macOS:
            source transformers_env/bin/activate
            ```
            
            **2. Install Transformers**
            ```bash
            # Basic installation
            pip install transformers
            
            # With PyTorch support (recommended)
            pip install transformers[torch]
            
            # Or install from our requirements file
            pip install -r requirements-transformers.txt
            ```
            
            **3. Test Installation**
            ```python
            from transformers import pipeline
            print(pipeline('sentiment-analysis')('Hugging Face is amazing!'))
            ```
            
            #### 📋 System Requirements
            - **Python**: 3.9+ (recommended 3.10+)
            - **PyTorch**: 2.2+ (or TensorFlow 2.12+)
            - **Memory**: Minimum 4GB RAM (8GB+ recommended)
            - **Storage**: 2GB+ free space for model cache
            
            #### 🎮 GPU Support (Optional)
            If you have an NVIDIA GPU:
            ```bash
            # Check GPU availability
            nvidia-smi
            ```
            
            #### 📖 Full Documentation
            - [Official Docs](https://huggingface.co/docs/transformers)
            - [Model Hub](https://huggingface.co/models)
            - [Installation Guide](https://huggingface.co/docs/transformers/installation)
            - [Quickstart](https://huggingface.co/docs/transformers/quickstart)
            
            #### 🔧 Troubleshooting
            - **Import Error**: Make sure you're in the correct virtual environment
            - **CUDA Issues**: Install appropriate CUDA drivers for PyTorch
            - **Memory Issues**: Use smaller models or enable gradient checkpointing
            - **Download Issues**: Check internet connection or use offline mode
            """)

    def render(self):
        """Render the complete HuggingFace page."""
        # Main header with installation guide
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🤗 HuggingFace Model Chat</h1>
            <p style="color: #f0f0f0; margin: 5px 0;">Access thousands of AI models • Powered by Transformers</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Installation guide section
        self.render_installation_guide()
        
        # API Key section in sidebar
        api_key = self.render_api_key_section()
        
        # Model selection in sidebar
        selected_model = self.render_model_selection(api_key)
        
        # Clear chat button in sidebar (only show if model is selected)
        if selected_model:
            self.render_clear_chat_button()
        
        # Main content area - Chat interface
        if api_key and selected_model:
            self.render_chat_interface(selected_model, api_key)
        elif api_key:
            st.info("👈 Please select a model in the sidebar to start chatting!")
        else:
            # Show instructions when no API key is configured
            st.info("👈 Please configure your HuggingFace API key in the sidebar to access models!")
            
            # Enhanced model showcase
            st.markdown("### 🌟 Popular Models You Can Try")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🤖 Text Generation Models:**
                - **microsoft/DialoGPT-medium** - Great for conversations
                - **gpt2** - Classic text generation model
                - **EleutherAI/gpt-neo-2.7B** - Larger, more capable model
                - **microsoft/DialoGPT-large** - Advanced conversational AI
                - **facebook/blenderbot-400M-distill** - Facebook's conversational model
                """)
            
            with col2:
                st.markdown("""
                **🎯 Specialized Models:**
                - **distilbert-base-uncased** - Fast sentiment analysis
                - **t5-small** - Text-to-text generation
                - **microsoft/DialoGPT-small** - Lightweight chat model
                - **google/flan-t5-small** - Instruction following
                - **facebook/blenderbot_small-90M** - Compact conversational AI
                """)
            
            st.markdown("""
            ### 🚀 Getting Started
            
            1. **Install Transformers** (see installation guide above)
            2. **Get API Token** from [HuggingFace Settings](https://huggingface.co/settings/tokens)
            3. **Configure API Key** in the sidebar
            4. **Fetch Models** and start chatting!
            
            Configure your API key in the sidebar and fetch models to see the full list!
            """)


def render_huggingface_page():
    """Render the HuggingFace page."""
    page = HuggingFacePage()
    page.render()
