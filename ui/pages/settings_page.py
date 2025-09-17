"""
Settings Page Module
Handles the settings configuration functionality
"""

import streamlit as st
import pyperclip
from utils.logger_service import get_log_stats, configure_logging


class SettingsPage:
    """Settings page implementation"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def validate_config_value(self, value, value_type, min_val=None, max_val=None):
        """Validate configuration values"""
        try:
            if value_type == "int":
                val = int(value)
                if min_val is not None and val < min_val:
                    return False, f"Value must be at least {min_val}"
                if max_val is not None and val > max_val:
                    return False, f"Value must be at most {max_val}"
                return True, val
            elif value_type == "float":
                val = float(value)
                if min_val is not None and val < min_val:
                    return False, f"Value must be at least {min_val}"
                if max_val is not None and val > max_val:
                    return False, f"Value must be at most {max_val}"
                return True, val
            elif value_type == "str":
                if not value or not value.strip():
                    return False, "Value cannot be empty"
                return True, value.strip()
            return True, value
        except (ValueError, TypeError):
            return False, "Invalid value format"
    
    def show_validation_error(self, message):
        """Show validation error message"""
        st.error(f"❌ {message}")
        return False
    
    def render(self):
        """Render the settings page"""
        st.title("⚙️ Settings")
        st.markdown("Configure DurgasAI application settings")
        
        # Configuration management
        config = self.config_manager.get_config()
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "🔑 API & Keys", "🤖 AI Models", "💬 Chat & UI", "🔧 Tools & Workflows", 
            "📊 System & Performance", "🛡️ Security & Logging","🗄️ Vector DB", "📋 Templates & Sessions", "🔗 Integrations & Dev"
        ])
        
        with tab1:
            self.render_api_keys_tab(config)
        
        with tab2:
            self.render_ai_models_tab(config)
        
        with tab3:
            self.render_chat_ui_tab(config)
        
        with tab4:
            self.render_tools_workflows_tab(config)
        
        with tab5:
            self.render_system_performance_tab(config)
        
        with tab6:
            self.render_security_logging_tab(config)
        
        # Additional tabs for comprehensive coverage
        # tab7, tab8, tab9 = st.tabs([])
        
        with tab7:
            self.render_vector_db_tab(config)
        
        with tab8:
            self.render_templates_sessions_tab(config)
        
        with tab9:
            self.render_integrations_dev_tab(config)
    
    def render_api_keys_tab(self, config):
        """Render API Keys and Authentication settings"""
        st.subheader("🔑 API Keys & Authentication")
        
        # Quick help section
        with st.expander("💡 Quick Help - How to Get API Keys", expanded=False):
            st.markdown("""
            **Need API keys? Here's where to get them:**
            - **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (Format: `sk-proj-...`)
            - **Anthropic**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) (Format: `sk-ant-api03-...`)
            - **Groq**: [console.groq.com/keys](https://console.groq.com/keys) (Format: `gsk_...`)
            - **HuggingFace**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Format: `hf_...`)
            - **Tavily**: [app.tavily.com/home](https://app.tavily.com/home) (Format: `tvly-dev-...`)
            - **Perplexity**: [Perplexity API Keys](https://www.perplexity.ai/account/api/keys) (Format: `pplx-...`)
            - **Google**: [Google AI Studio](https://aistudio.google.com/u/1/apikey?pli=1) (Format: `AIzaSy...`)
            - **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/) (Format: `sk-...` 32 chars)
            - **Mistral**: [console.mistral.ai](https://console.mistral.ai/) (Format: random string)
            - **GitHub**: [GitHub Personal Access Tokens](https://github.com/settings/tokens) (Format: `github_pat_...`)
            - **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys) (Format: `sk-or-v1-...`)
            - **NVIDIA**: [build.nvidia.com](https://build.nvidia.com/explore/discover) (Format: `nvapi-...`)
            - **Cohere**: [dashboard.cohere.ai/api-keys](https://dashboard.cohere.ai/api-keys) (Format: `32-50 chars`)
            - **Jina**: [jina.ai/embeddings](https://jina.ai/embeddings) (Format: `jina_...`)
            - **SerpAPI**: [serpapi.com/dashboard](https://serpapi.com/dashboard) (Format: `64 chars`)
            """)
        
        # API Key Management
        with st.expander("API Key Management", expanded=True):
            # Provider definitions with descriptions and validation info
            api_providers = {
                "openai": {
                    "name": "OpenAI",
                    "description": "GPT models, embeddings, and function calling",
                    "format": "sk-proj-...",
                    "required": False
                },
                "anthropic": {
                    "name": "Anthropic",
                    "description": "Claude models and advanced reasoning",
                    "format": "sk-ant-api03-...",
                    "required": False
                },
                "groq": {
                    "name": "Groq",
                    "description": "Fast inference for various models",
                    "format": "gsk_...",
                    "required": False
                },
                "google": {
                    "name": "Google",
                    "description": "Gemini models and Google AI services",
                    "format": "AIzaSy...",
                    "required": False
                },
                "deepseek": {
                    "name": "DeepSeek",
                    "description": "Advanced reasoning and coding models",
                    "format": "sk-... (32 chars)",
                    "required": False
                },
                "mistral": {
                    "name": "Mistral",
                    "description": "European AI models and embeddings",
                    "format": "Random string",
                    "required": False
                },
                "perplexity": {
                    "name": "Perplexity",
                    "description": "Real-time web search and answers",
                    "format": "pplx-... (40+ chars)",
                    "required": False
                },
                "huggingface": {
                    "name": "HuggingFace",
                    "description": "Open source models and datasets",
                    "format": "hf_...",
                    "required": False
                },
                "tavily": {
                    "name": "Tavily",
                    "description": "Web search and content discovery",
                    "format": "tvly-dev-...",
                    "required": False
                },
                "web_search": {
                    "name": "Web Search",
                    "description": "General web search API",
                    "format": "Various",
                    "required": False
                },
                "exa": {
                    "name": "Exa",
                    "description": "Neural search and content discovery",
                    "format": "Various",
                    "required": False
                },
                "serpapi": {
                    "name": "SerpAPI",
                    "description": "Google search results API",
                    "format": "64 chars",
                    "required": False
                },
                "jina": {
                    "name": "Jina",
                    "description": "Jina AI embeddings and multimodal models",
                    "format": "jina_...",
                    "required": False
                },
                "github": {
                    "name": "GitHub",
                    "description": "GitHub Models API for various AI models",
                    "format": "github_pat_...",
                    "required": False
                },
                "openrouter": {
                    "name": "OpenRouter",
                    "description": "Access to multiple AI models through OpenRouter",
                    "format": "sk-or-v1-...",
                    "required": False
                },
                "nvidia": {
                    "name": "NVIDIA",
                    "description": "NVIDIA NIM API for high-performance AI models",
                    "format": "nvapi-...",
                    "required": False
                },
                "cohere": {
                    "name": "Cohere",
                    "description": "Cohere's Command models for advanced language understanding",
                    "format": "32-50 chars",
                    "required": False
                }
            }
            
            # Display API keys in a grid
            cols = st.columns(3)
            api_key_inputs = {}
            validation_errors = {}
            
            for i, (provider, info) in enumerate(api_providers.items()):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"**{info['name']}** {'🔴' if info['required'] else '🟡'}")
                        # st.caption(info['description'])
                        
                        # Get current value from session state
                        current_value = st.session_state.api_keys.get(provider, "")
                        
                        # Create input field
                        api_key_inputs[provider] = st.text_input(
                            f"{info['name']} API Key",
                            value=current_value,
                            type="password",
                            key=f"api_key_{provider}",
                            help=f"Format: {info['format']}"
                        )
                        
                        # Validate API key format
                        if api_key_inputs[provider]:
                            is_valid, error_msg = self.config_manager.validate_api_key_format(
                                provider, api_key_inputs[provider]
                            )
                            if not is_valid:
                                validation_errors[provider] = error_msg
                                st.error(f"❌ {error_msg}")
                                # Add a small help text for common issues
                                if "format" in error_msg.lower():
                                    st.caption("💡 Check the format guide above or click 'View Error Details' for help")
                            else:
                                st.success("✅ Valid format")
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("💾 Save All Keys", type="primary"):
                    self._save_all_api_keys(api_key_inputs, validation_errors)
            
            with col2:
                if st.button("🔄 Reset to Saved"):
                    self._reset_api_keys_to_saved()
            
            with col3:
                if st.button("🗑️ Clear All Keys"):
                    self._clear_all_api_keys()
            
            with col4:
                if st.button("📋 Copy All Keys"):
                    self._copy_all_api_keys(api_key_inputs)
            
            # Show validation summary with detailed error instructions
            if validation_errors:
                st.error(f"❌ {len(validation_errors)} API key(s) have validation errors")
                
                # Show detailed error instructions
                with st.expander("🔍 View Error Details & Instructions", expanded=True):
                    st.markdown("### API Key Validation Errors:")
                    
                    for provider, error_msg in validation_errors.items():
                        st.markdown(f"**{provider.upper()}**: {error_msg}")
                        
                        # Provider-specific instructions
                        instructions = self._get_api_key_instructions(provider)
                        if instructions:
                            st.markdown(f"**Instructions for {provider.upper()}:**")
                            st.markdown(instructions)
                            st.markdown("---")
                    
                    # General troubleshooting
                    st.markdown("### 🔧 General Troubleshooting:")
                    st.markdown("""
                    1. **Check the format**: Each provider has a specific format requirement
                    2. **Verify length**: Most API keys are 20+ characters long
                    3. **Check for typos**: Copy-paste directly from your provider dashboard
                    4. **Remove extra spaces**: Ensure no leading/trailing whitespace
                    5. **Verify the key is active**: Check your provider account for key status
                    6. **Check permissions**: Ensure the key has the required permissions
                    """)
                    
                    # Common error solutions
                    st.markdown("### 💡 Common Solutions:")
                    st.markdown("""
                    - **OpenAI**: Format should be `sk-proj-` followed by 100+ alphanumeric characters, underscores, and hyphens
                    - **Anthropic**: Format should be `sk-ant-api03-` followed by 50+ alphanumeric characters, underscores, and hyphens  
                    - **Groq**: Format should be `gsk_` followed by 20+ alphanumeric characters
                    - **HuggingFace**: Format should be `hf_` followed by 20+ alphanumeric characters
                    - **Tavily**: Format should be `tvly-dev-` followed by 20+ alphanumeric characters
                    - **Perplexity**: Format should be `pplx-` followed by 40+ alphanumeric characters
                    - **Google**: Format should be `AIzaSy` followed by 33 alphanumeric characters, underscores, and hyphens
                    - **DeepSeek**: Format should be `sk-` followed by exactly 32 alphanumeric characters
                    - **GitHub**: Format should be `github_pat_` followed by 22+ alphanumeric characters, then 20+ alphanumeric characters or underscores
                    - **OpenRouter**: Format should be `sk-or-v1-` followed by exactly 64 alphanumeric characters
                    - **NVIDIA**: Format should be `nvapi-` followed by 65-75 alphanumeric characters, hyphens, and underscores
                    - **Cohere**: Format should be 32-50 alphanumeric characters
                    - **Jina**: Format should be `jina_` followed by 60-70 alphanumeric characters
                    - **SerpAPI**: Format should be exactly 64 alphanumeric characters
                    """)
                    
                    # Action buttons for fixing errors
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Clear Invalid Keys", help="Clear all keys with validation errors"):
                            self._clear_invalid_api_keys(validation_errors)
                    
                    with col2:
                        if st.button("📋 Copy Error Report", help="Copy error details to clipboard"):
                            self._copy_error_report(validation_errors)
            else:
                st.success("✅ All API keys have valid formats")
    
    def _save_all_api_keys(self, api_key_inputs, validation_errors):
        """Save all API keys with validation"""
        if validation_errors:
            st.error("Cannot save API keys with validation errors. Please fix them first.")
            return
        
        try:
            # Update session state
            for provider, key in api_key_inputs.items():
                if key and key.strip():
                    st.session_state.api_keys[provider] = key.strip()
                else:
                    st.session_state.api_keys[provider] = ""
            
            # Update config
            api_keys_dict = {}
            for provider, key in st.session_state.api_keys.items():
                if key and key.strip():
                    api_keys_dict[provider] = key.strip()
            
            success = self.config_manager.update_api_keys(api_keys_dict)
            
            if success:
                st.success("✅ All API keys saved successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to save API keys. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error saving API keys: {str(e)}")
    
    def _reset_api_keys_to_saved(self):
        """Reset API keys to saved values from config"""
        try:
            config = self.config_manager.get_config()
            api_keys_config = config.get("api_keys", {})
            
            for provider in st.session_state.api_keys.keys():
                st.session_state.api_keys[provider] = api_keys_config.get(provider, "")
            
            st.success("🔄 API keys reset to saved values")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error resetting API keys: {str(e)}")
    
    def _clear_all_api_keys(self):
        """Clear all API keys"""
        try:
            for provider in st.session_state.api_keys.keys():
                st.session_state.api_keys[provider] = ""
            
            # Also clear from config
            self.config_manager.update_api_keys({})
            
            st.success("🗑️ All API keys cleared")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing API keys: {str(e)}")
    
    def _copy_all_api_keys(self, api_key_inputs):
        """Copy all API keys to clipboard (for debugging)"""
        try:
            api_keys_text = "API Keys:\n"
            for provider, key in api_key_inputs.items():
                if key and key.strip():
                    api_keys_text += f"{provider}: {key}\n"
            
            pyperclip.copy(api_keys_text)
            st.success("📋 API keys copied to clipboard")
        except ImportError:
            st.warning("pyperclip not installed. Cannot copy to clipboard.")
        except Exception as e:
            st.error(f"❌ Error copying API keys: {str(e)}")
    
    def _get_api_key_instructions(self, provider: str) -> str:
        """Get detailed instructions for a specific API key provider"""
        instructions = {
            "openai": """
            **How to get your OpenAI API key:**
            1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign in to your account
            3. Click "Create new secret key"
            4. Copy the key (starts with `sk-proj-`)
            5. Paste it here
            
            **Format**: `sk-proj-` + 100+ alphanumeric characters, underscores, and hyphens
            """,
            
            "anthropic": """
            **How to get your Anthropic API key:**
            1. Go to [Anthropic Console - API Keys](https://console.anthropic.com/settings/keys)
            2. Sign in to your account
            3. Click "Create Key"
            4. Copy the key (starts with `sk-ant-api03-`)
            5. Paste it here
            
            **Format**: `sk-ant-api03-` + 50+ alphanumeric characters, underscores, and hyphens
            """,
            
            "groq": """
            **How to get your Groq API key:**
            1. Go to [Groq Console](https://console.groq.com/keys)
            2. Sign in to your account
            3. Click "Create API Key"
            4. Copy the key (starts with `gsk_`)
            5. Paste it here
            
            **Format**: `gsk_` + 20+ alphanumeric characters
            """,
            
            "huggingface": """
            **How to get your HuggingFace token:**
            1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
            2. Sign in to your account
            3. Click "New token"
            4. Select appropriate permissions
            5. Copy the token (starts with `hf_`)
            6. Paste it here
            
            **Format**: `hf_` + 20+ alphanumeric characters
            """,
            
            "tavily": """
            **How to get your Tavily API key:**
            1. Go to [Tavily App](https://app.tavily.com/home)
            2. Sign in to your account
            3. Navigate to "API Keys" section
            4. Click "Generate API Key"
            5. Copy the key (starts with `tvly-dev-`)
            6. Paste it here
            
            **Format**: `tvly-dev-` + 20+ alphanumeric characters
            """,
            
            "perplexity": """
            **How to get your Perplexity API key:**
            1. Go to [Perplexity API Keys](https://www.perplexity.ai/account/api/keys)
            2. Sign in to your account
            3. Navigate to "API Keys" section
            4. Click "Create API Key"
            5. Copy the key (starts with `pplx-`)
            6. Paste it here
            
            **Format**: `pplx-` + 40+ alphanumeric characters
            """,
            
            "google": """
            **How to get your Google API key:**
            1. Go to [Google AI Studio](https://aistudio.google.com/u/1/apikey?pli=1)
            2. Sign in with your Google account
            3. Click "Get API key" or "Create API key"
            4. Copy the key (starts with `AIzaSy`)
            5. Paste it here
            
            **Format**: `AIzaSy` + 33 alphanumeric characters, underscores, and hyphens
            """,
            
            "deepseek": """
            **How to get your DeepSeek API key:**
            1. Go to [DeepSeek Platform](https://platform.deepseek.com/)
            2. Sign in to your account
            3. Navigate to "API Keys" section
            4. Click "Create API Key"
            5. Copy the key (starts with `sk-`)
            6. Paste it here
            
            **Format**: `sk-` + exactly 32 alphanumeric characters
            """,
            
            "mistral": """
            **How to get your Mistral API key:**
            1. Go to [Mistral Console](https://console.mistral.ai/)
            2. Sign in to your account
            3. Navigate to "API Keys" section
            4. Click "Create API Key"
            5. Copy the key (random string format)
            6. Paste it here
            
            **Format**: 20+ alphanumeric characters
            """,
            
            "web_search": """
            **How to get your Web Search API key:**
            1. Choose a web search provider (Google, Bing, etc.)
            2. Go to their developer console
            3. Create a new API key
            4. Copy the key
            5. Paste it here
            
            **Format**: Varies by provider (10+ characters)
            """,
            
            "exa": """
            **How to get your Exa API key:**
            1. Go to [Exa Dashboard](https://dashboard.exa.ai/)
            2. Sign in to your account
            3. Navigate to "API Keys" section
            4. Click "Create API Key"
            5. Copy the key
            6. Paste it here
            
            **Format**: 20+ alphanumeric characters
            """,
            
            "serpapi": """
            **How to get your SerpAPI key:**
            1. Go to [SerpAPI Dashboard](https://serpapi.com/dashboard)
            2. Sign in to your account
            3. Navigate to "API Key" section
            4. Copy your existing key or create a new one
            5. Paste it here
            
            **Format**: Exactly 64 alphanumeric characters
            """,
            
            "jina": """
            **How to get your Jina API key:**
            1. Go to [Jina AI](https://jina.ai/embeddings)
            2. Sign in to your account (or create one)
            3. Navigate to "API Keys" or account settings
            4. Click "Create API Key"
            5. Copy the key (starts with `jina_`)
            6. Paste it here
            
            **Format**: `jina_` + 60-70 alphanumeric characters
            
            **Note**: Jina AI provides high-quality embeddings and multimodal AI models.
            """,
            
            "github": """
            **How to get your GitHub API key:**
            1. Go to [GitHub Settings - Personal Access Tokens](https://github.com/settings/tokens)
            2. Click "Generate new token (classic)"
            3. Select necessary scopes (at minimum: read:user, repo)
            4. Generate and copy the token (starts with `github_pat_`)
            5. Paste it here
            
            **Format**: `github_pat_` + 22+ alphanumeric characters + 20+ alphanumeric characters or underscores
            
            **Note**: GitHub Models requires a GitHub PAT with appropriate permissions for AI model access.
            """,
            
            "openrouter": """
            **How to get your OpenRouter API key:**
            1. Go to [OpenRouter Keys](https://openrouter.ai/keys)
            2. Sign in to your account (or create one)
            3. Click "Create Key"
            4. Give your key a name and set optional limits
            5. Copy the key (starts with `sk-or-v1-`)
            6. Paste it here
            
            **Format**: `sk-or-v1-` + exactly 64 alphanumeric characters
            
            **Note**: OpenRouter provides access to multiple AI models from different providers through a unified API.
            """,
            
            "nvidia": """
            **How to get your NVIDIA API key:**
            1. Go to [NVIDIA Build](https://build.nvidia.com/explore/discover)
            2. Sign in with your NVIDIA account (or create one)
            3. Navigate to "API Keys" or your account settings
            4. Click "Generate API Key" or "Create Key"
            5. Copy the key (starts with `nvapi-`)
            6. Paste it here
            
            **Format**: `nvapi-` + 65-75 alphanumeric characters, hyphens, and underscores
            
            **Note**: NVIDIA NIM provides access to high-performance AI models optimized for NVIDIA hardware.
            """,
            
            "cohere": """
            **How to get your Cohere API key:**
            1. Go to [Cohere Dashboard](https://dashboard.cohere.ai/api-keys)
            2. Sign in to your account (or create one)
            3. Navigate to "API Keys" section
            4. Click "Create API Key"
            5. Give your key a name and copy it
            6. Paste it here
            
            **Format**: 32-50 alphanumeric characters (no prefix)
            
            **Note**: Cohere provides advanced language models including Command R series for chat and reasoning tasks.
            """
        }
        
        return instructions.get(provider.lower(), "")
    
    def _clear_invalid_api_keys(self, validation_errors):
        """Clear all API keys that have validation errors"""
        try:
            for provider in validation_errors.keys():
                if provider in st.session_state.api_keys:
                    st.session_state.api_keys[provider] = ""
            
            st.success(f"🗑️ Cleared {len(validation_errors)} invalid API key(s)")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing invalid keys: {str(e)}")
    
    def _copy_error_report(self, validation_errors):
        """Copy error report to clipboard"""
        try:
            error_report = "API Key Validation Error Report:\n\n"
            for provider, error_msg in validation_errors.items():
                error_report += f"❌ {provider.upper()}: {error_msg}\n"
                instructions = self._get_api_key_instructions(provider)
                if instructions:
                    # Extract just the key steps from instructions
                    lines = instructions.split('\n')
                    key_steps = [line.strip() for line in lines if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.'))]
                    if key_steps:
                        error_report += f"   Steps to fix: {' | '.join(key_steps[:3])}\n"
                error_report += "\n"
            
            pyperclip.copy(error_report)
            st.success("📋 Error report copied to clipboard")
        except ImportError:
            st.warning("pyperclip not installed. Cannot copy error report.")
        except Exception as e:
            st.error(f"❌ Error copying error report: {str(e)}")
    
    def render_ai_models_tab(self, config):
        """Render AI Models and LangGraph settings"""
        st.subheader("🤖 AI Models & LangGraph")
        
        # Ollama Settings
        with st.expander("Ollama Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                ollama_host = st.text_input("Ollama Host", 
                    value=config.get("ollama", {}).get("host", "http://localhost:11434"))
                ollama_model = st.selectbox("Default Model", 
                    ["llama3.1", "gemma2", "qwen2.5", "mistralai/Mistral-7B-Instruct-v0.3"],
                    index=0)
                temperature = st.slider("Temperature", 0.0, 1.0, 
                    config.get("ollama", {}).get("temperature", 0.7), 0.1, key="ollama_temperature")
                max_tokens = st.number_input("Max Tokens", min_value=100, max_value=8192, 
                    value=config.get("ollama", {}).get("max_tokens", 2048), key="ollama_max_tokens")
            
            with col2:
                timeout = st.number_input("Timeout (seconds)", min_value=5, max_value=300, 
                    value=config.get("ollama", {}).get("timeout", 30), key="ollama_timeout")
                max_retries = st.number_input("Max Retries", min_value=1, max_value=10, 
                    value=config.get("ollama", {}).get("max_retries", 3))
                connection_pool = st.number_input("Connection Pool Size", min_value=1, max_value=20, 
                    value=config.get("ollama", {}).get("connection_pool_size", 5))
                enable_streaming = st.checkbox("Enable Streaming", 
                    value=config.get("ollama", {}).get("enable_streaming", True), key="ollama_streaming")
            
            if st.button("💾 Update Ollama Settings"):
                config["ollama"]["host"] = ollama_host
                config["ollama"]["model"] = ollama_model
                config["ollama"]["temperature"] = temperature
                config["ollama"]["max_tokens"] = max_tokens
                config["ollama"]["timeout"] = timeout
                config["ollama"]["max_retries"] = max_retries
                config["ollama"]["connection_pool_size"] = connection_pool
                config["ollama"]["enable_streaming"] = enable_streaming
                self.config_manager.save_config(config)
                st.success("Ollama settings updated!")
        
        # LangGraph Settings
        with st.expander("LangGraph Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_langgraph = st.checkbox("Enable LangGraph", 
                    value=config.get("langgraph", {}).get("enable_langgraph", True), key="langgraph_enable")
                enable_tool_calling = st.checkbox("Enable Tool Calling", 
                    value=config.get("langgraph", {}).get("enable_tool_calling", True), key="langgraph_tool_calling")
                enable_memory = st.checkbox("Enable Memory", 
                    value=config.get("langgraph", {}).get("enable_memory", True), key="langgraph_memory")
                enable_streaming = st.checkbox("Enable Streaming", 
                    value=config.get("langgraph", {}).get("enable_streaming", True), key="langgraph_streaming")
            
            with col2:
                max_sessions = st.number_input("Max Concurrent Sessions", min_value=1, max_value=50, 
                    value=config.get("langgraph", {}).get("max_concurrent_sessions", 10))
                session_timeout = st.number_input("Session Timeout (minutes)", min_value=5, max_value=480, 
                    value=config.get("langgraph", {}).get("session_timeout_minutes", 30), key="langgraph_session_timeout")
                max_iterations = st.number_input("Max Iterations", min_value=1, max_value=100, 
                    value=config.get("langgraph", {}).get("max_iterations", 20))
                tool_timeout = st.number_input("Tool Timeout (seconds)", min_value=5, max_value=300, 
                    value=config.get("langgraph", {}).get("tool_timeout_seconds", 30))
            
            if st.button("💾 Update LangGraph Settings"):
                config["langgraph"]["enable_langgraph"] = enable_langgraph
                config["langgraph"]["enable_tool_calling"] = enable_tool_calling
                config["langgraph"]["enable_memory"] = enable_memory
                config["langgraph"]["enable_streaming"] = enable_streaming
                config["langgraph"]["max_concurrent_sessions"] = max_sessions
                config["langgraph"]["session_timeout_minutes"] = session_timeout
                config["langgraph"]["max_iterations"] = max_iterations
                config["langgraph"]["tool_timeout_seconds"] = tool_timeout
                self.config_manager.save_config(config)
                st.success("LangGraph settings updated!")
    
    def render_chat_ui_tab(self, config):
        """Render Chat and UI settings"""
        st.subheader("💬 Chat & User Interface")
        
        # Chat Settings
        with st.expander("Chat Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_tools = st.checkbox("Enable Tools", 
                    value=config.get("chat", {}).get("enable_tools", True), key="chat_tools")
                enable_vector_search = st.checkbox("Enable Vector Search", 
                    value=config.get("chat", {}).get("enable_vector_search", True), key="chat_vector_search")
                enable_multimodal = st.checkbox("Enable Multimodal", 
                    value=config.get("chat", {}).get("enable_multimodal", True), key="chat_multimodal")
                enable_thinking = st.checkbox("Enable Thinking Mode", 
                    value=config.get("chat", {}).get("enable_thinking_mode", True), key="chat_thinking")
            
            with col2:
                max_tokens = st.number_input("Max Tokens", min_value=100, max_value=8192, 
                    value=config.get("chat", {}).get("default_max_tokens", 2048), key="chat_max_tokens")
                temperature = st.slider("Temperature", 0.0, 1.0, 
                    config.get("chat", {}).get("default_temperature", 0.7), 0.1, key="chat_temperature")
                top_p = st.slider("Top P", 0.0, 1.0, 
                    config.get("chat", {}).get("default_top_p", 0.9), 0.1)
                top_k = st.number_input("Top K", min_value=1, max_value=100, 
                    value=config.get("chat", {}).get("default_top_k", 40))
            
            if st.button("💾 Update Chat Settings"):
                config["chat"]["enable_tools"] = enable_tools
                config["chat"]["enable_vector_search"] = enable_vector_search
                config["chat"]["enable_multimodal"] = enable_multimodal
                config["chat"]["enable_thinking_mode"] = enable_thinking
                config["chat"]["default_max_tokens"] = max_tokens
                config["chat"]["default_temperature"] = temperature
                config["chat"]["default_top_p"] = top_p
                config["chat"]["default_top_k"] = top_k
                self.config_manager.save_config(config)
                st.success("Chat settings updated!")
        
        # UI Settings
        with st.expander("User Interface Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                theme = st.selectbox("Theme", ["light", "dark"], 
                    index=0 if config.get("ui", {}).get("theme", "light") == "light" else 1)
                page_title = st.text_input("Page Title", 
                    value=config.get("ui", {}).get("page_title", "DurgasAI"))
                page_icon = st.text_input("Page Icon", 
                    value=config.get("ui", {}).get("page_icon", "🤖"))
                default_page = st.selectbox("Default Page", 
                    ["Chat", "Tools", "Templates", "System Monitor", "Settings"],
                    index=["Chat", "Tools", "Templates", "System Monitor", "Settings"].index(
                        config.get("ui", {}).get("default_page", "Tools")))
            
            with col2:
                enable_animations = st.checkbox("Enable Animations", 
                    value=config.get("ui", {}).get("enable_animations", True), key="ui_animations")
                enable_sound = st.checkbox("Enable Sound Effects", 
                    value=config.get("ui", {}).get("enable_sound_effects", False), key="ui_sound")
                enable_keyboard = st.checkbox("Enable Keyboard Shortcuts", 
                    value=config.get("ui", {}).get("enable_keyboard_shortcuts", True), key="ui_keyboard")
                auto_refresh = st.checkbox("Enable Auto Refresh", 
                    value=config.get("ui", {}).get("enable_auto_refresh", True), key="ui_auto_refresh")
            
            if st.button("💾 Update UI Settings"):
                config["ui"]["theme"] = theme
                config["ui"]["page_title"] = page_title
                config["ui"]["page_icon"] = page_icon
                config["ui"]["default_page"] = default_page
                config["ui"]["enable_animations"] = enable_animations
                config["ui"]["enable_sound_effects"] = enable_sound
                config["ui"]["enable_keyboard_shortcuts"] = enable_keyboard
                config["ui"]["enable_auto_refresh"] = auto_refresh
                self.config_manager.save_config(config)
                st.success("UI settings updated!")
    
    def render_tools_workflows_tab(self, config):
        """Render Tools and Workflows settings"""
        st.subheader("🔧 Tools & Workflows")
        
        # Tools Settings
        with st.expander("Tools Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_tool_mgmt = st.checkbox("Enable Tool Management", 
                    value=config.get("tools", {}).get("enable_tool_management", True), key="tools_management")
                tools_dir = st.text_input("Tools Directory", 
                    value=config.get("tools", {}).get("tools_directory", "./output/tools"))
                max_execution_time = st.number_input("Max Tool Execution Time (seconds)", 
                    min_value=10, max_value=600, 
                    value=config.get("tools", {}).get("max_tool_execution_time", 60))
                enable_caching = st.checkbox("Enable Tool Caching", 
                    value=config.get("tools", {}).get("enable_tool_caching", True), key="tools_caching")
            
            with col2:
                cache_ttl = st.number_input("Cache TTL (seconds)", min_value=60, max_value=3600, 
                    value=config.get("tools", {}).get("tool_cache_ttl", 300), key="tools_cache_ttl")
                enable_logging = st.checkbox("Enable Tool Logging", 
                    value=config.get("tools", {}).get("enable_tool_logging", True), key="tools_logging")
                enable_validation = st.checkbox("Enable Tool Validation", 
                    value=config.get("tools", {}).get("enable_tool_validation", True), key="tools_validation")
                max_concurrent = st.number_input("Max Concurrent Tool Executions", 
                    min_value=1, max_value=20, 
                    value=config.get("tools", {}).get("max_concurrent_tool_executions", 5))
            
            if st.button("💾 Update Tools Settings"):
                config["tools"]["enable_tool_management"] = enable_tool_mgmt
                config["tools"]["tools_directory"] = tools_dir
                config["tools"]["max_tool_execution_time"] = max_execution_time
                config["tools"]["enable_tool_caching"] = enable_caching
                config["tools"]["tool_cache_ttl"] = cache_ttl
                config["tools"]["enable_tool_logging"] = enable_logging
                config["tools"]["enable_tool_validation"] = enable_validation
                config["tools"]["max_concurrent_tool_executions"] = max_concurrent
                self.config_manager.save_config(config)
                st.success("Tools settings updated!")
        
        # Workflows Settings
        with st.expander("Workflows Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_workflows = st.checkbox("Enable Workflow Management", 
                    value=config.get("workflows", {}).get("enable_workflow_management", True), key="workflows_management")
                workflows_dir = st.text_input("Workflows Directory", 
                    value=config.get("workflows", {}).get("workflows_directory", "./output/workflows"))
                max_steps = st.number_input("Max Workflow Steps", min_value=5, max_value=100, 
                    value=config.get("workflows", {}).get("max_workflow_steps", 20))
                enable_monitoring = st.checkbox("Enable Workflow Monitoring", 
                    value=config.get("workflows", {}).get("enable_workflow_monitoring", True), key="workflows_monitoring")
            
            with col2:
                max_concurrent_workflows = st.number_input("Max Concurrent Workflows", 
                    min_value=1, max_value=20, 
                    value=config.get("workflows", {}).get("max_concurrent_workflows", 3))
                workflow_timeout = st.number_input("Workflow Timeout (minutes)", 
                    min_value=5, max_value=480, 
                    value=config.get("workflows", {}).get("workflow_timeout_minutes", 30))
                enable_logging = st.checkbox("Enable Workflow Logging", 
                    value=config.get("workflows", {}).get("enable_workflow_logging", True), key="workflows_logging")
                enable_validation = st.checkbox("Enable Workflow Validation", 
                    value=config.get("workflows", {}).get("enable_workflow_validation", True), key="workflows_validation")
            
            if st.button("💾 Update Workflows Settings"):
                config["workflows"]["enable_workflow_management"] = enable_workflows
                config["workflows"]["workflows_directory"] = workflows_dir
                config["workflows"]["max_workflow_steps"] = max_steps
                config["workflows"]["enable_workflow_monitoring"] = enable_monitoring
                config["workflows"]["max_concurrent_workflows"] = max_concurrent_workflows
                config["workflows"]["workflow_timeout_minutes"] = workflow_timeout
                config["workflows"]["enable_workflow_logging"] = enable_logging
                config["workflows"]["enable_workflow_validation"] = enable_validation
                self.config_manager.save_config(config)
                st.success("Workflows settings updated!")
    
    def render_system_performance_tab(self, config):
        """Render System Monitor and Performance settings"""
        st.subheader("📊 System & Performance")
        
        # System Monitor Settings
        with st.expander("System Monitor Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                refresh_rate = st.number_input("Refresh Rate (seconds)", min_value=1, max_value=60, 
                    value=config.get("system_monitor", {}).get("refresh_rate", 5))
                max_history = st.number_input("Max History", min_value=10, max_value=1000, 
                    value=config.get("system_monitor", {}).get("max_history", 100))
                show_network = st.checkbox("Show Network Stats", 
                    value=config.get("system_monitor", {}).get("show_network", True), key="monitor_network")
                show_disk = st.checkbox("Show Disk Stats", 
                    value=config.get("system_monitor", {}).get("show_disk", True), key="monitor_disk")
            
            with col2:
                enable_alerts = st.checkbox("Enable Alerts", 
                    value=config.get("system_monitor", {}).get("enable_alerts", True), key="monitor_alerts")
                cpu_threshold = st.slider("CPU Alert Threshold (%)", 50.0, 100.0, 
                    config.get("system_monitor", {}).get("alert_cpu_threshold", 80.0), 5.0)
                memory_threshold = st.slider("Memory Alert Threshold (%)", 50.0, 100.0, 
                    config.get("system_monitor", {}).get("alert_memory_threshold", 85.0), 5.0)
                disk_threshold = st.slider("Disk Alert Threshold (%)", 50.0, 100.0, 
                    config.get("system_monitor", {}).get("alert_disk_threshold", 90.0), 5.0)
            
            if st.button("💾 Update System Monitor Settings"):
                config["system_monitor"]["refresh_rate"] = refresh_rate
                config["system_monitor"]["max_history"] = max_history
                config["system_monitor"]["show_network"] = show_network
                config["system_monitor"]["show_disk"] = show_disk
                config["system_monitor"]["enable_alerts"] = enable_alerts
                config["system_monitor"]["alert_cpu_threshold"] = cpu_threshold
                config["system_monitor"]["alert_memory_threshold"] = memory_threshold
                config["system_monitor"]["alert_disk_threshold"] = disk_threshold
                self.config_manager.save_config(config)
                st.success("System Monitor settings updated!")
        
        # Performance Settings
        with st.expander("Performance Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_caching = st.checkbox("Enable Caching", 
                    value=config.get("performance", {}).get("enable_caching", True), key="perf_caching")
                cache_size = st.number_input("Cache Size (MB)", min_value=10, max_value=1000, 
                    value=config.get("performance", {}).get("cache_size_mb", 100))
                cache_ttl = st.number_input("Cache TTL (seconds)", min_value=60, max_value=86400, 
                    value=config.get("performance", {}).get("cache_ttl_seconds", 3600), key="performance_cache_ttl")
                enable_memory_opt = st.checkbox("Enable Memory Optimization", 
                    value=config.get("performance", {}).get("enable_memory_optimization", True), key="perf_memory")
            
            with col2:
                max_memory = st.number_input("Max Memory Usage (MB)", min_value=512, max_value=8192, 
                    value=config.get("performance", {}).get("max_memory_usage_mb", 2048))
                enable_cpu_opt = st.checkbox("Enable CPU Optimization", 
                    value=config.get("performance", {}).get("enable_cpu_optimization", True), key="perf_cpu")
                max_cpu = st.slider("Max CPU Usage (%)", 10.0, 100.0, 
                    float(config.get("performance", {}).get("max_cpu_usage_percent", 80.0)), 5.0)
                enable_lazy_loading = st.checkbox("Enable Lazy Loading", 
                    value=config.get("performance", {}).get("enable_lazy_loading", True), key="perf_lazy")
            
            if st.button("💾 Update Performance Settings"):
                config["performance"]["enable_caching"] = enable_caching
                config["performance"]["cache_size_mb"] = cache_size
                config["performance"]["cache_ttl_seconds"] = cache_ttl
                config["performance"]["enable_memory_optimization"] = enable_memory_opt
                config["performance"]["max_memory_usage_mb"] = max_memory
                config["performance"]["enable_cpu_optimization"] = enable_cpu_opt
                config["performance"]["max_cpu_usage_percent"] = max_cpu
                config["performance"]["enable_lazy_loading"] = enable_lazy_loading
                self.config_manager.save_config(config)
                st.success("Performance settings updated!")
    
    def render_security_logging_tab(self, config):
        """Render Security and Logging settings"""
        st.subheader("🛡️ Security & Logging")
        
        # Security Settings
        with st.expander("Security Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_auth = st.checkbox("Enable API Authentication", 
                    value=config.get("security", {}).get("enable_api_authentication", False))
                require_key = st.checkbox("API Key Required", 
                    value=config.get("security", {}).get("api_key_required", False))
                enable_rate_limit = st.checkbox("Enable Rate Limiting", 
                    value=config.get("security", {}).get("enable_rate_limiting", False))
                max_requests = st.number_input("Max Requests per Minute", min_value=10, max_value=1000, 
                    value=config.get("security", {}).get("max_requests_per_minute", 60))
            
            with col2:
                enable_cors = st.checkbox("Enable CORS", 
                    value=config.get("security", {}).get("enable_cors", True))
                enable_input_sanitization = st.checkbox("Enable Input Sanitization", 
                    value=config.get("security", {}).get("enable_input_sanitization", True))
                enable_output_filtering = st.checkbox("Enable Output Filtering", 
                    value=config.get("security", {}).get("enable_output_filtering", False))
                session_timeout = st.number_input("Session Timeout (minutes)", min_value=5, max_value=1440, 
                    value=config.get("security", {}).get("session_timeout_minutes", 480), key="security_session_timeout")
            
            if st.button("💾 Update Security Settings"):
                config["security"]["enable_api_authentication"] = enable_auth
                config["security"]["api_key_required"] = require_key
                config["security"]["enable_rate_limiting"] = enable_rate_limit
                config["security"]["max_requests_per_minute"] = max_requests
                config["security"]["enable_cors"] = enable_cors
                config["security"]["enable_input_sanitization"] = enable_input_sanitization
                config["security"]["enable_output_filtering"] = enable_output_filtering
                config["security"]["session_timeout_minutes"] = session_timeout
                self.config_manager.save_config(config)
                st.success("Security settings updated!")
        
        # Logging Settings
        with st.expander("Advanced Logging Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Log Directory & Files")
                log_directory = st.text_input("Log Directory", 
                    value=config.get("logging", {}).get("log_directory", "./output/logs"))
                log_level = st.selectbox("Log Level", 
                    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                        config.get("logging", {}).get("log_level", "INFO")))
                log_file_naming = st.selectbox("Log File Naming", 
                    ["date_based", "component_based", "timestamp_based"],
                    index=["date_based", "component_based", "timestamp_based"].index(
                        config.get("logging", {}).get("log_file_naming", "date_based")))
                date_format = st.text_input("Date Format", 
                    value=config.get("logging", {}).get("date_format", "%Y%m%d"))
                
                st.subheader("📊 File Management")
                max_file_size = st.number_input("Max Log File Size (MB)", min_value=1, max_value=1000, 
                    value=config.get("logging", {}).get("max_log_file_size_mb", 50))
                max_log_files = st.number_input("Max Log Files", min_value=1, max_value=50, 
                    value=config.get("logging", {}).get("max_log_files", 5))
                retention_days = st.number_input("Log Retention (days)", min_value=1, max_value=365, 
                    value=config.get("logging", {}).get("log_retention_days", 30))
            
            with col2:
                st.subheader("🔧 Logging Features")
                enable_file_logging = st.checkbox("Enable File Logging", 
                    value=config.get("logging", {}).get("enable_file_logging", True))
                enable_console_logging = st.checkbox("Enable Console Logging", 
                    value=config.get("logging", {}).get("enable_console_logging", True))
                enable_component_logging = st.checkbox("Enable Component Logging", 
                    value=config.get("logging", {}).get("enable_component_logging", True))
                enable_performance_logging = st.checkbox("Enable Performance Logging", 
                    value=config.get("logging", {}).get("enable_performance_logging", True))
                
                st.subheader("⚙️ Advanced Options")
                enable_audit_logging = st.checkbox("Enable Audit Logging", 
                    value=config.get("logging", {}).get("enable_audit_logging", True))
                enable_error_tracking = st.checkbox("Enable Error Tracking", 
                    value=config.get("logging", {}).get("enable_error_tracking", True))
                enable_log_rotation = st.checkbox("Enable Log Rotation", 
                    value=config.get("logging", {}).get("enable_log_rotation", True))
                enable_log_compression = st.checkbox("Enable Log Compression", 
                    value=config.get("logging", {}).get("enable_log_compression", True))
                enable_log_cleanup = st.checkbox("Enable Log Cleanup", 
                    value=config.get("logging", {}).get("enable_log_cleanup", True))
            
            # Log Statistics
            if st.button("📈 Show Log Statistics"):
                try:
                    stats = get_log_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"Error getting log statistics: {e}")
            
            if st.button("💾 Update Logging Settings"):
                config["logging"]["log_directory"] = log_directory
                config["logging"]["log_level"] = log_level
                config["logging"]["log_file_naming"] = log_file_naming
                config["logging"]["date_format"] = date_format
                config["logging"]["enable_file_logging"] = enable_file_logging
                config["logging"]["enable_console_logging"] = enable_console_logging
                config["logging"]["enable_component_logging"] = enable_component_logging
                config["logging"]["enable_performance_logging"] = enable_performance_logging
                config["logging"]["enable_audit_logging"] = enable_audit_logging
                config["logging"]["enable_error_tracking"] = enable_error_tracking
                config["logging"]["max_log_file_size_mb"] = max_file_size
                config["logging"]["max_log_files"] = max_log_files
                config["logging"]["log_retention_days"] = retention_days
                config["logging"]["enable_log_rotation"] = enable_log_rotation
                config["logging"]["enable_log_compression"] = enable_log_compression
                config["logging"]["enable_log_cleanup"] = enable_log_cleanup
                
                # Reconfigure logging service
                try:
                    configure_logging(config["logging"])
                    self.config_manager.save_config(config)
                    st.success("Logging settings updated and reconfigured!")
                except Exception as e:
                    st.error(f"Error reconfiguring logging: {e}")
                    self.config_manager.save_config(config)
                    st.success("Logging settings saved (reconfiguration failed)")
    
    def render_vector_db_tab(self, config):
        """Render Vector Database settings"""
        st.subheader("🗄️ Vector Database Configuration")
        
        # Basic Vector DB Settings
        with st.expander("Basic Vector Database Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                db_path = st.text_input("Database Path", 
                    value=config.get("vector_db", {}).get("db_path", "./vector_db"))
                collection_name = st.text_input("Collection Name", 
                    value=config.get("vector_db", {}).get("collection_name", "documents"))
                embedding_model = st.selectbox("Embedding Model", 
                    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
                    index=0)
                max_results = st.number_input("Max Search Results", min_value=1, max_value=50, 
                    value=config.get("vector_db", {}).get("max_results", 5))
            
            with col2:
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 
                    config.get("vector_db", {}).get("similarity_threshold", 0.7), 0.1)
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, 
                    value=config.get("vector_db", {}).get("chunk_size", 1000))
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, 
                    value=config.get("vector_db", {}).get("chunk_overlap", 200))
                enable_persistent = st.checkbox("Enable Persistent Storage", 
                    value=config.get("vector_db", {}).get("enable_persistent_storage", True))
            
            if st.button("💾 Update Vector DB Settings"):
                config["vector_db"]["db_path"] = db_path
                config["vector_db"]["collection_name"] = collection_name
                config["vector_db"]["embedding_model"] = embedding_model
                config["vector_db"]["max_results"] = max_results
                config["vector_db"]["similarity_threshold"] = similarity_threshold
                config["vector_db"]["chunk_size"] = chunk_size
                config["vector_db"]["chunk_overlap"] = chunk_overlap
                config["vector_db"]["enable_persistent_storage"] = enable_persistent
                self.config_manager.save_config(config)
                st.success("Vector DB settings updated!")
        
        # Advanced Vector DB Settings
        with st.expander("Advanced Vector Database Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_hybrid = st.checkbox("Enable Hybrid Search", 
                    value=config.get("vector_db_advanced", {}).get("enable_hybrid_search", True))
                enable_contextual = st.checkbox("Enable Contextual Search", 
                    value=config.get("vector_db_advanced", {}).get("enable_contextual_search", True))
                enable_smart_chunking = st.checkbox("Enable Smart Chunking", 
                    value=config.get("vector_db_advanced", {}).get("enable_smart_chunking", True))
                chunking_strategy = st.selectbox("Chunking Strategy", 
                    ["semantic", "fixed", "recursive"],
                    index=0)
            
            with col2:
                semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 
                    config.get("vector_db_advanced", {}).get("semantic_weight", 0.7), 0.1)
                keyword_weight = st.slider("Keyword Weight", 0.0, 1.0, 
                    config.get("vector_db_advanced", {}).get("keyword_weight", 0.3), 0.1)
                context_window = st.number_input("Context Window", min_value=1, max_value=10, 
                    value=config.get("vector_db_advanced", {}).get("context_window", 3))
                enable_reranking = st.checkbox("Enable Reranking", 
                    value=config.get("vector_db_advanced", {}).get("enable_reranking", True))
            
            if st.button("💾 Update Advanced Vector DB Settings"):
                config["vector_db_advanced"]["enable_hybrid_search"] = enable_hybrid
                config["vector_db_advanced"]["enable_contextual_search"] = enable_contextual
                config["vector_db_advanced"]["enable_smart_chunking"] = enable_smart_chunking
                config["vector_db_advanced"]["default_chunking_strategy"] = chunking_strategy
                config["vector_db_advanced"]["semantic_weight"] = semantic_weight
                config["vector_db_advanced"]["keyword_weight"] = keyword_weight
                config["vector_db_advanced"]["context_window"] = context_window
                config["vector_db_advanced"]["enable_reranking"] = enable_reranking
                self.config_manager.save_config(config)
                st.success("Advanced Vector DB settings updated!")
    
    def render_templates_sessions_tab(self, config):
        """Render Templates and Sessions settings"""
        st.subheader("📋 Templates & Sessions Configuration")
        
        # Templates Settings
        with st.expander("Templates Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_template_mgmt = st.checkbox("Enable Template Management", 
                    value=config.get("templates", {}).get("enable_template_management", True))
                templates_dir = st.text_input("Templates Directory", 
                    value=config.get("templates", {}).get("templates_directory", "./output/templates"))
                max_template_size = st.number_input("Max Template Size (KB)", min_value=1, max_value=1000, 
                    value=config.get("templates", {}).get("max_template_size_kb", 100))
                enable_validation = st.checkbox("Enable Template Validation", 
                    value=config.get("templates", {}).get("enable_template_validation", True))
            
            with col2:
                enable_caching = st.checkbox("Enable Template Caching", 
                    value=config.get("templates", {}).get("enable_template_caching", True))
                cache_ttl = st.number_input("Template Cache TTL (seconds)", min_value=60, max_value=3600, 
                    value=config.get("templates", {}).get("template_cache_ttl", 600))
                enable_import_export = st.checkbox("Enable Import/Export", 
                    value=config.get("templates", {}).get("enable_template_import_export", True))
                enable_sharing = st.checkbox("Enable Template Sharing", 
                    value=config.get("templates", {}).get("enable_template_sharing", False))
            
            if st.button("💾 Update Templates Settings"):
                config["templates"]["enable_template_management"] = enable_template_mgmt
                config["templates"]["templates_directory"] = templates_dir
                config["templates"]["max_template_size_kb"] = max_template_size
                config["templates"]["enable_template_validation"] = enable_validation
                config["templates"]["enable_template_caching"] = enable_caching
                config["templates"]["template_cache_ttl"] = cache_ttl
                config["templates"]["enable_template_import_export"] = enable_import_export
                config["templates"]["enable_template_sharing"] = enable_sharing
                self.config_manager.save_config(config)
                st.success("Templates settings updated!")
        
        # Sessions Settings
        with st.expander("Sessions Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_session_mgmt = st.checkbox("Enable Session Management", 
                    value=config.get("sessions", {}).get("enable_session_management", True))
                sessions_dir = st.text_input("Sessions Directory", 
                    value=config.get("sessions", {}).get("sessions_directory", "./output/sessions"))
                retention_days = st.number_input("Session Retention (days)", min_value=1, max_value=365, 
                    value=config.get("sessions", {}).get("session_retention_days", 30))
                max_sessions = st.number_input("Max Sessions", min_value=10, max_value=1000, 
                    value=config.get("sessions", {}).get("max_sessions", 100))
            
            with col2:
                enable_backup = st.checkbox("Enable Session Backup", 
                    value=config.get("sessions", {}).get("enable_session_backup", True))
                enable_compression = st.checkbox("Enable Session Compression", 
                    value=config.get("sessions", {}).get("enable_session_compression", False))
                auto_save_interval = st.number_input("Auto Save Interval (seconds)", min_value=10, max_value=3600, 
                    value=config.get("sessions", {}).get("session_auto_save_interval", 60))
                max_session_size = st.number_input("Max Session Size (MB)", min_value=1, max_value=100, 
                    value=config.get("sessions", {}).get("max_session_size_mb", 10))
            
            if st.button("💾 Update Sessions Settings"):
                config["sessions"]["enable_session_management"] = enable_session_mgmt
                config["sessions"]["sessions_directory"] = sessions_dir
                config["sessions"]["session_retention_days"] = retention_days
                config["sessions"]["max_sessions"] = max_sessions
                config["sessions"]["enable_session_backup"] = enable_backup
                config["sessions"]["enable_session_compression"] = enable_compression
                config["sessions"]["session_auto_save_interval"] = auto_save_interval
                config["sessions"]["max_session_size_mb"] = max_session_size
                self.config_manager.save_config(config)
                st.success("Sessions settings updated!")
    
    def render_integrations_dev_tab(self, config):
        """Render Integrations and Development settings"""
        st.subheader("🔗 Integrations & Development Configuration")
        
        # Integrations Settings
        with st.expander("Integrations Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_web_search = st.checkbox("Enable Web Search", 
                    value=config.get("integrations", {}).get("enable_web_search", True))
                web_search_provider = st.selectbox("Web Search Provider", 
                    ["exa", "tavily", "serpapi", "google"],
                    index=0)
                enable_file_processing = st.checkbox("Enable File Processing", 
                    value=config.get("integrations", {}).get("enable_file_processing", True))
                max_file_size = st.number_input("Max File Size (MB)", min_value=1, max_value=1000, 
                    value=config.get("integrations", {}).get("max_file_size_mb", 50))
            
            with col2:
                enable_email = st.checkbox("Enable Email Integration", 
                    value=config.get("integrations", {}).get("enable_email_integration", False))
                email_provider = st.selectbox("Email Provider", 
                    ["smtp", "gmail", "outlook", "sendgrid"],
                    index=0)
                enable_calendar = st.checkbox("Enable Calendar Integration", 
                    value=config.get("integrations", {}).get("enable_calendar_integration", False))
                enable_database = st.checkbox("Enable Database Integration", 
                    value=config.get("integrations", {}).get("enable_database_integration", False))
            
            if st.button("💾 Update Integrations Settings"):
                config["integrations"]["enable_web_search"] = enable_web_search
                config["integrations"]["web_search_provider"] = web_search_provider
                config["integrations"]["enable_file_processing"] = enable_file_processing
                config["integrations"]["max_file_size_mb"] = max_file_size
                config["integrations"]["enable_email_integration"] = enable_email
                config["integrations"]["email_provider"] = email_provider
                config["integrations"]["enable_calendar_integration"] = enable_calendar
                config["integrations"]["enable_database_integration"] = enable_database
                self.config_manager.save_config(config)
                st.success("Integrations settings updated!")
        
        # Development Settings
        with st.expander("Development Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_debug = st.checkbox("Enable Debug Mode", 
                    value=config.get("development", {}).get("enable_debug_mode", False))
                enable_hot_reload = st.checkbox("Enable Hot Reload", 
                    value=config.get("development", {}).get("enable_hot_reload", False))
                enable_profiling = st.checkbox("Enable Profiling", 
                    value=config.get("development", {}).get("enable_profiling", False))
                enable_testing = st.checkbox("Enable Testing Mode", 
                    value=config.get("development", {}).get("enable_testing_mode", False))
            
            with col2:
                enable_dev_logging = st.checkbox("Enable Development Logging", 
                    value=config.get("development", {}).get("enable_development_logging", False))
                enable_feature_flags = st.checkbox("Enable Feature Flags", 
                    value=config.get("development", {}).get("enable_feature_flags", False))
                enable_experimental = st.checkbox("Enable Experimental Features", 
                    value=config.get("development", {}).get("enable_experimental_features", False))
                enable_async = st.checkbox("Enable Async Processing", 
                    value=config.get("async_config", {}).get("enable_async_chat", True))
            
            if st.button("💾 Update Development Settings"):
                config["development"]["enable_debug_mode"] = enable_debug
                config["development"]["enable_hot_reload"] = enable_hot_reload
                config["development"]["enable_profiling"] = enable_profiling
                config["development"]["enable_testing_mode"] = enable_testing
                config["development"]["enable_development_logging"] = enable_dev_logging
                config["development"]["enable_feature_flags"] = enable_feature_flags
                config["development"]["enable_experimental_features"] = enable_experimental
                config["async_config"]["enable_async_chat"] = enable_async
                self.config_manager.save_config(config)
                st.success("Development settings updated!")
        
        # Async Configuration
        with st.expander("Async Processing Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_async_chat = st.checkbox("Enable Async Chat", 
                    value=config.get("async_config", {}).get("enable_async_chat", True))
                enable_async_vector = st.checkbox("Enable Async Vector Search", 
                    value=config.get("async_config", {}).get("enable_async_vector_search", True))
                enable_async_docs = st.checkbox("Enable Async Document Processing", 
                    value=config.get("async_config", {}).get("enable_async_document_processing", True))
                max_concurrent_requests = st.number_input("Max Concurrent Requests", min_value=1, max_value=50, 
                    value=config.get("async_config", {}).get("max_concurrent_requests", 5))
            
            with col2:
                timeout_seconds = st.number_input("Timeout (seconds)", min_value=5, max_value=300, 
                    value=config.get("async_config", {}).get("timeout_seconds", 30), key="async_timeout")
                thread_pool_size = st.number_input("Thread Pool Size", min_value=1, max_value=20, 
                    value=config.get("async_config", {}).get("thread_pool_size", 4))
                async_queue_size = st.number_input("Async Queue Size", min_value=10, max_value=1000, 
                    value=config.get("async_config", {}).get("async_queue_size", 100))
                enable_async_monitoring = st.checkbox("Enable Async Monitoring", 
                    value=config.get("async_config", {}).get("enable_async_monitoring", True))
            
            if st.button("💾 Update Async Settings"):
                config["async_config"]["enable_async_chat"] = enable_async_chat
                config["async_config"]["enable_async_vector_search"] = enable_async_vector
                config["async_config"]["enable_async_document_processing"] = enable_async_docs
                config["async_config"]["max_concurrent_requests"] = max_concurrent_requests
                config["async_config"]["timeout_seconds"] = timeout_seconds
                config["async_config"]["thread_pool_size"] = thread_pool_size
                config["async_config"]["async_queue_size"] = async_queue_size
                config["async_config"]["enable_async_monitoring"] = enable_async_monitoring
                self.config_manager.save_config(config)
                st.success("Async settings updated!")
