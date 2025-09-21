"""
Settings Page for DurgasAI Application.

Handles application settings, API configuration, and user preferences.
"""

import streamlit as st
from pathlib import Path
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from core.application_state import ApplicationState
from components.common.export_tools import ExportTools


def render_settings_page() -> None:
    """
    Render the settings page with comprehensive configuration options and logging.
    
    This function creates a complete settings interface including:
    - API configuration for various AI providers
    - UI preferences and customization options
    - Chat settings and behavior configuration
    - Model default parameters
    - Data management and export tools
    
    All settings changes are logged for audit trails and debugging.
    """
    # Import logging utilities with fallback
    try:
        from utils.logger import debug, info, warning, error, log_user_action
        LOGGING_AVAILABLE = True
    except ImportError:
        LOGGING_AVAILABLE = False
        def debug(msg, component="settings_page", **kwargs): pass
        def info(msg, component="settings_page", **kwargs): pass
        def warning(msg, component="settings_page", **kwargs): pass
        def error(msg, component="settings_page", **kwargs): pass
        def log_user_action(action, **kwargs): pass
    
    debug("Starting settings page render", "settings_page")
    
    try:
        # Log page access for analytics
        log_user_action("settings_page_visited", 
                       timestamp=datetime.now().isoformat(),
                       logging_available=LOGGING_AVAILABLE)
        
        # Render main header
        debug("Rendering settings page header", "settings_page")
        st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
        
        # Create main tabs for different configuration files
        debug("Creating comprehensive settings tabs", "settings_page")
        api_tab, app_tab, model_tab, catalog_tab, comprehensive_tab, env_tab, data_tab = st.tabs([
            "🔑 API Config", 
            "⚙️ App Config", 
            "🤖 Model Config",
            "📚 Model Catalog",
            "🔧 Comprehensive Config",
            "🌐 Environment Variables",
            "📁 Data Management"
        ])
        
        with api_tab:
            # API Configuration Section (api_config.json)
            debug("Rendering API configuration section", "settings_page")
            try:
                _render_api_config_file()
                debug("API configuration section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render API configuration section", "settings_page", e)
                st.error("❌ Failed to load API configuration")
        
        with app_tab:
            # App Configuration Section (app_config.json)
            debug("Rendering app configuration section", "settings_page")
            try:
                _render_app_config_file()
                debug("App configuration section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render app configuration section", "settings_page", e)
                st.error("❌ Failed to load app configuration")
        
        with model_tab:
            # Model Configuration Section (model_config.json)
            debug("Rendering model configuration section", "settings_page")
            try:
                _render_model_config_file()
                debug("Model configuration section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render model configuration section", "settings_page", e)
                st.error("❌ Failed to load model configuration")
        
        with catalog_tab:
            # Model Catalog Section (comprehensive_model_catalog.json)
            debug("Rendering model catalog section", "settings_page")
            try:
                _render_model_catalog_file()
                debug("Model catalog section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render model catalog section", "settings_page", e)
                st.error("❌ Failed to load model catalog")
        
        with comprehensive_tab:
            # Comprehensive Configuration Section (comprehensive_config.json)
            debug("Rendering comprehensive configuration section", "settings_page")
            try:
                _render_comprehensive_config_file()
                debug("Comprehensive configuration section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render comprehensive configuration section", "settings_page", e)
                st.error("❌ Failed to load comprehensive configuration")
        
        with env_tab:
            # Environment Variables Section
            debug("Rendering environment variables section", "settings_page")
            try:
                _render_environment_variables_section()
                debug("Environment variables section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render environment variables section", "settings_page", e)
                st.error("❌ Failed to load environment variables")
        
        with data_tab:
            # Data Management Section
            debug("Rendering data management section", "settings_page")
            try:
                _render_data_management()
                debug("Data management section rendered successfully", "settings_page")
            except Exception as e:
                error("Failed to render data management section", "settings_page", e)
                st.error("❌ Failed to load data management")
        
        # Log successful page render
        info("Settings page rendered successfully", "settings_page",
             sections_rendered=["api_config", "ui_preferences", "chat_config", "model_defaults", "data_management"],
             render_timestamp=datetime.now().isoformat())
        
    except Exception as e:
        # Handle any unexpected errors during page render
        error("Critical error rendering settings page", "settings_page", e)
        st.error("🚨 An error occurred while loading the settings page. Please refresh and try again.")
        
        # Show basic fallback content
        st.markdown("## ⚙️ Settings")
        st.markdown("Please refresh the page to access the full settings interface.")


def _render_api_config_file() -> None:
    """Render API configuration from api_config.json."""
    st.markdown("### 🔑 API Configuration (`api_config.json`)")
    st.markdown("Configure API keys and endpoints for various AI providers.")
    
    # Load current config
    config_data = _load_config_file()
    
    # Create tabs for different API providers
    tab1, tab2, tab3, tab4 = st.tabs(["🤗 AI Models", "🔍 Search & Web", "🤖 Advanced AI", "🔧 Other Services"])
    
    with tab1:
        _render_ai_model_apis(config_data)
    
    with tab2:
        _render_search_apis(config_data)
    
    with tab3:
        _render_advanced_ai_apis(config_data)
    
    with tab4:
        _render_other_services_apis(config_data)
    
    # API Key Status and Validation
    _render_api_status(config_data)
    
    # Save button
    if st.button("💾 Save All API Keys", type="primary"):
        _save_config_file(config_data)
        st.success("✅ All API keys saved successfully!")
        st.rerun()


def _render_api_status(config_data: Dict[str, Any]) -> None:
    """Render API key status and validation section."""
    st.markdown("#### 📊 API Key Status")
    
    # Define API providers with their status
    api_providers = {
        "🤗 HuggingFace": config_data.get('huggingface', {}).get('api_keys', ''),
        "🧠 OpenAI": config_data.get('openai', {}).get('api_keys', ''),
        "🧬 Anthropic": config_data.get('anthropic', {}).get('api_keys', ''),
        "⚡ Groq": config_data.get('groq', {}).get('api_keys', ''),
        "💬 Cohere": config_data.get('cohere', {}).get('api_keys', ''),
        "🚀 NVIDIA": config_data.get('nvidia', {}).get('api_keys', ''),
        "🔗 OpenRouter": config_data.get('openrouter', {}).get('api_keys', ''),
        "🧠 DeepSeek": config_data.get('deepseek', {}).get('api_keys', ''),
        "🔍 Google": config_data.get('google', {}).get('api_keys', ''),
        "🔎 SerpAPI": config_data.get('serpapi', {}).get('api_keys', ''),
        "🌐 Tavily": config_data.get('tavily', {}).get('api_keys', ''),
        "🤔 Perplexity": config_data.get('perplexity', {}).get('api_keys', ''),
        "🌊 Jina": config_data.get('jina', {}).get('api_keys', ''),
        "🎨 FalAI": config_data.get('falai', {}).get('api_keys', ''),
        "🐙 GitHub": config_data.get('github', {}).get('api_keys', ''),
        "🔍 Exa": config_data.get('exa', {}).get('api_keys', '')
    }
    
    # Create columns for status display
    col1, col2, col3 = st.columns(3)
    
    configured_count = 0
    total_count = len(api_providers)
    
    with col1:
        st.markdown("**✅ Configured APIs:**")
        for provider, key in api_providers.items():
            if key and len(key) > 10:  # Basic validation
                st.write(f"✅ {provider}")
                configured_count += 1
    
    with col2:
        st.markdown("**❌ Missing APIs:**")
        for provider, key in api_providers.items():
            if not key or len(key) <= 10:
                st.write(f"❌ {provider}")
    
    with col3:
        st.markdown("**📊 Summary:**")
        st.metric("Configured", configured_count)
        st.metric("Total", total_count)
        st.metric("Coverage", f"{(configured_count/total_count)*100:.1f}%")
    
    # Quick actions
    st.markdown("#### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col2:
        if st.button("🧪 Test APIs"):
            st.info("API testing feature coming soon!")
    
    with col3:
        if st.button("📋 Export Config"):
            _export_config_backup(config_data)


def _render_ai_model_apis(config_data: Dict[str, Any]) -> None:
    """Render AI model API configurations."""
    st.markdown("#### 🤗 AI Model Providers")
    
    # HuggingFace
    hf_config = config_data.get('huggingface', {})
    hf_token = st.text_input(
        "🤗 HuggingFace API Token:",
        value=hf_config.get('api_keys', ''),
        type="password",
        help="Your HuggingFace API token for accessing models",
        key="hf_api_key"
    )
    config_data.setdefault('huggingface', {})['api_keys'] = hf_token
    
    # OpenAI
    openai_config = config_data.get('openai', {})
    openai_token = st.text_input(
        "🧠 OpenAI API Key:",
        value=openai_config.get('api_keys', ''),
        type="password",
        help="Your OpenAI API key for GPT models",
        key="openai_api_key"
    )
    config_data.setdefault('openai', {})['api_keys'] = openai_token
    
    # Anthropic
    anthropic_config = config_data.get('anthropic', {})
    anthropic_token = st.text_input(
        "🧬 Anthropic API Key:",
        value=anthropic_config.get('api_keys', ''),
        type="password",
        help="Your Anthropic API key for Claude models",
        key="anthropic_api_key"
    )
    config_data.setdefault('anthropic', {})['api_keys'] = anthropic_token
    
    # Groq
    groq_config = config_data.get('groq', {})
    groq_token = st.text_input(
        "⚡ Groq API Key:",
        value=groq_config.get('api_keys', ''),
        type="password",
        help="Your Groq API key for fast inference",
        key="groq_api_key"
    )
    config_data.setdefault('groq', {})['api_keys'] = groq_token


def _render_search_apis(config_data: Dict[str, Any]) -> None:
    """Render search and web API configurations."""
    st.markdown("#### 🔍 Search & Web Services")
    
    # Google
    google_config = config_data.get('google', {})
    google_token = st.text_input(
        "🔍 Google API Key:",
        value=google_config.get('api_keys', ''),
        type="password",
        help="Your Google API key for search and services",
        key="google_api_key"
    )
    config_data.setdefault('google', {})['api_keys'] = google_token
    
    # SerpAPI
    serpapi_config = config_data.get('serpapi', {})
    serpapi_token = st.text_input(
        "🔎 SerpAPI Key:",
        value=serpapi_config.get('api_keys', ''),
        type="password",
        help="Your SerpAPI key for Google search results",
        key="serpapi_api_key"
    )
    config_data.setdefault('serpapi', {})['api_keys'] = serpapi_token
    
    # Tavily
    tavily_config = config_data.get('tavily', {})
    tavily_token = st.text_input(
        "🌐 Tavily API Key:",
        value=tavily_config.get('api_keys', ''),
        type="password",
        help="Your Tavily API key for web search",
        key="tavily_api_key"
    )
    config_data.setdefault('tavily', {})['api_keys'] = tavily_token
    
    # Perplexity
    perplexity_config = config_data.get('perplexity', {})
    perplexity_token = st.text_input(
        "🤔 Perplexity API Key:",
        value=perplexity_config.get('api_keys', ''),
        type="password",
        help="Your Perplexity API key for AI-powered search",
        key="perplexity_api_key"
    )
    config_data.setdefault('perplexity', {})['api_keys'] = perplexity_token


def _render_advanced_ai_apis(config_data: Dict[str, Any]) -> None:
    """Render advanced AI API configurations."""
    st.markdown("#### 🤖 Advanced AI Services")
    
    # Cohere
    cohere_config = config_data.get('cohere', {})
    cohere_token = st.text_input(
        "💬 Cohere API Key:",
        value=cohere_config.get('api_keys', ''),
        type="password",
        help="Your Cohere API key for command models",
        key="cohere_api_key"
    )
    config_data.setdefault('cohere', {})['api_keys'] = cohere_token
    
    # NVIDIA
    nvidia_config = config_data.get('nvidia', {})
    nvidia_token = st.text_input(
        "🚀 NVIDIA API Key:",
        value=nvidia_config.get('api_keys', ''),
        type="password",
        help="Your NVIDIA API key for Nemotron models",
        key="nvidia_api_key"
    )
    config_data.setdefault('nvidia', {})['api_keys'] = nvidia_token
    
    # OpenRouter
    openrouter_config = config_data.get('openrouter', {})
    openrouter_token = st.text_input(
        "🔗 OpenRouter API Key:",
        value=openrouter_config.get('api_keys', ''),
        type="password",
        help="Your OpenRouter API key for multiple AI models",
        key="openrouter_api_key"
    )
    config_data.setdefault('openrouter', {})['api_keys'] = openrouter_token
    
    # DeepSeek
    deepseek_config = config_data.get('deepseek', {})
    deepseek_token = st.text_input(
        "🧠 DeepSeek API Key:",
        value=deepseek_config.get('api_keys', ''),
        type="password",
        help="Your DeepSeek API key for advanced reasoning",
        key="deepseek_api_key"
    )
    config_data.setdefault('deepseek', {})['api_keys'] = deepseek_token


def _render_other_services_apis(config_data: Dict[str, Any]) -> None:
    """Render other services API configurations."""
    st.markdown("#### 🔧 Other Services")
    
    # Jina
    jina_config = config_data.get('jina', {})
    jina_token = st.text_input(
        "🌊 Jina API Key:",
        value=jina_config.get('api_keys', ''),
        type="password",
        help="Your Jina API key for embeddings and search",
        key="jina_api_key"
    )
    config_data.setdefault('jina', {})['api_keys'] = jina_token
    
    # FalAI
    falai_config = config_data.get('falai', {})
    falai_token = st.text_input(
        "🎨 FalAI API Key:",
        value=falai_config.get('api_keys', ''),
        type="password",
        help="Your FalAI API key for image generation",
        key="falai_api_key"
    )
    config_data.setdefault('falai', {})['api_keys'] = falai_token
    
    # GitHub
    github_config = config_data.get('github', {})
    github_token = st.text_input(
        "🐙 GitHub API Key:",
        value=github_config.get('api_keys', ''),
        type="password",
        help="Your GitHub API key for repository access",
        key="github_api_key"
    )
    config_data.setdefault('github', {})['api_keys'] = github_token
    
    # Exa
    exa_config = config_data.get('exa', {})
    exa_token = st.text_input(
        "🔍 Exa API Key:",
        value=exa_config.get('api_keys', ''),
        type="password",
        help="Your Exa API key for web search",
        key="exa_api_key"
    )
    config_data.setdefault('exa', {})['api_keys'] = exa_token


def _load_config_file() -> Dict[str, Any]:
    """Load configuration from distributed config files using ConfigService."""
    try:
        from services import ConfigService
        config_service = ConfigService()
        
        # Get API configuration (contains all API keys)
        api_config = config_service.get_api_config()
        
        if api_config:
            return api_config
        else:
            st.warning("⚠️ API configuration not found. Using default configuration.")
            return {}
    except ImportError as e:
        st.error(f"❌ ConfigService not available: {e}")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading configuration: {e}")
        return {}


def _save_config_file(config_data: Dict[str, Any]) -> None:
    """Save configuration to distributed config files using ConfigService."""
    try:
        from services import ConfigService
        config_service = ConfigService()
        
        # Save to API config file
        success = config_service.save_config('api', config_data)
        
        if success:
            st.success("✅ Configuration saved successfully to config/api_config.json!")
        else:
            st.error("❌ Failed to save configuration")
        
    except ImportError as e:
        st.error(f"❌ ConfigService not available: {e}")
    except Exception as e:
        st.error(f"❌ Error saving configuration: {e}")


def _export_config_backup(config_data: Dict[str, Any]) -> None:
    """Export configuration as a downloadable backup file."""
    try:
        # Create a backup with masked sensitive data
        backup_data = {}
        for provider, config in config_data.items():
            if isinstance(config, dict) and 'api_keys' in config:
                backup_data[provider] = {
                    **config,
                    'api_keys': '***MASKED***' if config.get('api_keys') else ''
                }
            else:
                backup_data[provider] = config
        
        # Convert to JSON string
        json_str = json.dumps(backup_data, indent=2, ensure_ascii=False)
        
        # Create download button
        st.download_button(
            label="📥 Download Config Backup",
            data=json_str,
            file_name="durgasai_config_backup.json",
            mime="application/json",
            help="Download a backup of your configuration (API keys are masked for security)"
        )
        
    except Exception as e:
        st.error(f"❌ Error creating backup: {e}")


def _render_app_settings() -> None:
    """Render comprehensive app settings from app_config.json."""
    st.markdown("### ⚙️ Application Settings")
    st.markdown("Configure application-wide settings from `config/app_config.json`")
    
    try:
        from services import ConfigService
        config_service = ConfigService()
        app_config = config_service.get_app_config()
        
        # Create sub-tabs for different app setting categories
        ui_tab, chat_tab, perf_tab, log_tab = st.tabs([
            "🎨 UI Settings", 
            "💬 Chat Settings", 
            "⚡ Performance", 
            "📝 Logging"
        ])
        
        with ui_tab:
            _render_ui_settings(app_config, config_service)
        
        with chat_tab:
            _render_chat_settings_app(app_config, config_service)
        
        with perf_tab:
            _render_performance_settings(app_config, config_service)
        
        with log_tab:
            _render_logging_settings(app_config, config_service)
            
    except Exception as e:
        st.error(f"❌ Error loading app settings: {e}")


def _render_ui_preferences() -> None:
    """Render UI preferences section (legacy - kept for compatibility)."""
    st.markdown("### 🎨 UI Preferences")
    
    # Theme settings
    theme = st.selectbox(
        "Theme:",
        ["Auto", "Light", "Dark"],
        index=0
    )
    
    # Debug mode toggle
    debug_mode = st.checkbox(
        "Enable Debug Mode",
        value=ApplicationState.get("debug_mode", False),
        help="Show additional debugging information"
    )
    
    if debug_mode != ApplicationState.get("debug_mode", False):
        ApplicationState.set("debug_mode", debug_mode)
        st.success("Debug mode updated!")


def _render_chat_configuration() -> None:
    """Render chat configuration section."""
    st.markdown("### 💬 Chat Configuration")
    
    max_messages = st.slider(
        "Maximum chat history:",
        min_value=10,
        max_value=100,
        value=Config.MAX_MESSAGE_HISTORY,
        help="Maximum number of messages to keep in chat history"
    )
    
    auto_save = st.checkbox(
        "Auto-save conversations",
        value=True,
        help="Automatically save conversation history"
    )


def _render_model_defaults() -> None:
    """Render model default settings."""
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


def _render_data_management() -> None:
    """Render data management section."""
    st.markdown("### 📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data"):
            ApplicationState.clear()
            st.success("All data cleared!")
    
    with col2:
        messages = ApplicationState.get("messages", [])
        if messages:
            export_tools = ExportTools()
            export_tools.render_export_button(messages)


def _render_ui_settings(app_config: Dict[str, Any], config_service) -> None:
    """Render UI settings configuration."""
    st.markdown("#### 🎨 User Interface Settings")
    
    ui_config = app_config.get('ui', {})
    updated_ui_config = ui_config.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Theme settings
        theme_options = ["auto", "light", "dark"]
        current_theme = ui_config.get('theme', 'auto')
        theme = st.selectbox(
            "🎨 Theme:",
            theme_options,
            index=theme_options.index(current_theme) if current_theme in theme_options else 0,
            help="Application color theme"
        )
        updated_ui_config['theme'] = theme
        
        # Sidebar width
        sidebar_width = st.slider(
            "📏 Sidebar Width:",
            min_value=200,
            max_value=500,
            value=ui_config.get('sidebar_width', 300),
            help="Width of the sidebar in pixels"
        )
        updated_ui_config['sidebar_width'] = sidebar_width
        
        # Chat height
        chat_height = st.slider(
            "💬 Chat Height:",
            min_value=400,
            max_value=800,
            value=ui_config.get('chat_height', 600),
            help="Height of chat interface in pixels"
        )
        updated_ui_config['chat_height'] = chat_height
    
    with col2:
        # Enable animations
        enable_animations = st.checkbox(
            "✨ Enable Animations",
            value=ui_config.get('enable_animations', True),
            help="Enable UI animations and transitions"
        )
        updated_ui_config['enable_animations'] = enable_animations
        
        # Enable sound effects
        enable_sound_effects = st.checkbox(
            "🔊 Enable Sound Effects",
            value=ui_config.get('enable_sound_effects', False),
            help="Enable sound effects for interactions"
        )
        updated_ui_config['enable_sound_effects'] = enable_sound_effects
        
        # Enable keyboard shortcuts
        enable_keyboard_shortcuts = st.checkbox(
            "⌨️ Enable Keyboard Shortcuts",
            value=ui_config.get('enable_keyboard_shortcuts', True),
            help="Enable keyboard shortcuts for faster navigation"
        )
        updated_ui_config['enable_keyboard_shortcuts'] = enable_keyboard_shortcuts
        
        # Default page
        default_page_options = ["chat", "aiagent", "huggingface", "settings"]
        current_default = ui_config.get('default_page', 'chat')
        default_page = st.selectbox(
            "🏠 Default Page:",
            default_page_options,
            index=default_page_options.index(current_default) if current_default in default_page_options else 0,
            help="Page to show when app starts"
        )
        updated_ui_config['default_page'] = default_page
    
    # Save UI settings
    if st.button("💾 Save UI Settings", key="save_ui_settings"):
        app_config['ui'] = updated_ui_config
        success = config_service.save_config('app', app_config)
        if success:
            st.success("✅ UI settings saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save UI settings")


def _render_app_config_file() -> None:
    """Render complete app_config.json configuration interface."""
    st.markdown("### ⚙️ App Configuration (`app_config.json`)")
    st.markdown("Configure application-wide settings including UI, chat, performance, and logging.")
    
    try:
        from services import ConfigService
        config_service = ConfigService()
        app_config = config_service.get_app_config()
        
        # Create sub-tabs for app config sections
        ui_tab, chat_tab, perf_tab, log_tab, app_info_tab = st.tabs([
            "🎨 UI Settings", 
            "💬 Chat Settings", 
            "⚡ Performance", 
            "📝 Logging",
            "ℹ️ App Info"
        ])
        
        with ui_tab:
            _render_ui_settings(app_config, config_service)
        
        with chat_tab:
            _render_chat_settings_app(app_config, config_service)
        
        with perf_tab:
            _render_performance_settings(app_config, config_service)
        
        with log_tab:
            _render_logging_settings(app_config, config_service)
            
        with app_info_tab:
            _render_app_info_settings(app_config, config_service)
            
    except Exception as e:
        st.error(f"❌ Error loading app configuration: {e}")


def _render_model_config_file() -> None:
    """Render model_config.json configuration interface."""
    st.markdown("### 🤖 Model Configuration (`model_config.json`)")
    st.markdown("Configure AI models, system prompts, and model-specific settings.")
    
    try:
        from services import ConfigService
        config_service = ConfigService()
        model_config = config_service.get_model_config()
        
        # Create tabs for model config sections
        models_tab, prompts_tab, providers_tab = st.tabs([
            "🤖 Models", 
            "💭 System Prompts", 
            "🔌 Providers"
        ])
        
        with models_tab:
            _render_models_configuration(model_config, config_service)
        
        with prompts_tab:
            _render_system_prompts_configuration(model_config, config_service)
        
        with providers_tab:
            _render_providers_configuration(model_config, config_service)
            
    except Exception as e:
        st.error(f"❌ Error loading model configuration: {e}")


def _render_model_catalog_file() -> None:
    """Render comprehensive_model_catalog.json interface."""
    st.markdown("### 📚 Model Catalog (`comprehensive_model_catalog.json`)")
    st.markdown("Browse and manage the comprehensive model catalog with categories and metadata.")
    
    try:
        from services import ConfigService
        config_service = ConfigService()
        catalog = config_service.get_model_catalog()
        
        if not catalog:
            st.warning("⚠️ Model catalog not found")
            return
        
        # Display catalog statistics
        st.markdown("#### 📊 Catalog Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Catalog Version", catalog.get('catalog_version', 'Unknown'))
        with col2:
            st.metric("Total Models", f"{catalog.get('total_models', 0):,}")
        with col3:
            st.metric("Categories", len(catalog.get('categories', {})))
        
        # Display categories
        categories = catalog.get('categories', {})
        if categories:
            st.markdown("#### 📂 Model Categories")
            
            for category_key, category_data in categories.items():
                with st.expander(f"{category_data.get('icon', '📁')} {category_data.get('title', category_key)} ({category_data.get('model_count', 0):,} models)"):
                    st.markdown(f"**Description**: {category_data.get('description', 'No description')}")
                    
                    subcategories = category_data.get('subcategories', {})
                    if subcategories:
                        st.markdown("**Subcategories:**")
                        for subcat_key, subcat_data in subcategories.items():
                            st.write(f"- **{subcat_data.get('name', subcat_key)}**: {subcat_data.get('count', 0)} models")
                            
                            # Show sample models if available
                            models = subcat_data.get('models', [])
                            if models:
                                st.markdown(f"  *Sample models:*")
                                for model in models[:3]:  # Show first 3 models
                                    st.write(f"    - {model.get('name', model.get('id', 'Unknown'))}")
        
        # Catalog management
        st.markdown("#### 🔧 Catalog Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Catalog"):
                st.success("Catalog refreshed!")
                st.rerun()
        
        with col2:
            if st.button("📥 Export Catalog"):
                catalog_json = json.dumps(catalog, indent=2)
                st.download_button(
                    "💾 Download Catalog",
                    catalog_json,
                    "model_catalog.json",
                    "application/json"
                )
        
    except Exception as e:
        st.error(f"❌ Error loading model catalog: {e}")


def _render_comprehensive_config_file() -> None:
    """Render comprehensive_config.json interface."""
    st.markdown("### 🔧 Comprehensive Configuration (`comprehensive_config.json`)")
    st.markdown("Configure advanced features, tools, workflows, and system settings.")
    
    try:
        from services import ConfigService
        config_service = ConfigService()
        comprehensive_config = config_service.get_comprehensive_config()
        
        if not comprehensive_config:
            st.warning("⚠️ Comprehensive configuration not found")
            return
        
        # Create tabs for comprehensive config sections
        features_tab, tools_tab, system_tab, dev_tab = st.tabs([
            "🚀 Features", 
            "🛠️ Tools & Workflows", 
            "⚙️ System Settings",
            "🧪 Development"
        ])
        
        with features_tab:
            _render_features_configuration(comprehensive_config, config_service)
        
        with tools_tab:
            _render_tools_workflows_configuration(comprehensive_config, config_service)
        
        with system_tab:
            _render_system_configuration(comprehensive_config, config_service)
            
        with dev_tab:
            _render_development_configuration(comprehensive_config, config_service)
            
    except Exception as e:
        st.error(f"❌ Error loading comprehensive configuration: {e}")


def _render_app_info_settings(app_config: Dict[str, Any], config_service) -> None:
    """Render application information settings."""
    st.markdown("#### ℹ️ Application Information")
    
    app_info = app_config.get('application', {})
    updated_app_info = app_info.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # App title
        app_title = st.text_input(
            "📱 Application Title:",
            value=app_info.get('title', '🤖 DurgasAI - Advanced AI Agent'),
            help="Title shown in browser tab and header"
        )
        updated_app_info['title'] = app_title
        
        # App icon
        app_icon = st.text_input(
            "🎨 Application Icon:",
            value=app_info.get('icon', '🤖'),
            help="Emoji or icon for the application"
        )
        updated_app_info['icon'] = app_icon
        
        # Version
        version = st.text_input(
            "🔢 Version:",
            value=app_info.get('version', '2.0.0'),
            help="Application version number"
        )
        updated_app_info['version'] = version
    
    with col2:
        # Description
        description = st.text_area(
            "📝 Description:",
            value=app_info.get('description', 'Advanced AI Agent Platform'),
            help="Application description"
        )
        updated_app_info['description'] = description
        
        # Author
        author = st.text_input(
            "👤 Author:",
            value=app_info.get('author', 'DurgasAI Team'),
            help="Application author or team"
        )
        updated_app_info['author'] = author
        
        # Homepage
        homepage = st.text_input(
            "🌐 Homepage:",
            value=app_info.get('homepage', 'https://github.com/durgasai/durgasai'),
            help="Project homepage URL"
        )
        updated_app_info['homepage'] = homepage
    
    # Save app info
    if st.button("💾 Save App Info", key="save_app_info"):
        app_config['application'] = updated_app_info
        success = config_service.save_config('app', app_config)
        if success:
            st.success("✅ Application info saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save application info")


def _render_models_configuration(model_config: Dict[str, Any], config_service) -> None:
    """Render models configuration interface."""
    st.markdown("#### 🤖 Model Configurations")
    
    models = model_config.get('models', {})
    
    if not models:
        st.warning("⚠️ No models configured")
        return
    
    # Model selection for editing
    model_names = list(models.keys())
    selected_model = st.selectbox(
        "Select Model to Configure:",
        model_names,
        help="Choose a model to view and edit its configuration"
    )
    
    if selected_model and selected_model in models:
        model_data = models[selected_model]
        st.markdown(f"#### Editing: **{model_data.get('name', selected_model)}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Model ID:", value=model_data.get('model_id', ''), disabled=True)
            st.text_input("Provider:", value=model_data.get('provider', ''), disabled=True)
            st.text_area("Description:", value=model_data.get('description', ''), height=100)
            
        with col2:
            st.slider("Max Tokens:", 50, 4096, value=model_data.get('max_tokens', 512))
            st.slider("Temperature:", 0.0, 2.0, value=model_data.get('temperature', 0.7), step=0.1)
            st.slider("Top P:", 0.0, 1.0, value=model_data.get('top_p', 0.9), step=0.1)
            st.checkbox("Enabled:", value=model_data.get('enabled', True))
        
        if st.button(f"💾 Save {selected_model} Config", key=f"save_model_{selected_model}"):
            st.info("Model configuration saving will be implemented in next version")


def _render_system_prompts_configuration(model_config: Dict[str, Any], config_service) -> None:
    """Render system prompts configuration."""
    st.markdown("#### 💭 System Prompts")
    
    prompts = model_config.get('system_prompts', {})
    
    if prompts:
        for prompt_name, prompt_text in prompts.items():
            with st.expander(f"📝 {prompt_name.replace('_', ' ').title()}"):
                st.text_area(
                    "Prompt Text:",
                    value=prompt_text,
                    height=100,
                    key=f"prompt_{prompt_name}",
                    disabled=True  # Read-only for now
                )
    else:
        st.info("No system prompts configured")


def _render_providers_configuration(model_config: Dict[str, Any], config_service) -> None:
    """Render providers configuration."""
    st.markdown("#### 🔌 Model Providers")
    
    providers = model_config.get('providers', {})
    
    if providers:
        for provider_name, provider_data in providers.items():
            with st.expander(f"🔌 {provider_data.get('name', provider_name)}"):
                st.write(f"**Requires Token**: {provider_data.get('requires_token', 'Unknown')}")
                st.write(f"**Supports Streaming**: {provider_data.get('supports_streaming', 'Unknown')}")
                
                rate_limits = provider_data.get('rate_limits', {})
                if rate_limits:
                    st.write("**Rate Limits**:")
                    for limit_type, limit_value in rate_limits.items():
                        st.write(f"  - {limit_type}: {limit_value}")
    else:
        st.info("No providers configured")


def _render_features_configuration(comprehensive_config: Dict[str, Any], config_service) -> None:
    """Render features configuration from comprehensive config."""
    st.markdown("#### 🚀 Feature Settings")
    
    # Chat features
    chat_config = comprehensive_config.get('chat', {})
    if chat_config:
        with st.expander("💬 Chat Features"):
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Enable Tools", value=chat_config.get('enable_tools', True), disabled=True)
                st.checkbox("Enable Vector Search", value=chat_config.get('enable_vector_search', True), disabled=True)
                st.checkbox("Enable Multimodal", value=chat_config.get('enable_multimodal', True), disabled=True)
            with col2:
                st.checkbox("Enable Async Processing", value=chat_config.get('enable_async_processing', True), disabled=True)
                st.checkbox("Stream Responses", value=chat_config.get('stream_responses', True), disabled=True)
                st.checkbox("Enable Avatars", value=chat_config.get('enable_avatars', True), disabled=True)
    
    # Vector DB features
    vector_config = comprehensive_config.get('vector_db', {})
    if vector_config:
        with st.expander("🗃️ Vector Database"):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("DB Path", value=vector_config.get('db_path', ''), disabled=True)
                st.text_input("Collection Name", value=vector_config.get('collection_name', ''), disabled=True)
                st.text_input("Embedding Model", value=vector_config.get('embedding_model', ''), disabled=True)
            with col2:
                st.number_input("Max Results", value=vector_config.get('max_results', 5), disabled=True)
                st.number_input("Similarity Threshold", value=vector_config.get('similarity_threshold', 0.7), disabled=True)
                st.checkbox("Persistent Storage", value=vector_config.get('enable_persistent_storage', True), disabled=True)


def _render_tools_workflows_configuration(comprehensive_config: Dict[str, Any], config_service) -> None:
    """Render tools and workflows configuration."""
    st.markdown("#### 🛠️ Tools & Workflows")
    
    # Tools configuration
    tools_config = comprehensive_config.get('tools', {})
    if tools_config:
        with st.expander("🔧 Tools Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Enable Tool Management", value=tools_config.get('enable_tool_management', True), disabled=True)
                st.text_input("Tools Directory", value=tools_config.get('tools_directory', ''), disabled=True)
                st.number_input("Max Execution Time", value=tools_config.get('max_tool_execution_time', 60), disabled=True)
            with col2:
                st.checkbox("Enable Tool Caching", value=tools_config.get('enable_tool_caching', True), disabled=True)
                st.checkbox("Enable Tool Logging", value=tools_config.get('enable_tool_logging', True), disabled=True)
                st.checkbox("Enable Tool Validation", value=tools_config.get('enable_tool_validation', True), disabled=True)
    
    # Workflows configuration
    workflows_config = comprehensive_config.get('workflows', {})
    if workflows_config:
        with st.expander("🔄 Workflows Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Enable Workflow Management", value=workflows_config.get('enable_workflow_management', True), disabled=True)
                st.text_input("Workflows Directory", value=workflows_config.get('workflows_directory', ''), disabled=True)
                st.number_input("Max Workflow Steps", value=workflows_config.get('max_workflow_steps', 20), disabled=True)
            with col2:
                st.checkbox("Enable Workflow Monitoring", value=workflows_config.get('enable_workflow_monitoring', True), disabled=True)
                st.number_input("Max Concurrent Workflows", value=workflows_config.get('max_concurrent_workflows', 3), disabled=True)
                st.number_input("Workflow Timeout (min)", value=workflows_config.get('workflow_timeout_minutes', 30), disabled=True)


def _render_system_configuration(comprehensive_config: Dict[str, Any], config_service) -> None:
    """Render system configuration."""
    st.markdown("#### ⚙️ System Settings")
    
    # System monitor
    monitor_config = comprehensive_config.get('system_monitor', {})
    if monitor_config:
        with st.expander("📊 System Monitor"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Refresh Rate", value=monitor_config.get('refresh_rate', 5), disabled=True)
                st.number_input("Max History", value=monitor_config.get('max_history', 100), disabled=True)
                st.number_input("CPU Alert Threshold", value=monitor_config.get('alert_cpu_threshold', 80.0), disabled=True)
            with col2:
                st.checkbox("Show Network", value=monitor_config.get('show_network', True), disabled=True)
                st.checkbox("Show Disk", value=monitor_config.get('show_disk', True), disabled=True)
                st.checkbox("Enable Alerts", value=monitor_config.get('enable_alerts', True), disabled=True)
    
    # Integrations
    integrations_config = comprehensive_config.get('integrations', {})
    if integrations_config:
        with st.expander("🔗 Integrations"):
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Enable Web Search", value=integrations_config.get('enable_web_search', True), disabled=True)
                st.text_input("Web Search Provider", value=integrations_config.get('web_search_provider', ''), disabled=True)
                st.checkbox("Enable File Processing", value=integrations_config.get('enable_file_processing', True), disabled=True)
            with col2:
                st.number_input("Max File Size (MB)", value=integrations_config.get('max_file_size_mb', 50), disabled=True)
                st.checkbox("Enable Email Integration", value=integrations_config.get('enable_email_integration', False), disabled=True)
                st.checkbox("Enable Database Integration", value=integrations_config.get('enable_database_integration', False), disabled=True)


def _render_development_configuration(comprehensive_config: Dict[str, Any], config_service) -> None:
    """Render development configuration."""
    st.markdown("#### 🧪 Development Settings")
    
    dev_config = comprehensive_config.get('development', {})
    if dev_config:
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Debug Mode", value=dev_config.get('enable_debug_mode', False), disabled=True)
            st.checkbox("Enable Hot Reload", value=dev_config.get('enable_hot_reload', False), disabled=True)
            st.checkbox("Enable Profiling", value=dev_config.get('enable_profiling', False), disabled=True)
        
        with col2:
            st.checkbox("Enable Testing Mode", value=dev_config.get('enable_testing_mode', False), disabled=True)
            st.checkbox("Enable Development Logging", value=dev_config.get('enable_development_logging', False), disabled=True)
            st.checkbox("Enable Feature Flags", value=dev_config.get('enable_feature_flags', False), disabled=True)
        
        # Feature flags
        feature_flags = dev_config.get('feature_flags', {})
        if feature_flags:
            st.markdown("**Feature Flags:**")
            for flag_name, flag_value in feature_flags.items():
                st.write(f"- {flag_name}: {flag_value}")
        else:
            st.info("No feature flags configured")
    else:
        st.info("No development configuration found")


def _render_ui_settings(app_config: Dict[str, Any], config_service) -> None:
    """Render UI settings configuration."""
    st.markdown("#### 🎨 User Interface Settings")
    
    ui_config = app_config.get('ui', {})
    updated_ui_config = ui_config.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Theme settings
        theme_options = ["auto", "light", "dark"]
        current_theme = ui_config.get('theme', 'auto')
        theme = st.selectbox(
            "🎨 Theme:",
            theme_options,
            index=theme_options.index(current_theme) if current_theme in theme_options else 0,
            help="Application color theme"
        )
        updated_ui_config['theme'] = theme
        
        # Sidebar width
        sidebar_width = st.slider(
            "📏 Sidebar Width:",
            min_value=200,
            max_value=500,
            value=ui_config.get('sidebar_width', 300),
            help="Width of the sidebar in pixels"
        )
        updated_ui_config['sidebar_width'] = sidebar_width
    
    with col2:
        # Enable animations
        enable_animations = st.checkbox(
            "✨ Enable Animations",
            value=ui_config.get('enable_animations', True),
            help="Enable UI animations and transitions"
        )
        updated_ui_config['enable_animations'] = enable_animations
        
        # Enable keyboard shortcuts
        enable_keyboard_shortcuts = st.checkbox(
            "⌨️ Enable Keyboard Shortcuts",
            value=ui_config.get('enable_keyboard_shortcuts', True),
            help="Enable keyboard shortcuts for faster navigation"
        )
        updated_ui_config['enable_keyboard_shortcuts'] = enable_keyboard_shortcuts
    
    # Save UI settings
    if st.button("💾 Save UI Settings", key="save_ui_settings_main"):
        app_config['ui'] = updated_ui_config
        success = config_service.save_config('app', app_config)
        if success:
            st.success("✅ UI settings saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save UI settings")


def _render_chat_settings_app(app_config: Dict[str, Any], config_service) -> None:
    """Render chat settings configuration from app config."""
    st.markdown("#### 💬 Chat Configuration")
    
    chat_config = app_config.get('chat', {})
    updated_chat_config = chat_config.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Max message history
        max_message_history = st.slider(
            "📝 Max Message History:",
            min_value=10,
            max_value=200,
            value=chat_config.get('max_message_history', 50),
            help="Maximum number of messages to keep in history"
        )
        updated_chat_config['max_message_history'] = max_message_history
        
        # Auto save interval
        auto_save_interval = st.slider(
            "💾 Auto Save Interval (seconds):",
            min_value=10,
            max_value=300,
            value=chat_config.get('auto_save_interval', 30),
            help="How often to auto-save conversations"
        )
        updated_chat_config['auto_save_interval'] = auto_save_interval
    
    with col2:
        # Enable export
        enable_export = st.checkbox(
            "📤 Enable Export",
            value=chat_config.get('enable_export', True),
            help="Allow exporting chat conversations"
        )
        updated_chat_config['enable_export'] = enable_export
        
        # Enable import
        enable_import = st.checkbox(
            "📥 Enable Import",
            value=chat_config.get('enable_import', True),
            help="Allow importing chat conversations"
        )
        updated_chat_config['enable_import'] = enable_import
        
        # Enable typing indicator
        enable_typing_indicator = st.checkbox(
            "⌨️ Enable Typing Indicator",
            value=chat_config.get('enable_typing_indicator', True),
            help="Show typing indicator during AI response"
        )
        updated_chat_config['enable_typing_indicator'] = enable_typing_indicator
        
        # Enable message timestamps
        enable_message_timestamps = st.checkbox(
            "🕒 Enable Message Timestamps",
            value=chat_config.get('enable_message_timestamps', True),
            help="Show timestamps on chat messages"
        )
        updated_chat_config['enable_message_timestamps'] = enable_message_timestamps
    
    # Save chat settings
    if st.button("💾 Save Chat Settings", key="save_chat_settings_app"):
        app_config['chat'] = updated_chat_config
        success = config_service.save_config('app', app_config)
        if success:
            st.success("✅ Chat settings saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save chat settings")


def _render_performance_settings(app_config: Dict[str, Any], config_service) -> None:
    """Render performance settings configuration."""
    st.markdown("#### ⚡ Performance Settings")
    
    perf_config = app_config.get('performance', {})
    updated_perf_config = perf_config.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enable performance monitoring
        enable_performance_monitoring = st.checkbox(
            "📊 Enable Performance Monitoring",
            value=perf_config.get('enable_performance_monitoring', True),
            help="Monitor application performance metrics"
        )
        updated_perf_config['enable_performance_monitoring'] = enable_performance_monitoring
        
        # Log slow operations
        log_slow_operations = st.checkbox(
            "🐌 Log Slow Operations",
            value=perf_config.get('log_slow_operations', True),
            help="Log operations that exceed the threshold"
        )
        updated_perf_config['log_slow_operations'] = log_slow_operations
    
    with col2:
        # Slow operation threshold
        slow_operation_threshold = st.slider(
            "⏱️ Slow Operation Threshold (seconds):",
            min_value=0.1,
            max_value=10.0,
            value=perf_config.get('slow_operation_threshold', 2.0),
            step=0.1,
            help="Threshold for considering an operation slow"
        )
        updated_perf_config['slow_operation_threshold'] = slow_operation_threshold
        
        # Enable memory monitoring
        enable_memory_monitoring = st.checkbox(
            "🧠 Enable Memory Monitoring",
            value=perf_config.get('enable_memory_monitoring', True),
            help="Monitor memory usage"
        )
        updated_perf_config['enable_memory_monitoring'] = enable_memory_monitoring
    
    # Save performance settings
    if st.button("💾 Save Performance Settings", key="save_perf_settings_app"):
        app_config['performance'] = updated_perf_config
        success = config_service.save_config('app', app_config)
        if success:
            st.success("✅ Performance settings saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save performance settings")


def _render_logging_settings(app_config: Dict[str, Any], config_service) -> None:
    """Render logging settings configuration."""
    st.markdown("#### 📝 Logging Configuration")
    
    log_config = app_config.get('logging', {})
    updated_log_config = log_config.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Log level
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        current_level = log_config.get('log_level', 'INFO')
        log_level = st.selectbox(
            "📊 Log Level:",
            log_levels,
            index=log_levels.index(current_level) if current_level in log_levels else 1,
            help="Minimum log level to record"
        )
        updated_log_config['log_level'] = log_level
        
        # Log folder path
        log_folder_path = st.text_input(
            "📁 Log Folder Path:",
            value=log_config.get('log_folder_path', 'logs'),
            help="Directory where log files will be stored"
        )
        updated_log_config['log_folder_path'] = log_folder_path
        
        # Log retention days
        log_retention_days = st.slider(
            "🗓️ Log Retention (days):",
            min_value=1,
            max_value=365,
            value=log_config.get('log_retention_days', 30),
            help="How long to keep log files"
        )
        updated_log_config['log_retention_days'] = log_retention_days
    
    with col2:
        # Enable debug logging
        enable_debug_logging = st.checkbox(
            "🐛 Enable Debug Logging",
            value=log_config.get('enable_debug_logging', False),
            help="Enable detailed debug logging"
        )
        updated_log_config['enable_debug_logging'] = enable_debug_logging
        
        # Enable performance logging
        enable_performance_logging = st.checkbox(
            "⚡ Enable Performance Logging",
            value=log_config.get('enable_performance_logging', True),
            help="Log performance metrics and timing"
        )
        updated_log_config['enable_performance_logging'] = enable_performance_logging
        
        # Enable file logging
        enable_file_logging = st.checkbox(
            "📄 Enable File Logging",
            value=log_config.get('enable_file_logging', True),
            help="Write logs to files"
        )
        updated_log_config['enable_file_logging'] = enable_file_logging
        
        # Enable console logging
        enable_console_logging = st.checkbox(
            "🖥️ Enable Console Logging",
            value=log_config.get('enable_console_logging', True),
            help="Show logs in console/terminal"
        )
        updated_log_config['enable_console_logging'] = enable_console_logging
    
    # Current log directory status
    st.markdown("#### 📊 Current Log Status")
    current_log_path = Path(log_config.get('log_folder_path', 'logs'))
    
    if current_log_path.exists():
        log_files = list(current_log_path.rglob("*.log"))
        st.success(f"✅ Log directory exists: `{current_log_path}`")
        st.info(f"📄 Found {len(log_files)} log files")
        
        if log_files:
            # Show recent log files
            recent_files = sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            st.markdown("**Recent Log Files:**")
            for file in recent_files:
                file_size = file.stat().st_size
                mod_time = datetime.fromtimestamp(file.stat().st_mtime)
                st.write(f"- `{file.name}` ({file_size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        st.warning(f"⚠️ Log directory does not exist: `{current_log_path}`")
    
    # Save logging settings
    if st.button("💾 Save Logging Settings", key="save_log_settings"):
        app_config['logging'] = updated_log_config
        success = config_service.save_config('app', app_config)
        if success:
            st.success("✅ Logging settings saved successfully!")
            st.info("🔄 Restart the application to apply new logging settings")
            st.rerun()
        else:
            st.error("❌ Failed to save logging settings")


def _render_environment_variables_section() -> None:
    """Render environment variables management section."""
    st.markdown("### 🌐 Environment Variables")
    st.markdown("Manage system environment variables used by AI frameworks and libraries.")
    
    try:
        # Import config utilities
        from utils.config import Config
        
        # Get current environment variables
        current_env_vars = Config.get_environment_variables()
        
        # Create tabs for different categories
        tf_tab, pytorch_tab, hf_tab, python_tab, status_tab = st.tabs([
            "🧠 TensorFlow", 
            "🔥 PyTorch", 
            "🤗 HuggingFace", 
            "🐍 Python",
            "📊 Status"
        ])
        
        with tf_tab:
            _render_tensorflow_env_vars(current_env_vars)
        
        with pytorch_tab:
            _render_pytorch_env_vars(current_env_vars)
        
        with hf_tab:
            _render_huggingface_env_vars(current_env_vars)
        
        with python_tab:
            _render_python_env_vars(current_env_vars)
        
        with status_tab:
            _render_env_vars_status(current_env_vars)
            
    except Exception as e:
        st.error(f"❌ Error loading environment variables: {e}")


def _render_tensorflow_env_vars(current_env_vars: Dict[str, str]) -> None:
    """Render TensorFlow environment variables configuration."""
    st.markdown("#### 🧠 TensorFlow Configuration")
    st.markdown("Configure TensorFlow behavior and performance settings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TensorFlow logging level
        tf_log_level = st.selectbox(
            "🗣️ TensorFlow Log Level:",
            options=["0", "1", "2", "3"],
            index=["0", "1", "2", "3"].index(current_env_vars.get('TF_CPP_MIN_LOG_LEVEL', '2')),
            help="0=All, 1=Info+, 2=Warning+, 3=Error only",
            key="tf_log_level"
        )
        
        # OneDNN optimization
        onednn_enabled = st.checkbox(
            "⚡ Enable OneDNN Optimizations",
            value=current_env_vars.get('TF_ENABLE_ONEDNN_OPTS', '0') == '1',
            help="Enable Intel OneDNN optimizations for better CPU performance",
            key="onednn_enabled"
        )
        
        # TensorFlow 2.x behavior
        tf2_behavior = st.checkbox(
            "🔄 TensorFlow 2.x Behavior",
            value=current_env_vars.get('TF2_BEHAVIOR', '1') == '1',
            help="Enable TensorFlow 2.x behavior mode",
            key="tf2_behavior"
        )
    
    with col2:
        # CUDA visible devices
        cuda_devices = st.text_input(
            "🎮 CUDA Visible Devices:",
            value=current_env_vars.get('CUDA_VISIBLE_DEVICES', ''),
            help="Comma-separated GPU IDs (empty = all GPUs disabled)",
            placeholder="0,1 or leave empty",
            key="cuda_devices"
        )
        
        # Current values display
        st.markdown("**Current Values:**")
        st.code(f"""
TF_CPP_MIN_LOG_LEVEL = {current_env_vars.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')}
TF_ENABLE_ONEDNN_OPTS = {current_env_vars.get('TF_ENABLE_ONEDNN_OPTS', 'Not set')}
TF2_BEHAVIOR = {current_env_vars.get('TF2_BEHAVIOR', 'Not set')}
CUDA_VISIBLE_DEVICES = {current_env_vars.get('CUDA_VISIBLE_DEVICES', 'Not set')}
        """)
    
    # Update TensorFlow environment variables
    if st.button("💾 Update TensorFlow Settings", key="update_tf_env"):
        try:
            from utils.config import Config
            
            # Update environment variables
            Config.update_environment_variable('TF_CPP_MIN_LOG_LEVEL', tf_log_level)
            Config.update_environment_variable('TF_ENABLE_ONEDNN_OPTS', '1' if onednn_enabled else '0')
            Config.update_environment_variable('TF2_BEHAVIOR', '1' if tf2_behavior else '0')
            Config.update_environment_variable('CUDA_VISIBLE_DEVICES', cuda_devices)
            
            st.success("✅ TensorFlow environment variables updated!")
            st.info("🔄 Restart the application to apply changes to new processes")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to update TensorFlow settings: {e}")


def _render_pytorch_env_vars(current_env_vars: Dict[str, str]) -> None:
    """Render PyTorch environment variables configuration."""
    st.markdown("#### 🔥 PyTorch Configuration")
    st.markdown("Configure PyTorch behavior and CUDA settings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CUDA memory allocation
        cuda_alloc_conf = st.text_input(
            "💾 CUDA Allocation Config:",
            value=current_env_vars.get('PYTORCH_CUDA_ALLOC_CONF', ''),
            help="PyTorch CUDA memory allocation configuration",
            placeholder="expandable_segments:True",
            key="cuda_alloc_conf"
        )
        
        # Tokenizers parallelism
        tokenizers_parallel = st.selectbox(
            "🔄 Tokenizers Parallelism:",
            options=["true", "false"],
            index=["true", "false"].index(current_env_vars.get('TOKENIZERS_PARALLELISM', 'false')),
            help="Enable/disable tokenizers parallelism",
            key="tokenizers_parallel"
        )
    
    with col2:
        # Current values display
        st.markdown("**Current Values:**")
        st.code(f"""
PYTORCH_CUDA_ALLOC_CONF = {current_env_vars.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}
TOKENIZERS_PARALLELISM = {current_env_vars.get('TOKENIZERS_PARALLELISM', 'Not set')}
        """)
        
        # PyTorch info
        try:
            import torch
            st.markdown("**PyTorch Info:**")
            st.write(f"Version: {torch.__version__}")
            st.write(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"CUDA Version: {torch.version.cuda}")
                st.write(f"GPU Count: {torch.cuda.device_count()}")
        except ImportError:
            st.info("PyTorch not installed")
    
    # Update PyTorch environment variables
    if st.button("💾 Update PyTorch Settings", key="update_pytorch_env"):
        try:
            from utils.config import Config
            
            # Update environment variables
            Config.update_environment_variable('PYTORCH_CUDA_ALLOC_CONF', cuda_alloc_conf)
            Config.update_environment_variable('TOKENIZERS_PARALLELISM', tokenizers_parallel)
            
            st.success("✅ PyTorch environment variables updated!")
            st.info("🔄 Restart the application to apply changes to new processes")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to update PyTorch settings: {e}")


def _render_huggingface_env_vars(current_env_vars: Dict[str, str]) -> None:
    """Render HuggingFace environment variables configuration."""
    st.markdown("#### 🤗 HuggingFace Configuration")
    st.markdown("Configure HuggingFace cache directories and behavior.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # HuggingFace Hub cache
        hf_hub_cache = st.text_input(
            "🏠 HuggingFace Hub Cache:",
            value=current_env_vars.get('HUGGINGFACE_HUB_CACHE', ''),
            help="Directory for HuggingFace Hub downloads",
            placeholder="./output/cache/huggingface",
            key="hf_hub_cache"
        )
        
        # HuggingFace home
        hf_home = st.text_input(
            "🏡 HuggingFace Home:",
            value=current_env_vars.get('HF_HOME', ''),
            help="HuggingFace home directory",
            placeholder="./output/cache/huggingface",
            key="hf_home"
        )
        
        # Transformers cache
        transformers_cache = st.text_input(
            "🔄 Transformers Cache:",
            value=current_env_vars.get('TRANSFORMERS_CACHE', ''),
            help="Transformers library cache directory",
            placeholder="./output/cache/transformers",
            key="transformers_cache"
        )
    
    with col2:
        # Torch home
        torch_home = st.text_input(
            "🔥 Torch Home:",
            value=current_env_vars.get('TORCH_HOME', ''),
            help="PyTorch models cache directory",
            placeholder="./output/cache/torch",
            key="torch_home"
        )
        
        # Current values display
        st.markdown("**Current Values:**")
        st.code(f"""
HUGGINGFACE_HUB_CACHE = {current_env_vars.get('HUGGINGFACE_HUB_CACHE', 'Not set')}
HF_HOME = {current_env_vars.get('HF_HOME', 'Not set')}
TRANSFORMERS_CACHE = {current_env_vars.get('TRANSFORMERS_CACHE', 'Not set')}
TORCH_HOME = {current_env_vars.get('TORCH_HOME', 'Not set')}
        """)
    
    # Cache directory management
    st.markdown("#### 📁 Cache Directory Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Create Directories", key="create_cache_dirs"):
            try:
                from pathlib import Path
                directories = [hf_hub_cache, hf_home, transformers_cache, torch_home]
                created_count = 0
                
                for directory in directories:
                    if directory:
                        path = Path(directory)
                        if not path.exists():
                            path.mkdir(parents=True, exist_ok=True)
                            created_count += 1
                
                st.success(f"✅ Created {created_count} cache directories")
            except Exception as e:
                st.error(f"❌ Failed to create directories: {e}")
    
    with col2:
        if st.button("📊 Check Sizes", key="check_cache_sizes"):
            try:
                from pathlib import Path
                directories = [
                    ("HF Hub", hf_hub_cache),
                    ("HF Home", hf_home),
                    ("Transformers", transformers_cache),
                    ("Torch", torch_home)
                ]
                
                for name, directory in directories:
                    if directory and Path(directory).exists():
                        size = sum(f.stat().st_size for f in Path(directory).rglob('*') if f.is_file())
                        size_mb = size / (1024 * 1024)
                        st.write(f"{name}: {size_mb:.1f} MB")
                    else:
                        st.write(f"{name}: Directory not found")
                        
            except Exception as e:
                st.error(f"❌ Failed to check sizes: {e}")
    
    with col3:
        if st.button("🗑️ Clear Cache", key="clear_cache_dirs"):
            st.warning("⚠️ This will delete all cached models and data!")
            if st.button("⚠️ Confirm Clear", key="confirm_clear_cache"):
                try:
                    import shutil
                    from pathlib import Path
                    
                    directories = [hf_hub_cache, hf_home, transformers_cache, torch_home]
                    cleared_count = 0
                    
                    for directory in directories:
                        if directory and Path(directory).exists():
                            shutil.rmtree(directory)
                            Path(directory).mkdir(parents=True, exist_ok=True)
                            cleared_count += 1
                    
                    st.success(f"✅ Cleared {cleared_count} cache directories")
                except Exception as e:
                    st.error(f"❌ Failed to clear cache: {e}")
    
    # Update HuggingFace environment variables
    if st.button("💾 Update HuggingFace Settings", key="update_hf_env"):
        try:
            from utils.config import Config
            
            # Update environment variables
            Config.update_environment_variable('HUGGINGFACE_HUB_CACHE', hf_hub_cache)
            Config.update_environment_variable('HF_HOME', hf_home)
            Config.update_environment_variable('TRANSFORMERS_CACHE', transformers_cache)
            Config.update_environment_variable('TORCH_HOME', torch_home)
            
            st.success("✅ HuggingFace environment variables updated!")
            st.info("🔄 Restart the application to apply changes to new processes")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to update HuggingFace settings: {e}")


def _render_python_env_vars(current_env_vars: Dict[str, str]) -> None:
    """Render Python environment variables configuration."""
    st.markdown("#### 🐍 Python Configuration")
    st.markdown("Configure Python runtime behavior and warnings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Python warnings
        python_warnings = st.selectbox(
            "⚠️ Python Warnings:",
            options=["default", "ignore", "error", "always", "module", "once"],
            index=["default", "ignore", "error", "always", "module", "once"].index(
                current_env_vars.get('PYTHONWARNINGS', 'ignore').split(',')[0] if current_env_vars.get('PYTHONWARNINGS') else 0
            ),
            help="Control Python warning behavior",
            key="python_warnings"
        )
    
    with col2:
        # Current values display
        st.markdown("**Current Values:**")
        st.code(f"""
PYTHONWARNINGS = {current_env_vars.get('PYTHONWARNINGS', 'Not set')}
        """)
        
        # Python info
        import sys
        st.markdown("**Python Info:**")
        st.write(f"Version: {sys.version}")
        st.write(f"Executable: {sys.executable}")
    
    # Update Python environment variables
    if st.button("💾 Update Python Settings", key="update_python_env"):
        try:
            from utils.config import Config
            
            # Update environment variables
            Config.update_environment_variable('PYTHONWARNINGS', python_warnings)
            
            st.success("✅ Python environment variables updated!")
            st.info("🔄 Restart the application to apply changes to new processes")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to update Python settings: {e}")


def _render_env_vars_status(current_env_vars: Dict[str, str]) -> None:
    """Render environment variables status and management."""
    st.markdown("#### 📊 Environment Variables Status")
    
    # Environment variables summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Variables", len(current_env_vars))
    
    with col2:
        set_vars = sum(1 for v in current_env_vars.values() if v)
        st.metric("Set Variables", set_vars)
    
    with col3:
        unset_vars = sum(1 for v in current_env_vars.values() if not v)
        st.metric("Unset Variables", unset_vars)
    
    # Environment variables table
    st.markdown("#### 📋 All Environment Variables")
    
    # Create a dataframe for better display
    env_data = []
    for var_name, var_value in current_env_vars.items():
        status = "✅ Set" if var_value else "❌ Not Set"
        display_value = var_value if var_value else "Not set"
        # Truncate long values
        if len(display_value) > 50:
            display_value = display_value[:47] + "..."
        
        env_data.append({
            "Variable": var_name,
            "Status": status,
            "Value": display_value
        })
    
    # Display as table
    import pandas as pd
    df = pd.DataFrame(env_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Global actions
    st.markdown("#### ⚡ Global Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Status", key="refresh_env_status"):
            st.rerun()
    
    with col2:
        if st.button("📥 Load from Config", key="load_from_config"):
            try:
                from utils.config import Config
                Config.setup_environment_variables(force_update=True)
                st.success("✅ Environment variables loaded from configuration!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load from config: {e}")
    
    with col3:
        if st.button("💾 Save to Config", key="save_to_config"):
            try:
                # This would require implementing save functionality
                st.info("💡 Save to config functionality will be implemented in next version")
            except Exception as e:
                st.error(f"❌ Failed to save to config: {e}")
    
    with col4:
        if st.button("🔧 Reset to Defaults", key="reset_to_defaults"):
            try:
                from utils.config import Config
                Config._setup_default_environment_variables()
                st.success("✅ Environment variables reset to defaults!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to reset to defaults: {e}")
    
    # Export/Import functionality
    st.markdown("#### 📤 Export/Import")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Environment", key="export_env"):
            try:
                import json
                env_export = {
                    "environment_variables": current_env_vars,
                    "export_timestamp": datetime.now().isoformat(),
                    "export_source": "DurgasAI Settings"
                }
                
                json_str = json.dumps(env_export, indent=2)
                st.download_button(
                    label="💾 Download Environment Export",
                    data=json_str,
                    file_name="environment_variables.json",
                    mime="application/json",
                    key="download_env_export"
                )
            except Exception as e:
                st.error(f"❌ Failed to export environment: {e}")
    
    with col2:
        uploaded_file = st.file_uploader(
            "📥 Import Environment",
            type=["json"],
            help="Import environment variables from JSON file",
            key="import_env_file"
        )
        
        if uploaded_file is not None:
            try:
                import json
                env_data = json.load(uploaded_file)
                
                if "environment_variables" in env_data:
                    imported_vars = env_data["environment_variables"]
                    st.success(f"✅ Ready to import {len(imported_vars)} environment variables")
                    
                    if st.button("🔄 Apply Import", key="apply_env_import"):
                        from utils.config import Config
                        for var_name, var_value in imported_vars.items():
                            Config.update_environment_variable(var_name, var_value, save_to_config=False)
                        st.success("✅ Environment variables imported!")
                        st.rerun()
                else:
                    st.error("❌ Invalid file format")
            except Exception as e:
                st.error(f"❌ Failed to import environment: {e}")


