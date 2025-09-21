"""
Model Selector Component.

This module provides the model selection interface for DurgasAI, allowing users
to choose and configure AI models for their conversations.

Key Features:
- Interactive model selection dropdown
- API token configuration and validation
- Model information display with status
- System prompt selection and customization
- Real-time model status updates
- Integration with application state

Architecture:
- Integrates with Config system for available models
- Uses ApplicationState for persistent configuration
- Provides validation and error handling
- Supports both API and local model types
- Logs all user interactions for debugging

Components:
- Model selection dropdown with descriptions
- API token input with validation
- System prompt selector with presets
- Model status display with health indicators

Dependencies:
- utils.config: Model configuration and provider definitions
- utils.logger: Centralized logging system
- core.application_state: Global state management
"""

import streamlit as st
from typing import Dict, Any
from pathlib import Path
import sys

# Add utils to path for proper module resolution
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import Config, ModelProvider
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation
from core.application_state import ApplicationState


class ModelSelector:
    """Component for model selection and configuration."""
    
    def __init__(self):
        """Initialize the model selector."""
        debug("ModelSelector initialized", "model_selector")
    
    def render(self) -> Dict[str, Any]:
        """
        Render model selection interface with comprehensive logging and error handling.
        
        This method creates the complete model selection interface including API token
        input, model selection dropdown, model information display, and system prompt
        configuration. It includes detailed logging for debugging and analytics.
        
        Returns:
            Dict containing selected model configuration and user inputs
            
        Features:
        - API token input and validation
        - Model selection with descriptions
        - Model information display
        - System prompt configuration
        - Error handling and user feedback
        - User interaction tracking
        
        Logging:
        - API token configuration status
        - Model selection events
        - Configuration validation
        - Error handling for invalid inputs
        - User interaction analytics
        """
        debug("Starting model selection interface rendering", "model_selector")
        
        try:
            with LoggedOperation("model_selection_rendering", "model_selector"):
                # API Token input section with comprehensive logging
                debug("Rendering API token input section", "model_selector")
                
                current_api_token = ApplicationState.get("api_token", "")
                api_token_configured = bool(current_api_token)
                
                debug("Current API token status", "model_selector",
                      has_token=api_token_configured,
                      token_length=len(current_api_token) if current_api_token else 0)
                
                api_token = st.text_input(
                    "🔑 HuggingFace API Token:",
                    type="password",
                    value=current_api_token,
                    help="Get your token from https://huggingface.co/settings/tokens",
                    key="api_token_input"
                )
                
                debug("API token input field rendered", "model_selector",
                      token_provided=bool(api_token),
                      token_changed=api_token != current_api_token)
                
                # Handle API token changes
                if api_token and api_token != current_api_token:
                    debug("API token updated", "model_selector",
                          old_token_length=len(current_api_token),
                          new_token_length=len(api_token))
                    
                    ApplicationState.set("api_token", api_token)
                    st.success("✅ API token configured")
                    
                    # Log token configuration
                    log_user_action("api_token_configured",
                                  token_length=len(api_token),
                                  token_configured=True)
                    
                elif api_token:
                    debug("API token confirmed (no change)", "model_selector")
                    st.success("✅ API token configured")
                    
                else:
                    debug("No API token provided", "model_selector")
                    st.warning("⚠️ API token required for most models")
                    log_user_action("api_token_missing")
                
                st.markdown("---")
                debug("API token section completed", "model_selector")
                
                # Model selection section
                debug("Rendering model selection section", "model_selector")
                st.markdown("### 🎯 Model Selection")
                
                # Get available models with validation
                try:
                    model_names = list(Config.AVAILABLE_MODELS.keys())
                    model_display_names = [Config.AVAILABLE_MODELS[name].name for name in model_names]
                    
                    debug("Available models retrieved", "model_selector",
                          model_count=len(model_names),
                          model_names=model_names)
                    
                    if not model_names:
                        error("No models available in configuration", "model_selector")
                        st.error("🚨 No models available. Please check configuration.")
                        return {}
                    
                except Exception as e:
                    error("Error retrieving available models", "model_selector", e)
                    st.error("🚨 Error loading available models. Please try again.")
                    return {}
                
                # Model selection dropdown
                try:
                    selected_model_display = st.selectbox(
                        "Choose AI Model:",
                        model_display_names,
                        index=0,
                        help="Select the AI model for conversation",
                        key="model_selection"
                    )
                    
                    debug("Model selection dropdown rendered", "model_selector",
                          selected_display=selected_model_display,
                          available_count=len(model_display_names))
                    
                except Exception as e:
                    error("Error rendering model selection dropdown", "model_selector", e)
                    st.error("🚨 Error displaying model selection. Please try again.")
                    return {}
                
                # Get selected model configuration
                try:
                    selected_model = model_names[model_display_names.index(selected_model_display)]
                    model_config = Config.get_model_config(selected_model)
                    
                    debug("Selected model configuration retrieved", "model_selector",
                          selected_model=selected_model,
                          model_display_name=selected_model_display,
                          config_available=bool(model_config))
                    
                    if not model_config:
                        warning("No configuration found for selected model", "model_selector",
                               selected_model=selected_model)
                        st.warning(f"⚠️ No configuration found for {selected_model}")
                        return {}
                    
                except Exception as e:
                    error("Error retrieving model configuration", "model_selector", e,
                          selected_display=selected_model_display,
                          available_models=model_names)
                    st.error("🚨 Error loading model configuration. Please try again.")
                    return {}
                
                # Display model information
                debug("Rendering model information display", "model_selector")
                self._display_model_info(model_config)
                
                # Log model selection for analytics
                log_user_action("model_selected",
                              selected_model=selected_model,
                              model_display_name=selected_model_display,
                              provider=model_config.provider.value if hasattr(model_config, 'provider') else 'unknown',
                              api_token_configured=bool(api_token))
                
                # System prompt selection
                debug("Rendering system prompt selector", "model_selector")
                system_prompt = self._render_system_prompt_selector()
                
                debug("System prompt selector completed", "model_selector",
                      prompt_length=len(system_prompt) if system_prompt else 0)
                
                # Prepare return configuration
                configuration = {
                    'selected_model': selected_model,
                    'model_config': model_config,
                    'model_name': model_config.name if model_config else selected_model,
                    'api_token': api_token,
                    'system_prompt': system_prompt,
                    'requires_api_token': model_config.provider == ModelProvider.HUGGINGFACE_API if model_config else True
                }
                
                debug("Model selection configuration prepared", "model_selector",
                      selected_model=configuration['selected_model'],
                      model_name=configuration['model_name'],
                      requires_api_token=configuration['requires_api_token'],
                      api_token_configured=bool(configuration['api_token']),
                      system_prompt_length=len(configuration['system_prompt']) if configuration['system_prompt'] else 0)
                
                return configuration
                
        except Exception as e:
            error("Critical error in model selection rendering", "model_selector", e)
            st.error("🚨 An error occurred while loading model selection. Please try again.")
            log_user_action("model_selection_failed", error=str(e))
            return {}
        
        debug("Model selection interface rendering completed", "model_selector")
    
    def _display_model_info(self, model_config) -> None:
        """Display model information."""
        status = "ready" if ApplicationState.is_model_loaded() else "not_loaded"
        
        status_colors = {
            "ready": "🟢",
            "not_loaded": "🔴"
        }
        
        st.markdown(f"""
        **Model Info:**
        - {status_colors.get(status, '🔴')} **Status**: {status.replace('_', ' ').title()}
        - **Provider**: {model_config.provider.value}
        - **Description**: {model_config.description}
        """)
    
    def _render_system_prompt_selector(self) -> str:
        """Render system prompt selection interface."""
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
                value=ApplicationState.get("custom_system_prompt", ""),
                height=100,
                help="Define how the AI should behave"
            )
            ApplicationState.set("custom_system_prompt", system_prompt)
        else:
            prompt_key = prompt_options[prompt_display_names.index(selected_prompt_display)]
            system_prompt = Config.DEFAULT_SYSTEM_PROMPTS[prompt_key]
            
            with st.expander("📖 View System Prompt"):
                st.write(system_prompt)
        
        return system_prompt
