"""
Model Sharing Components for HuggingFace Hub Integration.

This module provides reusable UI components for model sharing functionality,
including repository configuration, model card creation, and sharing workflow management.
"""

import streamlit as st
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Import model manager for sharing functionality
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.model_manager import ModelManager
from utils.logger import debug, info, warning, error, log_user_action


class ModelSharingComponents:
    """
    Reusable UI components for model sharing functionality.
    
    This class provides modular components that can be used across different pages
    to implement HuggingFace model sharing capabilities.
    """
    
    def __init__(self, model_manager: ModelManager = None):
        """
        Initialize the model sharing components.
        
        Args:
            model_manager (ModelManager): Instance of ModelManager for sharing operations
        """
        self.model_manager = model_manager or ModelManager()
        debug("ModelSharingComponents initialized", "sharing")
    
    def render_sharing_configuration(self, selected_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the sharing configuration section.
        
        Args:
            selected_model (Dict[str, Any]): Selected model information
            
        Returns:
            Dict[str, Any]: Sharing configuration
        """
        st.markdown("### 🔧 Sharing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Repository Settings")
            
            # Username input
            username = st.text_input(
                "HuggingFace Username:",
                value=st.session_state.get('hf_username', ''),
                help="Your HuggingFace username",
                key="sharing_username"
            )
            
            # Model name input
            model_name = st.text_input(
                "Model Name:",
                value=st.session_state.get('model_name', selected_model['name'].split('/')[-1]),
                help="Name for your shared model",
                key="sharing_model_name"
            )
            
            # Repository ID (auto-generated)
            repo_id = f"{username}/{model_name}" if username and model_name else ""
            st.text_input(
                "Repository ID:",
                value=repo_id,
                help="Full repository ID (username/model-name)",
                disabled=True,
                key="sharing_repo_id"
            )
        
        with col2:
            st.markdown("#### ⚙️ Sharing Options")
            
            # Privacy options
            is_private = st.checkbox(
                "Private Repository",
                value=st.session_state.get('sharing_private', False),
                help="Make the repository private",
                key="sharing_private"
            )
            
            is_gated = st.checkbox(
                "Gated Model",
                value=st.session_state.get('sharing_gated', False),
                help="Require approval to access the model",
                key="sharing_gated"
            )
            
            # License selection
            license_type = st.selectbox(
                "License:",
                options=["apache-2.0", "mit", "gpl-3.0", "bsd-3-clause", "other"],
                index=0,
                help="Choose a license for your model",
                key="sharing_license"
            )
        
        # Store configuration in session state
        config = {
            "username": username,
            "model_name": model_name,
            "repo_id": repo_id,
            "private": is_private,
            "gated": is_gated,
            "license": license_type,
            "base_model": selected_model['name'],
            "pipeline": selected_model['pipeline_tag']
        }
        
        st.session_state.sharing_config = config
        
        return config
    
    def render_model_card_creation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the model card creation section.
        
        Args:
            config (Dict[str, Any]): Sharing configuration
            
        Returns:
            Dict[str, Any]: Model card configuration
        """
        st.markdown("### 📝 Model Card Creation")
        st.markdown("Create a comprehensive model card for your shared model.")
        
        # Model description
        model_description = st.text_area(
            "Model Description:",
            value=st.session_state.get('model_description', 
                f"This is a fine-tuned version of {config['base_model']} for improved performance."),
            height=100,
            help="Describe what your model does and how it's different",
            key="model_card_description"
        )
        
        # Performance metrics
        st.markdown("#### 📊 Performance Metrics")
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            accuracy = st.number_input(
                "Accuracy:", 
                value=st.session_state.get('model_accuracy', 0.95),
                min_value=0.0, 
                max_value=1.0, 
                step=0.01,
                key="model_card_accuracy"
            )
            f1_score = st.number_input(
                "F1 Score:", 
                value=st.session_state.get('model_f1_score', 0.94),
                min_value=0.0, 
                max_value=1.0, 
                step=0.01,
                key="model_card_f1"
            )
        
        with perf_col2:
            inference_time = st.number_input(
                "Inference Time (ms):", 
                value=st.session_state.get('model_inference_time', 2.3),
                min_value=0.0, 
                step=0.1,
                key="model_card_inference"
            )
            memory_usage = st.number_input(
                "Memory Usage (GB):", 
                value=st.session_state.get('model_memory_usage', 4.2),
                min_value=0.0, 
                step=0.1,
                key="model_card_memory"
            )
        
        # Usage example
        usage_example = st.text_area(
            "Usage Example:",
            value=st.session_state.get('model_usage_example', 
                f"""from transformers import pipeline

generator = pipeline('text-generation', model='{config['repo_id']}')
result = generator("Hello, I am")"""),
            height=100,
            help="Python code example showing how to use your model",
            key="model_card_usage"
        )
        
        # Limitations
        limitations = st.text_area(
            "Limitations and Bias:",
            value=st.session_state.get('model_limitations', 
                "- The model may exhibit bias present in the training data\n- Performance may vary across different domains\n- Limited to English language processing"),
            height=80,
            help="Describe any limitations or biases of your model",
            key="model_card_limitations"
        )
        
        # Store model card configuration
        model_card_config = {
            "description": model_description,
            "performance_metrics": {
                "Accuracy": f"{accuracy:.3f}",
                "F1 Score": f"{f1_score:.3f}",
                "Inference Time": f"{inference_time}ms",
                "Memory Usage": f"{memory_usage}GB"
            },
            "usage_example": usage_example,
            "limitations": limitations
        }
        
        st.session_state.model_card_config = model_card_config
        
        return model_card_config
    
    def render_sharing_manager_initialization(self, api_key: str) -> bool:
        """
        Render the sharing manager initialization section.
        
        Args:
            api_key (str): HuggingFace API key
            
        Returns:
            bool: True if sharing manager is initialized successfully
        """
        st.markdown("### 🔐 Sharing Manager Setup")
        
        if st.button("🔐 Initialize Sharing Manager", type="primary", use_container_width=True):
            if api_key:
                with st.spinner("Initializing HuggingFace sharing manager..."):
                    success = self.model_manager.initialize_sharing_manager(api_key)
                    
                    if success:
                        st.session_state.sharing_initialized = True
                        st.success("✅ Sharing manager initialized successfully!")
                        
                        # Show sharing status
                        status = self.model_manager.get_sharing_status()
                        if status.get('current_user'):
                            st.info(f"📝 Authenticated as: {status['current_user']}")
                            log_user_action("sharing_manager_initialized", 
                                          user=status['current_user'],
                                          success=True)
                        return True
                    else:
                        st.error("❌ Failed to initialize sharing manager. Please check your API key.")
                        log_user_action("sharing_manager_initialized", 
                                      success=False,
                                      error="initialization_failed")
                        return False
            else:
                st.error("❌ Please configure your HuggingFace API key first.")
                return False
        
        return st.session_state.get('sharing_initialized', False)
    
    def render_model_sharing_workflow(self, config: Dict[str, Any], model_card_config: Dict[str, Any]) -> bool:
        """
        Render the model sharing workflow section.
        
        Args:
            config (Dict[str, Any]): Sharing configuration
            model_card_config (Dict[str, Any]): Model card configuration
            
        Returns:
            bool: True if sharing was attempted
        """
        if not st.session_state.get('sharing_initialized', False):
            st.warning("⚠️ Please initialize the sharing manager first.")
            return False
        
        st.markdown("### 🚀 Share Your Model")
        
        # Show current configuration
        st.markdown("#### 📋 Current Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write(f"**Repository:** `{config['repo_id']}`")
            st.write(f"**Private:** {config['private']}")
            st.write(f"**Gated:** {config['gated']}")
        
        with config_col2:
            st.write(f"**License:** {config['license']}")
            st.write(f"**Base Model:** {config['base_model']}")
            st.write(f"**Pipeline:** {config['pipeline']}")
        
        # Share model button
        if st.button("📤 Share Model to HuggingFace Hub", type="primary", use_container_width=True):
            if config['repo_id'] and config['username'] and config['model_name']:
                with st.spinner("Creating model card and sharing model..."):
                    try:
                        # Create model card
                        model_card = self.model_manager.create_model_card(
                            model_name=config['model_name'],
                            description=model_card_config['description'],
                            performance_metrics=model_card_config['performance_metrics'],
                            usage_example=model_card_config['usage_example'],
                            limitations=model_card_config['limitations'],
                            license=config['license']
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
                        hub_url = f"https://huggingface.co/{config['repo_id']}"
                        st.markdown(f"🌐 **Model Hub URL:** [{hub_url}]({hub_url})")
                        
                        # Log the sharing event
                        log_user_action("model_sharing_attempted", 
                                      repo_id=config['repo_id'],
                                      private=config['private'],
                                      gated=config['gated'],
                                      license=config['license'],
                                      success=True)
                        
                        return True
                        
                    except Exception as e:
                        error_msg = f"Error in sharing workflow: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        error(error_msg, "sharing", e)
                        
                        log_user_action("model_sharing_attempted", 
                                      repo_id=config['repo_id'],
                                      success=False,
                                      error=error_msg)
                        return False
            else:
                st.error("❌ Please fill in all required fields (username, model name).")
                return False
        
        return False
    
    def render_sharing_guide(self):
        """Render the sharing guide section."""
        with st.expander("📚 Model Sharing Guide", expanded=False):
            st.markdown("""
            ### 🤗 How to Share Models with HuggingFace Hub
            
            **Step 1: Prepare Your Model**
            - Train or fine-tune your model
            - Save model files (pytorch_model.bin, config.json, etc.)
            - Test your model thoroughly
            
            **Step 2: Configure Repository**
            - Enter your HuggingFace username
            - Choose a descriptive model name
            - Set privacy and gating options
            - Select appropriate license
            
            **Step 3: Create Model Card**
            - Write a clear description
            - Add performance metrics
            - Include usage examples
            - Document limitations and biases
            
            **Step 4: Initialize Sharing Manager**
            - Click "Initialize Sharing Manager"
            - Verify authentication
            - Check sharing status
            
            **Step 5: Share Your Model**
            - Review configuration
            - Click "Share Model to HuggingFace Hub"
            - Wait for upload completion
            - Access your model on the Hub
            
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
            - [License Information](https://huggingface.co/docs/hub/repositories-licenses)
            """)
    
    def render_sharing_status(self) -> Dict[str, Any]:
        """
        Render the sharing status section.
        
        Returns:
            Dict[str, Any]: Current sharing status
        """
        st.markdown("### 📊 Sharing Status")
        
        if not self.model_manager.sharing_enabled:
            st.error("❌ Model sharing not available - missing dependencies")
            return {"error": "Sharing not available"}
        
        status = self.model_manager.get_sharing_status()
        
        # Display status information
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sharing Enabled", "✅ Yes" if status.get('sharing_enabled') else "❌ No")
            st.metric("API Available", "✅ Yes" if status.get('hf_api_available') else "❌ No")
        
        with col2:
            st.metric("User Authenticated", "✅ Yes" if status.get('user_authenticated') else "❌ No")
            if status.get('current_user'):
                st.metric("Current User", status['current_user'])
        
        # Show detailed status
        with st.expander("🔍 Detailed Status", expanded=False):
            st.json(status)
        
        return status
    
    def render_complete_sharing_workflow(self, selected_model: Dict[str, Any], api_key: str) -> bool:
        """
        Render the complete model sharing workflow.
        
        Args:
            selected_model (Dict[str, Any]): Selected model information
            api_key (str): HuggingFace API key
            
        Returns:
            bool: True if sharing was successful
        """
        st.markdown("### 📤 Complete Model Sharing Workflow")
        
        # Step 1: Sharing Configuration
        with st.expander("Step 1: Configure Repository", expanded=True):
            config = self.render_sharing_configuration(selected_model)
        
        # Step 2: Model Card Creation
        with st.expander("Step 2: Create Model Card", expanded=False):
            model_card_config = self.render_model_card_creation(config)
        
        # Step 3: Initialize Sharing Manager
        with st.expander("Step 3: Initialize Sharing Manager", expanded=False):
            sharing_initialized = self.render_sharing_manager_initialization(api_key)
        
        # Step 4: Share Model (only if initialized)
        if sharing_initialized:
            with st.expander("Step 4: Share Model", expanded=True):
                return self.render_model_sharing_workflow(config, model_card_config)
        
        # Step 5: Sharing Guide
        with st.expander("Step 5: Sharing Guide", expanded=False):
            self.render_sharing_guide()
        
        return False


def create_sharing_components(model_manager: ModelManager = None) -> ModelSharingComponents:
    """
    Factory function to create ModelSharingComponents instance.
    
    Args:
        model_manager (ModelManager): Optional ModelManager instance
        
    Returns:
        ModelSharingComponents: Configured sharing components instance
    """
    return ModelSharingComponents(model_manager)
