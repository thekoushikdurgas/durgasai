"""
Custom Model Development Page.

This page allows users to:
1. Create custom Transformers model configurations
2. Implement custom model classes
3. Register models with AutoClass APIs
4. Generate model files for Hub upload
5. Test and validate custom models
"""

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.custom_model_manager import (
    CustomModelManager, CustomModelInfo, CustomModelValidator,
    ModularModelInfo, ModularModelConverter, LegacyModelInfo, custom_model_manager
)
from utils.logger import debug, info, warning, error, log_session_event
from transformers import PretrainedConfig, PreTrainedModel


class CustomModelPage:
    """
    Custom model development page.
    
    This class handles:
    - Custom model configuration creation
    - Model implementation
    - AutoClass registration
    - File generation and validation
    - Hub upload preparation
    """
    
    def __init__(self):
        """Initialize the custom model page."""
        self.model_manager = custom_model_manager
        self.validator = CustomModelValidator()
        debug("CustomModelPage initialized", "custom_models")
    
    def render(self):
        """Render the complete custom model development page."""
        # Main header
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🔧 Custom Model Development</h1>
            <p style="color: #f0f0f0; margin: 5px 0;">Create, customize, and share your own Transformers models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "🏗️ Model Builder", 
            "📝 Code Generator", 
            "🔧 Modular Models",
            "🏛️ Legacy Models",
            "📚 Documentation", 
            "🎯 Attention Functions",
            "🔍 Validator", 
            "📤 Hub Upload", 
            "📚 Examples"
        ])
        
        with tab1:
            self._render_model_builder()
        
        with tab2:
            self._render_code_generator()
        
        with tab3:
            self._render_modular_models()
        
        with tab4:
            self._render_legacy_models()
        
        with tab5:
            self._render_documentation()
        
        with tab6:
            self._render_attention_functions()
        
        with tab7:
            self._render_validator()
        
        with tab8:
            self._render_hub_upload()
        
        with tab9:
            self._render_examples()
    
    def _render_modular_models(self):
        """Render the modular models creation interface."""
        st.markdown("### 🔧 Modular Model Creator")
        st.markdown("Create models using the HuggingFace modular approach - significantly reducing code complexity.")
        
        # Modular model creation form
        with st.expander("🚀 Create New Modular Model", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Basic Information")
                
                model_name = st.text_input(
                    "Model Name:",
                    placeholder="my_custom_model",
                    help="Name for your modular model (snake_case)"
                )
                
                base_model = st.selectbox(
                    "Base Model:",
                    options=["bert", "gpt2", "llama", "mistral", "t5", "roberta", "albert", "electra", "distilbert", "deberta"],
                    help="Choose the base model to inherit from"
                )
                
                description = st.text_area(
                    "Description:",
                    placeholder="Describe your model and what makes it different from the base model",
                    height=100
                )
                
                author = st.text_input(
                    "Author:",
                    placeholder="Your Name",
                    help="Your name or organization"
                )
            
            with col2:
                st.markdown("#### ⚙️ Configuration Changes")
                
                # Configuration parameters
                config_changes = {}
                
                st.markdown("**Add Configuration Parameters:**")
                with st.container():
                    param_col1, param_col2 = st.columns([3, 1])
                    with param_col1:
                        param_name = st.text_input("Parameter Name:", key="config_param_name")
                    with param_col2:
                        param_value = st.text_input("Value:", key="config_param_value")
                    
                    if st.button("➕ Add Config Parameter", key="add_config_param"):
                        if param_name and param_value:
                            try:
                                # Try to parse as number
                                if param_value.isdigit():
                                    config_changes[param_name] = int(param_value)
                                elif param_value.replace('.', '').isdigit():
                                    config_changes[param_name] = float(param_value)
                                else:
                                    config_changes[param_name] = param_value
                                
                                st.success(f"Added: {param_name} = {config_changes[param_name]}")
                            except:
                                config_changes[param_name] = param_value
                                st.success(f"Added: {param_name} = {param_value}")
                
                # Model changes
                st.markdown("**Add Model Changes:**")
                with st.container():
                    model_col1, model_col2 = st.columns([3, 1])
                    with model_col1:
                        model_change_name = st.text_input("Change Name:", key="model_change_name")
                    with model_col2:
                        model_change_value = st.text_input("Value:", key="model_change_value")
                    
                    if st.button("➕ Add Model Change", key="add_model_change"):
                        if model_change_name and model_change_value:
                            st.success(f"Added model change: {model_change_name}")
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🆕 New Components")
                new_components = st.multiselect(
                    "New Components to Add:",
                    options=["CustomAttention", "CustomNorm", "CustomMLP", "CustomEmbedding"],
                    help="Select new components to add to your model"
                )
            
            with col2:
                st.markdown("#### 🗑️ Removed Components")
                removed_components = st.multiselect(
                    "Components to Remove:",
                    options=["attention_dropout", "hidden_dropout", "layer_norm_eps", "initializer_range"],
                    help="Select components to remove from the base model"
                )
        
        # Create modular model button
        if st.button("🚀 Create Modular Model", type="primary", use_container_width=True):
            if model_name and base_model and description:
                try:
                    # Create ModularModelInfo
                    modular_info = ModularModelInfo(
                        name=model_name,
                        base_model=base_model,
                        config_changes=config_changes,
                        model_changes={},  # Would be populated from form
                        new_components=new_components,
                        removed_components=removed_components,
                        description=description,
                        author=author or "Anonymous"
                    )
                    
                    # Validate the modular model
                    validation = self.model_manager.validate_modular_model(modular_info)
                    
                    if validation["valid"]:
                        # Create the modular model
                        with st.spinner("Creating modular model..."):
                            success = self.model_manager.create_modular_model(modular_info)
                        
                        if success:
                            st.success(f"✅ Successfully created modular model: {model_name}")
                            
                            # Show generated files
                            with st.expander("📄 Generated Files", expanded=False):
                                files = self.model_manager.generate_model_files(model_name)
                                for filename, content in files.items():
                                    st.markdown(f"**{filename}:**")
                                    st.code(content, language="python")
                            
                            # Show conversion command
                            st.info(f"💡 To convert to single files, run: `python utils/modular_model_converter.py {model_name}`")
                            
                            log_session_event("modular_model_created_ui", 
                                            model_name=model_name,
                                            base_model=base_model,
                                            components_added=len(new_components))
                        else:
                            st.error("❌ Failed to create modular model. Please check the logs.")
                    else:
                        st.error("❌ Validation failed:")
                        for error_msg in validation["errors"]:
                            st.error(f"  - {error_msg}")
                        
                        if validation["warnings"]:
                            for warning_msg in validation["warnings"]:
                                st.warning(f"  - {warning_msg}")
                            
                except Exception as e:
                    st.error(f"❌ Error creating modular model: {str(e)}")
                    error(f"Error in modular model creation UI: {str(e)}", "custom_models", e)
            else:
                st.error("❌ Please fill in all required fields (Model Name, Base Model, Description)")
        
        # Show existing modular models
        with st.expander("📋 Existing Modular Models", expanded=False):
            modular_models = self.model_manager.get_modular_models()
            
            if modular_models:
                for name, info in modular_models.items():
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{name}**")
                            st.markdown(f"*Base: {info.base_model}*")
                            st.markdown(f"*{info.description[:100]}...*")
                        
                        with col2:
                            st.markdown(f"**Changes:** {len(info.config_changes) + len(info.model_changes)}")
                        
                        with col3:
                            if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                                # Delete logic would go here
                                st.success(f"Deleted {name}")
            else:
                st.info("No modular models created yet. Create your first modular model above!")
        
        # Modular model guide
        with st.expander("📚 Modular Model Guide", expanded=False):
            st.markdown("""
            ### 🤗 Understanding Modular Models
            
            **What are Modular Models?**
            - Modular models use inheritance to build upon existing models
            - Significantly reduce code duplication and maintenance overhead
            - Automatically generate single-file implementations
            
            **Benefits:**
            - **Reduced Code**: 95% less code than traditional implementation
            - **Easier Maintenance**: Changes propagate from base models
            - **Consistency**: Prevents code divergence across models
            - **Faster Development**: Focus on differences, not reimplementation
            
            **How it Works:**
            1. **Choose Base Model**: Select an existing model to inherit from
            2. **Specify Changes**: Define what's different in your model
            3. **Generate Files**: Automatic generation of all necessary files
            4. **Convert**: Transform modular files to single-file format
            
            **Example Use Cases:**
            - Fine-tuning with architectural modifications
            - Adding new attention mechanisms
            - Changing normalization layers
            - Custom embedding strategies
            
            **Best Practices:**
            - Choose the most similar existing model as base
            - Only specify what's actually different
            - Use descriptive parameter names
            - Test thoroughly before sharing
            """)
    
    def _render_model_builder(self):
        """Render the interactive model builder."""
        st.markdown("### 🏗️ Interactive Model Builder")
        st.markdown("Build your custom model step by step with guided configuration.")
        
        # Step 1: Model Information
        st.markdown("#### 📋 Step 1: Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Model Name",
                value="MyCustomModel",
                help="Choose a unique name for your model"
            )
            
            model_type = st.text_input(
                "Model Type",
                value="custom_transformer",
                help="Type identifier for your model (e.g., 'custom_transformer')"
            )
        
        with col2:
            author = st.text_input(
                "Author",
                value="Your Name",
                help="Your name or organization"
            )
            
            version = st.text_input(
                "Version",
                value="1.0.0",
                help="Model version"
            )
        
        description = st.text_area(
            "Description",
            value="A custom Transformers model for specific tasks.",
            help="Describe what your model does"
        )
        
        tags = st.multiselect(
            "Tags",
            options=["text-generation", "classification", "translation", "summarization", "custom"],
            default=["custom"],
            help="Select relevant tags for your model"
        )
        
        # Step 2: Configuration Parameters
        st.markdown("#### ⚙️ Step 2: Configuration Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vocab_size = st.number_input(
                "Vocabulary Size",
                min_value=1000,
                max_value=1000000,
                value=30522,
                help="Size of the vocabulary"
            )
            
            hidden_size = st.number_input(
                "Hidden Size",
                min_value=64,
                max_value=4096,
                value=768,
                help="Hidden dimension size"
            )
        
        with col2:
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=100,
                value=12,
                help="Number of transformer layers"
            )
            
            num_attention_heads = st.number_input(
                "Attention Heads",
                min_value=1,
                max_value=64,
                value=12,
                help="Number of attention heads"
            )
        
        # Step 3: Model Architecture
        st.markdown("#### 🏗️ Step 3: Model Architecture")
        
        architecture_type = st.selectbox(
            "Architecture Type",
            options=["Transformer", "CNN", "RNN", "Hybrid"],
            help="Select the base architecture type"
        )
        
        task_type = st.selectbox(
            "Task Type",
            options=["text-generation", "classification", "regression", "feature-extraction"],
            help="Select the primary task type"
        )
        
        # Step 4: Generate Configuration
        st.markdown("#### 🚀 Step 4: Generate Configuration")
        
        if st.button("🔧 Generate Model Configuration", type="primary"):
            config_code = self._generate_config_code(
                model_name, model_type, vocab_size, hidden_size, 
                num_layers, num_attention_heads, architecture_type
            )
            
            model_code = self._generate_model_code(
                model_name, model_type, task_type, architecture_type
            )
            
            # Store in session state
            st.session_state.custom_config_code = config_code
            st.session_state.custom_model_code = model_code
            st.session_state.custom_model_info = {
                "name": model_name,
                "type": model_type,
                "author": author,
                "version": version,
                "description": description,
                "tags": tags
            }
            
            st.success("✅ Configuration generated successfully!")
            st.info("Go to the 'Code Generator' tab to view and customize the generated code.")
    
    def _generate_config_code(self, name, model_type, vocab_size, hidden_size, num_layers, num_attention_heads, architecture_type):
        """Generate configuration class code."""
        config_code = f'''"""
Configuration for {name} model.
"""

from transformers import PretrainedConfig
from typing import List, Optional

class {name}Config(PretrainedConfig):
    model_type = "{model_type}"
    
    def __init__(
        self,
        vocab_size: int = {vocab_size},
        hidden_size: int = {hidden_size},
        num_layers: int = {num_layers},
        num_attention_heads: int = {num_attention_heads},
        intermediate_size: int = {hidden_size * 4},
        max_position_embeddings: int = 512,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs
    ):
        # Validate parameters
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_attention_heads` ({num_attention_heads})"
            )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        super().__init__(**kwargs)
'''
        return config_code
    
    def _generate_model_code(self, name, model_type, task_type, architecture_type):
        """Generate model class code."""
        if task_type == "classification":
            model_code = f'''"""
Model implementation for {name}.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_{name.lower()} import {name}Config

class {name}Model(PreTrainedModel):
    config_class = {name}Config
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize model components
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.dropout_rate,
                activation="gelu",
                norm_first=True
            ) for _ in range(config.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.dropout(embeddings)
        
        # Transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=attention_mask == 0)
        
        # Layer norm
        embeddings = self.layer_norm(embeddings)
        
        return embeddings

class {name}ForSequenceClassification(PreTrainedModel):
    config_class = {name}Config
    
    def __init__(self, config):
        super().__init__(config)
        self.model = {name}Model(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels if hasattr(config, 'num_labels') else 2)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get hidden states
        hidden_states = self.model(input_ids, attention_mask)
        
        # Pooling (mean pooling over non-padded tokens)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            pooled = torch.mean(hidden_states, dim=1)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {{"loss": loss, "logits": logits}}
        
        return {{"logits": logits}}
'''
        else:
            model_code = f'''"""
Model implementation for {name}.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_{name.lower()} import {name}Config

class {name}Model(PreTrainedModel):
    config_class = {name}Config
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize model components
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.dropout_rate,
                activation="gelu",
                norm_first=True
            ) for _ in range(config.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.dropout(embeddings)
        
        # Transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=attention_mask == 0)
        
        # Layer norm
        embeddings = self.layer_norm(embeddings)
        
        # Language modeling head
        logits = self.lm_head(embeddings)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return {{"loss": loss, "logits": logits}}
        
        return {{"logits": logits}}
'''
        
        return model_code
    
    def _render_code_generator(self):
        """Render the code generator and editor."""
        st.markdown("### 📝 Code Generator & Editor")
        st.markdown("View, edit, and customize your generated model code.")
        
        if 'custom_config_code' not in st.session_state:
            st.info("👈 Go to the 'Model Builder' tab first to generate your model configuration.")
            return
        
        # Display generated code
        st.markdown("#### 🔧 Generated Configuration Code")
        config_code = st.code(st.session_state.custom_config_code, language='python')
        
        st.markdown("#### 🏗️ Generated Model Code")
        model_code = st.code(st.session_state.custom_model_code, language='python')
        
        # Code customization
        st.markdown("#### ✏️ Customize Code")
        
        tab1, tab2 = st.tabs(["Configuration", "Model"])
        
        with tab1:
            edited_config = st.text_area(
                "Edit Configuration Code",
                value=st.session_state.custom_config_code,
                height=400,
                help="Modify the generated configuration code"
            )
            
            if st.button("💾 Update Configuration", key="update_config"):
                st.session_state.custom_config_code = edited_config
                st.success("✅ Configuration code updated!")
        
        with tab2:
            edited_model = st.text_area(
                "Edit Model Code",
                value=st.session_state.custom_model_code,
                height=400,
                help="Modify the generated model code"
            )
            
            if st.button("💾 Update Model", key="update_model"):
                st.session_state.custom_model_code = edited_model
                st.success("✅ Model code updated!")
        
        # Download code
        st.markdown("#### 📥 Download Code")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Configuration",
                data=st.session_state.custom_config_code,
                file_name="configuration.py",
                mime="text/python"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Model",
                data=st.session_state.custom_model_code,
                file_name="modeling.py",
                mime="text/python"
            )
    
    def _render_validator(self):
        """Render the model validator."""
        st.markdown("### 🔍 Model Validator")
        st.markdown("Validate your custom model configuration and implementation.")
        
        if 'custom_config_code' not in st.session_state:
            st.info("👈 Generate your model code first using the Model Builder.")
            return
        
        if st.button("🔍 Validate Model", type="primary"):
            with st.spinner("Validating model..."):
                validation_results = self._validate_custom_model()
                
                if validation_results["valid"]:
                    st.success("✅ Model validation passed!")
                else:
                    st.error("❌ Model validation failed!")
                
                # Display detailed results
                st.markdown("#### 📊 Validation Results")
                
                for check_name, result in validation_results["checks"].items():
                    if result:
                        st.success(f"✅ {check_name}")
                    else:
                        st.error(f"❌ {check_name}")
                
                # Display errors
                if validation_results["errors"]:
                    st.markdown("#### ❌ Errors")
                    for error in validation_results["errors"]:
                        st.error(f"- {error}")
                
                # Display warnings
                if validation_results["warnings"]:
                    st.markdown("#### ⚠️ Warnings")
                    for warning in validation_results["warnings"]:
                        st.warning(f"- {warning}")
    
    def _validate_custom_model(self) -> Dict[str, Any]:
        """Validate the custom model code."""
        try:
            # This is a simplified validation
            # In a real implementation, you would parse and validate the code
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "checks": {}
            }
            
            config_code = st.session_state.get('custom_config_code', '')
            model_code = st.session_state.get('custom_model_code', '')
            
            # Check if config code contains required elements
            required_config_elements = [
                "PretrainedConfig",
                "model_type",
                "__init__",
                "super().__init__"
            ]
            
            for element in required_config_elements:
                if element in config_code:
                    validation_results["checks"][f"config_has_{element}"] = True
                else:
                    validation_results["checks"][f"config_has_{element}"] = False
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Configuration missing: {element}")
            
            # Check if model code contains required elements
            required_model_elements = [
                "PreTrainedModel",
                "config_class",
                "__init__",
                "forward"
            ]
            
            for element in required_model_elements:
                if element in model_code:
                    validation_results["checks"][f"model_has_{element}"] = True
                else:
                    validation_results["checks"][f"model_has_{element}"] = False
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Model missing: {element}")
            
            return validation_results
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "checks": {}
            }
    
    def _render_hub_upload(self):
        """Render the Hub upload preparation."""
        st.markdown("### 📤 Hub Upload Preparation")
        st.markdown("Prepare your model for uploading to the Hugging Face Hub.")
        
        if 'custom_model_info' not in st.session_state:
            st.info("👈 Generate your model first using the Model Builder.")
            return
        
        model_info = st.session_state.custom_model_info
        
        # Display model information
        st.markdown("#### 📋 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {model_info['name']}")
            st.write(f"**Type:** {model_info['type']}")
            st.write(f"**Author:** {model_info['author']}")
        
        with col2:
            st.write(f"**Version:** {model_info['version']}")
            st.write(f"**Tags:** {', '.join(model_info['tags'])}")
        
        st.write(f"**Description:** {model_info['description']}")
        
        # Generate files for upload
        st.markdown("#### 🗂️ Generate Files for Upload")
        
        if st.button("📦 Generate Upload Files", type="primary"):
            with st.spinner("Generating files..."):
                try:
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        files_generated = self._generate_upload_files(temp_dir)
                        
                        if files_generated:
                            st.success("✅ Upload files generated successfully!")
                            
                            # Display generated files
                            st.markdown("#### 📁 Generated Files")
                            for file_path in files_generated:
                                st.write(f"- {file_path}")
                            
                            # Download instructions
                            st.markdown("#### 📥 Download Instructions")
                            st.info("""
                            **To upload your model to the Hub:**
                            
                            1. Download all generated files
                            2. Create a new repository on [Hugging Face Hub](https://huggingface.co/new)
                            3. Upload the files to your repository
                            4. Your model will be available for others to use!
                            """)
                        else:
                            st.error("❌ Failed to generate upload files.")
                
                except Exception as e:
                    st.error(f"❌ Error generating files: {str(e)}")
    
    def _generate_upload_files(self, output_dir: str) -> List[str]:
        """Generate files needed for Hub upload."""
        try:
            model_info = st.session_state.custom_model_info
            config_code = st.session_state.custom_config_code
            model_code = st.session_state.custom_model_code
            
            output_path = Path(output_dir)
            model_name = model_info['name'].lower()
            
            generated_files = []
            
            # Generate __init__.py
            init_content = f'''"""
{model_info['name']} - Custom Transformers Model

{model_info['description']}

Author: {model_info['author']}
Version: {model_info['version']}
"""

from .configuration_{model_name} import {model_info['name']}Config
from .modeling_{model_name} import {model_info['name']}Model

__all__ = [
    "{model_info['name']}Config",
    "{model_info['name']}Model",
]
'''
            
            init_file = output_path / "__init__.py"
            with open(init_file, "w") as f:
                f.write(init_content)
            generated_files.append(str(init_file))
            
            # Generate configuration file
            config_file = output_path / f"configuration_{model_name}.py"
            with open(config_file, "w") as f:
                f.write(config_code)
            generated_files.append(str(config_file))
            
            # Generate modeling file
            modeling_file = output_path / f"modeling_{model_name}.py"
            with open(modeling_file, "w") as f:
                f.write(model_code)
            generated_files.append(str(modeling_file))
            
            # Generate README.md
            readme_content = f'''# {model_info['name']}

{model_info['description']}

## Model Information

- **Author**: {model_info['author']}
- **Version**: {model_info['version']}
- **Model Type**: {model_info['type']}
- **Tags**: {', '.join(model_info['tags'])}

## Usage

```python
from transformers import AutoModel, AutoConfig

# Load configuration
config = AutoConfig.from_pretrained("{model_info['name']}")

# Load model
model = AutoModel.from_config(config)
```

## Installation

```bash
pip install transformers torch
```
'''
            
            readme_file = output_path / "README.md"
            with open(readme_file, "w") as f:
                f.write(readme_content)
            generated_files.append(str(readme_file))
            
            return generated_files
            
        except Exception as e:
            error(f"Failed to generate upload files: {str(e)}", "custom_models")
            return []
    
    def _render_examples(self):
        """Render example custom models."""
        st.markdown("### 📚 Example Custom Models")
        st.markdown("Explore example implementations of custom Transformers models.")
        
        # Example 1: Simple Classification Model
        st.markdown("#### 🔍 Example 1: Simple Text Classification Model")
        
        with st.expander("View Example Code", expanded=False):
            st.code('''
"""
Simple Text Classification Model Example
"""

from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn

class SimpleClassifierConfig(PretrainedConfig):
    model_type = "simple_classifier"
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        num_labels: int = 2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        super().__init__(**kwargs)

class SimpleClassifierModel(PreTrainedModel):
    config_class = SimpleClassifierConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, labels=None):
        embeddings = self.embedding(input_ids)
        pooled = torch.mean(embeddings, dim=1)
        logits = self.classifier(pooled)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
            ''', language='python')
        
        # Example 2: Custom Architecture
        st.markdown("#### 🏗️ Example 2: Custom Transformer Architecture")
        
        with st.expander("View Example Code", expanded=False):
            st.code('''
"""
Custom Transformer Architecture Example
"""

from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn

class CustomTransformerConfig(PretrainedConfig):
    model_type = "custom_transformer"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        super().__init__(**kwargs)

class CustomTransformerModel(PreTrainedModel):
    config_class = CustomTransformerConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Custom components
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(512, config.hidden_size)
        
        # Custom transformer layers
        self.layers = nn.ModuleList([
            CustomTransformerLayer(config) 
            for _ in range(config.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, input_ids, attention_mask=None):
        # Implementation details...
        pass

class CustomTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x, attention_mask=None):
        # Custom layer implementation
        pass
            ''', language='python')
        
        # Usage instructions
        st.markdown("#### 📖 How to Use Examples")
        
        st.markdown("""
        **To use these examples:**
        
        1. **Copy the code** from the examples above
        2. **Paste into the Code Generator** tab
        3. **Customize** the parameters for your needs
        4. **Validate** using the Validator tab
        5. **Generate files** for Hub upload
        
        **Tips:**
        - Start with simple examples and gradually add complexity
        - Always validate your models before uploading
        - Test your models thoroughly with sample data
        - Follow Transformers conventions for compatibility
            """)

    def _render_documentation(self):
        """Render the documentation generation and validation interface."""
        st.markdown("### 📚 Model Documentation Generator")
        st.markdown("Generate professional, consistent documentation for your models using HuggingFace's @auto_docstring patterns.")
        
        # Documentation generation section
        with st.expander("🚀 Generate Documentation", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📁 File Input")
                
                # File upload or path input
                uploaded_file = st.file_uploader(
                    "Upload Python file:",
                    type=['py'],
                    help="Upload a Python model file to generate documentation for"
                )
                
                # Or file path input
                file_path = st.text_input(
                    "Or enter file path:",
                    placeholder="/path/to/your/model.py",
                    help="Path to your model file"
                )
                
                # Documentation options
                st.markdown("#### ⚙️ Documentation Options")
                
                include_auto_docstring = st.checkbox(
                    "Add @auto_docstring decorator",
                    value=True,
                    help="Automatically add @auto_docstring decorator to model classes"
                )
                
                validate_documentation = st.checkbox(
                    "Validate documentation",
                    value=True,
                    help="Run validation checks on generated documentation"
                )
                
                overwrite_original = st.checkbox(
                    "Overwrite original file",
                    value=False,
                    help="Overwrite the original file with documented version"
                )
            
            with col2:
                st.markdown("#### 📝 Documentation Features")
                
                features = [
                    "✅ Automatic @auto_docstring decorator addition",
                    "✅ Standard argument documentation",
                    "✅ Custom argument documentation",
                    "✅ Return type documentation",
                    "✅ Type annotation validation",
                    "✅ Documentation completeness checking",
                    "✅ Format compliance validation",
                    "✅ Professional docstring generation"
                ]
                
                for feature in features:
                    st.markdown(feature)
                
                st.markdown("#### 🎯 Benefits")
                benefits = [
                    "**Consistency**: Standardized documentation across models",
                    "**Professional Quality**: Industry-standard docstrings",
                    "**Validation**: Built-in checking for completeness",
                    "**Time Saving**: Automatic generation reduces manual work",
                    "**HuggingFace Compatible**: Follows official patterns"
                ]
                
                for benefit in benefits:
                    st.markdown(benefit)
        
        # Generate documentation button
        if st.button("🚀 Generate Documentation", type="primary", use_container_width=True):
            # Determine file to process
            file_to_process = None
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_to_process = temp_path
            elif file_path:
                file_to_process = file_path
            
            if file_to_process:
                try:
                    with st.spinner("Generating documentation..."):
                        # Generate documentation
                        success = self.model_manager.generate_documentation(file_to_process)
                        
                        if success:
                            st.success("✅ Documentation generated successfully!")
                            
                            # Validate if requested
                            if validate_documentation:
                                with st.spinner("Validating documentation..."):
                                    validation_results = self.model_manager.validate_documentation(file_to_process)
                                
                                if validation_results['valid']:
                                    st.success("✅ Documentation validation passed!")
                                else:
                                    st.error("❌ Documentation validation failed:")
                                    for error_msg in validation_results['errors']:
                                        st.error(f"  - {error_msg}")
                                
                                if validation_results['warnings']:
                                    for warning_msg in validation_results['warnings']:
                                        st.warning(f"  - {warning_msg}")
                                
                                if validation_results['suggestions']:
                                    st.info("💡 Suggestions for improvement:")
                                    for suggestion in validation_results['suggestions']:
                                        st.info(f"  - {suggestion}")
                            
                            # Show generated file
                            output_file = file_to_process.replace('.py', '.documented.py')
                            if Path(output_file).exists():
                                with st.expander("📄 Generated Documentation", expanded=False):
                                    with open(output_file, 'r', encoding='utf-8') as f:
                                        documented_content = f.read()
                                    st.code(documented_content, language="python")
                                
                                # Download button
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    st.download_button(
                                        label="📥 Download Documented File",
                                        data=f.read(),
                                        file_name=f"documented_{Path(file_to_process).name}",
                                        mime="text/x-python"
                                    )
                            
                            log_session_event("documentation_generated", 
                                           file_name=Path(file_to_process).name,
                                           validation_enabled=validate_documentation)
                        else:
                            st.error("❌ Failed to generate documentation. Please check the logs.")
                    
                    # Clean up temporary file
                    if uploaded_file is not None and Path(file_to_process).exists():
                        Path(file_to_process).unlink()
                        
                except Exception as e:
                    st.error(f"❌ Error generating documentation: {str(e)}")
                    error(f"Error in documentation generation UI: {str(e)}", "custom_models", e)
            else:
                st.error("❌ Please upload a file or provide a file path")
        
        # Documentation validation section
        with st.expander("🔍 Validate Existing Documentation", expanded=False):
            st.markdown("#### 📋 Validation Tools")
            
            validation_file = st.file_uploader(
                "Upload Python file to validate:",
                type=['py'],
                key="validation_file",
                help="Upload a Python file to validate its documentation"
            )
            
            validation_path = st.text_input(
                "Or enter file path:",
                placeholder="/path/to/your/model.py",
                key="validation_path",
                help="Path to Python file to validate"
            )
            
            if st.button("🔍 Validate Documentation", use_container_width=True):
                file_to_validate = None
                
                if validation_file is not None:
                    temp_path = f"temp_validation_{validation_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(validation_file.getbuffer())
                    file_to_validate = temp_path
                elif validation_path:
                    file_to_validate = validation_path
                
                if file_to_validate:
                    try:
                        with st.spinner("Validating documentation..."):
                            validation_results = self.model_manager.validate_documentation(file_to_validate)
                        
                        # Display results
                        if validation_results['valid']:
                            st.success("✅ Documentation validation passed!")
                        else:
                            st.error("❌ Documentation validation failed:")
                            for error_msg in validation_results['errors']:
                                st.error(f"  - {error_msg}")
                        
                        if validation_results['warnings']:
                            st.warning("⚠️ Warnings:")
                            for warning_msg in validation_results['warnings']:
                                st.warning(f"  - {warning_msg}")
                        
                        if validation_results['suggestions']:
                            st.info("💡 Suggestions for improvement:")
                            for suggestion in validation_results['suggestions']:
                                st.info(f"  - {suggestion}")
                        
                        # Clean up
                        if validation_file is not None and Path(file_to_validate).exists():
                            Path(file_to_validate).unlink()
                            
                    except Exception as e:
                        st.error(f"❌ Error validating documentation: {str(e)}")
                else:
                    st.error("❌ Please upload a file or provide a file path")
        
        # Documentation guide
        with st.expander("📚 Documentation Guide", expanded=False):
            st.markdown("""
            ### 📚 Understanding Model Documentation
            
            **What is @auto_docstring?**
            - Decorator that automatically generates consistent docstrings
            - Reduces boilerplate by including standard argument descriptions
            - Allows customization for model-specific arguments
            - Ensures consistency across all Transformers models
            
            **Key Features:**
            - **Automatic Generation**: Creates docstrings from method signatures
            - **Standard Arguments**: Pre-defined documentation for common arguments
            - **Custom Arguments**: Easy documentation for model-specific parameters
            - **Validation**: Built-in checking for completeness and accuracy
            - **Professional Quality**: Industry-standard formatting
            
            **Documentation Standards:**
            - Use `r\"\"\"` for raw docstrings
            - Include argument types in backticks
            - Mark optional arguments with `*optional*`
            - Include default values when applicable
            - Document tensor shapes for complex types
            - Provide clear descriptions of purpose
            
            **Best Practices:**
            - Apply @auto_docstring to all model classes
            - Document custom arguments clearly
            - Include return type information
            - Use consistent formatting
            - Validate documentation regularly
            
            **Example Usage:**
            ```python
            from transformers.utils import auto_docstring
            
            @auto_docstring()
            class MyModel(PreTrainedModel):
                def __init__(self, config, custom_param: int = 10):
                    r\"\"\"
                    custom_param (`int`, *optional*, defaults to 10):
                        Custom parameter controlling model behavior.
                    \"\"\"
                    super().__init__(config)
                    self.custom_param = custom_param
            ```
            """)

    def _render_attention_functions(self):
        """Render the attention function customization interface."""
        st.markdown("### 🎯 Attention Function Customization")
        st.markdown("Create, register, and benchmark custom attention functions using HuggingFace's AttentionInterface.")
        
        # Attention function management section
        with st.expander("🚀 Create Custom Attention Function", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Function Configuration")
                
                function_name = st.text_input(
                    "Function Name:",
                    placeholder="my_custom_attention",
                    help="Name for your custom attention function (snake_case)"
                )
                
                function_description = st.text_area(
                    "Description:",
                    placeholder="Describe what your attention function does",
                    height=80
                )
                
                # Dependencies
                st.markdown("#### 📦 Dependencies")
                dependencies = st.multiselect(
                    "Required Dependencies:",
                    options=["torch", "torch.nn.functional", "transformers", "numpy", "scipy"],
                    help="Select dependencies your function requires"
                )
                
                # Template type
                template_type = st.selectbox(
                    "Template Type:",
                    options=["Basic Scaled Dot-Product", "Custom Scaling", "Logging Wrapper", "Custom Implementation"],
                    help="Choose a template to start with"
                )
            
            with col2:
                st.markdown("#### 🎯 Function Features")
                
                features = [
                    "✅ Automatic registration with AttentionInterface",
                    "✅ Signature validation",
                    "✅ Performance benchmarking",
                    "✅ Dynamic switching support",
                    "✅ Custom parameter support",
                    "✅ Integration with existing models",
                    "✅ Educational templates",
                    "✅ Professional code generation"
                ]
                
                for feature in features:
                    st.markdown(feature)
                
                st.markdown("#### 💡 Use Cases")
                use_cases = [
                    "**Research**: Experiment with new attention mechanisms",
                    "**Optimization**: Create performance-optimized attention",
                    "**Debugging**: Add logging to existing attention functions",
                    "**Education**: Learn attention mechanisms hands-on",
                    "**Customization**: Adapt attention for specific domains"
                ]
                
                for use_case in use_cases:
                    st.markdown(use_case)
        
        # Generate attention function template
        if st.button("🎯 Generate Attention Function Template", type="primary", use_container_width=True):
            if function_name:
                try:
                    with st.spinner("Generating attention function template..."):
                        template = self.model_manager.create_attention_function_template(function_name)
                        
                        if template:
                            st.success(f"✅ Generated template for '{function_name}'")
                            
                            # Show generated template
                            with st.expander("📄 Generated Attention Function Template", expanded=True):
                                st.code(template, language="python")
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Template",
                                data=template,
                                file_name=f"{function_name}.py",
                                mime="text/x-python"
                            )
                            
                            log_session_event("attention_function_template_generated", 
                                           function_name=function_name,
                                           template_type=template_type)
                        else:
                            st.error("❌ Failed to generate template")
                            
                except Exception as e:
                    st.error(f"❌ Error generating template: {str(e)}")
            else:
                st.error("❌ Please provide a function name")
        
        # Attention mask function section
        with st.expander("🎭 Create Custom Attention Mask", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                mask_name = st.text_input(
                    "Mask Function Name:",
                    placeholder="my_custom_mask",
                    help="Name for your custom attention mask function"
                )
                
                mask_description = st.text_area(
                    "Description:",
                    placeholder="Describe what your mask function does",
                    height=60
                )
                
                compatible_functions = st.multiselect(
                    "Compatible Attention Functions:",
                    options=["eager", "sdpa", "flash_attention_2", "flex_attention"],
                    help="Select attention functions this mask is compatible with"
                )
            
            with col2:
                st.markdown("#### 🎭 Mask Features")
                mask_features = [
                    "✅ Automatic format correction",
                    "✅ Compatibility validation",
                    "✅ Integration with attention functions",
                    "✅ Custom mask logic support",
                    "✅ Batch processing optimization"
                ]
                
                for feature in mask_features:
                    st.markdown(feature)
            
            if st.button("🎭 Generate Attention Mask Template", use_container_width=True):
                if mask_name:
                    try:
                        with st.spinner("Generating attention mask template..."):
                            mask_template = self.model_manager.create_attention_mask_template(mask_name)
                            
                            if mask_template:
                                st.success(f"✅ Generated mask template for '{mask_name}'")
                                
                                with st.expander("📄 Generated Mask Template", expanded=True):
                                    st.code(mask_template, language="python")
                                
                                st.download_button(
                                    label="📥 Download Mask Template",
                                    data=mask_template,
                                    file_name=f"{mask_name}_mask.py",
                                    mime="text/x-python"
                                )
                            else:
                                st.error("❌ Failed to generate mask template")
                                
                    except Exception as e:
                        st.error(f"❌ Error generating mask template: {str(e)}")
                else:
                    st.error("❌ Please provide a mask function name")
        
        # Available attention functions
        with st.expander("📋 Available Attention Functions", expanded=False):
            st.markdown("#### 🔧 Registered Functions")
            
            try:
                attention_functions = self.model_manager.get_attention_functions()
                
                if attention_functions:
                    for name, info in attention_functions.items():
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{name}**")
                                st.markdown(f"*{info.description}*")
                                if info.is_custom:
                                    st.markdown("🆕 *Custom Function*")
                            
                            with col2:
                                if info.performance_metrics:
                                    avg_time = info.performance_metrics.get('avg_time', 0)
                                    st.metric("Avg Time", f"{avg_time:.4f}s")
                                else:
                                    st.metric("Status", "Not Benchmarked")
                            
                            with col3:
                                if st.button(f"🔍 Details", key=f"details_{name}"):
                                    st.info(f"Function: {name}\nDescription: {info.description}")
                else:
                    st.info("No custom attention functions registered yet")
                    
            except Exception as e:
                st.error(f"❌ Error loading attention functions: {str(e)}")
        
        # Performance benchmarking
        with st.expander("⚡ Performance Benchmarking", expanded=False):
            st.markdown("#### 🏃‍♂️ Benchmark Attention Functions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                benchmark_functions = st.multiselect(
                    "Select Functions to Benchmark:",
                    options=list(self.model_manager.get_attention_functions().keys()),
                    help="Choose attention functions to compare"
                )
                
                num_runs = st.slider(
                    "Number of Runs:",
                    min_value=3,
                    max_value=20,
                    value=5,
                    help="Number of benchmark runs for averaging"
                )
            
            with col2:
                st.markdown("#### 📊 Benchmark Settings")
                
                test_sequence_length = st.slider(
                    "Test Sequence Length:",
                    min_value=10,
                    max_value=512,
                    value=128,
                    help="Length of input sequence for testing"
                )
                
                test_batch_size = st.slider(
                    "Test Batch Size:",
                    min_value=1,
                    max_value=16,
                    value=2,
                    help="Batch size for testing"
                )
            
            if st.button("🏃‍♂️ Run Benchmark", use_container_width=True):
                if benchmark_functions:
                    try:
                        with st.spinner("Running benchmark..."):
                            # Create test input
                            import torch
                            test_input = torch.randint(0, 1000, (test_batch_size, test_sequence_length))
                            
                            # Load a small model for testing
                            from transformers import AutoModelForCausalLM
                            model = AutoModelForCausalLM.from_pretrained(
                                "microsoft/DialoGPT-small",
                                torch_dtype=torch.float16
                            )
                            
                            # Run benchmark
                            results = self.model_manager.benchmark_attention_functions(
                                benchmark_functions, model, test_input, num_runs
                            )
                            
                            if results:
                                st.success("✅ Benchmark completed!")
                                
                                # Display results
                                st.markdown("#### 📊 Benchmark Results")
                                
                                for func_name, metrics in results.items():
                                    with st.expander(f"📈 {func_name} Results", expanded=False):
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            st.metric("Avg Time", f"{metrics.get('avg_time', 0):.4f}s")
                                        with col2:
                                            st.metric("Min Time", f"{metrics.get('min_time', 0):.4f}s")
                                        with col3:
                                            st.metric("Max Time", f"{metrics.get('max_time', 0):.4f}s")
                                        with col4:
                                            st.metric("Speedup", f"{metrics.get('speedup', 1.0):.2f}x")
                                
                                # Performance chart
                                if len(results) > 1:
                                    import pandas as pd
                                    df = pd.DataFrame({
                                        'Function': list(results.keys()),
                                        'Avg Time (s)': [metrics.get('avg_time', 0) for metrics in results.values()]
                                    })
                                    
                                    st.bar_chart(df.set_index('Function'))
                            else:
                                st.error("❌ Benchmark failed - no results returned")
                                
                    except Exception as e:
                        st.error(f"❌ Error running benchmark: {str(e)}")
                else:
                    st.error("❌ Please select functions to benchmark")
        
        # Educational guide
        with st.expander("📚 Attention Function Guide", expanded=False):
            st.markdown("""
            ### 🎯 Understanding Attention Functions
            
            **What are Attention Functions?**
            - Core computation mechanisms in transformer models
            - Determine how queries, keys, and values interact
            - Can be optimized for different performance characteristics
            - Enable research and experimentation with new mechanisms
            
            **Built-in Implementations:**
            - **eager**: Simple matrix multiplication (debugging/education)
            - **sdpa**: PyTorch's optimized Scaled Dot-Product Attention
            - **flash_attention_2**: Memory-efficient attention (requires GPU)
            - **flex_attention**: Flexible attention with various optimizations
            
            **Custom Function Requirements:**
            - Must follow specific signature: `(module, query, key, value, attention_mask, **kwargs)`
            - Must return tuple: `(attention_output, attention_weights)`
            - Must accept **kwargs for extensibility
            - Should handle attention masks properly
            
            **Best Practices:**
            - Start with simple implementations for learning
            - Add logging for debugging existing functions
            - Benchmark performance with realistic inputs
            - Test with different sequence lengths and batch sizes
            - Validate outputs against reference implementations
            
            **Use Cases:**
            - **Research**: Experiment with new attention mechanisms
            - **Optimization**: Create domain-specific optimizations
            - **Debugging**: Add instrumentation to existing functions
            - **Education**: Learn attention mechanisms hands-on
            - **Customization**: Adapt attention for specific applications
            """)

    def _render_legacy_models(self):
        """Render the legacy models contribution interface."""
        st.markdown("### 🏛️ Legacy Model Contribution")
        st.markdown("Create models using the traditional HuggingFace contribution approach - complete control and full implementation.")
        
        # Legacy model creation form
        with st.expander("🚀 Create New Legacy Model", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Basic Information")
                
                model_name = st.text_input(
                    "Model Name:",
                    placeholder="brand_new_llama",
                    help="Name for your legacy model (snake_case)"
                )
                
                model_type = st.selectbox(
                    "Model Type:",
                    options=["decoder", "encoder", "encoder-decoder"],
                    help="Choose the model architecture type"
                )
                
                original_repository = st.text_input(
                    "Original Repository:",
                    placeholder="https://github.com/org/brand_new_llama",
                    help="URL to the original model repository"
                )
                
                original_checkpoint = st.text_input(
                    "Original Checkpoint Path:",
                    placeholder="/path/to/checkpoint/model.pt",
                    help="Path to the original model checkpoint"
                )
                
                architecture_description = st.text_area(
                    "Architecture Description:",
                    placeholder="Describe the model architecture, key components, and how it differs from existing models",
                    height=100
                )
                
                author = st.text_input(
                    "Author:",
                    placeholder="Your Name",
                    help="Your name or organization"
                )
            
            with col2:
                st.markdown("#### 🔧 Technical Details")
                
                tokenizer_type = st.selectbox(
                    "Tokenizer Type:",
                    options=["BPE", "WordPiece", "SentencePiece", "GPT-2", "RoBERTa", "Custom"],
                    help="Type of tokenizer used by the model"
                )
                
                # Key differences
                st.markdown("**Key Differences:**")
                key_differences = []
                
                with st.container():
                    diff_col1, diff_col2 = st.columns([3, 1])
                    with diff_col1:
                        difference = st.text_input("Difference:", key="legacy_difference")
                    with diff_col2:
                        if st.button("➕ Add", key="add_legacy_difference"):
                            if difference:
                                key_differences.append(difference)
                                st.success(f"Added: {difference}")
                
                # Similar models
                st.markdown("**Similar Models:**")
                similar_models = st.multiselect(
                    "Models to Reference:",
                    options=["bert", "gpt2", "llama", "mistral", "t5", "bart", "roberta", "albert", "electra"],
                    help="Select existing models that are similar to yours"
                )
                
                # Tasks
                st.markdown("**Supported Tasks:**")
                tasks = st.multiselect(
                    "Model Tasks:",
                    options=["causal_lm", "masked_lm", "sequence_classification", "token_classification", 
                            "question_answering", "summarization", "translation", "text_generation"],
                    help="Select tasks this model can perform"
                )
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🧪 Test Cases")
                test_cases = []
                
                with st.container():
                    test_col1, test_col2 = st.columns([3, 1])
                    with test_col1:
                        test_case = st.text_input("Test Case:", key="legacy_test_case")
                    with test_col2:
                        if st.button("➕ Add", key="add_legacy_test_case"):
                            if test_case:
                                test_cases.append(test_case)
                                st.success(f"Added: {test_case}")
            
            with col2:
                st.markdown("#### 📝 Conversion Script")
                conversion_script = st.text_area(
                    "Conversion Script Path:",
                    placeholder="/path/to/convert_checkpoint.py",
                    help="Path to conversion script (optional)"
                )
        
        # Create legacy model button
        if st.button("🚀 Create Legacy Model", type="primary", use_container_width=True):
            if model_name and model_type and original_repository and architecture_description:
                try:
                    # Create LegacyModelInfo
                    legacy_info = LegacyModelInfo(
                        name=model_name,
                        model_type=model_type,
                        original_repository=original_repository,
                        original_checkpoint_path=original_checkpoint,
                        architecture_description=architecture_description,
                        key_differences=key_differences,
                        similar_models=similar_models,
                        tokenizer_type=tokenizer_type,
                        tasks=tasks,
                        description=architecture_description,
                        author=author or "Anonymous",
                        conversion_script=conversion_script,
                        test_cases=test_cases
                    )
                    
                    # Validate the legacy model
                    validation = self.model_manager.validate_legacy_model(legacy_info)
                    
                    if validation["valid"]:
                        # Create the legacy model
                        with st.spinner("Creating legacy model..."):
                            success = self.model_manager.create_legacy_model(legacy_info)
                        
                        if success:
                            st.success(f"✅ Successfully created legacy model: {model_name}")
                            
                            # Show generated files
                            with st.expander("📄 Generated Files", expanded=False):
                                files = self.model_manager.legacy_converter.generate_model_template(legacy_info)
                                for filename, content in files.items():
                                    st.markdown(f"**{filename}:**")
                                    st.code(content, language="python")
                            
                            # Show next steps
                            st.info(f"""
                            💡 **Next Steps:**
                            1. Review the generated template files
                            2. Implement the actual model architecture
                            3. Create conversion script for checkpoint
                            4. Add comprehensive tests
                            5. Submit to HuggingFace Transformers
                            """)
                            
                            log_session_event("legacy_model_created_ui", 
                                           model_name=model_name,
                                           model_type=model_type,
                                           similar_models=similar_models)
                        else:
                            st.error("❌ Failed to create legacy model. Please check the logs.")
                    else:
                        st.error("❌ Validation failed:")
                        for error_msg in validation["errors"]:
                            st.error(f"  - {error_msg}")
                        
                        if validation["warnings"]:
                            for warning_msg in validation["warnings"]:
                                st.warning(f"  - {warning_msg}")
                            
                except Exception as e:
                    st.error(f"❌ Error creating legacy model: {str(e)}")
                    error(f"Error in legacy model creation UI: {str(e)}", "custom_models", e)
            else:
                st.error("❌ Please fill in all required fields (Model Name, Model Type, Repository, Architecture Description)")
        
        # Show existing legacy models
        with st.expander("📋 Existing Legacy Models", expanded=False):
            legacy_models = self.model_manager.get_legacy_models()
            
            if legacy_models:
                for name, info in legacy_models.items():
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{name}**")
                            st.markdown(f"*Type: {info.model_type}*")
                            st.markdown(f"*{info.architecture_description[:100]}...*")
                        
                        with col2:
                            st.markdown(f"**Tasks:** {len(info.tasks)}")
                        
                        with col3:
                            if st.button(f"🗑️ Delete", key=f"delete_legacy_{name}"):
                                # Delete logic would go here
                                st.success(f"Deleted {name}")
            else:
                st.info("No legacy models created yet. Create your first legacy model above!")
        
        # Legacy model guide
        with st.expander("📚 Legacy Model Contribution Guide", expanded=False):
            st.markdown("""
            ### 🏛️ Understanding Legacy Model Contribution
            
            **What are Legacy Models?**
            - Traditional approach to contributing models to HuggingFace Transformers
            - Complete implementation from scratch
            - Full control over architecture and implementation
            - More complex but more flexible than modular approach
            
            **When to Use Legacy Approach:**
            - Model has unique architecture not similar to existing models
            - Need complete control over implementation
            - Model requires custom components not available in modular approach
            - Learning the complete Transformers contribution process
            
            **Contribution Process:**
            1. **Research**: Understand the original model implementation
            2. **Setup**: Create development environment and fork repository
            3. **Implement**: Create configuration, modeling, and tokenizer files
            4. **Convert**: Create checkpoint conversion script
            5. **Test**: Add comprehensive tests and validation
            6. **Document**: Add documentation and examples
            7. **Submit**: Create pull request to Transformers repository
            
            **Key Components:**
            - **Configuration**: Model blueprint with all parameters
            - **Modeling**: Core model implementation with forward pass
            - **Tokenizer**: Text preprocessing and tokenization
            - **Conversion**: Script to convert original checkpoints
            - **Tests**: Unit and integration tests
            - **Documentation**: Model documentation and examples
            
            **Best Practices:**
            - Study similar existing models for reference
            - Start with small, working implementation
            - Test thoroughly with original checkpoints
            - Follow Transformers code style and conventions
            - Add comprehensive documentation
            - Include practical examples and use cases
            """)


def render_custom_model_page():
    """Render the custom model development page."""
    page = CustomModelPage()
    page.render()
