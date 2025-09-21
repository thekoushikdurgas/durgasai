"""
Custom Model Manager for handling custom Transformers models.

This module provides comprehensive support for:
- Custom model configuration creation
- Custom model implementation
- AutoClass registration and management
- Custom model upload and sharing
- Model validation and testing

Key Classes:
- CustomModelManager: Main interface for custom model operations
- CustomModelConfig: Base class for custom configurations
- CustomModelValidator: Validation utilities for custom models
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union, Callable
from dataclasses import dataclass, asdict
import importlib.util
import sys

import streamlit as st
import torch
from transformers import (
    PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel,
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForImageClassification
)

from .logger import debug, info, warning, error, log_session_event
from .docstring_generator import docstring_generator, docstring_validator
from .attention_manager import attention_manager

# Import for modular model support
try:
    import ast
    import inspect
    from typing import get_type_hints
    MODULAR_AVAILABLE = True
except ImportError:
    MODULAR_AVAILABLE = False


@dataclass
class CustomModelInfo:
    """Information about a custom model."""
    name: str
    model_type: str
    config_class: str
    model_class: str
    description: str
    author: str
    version: str = "1.0.0"
    tags: List[str] = None
    dependencies: List[str] = None
    is_modular: bool = False
    base_model: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ModularModelInfo:
    """Information about a modular model."""
    name: str
    base_model: str
    config_changes: Dict[str, Any]
    model_changes: Dict[str, Any]
    new_components: List[str]
    removed_components: List[str]
    description: str
    author: str
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.config_changes is None:
            self.config_changes = {}
        if self.model_changes is None:
            self.model_changes = {}
        if self.new_components is None:
            self.new_components = []
        if self.removed_components is None:
            self.removed_components = []


@dataclass
class LegacyModelInfo:
    """Information about a legacy model contribution."""
    name: str
    model_type: str  # encoder, decoder, encoder-decoder
    original_repository: str
    original_checkpoint_path: str
    architecture_description: str
    key_differences: List[str]
    similar_models: List[str]
    tokenizer_type: str
    tasks: List[str]
    description: str
    author: str
    version: str = "1.0.0"
    conversion_script: str = None
    test_cases: List[str] = None
    
    def __post_init__(self):
        if self.key_differences is None:
            self.key_differences = []
        if self.similar_models is None:
            self.similar_models = []
        if self.tasks is None:
            self.tasks = []
        if self.test_cases is None:
            self.test_cases = []


class CustomModelValidator:
    """Validator for custom model configurations and implementations."""
    
    @staticmethod
    def validate_config(config_class: Type[PretrainedConfig]) -> Dict[str, Any]:
        """
        Validate a custom configuration class.
        
        Args:
            config_class: Custom configuration class
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Check if it subclasses PretrainedConfig
            if not issubclass(config_class, PretrainedConfig):
                validation_results["valid"] = False
                validation_results["errors"].append("Config class must subclass PretrainedConfig")
            else:
                validation_results["checks"]["subclasses_pretrained_config"] = True
            
            # Check for model_type attribute
            if not hasattr(config_class, 'model_type'):
                validation_results["valid"] = False
                validation_results["errors"].append("Config class must have 'model_type' attribute")
            else:
                validation_results["checks"]["has_model_type"] = True
                validation_results["checks"]["model_type"] = getattr(config_class, 'model_type')
            
            # Check __init__ method accepts **kwargs
            import inspect
            init_signature = inspect.signature(config_class.__init__)
            if 'kwargs' not in init_signature.parameters:
                validation_results["warnings"].append("Config __init__ should accept **kwargs")
            
            # Test instantiation
            try:
                test_config = config_class()
                validation_results["checks"]["instantiation"] = True
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Config instantiation failed: {str(e)}")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def validate_model(model_class: Type[PreTrainedModel]) -> Dict[str, Any]:
        """
        Validate a custom model class.
        
        Args:
            model_class: Custom model class
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Check if it subclasses PreTrainedModel
            if not issubclass(model_class, PreTrainedModel):
                validation_results["valid"] = False
                validation_results["errors"].append("Model class must subclass PreTrainedModel")
            else:
                validation_results["checks"]["subclasses_pretrained_model"] = True
            
            # Check for config_class attribute
            if not hasattr(model_class, 'config_class'):
                validation_results["valid"] = False
                validation_results["errors"].append("Model class must have 'config_class' attribute")
            else:
                validation_results["checks"]["has_config_class"] = True
                validation_results["checks"]["config_class"] = getattr(model_class, 'config_class')
            
            # Check __init__ method accepts config
            import inspect
            init_signature = inspect.signature(model_class.__init__)
            params = list(init_signature.parameters.keys())
            if len(params) < 2 or params[1] != 'config':
                validation_results["warnings"].append("Model __init__ should accept 'config' as first parameter")
            
            # Check for forward method
            if not hasattr(model_class, 'forward'):
                validation_results["valid"] = False
                validation_results["errors"].append("Model class must have 'forward' method")
            else:
                validation_results["checks"]["has_forward_method"] = True
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def validate_modular_model(modular_info: ModularModelInfo) -> Dict[str, Any]:
        """
        Validate a modular model configuration.
        
        Args:
            modular_info: Modular model information
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Check if modular support is available
            if not MODULAR_AVAILABLE:
                validation_results["valid"] = False
                validation_results["errors"].append("Modular model support not available")
                return validation_results
            
            # Validate base model
            if not modular_info.base_model:
                validation_results["valid"] = False
                validation_results["errors"].append("Base model must be specified")
            else:
                validation_results["checks"]["has_base_model"] = True
            
            # Validate model name
            if not modular_info.name:
                validation_results["valid"] = False
                validation_results["errors"].append("Model name must be specified")
            else:
                validation_results["checks"]["has_name"] = True
            
            # Check for circular dependencies
            if modular_info.base_model == modular_info.name:
                validation_results["valid"] = False
                validation_results["errors"].append("Model cannot inherit from itself")
            
            # Validate component changes
            if modular_info.config_changes:
                validation_results["checks"]["has_config_changes"] = True
            
            if modular_info.model_changes:
                validation_results["checks"]["has_model_changes"] = True
            
            if modular_info.new_components:
                validation_results["checks"]["has_new_components"] = True
                validation_results["checks"]["new_components_count"] = len(modular_info.new_components)
            
            if modular_info.removed_components:
                validation_results["checks"]["has_removed_components"] = True
                validation_results["checks"]["removed_components_count"] = len(modular_info.removed_components)
            
            # Check if any changes are specified
            if not any([
                modular_info.config_changes,
                modular_info.model_changes,
                modular_info.new_components,
                modular_info.removed_components
            ]):
                validation_results["warnings"].append("No changes specified - model will be identical to base model")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def validate_legacy_model(legacy_info: LegacyModelInfo) -> Dict[str, Any]:
        """
        Validate a legacy model contribution configuration.
        
        Args:
            legacy_info: Legacy model information
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Validate model type
            valid_types = ["encoder", "decoder", "encoder-decoder"]
            if legacy_info.model_type not in valid_types:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Invalid model type: {legacy_info.model_type}. Must be one of {valid_types}")
            else:
                validation_results["checks"]["valid_model_type"] = True
            
            # Validate repository URL
            if not legacy_info.original_repository:
                validation_results["valid"] = False
                validation_results["errors"].append("Original repository URL is required")
            else:
                validation_results["checks"]["has_repository"] = True
            
            # Validate checkpoint path
            if not legacy_info.original_checkpoint_path:
                validation_results["valid"] = False
                validation_results["errors"].append("Original checkpoint path is required")
            else:
                validation_results["checks"]["has_checkpoint_path"] = True
            
            # Validate architecture description
            if not legacy_info.architecture_description:
                validation_results["warnings"].append("Architecture description is missing")
            else:
                validation_results["checks"]["has_architecture_description"] = True
            
            # Validate key differences
            if not legacy_info.key_differences:
                validation_results["warnings"].append("No key differences specified - model may be identical to existing models")
            else:
                validation_results["checks"]["has_key_differences"] = True
                validation_results["checks"]["key_differences_count"] = len(legacy_info.key_differences)
            
            # Validate similar models
            if not legacy_info.similar_models:
                validation_results["warnings"].append("No similar models specified - this may make implementation more difficult")
            else:
                validation_results["checks"]["has_similar_models"] = True
                validation_results["checks"]["similar_models_count"] = len(legacy_info.similar_models)
            
            # Validate tokenizer type
            if not legacy_info.tokenizer_type:
                validation_results["warnings"].append("Tokenizer type not specified")
            else:
                validation_results["checks"]["has_tokenizer_type"] = True
            
            # Validate tasks
            if not legacy_info.tasks:
                validation_results["warnings"].append("No tasks specified")
            else:
                validation_results["checks"]["has_tasks"] = True
                validation_results["checks"]["tasks_count"] = len(legacy_info.tasks)
            
            # Validate conversion script
            if not legacy_info.conversion_script:
                validation_results["warnings"].append("No conversion script provided - manual conversion required")
            else:
                validation_results["checks"]["has_conversion_script"] = True
            
            # Validate test cases
            if not legacy_info.test_cases:
                validation_results["warnings"].append("No test cases provided")
            else:
                validation_results["checks"]["has_test_cases"] = True
                validation_results["checks"]["test_cases_count"] = len(legacy_info.test_cases)
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results


class LegacyModelConverter:
    """Converter for legacy model contributions to Transformers format."""
    
    def __init__(self):
        self.supported_frameworks = ["pytorch", "tensorflow", "jax"]
        self.model_types = ["encoder", "decoder", "encoder-decoder"]
        debug("LegacyModelConverter initialized", "legacy_models")
    
    def generate_model_template(self, legacy_info: LegacyModelInfo) -> Dict[str, str]:
        """
        Generate model template files for legacy contribution.
        
        Args:
            legacy_info: Legacy model information
            
        Returns:
            Dict of filename -> content mappings
        """
        try:
            debug(f"Generating model template for {legacy_info.name}", "legacy_models")
            
            files = {}
            
            # Generate configuration file
            config_content = self._generate_config_file(legacy_info)
            files["configuration.py"] = config_content
            
            # Generate modeling file
            modeling_content = self._generate_modeling_file(legacy_info)
            files["modeling.py"] = modeling_content
            
            # Generate tokenizer file
            tokenizer_content = self._generate_tokenizer_file(legacy_info)
            files["tokenization.py"] = tokenizer_content
            
            # Generate __init__.py file
            init_content = self._generate_init_file(legacy_info)
            files["__init__.py"] = init_content
            
            # Generate conversion script
            conversion_content = self._generate_conversion_script(legacy_info)
            files["convert_checkpoint.py"] = conversion_content
            
            # Generate test file
            test_content = self._generate_test_file(legacy_info)
            files["test_modeling.py"] = test_content
            
            info(f"Generated template files for {legacy_info.name}", "legacy_models")
            return files
            
        except Exception as e:
            error(f"Error generating model template: {str(e)}", "legacy_models", e)
            return {}
    
    def _generate_config_file(self, legacy_info: LegacyModelInfo) -> str:
        """Generate configuration file content."""
        model_name_pascal = legacy_info.name.replace("_", "").title()
        model_name_lower = legacy_info.name.lower()
        
        config_content = f'''"""
Configuration for {model_name_pascal} model.
"""

from transformers import PretrainedConfig


class {model_name_pascal}Config(PretrainedConfig):
    """
    Configuration class for {model_name_pascal}.
    
    {legacy_info.architecture_description}
    
    Args:
        vocab_size (int): Vocabulary size of the model.
        hidden_size (int): Dimension of the hidden states.
        intermediate_size (int): Dimension of the intermediate layer.
        num_hidden_layers (int): Number of hidden layers.
        num_attention_heads (int): Number of attention heads.
        max_position_embeddings (int): Maximum position embeddings.
        type_vocab_size (int): Vocabulary size for token type embeddings.
        initializer_range (float): Standard deviation for weight initialization.
        layer_norm_eps (float): Epsilon for layer normalization.
        use_cache (bool): Whether to use caching for past key values.
        **kwargs: Additional configuration parameters.
    """
    
    model_type = "{model_name_lower}"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
'''
        return config_content
    
    def _generate_modeling_file(self, legacy_info: LegacyModelInfo) -> str:
        """Generate modeling file content."""
        model_name_pascal = legacy_info.name.replace("_", "").title()
        
        modeling_content = f'''"""
{model_name_pascal} model implementation.
"""

import torch
import torch.nn as nn
from transformers import PretrainedModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput

from .configuration_{legacy_info.name} import {model_name_pascal}Config


class {model_name_pascal}PreTrainedModel(PreTrainedModel):
    """
    Base class for {model_name_pascal} models.
    """
    
    config_class = {model_name_pascal}Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize weights for different layer types."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class {model_name_pascal}Model({model_name_pascal}PreTrainedModel):
    """
    {model_name_pascal} model implementation.
    
    {legacy_info.architecture_description}
    """
    
    def __init__(self, config: {model_name_pascal}Config):
        super().__init__(config)
        
        # Model components will be implemented here
        # This is a template - actual implementation depends on the specific architecture
        
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        # Forward pass implementation
        # This is a template - actual implementation depends on the specific architecture
        pass


class {model_name_pascal}ForCausalLM({model_name_pascal}PreTrainedModel):
    """
    {model_name_pascal} model for causal language modeling.
    """
    
    def __init__(self, config: {model_name_pascal}Config):
        super().__init__(config)
        
        self.model = {model_name_pascal}Model(config)
        # Add language modeling head here
        
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        # Causal LM forward pass
        pass
'''
        return modeling_content
    
    def _generate_tokenizer_file(self, legacy_info: LegacyModelInfo) -> str:
        """Generate tokenizer file content."""
        model_name_pascal = legacy_info.name.replace("_", "").title()
        
        tokenizer_content = f'''"""
Tokenizer for {model_name_pascal} model.
"""

from transformers import PreTrainedTokenizer


class {model_name_pascal}Tokenizer(PreTrainedTokenizer):
    """
    Tokenizer for {model_name_pascal}.
    
    Tokenizer type: {legacy_info.tokenizer_type}
    """
    
    def __init__(self, vocab_file=None, merges_file=None, **kwargs):
        super().__init__(**kwargs)
        
        # Tokenizer implementation based on {legacy_info.tokenizer_type}
        # This is a template - actual implementation depends on the tokenizer type
        
    def _tokenize(self, text):
        # Tokenization implementation
        pass
    
    def _convert_token_to_id(self, token):
        # Convert token to ID
        pass
    
    def _convert_id_to_token(self, index):
        # Convert ID to token
        pass
'''
        return tokenizer_content
    
    def _generate_init_file(self, legacy_info: LegacyModelInfo) -> str:
        """Generate __init__.py file content."""
        model_name_pascal = legacy_info.name.replace("_", "").title()
        
        init_content = f'''"""
{model_name_pascal} model package.
"""

from .configuration_{legacy_info.name} import {model_name_pascal}Config
from .modeling_{legacy_info.name} import (
    {model_name_pascal}Model,
    {model_name_pascal}ForCausalLM,
    {model_name_pascal}PreTrainedModel,
)

__all__ = [
    "{model_name_pascal}Config",
    "{model_name_pascal}Model",
    "{model_name_pascal}ForCausalLM",
    "{model_name_pascal}PreTrainedModel",
]
'''
        return init_content
    
    def _generate_conversion_script(self, legacy_info: LegacyModelInfo) -> str:
        """Generate conversion script content."""
        model_name_pascal = legacy_info.name.replace("_", "").title()
        
        conversion_content = f'''"""
Conversion script for {model_name_pascal} model.
"""

import torch
from transformers import {model_name_pascal}Config, {model_name_pascal}Model


def convert_checkpoint(original_checkpoint_path, output_path):
    """
    Convert original checkpoint to Transformers format.
    
    Args:
        original_checkpoint_path (str): Path to original checkpoint
        output_path (str): Path to save converted checkpoint
    """
    
    # Load original checkpoint
    # original_weights = torch.load(original_checkpoint_path, map_location="cpu")
    
    # Create Transformers model
    config = {model_name_pascal}Config()
    model = {model_name_pascal}Model(config)
    
    # Map weights from original to Transformers format
    # This is a template - actual mapping depends on the original implementation
    
    # Save converted model
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    print(f"Converted model saved to {{output_path}}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert {model_name_pascal} checkpoint")
    parser.add_argument("--input", required=True, help="Path to original checkpoint")
    parser.add_argument("--output", required=True, help="Path to save converted checkpoint")
    
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)
'''
        return conversion_content
    
    def _generate_test_file(self, legacy_info: LegacyModelInfo) -> str:
        """Generate test file content."""
        model_name_pascal = legacy_info.name.replace("_", "").title()
        
        test_content = f'''"""
Tests for {model_name_pascal} model.
"""

import unittest
import torch
from transformers import {model_name_pascal}Config, {model_name_pascal}Model, {model_name_pascal}Tokenizer


class {model_name_pascal}ModelTest(unittest.TestCase):
    """Test cases for {model_name_pascal} model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {model_name_pascal}Config(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4
        )
    
    def test_model_creation(self):
        """Test model can be created."""
        model = {model_name_pascal}Model(self.config)
        self.assertIsNotNone(model)
    
    def test_forward_pass(self):
        """Test forward pass works."""
        model = {model_name_pascal}Model(self.config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        with torch.no_grad():
            output = model(input_ids)
            self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
'''
        return test_content


class ModularModelConverter:
    """Converter for modular model files to single-file implementations."""
    
    def __init__(self):
        self.available_base_models = [
            "bert", "gpt2", "llama", "mistral", "t5", "roberta", 
            "albert", "electra", "distilbert", "deberta"
        ]
        debug("ModularModelConverter initialized", "modular_models")
    
    def generate_modular_file(self, modular_info: ModularModelInfo) -> str:
        """
        Generate a modular model file from ModularModelInfo.
        
        Args:
            modular_info: Modular model information
            
        Returns:
            str: Generated modular file content
        """
        try:
            debug(f"Generating modular file for {modular_info.name}", "modular_models")
            
            # Generate imports
            imports = self._generate_imports(modular_info)
            
            # Generate configuration class
            config_class = self._generate_config_class(modular_info)
            
            # Generate model classes
            model_classes = self._generate_model_classes(modular_info)
            
            # Combine all parts
            modular_content = f"""# Modular {modular_info.name} Implementation
# Generated by DurgasAI Modular Model Converter

{imports}

{config_class}

{model_classes}

# Additional utility functions and classes can be added here
"""
            
            info(f"Generated modular file for {modular_info.name}", "modular_models")
            return modular_content
            
        except Exception as e:
            error(f"Error generating modular file: {str(e)}", "modular_models", e)
            return ""
    
    def _generate_imports(self, modular_info: ModularModelInfo) -> str:
        """Generate import statements for the modular file."""
        base_model_lower = modular_info.base_model.lower()
        
        imports = f"""from torch import nn
from ..{base_model_lower}.configuration_{base_model_lower} import {modular_info.base_model.title()}Config
from ..{base_model_lower}.modeling_{base_model_lower} import (
    {modular_info.base_model.title()}Model,
    {modular_info.base_model.title()}ForCausalLM,
    {modular_info.base_model.title()}ForSequenceClassification
)"""
        
        # Add specific imports based on components
        if modular_info.new_components:
            for component in modular_info.new_components:
                if "attention" in component.lower():
                    imports += "\nfrom transformers import Attention"
                elif "norm" in component.lower():
                    imports += "\nfrom transformers import LayerNorm, RMSNorm"
        
        return imports
    
    def _generate_config_class(self, modular_info: ModularModelInfo) -> str:
        """Generate configuration class."""
        base_config = f"{modular_info.base_model.title()}Config"
        new_config = f"{modular_info.name.title()}Config"
        
        config_class = f"""# {modular_info.name.title()} configuration class
class {new_config}({base_config}):
    \"\"\"
    Configuration class for {modular_info.name}.
    
    {modular_info.description}
    \"\"\"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "{modular_info.name.lower()}"
        
        # Configuration changes
"""
        
        # Add configuration changes
        for param, value in modular_info.config_changes.items():
            if isinstance(value, str):
                config_class += f'        self.{param} = "{value}"\n'
            else:
                config_class += f"        self.{param} = {value}\n"
        
        # Add removed parameters
        for param in modular_info.removed_components:
            if param in modular_info.config_changes:
                config_class += f"        del self.{param}\n"
        
        config_class += "\n"
        return config_class
    
    def _generate_model_classes(self, modular_info: ModularModelInfo) -> str:
        """Generate model classes."""
        model_classes = ""
        
        # Generate main model class
        base_model = f"{modular_info.base_model.title()}Model"
        new_model = f"{modular_info.name.title()}Model"
        
        model_classes += f"""# {modular_info.name.title()} model class
class {new_model}({base_model}):
    \"\"\"
    {modular_info.name.title()} model implementation.
    \"\"\"
    
    def __init__(self, config: {modular_info.name.title()}Config):
        super().__init__(config)
        # Model-specific changes
"""
        
        # Add model changes
        for change, value in modular_info.model_changes.items():
            model_classes += f"        self.{change} = {value}\n"
        
        model_classes += "\n"
        
        # Generate task-specific model heads
        tasks = ["ForCausalLM", "ForSequenceClassification", "ForMaskedLM"]
        for task in tasks:
            base_task_model = f"{modular_info.base_model.title()}{task}"
            new_task_model = f"{modular_info.name.title()}{task}"
            
            model_classes += f"""class {new_task_model}({base_task_model}):
    \"\"\"
    {modular_info.name.title()} for {task.replace('For', '')}.
    \"\"\"
    
    def __init__(self, config: {modular_info.name.title()}Config):
        super().__init__(config)
        self.model = {new_model}(config)

"""
        
        return model_classes
    
    def convert_modular_to_single_file(self, modular_file_path: str, output_path: str) -> bool:
        """
        Convert a modular file to single-file implementation.
        
        Args:
            modular_file_path: Path to the modular file
            output_path: Path for the output single file
            
        Returns:
            bool: True if conversion successful
        """
        try:
            debug(f"Converting modular file: {modular_file_path}", "modular_models")
            
            # Read modular file
            with open(modular_file_path, 'r', encoding='utf-8') as f:
                modular_content = f.read()
            
            # Parse and convert
            converted_content = self._parse_and_convert(modular_content)
            
            # Write converted file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(converted_content)
            
            info(f"Successfully converted modular file to {output_path}", "modular_models")
            return True
            
        except Exception as e:
            error(f"Error converting modular file: {str(e)}", "modular_models", e)
            return False
    
    def _parse_and_convert(self, modular_content: str) -> str:
        """Parse modular content and convert to single-file format."""
        # This is a simplified implementation
        # In practice, this would use AST parsing and more sophisticated conversion
        
        converted_content = modular_content
        
        # Replace relative imports with absolute imports
        converted_content = converted_content.replace("from ..", "from transformers.models.")
        
        # Add additional imports that might be needed
        additional_imports = """import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
"""
        
        # Insert additional imports at the beginning
        lines = converted_content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                import_end = i + 1
            elif line.strip() and not line.startswith('#'):
                break
        
        lines.insert(import_end, additional_imports)
        converted_content = '\n'.join(lines)
        
        return converted_content


class CustomModelManager:
    """
    Manager for custom Transformers models.
    
    This class handles:
    - Custom model registration and validation
    - Modular model creation and conversion
    - Legacy model contribution and conversion
    - Model file generation and management
    - AutoClass integration
    - Model file generation
    - Hub upload preparation
    """
    
    def __init__(self):
        """Initialize the custom model manager."""
        self.registered_models: Dict[str, CustomModelInfo] = {}
        self.modular_models: Dict[str, ModularModelInfo] = {}
        self.legacy_models: Dict[str, LegacyModelInfo] = {}
        self.validator = CustomModelValidator()
        self.modular_converter = ModularModelConverter()
        self.legacy_converter = LegacyModelConverter()
        self.attention_manager = attention_manager
        self.models_dir = Path("output/models")
        self.modular_dir = Path("output/modular_models")
        self.legacy_dir = Path("output/legacy_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.modular_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_dir.mkdir(parents=True, exist_ok=True)
        info("CustomModelManager initialized with modular, legacy, and attention support", "custom_models")
    
    def register_custom_model(
        self, 
        config_class: Type[PretrainedConfig],
        model_class: Type[PreTrainedModel],
        model_info: CustomModelInfo
    ) -> bool:
        """
        Register a custom model with the manager.
        
        Args:
            config_class: Custom configuration class
            model_class: Custom model class
            model_info: Model information
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            debug(f"Starting custom model registration: {model_info.name}", "custom_models",
                  model_type=model_info.model_type,
                  config_class=config_class.__name__,
                  model_class=model_class.__name__)
            
            # Step 1: Validate configuration class
            # Ensures the config class follows HuggingFace conventions
            debug("Validating configuration class", "custom_models")
            config_validation = self.validator.validate_config(config_class)
            
            if not config_validation["valid"]:
                error(f"Config validation failed: {config_validation['errors']}", "custom_models",
                      validation_errors=config_validation['errors'],
                      validation_warnings=config_validation.get('warnings', []))
                return False
            
            debug("Configuration class validation passed", "custom_models",
                  validation_checks=config_validation.get('checks', {}))
            
            # Step 2: Validate model class
            # Ensures the model class is compatible with HuggingFace framework
            debug("Validating model class", "custom_models")
            model_validation = self.validator.validate_model(model_class)
            
            if not model_validation["valid"]:
                error(f"Model validation failed: {model_validation['errors']}", "custom_models",
                      validation_errors=model_validation['errors'],
                      validation_warnings=model_validation.get('warnings', []))
                return False
            
            debug("Model class validation passed", "custom_models",
                  validation_checks=model_validation.get('checks', {}))
            
            # Step 3: Store model information in registry
            # This enables later retrieval and management
            debug("Storing model information in registry", "custom_models")
            self.registered_models[model_info.name] = model_info
            debug(f"Model stored in registry, total models: {len(self.registered_models)}", "custom_models")
            
            # Step 4: Register with AutoClass system
            # This enables automatic model loading via AutoModel.from_pretrained()
            debug("Registering with AutoClass system", "custom_models")
            self._register_with_autoclass(config_class, model_class, model_info)
            debug("AutoClass registration completed", "custom_models")
            
            # Log successful registration
            info(f"Custom model '{model_info.name}' registered successfully", "custom_models",
                 model_name=model_info.name,
                 model_type=model_info.model_type,
                 total_registered=len(self.registered_models))
            
            log_session_event("custom_model_registered", 
                            model_name=model_info.name,
                            model_type=model_info.model_type,
                            config_class=config_class.__name__,
                            model_class=model_class.__name__)
            
            return True
            
        except Exception as e:
            error(f"Failed to register custom model: {str(e)}", "custom_models", e,
                  model_name=model_info.name,
                  config_class=config_class.__name__,
                  model_class=model_class.__name__)
            return False
    
    def _register_with_autoclass(
        self, 
        config_class: Type[PretrainedConfig],
        model_class: Type[PreTrainedModel],
        model_info: CustomModelInfo
    ):
        """Register custom model with AutoClass APIs."""
        try:
            # Register configuration
            AutoConfig.register(model_info.model_type, config_class)
            
            # Register model based on task type
            if "causal" in model_info.model_type.lower() or "generation" in model_info.tags:
                AutoModelForCausalLM.register(config_class, model_class)
            elif "classification" in model_info.tags or "sequence" in model_info.model_type.lower():
                AutoModelForSequenceClassification.register(config_class, model_class)
            elif "image" in model_info.tags or "vision" in model_info.model_type.lower():
                AutoModelForImageClassification.register(config_class, model_class)
            else:
                # Default to general AutoModel
                AutoModel.register(config_class, model_class)
            
            debug(f"AutoClass registration completed for {model_info.name}", "custom_models")
            
        except Exception as e:
            warning(f"AutoClass registration failed: {str(e)}", "custom_models")
    
    def generate_model_files(
        self, 
        model_info: CustomModelInfo,
        config_class: Type[PretrainedConfig],
        model_class: Type[PreTrainedModel],
        output_dir: str = "custom_models"
    ) -> bool:
        """
        Generate model files for Hub upload.
        
        Args:
            model_info: Model information
            config_class: Configuration class
            model_class: Model class
            output_dir: Output directory for model files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory
            model_path = Path(output_dir) / model_info.name
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Generate __init__.py
            self._generate_init_file(model_path, model_info)
            
            # Generate configuration file
            self._generate_config_file(model_path, config_class, model_info)
            
            # Generate modeling file
            self._generate_modeling_file(model_path, model_class, model_info)
            
            # Generate README.md
            self._generate_readme(model_path, model_info)
            
            info(f"Model files generated in {model_path}", "custom_models")
            return True
            
        except Exception as e:
            error(f"Failed to generate model files: {str(e)}", "custom_models", e)
            return False
    
    def _generate_init_file(self, model_path: Path, model_info: CustomModelInfo):
        """Generate __init__.py file."""
        init_content = f'''"""
{model_info.name} - Custom Transformers Model

{model_info.description}

Author: {model_info.author}
Version: {model_info.version}
"""

from .configuration_{model_info.name.lower()} import {model_info.config_class}
from .modeling_{model_info.name.lower()} import {model_info.model_class}

__all__ = [
    "{model_info.config_class}",
    "{model_info.model_class}",
]
'''
        
        with open(model_path / "__init__.py", "w") as f:
            f.write(init_content)
    
    def _generate_config_file(self, model_path: Path, config_class: Type[PretrainedConfig], model_info: CustomModelInfo):
        """Generate configuration file."""
        # This would need to extract the actual class code
        # For now, we'll create a template
        config_content = f'''"""
Configuration for {model_info.name} model.
"""

from transformers import PretrainedConfig

class {config_class.__name__}(PretrainedConfig):
    model_type = "{model_info.model_type}"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your custom configuration parameters here
'''
        
        config_file = model_path / f"configuration_{model_info.name.lower()}.py"
        with open(config_file, "w") as f:
            f.write(config_content)
    
    def _generate_modeling_file(self, model_path: Path, model_class: Type[PreTrainedModel], model_info: CustomModelInfo):
        """Generate modeling file."""
        # This would need to extract the actual class code
        # For now, we'll create a template
        modeling_content = f'''"""
Model implementation for {model_info.name}.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_{model_info.name.lower()} import {model_info.config_class}

class {model_class.__name__}(PreTrainedModel):
    config_class = {model_info.config_class}
    
    def __init__(self, config):
        super().__init__(config)
        # Add your custom model implementation here
        
    def forward(self, **kwargs):
        # Implement your forward pass here
        pass
'''
        
        modeling_file = model_path / f"modeling_{model_info.name.lower()}.py"
        with open(modeling_file, "w") as f:
            f.write(modeling_content)
    
    def _generate_readme(self, model_path: Path, model_info: CustomModelInfo):
        """Generate README.md file."""
        readme_content = f'''# {model_info.name}

{model_info.description}

## Model Information

- **Author**: {model_info.author}
- **Version**: {model_info.version}
- **Model Type**: {model_info.model_type}
- **Tags**: {', '.join(model_info.tags)}

## Usage

```python
from transformers import AutoModel, AutoConfig

# Load configuration
config = AutoConfig.from_pretrained("{model_info.name}")

# Load model
model = AutoModel.from_config(config)
```

## Dependencies

{chr(10).join(f"- {dep}" for dep in model_info.dependencies)}

## Installation

```bash
pip install {' '.join(model_info.dependencies)}
```
'''
        
        with open(model_path / "README.md", "w") as f:
            f.write(readme_content)
    
    def prepare_for_hub_upload(
        self, 
        model_info: CustomModelInfo,
        config_class: Type[PretrainedConfig],
        model_class: Type[PreTrainedModel]
    ) -> bool:
        """
        Prepare model for Hugging Face Hub upload.
        
        Args:
            model_info: Model information
            config_class: Configuration class
            model_class: Model class
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Register for AutoClass (modifies config JSON)
            config_class.register_for_auto_class()
            model_class.register_for_auto_class("AutoModel")
            
            # Create temporary directory for upload preparation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Generate files
                self.generate_model_files(model_info, config_class, model_class, temp_dir)
                
                # Save configuration
                config_instance = config_class()
                config_instance.save_pretrained(temp_path)
                
                # Save model (if weights available)
                try:
                    model_instance = model_class(config_instance)
                    model_instance.save_pretrained(temp_path)
                except Exception as e:
                    warning(f"Could not save model weights: {str(e)}", "custom_models")
                
                info(f"Model prepared for Hub upload in {temp_path}", "custom_models")
                return True
                
        except Exception as e:
            error(f"Failed to prepare model for Hub upload: {str(e)}", "custom_models", e)
            return False
    
    # ===== MODULAR MODEL METHODS =====
    
    def create_modular_model(self, modular_info: ModularModelInfo) -> bool:
        """
        Create a modular model from ModularModelInfo.
        
        Args:
            modular_info: Modular model information
            
        Returns:
            bool: True if creation successful, False otherwise
        """
        try:
            # Validate modular model
            validation = self.validator.validate_modular_model(modular_info)
            if not validation["valid"]:
                error(f"Modular model validation failed: {validation['errors']}", "custom_models")
                return False
            
            # Generate modular file content
            modular_content = self.converter.generate_modular_file(modular_info)
            if not modular_content:
                error("Failed to generate modular file content", "custom_models")
                return False
            
            # Save modular file
            modular_file_path = self.modular_dir / f"modular_{modular_info.name.lower()}.py"
            with open(modular_file_path, 'w', encoding='utf-8') as f:
                f.write(modular_content)
            
            # Store modular model info
            self.modular_models[modular_info.name] = modular_info
            
            # Generate single-file implementation
            single_file_path = self.models_dir / f"modeling_{modular_info.name.lower()}.py"
            success = self.converter.convert_modular_to_single_file(
                str(modular_file_path), 
                str(single_file_path)
            )
            
            if success:
                info(f"Modular model '{modular_info.name}' created successfully", "custom_models")
                log_session_event("modular_model_created", 
                                model_name=modular_info.name,
                                base_model=modular_info.base_model,
                                changes_count=len(modular_info.config_changes) + len(modular_info.model_changes))
                return True
            else:
                error(f"Failed to convert modular file to single file", "custom_models")
                return False
                
        except Exception as e:
            error(f"Error creating modular model: {str(e)}", "custom_models", e)
            return False
    
    def get_modular_models(self) -> Dict[str, ModularModelInfo]:
        """Get all registered modular models."""
        return self.modular_models.copy()
    
    def get_modular_model(self, name: str) -> Optional[ModularModelInfo]:
        """Get modular model information by name."""
        return self.modular_models.get(name)
    
    def validate_modular_model(self, modular_info: ModularModelInfo) -> Dict[str, Any]:
        """Validate a modular model configuration."""
        return self.validator.validate_modular_model(modular_info)

    def get_registered_models(self) -> Dict[str, CustomModelInfo]:
        """Get all registered custom models."""
        return self.registered_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[CustomModelInfo]:
        """Get information about a specific model."""
        return self.registered_models.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available custom model names."""
        return list(self.registered_models.keys())
    
    def validate_model_directory(self, model_dir: str) -> Dict[str, Any]:
        """
        Validate a model directory structure.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "files": {}
        }
        
        model_path = Path(model_dir)
        
        if not model_path.exists():
            validation_results["valid"] = False
            validation_results["errors"].append(f"Directory {model_dir} does not exist")
            return validation_results
        
        # Check for required files
        required_files = ["__init__.py", "config.json"]
        optional_files = ["modeling_*.py", "configuration_*.py", "README.md"]
        
        for file_pattern in required_files:
            if file_pattern.endswith("*.py"):
                # Pattern matching for Python files
                matching_files = list(model_path.glob(file_pattern))
                if not matching_files:
                    validation_results["warnings"].append(f"No {file_pattern} files found")
                else:
                    validation_results["files"][file_pattern] = [str(f) for f in matching_files]
            else:
                file_path = model_path / file_pattern
                if file_path.exists():
                    validation_results["files"][file_pattern] = str(file_path)
                else:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Required file {file_pattern} not found")
        
        # Check config.json
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Check for model_type
                if 'model_type' not in config_data:
                    validation_results["warnings"].append("config.json missing 'model_type'")
                
                # Check for auto_map
                if 'auto_map' not in config_data:
                    validation_results["warnings"].append("config.json missing 'auto_map' for AutoClass support")
                
                validation_results["files"]["config.json"] = config_data
                
            except json.JSONDecodeError as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Invalid JSON in config.json: {str(e)}")
        
        return validation_results
    
    def create_legacy_model(self, legacy_info: LegacyModelInfo) -> bool:
        """
        Create a legacy model contribution from LegacyModelInfo.
        
        Args:
            legacy_info: Legacy model information
            
        Returns:
            bool: True if creation successful, False otherwise
        """
        try:
            # Validate legacy model
            validation = self.validator.validate_legacy_model(legacy_info)
            if not validation["valid"]:
                error(f"Legacy model validation failed: {validation['errors']}", "custom_models")
                return False
            
            # Generate model template files
            template_files = self.legacy_converter.generate_model_template(legacy_info)
            if not template_files:
                error("Failed to generate model template files", "custom_models")
                return False
            
            # Create model directory
            model_dir = self.legacy_dir / legacy_info.name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save template files
            for filename, content in template_files.items():
                file_path = model_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Store legacy model info
            self.legacy_models[legacy_info.name] = legacy_info
            
            info(f"Legacy model '{legacy_info.name}' created successfully", "custom_models")
            log_session_event("legacy_model_created", 
                           model_name=legacy_info.name,
                           model_type=legacy_info.model_type,
                           similar_models=legacy_info.similar_models)
            return True
            
        except Exception as e:
            error(f"Error creating legacy model: {str(e)}", "custom_models", e)
            return False
    
    def get_legacy_models(self) -> Dict[str, LegacyModelInfo]:
        """
        Get all legacy models.
        
        Returns:
            Dict of legacy model name -> LegacyModelInfo
        """
        return self.legacy_models.copy()
    
    def get_legacy_model(self, name: str) -> Optional[LegacyModelInfo]:
        """
        Get a specific legacy model.
        
        Args:
            name: Legacy model name
            
        Returns:
            LegacyModelInfo if found, None otherwise
        """
        return self.legacy_models.get(name)
    
    def validate_legacy_model(self, legacy_info: LegacyModelInfo) -> Dict[str, Any]:
        """
        Validate a legacy model configuration.
        
        Args:
            legacy_info: Legacy model information
            
        Returns:
            Dict containing validation results
        """
        return self.validator.validate_legacy_model(legacy_info)
    
    def generate_documentation(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """
        Generate documentation for a model file.
        
        Args:
            file_path: Path to the model file
            output_path: Optional output path for documented file
            
        Returns:
            bool: True if documentation generation successful
        """
        try:
            debug(f"Generating documentation for {file_path}", "custom_models")
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                error(f"File not found: {file_path}", "custom_models")
                return False
            
            # Analyze the file
            docstring_infos = docstring_generator.analyze_python_file(file_path_obj)
            
            # Generate documentation
            documented_content = self._add_documentation_to_file(file_path_obj, docstring_infos)
            
            # Write output
            if output_path:
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path_obj = file_path_obj.with_suffix('.documented.py')
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(documented_content)
            
            info(f"Documentation generated: {output_path_obj}", "custom_models")
            return True
            
        except Exception as e:
            error(f"Error generating documentation: {str(e)}", "custom_models", e)
            return False
    
    def validate_documentation(self, file_path: str) -> Dict[str, Any]:
        """
        Validate documentation in a model file.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Dict containing validation results
        """
        try:
            debug(f"Validating documentation in {file_path}", "custom_models")
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    'valid': False,
                    'errors': [f"File not found: {file_path}"],
                    'warnings': [],
                    'checks': {},
                    'suggestions': []
                }
            
            validation_results = docstring_validator.validate_file(file_path_obj)
            
            info(f"Documentation validation complete for {file_path}", "custom_models")
            return validation_results
            
        except Exception as e:
            error(f"Error validating documentation: {str(e)}", "custom_models", e)
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'checks': {},
                'suggestions': []
            }
    
    def _add_documentation_to_file(self, file_path: Path, docstring_infos: List) -> str:
        """
        Add documentation to a Python file content.
        
        Args:
            file_path: Path to the file
            docstring_infos: List of DocstringInfo objects
            
        Returns:
            Documented file content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # This is a simplified implementation
            # In practice, you'd parse the AST and insert docstrings appropriately
            
            documented_content = content
            
            # Add import for auto_docstring if not present
            if 'from transformers.utils import auto_docstring' not in content:
                import_line = 'from transformers.utils import auto_docstring\n'
                # Find the last import statement
                lines = content.split('\n')
                last_import_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        last_import_idx = i
                
                lines.insert(last_import_idx + 1, import_line)
                documented_content = '\n'.join(lines)
            
            return documented_content
            
        except Exception as e:
            error(f"Error adding documentation: {str(e)}", "custom_models", e)
            return content
    
    def register_attention_function(
        self, 
        name: str, 
        function: Callable, 
        description: str = "",
        dependencies: List[str] = None
    ) -> bool:
        """
        Register a custom attention function.
        
        Args:
            name: Name of the attention function
            function: The attention function implementation
            description: Description of the function
            dependencies: List of dependencies
            
        Returns:
            bool: True if registration successful
        """
        try:
            debug(f"Registering attention function: {name}", "custom_models")
            success = self.attention_manager.register_attention_function(
                name, function, description, dependencies
            )
            
            if success:
                log_session_event("attention_function_registered", 
                               function_name=name,
                               is_custom=True)
            
            return success
            
        except Exception as e:
            error(f"Error registering attention function: {str(e)}", "custom_models", e)
            return False
    
    def register_attention_mask(
        self, 
        name: str, 
        function: Callable, 
        description: str = "",
        compatible_attention_functions: List[str] = None
    ) -> bool:
        """
        Register a custom attention mask function.
        
        Args:
            name: Name of the mask function
            function: The mask function implementation
            description: Description of the function
            compatible_attention_functions: List of compatible attention functions
            
        Returns:
            bool: True if registration successful
        """
        try:
            debug(f"Registering attention mask: {name}", "custom_models")
            success = self.attention_manager.register_attention_mask(
                name, function, description, compatible_attention_functions
            )
            
            if success:
                log_session_event("attention_mask_registered", 
                               mask_name=name,
                               is_custom=True)
            
            return success
            
        except Exception as e:
            error(f"Error registering attention mask: {str(e)}", "custom_models", e)
            return False
    
    def get_attention_functions(self) -> Dict[str, Any]:
        """Get all registered attention functions."""
        try:
            return self.attention_manager.get_attention_functions()
        except Exception as e:
            error(f"Error getting attention functions: {str(e)}", "custom_models", e)
            return {}
    
    def get_attention_masks(self) -> Dict[str, Any]:
        """Get all registered attention masks."""
        try:
            return self.attention_manager.get_attention_masks()
        except Exception as e:
            error(f"Error getting attention masks: {str(e)}", "custom_models", e)
            return {}
    
    def benchmark_attention_functions(
        self, 
        function_names: List[str], 
        model, 
        input_ids: torch.Tensor,
        num_runs: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple attention functions.
        
        Args:
            function_names: List of attention function names to benchmark
            model: The model to test with
            input_ids: Input tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Dict containing benchmark results
        """
        try:
            debug(f"Benchmarking attention functions: {function_names}", "custom_models")
            results = self.attention_manager.compare_attention_functions(
                function_names, model, input_ids
            )
            
            log_session_event("attention_functions_benchmarked", 
                           function_names=function_names,
                           num_runs=num_runs)
            
            return results
            
        except Exception as e:
            error(f"Error benchmarking attention functions: {str(e)}", "custom_models", e)
            return {}
    
    def create_attention_function_template(self, name: str) -> str:
        """
        Create a template for a custom attention function.
        
        Args:
            name: Name for the attention function
            
        Returns:
            String template for the attention function
        """
        try:
            return self.attention_manager.create_attention_function_template(name)
        except Exception as e:
            error(f"Error creating attention function template: {str(e)}", "custom_models", e)
            return ""
    
    def create_attention_mask_template(self, name: str) -> str:
        """
        Create a template for a custom attention mask function.
        
        Args:
            name: Name for the attention mask function
            
        Returns:
            String template for the attention mask function
        """
        try:
            return self.attention_manager.create_attention_mask_template(name)
        except Exception as e:
            error(f"Error creating attention mask template: {str(e)}", "custom_models", e)
            return ""


# Global instance
custom_model_manager = CustomModelManager()
