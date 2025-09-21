"""
Component Manager for handling custom Transformers model components.

This module provides comprehensive support for:
- Custom attention mechanism creation and replacement
- LoRA (Low-Rank Adaptation) integration
- Component customization and validation
- Model component replacement strategies
- Performance optimization for custom components

Key Classes:
- ComponentManager: Main interface for component operations
- AttentionCustomizer: Specialized attention mechanism customization
- LoRAManager: LoRA integration and management
- ComponentValidator: Validation utilities for custom components
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
import torch.nn as nn
from transformers import (
    PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel,
    AutoModelForCausalLM, AutoModelForSequenceClassification
)

from .logger import debug, info, warning, error, log_session_event


@dataclass
class ComponentInfo:
    """Information about a custom component."""
    name: str
    component_type: str
    original_class: str
    custom_class: str
    description: str
    author: str
    version: str = "1.0.0"
    tags: List[str] = None
    dependencies: List[str] = None
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.target_modules is None:
            self.target_modules = []


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    task_type: str = "FEATURE_EXTRACTION"
    bias: str = "none"
    fan_in_fan_out: bool = False
    inference_mode: bool = False
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q", "v"]


class ComponentValidator:
    """Validator for custom component implementations."""
    
    @staticmethod
    def validate_attention_component(attention_class: Type[nn.Module]) -> Dict[str, Any]:
        """
        Validate a custom attention component.
        
        Args:
            attention_class: Custom attention class
            
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
            # Check if it's a PyTorch module
            if not issubclass(attention_class, nn.Module):
                validation_results["valid"] = False
                validation_results["errors"].append("Attention component must inherit from nn.Module")
            else:
                validation_results["checks"]["inherits_nn_module"] = True
            
            # Check for required methods
            required_methods = ["__init__", "forward"]
            for method_name in required_methods:
                if hasattr(attention_class, method_name):
                    validation_results["checks"][f"has_{method_name}"] = True
                else:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Missing required method: {method_name}")
            
            # Check for QKV projections
            qkv_components = ["q", "k", "v"]
            has_qkv = all(hasattr(attention_class, comp) for comp in qkv_components)
            validation_results["checks"]["has_qkv_projections"] = has_qkv
            
            if not has_qkv:
                validation_results["warnings"].append("Custom attention should have separate q, k, v projections for LoRA compatibility")
            
            # Test instantiation
            try:
                # Create a mock config for testing
                class MockConfig:
                    hidden_size = 768
                    num_attention_heads = 12
                    qkv_bias = True
                    attention_dropout = 0.1
                
                test_config = MockConfig()
                test_instance = attention_class(test_config)
                validation_results["checks"]["instantiation"] = True
                
                # Test forward pass with dummy input
                dummy_input = torch.randn(1, 10, 768)
                try:
                    output = test_instance(dummy_input)
                    validation_results["checks"]["forward_pass"] = True
                except Exception as e:
                    validation_results["warnings"].append(f"Forward pass test failed: {str(e)}")
                
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Component instantiation failed: {str(e)}")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def validate_lora_config(lora_config: LoRAConfig) -> Dict[str, Any]:
        """
        Validate LoRA configuration.
        
        Args:
            lora_config: LoRA configuration
            
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
            # Validate rank
            if lora_config.r <= 0:
                validation_results["valid"] = False
                validation_results["errors"].append("LoRA rank (r) must be positive")
            elif lora_config.r > 1024:
                validation_results["warnings"].append("Very high LoRA rank may lead to overfitting")
            else:
                validation_results["checks"]["valid_rank"] = True
            
            # Validate alpha
            if lora_config.lora_alpha <= 0:
                validation_results["valid"] = False
                validation_results["errors"].append("LoRA alpha must be positive")
            else:
                validation_results["checks"]["valid_alpha"] = True
            
            # Validate dropout
            if not 0 <= lora_config.lora_dropout <= 1:
                validation_results["valid"] = False
                validation_results["errors"].append("LoRA dropout must be between 0 and 1")
            else:
                validation_results["checks"]["valid_dropout"] = True
            
            # Validate target modules
            if not lora_config.target_modules:
                validation_results["valid"] = False
                validation_results["errors"].append("Target modules cannot be empty")
            else:
                validation_results["checks"]["has_target_modules"] = True
            
            # Check for common target modules
            common_targets = ["q", "k", "v", "q_proj", "k_proj", "v_proj", "o_proj"]
            has_common_targets = any(target in lora_config.target_modules for target in common_targets)
            if has_common_targets:
                validation_results["checks"]["has_common_targets"] = True
            else:
                validation_results["warnings"].append("No common attention projection targets found")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"LoRA validation error: {str(e)}")
        
        return validation_results


class AttentionCustomizer:
    """Specialized attention mechanism customizer."""
    
    def __init__(self):
        """Initialize the attention customizer."""
        self.custom_attentions: Dict[str, Type[nn.Module]] = {}
        info("AttentionCustomizer initialized", "component_manager")
    
    def register_custom_attention(self, name: str, attention_class: Type[nn.Module]) -> bool:
        """
        Register a custom attention mechanism.
        
        Args:
            name: Name for the custom attention
            attention_class: Custom attention class
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Validate the attention component
            validator = ComponentValidator()
            validation_results = validator.validate_attention_component(attention_class)
            
            if not validation_results["valid"]:
                error(f"Attention validation failed: {validation_results['errors']}", "component_manager")
                return False
            
            # Register the attention class
            self.custom_attentions[name] = attention_class
            
            info(f"Custom attention '{name}' registered successfully", "component_manager")
            log_session_event("custom_attention_registered", 
                            attention_name=name,
                            validation_passed=True)
            
            return True
            
        except Exception as e:
            error(f"Failed to register custom attention: {str(e)}", "component_manager", e)
            return False
    
    def create_split_attention(self, config, window_size=None):
        """
        Create a custom attention with split QKV projections.
        
        Args:
            config: Model configuration
            window_size: Optional window size for windowed attention
            
        Returns:
            Custom attention instance
        """
        class SplitAttention(nn.Module):
            """Custom attention with separate Q, K, V projections."""
            
            def __init__(self, config, window_size=None):
                super().__init__()
                
                self.config = config
                self.window_size = window_size
                
                # Attention parameters
                self.hidden_size = getattr(config, 'hidden_size', 768)
                self.num_attention_heads = getattr(config, 'num_attention_heads', 12)
                self.attention_head_size = self.hidden_size // self.num_attention_heads
                self.all_head_size = self.num_attention_heads * self.attention_head_size
                
                # Separate Q, K, V projections
                self.q = nn.Linear(self.hidden_size, self.all_head_size, bias=getattr(config, 'qkv_bias', True))
                self.k = nn.Linear(self.hidden_size, self.all_head_size, bias=getattr(config, 'qkv_bias', True))
                self.v = nn.Linear(self.hidden_size, self.all_head_size, bias=getattr(config, 'qkv_bias', True))
                
                # Output projection
                self.proj = nn.Linear(self.all_head_size, self.hidden_size)
                
                # Dropout and scaling
                self.dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.1))
                self.scale = self.attention_head_size ** -0.5
                
                # Register hook for weight loading
                self._register_load_state_dict_pre_hook(self._split_qkv_load_hook)
            
            def _split_qkv_load_hook(self, state_dict, prefix, *args):
                """
                Hook to handle loading of pretrained QKV weights.
                
                This hook automatically converts fused QKV weight matrices from pretrained
                models into separate Q, K, V projections for LoRA compatibility.
                
                Process:
                1. Identifies fused QKV weight tensors in the state dict
                2. Splits them into separate Q, K, V components
                3. Updates state dict with separated weights
                4. Removes original fused weights to prevent conflicts
                
                This enables loading pretrained models that use fused QKV layers
                into custom attention implementations with separate projections.
                """
                debug("Starting QKV weight splitting hook", "component_manager")
                
                keys_to_delete = []
                keys_processed = 0
                
                # Scan through all state dict keys to find QKV weights
                for key in list(state_dict.keys()):
                    if "qkv." in key:
                        debug(f"Found QKV weight to split: {key}", "component_manager",
                              weight_shape=state_dict[key].shape)
                        
                        # Extract the fused QKV weight tensor
                        qkv_weight = state_dict[key]
                        
                        # Split into Q, K, V components along the first dimension
                        # Assumes QKV weights are concatenated along dim=0
                        q, k, v = qkv_weight.chunk(3, dim=0)
                        debug(f"Split QKV weights", "component_manager",
                              q_shape=q.shape, k_shape=k.shape, v_shape=v.shape)
                        
                        # Create new state dict entries for separated weights
                        state_dict[key.replace("qkv.", "q.")] = q
                        state_dict[key.replace("qkv.", "k.")] = k
                        state_dict[key.replace("qkv.", "v.")] = v
                        
                        # Mark original fused weight for deletion
                        keys_to_delete.append(key)
                        keys_processed += 1
                
                # Remove original fused QKV weights from state dict
                for key in keys_to_delete:
                    debug(f"Removing original QKV weight: {key}", "component_manager")
                    del state_dict[key]
                
                debug(f"QKV weight splitting completed", "component_manager",
                      keys_processed=keys_processed,
                      keys_deleted=len(keys_to_delete))
            
            def forward(self, hidden_states, attention_mask=None, output_attentions=False):
                """
                Forward pass of the split attention mechanism.
                
                This method implements multi-head attention with separate Q, K, V projections
                for enhanced LoRA compatibility and customization flexibility.
                
                Args:
                    hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
                    attention_mask: Optional attention mask [batch_size, seq_len, seq_len]
                    output_attentions: Whether to return attention weights
                
                Returns:
                    Tuple of (attention_output, attention_weights)
                """
                # Extract input dimensions for processing
                batch_size, seq_len, _ = hidden_states.shape
                debug("Starting split attention forward pass", "component_manager",
                      batch_size=batch_size, seq_len=seq_len,
                      num_heads=self.num_attention_heads)
                
                # Step 1: Compute Q, K, V projections using separate linear layers
                # This separation enables LoRA fine-tuning on individual projections
                debug("Computing Q, K, V projections", "component_manager")
                query = self.q(hidden_states)  # [batch_size, seq_len, all_head_size]
                key = self.k(hidden_states)    # [batch_size, seq_len, all_head_size]
                value = self.v(hidden_states)  # [batch_size, seq_len, all_head_size]
                
                # Step 2: Reshape for multi-head attention computation
                # Split the all_head_size into num_heads × head_size
                debug("Reshaping for multi-head attention", "component_manager")
                query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
                key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
                value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
                
                # Step 3: Transpose for efficient attention computation
                # Move head dimension to position 1: [batch_size, num_heads, seq_len, head_size]
                debug("Transposing tensors for attention computation", "component_manager")
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                
                # Step 4: Compute scaled dot-product attention scores
                # Scale by sqrt(head_size) to prevent softmax saturation
                debug("Computing attention scores", "component_manager", scale=self.scale)
                attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
                
                # Step 5: Apply attention mask if provided
                # Mask prevents attention to certain positions (e.g., padding tokens)
                if attention_mask is not None:
                    debug("Applying attention mask", "component_manager",
                          mask_shape=attention_mask.shape)
                    attention_scores = attention_scores + attention_mask
                
                # Step 6: Apply softmax to get attention probabilities
                debug("Applying softmax to attention scores", "component_manager")
                attention_probs = nn.functional.softmax(attention_scores, dim=-1)
                
                # Step 7: Apply dropout for regularization
                attention_probs = self.dropout(attention_probs)
                debug("Applied dropout to attention probabilities", "component_manager")
                
                # Step 8: Apply attention weights to values
                debug("Applying attention weights to values", "component_manager")
                attention_output = torch.matmul(attention_probs, value)
                
                # Step 9: Reshape and apply output projection
                # Transpose back and reshape to original dimensions
                debug("Reshaping and applying output projection", "component_manager")
                attention_output = attention_output.transpose(1, 2).contiguous()
                attention_output = attention_output.view(batch_size, seq_len, self.all_head_size)
                
                # Final linear projection to output space
                attention_output = self.proj(attention_output)
                
                debug("Split attention forward pass completed", "component_manager",
                      output_shape=attention_output.shape,
                      return_attentions=output_attentions)
                
                # Return output with optional attention weights
                if output_attentions:
                    return (attention_output, attention_probs)
                else:
                    return (attention_output, None)
        
        return SplitAttention(config, window_size)
    
    def get_registered_attentions(self) -> List[str]:
        """Get list of registered custom attention mechanisms."""
        return list(self.custom_attentions.keys())
    
    def get_attention_class(self, name: str) -> Optional[Type[nn.Module]]:
        """Get a registered attention class by name."""
        return self.custom_attentions.get(name)


class LoRAManager:
    """Manager for LoRA (Low-Rank Adaptation) integration."""
    
    def __init__(self):
        """Initialize the LoRA manager."""
        self.lora_configs: Dict[str, LoRAConfig] = {}
        info("LoRAManager initialized", "component_manager")
    
    def create_lora_config(
        self, 
        name: str,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: List[str] = None,
        lora_dropout: float = 0.1,
        task_type: str = "FEATURE_EXTRACTION"
    ) -> LoRAConfig:
        """
        Create a LoRA configuration.
        
        Args:
            name: Name for the LoRA configuration
            r: Rank of adaptation
            lora_alpha: Scaling factor
            target_modules: Modules to apply LoRA to
            lora_dropout: Dropout rate
            task_type: Type of task
            
        Returns:
            LoRA configuration
        """
        lora_config = LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["q", "v"],
            lora_dropout=lora_dropout,
            task_type=task_type
        )
        
        # Validate the configuration
        validator = ComponentValidator()
        validation_results = validator.validate_lora_config(lora_config)
        
        if validation_results["valid"]:
            self.lora_configs[name] = lora_config
            info(f"LoRA configuration '{name}' created successfully", "component_manager")
        else:
            warning(f"LoRA configuration validation failed: {validation_results['errors']}", "component_manager")
        
        return lora_config
    
    def apply_lora_to_model(self, model, lora_config: LoRAConfig) -> Any:
        """
        Apply LoRA to a model.
        
        Args:
            model: Model to apply LoRA to
            lora_config: LoRA configuration
            
        Returns:
            Model with LoRA applied
        """
        try:
            # Check if PEFT is available
            try:
                from peft import LoraConfig, get_peft_model
                
                # Create PEFT LoRA configuration
                peft_config = LoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    target_modules=lora_config.target_modules,
                    lora_dropout=lora_config.lora_dropout,
                    task_type=lora_config.task_type,
                    bias=lora_config.bias,
                    fan_in_fan_out=lora_config.fan_in_fan_out,
                    inference_mode=lora_config.inference_mode
                )
                
                # Apply LoRA to model
                model_with_lora = get_peft_model(model, peft_config)
                
                info("LoRA applied to model successfully", "component_manager")
                return model_with_lora
                
            except ImportError:
                warning("PEFT library not available. LoRA cannot be applied.", "component_manager")
                return model
                
        except Exception as e:
            error(f"Failed to apply LoRA to model: {str(e)}", "component_manager", e)
            return model
    
    def get_lora_config(self, name: str) -> Optional[LoRAConfig]:
        """Get a LoRA configuration by name."""
        return self.lora_configs.get(name)
    
    def list_lora_configs(self) -> List[str]:
        """List all LoRA configuration names."""
        return list(self.lora_configs.keys())


class ComponentManager:
    """
    Main manager for component customization operations.
    
    This class handles:
    - Component registration and validation
    - Attention mechanism customization
    - LoRA integration
    - Component replacement strategies
    """
    
    def __init__(self):
        """Initialize the component manager."""
        self.attention_customizer = AttentionCustomizer()
        self.lora_manager = LoRAManager()
        self.validator = ComponentValidator()
        info("ComponentManager initialized", "component_manager")
    
    def replace_model_components(
        self, 
        model: Any, 
        component_type: str,
        custom_component_class: Type[nn.Module],
        config: Any
    ) -> int:
        """
        Replace components in a model with custom implementations.
        
        Args:
            model: Model to modify
            component_type: Type of component to replace (e.g., 'attention')
            custom_component_class: Custom component class
            config: Model configuration
            
        Returns:
            Number of components replaced
        """
        replacements_made = 0
        
        try:
            def replace_in_module(module, name, parent):
                nonlocal replacements_made
                
                # Check if this module has the component type we want to replace
                if hasattr(module, component_type):
                    old_component = getattr(module, component_type)
                    new_component = custom_component_class(config)
                    
                    # Copy relevant attributes
                    self._copy_component_attributes(old_component, new_component)
                    
                    # Replace the component
                    setattr(module, component_type, new_component)
                    replacements_made += 1
                    debug(f"Replaced {component_type} in {name}", "component_manager")
                
                # Recursively process child modules
                for child_name, child_module in module.named_children():
                    replace_in_module(child_module, f"{name}.{child_name}", module)
            
            # Start replacement from root
            replace_in_module(model, "model", None)
            
            info(f"Replaced {replacements_made} {component_type} components", "component_manager")
            log_session_event("components_replaced", 
                            component_type=component_type,
                            count=replacements_made)
            
        except Exception as e:
            error(f"Failed to replace components: {str(e)}", "component_manager", e)
        
        return replacements_made
    
    def _copy_component_attributes(self, old_component: nn.Module, new_component: nn.Module):
        """Copy relevant attributes from old component to new component."""
        attributes_to_copy = [
            'num_attention_heads', 'attention_head_size', 'all_head_size',
            'hidden_size', 'dropout', 'scale'
        ]
        
        for attr in attributes_to_copy:
            if hasattr(old_component, attr):
                setattr(new_component, attr, getattr(old_component, attr))
    
    def create_attention_with_lora(
        self, 
        config: Any,
        lora_config: LoRAConfig,
        window_size=None
    ) -> nn.Module:
        """
        Create a custom attention mechanism with LoRA applied.
        
        Args:
            config: Model configuration
            lora_config: LoRA configuration
            window_size: Optional window size
            
        Returns:
            Custom attention with LoRA
        """
        try:
            # Create split attention
            attention = self.attention_customizer.create_split_attention(config, window_size)
            
            # Apply LoRA if PEFT is available
            try:
                from peft import LoraConfig as PEFTLoraConfig, get_peft_model
                
                peft_config = PEFTLoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    target_modules=lora_config.target_modules,
                    lora_dropout=lora_config.lora_dropout,
                    task_type=lora_config.task_type
                )
                
                # Apply LoRA to attention
                attention_with_lora = get_peft_model(attention, peft_config)
                
                info("Created attention with LoRA successfully", "component_manager")
                return attention_with_lora
                
            except ImportError:
                warning("PEFT not available, returning attention without LoRA", "component_manager")
                return attention
                
        except Exception as e:
            error(f"Failed to create attention with LoRA: {str(e)}", "component_manager", e)
            return self.attention_customizer.create_split_attention(config, window_size)
    
    def validate_component(self, component_class: Type[nn.Module]) -> Dict[str, Any]:
        """Validate a custom component."""
        return self.validator.validate_attention_component(component_class)
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about available components."""
        return {
            "custom_attentions": self.attention_customizer.get_registered_attentions(),
            "lora_configs": self.lora_manager.list_lora_configs(),
            "total_custom_components": len(self.attention_customizer.get_registered_attentions())
        }


# Global instance
component_manager = ComponentManager()
