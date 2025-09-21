"""
Parameter Controls Component.

Handles model parameter configuration UI (temperature, max_tokens, etc.).
"""

import streamlit as st
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import ModelConfig
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation


class ParameterControls:
    """Component for model parameter configuration."""
    
    def __init__(self):
        """Initialize parameter controls."""
        debug("ParameterControls initialized", "parameter_controls")
    
    def render(self, model_config: Optional[Any] = None) -> Dict[str, Any]:
        """
        Render parameter control interface with comprehensive logging and error handling.
        
        This method creates the model parameter configuration interface including
        temperature, max tokens, top-p, and repetition penalty controls. It includes
        detailed logging for debugging and user interaction tracking.
        
        Args:
            model_config: Optional model configuration (ModelConfig object or dict)
            
        Returns:
            Dict containing parameter values with validation and logging
            
        Features:
        - Parameter slider controls with validation
        - Default value extraction from model config
        - Real-time parameter value tracking
        - Error handling for invalid configurations
        - User interaction logging
        
        Logging:
        - Parameter control rendering status
        - Default value extraction and validation
        - User parameter adjustments
        - Configuration validation results
        - Error handling for malformed configs
        """
        debug("Starting parameter controls rendering", "parameter_controls",
              config_provided=bool(model_config),
              config_type=type(model_config).__name__ if model_config else "None")
        
        try:
            with LoggedOperation("parameter_controls_rendering", "parameter_controls"):
                # Render parameter controls header
                st.markdown("### ⚙️ Generation Parameters")
                debug("Parameter controls header rendered", "parameter_controls")
                
                # Extract default values from model config with comprehensive validation
                debug("Extracting default values from model configuration", "parameter_controls")
                
                try:
                    if model_config:
                        if hasattr(model_config, 'temperature'):
                            # ModelConfig object - extract attributes directly
                            debug("Processing ModelConfig object", "parameter_controls",
                                  has_temperature=hasattr(model_config, 'temperature'),
                                  has_max_tokens=hasattr(model_config, 'max_tokens'),
                                  has_top_p=hasattr(model_config, 'top_p'),
                                  has_repetition_penalty=hasattr(model_config, 'repetition_penalty'))
                            
                            default_temp = getattr(model_config, 'temperature', 0.7)
                            default_tokens = getattr(model_config, 'max_tokens', 512)
                            default_top_p = getattr(model_config, 'top_p', 0.9)
                            default_rep_penalty = getattr(model_config, 'repetition_penalty', 1.1)
                            
                        elif isinstance(model_config, dict):
                            # Dictionary configuration - use get method
                            debug("Processing dictionary configuration", "parameter_controls",
                                  dict_keys=list(model_config.keys()))
                            
                            default_temp = model_config.get('temperature', 0.7)
                            default_tokens = model_config.get('max_tokens', 512)
                            default_top_p = model_config.get('top_p', 0.9)
                            default_rep_penalty = model_config.get('repetition_penalty', 1.1)
                            
                        else:
                            warning("Unknown model config type, using defaults", "parameter_controls",
                                   config_type=type(model_config).__name__)
                            default_temp = 0.7
                            default_tokens = 512
                            default_top_p = 0.9
                            default_rep_penalty = 1.1
                    else:
                        # No model config provided, use defaults
                        debug("No model config provided, using default values", "parameter_controls")
                        default_temp = 0.7
                        default_tokens = 512
                        default_top_p = 0.9
                        default_rep_penalty = 1.1
                    
                    debug("Default values extracted successfully", "parameter_controls",
                          temperature=default_temp,
                          max_tokens=default_tokens,
                          top_p=default_top_p,
                          repetition_penalty=default_rep_penalty)
                    
                except Exception as e:
                    error("Error extracting default values from model config", "parameter_controls", e,
                          config_type=type(model_config).__name__ if model_config else "None")
                    
                    # Fallback to safe defaults
                    warning("Using fallback default values due to extraction error", "parameter_controls")
                    default_temp = 0.7
                    default_tokens = 512
                    default_top_p = 0.9
                    default_rep_penalty = 1.1
                
                # Validate default values
                defaults = {
                    'temperature': default_temp,
                    'max_tokens': default_tokens,
                    'top_p': default_top_p,
                    'repetition_penalty': default_rep_penalty
                }
                
                validated_defaults = self._validate_parameter_defaults(defaults)
                debug("Parameter defaults validated", "parameter_controls",
                      original_defaults=defaults,
                      validated_defaults=validated_defaults)
                
                # Render parameter controls with error handling
                parameters = {}
                
                try:
                    # Temperature slider
                    debug("Rendering temperature slider", "parameter_controls")
                    temperature = st.slider(
                        "🌡️ Temperature:",
                        min_value=0.1,
                        max_value=2.0,
                        value=validated_defaults['temperature'],
                        step=0.1,
                        help="Controls randomness (lower = more focused, higher = more creative)",
                        key="param_temperature"
                    )
                    parameters['temperature'] = temperature
                    debug("Temperature slider rendered", "parameter_controls", value=temperature)
                    
                except Exception as e:
                    error("Error rendering temperature slider", "parameter_controls", e)
                    temperature = validated_defaults['temperature']
                    parameters['temperature'] = temperature
                
                try:
                    # Max tokens slider
                    debug("Rendering max tokens slider", "parameter_controls")
                    max_tokens = st.slider(
                        "📝 Max Tokens:",
                        min_value=50,
                        max_value=1000,
                        value=validated_defaults['max_tokens'],
                        help="Maximum length of generated response",
                        key="param_max_tokens"
                    )
                    parameters['max_tokens'] = max_tokens
                    debug("Max tokens slider rendered", "parameter_controls", value=max_tokens)
                    
                except Exception as e:
                    error("Error rendering max tokens slider", "parameter_controls", e)
                    max_tokens = validated_defaults['max_tokens']
                    parameters['max_tokens'] = max_tokens
                
                try:
                    # Top-p slider
                    debug("Rendering top-p slider", "parameter_controls")
                    top_p = st.slider(
                        "🎯 Top-p:",
                        min_value=0.1,
                        max_value=1.0,
                        value=validated_defaults['top_p'],
                        step=0.1,
                        help="Controls diversity (lower = more focused)",
                        key="param_top_p"
                    )
                    parameters['top_p'] = top_p
                    debug("Top-p slider rendered", "parameter_controls", value=top_p)
                    
                except Exception as e:
                    error("Error rendering top-p slider", "parameter_controls", e)
                    top_p = validated_defaults['top_p']
                    parameters['top_p'] = top_p
                
                try:
                    # Repetition penalty slider
                    debug("Rendering repetition penalty slider", "parameter_controls")
                    repetition_penalty = st.slider(
                        "🔄 Repetition Penalty:",
                        min_value=1.0,
                        max_value=2.0,
                        value=validated_defaults['repetition_penalty'],
                        step=0.1,
                        help="Penalty for repeating text",
                        key="param_repetition_penalty"
                    )
                    parameters['repetition_penalty'] = repetition_penalty
                    debug("Repetition penalty slider rendered", "parameter_controls", value=repetition_penalty)
                    
                except Exception as e:
                    error("Error rendering repetition penalty slider", "parameter_controls", e)
                    repetition_penalty = validated_defaults['repetition_penalty']
                    parameters['repetition_penalty'] = repetition_penalty
                
                debug("All parameter controls rendered successfully", "parameter_controls",
                      parameters=parameters)
                
                # Log parameter configuration for analytics
                log_user_action("parameter_controls_rendered",
                              temperature=parameters['temperature'],
                              max_tokens=parameters['max_tokens'],
                              top_p=parameters['top_p'],
                              repetition_penalty=parameters['repetition_penalty'],
                              config_provided=bool(model_config))
                
                return parameters
                
        except Exception as e:
            error("Critical error in parameter controls rendering", "parameter_controls", e,
                  config_provided=bool(model_config))
            st.error("🚨 An error occurred while loading parameter controls.")
            log_user_action("parameter_controls_failed", error=str(e))
            
            # Return safe fallback parameters
            return {
                'temperature': 0.7,
                'max_tokens': 512,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            }
        
        debug("Parameter controls rendering completed", "parameter_controls")
    
    def _validate_parameter_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize parameter default values.
        
        Args:
            defaults: Dictionary of parameter default values
            
        Returns:
            Dict containing validated and sanitized default values
        """
        debug("Validating parameter defaults", "parameter_controls", defaults=defaults)
        
        validated = {}
        
        # Validate temperature
        temp = defaults.get('temperature', 0.7)
        validated['temperature'] = max(0.1, min(2.0, float(temp)))
        
        # Validate max tokens
        tokens = defaults.get('max_tokens', 512)
        validated['max_tokens'] = max(50, min(1000, int(tokens)))
        
        # Validate top-p
        top_p = defaults.get('top_p', 0.9)
        validated['top_p'] = max(0.1, min(1.0, float(top_p)))
        
        # Validate repetition penalty
        rep_penalty = defaults.get('repetition_penalty', 1.1)
        validated['repetition_penalty'] = max(1.0, min(2.0, float(rep_penalty)))
        
        debug("Parameter defaults validation completed", "parameter_controls",
              original=defaults,
              validated=validated)
        
        return validated
