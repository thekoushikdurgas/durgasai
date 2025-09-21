"""
Auto Classes Integration Module for DurgasAI.

This module provides integration between the existing DurgasAI model management
system and the new Auto Classes approach from HuggingFace Transformers.
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForImageTextToText,
    AutoProcessor
)

# Import existing DurgasAI components
from .model_manager import ModelManager, ModelResponse, ModelConfig
from .model_manager import (
    EnhancedModelRegistry,
    EnhancedModelConfig,
    AutoModelFactory,
    TaskType,
    ModelProvider,
    ModelLoadResult
)
from .logger import debug, info, error, log_model_operation


class AutoClassesModelAdapter:
    """
    Adapter that bridges the existing ModelManager with Auto Classes functionality.
    
    This class allows gradual migration to Auto Classes while maintaining
    compatibility with existing code.
    """
    
    def __init__(self, model_manager: ModelManager):
        """Initialize the adapter with existing model manager."""
        self.model_manager = model_manager
        self.enhanced_registry = EnhancedModelRegistry()
        self._setup_enhanced_models()
    
    def _setup_enhanced_models(self) -> None:
        """Set up enhanced model configurations from existing configs."""
        try:
            # Load existing model configurations
            config_path = Path("config/model_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    existing_configs = json.load(f)
                
                # Convert existing configs to enhanced format
                for model_key, model_data in existing_configs.get('models', {}).items():
                    enhanced_config = self._convert_to_enhanced_config(model_key, model_data)
                    self.enhanced_registry.register_model(model_key, enhanced_config)
                    
                info(f"Converted {len(existing_configs.get('models', {}))} models to enhanced format", "auto_classes")
            
        except Exception as e:
            error(f"Failed to setup enhanced models: {str(e)}", "auto_classes", e)
    
    def _convert_to_enhanced_config(self, model_key: str, model_data: Dict[str, Any]) -> EnhancedModelConfig:
        """
        Convert existing model configuration to enhanced format.
        
        This method transforms legacy model configurations into the new Enhanced Model Config
        format that supports Auto Classes and advanced features. The conversion process
        includes intelligent inference of model capabilities and optimal settings.
        
        Args:
            model_key: Unique identifier for the model
            model_data: Legacy model configuration data
            
        Returns:
            EnhancedModelConfig: Converted configuration with Auto Classes support
        """
        debug(f"Converting model config to enhanced format: {model_key}", "auto_classes",
              model_id=model_data.get('model_id', 'unknown'),
              original_provider=model_data.get('provider', 'unknown'))
        
        # Step 1: Determine task type based on model characteristics
        # This enables automatic selection of the appropriate Auto Class
        debug("Inferring task type from model data", "auto_classes")
        task_type = self._infer_task_type(model_data)
        debug(f"Inferred task type: {task_type}", "auto_classes")
        
        # Step 2: Determine provider mapping
        # Convert legacy provider strings to new enum values
        provider_str = model_data.get('provider', 'huggingface_api')
        debug(f"Converting provider: {provider_str}", "auto_classes")
        
        provider = ModelProvider.HUGGINGFACE_HUB if provider_str == 'huggingface_api' else ModelProvider.HUGGINGFACE_LOCAL
        debug(f"Mapped to provider: {provider}", "auto_classes")
        
        # Step 3: Import required classes for enhanced configuration
        debug("Importing enhanced model manager classes", "auto_classes")
        from utils.model_manager import LoadingOptions, ModelCapabilities, ResourceRequirements
        
        # Step 4: Create enhanced configuration with comprehensive settings
        debug("Creating enhanced model configuration", "auto_classes")
        
        # Extract model parameters with defaults
        max_tokens = model_data.get('max_tokens', 512)
        temperature = model_data.get('temperature', 0.7)
        top_p = model_data.get('top_p', 0.9)
        repetition_penalty = model_data.get('repetition_penalty', 1.1)
        
        debug("Model parameters extracted", "auto_classes",
              max_tokens=max_tokens, temperature=temperature,
              top_p=top_p, repetition_penalty=repetition_penalty)
        
        # Create enhanced configuration object
        enhanced_config = EnhancedModelConfig(
            model_id=model_data['model_id'],
            display_name=model_data.get('name', model_key),
            provider=provider,
            task_type=task_type,
            description=model_data.get('description', ''),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            loading_options=LoadingOptions(),  # Use default optimized loading options
            capabilities=ModelCapabilities(
                text_generation=True,  # Assume text generation capability
                conversation=True,     # Assume conversation capability
                multimodal=model_data.get('supports_images', False)  # Check for image support
            ),
            resource_requirements=ResourceRequirements(
                min_memory_gb=2.0,      # Conservative minimum memory requirement
                recommended_memory_gb=4.0  # Recommended memory for optimal performance
            ),
            tags=model_data.get('use_cases', []),      # Use cases as tags
            use_cases=model_data.get('use_cases', []), # Preserve use cases
            enabled=model_data.get('enabled', True)    # Default to enabled
        )
        
        debug(f"Enhanced configuration created successfully", "auto_classes",
              model_name=enhanced_config.display_name,
              task_type=enhanced_config.task_type.value,
              provider=enhanced_config.provider.value)
        
        return enhanced_config
    
    def _infer_task_type(self, model_data: Dict[str, Any]) -> TaskType:
        """
        Infer task type from model data using pattern matching.
        
        This method analyzes the model ID and metadata to determine the most
        appropriate task type for Auto Class selection. It uses pattern matching
        on model names and architectures to classify models correctly.
        
        Args:
            model_data: Model configuration data containing model_id and metadata
            
        Returns:
            TaskType: Inferred task type for the model
        """
        model_id = model_data.get('model_id', '').lower()
        debug(f"Inferring task type for model: {model_id}", "auto_classes")
        
        # Pattern matching for different model types
        # Each pattern corresponds to specific model architectures and their capabilities
        
        # Multimodal models (image + text processing)
        multimodal_patterns = ['idefics', 'llava', 'blip']
        for pattern in multimodal_patterns:
            if pattern in model_id:
                debug(f"Matched multimodal pattern: {pattern}", "auto_classes",
                      model_id=model_id, task_type="IMAGE_TEXT_TO_TEXT")
                return TaskType.IMAGE_TEXT_TO_TEXT
        
        # Sequence-to-sequence models (encoder-decoder architectures)
        seq2seq_patterns = ['t5', 'bart', 'pegasus']
        for pattern in seq2seq_patterns:
            if pattern in model_id:
                debug(f"Matched seq2seq pattern: {pattern}", "auto_classes",
                      model_id=model_id, task_type="SEQ2SEQ")
                return TaskType.SEQ2SEQ
        
        # Speech processing models
        speech_patterns = ['whisper', 'speech']
        for pattern in speech_patterns:
            if pattern in model_id:
                debug(f"Matched speech pattern: {pattern}", "auto_classes",
                      model_id=model_id, task_type="SPEECH_SEQ2SEQ")
                return TaskType.SPEECH_SEQ2SEQ
        
        # Classification models
        classification_patterns = ['sentiment', 'classification', 'classifier']
        for pattern in classification_patterns:
            if pattern in model_id:
                debug(f"Matched classification pattern: {pattern}", "auto_classes",
                      model_id=model_id, task_type="CLASSIFICATION")
                return TaskType.CLASSIFICATION
        
        # Default to text generation for unmatched models
        # This is the most common task type and works for most language models
        debug(f"No specific pattern matched, defaulting to text generation", "auto_classes",
              model_id=model_id, task_type="TEXT_GENERATION")
        return TaskType.TEXT_GENERATION
    
    def load_model_with_auto_classes(self, model_key: str) -> Optional[ModelLoadResult]:
        """Load a model using Auto Classes approach."""
        try:
            debug(f"Loading model with Auto Classes: {model_key}", "auto_classes")
            
            # Check if model is registered in enhanced registry
            if model_key not in self.enhanced_registry.configs:
                error(f"Model {model_key} not found in enhanced registry", "auto_classes")
                return None
            
            # Load using enhanced registry
            result = self.enhanced_registry.load_model(model_key)
            
            if result and result.success:
                info(f"Successfully loaded {model_key} using {result.auto_class_used}", "auto_classes")
                log_model_operation("auto_classes_load_success", model_key,
                    auto_class=result.auto_class_used,
                    model_type=result.model_type,
                    loading_time=result.loading_time,
                    memory_usage_mb=result.memory_usage_mb)
            else:
                error(f"Failed to load {model_key} with Auto Classes", "auto_classes")
                log_model_operation("auto_classes_load_failed", model_key,
                    error=result.error if result else "Unknown error")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in Auto Classes loading for {model_key}: {str(e)}"
            error(error_msg, "auto_classes", e)
            return None
    
    def get_model_info_enhanced(self, model_key: str) -> Dict[str, Any]:
        """Get enhanced model information including Auto Classes details."""
        base_info = self.enhanced_registry.get_model_info(model_key)
        
        # Add compatibility information
        base_info['auto_classes_compatible'] = model_key in self.enhanced_registry.configs
        base_info['legacy_compatible'] = True  # All models are backward compatible
        
        return base_info
    
    def list_auto_classes_models(self) -> List[str]:
        """List all models that support Auto Classes."""
        return self.enhanced_registry.list_models()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including Auto Classes info."""
        base_status = self.enhanced_registry.get_system_info()
        
        # Add integration status
        base_status.update({
            'auto_classes_enabled': True,
            'enhanced_models_count': len(self.enhanced_registry.configs),
            'loaded_enhanced_models': len(self.enhanced_registry.models),
            'legacy_models_count': len(self.model_manager._models) if hasattr(self.model_manager, '_models') else 0
        })
        
        return base_status


class AutoClassesResponseGenerator:
    """
    Enhanced response generator that uses Auto Classes models.
    
    This class provides improved response generation capabilities
    using the Auto Classes approach for better model handling.
    """
    
    def __init__(self, adapter: AutoClassesModelAdapter):
        """Initialize with Auto Classes adapter."""
        self.adapter = adapter
    
    def generate_response(
        self, 
        model_key: str, 
        prompt: str, 
        **generation_kwargs
    ) -> ModelResponse:
        """Generate response using Auto Classes model."""
        start_time = time.time()
        
        try:
            debug(f"Generating response with Auto Classes model: {model_key}", "auto_classes")
            
            # Get loaded model
            model_result = self.adapter.enhanced_registry.get_model(model_key)
            
            if not model_result or not model_result.success:
                # Try to load the model
                model_result = self.adapter.load_model_with_auto_classes(model_key)
                
                if not model_result or not model_result.success:
                    return ModelResponse(
                        content="",
                        success=False,
                        error=f"Failed to load model {model_key}",
                        metadata={"model_key": model_key, "error_type": "model_loading"}
                    )
            
            # Generate response based on model type
            response_content = self._generate_with_model(
                model_result, prompt, **generation_kwargs
            )
            
            generation_time = time.time() - start_time
            
            # Log successful generation
            log_model_operation("auto_classes_generation_success", model_key,
                generation_time=generation_time,
                prompt_length=len(prompt),
                response_length=len(response_content))
            
            return ModelResponse(
                content=response_content,
                success=True,
                metadata={
                    "model_key": model_key,
                    "model_type": model_result.model_type,
                    "auto_class": model_result.auto_class_used,
                    "generation_time": generation_time,
                    "prompt_length": len(prompt),
                    "response_length": len(response_content)
                }
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Error generating response with {model_key}: {str(e)}"
            error(error_msg, "auto_classes", e)
            
            log_model_operation("auto_classes_generation_failed", model_key,
                error=str(e),
                generation_time=generation_time)
            
            return ModelResponse(
                content="",
                success=False,
                error=error_msg,
                metadata={"model_key": model_key, "error_type": "generation"}
            )
    
    def _generate_with_model(
        self, 
        model_result: ModelLoadResult, 
        prompt: str, 
        **generation_kwargs
    ) -> str:
        """Generate response using the loaded model."""
        
        model = model_result.model
        tokenizer = model_result.tokenizer
        processor = model_result.processor
        
        # Handle different model types
        if model_result.auto_class_used == "AutoModelForCausalLM":
            return self._generate_causal_lm(model, tokenizer, prompt, **generation_kwargs)
        elif model_result.auto_class_used == "AutoModelForSeq2SeqLM":
            return self._generate_seq2seq(model, tokenizer, prompt, **generation_kwargs)
        elif model_result.auto_class_used == "AutoModelForImageTextToText":
            return self._generate_multimodal(model, processor, prompt, **generation_kwargs)
        else:
            # Fallback to basic generation
            return self._generate_basic(model, tokenizer, prompt, **generation_kwargs)
    
    def _generate_causal_lm(
        self, 
        model, 
        tokenizer, 
        prompt: str, 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text using causal language model."""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        
        # Set pad token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode response (exclude input tokens)
        response_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _generate_seq2seq(
        self, 
        model, 
        tokenizer, 
        prompt: str, 
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text using sequence-to-sequence model."""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()
    
    def _generate_multimodal(
        self, 
        model, 
        processor, 
        prompt: str, 
        images: Optional[List] = None,
        **kwargs
    ) -> str:
        """Generate text using multimodal model."""
        
        # This is a placeholder for multimodal generation
        # Actual implementation would handle image inputs
        if processor:
            # Process text input
            inputs = processor(text=prompt, return_tensors="pt")
            
            # Generate (simplified)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, **kwargs)
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        else:
            return "Multimodal generation not fully implemented"
    
    def _generate_basic(
        self, 
        model, 
        tokenizer, 
        prompt: str, 
        **kwargs
    ) -> str:
        """Basic generation fallback."""
        return f"Generated response for: {prompt[:50]}..."


class DurgasAIAutoClassesIntegrator:
    """
    Main integration class that combines Auto Classes with DurgasAI architecture.
    
    This class provides a unified interface that can use both legacy and
    Auto Classes approaches based on configuration.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the integrator."""
        self.legacy_manager = ModelManager()
        self.adapter = AutoClassesModelAdapter(self.legacy_manager)
        self.response_generator = AutoClassesResponseGenerator(self.adapter)
        
        # Load enhanced configurations if available
        if config_file:
            self._load_enhanced_config(config_file)
    
    def _load_enhanced_config(self, config_file: str) -> None:
        """Load enhanced model configurations."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                self.adapter.enhanced_registry.load_from_file(str(config_path))
                info(f"Loaded enhanced configurations from {config_file}", "auto_classes")
            else:
                debug(f"Enhanced config file not found: {config_file}", "auto_classes")
        except Exception as e:
            error(f"Failed to load enhanced config: {str(e)}", "auto_classes", e)
    
    def generate_response(
        self, 
        model_key: str, 
        prompt: str, 
        use_auto_classes: bool = True,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response using either Auto Classes or legacy approach.
        
        Args:
            model_key: Model identifier
            prompt: Input prompt
            use_auto_classes: Whether to use Auto Classes (default: True)
            **kwargs: Additional generation parameters
        
        Returns:
            ModelResponse with generated content
        """
        
        if use_auto_classes and model_key in self.adapter.enhanced_registry.configs:
            debug(f"Using Auto Classes for {model_key}", "auto_classes")
            return self.response_generator.generate_response(model_key, prompt, **kwargs)
        else:
            debug(f"Using legacy approach for {model_key}", "auto_classes")
            return self.legacy_manager.generate_response(prompt, model_key, **kwargs)
    
    def get_available_models(self, include_auto_classes: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get all available models with their capabilities."""
        models = {}
        
        # Add Auto Classes models
        if include_auto_classes:
            for model_name in self.adapter.enhanced_registry.list_models():
                info = self.adapter.get_model_info_enhanced(model_name)
                info['source'] = 'auto_classes'
                models[model_name] = info
        
        # Add legacy models (if any not covered by Auto Classes)
        # This would require accessing legacy model configurations
        
        return models
    
    def get_model_capabilities(self, model_key: str) -> Dict[str, Any]:
        """Get detailed model capabilities."""
        if model_key in self.adapter.enhanced_registry.configs:
            config = self.adapter.enhanced_registry.configs[model_key]
            return {
                'task_type': config.task_type.value,
                'auto_class': config.auto_class_type,
                'capabilities': config.capabilities.__dict__,
                'resource_requirements': config.resource_requirements.__dict__,
                'multimodal': config.capabilities.multimodal,
                'streaming': config.capabilities.streaming
            }
        else:
            return {'error': 'Model not found in enhanced registry'}
    
    def validate_system_compatibility(self) -> Dict[str, Any]:
        """Validate system compatibility with Auto Classes."""
        validation_results = {
            'compatible': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check PyTorch version
        torch_version = torch.__version__
        if torch.__version__ < "2.0.0":
            validation_results['issues'].append(f"PyTorch version {torch_version} < 2.0.0")
            validation_results['recommendations'].append("Upgrade PyTorch to 2.0+")
            validation_results['compatible'] = False
        
        # Check transformers version
        try:
            import transformers
            if hasattr(transformers, '__version__'):
                version = transformers.__version__
                if version < "4.40.0":
                    validation_results['issues'].append(f"Transformers version {version} < 4.40.0")
                    validation_results['recommendations'].append("Upgrade transformers to 4.40+")
        except:
            validation_results['issues'].append("Could not determine transformers version")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            validation_results['cuda_info'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            }
        else:
            validation_results['cuda_info'] = {'available': False}
            validation_results['recommendations'].append("Consider GPU acceleration for better performance")
        
        # Check memory
        system_info = self.adapter.enhanced_registry.get_system_info()
        if system_info['memory_available_gb'] < 4:
            validation_results['issues'].append("Low available memory (< 4GB)")
            validation_results['recommendations'].append("Close other applications or add more RAM")
        
        return validation_results
    
    def create_migration_plan(self) -> Dict[str, Any]:
        """Create a plan for migrating to Auto Classes."""
        plan = {
            'steps': [],
            'estimated_time': '30-60 minutes',
            'backup_required': True,
            'rollback_available': True
        }
        
        # Step 1: Backup existing configurations
        plan['steps'].append({
            'step': 1,
            'title': 'Backup Existing Configurations',
            'description': 'Create backup of current model configurations',
            'files': ['config/model_config.json', 'utils/model_manager.py'],
            'risk': 'low'
        })
        
        # Step 2: Install dependencies
        plan['steps'].append({
            'step': 2,
            'title': 'Install/Update Dependencies',
            'description': 'Ensure transformers, torch, and other dependencies are up to date',
            'commands': ['pip install transformers>=4.40.0', 'pip install torch>=2.0.0'],
            'risk': 'medium'
        })
        
        # Step 3: Deploy enhanced model manager
        plan['steps'].append({
            'step': 3,
            'title': 'Deploy Enhanced Model Manager',
            'description': 'Add the enhanced model manager with Auto Classes support',
            'files': ['utils/enhanced_model_manager.py', 'utils/auto_classes_integration.py'],
            'risk': 'low'
        })
        
        # Step 4: Update configurations
        plan['steps'].append({
            'step': 4,
            'title': 'Update Model Configurations',
            'description': 'Convert existing model configs to enhanced format',
            'files': ['config/model_config.json'],
            'risk': 'medium'
        })
        
        # Step 5: Update application code
        plan['steps'].append({
            'step': 5,
            'title': 'Update Application Code',
            'description': 'Integrate Auto Classes into main application',
            'files': ['app.py', 'core/app_controller.py'],
            'risk': 'medium'
        })
        
        # Step 6: Testing and validation
        plan['steps'].append({
            'step': 6,
            'title': 'Testing and Validation',
            'description': 'Test all models and functionality',
            'risk': 'low'
        })
        
        return plan


# Example usage and testing functions
def test_auto_classes_integration():
    """Test the Auto Classes integration."""
    print("=== Testing Auto Classes Integration ===")
    
    try:
        # Initialize integrator
        integrator = DurgasAIAutoClassesIntegrator("config/model_config.json")
        
        # Validate system
        validation = integrator.validate_system_compatibility()
        print(f"\nSystem Validation:")
        print(f"  Compatible: {validation['compatible']}")
        print(f"  Issues: {len(validation['issues'])}")
        for issue in validation['issues']:
            print(f"    - {issue}")
        
        # Get available models
        models = integrator.get_available_models()
        print(f"\nAvailable Models: {len(models)}")
        for name, info in models.items():
            print(f"  {name}: {info.get('task_type', 'unknown')} ({info.get('source', 'unknown')})")
        
        # Test model loading (with a small model)
        test_model = "dialogpt_medium"  # Assuming this exists in config
        if test_model in models:
            print(f"\nTesting model loading: {test_model}")
            result = integrator.adapter.load_model_with_auto_classes(test_model)
            if result and result.success:
                print(f"  ✓ Loaded successfully with {result.auto_class_used}")
                print(f"  ✓ Model type: {result.model_type}")
                print(f"  ✓ Loading time: {result.loading_time:.2f}s")
                print(f"  ✓ Memory usage: {result.memory_usage_mb:.1f}MB")
            else:
                print(f"  ❌ Failed to load: {result.error if result else 'Unknown error'}")
        
        print("\n✓ Auto Classes integration test completed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")


if __name__ == "__main__":
    test_auto_classes_integration()
