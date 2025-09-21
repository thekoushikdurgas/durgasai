"""
Custom Pipeline Manager for DurgasAI.

This module provides utilities for creating, managing, and deploying custom Hugging Face pipelines:
- Pipeline creation and registration
- Custom pipeline templates
- Integration with web server inference
- Pipeline sharing and Hub integration
- Usage statistics and monitoring
"""

import json
import time
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
import inspect

from transformers import Pipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import PIPELINE_REGISTRY
import numpy as np

from .logger import debug, info, warning, error, log_user_action
from .config import Config


class CustomPipelineManager:
    """
    Manager for custom Hugging Face pipelines in DurgasAI.
    
    This class provides:
    - Pipeline creation and registration
    - Custom pipeline templates
    - Integration with web server inference
    - Usage statistics and monitoring
    - Pipeline sharing capabilities
    """
    
    def __init__(self):
        """Initialize the custom pipeline manager."""
        self.custom_pipelines: Dict[str, Dict[str, Any]] = {}
        self.pipeline_stats: Dict[str, Dict[str, Any]] = {}
        self.pipeline_templates: Dict[str, Dict[str, Any]] = {}
        self.custom_pipeline_dir = Path("output/custom_pipelines")
        self.custom_pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing pipelines
        self._load_existing_pipelines()
        
        # Initialize built-in templates
        self._initialize_templates()
        
        debug("CustomPipelineManager initialized", "custom_pipeline_manager")
    
    def _load_existing_pipelines(self):
        """Load existing custom pipelines from storage."""
        pipeline_config_file = self.custom_pipeline_dir / "pipelines.json"
        
        if pipeline_config_file.exists():
            try:
                with open(pipeline_config_file, 'r') as f:
                    data = json.load(f)
                    self.custom_pipelines = data.get("pipelines", {})
                    self.pipeline_stats = data.get("stats", {})
                
                info(f"Loaded {len(self.custom_pipelines)} custom pipelines", "custom_pipeline_manager")
            except Exception as e:
                warning(f"Failed to load existing pipelines: {e}", "custom_pipeline_manager")
    
    def _save_pipelines(self):
        """Save pipeline configurations to storage."""
        pipeline_config_file = self.custom_pipeline_dir / "pipelines.json"
        
        try:
            data = {
                "pipelines": self.custom_pipelines,
                "stats": self.pipeline_stats,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(pipeline_config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            debug("Pipeline configurations saved", "custom_pipeline_manager")
        except Exception as e:
            error(f"Failed to save pipeline configurations: {e}", "custom_pipeline_manager")
    
    def _initialize_templates(self):
        """Initialize built-in pipeline templates."""
        self.pipeline_templates = {
            "pair_classification": {
                "name": "Pair Classification Pipeline",
                "description": "Classify relationships between two text inputs",
                "category": "text",
                "difficulty": "intermediate",
                "example_code": self._get_pair_classification_template(),
                "required_methods": ["_sanitize_parameters", "preprocess", "_forward", "postprocess"],
                "example_usage": {
                    "inputs": "First text",
                    "parameters": {"second_text": "Second text", "top_k": 3}
                }
            },
            "enhanced_sentiment": {
                "name": "Enhanced Sentiment Analysis",
                "description": "Sentiment analysis with emotion detection and confidence scoring",
                "category": "text",
                "difficulty": "beginner",
                "example_code": self._get_enhanced_sentiment_template(),
                "required_methods": ["_sanitize_parameters", "preprocess", "_forward", "postprocess"],
                "example_usage": {
                    "inputs": "I love this product!",
                    "parameters": {"include_emotions": True, "top_k": 2}
                }
            },
            "document_classifier": {
                "name": "Document Classification Pipeline",
                "description": "Classify documents by topic, category, or intent",
                "category": "text",
                "difficulty": "intermediate",
                "example_code": self._get_document_classifier_template(),
                "required_methods": ["_sanitize_parameters", "preprocess", "_forward", "postprocess"],
                "example_usage": {
                    "inputs": "Long document text here...",
                    "parameters": {"chunk_size": 512, "overlap": 50}
                }
            },
            "multi_label_classifier": {
                "name": "Multi-Label Classification Pipeline",
                "description": "Assign multiple labels to a single input",
                "category": "text",
                "difficulty": "advanced",
                "example_code": self._get_multi_label_template(),
                "required_methods": ["_sanitize_parameters", "preprocess", "_forward", "postprocess"],
                "example_usage": {
                    "inputs": "This is a technical article about machine learning",
                    "parameters": {"threshold": 0.5, "max_labels": 5}
                }
            }
        }
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available pipeline templates."""
        return {
            template_id: {
                "name": template["name"],
                "description": template["description"],
                "category": template["category"],
                "difficulty": template["difficulty"],
                "example_usage": template["example_usage"]
            }
            for template_id, template in self.pipeline_templates.items()
        }
    
    def get_template_code(self, template_id: str) -> Optional[str]:
        """Get the example code for a specific template."""
        template = self.pipeline_templates.get(template_id)
        return template["example_code"] if template else None
    
    def create_pipeline_from_template(self, template_id: str, pipeline_name: str, 
                                    model_name: str, customizations: Dict[str, Any] = None) -> bool:
        """
        Create a new pipeline from a template.
        
        Args:
            template_id: ID of the template to use
            pipeline_name: Name for the new pipeline
            model_name: Model to use with the pipeline
            customizations: Custom modifications to apply
            
        Returns:
            bool: Success status
        """
        try:
            template = self.pipeline_templates.get(template_id)
            if not template:
                error(f"Template {template_id} not found", "custom_pipeline_manager")
                return False
            
            # Create pipeline configuration
            pipeline_config = {
                "name": pipeline_name,
                "template_id": template_id,
                "model_name": model_name,
                "description": template["description"],
                "category": template["category"],
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "customizations": customizations or {},
                "usage_count": 0
            }
            
            # Generate pipeline code
            pipeline_code = self._customize_template_code(
                template["example_code"], 
                pipeline_name, 
                customizations or {}
            )
            
            # Save pipeline code to file
            pipeline_file = self.custom_pipeline_dir / f"{pipeline_name}.py"
            with open(pipeline_file, 'w') as f:
                f.write(pipeline_code)
            
            # Register pipeline
            self.custom_pipelines[pipeline_name] = pipeline_config
            self.pipeline_stats[pipeline_name] = {
                "calls": 0,
                "errors": 0,
                "avg_execution_time": 0,
                "last_used": None
            }
            
            # Save configurations
            self._save_pipelines()
            
            info(f"Custom pipeline '{pipeline_name}' created successfully", "custom_pipeline_manager")
            log_user_action("custom_pipeline_created", pipeline_name=pipeline_name, template_id=template_id)
            
            return True
            
        except Exception as e:
            error(f"Failed to create pipeline from template: {e}", "custom_pipeline_manager", e)
            return False
    
    def register_custom_pipeline(self, pipeline_name: str, pipeline_class: Type[Pipeline], 
                                model_class: Type, default_model: str = None,
                                description: str = "", version: str = "1.0.0") -> bool:
        """
        Register a custom pipeline class.
        
        Args:
            pipeline_name: Unique name for the pipeline
            pipeline_class: Pipeline class to register
            model_class: Model class to use
            default_model: Default model identifier
            description: Pipeline description
            version: Pipeline version
            
        Returns:
            bool: Success status
        """
        try:
            # Validate pipeline class
            if not issubclass(pipeline_class, Pipeline):
                raise ValueError("pipeline_class must inherit from transformers.Pipeline")
            
            # Check required methods
            required_methods = ["_sanitize_parameters", "preprocess", "_forward", "postprocess"]
            for method in required_methods:
                if not hasattr(pipeline_class, method):
                    raise ValueError(f"Pipeline class missing required method: {method}")
            
            # Register with transformers registry
            try:
                PIPELINE_REGISTRY.register_pipeline(
                    pipeline_name,
                    pipeline_class=pipeline_class,
                    pt_model=model_class,
                    default={"pt": (default_model, "main")} if default_model else None,
                    type="text"
                )
            except Exception as e:
                warning(f"Could not register with transformers registry: {e}", "custom_pipeline_manager")
            
            # Store in custom registry
            pipeline_config = {
                "name": pipeline_name,
                "class_name": pipeline_class.__name__,
                "model_class": model_class.__name__,
                "default_model": default_model,
                "description": description,
                "version": version,
                "registered_at": datetime.now().isoformat(),
                "usage_count": 0,
                "category": "custom"
            }
            
            self.custom_pipelines[pipeline_name] = pipeline_config
            self.pipeline_stats[pipeline_name] = {
                "calls": 0,
                "errors": 0,
                "avg_execution_time": 0,
                "last_used": None
            }
            
            # Save configurations
            self._save_pipelines()
            
            info(f"Custom pipeline '{pipeline_name}' registered successfully", "custom_pipeline_manager")
            log_user_action("custom_pipeline_registered", pipeline_name=pipeline_name)
            
            return True
            
        except Exception as e:
            error(f"Failed to register custom pipeline: {e}", "custom_pipeline_manager", e)
            return False
    
    def get_custom_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered custom pipelines."""
        return {
            name: {
                "name": config["name"],
                "description": config["description"],
                "version": config["version"],
                "category": config["category"],
                "usage_count": config["usage_count"],
                "default_model": config.get("default_model"),
                "registered_at": config["registered_at"]
            }
            for name, config in self.custom_pipelines.items()
        }
    
    def get_pipeline_stats(self, pipeline_name: str) -> Dict[str, Any]:
        """Get usage statistics for a specific pipeline."""
        return self.pipeline_stats.get(pipeline_name, {})
    
    def update_pipeline_stats(self, pipeline_name: str, execution_time: float, success: bool = True):
        """Update usage statistics for a pipeline."""
        if pipeline_name in self.pipeline_stats:
            stats = self.pipeline_stats[pipeline_name]
            stats["calls"] += 1
            stats["last_used"] = datetime.now().isoformat()
            
            if not success:
                stats["errors"] += 1
            
            # Update average execution time
            if stats["calls"] == 1:
                stats["avg_execution_time"] = execution_time
            else:
                current_avg = stats["avg_execution_time"]
                stats["avg_execution_time"] = (current_avg * (stats["calls"] - 1) + execution_time) / stats["calls"]
            
            # Update usage count in pipeline config
            if pipeline_name in self.custom_pipelines:
                self.custom_pipelines[pipeline_name]["usage_count"] = stats["calls"]
            
            # Save updated stats
            self._save_pipelines()
    
    def delete_custom_pipeline(self, pipeline_name: str) -> bool:
        """
        Delete a custom pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to delete
            
        Returns:
            bool: Success status
        """
        try:
            # Remove from registries
            if pipeline_name in self.custom_pipelines:
                del self.custom_pipelines[pipeline_name]
            
            if pipeline_name in self.pipeline_stats:
                del self.pipeline_stats[pipeline_name]
            
            # Remove pipeline file
            pipeline_file = self.custom_pipeline_dir / f"{pipeline_name}.py"
            if pipeline_file.exists():
                pipeline_file.unlink()
            
            # Save configurations
            self._save_pipelines()
            
            info(f"Custom pipeline '{pipeline_name}' deleted successfully", "custom_pipeline_manager")
            log_user_action("custom_pipeline_deleted", pipeline_name=pipeline_name)
            
            return True
            
        except Exception as e:
            error(f"Failed to delete custom pipeline: {e}", "custom_pipeline_manager", e)
            return False
    
    def load_pipeline_from_file(self, pipeline_file: Path) -> bool:
        """
        Load a custom pipeline from a Python file.
        
        Args:
            pipeline_file: Path to the pipeline Python file
            
        Returns:
            bool: Success status
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_pipeline", pipeline_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find Pipeline classes in the module
            pipeline_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Pipeline) and 
                    obj != Pipeline):
                    pipeline_classes.append((name, obj))
            
            if not pipeline_classes:
                raise ValueError("No Pipeline classes found in file")
            
            # Register found pipelines
            success_count = 0
            for class_name, pipeline_class in pipeline_classes:
                pipeline_name = pipeline_file.stem + "_" + class_name.lower()
                
                if self.register_custom_pipeline(
                    pipeline_name,
                    pipeline_class,
                    AutoModelForSequenceClassification,  # Default model class
                    description=f"Custom pipeline loaded from {pipeline_file.name}"
                ):
                    success_count += 1
            
            info(f"Loaded {success_count} custom pipelines from {pipeline_file.name}", "custom_pipeline_manager")
            return success_count > 0
            
        except Exception as e:
            error(f"Failed to load pipeline from file: {e}", "custom_pipeline_manager", e)
            return False
    
    def _customize_template_code(self, template_code: str, pipeline_name: str, 
                               customizations: Dict[str, Any]) -> str:
        """Customize template code with user modifications."""
        # Basic customization - replace class name
        customized_code = template_code.replace("TemplatePipeline", f"{pipeline_name}Pipeline")
        
        # Apply additional customizations based on the customizations dict
        # This is a simplified version - in practice, you'd want more sophisticated code generation
        
        return customized_code
    
    def _get_pair_classification_template(self) -> str:
        """Get pair classification pipeline template code."""
        return '''
import numpy as np
from transformers import Pipeline

def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

class TemplatePipeline(Pipeline):
    """Custom pair classification pipeline."""
    
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        if "top_k" in kwargs:
            postprocess_kwargs["top_k"] = kwargs["top_k"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, text, second_text=None):
        if second_text is None:
            raise ValueError("second_text is required")
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs, top_k=1):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)
        
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            results.append({
                "label": self.model.config.id2label[idx],
                "score": float(probabilities[idx])
            })
        
        return results[0] if top_k == 1 else results
'''
    
    def _get_enhanced_sentiment_template(self) -> str:
        """Get enhanced sentiment analysis template code."""
        return '''
import numpy as np
from transformers import Pipeline
from datetime import datetime

def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

class TemplatePipeline(Pipeline):
    """Enhanced sentiment analysis pipeline."""
    
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]
        if "include_emotions" in kwargs:
            postprocess_kwargs["include_emotions"] = kwargs["include_emotions"]
        if "top_k" in kwargs:
            postprocess_kwargs["top_k"] = kwargs["top_k"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, inputs, max_length=512):
        return self.tokenizer(inputs, return_tensors=self.framework, 
                            truncation=True, padding=True, max_length=max_length)
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs, include_emotions=False, top_k=1):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)
        
        best_idx = np.argmax(probabilities)
        result = {
            "label": self.model.config.id2label[best_idx],
            "score": float(probabilities[best_idx])
        }
        
        if include_emotions:
            emotion_mapping = {
                "POSITIVE": ["joy", "satisfaction"],
                "NEGATIVE": ["anger", "sadness"],
                "NEUTRAL": ["calm", "neutral"]
            }
            result["emotions"] = emotion_mapping.get(result["label"], ["neutral"])
        
        result["metadata"] = {
            "confidence": "high" if result["score"] > 0.8 else "medium" if result["score"] > 0.5 else "low",
            "timestamp": datetime.now().isoformat()
        }
        
        return result
'''
    
    def _get_document_classifier_template(self) -> str:
        """Get document classification template code."""
        return '''
import numpy as np
from transformers import Pipeline

class TemplatePipeline(Pipeline):
    """Document classification pipeline with chunking support."""
    
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        if "chunk_size" in kwargs:
            preprocess_kwargs["chunk_size"] = kwargs["chunk_size"]
        if "overlap" in kwargs:
            preprocess_kwargs["overlap"] = kwargs["overlap"]
        if "aggregation" in kwargs:
            postprocess_kwargs["aggregation"] = kwargs["aggregation"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, text, chunk_size=512, overlap=50):
        # Simple chunking implementation
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Tokenize all chunks
        model_inputs = self.tokenizer(chunks, return_tensors=self.framework,
                                    truncation=True, padding=True, max_length=chunk_size)
        model_inputs["chunks"] = chunks
        return model_inputs
    
    def _forward(self, model_inputs):
        chunks = model_inputs.pop("chunks")
        outputs = self.model(**model_inputs)
        outputs.chunks = chunks
        return outputs
    
    def postprocess(self, model_outputs, aggregation="mean"):
        logits = model_outputs.logits.numpy()
        
        if aggregation == "mean":
            avg_logits = np.mean(logits, axis=0)
            probabilities = softmax(avg_logits.reshape(1, -1))[0]
        elif aggregation == "max":
            max_logits = np.max(logits, axis=0)
            probabilities = softmax(max_logits.reshape(1, -1))[0]
        else:
            # Use first chunk
            probabilities = softmax(logits[0:1])[0]
        
        best_idx = np.argmax(probabilities)
        return {
            "label": self.model.config.id2label[best_idx],
            "score": float(probabilities[best_idx]),
            "chunks_processed": len(model_outputs.chunks),
            "aggregation_method": aggregation
        }
'''
    
    def _get_multi_label_template(self) -> str:
        """Get multi-label classification template code."""
        return '''
import numpy as np
from transformers import Pipeline

class TemplatePipeline(Pipeline):
    """Multi-label classification pipeline."""
    
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "max_labels" in kwargs:
            postprocess_kwargs["max_labels"] = kwargs["max_labels"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, text, max_length=512):
        return self.tokenizer(text, return_tensors=self.framework,
                            truncation=True, padding=True, max_length=max_length)
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs, threshold=0.5, max_labels=5):
        logits = model_outputs.logits[0].numpy()
        # Apply sigmoid for multi-label
        probabilities = 1 / (1 + np.exp(-logits))
        
        # Get labels above threshold
        above_threshold = np.where(probabilities > threshold)[0]
        
        # Sort by probability and limit to max_labels
        sorted_indices = above_threshold[np.argsort(probabilities[above_threshold])[::-1]]
        final_indices = sorted_indices[:max_labels] if max_labels > 0 else sorted_indices
        
        labels = []
        for idx in final_indices:
            labels.append({
                "label": self.model.config.id2label[idx],
                "score": float(probabilities[idx])
            })
        
        return {
            "labels": labels,
            "threshold_used": threshold,
            "total_candidates": len(above_threshold)
        }
'''


# Global instance
custom_pipeline_manager = CustomPipelineManager()
