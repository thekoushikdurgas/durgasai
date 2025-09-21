"""
Enhanced Pipeline Manager for DurgasAI.

This module provides advanced pipeline management capabilities including:
- Intelligent caching (memory + disk)
- Advanced pipeline configuration and management
- Performance monitoring and optimization
- Advanced error handling and recovery
- Memory optimization and batch processing
- Task-specific pipeline optimization
- Hardware acceleration and device management

Based on Hugging Face Transformers Pipeline documentation best practices.
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
import warnings

# Transformers imports
try:
    from transformers import pipeline, Pipeline
    from transformers.pipelines import PIPELINE_REGISTRY
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# PyTorch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Accelerate for device management
try:
    from accelerate import infer_device
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# Local imports
from .logger import debug, info, warning, error
from .config import Config


@dataclass
class PipelineConfig:
    """Configuration for pipeline setup and execution."""
    task: str
    model: Optional[str] = None
    device: Union[int, str] = -1
    batch_size: Optional[int] = None
    torch_dtype: Optional[str] = None
    device_map: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    use_fast: bool = True
    use_auth_token: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    pipeline_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    outputs: Any
    config: PipelineConfig
    processing_time: float
    batch_size: int
    input_count: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    device_used: Optional[str] = None
    memory_usage: Optional[float] = None


@dataclass
class PipelineInfo:
    """Information about a cached pipeline."""
    task: str
    model: str
    device: Union[int, str]
    config: PipelineConfig
    created_at: float
    last_used: float
    usage_count: int
    memory_usage: float
    success_rate: float


class EnhancedPipelineManager:
    """
    Enhanced pipeline manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Advanced pipeline configuration and management
    - Performance monitoring and optimization
    - Advanced error handling and recovery
    - Memory optimization and batch processing
    - Task-specific pipeline optimization
    - Hardware acceleration and device management
    """
    
    def __init__(self, cache_dir: str = "./output/cache/pipelines", max_cache_size: int = 5):
        """
        Initialize the enhanced pipeline manager.
        
        Args:
            cache_dir: Directory for persistent pipeline cache
            max_cache_size: Maximum number of pipelines to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active pipelines
        self.pipeline_cache: Dict[str, Pipeline] = {}
        self.pipeline_info_cache: Dict[str, PipelineInfo] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_executions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_execution_time": 0.0,
            "total_inputs_processed": 0,
            "total_outputs_generated": 0,
            "error_count": 0,
            "memory_optimizations": 0,
            "batch_optimizations": 0,
            "device_switches": 0,
            "quantization_usage": 0
        }
        
        # Available tasks
        self.available_tasks = self._get_available_tasks()
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedPipelineManager initialized", "pipeline_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             torch_available=TORCH_AVAILABLE,
             accelerate_available=ACCELERATE_AVAILABLE)
    
    def _get_available_tasks(self) -> List[str]:
        """Get list of available pipeline tasks."""
        if not TRANSFORMERS_AVAILABLE:
            return []
        
        try:
            return list(PIPELINE_REGISTRY.keys())
        except Exception:
            # Fallback list of common tasks
            return [
                "text-generation",
                "text-classification",
                "question-answering",
                "summarization",
                "translation",
                "conversational",
                "visual-question-answering",
                "image-classification",
                "object-detection",
                "image-segmentation",
                "automatic-speech-recognition",
                "text-to-speech",
                "feature-extraction",
                "fill-mask"
            ]
    
    def _get_cache_key(self, config: PipelineConfig) -> str:
        """Generate unique cache key for pipeline configuration."""
        config_dict = asdict(config)
        # Sort keys for consistent hashing
        sorted_config = {k: config_dict[k] for k in sorted(config_dict.keys())}
        config_str = json.dumps(sorted_config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_pipeline_cache(self, cache_key: str, pipeline: Pipeline, info: PipelineInfo):
        """Save pipeline to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        info_file = self.cache_dir / f"{cache_key}_info.json"
        
        try:
            # Save pipeline (this might not work for all pipeline types)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(pipeline, f)
            except Exception as e:
                debug(f"Could not pickle pipeline: {e}", "pipeline_manager")
                # Skip pipeline saving but continue with info
            
            # Save pipeline info
            info_dict = asdict(info)
            with open(info_file, 'w') as f:
                json.dump(info_dict, f, indent=2, default=str)
            
            debug(f"Pipeline cached successfully", "pipeline_manager",
                  cache_key=cache_key)
                
        except Exception as e:
            error(f"Failed to cache pipeline", "pipeline_manager", e,
                  cache_key=cache_key)
    
    def _load_pipeline_cache(self, cache_key: str) -> Optional[Pipeline]:
        """Load pipeline from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        info_file = self.cache_dir / f"{cache_key}_info.json"
        
        if not cache_file.exists() or not info_file.exists():
            return None
            
        try:
            # Load pipeline info first
            with open(info_file, 'r') as f:
                info_dict = json.load(f)
            
            # Try to load pipeline
            with open(cache_file, 'rb') as f:
                pipeline = pickle.load(f)
            
            debug(f"Pipeline loaded from cache", "pipeline_manager",
                  cache_key=cache_key)
            
            return pipeline
            
        except Exception as e:
            debug(f"Failed to load cached pipeline: {e}", "pipeline_manager",
                  cache_key=cache_key)
            return None
    
    def _load_cache_metadata(self):
        """Load cache metadata for performance tracking."""
        try:
            metadata_file = self.cache_dir / "cache_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.performance_stats.update(metadata.get('performance_stats', {}))
                    
        except Exception as e:
            warning(f"Failed to load cache metadata", "pipeline_manager", e)
    
    def _save_cache_metadata(self):
        """Save cache metadata for persistence."""
        try:
            metadata_file = self.cache_dir / "cache_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'performance_stats': self.performance_stats,
                    'last_updated': time.time()
                }, f, indent=2)
                
        except Exception as e:
            warning(f"Failed to save cache metadata", "pipeline_manager", e)
    
    def _validate_configuration(self, config: PipelineConfig) -> bool:
        """Validate pipeline configuration."""
        try:
            # Validate task
            if config.task not in self.available_tasks:
                error(f"Invalid task: {config.task}", "pipeline_manager")
                return False
            
            # Validate device
            if isinstance(config.device, int) and config.device < -1:
                error(f"Invalid device: {config.device}", "pipeline_manager")
                return False
            
            # Validate batch_size
            if config.batch_size is not None and config.batch_size <= 0:
                error(f"Invalid batch_size: {config.batch_size}", "pipeline_manager")
                return False
            
            # Validate torch_dtype
            if config.torch_dtype is not None:
                valid_dtypes = ['float16', 'float32', 'bfloat16']
                if config.torch_dtype not in valid_dtypes:
                    error(f"Invalid torch_dtype: {config.torch_dtype}", "pipeline_manager")
                    return False
            
            return True
            
        except Exception as e:
            error(f"Configuration validation failed", "pipeline_manager", e)
            return False
    
    def _optimize_configuration(self, config: PipelineConfig) -> PipelineConfig:
        """Optimize pipeline configuration for better performance."""
        optimized_config = asdict(config)
        
        # Memory optimization
        if config.load_in_8bit or config.load_in_4bit:
            self.performance_stats["quantization_usage"] += 1
        
        # Device optimization
        if config.device == -1 and TORCH_AVAILABLE and torch.cuda.is_available():
            # Auto-detect GPU if available
            optimized_config['device'] = 0
            self.performance_stats["device_switches"] += 1
        
        # Batch optimization
        if config.batch_size is None and config.device != -1:
            # Set default batch size for GPU
            optimized_config['batch_size'] = 4
            self.performance_stats["batch_optimizations"] += 1
        
        return PipelineConfig(**optimized_config)
    
    def _estimate_memory_usage(self, config: PipelineConfig) -> float:
        """Estimate memory usage for pipeline configuration."""
        # Rough estimation based on model type and configuration
        base_memory = 1000  # MB base memory
        
        # Add memory based on quantization
        if config.load_in_8bit:
            base_memory *= 0.5
        elif config.load_in_4bit:
            base_memory *= 0.25
        
        # Add memory based on batch size
        if config.batch_size:
            base_memory *= (1 + config.batch_size * 0.1)
        
        # Add memory based on precision
        if config.torch_dtype == 'float16':
            base_memory *= 0.5
        elif config.torch_dtype == 'bfloat16':
            base_memory *= 0.5
        
        return base_memory
    
    def load_pipeline(self, config: PipelineConfig) -> Optional[Pipeline]:
        """
        Load a pipeline with the specified configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline instance or None if failed
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for pipeline loading", "pipeline_manager")
            return None
        
        # Validate configuration
        if not self._validate_configuration(config):
            return None
        
        # Optimize configuration
        optimized_config = self._optimize_configuration(config)
        cache_key = self._get_cache_key(optimized_config)
        
        # Check cache first
        if cache_key in self.pipeline_cache:
            self.performance_stats["cache_hits"] += 1
            self.pipeline_info_cache[cache_key].last_used = time.time()
            self.pipeline_info_cache[cache_key].usage_count += 1
            
            debug(f"Pipeline loaded from memory cache", "pipeline_manager",
                  cache_key=cache_key, task=optimized_config.task)
            return self.pipeline_cache[cache_key]
        
        # Try disk cache
        cached_pipeline = self._load_pipeline_cache(cache_key)
        if cached_pipeline:
            self.pipeline_cache[cache_key] = cached_pipeline
            self.performance_stats["cache_hits"] += 1
            
            # Update info
            info = self.pipeline_info_cache.get(cache_key, PipelineInfo(
                task=optimized_config.task,
                model=optimized_config.model or "default",
                device=optimized_config.device,
                config=optimized_config,
                created_at=time.time(),
                last_used=time.time(),
                usage_count=1,
                memory_usage=self._estimate_memory_usage(optimized_config),
                success_rate=1.0
            ))
            info.last_used = time.time()
            info.usage_count += 1
            self.pipeline_info_cache[cache_key] = info
            
            debug(f"Pipeline loaded from disk cache", "pipeline_manager",
                  cache_key=cache_key, task=optimized_config.task)
            return cached_pipeline
        
        # Load new pipeline
        self.performance_stats["cache_misses"] += 1
        start_time = time.time()
        
        try:
            # Prepare pipeline arguments
            pipeline_kwargs = {
                "task": optimized_config.task,
                "device": optimized_config.device,
                "trust_remote_code": optimized_config.trust_remote_code,
                "use_fast": optimized_config.use_fast
            }
            
            # Add optional parameters
            if optimized_config.model:
                pipeline_kwargs["model"] = optimized_config.model
            if optimized_config.batch_size:
                pipeline_kwargs["batch_size"] = optimized_config.batch_size
            if optimized_config.torch_dtype:
                pipeline_kwargs["torch_dtype"] = getattr(torch, optimized_config.torch_dtype)
            if optimized_config.device_map:
                pipeline_kwargs["device_map"] = optimized_config.device_map
            if optimized_config.use_auth_token:
                pipeline_kwargs["use_auth_token"] = optimized_config.use_auth_token
            
            # Add model kwargs
            model_kwargs = optimized_config.model_kwargs or {}
            if optimized_config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            if optimized_config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            if model_kwargs:
                pipeline_kwargs["model_kwargs"] = model_kwargs
            
            # Add additional pipeline kwargs
            if optimized_config.pipeline_kwargs:
                pipeline_kwargs.update(optimized_config.pipeline_kwargs)
            
            # Create pipeline
            pipeline_instance = pipeline(**pipeline_kwargs)
            
            loading_time = time.time() - start_time
            
            # Cache the pipeline
            self.pipeline_cache[cache_key] = pipeline_instance
            
            # Create pipeline info
            pipeline_info = PipelineInfo(
                task=optimized_config.task,
                model=optimized_config.model or "default",
                device=optimized_config.device,
                config=optimized_config,
                created_at=time.time(),
                last_used=time.time(),
                usage_count=1,
                memory_usage=self._estimate_memory_usage(optimized_config),
                success_rate=1.0
            )
            self.pipeline_info_cache[cache_key] = pipeline_info
            
            # Save to disk cache
            self._save_pipeline_cache(cache_key, pipeline_instance, pipeline_info)
            
            # Manage cache size
            if len(self.pipeline_cache) > self.max_cache_size:
                self._evict_least_used_pipeline()
            
            info(f"Pipeline loaded successfully", "pipeline_manager",
                 task=optimized_config.task,
                 model=optimized_config.model,
                 device=optimized_config.device,
                 loading_time=loading_time)
            
            return pipeline_instance
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load pipeline", "pipeline_manager", e,
                  task=optimized_config.task,
                  model=optimized_config.model,
                  device=optimized_config.device)
            return None
    
    def _evict_least_used_pipeline(self):
        """Evict the least recently used pipeline from cache."""
        if not self.pipeline_info_cache:
            return
        
        # Find least recently used pipeline
        lru_key = min(self.pipeline_info_cache.keys(),
                     key=lambda k: self.pipeline_info_cache[k].last_used)
        
        # Remove from cache
        del self.pipeline_cache[lru_key]
        del self.pipeline_info_cache[lru_key]
        
        debug(f"Evicted least recently used pipeline", "pipeline_manager",
              cache_key=lru_key)
    
    def execute_pipeline(self, config: PipelineConfig, inputs: Any, 
                        **kwargs) -> Optional[PipelineResult]:
        """
        Execute a pipeline with the specified inputs.
        
        Args:
            config: Pipeline configuration
            inputs: Input data for the pipeline
            **kwargs: Additional pipeline-specific parameters
            
        Returns:
            PipelineResult object with execution results
        """
        start_time = time.time()
        
        # Load pipeline
        pipeline_instance = self.load_pipeline(config)
        if not pipeline_instance:
            return PipelineResult(
                outputs=None,
                config=config,
                processing_time=time.time() - start_time,
                batch_size=1,
                input_count=0,
                success=False,
                error="Failed to load pipeline"
            )
        
        try:
            # Prepare inputs
            if isinstance(inputs, (list, tuple)):
                input_count = len(inputs)
                batch_size = input_count
            else:
                input_count = 1
                batch_size = 1
            
            # Execute pipeline
            outputs = pipeline_instance(inputs, **kwargs)
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats["total_executions"] += 1
            self.performance_stats["total_inputs_processed"] += input_count
            self.performance_stats["total_outputs_generated"] += input_count
            self._update_average_execution_time(processing_time)
            
            # Get device info
            device_used = str(config.device)
            if hasattr(pipeline_instance, 'device'):
                device_used = str(pipeline_instance.device)
            
            # Create result
            result = PipelineResult(
                outputs=outputs,
                config=config,
                processing_time=processing_time,
                batch_size=batch_size,
                input_count=input_count,
                success=True,
                metadata={
                    'memory_usage_mb': self._estimate_memory_usage(config),
                    'device_used': device_used,
                    'pipeline_type': type(pipeline_instance).__name__,
                    'task': config.task
                },
                device_used=device_used,
                memory_usage=self._estimate_memory_usage(config)
            )
            
            debug(f"Pipeline executed successfully", "pipeline_manager",
                  task=config.task,
                  input_count=input_count,
                  processing_time=processing_time)
            
            return result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Pipeline execution failed", "pipeline_manager", e,
                  task=config.task,
                  input_count=input_count,
                  processing_time=time.time() - start_time)
            
            return PipelineResult(
                outputs=None,
                config=config,
                processing_time=time.time() - start_time,
                batch_size=batch_size,
                input_count=input_count,
                success=False,
                error=str(e)
            )
    
    def _update_average_execution_time(self, execution_time: float):
        """Update running average of execution times."""
        total_executions = self.performance_stats["total_executions"]
        if total_executions == 1:
            self.performance_stats["average_execution_time"] = execution_time
        else:
            # Running average
            current_avg = self.performance_stats["average_execution_time"]
            self.performance_stats["average_execution_time"] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available pipeline tasks."""
        return self.available_tasks.copy()
    
    def get_pipeline_info(self, cache_key: str) -> Optional[PipelineInfo]:
        """Get information about a cached pipeline."""
        return self.pipeline_info_cache.get(cache_key)
    
    def list_cached_pipelines(self) -> List[PipelineInfo]:
        """List all cached pipelines."""
        return list(self.pipeline_info_cache.values())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_hit_rate = 0.0
        total_cache_operations = (self.performance_stats["cache_hits"] + 
                                 self.performance_stats["cache_misses"])
        if total_cache_operations > 0:
            cache_hit_rate = (self.performance_stats["cache_hits"] / 
                             total_cache_operations * 100)
        
        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'memory_cache_size': len(self.pipeline_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.pkl"))),
            'average_inputs_per_execution': (
                self.performance_stats["total_inputs_processed"] / 
                max(self.performance_stats["total_executions"], 1)
            ),
            'success_rate': (
                (self.performance_stats["total_executions"] - self.performance_stats["error_count"]) / 
                max(self.performance_stats["total_executions"], 1) * 100
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear pipeline cache."""
        if memory_only:
            self.pipeline_cache.clear()
            self.pipeline_info_cache.clear()
            info("Memory cache cleared", "pipeline_manager")
        else:
            # Clear both memory and disk cache
            self.pipeline_cache.clear()
            self.pipeline_info_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            for info_file in self.cache_dir.glob("*_info.json"):
                info_file.unlink()
            
            info("All caches cleared", "pipeline_manager")
    
    def optimize_for_task(self, task: str, model: Optional[str] = None) -> PipelineConfig:
        """Get optimized configuration for a specific task."""
        # Task-specific optimizations
        optimizations = {
            "text-generation": {
                "batch_size": 4,
                "torch_dtype": "float16",
                "device_map": "auto" if ACCELERATE_AVAILABLE else None
            },
            "text-classification": {
                "batch_size": 16,
                "torch_dtype": "float16"
            },
            "question-answering": {
                "batch_size": 8,
                "torch_dtype": "float16"
            },
            "image-classification": {
                "batch_size": 8,
                "torch_dtype": "float16"
            },
            "object-detection": {
                "batch_size": 2,
                "torch_dtype": "float16"
            }
        }
        
        config_dict = {
            "task": task,
            "model": model,
            "device": 0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1
        }
        
        # Apply task-specific optimizations
        if task in optimizations:
            config_dict.update(optimizations[task])
        
        return PipelineConfig(**config_dict)


# Global instance for easy access
pipeline_manager = EnhancedPipelineManager()
