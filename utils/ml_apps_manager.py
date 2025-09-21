"""
Enhanced ML Apps Manager for DurgasAI.

This module provides advanced ML app management capabilities including:
- Intelligent caching (memory + disk)
- Advanced ML app configuration and management
- Performance monitoring and optimization
- Advanced error handling and recovery
- Memory optimization and batch processing
- Task-specific ML app optimization
- Hardware acceleration and device management
- Integration with enhanced pipeline manager

Based on Hugging Face Transformers ML Apps documentation best practices.
"""

import hashlib
import pickle
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
import json
import warnings
import subprocess
import webbrowser
from urllib.parse import urlparse

# Transformers imports
try:
    from transformers import pipeline, Pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Gradio imports
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

# PyTorch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local imports
from .logger import debug, info, warning, error
from .config import Config
from .pipeline_manager import pipeline_manager, PipelineConfig


@dataclass
class MLAppConfig:
    """Configuration for ML app setup and deployment."""
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
    # Gradio-specific options
    title: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[Any]] = None
    cache_examples: bool = True
    theme: Optional[str] = None
    css: Optional[str] = None
    # Deployment options
    server_name: str = "127.0.0.1"
    server_port: Optional[int] = None
    share: bool = False
    debug: bool = False
    show_error: bool = True
    quiet: bool = False
    show_tips: bool = True
    enable_queue: bool = True
    max_threads: int = 40
    auth: Optional[Tuple[str, str]] = None
    auth_message: Optional[str] = None
    ssl_verify: bool = True
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None


@dataclass
class MLAppResult:
    """Result of ML app creation and deployment."""
    app: Optional[Any]  # Gradio interface
    config: MLAppConfig
    creation_time: float
    deployment_url: Optional[str] = None
    process_id: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MLAppInfo:
    """Information about a cached ML app."""
    task: str
    model: str
    device: Union[int, str]
    config: MLAppConfig
    created_at: float
    last_used: float
    usage_count: int
    memory_usage: float
    success_rate: float
    deployment_status: str = "stopped"
    deployment_url: Optional[str] = None
    process_id: Optional[int] = None


class EnhancedMLAppsManager:
    """
    Enhanced ML apps manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Advanced ML app configuration and management
    - Performance monitoring and optimization
    - Advanced error handling and recovery
    - Memory optimization and batch processing
    - Task-specific ML app optimization
    - Hardware acceleration and device management
    - Integration with enhanced pipeline manager
    """
    
    def __init__(self, cache_dir: str = "./output/cache/ml_apps", max_cache_size: int = 5):
        """
        Initialize the enhanced ML apps manager.
        
        Args:
            cache_dir: Directory for persistent ML app cache
            max_cache_size: Maximum number of ML apps to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active ML apps
        self.ml_apps_cache: Dict[str, Any] = {}
        self.ml_apps_info_cache: Dict[str, MLAppInfo] = {}
        
        # Active deployments
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_app_creations": 0,
            "total_deployments": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_creation_time": 0.0,
            "average_deployment_time": 0.0,
            "total_users_served": 0,
            "error_count": 0,
            "memory_optimizations": 0,
            "batch_optimizations": 0,
            "device_switches": 0,
            "quantization_usage": 0,
            "active_deployments": 0
        }
        
        # Available tasks (same as pipeline manager)
        self.available_tasks = self._get_available_tasks()
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedMLAppsManager initialized", "ml_apps_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             gradio_available=GRADIO_AVAILABLE,
             torch_available=TORCH_AVAILABLE)
    
    def _get_available_tasks(self) -> List[str]:
        """Get list of available ML app tasks."""
        if not TRANSFORMERS_AVAILABLE:
            return []
        
        try:
            from transformers.pipelines import PIPELINE_REGISTRY
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
    
    def _get_cache_key(self, config: MLAppConfig) -> str:
        """Generate unique cache key for ML app configuration."""
        config_dict = asdict(config)
        # Remove deployment-specific options for caching
        deployment_keys = ['server_name', 'server_port', 'share', 'debug', 'auth']
        for key in deployment_keys:
            config_dict.pop(key, None)
        
        # Sort keys for consistent hashing
        sorted_config = {k: config_dict[k] for k in sorted(config_dict.keys())}
        config_str = json.dumps(sorted_config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_ml_app_cache(self, cache_key: str, ml_app: Any, info: MLAppInfo):
        """Save ML app to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        info_file = self.cache_dir / f"{cache_key}_info.json"
        
        try:
            # Save ML app (this might not work for all app types)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(ml_app, f)
            except Exception as e:
                debug(f"Could not pickle ML app: {e}", "ml_apps_manager")
                # Skip app saving but continue with info
            
            # Save ML app info
            info_dict = asdict(info)
            with open(info_file, 'w') as f:
                json.dump(info_dict, f, indent=2, default=str)
            
            debug(f"ML app cached successfully", "ml_apps_manager",
                  cache_key=cache_key)
                
        except Exception as e:
            error(f"Failed to cache ML app", "ml_apps_manager", e,
                  cache_key=cache_key)
    
    def _load_ml_app_cache(self, cache_key: str) -> Optional[Any]:
        """Load ML app from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        info_file = self.cache_dir / f"{cache_key}_info.json"
        
        if not cache_file.exists() or not info_file.exists():
            return None
            
        try:
            # Load ML app info first
            with open(info_file, 'r') as f:
                info_dict = json.load(f)
            
            # Try to load ML app
            with open(cache_file, 'rb') as f:
                ml_app = pickle.load(f)
            
            debug(f"ML app loaded from cache", "ml_apps_manager",
                  cache_key=cache_key)
            
            return ml_app
            
        except Exception as e:
            debug(f"Failed to load cached ML app: {e}", "ml_apps_manager",
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
            warning(f"Failed to load cache metadata", "ml_apps_manager", e)
    
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
            warning(f"Failed to save cache metadata", "ml_apps_manager", e)
    
    def _validate_configuration(self, config: MLAppConfig) -> bool:
        """Validate ML app configuration."""
        try:
            # Validate task
            if config.task not in self.available_tasks:
                error(f"Invalid task: {config.task}", "ml_apps_manager")
                return False
            
            # Validate device
            if isinstance(config.device, int) and config.device < -1:
                error(f"Invalid device: {config.device}", "ml_apps_manager")
                return False
            
            # Validate batch_size
            if config.batch_size is not None and config.batch_size <= 0:
                error(f"Invalid batch_size: {config.batch_size}", "ml_apps_manager")
                return False
            
            # Validate torch_dtype
            if config.torch_dtype is not None:
                valid_dtypes = ['float16', 'float32', 'bfloat16']
                if config.torch_dtype not in valid_dtypes:
                    error(f"Invalid torch_dtype: {config.torch_dtype}", "ml_apps_manager")
                    return False
            
            # Validate Gradio availability
            if not GRADIO_AVAILABLE:
                error("Gradio not available for ML app creation", "ml_apps_manager")
                return False
            
            return True
            
        except Exception as e:
            error(f"Configuration validation failed", "ml_apps_manager", e)
            return False
    
    def _optimize_configuration(self, config: MLAppConfig) -> MLAppConfig:
        """Optimize ML app configuration for better performance."""
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
        
        # Set default port if not specified
        if config.server_port is None:
            optimized_config['server_port'] = self._find_available_port()
        
        return MLAppConfig(**optimized_config)
    
    def _find_available_port(self, start_port: int = 7860) -> int:
        """Find an available port for deployment."""
        import socket
        
        port = start_port
        while port < start_port + 100:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                port += 1
        
        return start_port  # Fallback
    
    def _estimate_memory_usage(self, config: MLAppConfig) -> float:
        """Estimate memory usage for ML app configuration."""
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
        
        # Add extra memory for Gradio interface
        base_memory += 200  # MB for Gradio
        
        return base_memory
    
    def create_ml_app(self, config: MLAppConfig) -> Optional[MLAppResult]:
        """
        Create an ML app with the specified configuration.
        
        Args:
            config: ML app configuration
            
        Returns:
            MLAppResult object with creation results
        """
        if not GRADIO_AVAILABLE:
            error("Gradio not available for ML app creation", "ml_apps_manager")
            return MLAppResult(
                app=None,
                config=config,
                creation_time=0.0,
                success=False,
                error="Gradio not available"
            )
        
        # Validate configuration
        if not self._validate_configuration(config):
            return MLAppResult(
                app=None,
                config=config,
                creation_time=0.0,
                success=False,
                error="Invalid configuration"
            )
        
        # Optimize configuration
        optimized_config = self._optimize_configuration(config)
        cache_key = self._get_cache_key(optimized_config)
        
        start_time = time.time()
        
        # Check cache first
        if cache_key in self.ml_apps_cache:
            self.performance_stats["cache_hits"] += 1
            self.ml_apps_info_cache[cache_key].last_used = time.time()
            self.ml_apps_info_cache[cache_key].usage_count += 1
            
            debug(f"ML app loaded from memory cache", "ml_apps_manager",
                  cache_key=cache_key, task=optimized_config.task)
            
            creation_time = time.time() - start_time
            return MLAppResult(
                app=self.ml_apps_cache[cache_key],
                config=optimized_config,
                creation_time=creation_time,
                success=True
            )
        
        # Try disk cache
        cached_app = self._load_ml_app_cache(cache_key)
        if cached_app:
            self.ml_apps_cache[cache_key] = cached_app
            self.performance_stats["cache_hits"] += 1
            
            # Update info
            info = self.ml_apps_info_cache.get(cache_key, MLAppInfo(
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
            self.ml_apps_info_cache[cache_key] = info
            
            debug(f"ML app loaded from disk cache", "ml_apps_manager",
                  cache_key=cache_key, task=optimized_config.task)
            
            creation_time = time.time() - start_time
            return MLAppResult(
                app=cached_app,
                config=optimized_config,
                creation_time=creation_time,
                success=True
            )
        
        # Create new ML app
        self.performance_stats["cache_misses"] += 1
        
        try:
            # Create pipeline first
            pipeline_config = PipelineConfig(
                task=optimized_config.task,
                model=optimized_config.model,
                device=optimized_config.device,
                batch_size=optimized_config.batch_size,
                torch_dtype=optimized_config.torch_dtype,
                device_map=optimized_config.device_map,
                load_in_8bit=optimized_config.load_in_8bit,
                load_in_4bit=optimized_config.load_in_4bit,
                trust_remote_code=optimized_config.trust_remote_code,
                use_fast=optimized_config.use_fast,
                use_auth_token=optimized_config.use_auth_token,
                model_kwargs=optimized_config.model_kwargs,
                pipeline_kwargs=optimized_config.pipeline_kwargs
            )
            
            pipeline_instance = pipeline_manager.load_pipeline(pipeline_config)
            if not pipeline_instance:
                return MLAppResult(
                    app=None,
                    config=optimized_config,
                    creation_time=time.time() - start_time,
                    success=False,
                    error="Failed to load pipeline"
                )
            
            # Create Gradio interface
            gradio_kwargs = {}
            if optimized_config.title:
                gradio_kwargs['title'] = optimized_config.title
            if optimized_config.description:
                gradio_kwargs['description'] = optimized_config.description
            if optimized_config.examples:
                gradio_kwargs['examples'] = optimized_config.examples
            if optimized_config.cache_examples is not None:
                gradio_kwargs['cache_examples'] = optimized_config.cache_examples
            if optimized_config.theme:
                gradio_kwargs['theme'] = optimized_config.theme
            if optimized_config.css:
                gradio_kwargs['css'] = optimized_config.css
            
            ml_app = gr.Interface.from_pipeline(pipeline_instance, **gradio_kwargs)
            
            creation_time = time.time() - start_time
            
            # Cache the ML app
            self.ml_apps_cache[cache_key] = ml_app
            
            # Create ML app info
            ml_app_info = MLAppInfo(
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
            self.ml_apps_info_cache[cache_key] = ml_app_info
            
            # Save to disk cache
            self._save_ml_app_cache(cache_key, ml_app, ml_app_info)
            
            # Manage cache size
            if len(self.ml_apps_cache) > self.max_cache_size:
                self._evict_least_used_ml_app()
            
            # Update performance stats
            self.performance_stats["total_app_creations"] += 1
            self._update_average_creation_time(creation_time)
            
            info(f"ML app created successfully", "ml_apps_manager",
                 task=optimized_config.task,
                 model=optimized_config.model,
                 device=optimized_config.device,
                 creation_time=creation_time)
            
            return MLAppResult(
                app=ml_app,
                config=optimized_config,
                creation_time=creation_time,
                success=True
            )
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to create ML app", "ml_apps_manager", e,
                  task=optimized_config.task,
                  model=optimized_config.model,
                  device=optimized_config.device)
            
            return MLAppResult(
                app=None,
                config=optimized_config,
                creation_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _evict_least_used_ml_app(self):
        """Evict the least recently used ML app from cache."""
        if not self.ml_apps_info_cache:
            return
        
        # Find least recently used ML app
        lru_key = min(self.ml_apps_info_cache.keys(),
                     key=lambda k: self.ml_apps_info_cache[k].last_used)
        
        # Remove from cache
        del self.ml_apps_cache[lru_key]
        del self.ml_apps_info_cache[lru_key]
        
        debug(f"Evicted least recently used ML app", "ml_apps_manager",
              cache_key=lru_key)
    
    def _update_average_creation_time(self, creation_time: float):
        """Update running average of creation times."""
        total_creations = self.performance_stats["total_app_creations"]
        if total_creations == 1:
            self.performance_stats["average_creation_time"] = creation_time
        else:
            # Running average
            current_avg = self.performance_stats["average_creation_time"]
            self.performance_stats["average_creation_time"] = (
                (current_avg * (total_creations - 1) + creation_time) / total_creations
            )
    
    def _update_average_deployment_time(self, deployment_time: float):
        """Update running average of deployment times."""
        total_deployments = self.performance_stats["total_deployments"]
        if total_deployments == 1:
            self.performance_stats["average_deployment_time"] = deployment_time
        else:
            # Running average
            current_avg = self.performance_stats["average_deployment_time"]
            self.performance_stats["average_deployment_time"] = (
                (current_avg * (total_deployments - 1) + deployment_time) / total_deployments
            )
    
    def deploy_ml_app(self, config: MLAppConfig, auto_open: bool = True) -> Optional[MLAppResult]:
        """
        Deploy an ML app with the specified configuration.
        
        Args:
            config: ML app configuration
            auto_open: Whether to automatically open the app in browser
            
        Returns:
            MLAppResult object with deployment results
        """
        # Create ML app first
        result = self.create_ml_app(config)
        if not result.success or not result.app:
            return result
        
        cache_key = self._get_cache_key(config)
        start_time = time.time()
        
        try:
            # Prepare launch arguments
            launch_kwargs = {
                'server_name': config.server_name,
                'share': config.share,
                'debug': config.debug,
                'show_error': config.show_error,
                'quiet': config.quiet,
                'show_tips': config.show_tips,
                'enable_queue': config.enable_queue,
                'max_threads': config.max_threads,
                'ssl_verify': config.ssl_verify
            }
            
            # Add optional parameters
            if config.server_port:
                launch_kwargs['server_port'] = config.server_port
            if config.auth:
                launch_kwargs['auth'] = config.auth
            if config.auth_message:
                launch_kwargs['auth_message'] = config.auth_message
            if config.ssl_keyfile:
                launch_kwargs['ssl_keyfile'] = config.ssl_keyfile
            if config.ssl_certfile:
                launch_kwargs['ssl_certfile'] = config.ssl_certfile
            
            # Launch the app
            deployment_info = result.app.launch(**launch_kwargs)
            
            deployment_time = time.time() - start_time
            
            # Extract deployment URL
            deployment_url = None
            if hasattr(deployment_info, 'local_url'):
                deployment_url = deployment_info.local_url
            elif hasattr(deployment_info, 'public_url'):
                deployment_url = deployment_info.public_url
            elif isinstance(deployment_info, str):
                deployment_url = deployment_info
            
            # Store deployment info
            self.active_deployments[cache_key] = {
                'config': config,
                'deployment_url': deployment_url,
                'started_at': time.time(),
                'deployment_info': deployment_info
            }
            
            # Update ML app info
            if cache_key in self.ml_apps_info_cache:
                self.ml_apps_info_cache[cache_key].deployment_status = "running"
                self.ml_apps_info_cache[cache_key].deployment_url = deployment_url
            
            # Update performance stats
            self.performance_stats["total_deployments"] += 1
            self.performance_stats["active_deployments"] = len(self.active_deployments)
            self._update_average_deployment_time(deployment_time)
            
            # Auto-open browser if requested
            if auto_open and deployment_url:
                try:
                    webbrowser.open(deployment_url)
                except Exception as e:
                    debug(f"Failed to open browser: {e}", "ml_apps_manager")
            
            info(f"ML app deployed successfully", "ml_apps_manager",
                 task=config.task,
                 model=config.model,
                 deployment_url=deployment_url,
                 deployment_time=deployment_time)
            
            # Update result
            result.deployment_url = deployment_url
            result.metadata = {
                'deployment_time': deployment_time,
                'auto_opened': auto_open,
                'launch_kwargs': launch_kwargs
            }
            
            return result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to deploy ML app", "ml_apps_manager", e,
                  task=config.task,
                  model=config.model,
                  deployment_time=time.time() - start_time)
            
            return MLAppResult(
                app=result.app,
                config=config,
                creation_time=result.creation_time,
                deployment_time=time.time() - start_time,
                success=False,
                error=f"Deployment failed: {str(e)}"
            )
    
    def stop_deployment(self, cache_key: str) -> bool:
        """
        Stop a deployed ML app.
        
        Args:
            cache_key: Cache key of the ML app to stop
            
        Returns:
            True if successfully stopped, False otherwise
        """
        if cache_key not in self.active_deployments:
            warning(f"No active deployment found for cache key: {cache_key}", "ml_apps_manager")
            return False
        
        try:
            deployment_info = self.active_deployments[cache_key]['deployment_info']
            
            # Try to stop the deployment
            if hasattr(deployment_info, 'close'):
                deployment_info.close()
            elif hasattr(deployment_info, 'stop'):
                deployment_info.stop()
            
            # Remove from active deployments
            del self.active_deployments[cache_key]
            
            # Update ML app info
            if cache_key in self.ml_apps_info_cache:
                self.ml_apps_info_cache[cache_key].deployment_status = "stopped"
                self.ml_apps_info_cache[cache_key].deployment_url = None
            
            # Update performance stats
            self.performance_stats["active_deployments"] = len(self.active_deployments)
            
            info(f"ML app deployment stopped", "ml_apps_manager",
                 cache_key=cache_key)
            
            return True
            
        except Exception as e:
            error(f"Failed to stop ML app deployment", "ml_apps_manager", e,
                  cache_key=cache_key)
            return False
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available ML app tasks."""
        return self.available_tasks.copy()
    
    def get_ml_app_info(self, cache_key: str) -> Optional[MLAppInfo]:
        """Get information about a cached ML app."""
        return self.ml_apps_info_cache.get(cache_key)
    
    def list_cached_ml_apps(self) -> List[MLAppInfo]:
        """List all cached ML apps."""
        return list(self.ml_apps_info_cache.values())
    
    def list_active_deployments(self) -> Dict[str, Dict[str, Any]]:
        """List all active deployments."""
        return self.active_deployments.copy()
    
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
            'memory_cache_size': len(self.ml_apps_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.pkl"))),
            'success_rate': (
                (self.performance_stats["total_app_creations"] - self.performance_stats["error_count"]) / 
                max(self.performance_stats["total_app_creations"], 1) * 100
            ),
            'deployment_success_rate': (
                (self.performance_stats["total_deployments"] - self.performance_stats["error_count"]) / 
                max(self.performance_stats["total_deployments"], 1) * 100
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear ML app cache."""
        if memory_only:
            self.ml_apps_cache.clear()
            self.ml_apps_info_cache.clear()
            info("Memory cache cleared", "ml_apps_manager")
        else:
            # Clear both memory and disk cache
            self.ml_apps_cache.clear()
            self.ml_apps_info_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            for info_file in self.cache_dir.glob("*_info.json"):
                info_file.unlink()
            
            info("All caches cleared", "ml_apps_manager")
    
    def optimize_for_task(self, task: str, model: Optional[str] = None) -> MLAppConfig:
        """Get optimized configuration for a specific task."""
        # Task-specific optimizations
        optimizations = {
            "text-generation": {
                "batch_size": 4,
                "torch_dtype": "float16",
                "device_map": "auto",
                "title": "Text Generation App",
                "description": "Generate text using AI models"
            },
            "text-classification": {
                "batch_size": 16,
                "torch_dtype": "float16",
                "title": "Text Classification App",
                "description": "Classify text into different categories"
            },
            "question-answering": {
                "batch_size": 8,
                "torch_dtype": "float16",
                "title": "Question Answering App",
                "description": "Answer questions based on given context"
            },
            "image-classification": {
                "batch_size": 8,
                "torch_dtype": "float16",
                "title": "Image Classification App",
                "description": "Classify images into different categories"
            },
            "object-detection": {
                "batch_size": 2,
                "torch_dtype": "float16",
                "title": "Object Detection App",
                "description": "Detect and locate objects in images"
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
        
        return MLAppConfig(**config_dict)


# Global instance for easy access
ml_apps_manager = EnhancedMLAppsManager()
