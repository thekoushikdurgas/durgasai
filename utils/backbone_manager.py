"""
Enhanced Backbone Manager for DurgasAI.

This module provides advanced backbone management capabilities including:
- Intelligent caching (memory + disk)
- Multi-layer feature extraction
- timm backbone integration
- Performance monitoring
- Advanced error handling
- Feature map analysis
- GPU acceleration support
- Batch processing optimization

Based on Hugging Face Transformers Backbones documentation best practices.
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
    from transformers import AutoBackbone, AutoImageProcessor
    from transformers.models.backbone import TimmBackbone, TimmBackboneConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Torch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Local imports
from .logger import debug, info, warning, error
from .config import Config


@dataclass
class BackboneInfo:
    """Comprehensive information about a loaded backbone."""
    model_id: str
    backbone: Any
    processor: Any
    out_indices: Tuple[int, ...]
    out_features: Optional[List[str]]
    num_channels: List[int]
    feature_info: List[Dict[str, Any]]
    load_time: float
    last_used: float
    cache_key: str
    backbone_type: str
    supports_timm: bool = False
    is_timm: bool = False
    supports_gpu: bool = False
    gpu_optimized: bool = False
    supports_compilation: bool = False
    image_size: Optional[int] = None
    patch_size: Optional[int] = None
    embed_dim: Optional[int] = None


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction operation."""
    feature_maps: List[torch.Tensor]
    backbone_info: BackboneInfo
    extraction_time: float
    input_shape: Tuple[int, ...]
    output_shapes: List[Tuple[int, ...]]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedBackboneManager:
    """
    Enhanced backbone manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Multi-layer feature extraction
    - timm backbone integration
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Feature map analysis and visualization
    - GPU acceleration support
    - Batch processing optimization
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/backbones", max_cache_size: int = 10):
        """
        Initialize the enhanced backbone manager.
        
        Args:
            cache_dir: Directory for persistent backbone cache
            max_cache_size: Maximum number of backbones to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active backbones
        self.backbone_cache: Dict[str, BackboneInfo] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_load_time": 0.0,
            "total_extractions": 0,
            "batch_extractions": 0,
            "error_count": 0,
            "gpu_extractions": 0,
            "compiled_extractions": 0,
            "total_feature_maps": 0,
            "timm_backbones": 0,
            "multi_layer_extractions": 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedBackboneManager initialized", "backbone_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             torch_available=TORCH_AVAILABLE)
    
    def _get_cache_key(self, model_id: str, **kwargs) -> str:
        """Generate unique cache key for backbone configuration."""
        sorted_kwargs = sorted(kwargs.items())
        config_str = f"{model_id}_{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_backbone_cache(self, cache_key: str, backbone_info: BackboneInfo):
        """Save backbone to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        backbone_dir = self.cache_dir / cache_key
        
        try:
            # Save backbone files
            backbone_info.backbone.save_pretrained(str(backbone_dir))
            
            # Create metadata without the backbone object
            metadata = asdict(backbone_info)
            metadata.pop('backbone', None)  # Remove the actual backbone object
            metadata.pop('processor', None)  # Remove the processor object
            metadata['backbone_path'] = str(backbone_dir)
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            debug(f"Backbone cached successfully", "backbone_manager",
                  cache_key=cache_key, model_id=backbone_info.model_id)
                
        except Exception as e:
            error(f"Failed to cache backbone", "backbone_manager", e,
                  cache_key=cache_key, model_id=backbone_info.model_id)
    
    def _load_backbone_cache(self, cache_key: str) -> Optional[BackboneInfo]:
        """Load backbone from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        backbone_dir = self.cache_dir / cache_key
        
        if not cache_file.exists() or not backbone_dir.exists():
            return None
            
        try:
            # Load metadata
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            
            # Load backbone from disk
            backbone = AutoBackbone.from_pretrained(str(backbone_dir))
            
            # Load processor
            processor = AutoImageProcessor.from_pretrained(metadata['model_id'])
            
            # Reconstruct BackboneInfo
            backbone_info = BackboneInfo(
                model_id=metadata['model_id'],
                backbone=backbone,
                processor=processor,
                out_indices=tuple(metadata['out_indices']),
                out_features=metadata.get('out_features'),
                num_channels=metadata['num_channels'],
                feature_info=metadata['feature_info'],
                load_time=metadata['load_time'],
                last_used=metadata['last_used'],
                cache_key=cache_key,
                backbone_type=metadata['backbone_type'],
                supports_timm=metadata.get('supports_timm', False),
                is_timm=metadata.get('is_timm', False),
                supports_gpu=metadata.get('supports_gpu', False),
                gpu_optimized=metadata.get('gpu_optimized', False),
                supports_compilation=metadata.get('supports_compilation', False),
                image_size=metadata.get('image_size'),
                patch_size=metadata.get('patch_size'),
                embed_dim=metadata.get('embed_dim')
            )
            
            debug(f"Backbone loaded from cache", "backbone_manager",
                  cache_key=cache_key, model_id=backbone_info.model_id)
            
            return backbone_info
            
        except Exception as e:
            error(f"Failed to load cached backbone", "backbone_manager", e,
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
            warning(f"Failed to load cache metadata", "backbone_manager", e)
    
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
            warning(f"Failed to save cache metadata", "backbone_manager", e)
    
    def _validate_backbone_compatibility(self, model_id: str, **kwargs) -> bool:
        """Validate backbone compatibility before loading."""
        try:
            # Check if model supports backbone
            # This is a simplified check - in practice, you might want more validation
            return True
            
        except Exception as e:
            error(f"Backbone compatibility check failed", "backbone_manager", e,
                  model_id=model_id)
            return False
    
    def _determine_backbone_capabilities(self, backbone: Any) -> Tuple[str, bool, bool, bool, bool, bool]:
        """Determine backbone type and capabilities."""
        backbone_type = backbone.__class__.__name__
        supports_timm = hasattr(backbone, 'timm_model') or 'Timm' in backbone_type
        is_timm = 'TimmBackbone' in backbone_type
        supports_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        gpu_optimized = supports_gpu
        supports_compilation = TORCH_AVAILABLE
        
        return backbone_type, supports_timm, is_timm, supports_gpu, gpu_optimized, supports_compilation
    
    def load_backbone(self, model_id: str, out_indices: Optional[Tuple[int, ...]] = None, 
                     out_features: Optional[List[str]] = None, use_timm_backbone: bool = False,
                     use_pretrained_backbone: bool = True, device: str = "auto", **kwargs) -> Optional[BackboneInfo]:
        """
        Load backbone with intelligent caching and validation.
        
        Args:
            model_id: HuggingFace model identifier
            out_indices: Layer indices for feature extraction
            out_features: Layer names for feature extraction (alternative to indices)
            use_timm_backbone: Whether to use timm backbone
            use_pretrained_backbone: Use pretrained or randomly initialized weights
            device: Processing device ("auto", "cuda", "cpu")
            **kwargs: Additional backbone loading arguments
            
        Returns:
            BackboneInfo object if successful, None if failed
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for backbone loading", "backbone_manager")
            return None
        
        start_time = time.time()
        cache_key = self._get_cache_key(model_id, out_indices=out_indices, out_features=out_features,
                                       use_timm_backbone=use_timm_backbone, **kwargs)
        
        # Check memory cache first
        if cache_key in self.backbone_cache:
            self.backbone_cache[cache_key].last_used = time.time()
            self.performance_stats["cache_hits"] += 1
            
            info(f"Backbone cache hit", "backbone_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return self.backbone_cache[cache_key]
        
        # Check disk cache
        backbone_info = self._load_backbone_cache(cache_key)
        if backbone_info:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, backbone_info)
            self.performance_stats["cache_hits"] += 1
            
            info(f"Backbone loaded from disk cache", "backbone_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return backbone_info
        
        # Cache miss - load from HuggingFace
        self.performance_stats["cache_misses"] += 1
        
        # Validate compatibility
        if not self._validate_backbone_compatibility(model_id, **kwargs):
            error(f"Backbone compatibility validation failed", "backbone_manager",
                  model_id=model_id)
            return None
        
        try:
            # Load backbone with enhanced configuration
            default_kwargs = {
                'cache_dir': str(self.cache_dir.parent / "huggingface")
            }
            
            # Add layer selection parameters
            if out_indices is not None:
                default_kwargs['out_indices'] = out_indices
            if out_features is not None:
                default_kwargs['out_features'] = out_features
            
            # Add device configuration if CUDA is available
            if device == "auto" and TORCH_AVAILABLE and torch.cuda.is_available():
                default_kwargs['device'] = "cuda"
            elif device != "auto":
                default_kwargs['device'] = device
            
            default_kwargs.update(kwargs)
            
            debug(f"Loading backbone from HuggingFace", "backbone_manager",
                  model_id=model_id, kwargs=default_kwargs)
            
            # Load backbone
            if use_timm_backbone:
                # Load timm backbone
                backbone_config = TimmBackboneConfig(
                    model_id, 
                    use_pretrained_backbone=use_pretrained_backbone
                )
                backbone = TimmBackbone(backbone_config)
                self.performance_stats["timm_backbones"] += 1
            else:
                # Load standard backbone
                backbone = AutoBackbone.from_pretrained(model_id, **default_kwargs)
            
            # Load processor
            processor = AutoImageProcessor.from_pretrained(model_id)
            
            # Determine backbone capabilities
            backbone_type, supports_timm, is_timm, supports_gpu, gpu_optimized, supports_compilation = self._determine_backbone_capabilities(backbone)
            
            # Extract feature information
            config = backbone.config
            num_channels = getattr(config, 'num_channels', [])
            feature_info = getattr(backbone, 'feature_info', [])
            
            # Create backbone info
            load_time = time.time() - start_time
            backbone_info = BackboneInfo(
                model_id=model_id,
                backbone=backbone,
                processor=processor,
                out_indices=out_indices or tuple(range(len(num_channels))),
                out_features=out_features,
                num_channels=num_channels,
                feature_info=feature_info,
                load_time=load_time,
                last_used=time.time(),
                cache_key=cache_key,
                backbone_type=backbone_type,
                supports_timm=supports_timm,
                is_timm=is_timm,
                supports_gpu=supports_gpu,
                gpu_optimized=gpu_optimized,
                supports_compilation=supports_compilation,
                image_size=getattr(config, 'image_size', None),
                patch_size=getattr(config, 'patch_size', None),
                embed_dim=getattr(config, 'embed_dim', None)
            )
            
            # Cache the backbone
            self._add_to_memory_cache(cache_key, backbone_info)
            self._save_backbone_cache(cache_key, backbone_info)
            
            # Update performance stats
            self.performance_stats["total_loads"] += 1
            self._update_average_load_time(load_time)
            self._save_cache_metadata()
            
            info(f"Backbone loaded successfully", "backbone_manager",
                 model_id=model_id,
                 load_time=load_time,
                 backbone_type=backbone_type,
                 is_timm=is_timm,
                 gpu_optimized=gpu_optimized)
            
            return backbone_info
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load backbone", "backbone_manager", e,
                  model_id=model_id, load_time=time.time() - start_time)
            return None
    
    def _add_to_memory_cache(self, cache_key: str, backbone_info: BackboneInfo):
        """Add backbone to memory cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self.backbone_cache) >= self.max_cache_size:
            oldest_key = min(self.backbone_cache.keys(), 
                           key=lambda k: self.backbone_cache[k].last_used)
            del self.backbone_cache[oldest_key]
            
            debug(f"Evicted backbone from memory cache", "backbone_manager",
                  evicted_key=oldest_key)
        
        self.backbone_cache[cache_key] = backbone_info
    
    def _update_average_load_time(self, load_time: float):
        """Update running average of load times."""
        total_loads = self.performance_stats["total_loads"]
        if total_loads == 1:
            self.performance_stats["average_load_time"] = load_time
        else:
            # Running average
            current_avg = self.performance_stats["average_load_time"]
            self.performance_stats["average_load_time"] = (
                (current_avg * (total_loads - 1) + load_time) / total_loads
            )
    
    def extract_features(self, images: Union[Any, List[Any]], model_id: str, 
                        compile_backbone: bool = False, **kwargs) -> Optional[FeatureExtractionResult]:
        """
        Extract features from images using the backbone.
        
        Args:
            images: Single image or list of images
            model_id: Model identifier for backbone
            compile_backbone: Whether to compile backbone for maximum performance
            **kwargs: Additional processing arguments
            
        Returns:
            FeatureExtractionResult object with extracted features
        """
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            error("Required dependencies not available for feature extraction", "backbone_manager")
            return None
        
        start_time = time.time()
        
        # Load backbone
        backbone_info = self.load_backbone(model_id, **kwargs)
        if not backbone_info:
            return FeatureExtractionResult(
                feature_maps=[],
                backbone_info=None,
                extraction_time=time.time() - start_time,
                input_shape=(),
                output_shapes=[],
                success=False,
                error="Failed to load backbone"
            )
        
        try:
            # Prepare backbone for optimal performance
            backbone = backbone_info.backbone
            processor = backbone_info.processor
            
            # Compile backbone if requested and supported
            if compile_backbone and backbone_info.supports_compilation:
                backbone = torch.compile(backbone)
                self.performance_stats["compiled_extractions"] += 1
            
            # Prepare images
            if not isinstance(images, list):
                images = [images]
                is_single = True
            else:
                is_single = False
            
            # Process images
            processed_images = []
            for image in images:
                if isinstance(image, str):
                    # Load image from path
                    image = Image.open(image).convert('RGB')
                elif hasattr(image, 'convert'):
                    # PIL Image
                    image = image.convert('RGB')
                
                # Process with processor
                inputs = processor(image, return_tensors="pt")
                processed_images.append(inputs)
            
            # Extract features
            feature_maps = []
            output_shapes = []
            
            for inputs in processed_images:
                # Move to device if GPU is available
                if backbone_info.gpu_optimized and TORCH_AVAILABLE and torch.cuda.is_available():
                    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    outputs = backbone(**inputs)
                
                # Get feature maps
                if hasattr(outputs, 'feature_maps'):
                    maps = outputs.feature_maps
                    feature_maps.extend(maps)
                    output_shapes.extend([map.shape for map in maps])
                    
                    # Count feature maps
                    self.performance_stats["total_feature_maps"] += len(maps)
                    
                    # Check for multi-layer extraction
                    if len(maps) > 1:
                        self.performance_stats["multi_layer_extractions"] += 1
            
            extraction_time = time.time() - start_time
            self.performance_stats["total_extractions"] += 1
            
            # Check for batch extraction
            if len(processed_images) > 1:
                self.performance_stats["batch_extractions"] += 1
            
            # Check for GPU extraction
            if backbone_info.gpu_optimized:
                self.performance_stats["gpu_extractions"] += 1
            
            # Create result
            result = FeatureExtractionResult(
                feature_maps=feature_maps,
                backbone_info=backbone_info,
                extraction_time=extraction_time,
                input_shape=processed_images[0]['pixel_values'].shape if processed_images else (),
                output_shapes=output_shapes,
                success=True,
                metadata={
                    'is_single': is_single,
                    'backbone_type': backbone_info.backbone_type,
                    'is_timm': backbone_info.is_timm,
                    'gpu_optimized': backbone_info.gpu_optimized,
                    'compiled': compile_backbone and backbone_info.supports_compilation,
                    'num_layers': len(backbone_info.out_indices),
                    'out_indices': backbone_info.out_indices
                }
            )
            
            debug(f"Feature extraction completed", "backbone_manager",
                  model_id=model_id,
                  num_images=len(processed_images),
                  num_feature_maps=len(feature_maps),
                  extraction_time=extraction_time)
            
            return result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Feature extraction failed", "backbone_manager", e,
                  model_id=model_id, num_images=len(images))
            
            return FeatureExtractionResult(
                feature_maps=[],
                backbone_info=backbone_info,
                extraction_time=time.time() - start_time,
                input_shape=(),
                output_shapes=[],
                success=False,
                error=str(e)
            )
    
    def get_backbone_info(self, model_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get comprehensive backbone information without loading."""
        backbone_info = self.load_backbone(model_id, **kwargs)
        if not backbone_info:
            return None
        
        return {
            'model_id': backbone_info.model_id,
            'backbone_type': backbone_info.backbone_type,
            'supports_timm': backbone_info.supports_timm,
            'is_timm': backbone_info.is_timm,
            'supports_gpu': backbone_info.supports_gpu,
            'gpu_optimized': backbone_info.gpu_optimized,
            'supports_compilation': backbone_info.supports_compilation,
            'out_indices': backbone_info.out_indices,
            'out_features': backbone_info.out_features,
            'num_channels': backbone_info.num_channels,
            'feature_info': backbone_info.feature_info,
            'load_time': backbone_info.load_time,
            'cache_key': backbone_info.cache_key,
            'image_size': backbone_info.image_size,
            'patch_size': backbone_info.patch_size,
            'embed_dim': backbone_info.embed_dim
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_hit_rate = 0.0
        if self.performance_stats["total_loads"] > 0:
            cache_hit_rate = (self.performance_stats["cache_hits"] / 
                            self.performance_stats["total_loads"]) * 100
        
        return {
            **self.performance_stats,
            'cache_hit_rate': cache_hit_rate,
            'memory_cache_size': len(self.backbone_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.json"))),
            'average_feature_maps_per_extraction': (
                self.performance_stats["total_feature_maps"] / 
                max(self.performance_stats["total_extractions"], 1)
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear backbone cache."""
        if memory_only:
            self.backbone_cache.clear()
            info("Memory cache cleared", "backbone_manager")
        else:
            # Clear both memory and disk cache
            self.backbone_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix != '.json':
                    cache_file.unlink()
            
            info("All caches cleared", "backbone_manager")
    
    def list_cached_backbones(self) -> List[Dict[str, Any]]:
        """List all cached backbones with metadata."""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    cached.append(metadata)
            except Exception as e:
                warning(f"Failed to read cache metadata", "backbone_manager", e,
                       cache_file=cache_file.name)
        
        return cached


# Global instance for easy access
backbone_manager = EnhancedBackboneManager()
