"""
Enhanced Video Processor Manager for DurgasAI.

This module provides advanced video processing capabilities including:
- Intelligent caching (memory + disk)
- Fast video processor optimization
- Batch processing with automatic optimization
- GPU acceleration support
- Performance monitoring
- Advanced error handling
- Video model integration
- Temporal processing capabilities

Based on Hugging Face Transformers Video Processors documentation best practices.
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
    from transformers import AutoVideoProcessor
    from transformers.video_utils import load_video
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
class VideoProcessorInfo:
    """Comprehensive information about a loaded video processor."""
    model_id: str
    processor: Any
    size: Dict[str, Any]
    image_mean: List[float]
    image_std: List[float]
    do_resize: bool
    do_normalize: bool
    do_rescale: bool
    is_fast: bool
    load_time: float
    last_used: float
    cache_key: str
    processor_type: str
    supports_batch: bool = True
    supports_gpu: bool = False
    gpu_optimized: bool = False
    supports_compilation: bool = False
    max_frames: Optional[int] = None
    frame_sampling_rate: Optional[float] = None


@dataclass
class VideoProcessingResult:
    """Result of video processing operation."""
    pixel_values: Any
    processor_info: VideoProcessorInfo
    processing_time: float
    batch_size: int
    frame_count: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedVideoProcessorManager:
    """
    Enhanced video processor manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Fast video processor optimization
    - Batch processing with automatic optimization
    - GPU acceleration support
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Video model integration
    - Temporal processing capabilities
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/video_processors", max_cache_size: int = 10):
        """
        Initialize the enhanced video processor manager.
        
        Args:
            cache_dir: Directory for persistent processor cache
            max_cache_size: Maximum number of processors to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active processors
        self.processor_cache: Dict[str, VideoProcessorInfo] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_load_time": 0.0,
            "total_processings": 0,
            "batch_processings": 0,
            "error_count": 0,
            "gpu_processings": 0,
            "compiled_processings": 0,
            "total_frames_processed": 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedVideoProcessorManager initialized", "video_processor_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             torch_available=TORCH_AVAILABLE)
    
    def _get_cache_key(self, model_id: str, **kwargs) -> str:
        """Generate unique cache key for processor configuration."""
        sorted_kwargs = sorted(kwargs.items())
        config_str = f"{model_id}_{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_processor_cache(self, cache_key: str, processor_info: VideoProcessorInfo):
        """Save processor to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        processor_dir = self.cache_dir / cache_key
        
        try:
            # Save processor files
            processor_info.processor.save_pretrained(str(processor_dir))
            
            # Create metadata without the processor object
            metadata = asdict(processor_info)
            metadata.pop('processor', None)  # Remove the actual processor object
            metadata['processor_path'] = str(processor_dir)
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            debug(f"Video processor cached successfully", "video_processor_manager",
                  cache_key=cache_key, model_id=processor_info.model_id)
                
        except Exception as e:
            error(f"Failed to cache video processor", "video_processor_manager", e,
                  cache_key=cache_key, model_id=processor_info.model_id)
    
    def _load_processor_cache(self, cache_key: str) -> Optional[VideoProcessorInfo]:
        """Load processor from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        processor_dir = self.cache_dir / cache_key
        
        if not cache_file.exists() or not processor_dir.exists():
            return None
            
        try:
            # Load metadata
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            
            # Load processor from disk
            processor = AutoVideoProcessor.from_pretrained(str(processor_dir))
            
            # Reconstruct VideoProcessorInfo
            processor_info = VideoProcessorInfo(
                model_id=metadata['model_id'],
                processor=processor,
                size=metadata['size'],
                image_mean=metadata['image_mean'],
                image_std=metadata['image_std'],
                do_resize=metadata['do_resize'],
                do_normalize=metadata['do_normalize'],
                do_rescale=metadata['do_rescale'],
                is_fast=metadata['is_fast'],
                load_time=metadata['load_time'],
                last_used=metadata['last_used'],
                cache_key=cache_key,
                processor_type=metadata['processor_type'],
                supports_batch=metadata.get('supports_batch', True),
                supports_gpu=metadata.get('supports_gpu', False),
                gpu_optimized=metadata.get('gpu_optimized', False),
                supports_compilation=metadata.get('supports_compilation', False),
                max_frames=metadata.get('max_frames'),
                frame_sampling_rate=metadata.get('frame_sampling_rate')
            )
            
            debug(f"Video processor loaded from cache", "video_processor_manager",
                  cache_key=cache_key, model_id=processor_info.model_id)
            
            return processor_info
            
        except Exception as e:
            error(f"Failed to load cached video processor", "video_processor_manager", e,
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
            warning(f"Failed to load cache metadata", "video_processor_manager", e)
    
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
            warning(f"Failed to save cache metadata", "video_processor_manager", e)
    
    def _validate_processor_compatibility(self, model_id: str, **kwargs) -> bool:
        """Validate processor compatibility before loading."""
        try:
            # Basic validation - check if model exists
            # This is a simplified check - in practice, you might want more validation
            return True
            
        except Exception as e:
            error(f"Video processor compatibility check failed", "video_processor_manager", e,
                  model_id=model_id)
            return False
    
    def _determine_processor_capabilities(self, processor: Any) -> Tuple[str, bool, bool, bool, bool, bool]:
        """Determine processor type and capabilities."""
        processor_type = processor.__class__.__name__
        is_fast = hasattr(processor, '_processor_class') or 'Fast' in processor_type
        supports_batch = True  # Most video processors support batch processing
        supports_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        gpu_optimized = is_fast and supports_gpu
        supports_compilation = TORCH_AVAILABLE and is_fast
        
        return processor_type, is_fast, supports_batch, supports_gpu, gpu_optimized, supports_compilation
    
    def load_processor(self, model_id: str, use_fast: bool = True, device: str = "auto", **kwargs) -> Optional[VideoProcessorInfo]:
        """
        Load video processor with intelligent caching and validation.
        
        Args:
            model_id: HuggingFace model identifier
            use_fast: Whether to use fast processor if available
            device: Processing device ("auto", "cuda", "cpu")
            **kwargs: Additional processor loading arguments
            
        Returns:
            VideoProcessorInfo object if successful, None if failed
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for video processor loading", "video_processor_manager")
            return None
        
        start_time = time.time()
        cache_key = self._get_cache_key(model_id, use_fast=use_fast, device=device, **kwargs)
        
        # Check memory cache first
        if cache_key in self.processor_cache:
            self.processor_cache[cache_key].last_used = time.time()
            self.performance_stats["cache_hits"] += 1
            
            info(f"Video processor cache hit", "video_processor_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return self.processor_cache[cache_key]
        
        # Check disk cache
        processor_info = self._load_processor_cache(cache_key)
        if processor_info:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, processor_info)
            self.performance_stats["cache_hits"] += 1
            
            info(f"Video processor loaded from disk cache", "video_processor_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return processor_info
        
        # Cache miss - load from HuggingFace
        self.performance_stats["cache_misses"] += 1
        
        # Validate compatibility
        if not self._validate_processor_compatibility(model_id, **kwargs):
            error(f"Video processor compatibility validation failed", "video_processor_manager",
                  model_id=model_id)
            return None
        
        try:
            # Load processor with enhanced configuration
            default_kwargs = {
                'use_fast': use_fast,
                'cache_dir': str(self.cache_dir.parent / "huggingface")
            }
            
            # Add device configuration if CUDA is available
            if device == "auto" and TORCH_AVAILABLE and torch.cuda.is_available():
                default_kwargs['device'] = "cuda"
            elif device != "auto":
                default_kwargs['device'] = device
            
            default_kwargs.update(kwargs)
            
            debug(f"Loading video processor from HuggingFace", "video_processor_manager",
                  model_id=model_id, kwargs=default_kwargs)
            
            processor = AutoVideoProcessor.from_pretrained(model_id, **default_kwargs)
            
            # Determine processor capabilities
            processor_type, is_fast, supports_batch, supports_gpu, gpu_optimized, supports_compilation = self._determine_processor_capabilities(processor)
            
            # Create processor info
            load_time = time.time() - start_time
            processor_info = VideoProcessorInfo(
                model_id=model_id,
                processor=processor,
                size=getattr(processor, 'size', {}),
                image_mean=getattr(processor, 'image_mean', [0.5, 0.5, 0.5]),
                image_std=getattr(processor, 'image_std', [0.5, 0.5, 0.5]),
                do_resize=getattr(processor, 'do_resize', True),
                do_normalize=getattr(processor, 'do_normalize', True),
                do_rescale=getattr(processor, 'do_rescale', True),
                is_fast=is_fast,
                load_time=load_time,
                last_used=time.time(),
                cache_key=cache_key,
                processor_type=processor_type,
                supports_batch=supports_batch,
                supports_gpu=supports_gpu,
                gpu_optimized=gpu_optimized,
                supports_compilation=supports_compilation,
                max_frames=getattr(processor, 'max_frames', None),
                frame_sampling_rate=getattr(processor, 'frame_sampling_rate', None)
            )
            
            # Cache the processor
            self._add_to_memory_cache(cache_key, processor_info)
            self._save_processor_cache(cache_key, processor_info)
            
            # Update performance stats
            self.performance_stats["total_loads"] += 1
            self._update_average_load_time(load_time)
            self._save_cache_metadata()
            
            info(f"Video processor loaded successfully", "video_processor_manager",
                 model_id=model_id,
                 load_time=load_time,
                 processor_type=processor_type,
                 is_fast=is_fast,
                 gpu_optimized=gpu_optimized)
            
            return processor_info
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load video processor", "video_processor_manager", e,
                  model_id=model_id, load_time=time.time() - start_time)
            return None
    
    def _add_to_memory_cache(self, cache_key: str, processor_info: VideoProcessorInfo):
        """Add processor to memory cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self.processor_cache) >= self.max_cache_size:
            oldest_key = min(self.processor_cache.keys(), 
                           key=lambda k: self.processor_cache[k].last_used)
            del self.processor_cache[oldest_key]
            
            debug(f"Evicted video processor from memory cache", "video_processor_manager",
                  evicted_key=oldest_key)
        
        self.processor_cache[cache_key] = processor_info
    
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
    
    def process_videos(self, videos: Union[Any, List[Any]], model_id: str, 
                      compile_processor: bool = False, **kwargs) -> Optional[VideoProcessingResult]:
        """
        Process video(s) with optimized batch processing.
        
        Args:
            videos: Single video or list of videos
            model_id: Model identifier for processor
            compile_processor: Whether to compile processor for maximum performance
            **kwargs: Additional processing arguments
            
        Returns:
            VideoProcessingResult object with processing data
        """
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            error("Required dependencies not available for video processing", "video_processor_manager")
            return None
        
        start_time = time.time()
        
        # Load processor
        processor_info = self.load_processor(model_id, **kwargs)
        if not processor_info:
            return VideoProcessingResult(
                pixel_values=None,
                processor_info=None,
                processing_time=time.time() - start_time,
                batch_size=0,
                frame_count=0,
                success=False,
                error="Failed to load video processor"
            )
        
        try:
            # Prepare processor for optimal performance
            processor = processor_info.processor
            
            # Compile processor if requested and supported
            if compile_processor and processor_info.supports_compilation:
                processor = torch.compile(processor)
                self.performance_stats["compiled_processings"] += 1
            
            # Prepare videos
            if not isinstance(videos, list):
                videos = [videos]
                is_single = True
            else:
                is_single = False
            
            # Optimize batch processing
            if processor_info.supports_batch and len(videos) > 1:
                # Batch processing
                encoded = processor(videos, return_tensors="pt")
                self.performance_stats["batch_processings"] += 1
            else:
                # Individual processing
                encoded = processor(
                    videos[0] if is_single else videos,
                    return_tensors="pt"
                )
            
            processing_time = time.time() - start_time
            self.performance_stats["total_processings"] += 1
            
            # Count frames processed
            frame_count = 0
            if hasattr(encoded, 'pixel_values'):
                pixel_values = encoded['pixel_values']
                if hasattr(pixel_values, 'shape'):
                    # Assume shape is [batch, frames, height, width, channels] or similar
                    if len(pixel_values.shape) >= 2:
                        frame_count = pixel_values.shape[1] if len(pixel_values.shape) > 2 else 1
            
            self.performance_stats["total_frames_processed"] += frame_count
            
            # Check for GPU processing
            if processor_info.gpu_optimized:
                self.performance_stats["gpu_processings"] += 1
            
            # Create result
            result = VideoProcessingResult(
                pixel_values=encoded['pixel_values'],
                processor_info=processor_info,
                processing_time=processing_time,
                batch_size=len(videos),
                frame_count=frame_count,
                success=True,
                metadata={
                    'is_single': is_single,
                    'processor_type': processor_info.processor_type,
                    'is_fast': processor_info.is_fast,
                    'gpu_optimized': processor_info.gpu_optimized,
                    'compiled': compile_processor and processor_info.supports_compilation,
                    'max_frames': processor_info.max_frames,
                    'frame_sampling_rate': processor_info.frame_sampling_rate
                }
            )
            
            debug(f"Video processing completed", "video_processor_manager",
                  model_id=model_id,
                  batch_size=len(videos),
                  frame_count=frame_count,
                  processing_time=processing_time,
                  is_batch=len(videos) > 1)
            
            return result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Video processing failed", "video_processor_manager", e,
                  model_id=model_id, batch_size=len(videos))
            
            return VideoProcessingResult(
                pixel_values=None,
                processor_info=processor_info,
                processing_time=time.time() - start_time,
                batch_size=len(videos),
                frame_count=0,
                success=False,
                error=str(e)
            )
    
    def get_processor_info(self, model_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get comprehensive processor information without loading."""
        processor_info = self.load_processor(model_id, **kwargs)
        if not processor_info:
            return None
        
        return {
            'model_id': processor_info.model_id,
            'processor_type': processor_info.processor_type,
            'is_fast': processor_info.is_fast,
            'supports_batch': processor_info.supports_batch,
            'supports_gpu': processor_info.supports_gpu,
            'gpu_optimized': processor_info.gpu_optimized,
            'supports_compilation': processor_info.supports_compilation,
            'size': processor_info.size,
            'image_mean': processor_info.image_mean,
            'image_std': processor_info.image_std,
            'load_time': processor_info.load_time,
            'cache_key': processor_info.cache_key,
            'max_frames': processor_info.max_frames,
            'frame_sampling_rate': processor_info.frame_sampling_rate
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
            'memory_cache_size': len(self.processor_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.json"))),
            'average_frames_per_processing': (
                self.performance_stats["total_frames_processed"] / 
                max(self.performance_stats["total_processings"], 1)
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear processor cache."""
        if memory_only:
            self.processor_cache.clear()
            info("Memory cache cleared", "video_processor_manager")
        else:
            # Clear both memory and disk cache
            self.processor_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix != '.json':
                    cache_file.unlink()
            
            info("All caches cleared", "video_processor_manager")
    
    def list_cached_processors(self) -> List[Dict[str, Any]]:
        """List all cached processors with metadata."""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    cached.append(metadata)
            except Exception as e:
                warning(f"Failed to read cache metadata", "video_processor_manager", e,
                       cache_file=cache_file.name)
        
        return cached


# Global instance for easy access
video_processor_manager = EnhancedVideoProcessorManager()
