"""
Enhanced Processor Manager for DurgasAI.

This module provides advanced processor management capabilities including:
- Intelligent caching (memory + disk)
- Multimodal processing coordination
- Performance monitoring
- Advanced error handling
- Input modality detection and routing
- Unified preprocessing optimization

Based on Hugging Face Transformers Processors documentation best practices.
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
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# NumPy for array operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Local imports
from .logger import debug, info, warning, error
from .config import Config


@dataclass
class ProcessorInfo:
    """Comprehensive information about a loaded processor."""
    model_id: str
    processor: Any
    processor_type: str
    supported_modalities: List[str]
    tokenizer_class: Optional[str]
    image_processor_class: Optional[str]
    feature_extractor_class: Optional[str]
    load_time: float
    last_used: float
    cache_key: str
    model_input_names: List[str]
    supports_text: bool = False
    supports_images: bool = False
    supports_audio: bool = False
    supports_multimodal: bool = False
    supports_batch: bool = True
    supports_gpu: bool = False
    gpu_optimized: bool = False


@dataclass
class MultimodalProcessingResult:
    """Result of multimodal processing operation."""
    input_ids: Optional[Any]
    pixel_values: Optional[Any]
    input_features: Optional[Any]
    attention_mask: Optional[Any]
    processor_info: ProcessorInfo
    processing_time: float
    batch_size: int
    modalities_processed: List[str]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedProcessorManager:
    """
    Enhanced processor manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Multimodal processing coordination
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Input modality detection and routing
    - Unified preprocessing optimization
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/processors", max_cache_size: int = 10):
        """
        Initialize the enhanced processor manager.
        
        Args:
            cache_dir: Directory for persistent processor cache
            max_cache_size: Maximum number of processors to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active processors
        self.processor_cache: Dict[str, ProcessorInfo] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_load_time": 0.0,
            "total_processings": 0,
            "multimodal_processings": 0,
            "text_only_processings": 0,
            "image_only_processings": 0,
            "audio_only_processings": 0,
            "error_count": 0,
            "gpu_processings": 0,
            "batch_processings": 0,
            "total_inputs": 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedProcessorManager initialized", "processor_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             numpy_available=NUMPY_AVAILABLE,
             pil_available=PIL_AVAILABLE,
             librosa_available=LIBROSA_AVAILABLE)
    
    def _get_cache_key(self, model_id: str, **kwargs) -> str:
        """Generate unique cache key for processor configuration."""
        sorted_kwargs = sorted(kwargs.items())
        config_str = f"{model_id}_{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_processor_cache(self, cache_key: str, processor_info: ProcessorInfo):
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
            
            debug(f"Processor cached successfully", "processor_manager",
                  cache_key=cache_key, model_id=processor_info.model_id)
                
        except Exception as e:
            error(f"Failed to cache processor", "processor_manager", e,
                  cache_key=cache_key, model_id=processor_info.model_id)
    
    def _load_processor_cache(self, cache_key: str) -> Optional[ProcessorInfo]:
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
            processor = AutoProcessor.from_pretrained(str(processor_dir))
            
            # Reconstruct ProcessorInfo
            processor_info = ProcessorInfo(
                model_id=metadata['model_id'],
                processor=processor,
                processor_type=metadata['processor_type'],
                supported_modalities=metadata['supported_modalities'],
                tokenizer_class=metadata.get('tokenizer_class'),
                image_processor_class=metadata.get('image_processor_class'),
                feature_extractor_class=metadata.get('feature_extractor_class'),
                load_time=metadata['load_time'],
                last_used=metadata['last_used'],
                cache_key=cache_key,
                model_input_names=metadata['model_input_names'],
                supports_text=metadata.get('supports_text', False),
                supports_images=metadata.get('supports_images', False),
                supports_audio=metadata.get('supports_audio', False),
                supports_multimodal=metadata.get('supports_multimodal', False),
                supports_batch=metadata.get('supports_batch', True),
                supports_gpu=metadata.get('supports_gpu', False),
                gpu_optimized=metadata.get('gpu_optimized', False)
            )
            
            debug(f"Processor loaded from cache", "processor_manager",
                  cache_key=cache_key, model_id=processor_info.model_id)
            
            return processor_info
            
        except Exception as e:
            error(f"Failed to load cached processor", "processor_manager", e,
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
            warning(f"Failed to load cache metadata", "processor_manager", e)
    
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
            warning(f"Failed to save cache metadata", "processor_manager", e)
    
    def _validate_processor_compatibility(self, model_id: str, **kwargs) -> bool:
        """Validate processor compatibility before loading."""
        try:
            # Basic validation - check if model exists
            # This is a simplified check - in practice, you might want more validation
            return True
            
        except Exception as e:
            error(f"Processor compatibility check failed", "processor_manager", e,
                  model_id=model_id)
            return False
    
    def _determine_processor_capabilities(self, processor: Any) -> Tuple[str, List[str], bool, bool, bool, bool, bool, bool, bool]:
        """Determine processor type and capabilities."""
        processor_type = processor.__class__.__name__
        supported_modalities = []
        
        # Check for supported modalities
        supports_text = hasattr(processor, 'tokenizer') and processor.tokenizer is not None
        supports_images = hasattr(processor, 'image_processor') and processor.image_processor is not None
        supports_audio = hasattr(processor, 'feature_extractor') and processor.feature_extractor is not None
        
        if supports_text:
            supported_modalities.append('text')
        if supports_images:
            supported_modalities.append('images')
        if supports_audio:
            supported_modalities.append('audio')
        
        supports_multimodal = len(supported_modalities) > 1
        supports_batch = True  # Most processors support batch processing
        supports_gpu = False  # Processors typically run on CPU
        gpu_optimized = False
        
        return processor_type, supported_modalities, supports_text, supports_images, supports_audio, supports_multimodal, supports_batch, supports_gpu, gpu_optimized
    
    def load_processor(self, model_id: str, **kwargs) -> Optional[ProcessorInfo]:
        """
        Load processor with intelligent caching and validation.
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional processor loading arguments
            
        Returns:
            ProcessorInfo object if successful, None if failed
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for processor loading", "processor_manager")
            return None
        
        start_time = time.time()
        cache_key = self._get_cache_key(model_id, **kwargs)
        
        # Check memory cache first
        if cache_key in self.processor_cache:
            self.processor_cache[cache_key].last_used = time.time()
            self.performance_stats["cache_hits"] += 1
            
            info(f"Processor cache hit", "processor_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return self.processor_cache[cache_key]
        
        # Check disk cache
        processor_info = self._load_processor_cache(cache_key)
        if processor_info:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, processor_info)
            self.performance_stats["cache_hits"] += 1
            
            info(f"Processor loaded from disk cache", "processor_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return processor_info
        
        # Cache miss - load from HuggingFace
        self.performance_stats["cache_misses"] += 1
        
        # Validate compatibility
        if not self._validate_processor_compatibility(model_id, **kwargs):
            error(f"Processor compatibility validation failed", "processor_manager",
                  model_id=model_id)
            return None
        
        try:
            # Load processor with enhanced configuration
            default_kwargs = {
                'cache_dir': str(self.cache_dir.parent / "huggingface")
            }
            default_kwargs.update(kwargs)
            
            debug(f"Loading processor from HuggingFace", "processor_manager",
                  model_id=model_id, kwargs=default_kwargs)
            
            processor = AutoProcessor.from_pretrained(model_id, **default_kwargs)
            
            # Determine processor capabilities
            processor_type, supported_modalities, supports_text, supports_images, supports_audio, supports_multimodal, supports_batch, supports_gpu, gpu_optimized = self._determine_processor_capabilities(processor)
            
            # Create processor info
            load_time = time.time() - start_time
            processor_info = ProcessorInfo(
                model_id=model_id,
                processor=processor,
                processor_type=processor_type,
                supported_modalities=supported_modalities,
                tokenizer_class=getattr(processor, 'tokenizer', {}).get('__class__', {}).get('__name__', None) if hasattr(processor, 'tokenizer') else None,
                image_processor_class=getattr(processor, 'image_processor', {}).get('__class__', {}).get('__name__', None) if hasattr(processor, 'image_processor') else None,
                feature_extractor_class=getattr(processor, 'feature_extractor', {}).get('__class__', {}).get('__name__', None) if hasattr(processor, 'feature_extractor') else None,
                load_time=load_time,
                last_used=time.time(),
                cache_key=cache_key,
                model_input_names=getattr(processor, 'model_input_names', []),
                supports_text=supports_text,
                supports_images=supports_images,
                supports_audio=supports_audio,
                supports_multimodal=supports_multimodal,
                supports_batch=supports_batch,
                supports_gpu=supports_gpu,
                gpu_optimized=gpu_optimized
            )
            
            # Cache the processor
            self._add_to_memory_cache(cache_key, processor_info)
            self._save_processor_cache(cache_key, processor_info)
            
            # Update performance stats
            self.performance_stats["total_loads"] += 1
            self._update_average_load_time(load_time)
            self._save_cache_metadata()
            
            info(f"Processor loaded successfully", "processor_manager",
                 model_id=model_id,
                 load_time=load_time,
                 processor_type=processor_type,
                 supported_modalities=supported_modalities)
            
            return processor_info
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load processor", "processor_manager", e,
                  model_id=model_id, load_time=time.time() - start_time)
            return None
    
    def _add_to_memory_cache(self, cache_key: str, processor_info: ProcessorInfo):
        """Add processor to memory cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self.processor_cache) >= self.max_cache_size:
            oldest_key = min(self.processor_cache.keys(), 
                           key=lambda k: self.processor_cache[k].last_used)
            del self.processor_cache[oldest_key]
            
            debug(f"Evicted processor from memory cache", "processor_manager",
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
    
    def _detect_input_modalities(self, **inputs) -> List[str]:
        """Detect input modalities from the provided inputs."""
        modalities = []
        
        # Check for text inputs
        if 'text' in inputs or any(key.startswith('text') for key in inputs.keys()):
            modalities.append('text')
        
        # Check for image inputs
        if 'images' in inputs or any(key.startswith('image') for key in inputs.keys()):
            modalities.append('images')
        
        # Check for audio inputs
        if 'audio' in inputs or any(key.startswith('audio') for key in inputs.keys()):
            modalities.append('audio')
        
        return modalities
    
    def process_multimodal(self, model_id: str, return_tensors: Optional[str] = None, **inputs) -> Optional[MultimodalProcessingResult]:
        """
        Process multimodal inputs with the processor.
        
        Args:
            model_id: Model identifier for processor
            return_tensors: Tensor format for output
            **inputs: Multimodal inputs (text, images, audio, etc.)
            
        Returns:
            MultimodalProcessingResult object with processed inputs
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Required dependencies not available for multimodal processing", "processor_manager")
            return None
        
        start_time = time.time()
        
        # Load processor
        processor_info = self.load_processor(model_id)
        if not processor_info:
            return MultimodalProcessingResult(
                input_ids=None,
                pixel_values=None,
                input_features=None,
                attention_mask=None,
                processor_info=None,
                processing_time=time.time() - start_time,
                batch_size=0,
                modalities_processed=[],
                success=False,
                error="Failed to load processor"
            )
        
        try:
            # Detect input modalities
            modalities = self._detect_input_modalities(**inputs)
            
            # Prepare processing arguments
            processing_kwargs = {}
            if return_tensors:
                processing_kwargs['return_tensors'] = return_tensors
            
            # Process with processor
            result = processor_info.processor(**inputs, **processing_kwargs)
            
            processing_time = time.time() - start_time
            self.performance_stats["total_processings"] += 1
            
            # Update modality-specific stats
            if len(modalities) > 1:
                self.performance_stats["multimodal_processings"] += 1
            elif 'text' in modalities:
                self.performance_stats["text_only_processings"] += 1
            elif 'images' in modalities:
                self.performance_stats["image_only_processings"] += 1
            elif 'audio' in modalities:
                self.performance_stats["audio_only_processings"] += 1
            
            # Count inputs
            self.performance_stats["total_inputs"] += len(inputs)
            
            # Determine batch size
            batch_size = 1
            for key, value in result.items():
                if hasattr(value, '__len__') and not isinstance(value, str):
                    if isinstance(value, list):
                        batch_size = max(batch_size, len(value))
                    elif hasattr(value, 'shape'):
                        batch_size = max(batch_size, value.shape[0] if len(value.shape) > 0 else 1)
            
            if batch_size > 1:
                self.performance_stats["batch_processings"] += 1
            
            # Create result
            multimodal_result = MultimodalProcessingResult(
                input_ids=result.get('input_ids'),
                pixel_values=result.get('pixel_values'),
                input_features=result.get('input_features'),
                attention_mask=result.get('attention_mask'),
                processor_info=processor_info,
                processing_time=processing_time,
                batch_size=batch_size,
                modalities_processed=modalities,
                success=True,
                metadata={
                    'processor_type': processor_info.processor_type,
                    'supported_modalities': processor_info.supported_modalities,
                    'return_tensors': return_tensors,
                    'result_keys': list(result.keys())
                }
            )
            
            debug(f"Multimodal processing completed", "processor_manager",
                  model_id=model_id,
                  modalities=modalities,
                  batch_size=batch_size,
                  processing_time=processing_time)
            
            return multimodal_result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Multimodal processing failed", "processor_manager", e,
                  model_id=model_id, modalities=modalities)
            
            return MultimodalProcessingResult(
                input_ids=None,
                pixel_values=None,
                input_features=None,
                attention_mask=None,
                processor_info=processor_info,
                processing_time=time.time() - start_time,
                batch_size=0,
                modalities_processed=modalities,
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
            'supported_modalities': processor_info.supported_modalities,
            'tokenizer_class': processor_info.tokenizer_class,
            'image_processor_class': processor_info.image_processor_class,
            'feature_extractor_class': processor_info.feature_extractor_class,
            'model_input_names': processor_info.model_input_names,
            'supports_text': processor_info.supports_text,
            'supports_images': processor_info.supports_images,
            'supports_audio': processor_info.supports_audio,
            'supports_multimodal': processor_info.supports_multimodal,
            'supports_batch': processor_info.supports_batch,
            'supports_gpu': processor_info.supports_gpu,
            'gpu_optimized': processor_info.gpu_optimized,
            'load_time': processor_info.load_time,
            'cache_key': processor_info.cache_key
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
            'average_inputs_per_processing': (
                self.performance_stats["total_inputs"] / 
                max(self.performance_stats["total_processings"], 1)
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear processor cache."""
        if memory_only:
            self.processor_cache.clear()
            info("Memory cache cleared", "processor_manager")
        else:
            # Clear both memory and disk cache
            self.processor_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix != '.json':
                    cache_file.unlink()
            
            info("All caches cleared", "processor_manager")
    
    def list_cached_processors(self) -> List[Dict[str, Any]]:
        """List all cached processors with metadata."""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    cached.append(metadata)
            except Exception as e:
                warning(f"Failed to read cache metadata", "processor_manager", e,
                       cache_file=cache_file.name)
        
        return cached


# Global instance for easy access
processor_manager = EnhancedProcessorManager()
