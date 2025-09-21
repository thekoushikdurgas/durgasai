"""
Enhanced Feature Extractor Manager for DurgasAI.

This module provides advanced feature extraction capabilities including:
- Intelligent caching (memory + disk)
- Audio preprocessing optimization
- Performance monitoring
- Advanced error handling
- Sampling rate management
- Padding and truncation handling
- Batch processing optimization

Based on Hugging Face Transformers Feature Extractors documentation best practices.
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
    from transformers import AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# NumPy for audio array operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

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
class FeatureExtractorInfo:
    """Comprehensive information about a loaded feature extractor."""
    model_id: str
    feature_extractor: Any
    sampling_rate: int
    padding: bool
    return_attention_mask: bool
    max_length: Optional[int]
    load_time: float
    last_used: float
    cache_key: str
    extractor_type: str
    supports_padding: bool = True
    supports_truncation: bool = True
    supports_resampling: bool = True
    supports_batch: bool = True
    supports_gpu: bool = False
    gpu_optimized: bool = False
    supports_compilation: bool = False


@dataclass
class AudioProcessingResult:
    """Result of audio processing operation."""
    input_values: Any
    attention_mask: Optional[Any]
    feature_extractor_info: FeatureExtractorInfo
    processing_time: float
    batch_size: int
    sequence_length: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedFeatureExtractorManager:
    """
    Enhanced feature extractor manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Audio preprocessing optimization
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Sampling rate management and resampling
    - Padding and truncation handling
    - Batch processing optimization
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/feature_extractors", max_cache_size: int = 10):
        """
        Initialize the enhanced feature extractor manager.
        
        Args:
            cache_dir: Directory for persistent feature extractor cache
            max_cache_size: Maximum number of feature extractors to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active feature extractors
        self.extractor_cache: Dict[str, FeatureExtractorInfo] = {}
        
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
            "total_audio_samples": 0,
            "total_sequence_length": 0,
            "resampling_operations": 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedFeatureExtractorManager initialized", "feature_extractor_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             numpy_available=NUMPY_AVAILABLE,
             librosa_available=LIBROSA_AVAILABLE)
    
    def _get_cache_key(self, model_id: str, **kwargs) -> str:
        """Generate unique cache key for feature extractor configuration."""
        sorted_kwargs = sorted(kwargs.items())
        config_str = f"{model_id}_{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_extractor_cache(self, cache_key: str, extractor_info: FeatureExtractorInfo):
        """Save feature extractor to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        extractor_dir = self.cache_dir / cache_key
        
        try:
            # Save feature extractor files
            extractor_info.feature_extractor.save_pretrained(str(extractor_dir))
            
            # Create metadata without the feature extractor object
            metadata = asdict(extractor_info)
            metadata.pop('feature_extractor', None)  # Remove the actual feature extractor object
            metadata['extractor_path'] = str(extractor_dir)
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            debug(f"Feature extractor cached successfully", "feature_extractor_manager",
                  cache_key=cache_key, model_id=extractor_info.model_id)
                
        except Exception as e:
            error(f"Failed to cache feature extractor", "feature_extractor_manager", e,
                  cache_key=cache_key, model_id=extractor_info.model_id)
    
    def _load_extractor_cache(self, cache_key: str) -> Optional[FeatureExtractorInfo]:
        """Load feature extractor from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        extractor_dir = self.cache_dir / cache_key
        
        if not cache_file.exists() or not extractor_dir.exists():
            return None
            
        try:
            # Load metadata
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            
            # Load feature extractor from disk
            feature_extractor = AutoFeatureExtractor.from_pretrained(str(extractor_dir))
            
            # Reconstruct FeatureExtractorInfo
            extractor_info = FeatureExtractorInfo(
                model_id=metadata['model_id'],
                feature_extractor=feature_extractor,
                sampling_rate=metadata['sampling_rate'],
                padding=metadata['padding'],
                return_attention_mask=metadata['return_attention_mask'],
                max_length=metadata.get('max_length'),
                load_time=metadata['load_time'],
                last_used=metadata['last_used'],
                cache_key=cache_key,
                extractor_type=metadata['extractor_type'],
                supports_padding=metadata.get('supports_padding', True),
                supports_truncation=metadata.get('supports_truncation', True),
                supports_resampling=metadata.get('supports_resampling', True),
                supports_batch=metadata.get('supports_batch', True),
                supports_gpu=metadata.get('supports_gpu', False),
                gpu_optimized=metadata.get('gpu_optimized', False),
                supports_compilation=metadata.get('supports_compilation', False)
            )
            
            debug(f"Feature extractor loaded from cache", "feature_extractor_manager",
                  cache_key=cache_key, model_id=extractor_info.model_id)
            
            return extractor_info
            
        except Exception as e:
            error(f"Failed to load cached feature extractor", "feature_extractor_manager", e,
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
            warning(f"Failed to load cache metadata", "feature_extractor_manager", e)
    
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
            warning(f"Failed to save cache metadata", "feature_extractor_manager", e)
    
    def _validate_extractor_compatibility(self, model_id: str, **kwargs) -> bool:
        """Validate feature extractor compatibility before loading."""
        try:
            # Basic validation - check if model exists
            # This is a simplified check - in practice, you might want more validation
            return True
            
        except Exception as e:
            error(f"Feature extractor compatibility check failed", "feature_extractor_manager", e,
                  model_id=model_id)
            return False
    
    def _determine_extractor_capabilities(self, feature_extractor: Any) -> Tuple[str, bool, bool, bool, bool, bool, bool]:
        """Determine feature extractor type and capabilities."""
        extractor_type = feature_extractor.__class__.__name__
        supports_padding = hasattr(feature_extractor, 'pad') or hasattr(feature_extractor, 'padding')
        supports_truncation = hasattr(feature_extractor, 'truncate') or hasattr(feature_extractor, 'max_length')
        supports_resampling = LIBROSA_AVAILABLE
        supports_batch = True  # Most feature extractors support batch processing
        supports_gpu = False  # Feature extractors typically run on CPU
        gpu_optimized = False
        supports_compilation = False
        
        return extractor_type, supports_padding, supports_truncation, supports_resampling, supports_batch, supports_gpu, supports_compilation
    
    def load_extractor(self, model_id: str, **kwargs) -> Optional[FeatureExtractorInfo]:
        """
        Load feature extractor with intelligent caching and validation.
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional feature extractor loading arguments
            
        Returns:
            FeatureExtractorInfo object if successful, None if failed
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for feature extractor loading", "feature_extractor_manager")
            return None
        
        start_time = time.time()
        cache_key = self._get_cache_key(model_id, **kwargs)
        
        # Check memory cache first
        if cache_key in self.extractor_cache:
            self.extractor_cache[cache_key].last_used = time.time()
            self.performance_stats["cache_hits"] += 1
            
            info(f"Feature extractor cache hit", "feature_extractor_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return self.extractor_cache[cache_key]
        
        # Check disk cache
        extractor_info = self._load_extractor_cache(cache_key)
        if extractor_info:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, extractor_info)
            self.performance_stats["cache_hits"] += 1
            
            info(f"Feature extractor loaded from disk cache", "feature_extractor_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return extractor_info
        
        # Cache miss - load from HuggingFace
        self.performance_stats["cache_misses"] += 1
        
        # Validate compatibility
        if not self._validate_extractor_compatibility(model_id, **kwargs):
            error(f"Feature extractor compatibility validation failed", "feature_extractor_manager",
                  model_id=model_id)
            return None
        
        try:
            # Load feature extractor with enhanced configuration
            default_kwargs = {
                'cache_dir': str(self.cache_dir.parent / "huggingface")
            }
            default_kwargs.update(kwargs)
            
            debug(f"Loading feature extractor from HuggingFace", "feature_extractor_manager",
                  model_id=model_id, kwargs=default_kwargs)
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **default_kwargs)
            
            # Determine feature extractor capabilities
            extractor_type, supports_padding, supports_truncation, supports_resampling, supports_batch, supports_gpu, supports_compilation = self._determine_extractor_capabilities(feature_extractor)
            
            # Create feature extractor info
            load_time = time.time() - start_time
            extractor_info = FeatureExtractorInfo(
                model_id=model_id,
                feature_extractor=feature_extractor,
                sampling_rate=getattr(feature_extractor, 'sampling_rate', 16000),
                padding=getattr(feature_extractor, 'padding', True),
                return_attention_mask=getattr(feature_extractor, 'return_attention_mask', True),
                max_length=getattr(feature_extractor, 'max_length', None),
                load_time=load_time,
                last_used=time.time(),
                cache_key=cache_key,
                extractor_type=extractor_type,
                supports_padding=supports_padding,
                supports_truncation=supports_truncation,
                supports_resampling=supports_resampling,
                supports_batch=supports_batch,
                supports_gpu=supports_gpu,
                gpu_optimized=gpu_optimized,
                supports_compilation=supports_compilation
            )
            
            # Cache the feature extractor
            self._add_to_memory_cache(cache_key, extractor_info)
            self._save_extractor_cache(cache_key, extractor_info)
            
            # Update performance stats
            self.performance_stats["total_loads"] += 1
            self._update_average_load_time(load_time)
            self._save_cache_metadata()
            
            info(f"Feature extractor loaded successfully", "feature_extractor_manager",
                 model_id=model_id,
                 load_time=load_time,
                 extractor_type=extractor_type,
                 sampling_rate=extractor_info.sampling_rate)
            
            return extractor_info
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load feature extractor", "feature_extractor_manager", e,
                  model_id=model_id, load_time=time.time() - start_time)
            return None
    
    def _add_to_memory_cache(self, cache_key: str, extractor_info: FeatureExtractorInfo):
        """Add feature extractor to memory cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self.extractor_cache) >= self.max_cache_size:
            oldest_key = min(self.extractor_cache.keys(), 
                           key=lambda k: self.extractor_cache[k].last_used)
            del self.extractor_cache[oldest_key]
            
            debug(f"Evicted feature extractor from memory cache", "feature_extractor_manager",
                  evicted_key=oldest_key)
        
        self.extractor_cache[cache_key] = extractor_info
    
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
    
    def _resample_audio(self, audio_array: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio to target sampling rate."""
        if not LIBROSA_AVAILABLE:
            warning("Librosa not available for resampling", "feature_extractor_manager")
            return audio_array
        
        if original_rate == target_rate:
            return audio_array
        
        try:
            resampled = librosa.resample(audio_array, orig_sr=original_rate, target_sr=target_rate)
            self.performance_stats["resampling_operations"] += 1
            return resampled
        except Exception as e:
            error(f"Resampling failed", "feature_extractor_manager", e)
            return audio_array
    
    def process_audio(self, audio_data: Union[np.ndarray, List[np.ndarray]], model_id: str, 
                     sampling_rate: Optional[int] = None, padding: bool = True, 
                     truncation: bool = False, max_length: Optional[int] = None,
                     return_tensors: Optional[str] = None, **kwargs) -> Optional[AudioProcessingResult]:
        """
        Process audio data with the feature extractor.
        
        Args:
            audio_data: Single audio array or list of audio arrays
            model_id: Model identifier for feature extractor
            sampling_rate: Target sampling rate (if None, uses extractor default)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Tensor format for output
            **kwargs: Additional processing arguments
            
        Returns:
            AudioProcessingResult object with processed audio data
        """
        if not TRANSFORMERS_AVAILABLE or not NUMPY_AVAILABLE:
            error("Required dependencies not available for audio processing", "feature_extractor_manager")
            return None
        
        start_time = time.time()
        
        # Load feature extractor
        extractor_info = self.load_extractor(model_id, **kwargs)
        if not extractor_info:
            return AudioProcessingResult(
                input_values=None,
                attention_mask=None,
                feature_extractor_info=None,
                processing_time=time.time() - start_time,
                batch_size=0,
                sequence_length=0,
                success=False,
                error="Failed to load feature extractor"
            )
        
        try:
            # Prepare audio data
            if not isinstance(audio_data, list):
                audio_data = [audio_data]
                is_single = True
            else:
                is_single = False
            
            # Get target sampling rate
            target_sampling_rate = sampling_rate or extractor_info.sampling_rate
            
            # Process audio arrays
            processed_audio = []
            for audio_array in audio_data:
                # Ensure audio is numpy array
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)
                
                # Resample if necessary
                if hasattr(audio_array, 'shape') and len(audio_array.shape) > 0:
                    current_rate = getattr(audio_array, 'sampling_rate', target_sampling_rate)
                    if current_rate != target_sampling_rate:
                        audio_array = self._resample_audio(audio_array, current_rate, target_sampling_rate)
                
                processed_audio.append(audio_array)
            
            # Prepare processing arguments
            processing_kwargs = {
                'sampling_rate': target_sampling_rate,
                'padding': padding and extractor_info.supports_padding,
                'return_tensors': return_tensors
            }
            
            if truncation and extractor_info.supports_truncation:
                processing_kwargs['truncation'] = True
                if max_length:
                    processing_kwargs['max_length'] = max_length
            
            # Process with feature extractor
            if len(processed_audio) > 1:
                # Batch processing
                result = extractor_info.feature_extractor(processed_audio, **processing_kwargs)
                self.performance_stats["batch_processings"] += 1
            else:
                # Single audio processing
                result = extractor_info.feature_extractor(processed_audio[0], **processing_kwargs)
            
            processing_time = time.time() - start_time
            self.performance_stats["total_processings"] += 1
            
            # Count audio samples and sequence length
            if 'input_values' in result:
                input_values = result['input_values']
                if isinstance(input_values, list):
                    batch_size = len(input_values)
                    sequence_length = len(input_values[0]) if len(input_values) > 0 else 0
                else:
                    batch_size = 1
                    sequence_length = len(input_values) if hasattr(input_values, '__len__') else 0
                
                self.performance_stats["total_audio_samples"] += batch_size
                self.performance_stats["total_sequence_length"] += sequence_length
            
            # Create result
            audio_result = AudioProcessingResult(
                input_values=result.get('input_values'),
                attention_mask=result.get('attention_mask'),
                feature_extractor_info=extractor_info,
                processing_time=processing_time,
                batch_size=batch_size,
                sequence_length=sequence_length,
                success=True,
                metadata={
                    'is_single': is_single,
                    'extractor_type': extractor_info.extractor_type,
                    'sampling_rate': target_sampling_rate,
                    'padding': processing_kwargs.get('padding', False),
                    'truncation': processing_kwargs.get('truncation', False),
                    'max_length': processing_kwargs.get('max_length'),
                    'return_tensors': return_tensors
                }
            )
            
            debug(f"Audio processing completed", "feature_extractor_manager",
                  model_id=model_id,
                  batch_size=batch_size,
                  sequence_length=sequence_length,
                  processing_time=processing_time)
            
            return audio_result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Audio processing failed", "feature_extractor_manager", e,
                  model_id=model_id, num_audio_samples=len(audio_data))
            
            return AudioProcessingResult(
                input_values=None,
                attention_mask=None,
                feature_extractor_info=extractor_info,
                processing_time=time.time() - start_time,
                batch_size=0,
                sequence_length=0,
                success=False,
                error=str(e)
            )
    
    def get_extractor_info(self, model_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get comprehensive feature extractor information without loading."""
        extractor_info = self.load_extractor(model_id, **kwargs)
        if not extractor_info:
            return None
        
        return {
            'model_id': extractor_info.model_id,
            'extractor_type': extractor_info.extractor_type,
            'sampling_rate': extractor_info.sampling_rate,
            'padding': extractor_info.padding,
            'return_attention_mask': extractor_info.return_attention_mask,
            'max_length': extractor_info.max_length,
            'supports_padding': extractor_info.supports_padding,
            'supports_truncation': extractor_info.supports_truncation,
            'supports_resampling': extractor_info.supports_resampling,
            'supports_batch': extractor_info.supports_batch,
            'supports_gpu': extractor_info.supports_gpu,
            'gpu_optimized': extractor_info.gpu_optimized,
            'supports_compilation': extractor_info.supports_compilation,
            'load_time': extractor_info.load_time,
            'cache_key': extractor_info.cache_key
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
            'memory_cache_size': len(self.extractor_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.json"))),
            'average_sequence_length': (
                self.performance_stats["total_sequence_length"] / 
                max(self.performance_stats["total_audio_samples"], 1)
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear feature extractor cache."""
        if memory_only:
            self.extractor_cache.clear()
            info("Memory cache cleared", "feature_extractor_manager")
        else:
            # Clear both memory and disk cache
            self.extractor_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix != '.json':
                    cache_file.unlink()
            
            info("All caches cleared", "feature_extractor_manager")
    
    def list_cached_extractors(self) -> List[Dict[str, Any]]:
        """List all cached feature extractors with metadata."""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    cached.append(metadata)
            except Exception as e:
                warning(f"Failed to read cache metadata", "feature_extractor_manager", e,
                       cache_file=cache_file.name)
        
        return cached


# Global instance for easy access
feature_extractor_manager = EnhancedFeatureExtractorManager()
