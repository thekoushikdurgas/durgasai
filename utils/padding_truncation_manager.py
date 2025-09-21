"""
Enhanced Padding and Truncation Manager for DurgasAI.

This module provides advanced padding and truncation capabilities including:
- Intelligent caching (memory + disk)
- Advanced padding and truncation strategies
- Performance monitoring
- Advanced error handling
- Memory optimization
- Batch processing optimization
- Strategy validation and testing

Based on Hugging Face Transformers Padding and Truncation documentation best practices.
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
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# NumPy for array operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# PyTorch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local imports
from .logger import debug, info, warning, error
from .config import Config


@dataclass
class PaddingTruncationConfig:
    """Configuration for padding and truncation strategies."""
    padding: Union[bool, str] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: Optional[str] = None
    return_attention_mask: bool = True
    return_length: bool = False
    padding_side: str = "right"
    truncation_side: str = "right"
    stride: int = 0
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False


@dataclass
class PaddingTruncationResult:
    """Result of padding and truncation operation."""
    input_ids: Any
    attention_mask: Optional[Any]
    token_type_ids: Optional[Any]
    length: Optional[int]
    overflowing_tokens: Optional[List[Any]]
    special_tokens_mask: Optional[Any]
    offsets_mapping: Optional[Any]
    config: PaddingTruncationConfig
    processing_time: float
    batch_size: int
    sequence_length: int
    padding_applied: bool
    truncation_applied: bool
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StrategyAnalysisResult:
    """Result of strategy analysis operation."""
    strategy_name: str
    config: PaddingTruncationConfig
    memory_usage: float
    processing_time: float
    sequence_length: int
    batch_size: int
    padding_tokens_added: int
    truncation_tokens_removed: int
    efficiency_score: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedPaddingTruncationManager:
    """
    Enhanced padding and truncation manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Advanced padding and truncation strategies
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Memory optimization and batch processing
    - Strategy validation and testing
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/padding_truncation", max_cache_size: int = 10):
        """
        Initialize the enhanced padding and truncation manager.
        
        Args:
            cache_dir: Directory for persistent padding/truncation cache
            max_cache_size: Maximum number of configurations to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active configurations
        self.config_cache: Dict[str, PaddingTruncationConfig] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0,
            "total_tokens_processed": 0,
            "total_padding_tokens": 0,
            "total_truncation_tokens": 0,
            "error_count": 0,
            "strategy_comparisons": 0,
            "memory_optimizations": 0,
            "batch_optimizations": 0
        }
        
        # Predefined strategies
        self.predefined_strategies = {
            "training": PaddingTruncationConfig(
                padding=True,
                truncation=True,
                max_length=None,
                return_tensors="pt",
                return_attention_mask=True
            ),
            "inference": PaddingTruncationConfig(
                padding=True,
                truncation=True,
                max_length=None,
                return_tensors="pt",
                return_attention_mask=True
            ),
            "long_documents": PaddingTruncationConfig(
                padding=True,
                truncation='only_first',
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True
            ),
            "short_sequences": PaddingTruncationConfig(
                padding='max_length',
                truncation=False,
                max_length=128,
                return_tensors="pt",
                return_attention_mask=True
            ),
            "memory_constrained": PaddingTruncationConfig(
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                return_attention_mask=True
            ),
            "no_padding": PaddingTruncationConfig(
                padding=False,
                truncation=False,
                return_tensors=None,
                return_attention_mask=False
            ),
            "fixed_length": PaddingTruncationConfig(
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True
            )
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedPaddingTruncationManager initialized", "padding_truncation_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE,
             numpy_available=NUMPY_AVAILABLE,
             torch_available=TORCH_AVAILABLE)
    
    def _get_cache_key(self, config: PaddingTruncationConfig) -> str:
        """Generate unique cache key for configuration."""
        config_dict = asdict(config)
        # Sort keys for consistent hashing
        sorted_config = {k: config_dict[k] for k in sorted(config_dict.keys())}
        config_str = json.dumps(sorted_config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_config_cache(self, cache_key: str, config: PaddingTruncationConfig):
        """Save configuration to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Create metadata
            metadata = asdict(config)
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            debug(f"Configuration cached successfully", "padding_truncation_manager",
                  cache_key=cache_key)
                
        except Exception as e:
            error(f"Failed to cache configuration", "padding_truncation_manager", e,
                  cache_key=cache_key)
    
    def _load_config_cache(self, cache_key: str) -> Optional[PaddingTruncationConfig]:
        """Load configuration from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            # Load metadata
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            
            # Reconstruct PaddingTruncationConfig
            config = PaddingTruncationConfig(**metadata)
            
            debug(f"Configuration loaded from cache", "padding_truncation_manager",
                  cache_key=cache_key)
            
            return config
            
        except Exception as e:
            error(f"Failed to load cached configuration", "padding_truncation_manager", e,
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
            warning(f"Failed to load cache metadata", "padding_truncation_manager", e)
    
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
            warning(f"Failed to save cache metadata", "padding_truncation_manager", e)
    
    def _validate_configuration(self, config: PaddingTruncationConfig) -> bool:
        """Validate padding and truncation configuration."""
        try:
            # Validate padding parameter
            if isinstance(config.padding, str):
                valid_padding = ['longest', 'max_length', 'do_not_pad']
                if config.padding not in valid_padding:
                    error(f"Invalid padding value: {config.padding}", "padding_truncation_manager")
                    return False
            
            # Validate truncation parameter
            if isinstance(config.truncation, str):
                valid_truncation = ['longest_first', 'only_first', 'only_second', 'do_not_truncate']
                if config.truncation not in valid_truncation:
                    error(f"Invalid truncation value: {config.truncation}", "padding_truncation_manager")
                    return False
            
            # Validate max_length
            if config.max_length is not None and config.max_length <= 0:
                error(f"Invalid max_length: {config.max_length}", "padding_truncation_manager")
                return False
            
            # Validate return_tensors
            if config.return_tensors is not None:
                valid_tensors = ['pt', 'tf', 'np', None]
                if config.return_tensors not in valid_tensors:
                    error(f"Invalid return_tensors: {config.return_tensors}", "padding_truncation_manager")
                    return False
            
            return True
            
        except Exception as e:
            error(f"Configuration validation failed", "padding_truncation_manager", e)
            return False
    
    def _calculate_memory_usage(self, batch_size: int, sequence_length: int, return_tensors: str) -> float:
        """Calculate estimated memory usage for the operation."""
        # Rough estimation based on tensor size
        if return_tensors == 'pt':
            # PyTorch tensor: 4 bytes per float32, plus overhead
            bytes_per_token = 4
        elif return_tensors == 'tf':
            # TensorFlow tensor: similar to PyTorch
            bytes_per_token = 4
        elif return_tensors == 'np':
            # NumPy array: similar
            bytes_per_token = 4
        else:
            # List: more overhead
            bytes_per_token = 8
        
        total_tokens = batch_size * sequence_length
        memory_bytes = total_tokens * bytes_per_token
        
        # Add overhead for metadata
        memory_bytes += batch_size * 100  # Attention masks, etc.
        
        return memory_bytes / (1024 * 1024)  # Convert to MB
    
    def _optimize_configuration(self, config: PaddingTruncationConfig, batch_size: int, max_sequence_length: int) -> PaddingTruncationConfig:
        """Optimize configuration based on batch size and sequence length."""
        optimized_config = asdict(config)
        
        # Memory optimization
        if batch_size > 32 and max_sequence_length > 512:
            # Large batch, long sequences - use memory-efficient settings
            if config.max_length is None or config.max_length > 512:
                optimized_config['max_length'] = 512
            self.performance_stats["memory_optimizations"] += 1
        
        # Batch optimization
        if batch_size > 16:
            # Large batch - ensure consistent padding
            if config.padding is False:
                optimized_config['padding'] = True
            self.performance_stats["batch_optimizations"] += 1
        
        return PaddingTruncationConfig(**optimized_config)
    
    def apply_padding_truncation(self, tokenizer: Any, texts: Union[str, List[str]], 
                               config: PaddingTruncationConfig) -> Optional[PaddingTruncationResult]:
        """
        Apply padding and truncation to texts using the specified configuration.
        
        Args:
            tokenizer: HuggingFace tokenizer
            texts: Text or list of texts to process
            config: Padding and truncation configuration
            
        Returns:
            PaddingTruncationResult object with processed data
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for padding and truncation", "padding_truncation_manager")
            return None
        
        start_time = time.time()
        
        # Validate configuration
        if not self._validate_configuration(config):
            return PaddingTruncationResult(
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                length=None,
                overflowing_tokens=None,
                special_tokens_mask=None,
                offsets_mapping=None,
                config=config,
                processing_time=time.time() - start_time,
                batch_size=0,
                sequence_length=0,
                padding_applied=False,
                truncation_applied=False,
                success=False,
                error="Invalid configuration"
            )
        
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            batch_size = len(texts)
            
            # Calculate max sequence length for optimization
            max_sequence_length = 0
            if config.max_length is None:
                # Estimate max length from texts
                for text in texts:
                    estimated_length = len(text.split()) * 1.3  # Rough estimation
                    max_sequence_length = max(max_sequence_length, int(estimated_length))
            else:
                max_sequence_length = config.max_length
            
            # Optimize configuration
            optimized_config = self._optimize_configuration(config, batch_size, max_sequence_length)
            
            # Prepare tokenizer arguments
            tokenizer_kwargs = {
                'padding': optimized_config.padding,
                'truncation': optimized_config.truncation,
                'return_tensors': optimized_config.return_tensors,
                'return_attention_mask': optimized_config.return_attention_mask,
                'return_length': optimized_config.return_length,
                'return_overflowing_tokens': optimized_config.return_overflowing_tokens,
                'return_special_tokens_mask': optimized_config.return_special_tokens_mask,
                'return_offsets_mapping': optimized_config.return_offsets_mapping
            }
            
            # Add optional parameters
            if optimized_config.max_length is not None:
                tokenizer_kwargs['max_length'] = optimized_config.max_length
            if optimized_config.pad_to_multiple_of is not None:
                tokenizer_kwargs['pad_to_multiple_of'] = optimized_config.pad_to_multiple_of
            if optimized_config.stride > 0:
                tokenizer_kwargs['stride'] = optimized_config.stride
            
            # Apply padding and truncation
            result = tokenizer(texts, **tokenizer_kwargs)
            
            processing_time = time.time() - start_time
            
            # Analyze results
            if 'input_ids' in result:
                input_ids = result['input_ids']
                if TORCH_AVAILABLE and hasattr(input_ids, 'shape'):
                    sequence_length = input_ids.shape[1] if len(input_ids.shape) > 1 else len(input_ids)
                else:
                    sequence_length = len(input_ids[0]) if isinstance(input_ids, list) else len(input_ids)
            else:
                sequence_length = 0
            
            # Count padding and truncation tokens
            padding_tokens_added = 0
            truncation_tokens_removed = 0
            
            if 'input_ids' in result and hasattr(tokenizer, 'pad_token_id'):
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is not None:
                    input_ids = result['input_ids']
                    if TORCH_AVAILABLE and hasattr(input_ids, 'eq'):
                        padding_tokens_added = int(input_ids.eq(pad_token_id).sum())
                    else:
                        # Fallback for non-tensor inputs
                        padding_tokens_added = sum(1 for seq in input_ids for token in seq if token == pad_token_id)
            
            # Update performance stats
            self.performance_stats["total_operations"] += 1
            self.performance_stats["total_tokens_processed"] += batch_size * sequence_length
            self.performance_stats["total_padding_tokens"] += padding_tokens_added
            self._update_average_processing_time(processing_time)
            
            # Determine if padding/truncation was applied
            padding_applied = optimized_config.padding is not False and padding_tokens_added > 0
            truncation_applied = optimized_config.truncation is not False and sequence_length < max_sequence_length
            
            # Create result
            padding_truncation_result = PaddingTruncationResult(
                input_ids=result.get('input_ids'),
                attention_mask=result.get('attention_mask'),
                token_type_ids=result.get('token_type_ids'),
                length=result.get('length'),
                overflowing_tokens=result.get('overflowing_tokens'),
                special_tokens_mask=result.get('special_tokens_mask'),
                offsets_mapping=result.get('offsets_mapping'),
                config=optimized_config,
                processing_time=processing_time,
                batch_size=batch_size,
                sequence_length=sequence_length,
                padding_applied=padding_applied,
                truncation_applied=truncation_applied,
                success=True,
                metadata={
                    'memory_usage_mb': self._calculate_memory_usage(batch_size, sequence_length, optimized_config.return_tensors),
                    'padding_tokens_added': padding_tokens_added,
                    'truncation_tokens_removed': truncation_tokens_removed,
                    'efficiency_score': self._calculate_efficiency_score(processing_time, batch_size, sequence_length)
                }
            )
            
            debug(f"Padding and truncation completed", "padding_truncation_manager",
                  batch_size=batch_size,
                  sequence_length=sequence_length,
                  processing_time=processing_time,
                  padding_applied=padding_applied,
                  truncation_applied=truncation_applied)
            
            return padding_truncation_result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Padding and truncation failed", "padding_truncation_manager", e,
                  batch_size=batch_size, processing_time=time.time() - start_time)
            
            return PaddingTruncationResult(
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                length=None,
                overflowing_tokens=None,
                special_tokens_mask=None,
                offsets_mapping=None,
                config=config,
                processing_time=time.time() - start_time,
                batch_size=batch_size if 'batch_size' in locals() else 0,
                sequence_length=0,
                padding_applied=False,
                truncation_applied=False,
                success=False,
                error=str(e)
            )
    
    def _update_average_processing_time(self, processing_time: float):
        """Update running average of processing times."""
        total_operations = self.performance_stats["total_operations"]
        if total_operations == 1:
            self.performance_stats["average_processing_time"] = processing_time
        else:
            # Running average
            current_avg = self.performance_stats["average_processing_time"]
            self.performance_stats["average_processing_time"] = (
                (current_avg * (total_operations - 1) + processing_time) / total_operations
            )
    
    def _calculate_efficiency_score(self, processing_time: float, batch_size: int, sequence_length: int) -> float:
        """Calculate efficiency score based on processing time and throughput."""
        if processing_time == 0:
            return 1.0
        
        tokens_per_second = (batch_size * sequence_length) / processing_time
        
        # Normalize score (higher is better, max around 1000 tokens/sec)
        efficiency_score = min(tokens_per_second / 1000.0, 1.0)
        
        return efficiency_score
    
    def compare_strategies(self, tokenizer: Any, texts: Union[str, List[str]], 
                         strategy_names: List[str]) -> Dict[str, StrategyAnalysisResult]:
        """
        Compare different padding and truncation strategies.
        
        Args:
            tokenizer: HuggingFace tokenizer
            texts: Text or list of texts to process
            strategy_names: List of strategy names to compare
            
        Returns:
            Dictionary mapping strategy names to analysis results
        """
        results = {}
        
        for strategy_name in strategy_names:
            if strategy_name in self.predefined_strategies:
                config = self.predefined_strategies[strategy_name]
                
                try:
                    result = self.apply_padding_truncation(tokenizer, texts, config)
                    if result and result.success:
                        # Create strategy analysis result
                        strategy_result = StrategyAnalysisResult(
                            strategy_name=strategy_name,
                            config=config,
                            memory_usage=result.metadata.get('memory_usage_mb', 0),
                            processing_time=result.processing_time,
                            sequence_length=result.sequence_length,
                            batch_size=result.batch_size,
                            padding_tokens_added=result.metadata.get('padding_tokens_added', 0),
                            truncation_tokens_removed=result.metadata.get('truncation_tokens_removed', 0),
                            efficiency_score=result.metadata.get('efficiency_score', 0),
                            success=True,
                            metadata=result.metadata
                        )
                        results[strategy_name] = strategy_result
                    else:
                        error_msg = result.error if result else "Unknown error"
                        results[strategy_name] = StrategyAnalysisResult(
                            strategy_name=strategy_name,
                            config=config,
                            memory_usage=0,
                            processing_time=0,
                            sequence_length=0,
                            batch_size=0,
                            padding_tokens_added=0,
                            truncation_tokens_removed=0,
                            efficiency_score=0,
                            success=False,
                            error=error_msg
                        )
                        
                except Exception as e:
                    error(f"Failed to analyze strategy {strategy_name}", "padding_truncation_manager", e)
                    results[strategy_name] = StrategyAnalysisResult(
                        strategy_name=strategy_name,
                        config=config,
                        memory_usage=0,
                        processing_time=0,
                        sequence_length=0,
                        batch_size=0,
                        padding_tokens_added=0,
                        truncation_tokens_removed=0,
                        efficiency_score=0,
                        success=False,
                        error=str(e)
                    )
        
        self.performance_stats["strategy_comparisons"] += 1
        return results
    
    def get_predefined_strategy(self, strategy_name: str) -> Optional[PaddingTruncationConfig]:
        """Get a predefined strategy configuration."""
        return self.predefined_strategies.get(strategy_name)
    
    def list_predefined_strategies(self) -> List[str]:
        """List all available predefined strategies."""
        return list(self.predefined_strategies.keys())
    
    def create_custom_config(self, **kwargs) -> PaddingTruncationConfig:
        """Create a custom padding and truncation configuration."""
        return PaddingTruncationConfig(**kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            **self.performance_stats,
            'memory_cache_size': len(self.config_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.json"))),
            'average_tokens_per_operation': (
                self.performance_stats["total_tokens_processed"] / 
                max(self.performance_stats["total_operations"], 1)
            ),
            'padding_ratio': (
                self.performance_stats["total_padding_tokens"] / 
                max(self.performance_stats["total_tokens_processed"], 1)
            ),
            'truncation_ratio': (
                self.performance_stats["total_truncation_tokens"] / 
                max(self.performance_stats["total_tokens_processed"], 1)
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear padding and truncation cache."""
        if memory_only:
            self.config_cache.clear()
            info("Memory cache cleared", "padding_truncation_manager")
        else:
            # Clear both memory and disk cache
            self.config_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.is_file() and cache_file.name != "cache_metadata.json":
                    cache_file.unlink()
            
            info("All caches cleared", "padding_truncation_manager")
    
    def validate_strategy(self, config: PaddingTruncationConfig, 
                         sample_texts: List[str]) -> bool:
        """Validate a padding and truncation strategy with sample texts."""
        try:
            # Create a simple tokenizer for testing
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            
            # Test with sample texts
            result = self.apply_padding_truncation(tokenizer, sample_texts, config)
            
            return result is not None and result.success
            
        except Exception as e:
            error(f"Strategy validation failed", "padding_truncation_manager", e)
            return False


# Global instance for easy access
padding_truncation_manager = EnhancedPaddingTruncationManager()
