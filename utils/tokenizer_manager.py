"""
Enhanced Tokenizer Manager for DurgasAI.

This module provides advanced tokenizer management capabilities including:
- Intelligent caching (memory + disk)
- Batch processing optimization
- Multimodal tokenizer support
- Performance monitoring
- Advanced error handling
- Special token management
- Tokenizer validation and compatibility checks

Based on Hugging Face Transformers documentation best practices.
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
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer,
    AutoConfig
)

# Local imports
from .logger import debug, info, warning, error
from .config import Config


@dataclass
class TokenizerInfo:
    """Comprehensive information about a loaded tokenizer."""
    model_id: str
    tokenizer: Any
    vocab_size: int
    model_max_length: int
    is_fast: bool
    special_tokens: Dict[str, str]
    load_time: float
    last_used: float
    cache_key: str
    tokenizer_type: str
    supports_batch: bool = True
    multimodal_support: bool = False
    custom_tokens: Dict[str, str] = None


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    input_ids: Any
    attention_mask: Any
    tokenizer_info: TokenizerInfo
    processing_time: float
    batch_size: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    input_text: Optional[str] = None  # Add input_text attribute


class EnhancedTokenizerManager:
    """
    Enhanced tokenizer manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Batch processing optimization
    - Multimodal tokenizer support
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Special token management
    - Tokenizer validation and compatibility
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/tokenizers", max_cache_size: int = 10):
        """
        Initialize the enhanced tokenizer manager.
        
        Args:
            cache_dir: Directory for persistent tokenizer cache
            max_cache_size: Maximum number of tokenizers to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active tokenizers
        self.tokenizer_cache: Dict[str, TokenizerInfo] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_load_time": 0.0,
            "total_tokenizations": 0,
            "batch_tokenizations": 0,
            "error_count": 0
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedTokenizerManager initialized", "tokenizer_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size)
    
    def _get_cache_key(self, model_id: str, **kwargs) -> str:
        """Generate unique cache key for tokenizer configuration."""
        # Sort kwargs for consistent hashing
        sorted_kwargs = sorted(kwargs.items())
        config_str = f"{model_id}_{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_tokenizer_cache(self, cache_key: str, tokenizer_info: TokenizerInfo):
        """Save tokenizer to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        tokenizer_dir = self.cache_dir / cache_key
        
        try:
            # Save tokenizer files
            tokenizer_info.tokenizer.save_pretrained(str(tokenizer_dir))
            
            # Create metadata without the tokenizer object
            metadata = asdict(tokenizer_info)
            metadata.pop('tokenizer', None)  # Remove the actual tokenizer object
            metadata['tokenizer_path'] = str(tokenizer_dir)
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            debug(f"Tokenizer cached successfully", "tokenizer_manager",
                  cache_key=cache_key, model_id=tokenizer_info.model_id)
                
        except Exception as e:
            error(f"Failed to cache tokenizer", "tokenizer_manager", e,
                  cache_key=cache_key, model_id=tokenizer_info.model_id)
    
    def _load_tokenizer_cache(self, cache_key: str) -> Optional[TokenizerInfo]:
        """Load tokenizer from persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        tokenizer_dir = self.cache_dir / cache_key
        
        if not cache_file.exists() or not tokenizer_dir.exists():
            return None
            
        try:
            # Load metadata
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            
            # Load tokenizer from disk
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            
            # Reconstruct TokenizerInfo
            tokenizer_info = TokenizerInfo(
                model_id=metadata['model_id'],
                tokenizer=tokenizer,
                vocab_size=metadata['vocab_size'],
                model_max_length=metadata['model_max_length'],
                is_fast=metadata['is_fast'],
                special_tokens=metadata['special_tokens'],
                load_time=metadata['load_time'],
                last_used=metadata['last_used'],
                cache_key=cache_key,
                tokenizer_type=metadata['tokenizer_type'],
                supports_batch=metadata.get('supports_batch', True),
                multimodal_support=metadata.get('multimodal_support', False),
                custom_tokens=metadata.get('custom_tokens', {})
            )
            
            debug(f"Tokenizer loaded from cache", "tokenizer_manager",
                  cache_key=cache_key, model_id=tokenizer_info.model_id)
            
            return tokenizer_info
            
        except Exception as e:
            error(f"Failed to load cached tokenizer", "tokenizer_manager", e,
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
            warning(f"Failed to load cache metadata", "tokenizer_manager", e)
    
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
            warning(f"Failed to save cache metadata", "tokenizer_manager", e)
    
    def _validate_tokenizer_compatibility(self, model_id: str, **kwargs) -> bool:
        """Validate tokenizer compatibility before loading."""
        try:
            # Check if model exists and is accessible
            config = AutoConfig.from_pretrained(model_id, **kwargs)
            
            # Basic validation checks
            if not hasattr(config, 'vocab_size'):
                warning(f"Model {model_id} may not have proper tokenizer support", 
                       "tokenizer_manager")
                return False
                
            return True
            
        except Exception as e:
            error(f"Tokenizer compatibility check failed", "tokenizer_manager", e,
                  model_id=model_id)
            return False
    
    def _configure_special_tokens(self, tokenizer: Any, model_id: str) -> Dict[str, str]:
        """Configure and extract special token information."""
        special_tokens = {}
        
        try:
            # Configure pad_token if missing
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    debug("Set pad_token to eos_token", "tokenizer_manager", model_id=model_id)
                else:
                    warning(f"No suitable pad_token found for {model_id}", "tokenizer_manager")
            
            # Extract special token information
            special_tokens = {
                'bos_token': tokenizer.bos_token,
                'eos_token': tokenizer.eos_token,
                'pad_token': tokenizer.pad_token,
                'unk_token': tokenizer.unk_token,
                'sep_token': tokenizer.sep_token,
                'cls_token': tokenizer.cls_token,
                'mask_token': tokenizer.mask_token
            }
            
            # Remove None values
            special_tokens = {k: v for k, v in special_tokens.items() if v is not None}
            
        except Exception as e:
            error(f"Failed to configure special tokens", "tokenizer_manager", e,
                  model_id=model_id)
        
        return special_tokens
    
    def _determine_tokenizer_type(self, tokenizer: Any) -> Tuple[str, bool, bool]:
        """Determine tokenizer type and capabilities."""
        tokenizer_type = tokenizer.__class__.__name__
        is_fast = hasattr(tokenizer, 'is_fast') and tokenizer.is_fast
        supports_batch = is_fast  # Fast tokenizers support better batch processing
        
        # Check for multimodal capabilities
        multimodal_support = False
        try:
            if hasattr(tokenizer, 'image_token') or hasattr(tokenizer, 'extra_special_tokens'):
                multimodal_support = True
        except:
            pass
        
        return tokenizer_type, is_fast, supports_batch, multimodal_support
    
    def load_tokenizer(self, model_id: str, **kwargs) -> Optional[TokenizerInfo]:
        """
        Load tokenizer with intelligent caching and validation.
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional tokenizer loading arguments
            
        Returns:
            TokenizerInfo object if successful, None if failed
        """
        start_time = time.time()
        cache_key = self._get_cache_key(model_id, **kwargs)
        
        # Check memory cache first
        if cache_key in self.tokenizer_cache:
            self.tokenizer_cache[cache_key].last_used = time.time()
            self.performance_stats["cache_hits"] += 1
            
            info(f"Tokenizer cache hit", "tokenizer_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return self.tokenizer_cache[cache_key]
        
        # Check disk cache
        tokenizer_info = self._load_tokenizer_cache(cache_key)
        if tokenizer_info:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, tokenizer_info)
            self.performance_stats["cache_hits"] += 1
            
            info(f"Tokenizer loaded from disk cache", "tokenizer_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return tokenizer_info
        
        # Cache miss - load from HuggingFace
        self.performance_stats["cache_misses"] += 1
        
        # Validate compatibility
        if not self._validate_tokenizer_compatibility(model_id, **kwargs):
            error(f"Tokenizer compatibility validation failed", "tokenizer_manager",
                  model_id=model_id)
            return None
        
        try:
            # Load tokenizer with enhanced configuration
            default_kwargs = {
                'use_fast': True,
                'trust_remote_code': False,
                'cache_dir': str(self.cache_dir.parent / "huggingface")
            }
            default_kwargs.update(kwargs)
            
            # Filter out problematic parameters for specific tokenizer types
            if 'gemma' in model_id.lower():
                # Remove add_special_tokens for Gemma tokenizers as it conflicts with the method
                default_kwargs.pop('add_special_tokens', None)
            
            debug(f"Loading tokenizer from HuggingFace", "tokenizer_manager",
                  model_id=model_id, kwargs=default_kwargs)
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, **default_kwargs)
            
            # Configure special tokens
            special_tokens = self._configure_special_tokens(tokenizer, model_id)
            
            # Determine tokenizer capabilities
            tokenizer_type, is_fast, supports_batch, multimodal_support = self._determine_tokenizer_type(tokenizer)
            
            # Create tokenizer info
            load_time = time.time() - start_time
            tokenizer_info = TokenizerInfo(
                model_id=model_id,
                tokenizer=tokenizer,
                vocab_size=tokenizer.vocab_size,
                model_max_length=tokenizer.model_max_length,
                is_fast=is_fast,
                special_tokens=special_tokens,
                load_time=load_time,
                last_used=time.time(),
                cache_key=cache_key,
                tokenizer_type=tokenizer_type,
                supports_batch=supports_batch,
                multimodal_support=multimodal_support
            )
            
            # Cache the tokenizer
            self._add_to_memory_cache(cache_key, tokenizer_info)
            self._save_tokenizer_cache(cache_key, tokenizer_info)
            
            # Update performance stats
            self.performance_stats["total_loads"] += 1
            self._update_average_load_time(load_time)
            self._save_cache_metadata()
            
            info(f"Tokenizer loaded successfully", "tokenizer_manager",
                 model_id=model_id,
                 load_time=load_time,
                 tokenizer_type=tokenizer_type,
                 is_fast=is_fast,
                 vocab_size=tokenizer.vocab_size)
            
            return tokenizer_info
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load tokenizer", "tokenizer_manager", e,
                  model_id=model_id, load_time=time.time() - start_time)
            return None
    
    def _add_to_memory_cache(self, cache_key: str, tokenizer_info: TokenizerInfo):
        """Add tokenizer to memory cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self.tokenizer_cache) >= self.max_cache_size:
            oldest_key = min(self.tokenizer_cache.keys(), 
                           key=lambda k: self.tokenizer_cache[k].last_used)
            del self.tokenizer_cache[oldest_key]
            
            debug(f"Evicted tokenizer from memory cache", "tokenizer_manager",
                  evicted_key=oldest_key)
        
        self.tokenizer_cache[cache_key] = tokenizer_info
    
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
    
    def tokenize(self, texts: Union[str, List[str]], model_id: str, 
                **kwargs) -> Optional[TokenizationResult]:
        """
        Tokenize text(s) with optimized batch processing.
        
        Args:
            texts: Single text string or list of texts
            model_id: Model identifier for tokenizer
            **kwargs: Additional tokenization arguments
            
        Returns:
            TokenizationResult object with tokenization data
        """
        start_time = time.time()
        
        # Load tokenizer
        tokenizer_info = self.load_tokenizer(model_id, **kwargs)
        if not tokenizer_info:
            return TokenizationResult(
                input_ids=None,
                attention_mask=None,
                tokenizer_info=None,
                processing_time=time.time() - start_time,
                batch_size=0,
                success=False,
                error="Failed to load tokenizer",
                input_text=texts if isinstance(texts, str) else (texts[0] if isinstance(texts, list) and len(texts) > 0 else '')
            )
        
        try:
            # Prepare texts
            if isinstance(texts, str):
                texts = [texts]
                is_single = True
            else:
                is_single = False
            
            # Optimize batch processing for fast tokenizers
            # Always use padding=True when return_tensors="pt" to avoid tensor conversion errors
            default_kwargs = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt"
            }
            default_kwargs.update(kwargs)
            
            # Filter out problematic parameters for specific tokenizer types
            if 'gemma' in model_id.lower():
                # Remove add_special_tokens for Gemma tokenizers as it conflicts with the method
                default_kwargs.pop('add_special_tokens', None)
            
            if tokenizer_info.supports_batch and len(texts) > 1:
                # Batch processing
                encoded = tokenizer_info.tokenizer(texts, **default_kwargs)
                self.performance_stats["batch_tokenizations"] += 1
            else:
                # Individual processing
                encoded = tokenizer_info.tokenizer(
                    texts[0] if is_single else texts,
                    **default_kwargs
                )
            
            processing_time = time.time() - start_time
            self.performance_stats["total_tokenizations"] += 1
            
            # Create result
            result = TokenizationResult(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                tokenizer_info=tokenizer_info,
                processing_time=processing_time,
                batch_size=len(texts),
                success=True,
                input_text=texts[0] if is_single else '\n'.join(texts),
                metadata={
                    'is_single': is_single,
                    'vocab_size': tokenizer_info.vocab_size,
                    'model_max_length': tokenizer_info.model_max_length,
                    'special_tokens': tokenizer_info.special_tokens
                }
            )
            
            debug(f"Tokenization completed", "tokenizer_manager",
                  model_id=model_id,
                  batch_size=len(texts),
                  processing_time=processing_time,
                  is_batch=len(texts) > 1)
            
            return result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Tokenization failed", "tokenizer_manager", e,
                  model_id=model_id, batch_size=len(texts))
            
            return TokenizationResult(
                input_ids=None,
                attention_mask=None,
                tokenizer_info=tokenizer_info,
                processing_time=time.time() - start_time,
                batch_size=len(texts),
                success=False,
                error=str(e),
                input_text=texts[0] if isinstance(texts, list) and len(texts) > 0 else (texts if isinstance(texts, str) else '')
            )
    
    def get_tokenizer_info(self, model_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get comprehensive tokenizer information without loading."""
        tokenizer_info = self.load_tokenizer(model_id, **kwargs)
        if not tokenizer_info:
            return None
        
        return {
            'model_id': tokenizer_info.model_id,
            'vocab_size': tokenizer_info.vocab_size,
            'model_max_length': tokenizer_info.model_max_length,
            'is_fast': tokenizer_info.is_fast,
            'tokenizer_type': tokenizer_info.tokenizer_type,
            'supports_batch': tokenizer_info.supports_batch,
            'multimodal_support': tokenizer_info.multimodal_support,
            'special_tokens': tokenizer_info.special_tokens,
            'load_time': tokenizer_info.load_time,
            'cache_key': tokenizer_info.cache_key
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
            'memory_cache_size': len(self.tokenizer_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.json")))
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear tokenizer cache."""
        if memory_only:
            self.tokenizer_cache.clear()
            info("Memory cache cleared", "tokenizer_manager")
        else:
            # Clear both memory and disk cache
            self.tokenizer_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix != '.json':
                    cache_file.unlink()
            
            info("All caches cleared", "tokenizer_manager")
    
    def list_cached_tokenizers(self) -> List[Dict[str, Any]]:
        """List all cached tokenizers with metadata."""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    cached.append(metadata)
            except Exception as e:
                warning(f"Failed to read cache metadata", "tokenizer_manager", e,
                       cache_file=cache_file.name)
        
        return cached
    
    def safe_batch_tokenize(self, texts: Union[str, List[str]], model_id: str, 
                           **kwargs) -> Optional[TokenizationResult]:
        """
        Safe batch tokenization that handles common errors automatically.
        
        This method automatically applies proper padding and truncation to prevent
        tensor conversion errors, making it safe for batch processing.
        
        Args:
            texts: Single text string or list of texts
            model_id: Model identifier for tokenizer
            **kwargs: Additional tokenization arguments
            
        Returns:
            TokenizationResult object with tokenization data
        """
        # Ensure safe defaults for batch processing
        safe_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": kwargs.get('max_length', 512),
            "return_tensors": "pt"
        }
        safe_kwargs.update(kwargs)
        
        return self.tokenize(texts, model_id, **safe_kwargs)


# Global instance for easy access
tokenizer_manager = EnhancedTokenizerManager()
