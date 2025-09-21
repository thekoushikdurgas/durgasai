"""
Enhanced Tokenizer Summary Manager for DurgasAI.

This module provides advanced tokenizer analysis and comparison capabilities including:
- Intelligent caching (memory + disk)
- Tokenizer algorithm analysis
- Performance monitoring
- Advanced error handling
- Vocabulary analysis and comparison
- Subword tokenization insights
- Language support analysis

Based on Hugging Face Transformers Tokenizer Summary documentation best practices.
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
import warnings
import re
from collections import Counter

# Transformers imports
try:
    from transformers import AutoTokenizer, BertTokenizer, XLNetTokenizer, GPT2Tokenizer
    from transformers import T5Tokenizer, AlbertTokenizer, RoBERTaTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
from .logger import debug, info, warning, error
from .config import Config


@dataclass
class TokenizerSummaryInfo:
    """Comprehensive information about a tokenizer's characteristics."""
    model_id: str
    tokenizer: Any
    tokenizer_type: str
    algorithm: str
    vocabulary_size: int
    base_vocabulary_size: int
    merge_rules_count: int
    language_support: List[str]
    pre_tokenizer: str
    special_tokens: Dict[str, str]
    load_time: float
    last_used: float
    cache_key: str
    supports_fast: bool
    supports_unicode: bool
    supports_multilingual: bool
    byte_level: bool
    sentence_piece: bool
    wordpiece: bool
    bpe: bool
    unigram: bool


@dataclass
class TokenizationAnalysisResult:
    """Result of tokenization analysis operation."""
    tokens: List[str]
    token_count: int
    subword_ratio: float
    unknown_tokens: List[str]
    special_token_count: int
    vocabulary_coverage: float
    tokenizer_info: TokenizerSummaryInfo
    analysis_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedTokenizerSummaryManager:
    """
    Enhanced tokenizer summary manager with advanced features for DurgasAI.
    
    Features:
    - Intelligent caching (memory + disk persistence)
    - Tokenizer algorithm analysis and comparison
    - Performance monitoring and metrics
    - Advanced error handling and recovery
    - Vocabulary analysis and insights
    - Subword tokenization analysis
    - Language support detection
    - Memory-efficient operations
    """
    
    def __init__(self, cache_dir: str = "./output/cache/tokenizer_summaries", max_cache_size: int = 10):
        """
        Initialize the enhanced tokenizer summary manager.
        
        Args:
            cache_dir: Directory for persistent tokenizer summary cache
            max_cache_size: Maximum number of tokenizer summaries to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # In-memory cache for active tokenizer summaries
        self.summary_cache: Dict[str, TokenizerSummaryInfo] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_load_time": 0.0,
            "total_analyses": 0,
            "vocabulary_analyses": 0,
            "subword_analyses": 0,
            "error_count": 0,
            "algorithm_comparisons": 0,
            "total_tokens_analyzed": 0,
            "total_unknown_tokens": 0
        }
        
        # Algorithm mappings
        self.algorithm_mappings = {
            'BertTokenizer': 'WordPiece',
            'BertTokenizerFast': 'WordPiece',
            'GPT2Tokenizer': 'BPE',
            'GPT2TokenizerFast': 'BPE',
            'XLNetTokenizer': 'SentencePiece',
            'XLNetTokenizerFast': 'SentencePiece',
            'T5Tokenizer': 'SentencePiece',
            'T5TokenizerFast': 'SentencePiece',
            'AlbertTokenizer': 'SentencePiece',
            'AlbertTokenizerFast': 'SentencePiece',
            'RoBERTaTokenizer': 'BPE',
            'RoBERTaTokenizerFast': 'BPE'
        }
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        info("EnhancedTokenizerSummaryManager initialized", "tokenizer_summary_manager",
             cache_dir=str(self.cache_dir),
             max_cache_size=max_cache_size,
             transformers_available=TRANSFORMERS_AVAILABLE)
    
    def _get_cache_key(self, model_id: str, **kwargs) -> str:
        """Generate unique cache key for tokenizer summary configuration."""
        sorted_kwargs = sorted(kwargs.items())
        config_str = f"{model_id}_{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_summary_cache(self, cache_key: str, summary_info: TokenizerSummaryInfo):
        """Save tokenizer summary to persistent disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        tokenizer_dir = self.cache_dir / cache_key
        
        try:
            # Save tokenizer files
            summary_info.tokenizer.save_pretrained(str(tokenizer_dir))
            
            # Create metadata without the tokenizer object
            metadata = asdict(summary_info)
            metadata.pop('tokenizer', None)  # Remove the actual tokenizer object
            metadata['tokenizer_path'] = str(tokenizer_dir)
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            debug(f"Tokenizer summary cached successfully", "tokenizer_summary_manager",
                  cache_key=cache_key, model_id=summary_info.model_id)
                
        except Exception as e:
            error(f"Failed to cache tokenizer summary", "tokenizer_summary_manager", e,
                  cache_key=cache_key, model_id=summary_info.model_id)
    
    def _load_summary_cache(self, cache_key: str) -> Optional[TokenizerSummaryInfo]:
        """Load tokenizer summary from persistent disk cache."""
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
            
            # Reconstruct TokenizerSummaryInfo
            summary_info = TokenizerSummaryInfo(
                model_id=metadata['model_id'],
                tokenizer=tokenizer,
                tokenizer_type=metadata['tokenizer_type'],
                algorithm=metadata['algorithm'],
                vocabulary_size=metadata['vocabulary_size'],
                base_vocabulary_size=metadata['base_vocabulary_size'],
                merge_rules_count=metadata['merge_rules_count'],
                language_support=metadata['language_support'],
                pre_tokenizer=metadata['pre_tokenizer'],
                special_tokens=metadata['special_tokens'],
                load_time=metadata['load_time'],
                last_used=metadata['last_used'],
                cache_key=cache_key,
                supports_fast=metadata.get('supports_fast', False),
                supports_unicode=metadata.get('supports_unicode', True),
                supports_multilingual=metadata.get('supports_multilingual', False),
                byte_level=metadata.get('byte_level', False),
                sentence_piece=metadata.get('sentence_piece', False),
                wordpiece=metadata.get('wordpiece', False),
                bpe=metadata.get('bpe', False),
                unigram=metadata.get('unigram', False)
            )
            
            debug(f"Tokenizer summary loaded from cache", "tokenizer_summary_manager",
                  cache_key=cache_key, model_id=summary_info.model_id)
            
            return summary_info
            
        except Exception as e:
            error(f"Failed to load cached tokenizer summary", "tokenizer_summary_manager", e,
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
            warning(f"Failed to load cache metadata", "tokenizer_summary_manager", e)
    
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
            warning(f"Failed to save cache metadata", "tokenizer_summary_manager", e)
    
    def _analyze_tokenizer_algorithm(self, tokenizer: Any) -> Dict[str, Any]:
        """Analyze tokenizer algorithm and characteristics."""
        tokenizer_type = tokenizer.__class__.__name__
        algorithm = self.algorithm_mappings.get(tokenizer_type, "Unknown")
        
        # Determine algorithm characteristics
        bpe = 'BPE' in algorithm or 'bpe' in str(type(tokenizer)).lower()
        wordpiece = 'WordPiece' in algorithm or 'wordpiece' in str(type(tokenizer)).lower()
        sentence_piece = 'SentencePiece' in algorithm or 'sentencepiece' in str(type(tokenizer)).lower()
        unigram = 'Unigram' in algorithm or 'unigram' in str(type(tokenizer)).lower()
        byte_level = hasattr(tokenizer, 'byte_level') and tokenizer.byte_level
        
        # Analyze vocabulary
        vocabulary_size = getattr(tokenizer, 'vocab_size', 0)
        base_vocabulary_size = 0
        merge_rules_count = 0
        
        if hasattr(tokenizer, 'merges'):
            merge_rules_count = len(tokenizer.merges)
        elif hasattr(tokenizer, 'merge_rules'):
            merge_rules_count = len(tokenizer.merge_rules)
        
        # Estimate base vocabulary size
        if byte_level:
            base_vocabulary_size = 256  # Byte-level BPE
        elif sentence_piece:
            base_vocabulary_size = len(tokenizer.get_vocab()) - merge_rules_count
        else:
            # Estimate based on common patterns
            base_vocabulary_size = max(0, vocabulary_size - merge_rules_count)
        
        # Analyze language support
        language_support = self._detect_language_support(tokenizer)
        
        # Analyze pre-tokenizer
        pre_tokenizer = self._detect_pre_tokenizer(tokenizer)
        
        # Analyze special tokens
        special_tokens = self._analyze_special_tokens(tokenizer)
        
        return {
            'tokenizer_type': tokenizer_type,
            'algorithm': algorithm,
            'vocabulary_size': vocabulary_size,
            'base_vocabulary_size': base_vocabulary_size,
            'merge_rules_count': merge_rules_count,
            'language_support': language_support,
            'pre_tokenizer': pre_tokenizer,
            'special_tokens': special_tokens,
            'supports_fast': 'Fast' in tokenizer_type,
            'supports_unicode': True,  # Most modern tokenizers support Unicode
            'supports_multilingual': len(language_support) > 1,
            'byte_level': byte_level,
            'sentence_piece': sentence_piece,
            'wordpiece': wordpiece,
            'bpe': bpe,
            'unigram': unigram
        }
    
    def _detect_language_support(self, tokenizer: Any) -> List[str]:
        """Detect language support based on tokenizer characteristics."""
        languages = []
        
        # Check for common language indicators
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            
            # Check for Chinese characters
            if any('\u4e00' <= char <= '\u9fff' for token in vocab.keys() for char in token):
                languages.append('Chinese')
            
            # Check for Arabic characters
            if any('\u0600' <= char <= '\u06ff' for token in vocab.keys() for char in token):
                languages.append('Arabic')
            
            # Check for Cyrillic characters
            if any('\u0400' <= char <= '\u04ff' for token in vocab.keys() for char in token):
                languages.append('Russian')
            
            # Check for Japanese characters
            if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' 
                   for token in vocab.keys() for char in token):
                languages.append('Japanese')
        
        # Default to English if no specific languages detected
        if not languages:
            languages.append('English')
        
        return languages
    
    def _detect_pre_tokenizer(self, tokenizer: Any) -> str:
        """Detect pre-tokenizer type."""
        if hasattr(tokenizer, 'pre_tokenizer'):
            pre_tokenizer = str(tokenizer.pre_tokenizer)
            if 'Whitespace' in pre_tokenizer:
                return 'Whitespace'
            elif 'Punctuation' in pre_tokenizer:
                return 'Punctuation'
            elif 'BertPreTokenizer' in pre_tokenizer:
                return 'BERT'
            elif 'GPT2PreTokenizer' in pre_tokenizer:
                return 'GPT-2'
            else:
                return 'Custom'
        
        return 'Unknown'
    
    def _analyze_special_tokens(self, tokenizer: Any) -> Dict[str, str]:
        """Analyze special tokens in the tokenizer."""
        special_tokens = {}
        
        # Common special tokens
        special_token_attrs = [
            'pad_token', 'unk_token', 'bos_token', 'eos_token',
            'sep_token', 'cls_token', 'mask_token'
        ]
        
        for attr in special_token_attrs:
            if hasattr(tokenizer, attr):
                token = getattr(tokenizer, attr)
                if token:
                    special_tokens[attr] = token
        
        return special_tokens
    
    def _validate_tokenizer_compatibility(self, model_id: str, **kwargs) -> bool:
        """Validate tokenizer compatibility before loading."""
        try:
            # Basic validation - check if model exists
            # This is a simplified check - in practice, you might want more validation
            return True
            
        except Exception as e:
            error(f"Tokenizer compatibility check failed", "tokenizer_summary_manager", e,
                  model_id=model_id)
            return False
    
    def load_tokenizer_summary(self, model_id: str, **kwargs) -> Optional[TokenizerSummaryInfo]:
        """
        Load tokenizer with comprehensive analysis and intelligent caching.
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional tokenizer loading arguments
            
        Returns:
            TokenizerSummaryInfo object if successful, None if failed
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Transformers not available for tokenizer summary loading", "tokenizer_summary_manager")
            return None
        
        start_time = time.time()
        cache_key = self._get_cache_key(model_id, **kwargs)
        
        # Check memory cache first
        if cache_key in self.summary_cache:
            self.summary_cache[cache_key].last_used = time.time()
            self.performance_stats["cache_hits"] += 1
            
            info(f"Tokenizer summary cache hit", "tokenizer_summary_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return self.summary_cache[cache_key]
        
        # Check disk cache
        summary_info = self._load_summary_cache(cache_key)
        if summary_info:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, summary_info)
            self.performance_stats["cache_hits"] += 1
            
            info(f"Tokenizer summary loaded from disk cache", "tokenizer_summary_manager",
                 model_id=model_id, cache_key=cache_key)
            
            return summary_info
        
        # Cache miss - load from HuggingFace
        self.performance_stats["cache_misses"] += 1
        
        # Validate compatibility
        if not self._validate_tokenizer_compatibility(model_id, **kwargs):
            error(f"Tokenizer compatibility validation failed", "tokenizer_summary_manager",
                  model_id=model_id)
            return None
        
        try:
            # Load tokenizer with enhanced configuration
            default_kwargs = {
                'cache_dir': str(self.cache_dir.parent / "huggingface")
            }
            default_kwargs.update(kwargs)
            
            debug(f"Loading tokenizer from HuggingFace", "tokenizer_summary_manager",
                  model_id=model_id, kwargs=default_kwargs)
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, **default_kwargs)
            
            # Analyze tokenizer characteristics
            analysis = self._analyze_tokenizer_algorithm(tokenizer)
            
            # Create summary info
            load_time = time.time() - start_time
            summary_info = TokenizerSummaryInfo(
                model_id=model_id,
                tokenizer=tokenizer,
                load_time=load_time,
                last_used=time.time(),
                cache_key=cache_key,
                **analysis
            )
            
            # Cache the tokenizer summary
            self._add_to_memory_cache(cache_key, summary_info)
            self._save_summary_cache(cache_key, summary_info)
            
            # Update performance stats
            self.performance_stats["total_loads"] += 1
            self._update_average_load_time(load_time)
            self._save_cache_metadata()
            
            info(f"Tokenizer summary loaded successfully", "tokenizer_summary_manager",
                 model_id=model_id,
                 load_time=load_time,
                 tokenizer_type=summary_info.tokenizer_type,
                 algorithm=summary_info.algorithm,
                 vocabulary_size=summary_info.vocabulary_size)
            
            return summary_info
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Failed to load tokenizer summary", "tokenizer_summary_manager", e,
                  model_id=model_id, load_time=time.time() - start_time)
            return None
    
    def _add_to_memory_cache(self, cache_key: str, summary_info: TokenizerSummaryInfo):
        """Add tokenizer summary to memory cache with LRU eviction."""
        # Remove oldest entries if cache is full
        if len(self.summary_cache) >= self.max_cache_size:
            oldest_key = min(self.summary_cache.keys(), 
                           key=lambda k: self.summary_cache[k].last_used)
            del self.summary_cache[oldest_key]
            
            debug(f"Evicted tokenizer summary from memory cache", "tokenizer_summary_manager",
                  evicted_key=oldest_key)
        
        self.summary_cache[cache_key] = summary_info
    
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
    
    def analyze_tokenization(self, text: str, model_id: str, **kwargs) -> Optional[TokenizationAnalysisResult]:
        """
        Analyze tokenization characteristics of text.
        
        Args:
            text: Text to analyze
            model_id: Model identifier for tokenizer
            **kwargs: Additional analysis arguments
            
        Returns:
            TokenizationAnalysisResult object with analysis details
        """
        if not TRANSFORMERS_AVAILABLE:
            error("Required dependencies not available for tokenization analysis", "tokenizer_summary_manager")
            return None
        
        start_time = time.time()
        
        # Load tokenizer summary
        summary_info = self.load_tokenizer_summary(model_id, **kwargs)
        if not summary_info:
            return TokenizationAnalysisResult(
                tokens=[],
                token_count=0,
                subword_ratio=0.0,
                unknown_tokens=[],
                special_token_count=0,
                vocabulary_coverage=0.0,
                tokenizer_info=None,
                analysis_time=time.time() - start_time,
                success=False,
                error="Failed to load tokenizer summary"
            )
        
        try:
            # Tokenize text
            tokens = summary_info.tokenizer.tokenize(text)
            token_count = len(tokens)
            
            # Analyze subword ratio
            words = text.split()
            subword_ratio = token_count / len(words) if words else 0.0
            
            # Find unknown tokens
            unknown_tokens = []
            if hasattr(summary_info.tokenizer, 'unk_token_id'):
                token_ids = summary_info.tokenizer.convert_tokens_to_ids(tokens)
                unk_id = summary_info.tokenizer.unk_token_id
                if unk_id is not None:
                    unknown_tokens = [tokens[i] for i, token_id in enumerate(token_ids) if token_id == unk_id]
            
            # Count special tokens
            special_token_count = 0
            special_tokens = set(summary_info.special_tokens.values())
            for token in tokens:
                if token in special_tokens:
                    special_token_count += 1
            
            # Calculate vocabulary coverage
            if hasattr(summary_info.tokenizer, 'get_vocab'):
                vocab = summary_info.tokenizer.get_vocab()
                known_tokens = sum(1 for token in tokens if token in vocab)
                vocabulary_coverage = known_tokens / token_count if token_count > 0 else 0.0
            else:
                vocabulary_coverage = 1.0 - (len(unknown_tokens) / token_count) if token_count > 0 else 0.0
            
            analysis_time = time.time() - start_time
            self.performance_stats["total_analyses"] += 1
            self.performance_stats["total_tokens_analyzed"] += token_count
            self.performance_stats["total_unknown_tokens"] += len(unknown_tokens)
            
            # Update specific analysis counts
            if subword_ratio > 1.0:
                self.performance_stats["subword_analyses"] += 1
            
            self.performance_stats["vocabulary_analyses"] += 1
            
            # Create result
            analysis_result = TokenizationAnalysisResult(
                tokens=tokens,
                token_count=token_count,
                subword_ratio=subword_ratio,
                unknown_tokens=unknown_tokens,
                special_token_count=special_token_count,
                vocabulary_coverage=vocabulary_coverage,
                tokenizer_info=summary_info,
                analysis_time=analysis_time,
                success=True,
                metadata={
                    'text_length': len(text),
                    'word_count': len(words),
                    'algorithm': summary_info.algorithm,
                    'tokenizer_type': summary_info.tokenizer_type,
                    'vocabulary_size': summary_info.vocabulary_size
                }
            )
            
            debug(f"Tokenization analysis completed", "tokenizer_summary_manager",
                  model_id=model_id,
                  token_count=token_count,
                  subword_ratio=subword_ratio,
                  analysis_time=analysis_time)
            
            return analysis_result
            
        except Exception as e:
            self.performance_stats["error_count"] += 1
            error(f"Tokenization analysis failed", "tokenizer_summary_manager", e,
                  model_id=model_id, text_length=len(text))
            
            return TokenizationAnalysisResult(
                tokens=[],
                token_count=0,
                subword_ratio=0.0,
                unknown_tokens=[],
                special_token_count=0,
                vocabulary_coverage=0.0,
                tokenizer_info=summary_info,
                analysis_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def compare_tokenizers(self, text: str, model_ids: List[str], **kwargs) -> Dict[str, TokenizationAnalysisResult]:
        """
        Compare tokenization across multiple tokenizers.
        
        Args:
            text: Text to analyze
            model_ids: List of model identifiers to compare
            **kwargs: Additional analysis arguments
            
        Returns:
            Dictionary mapping model_id to TokenizationAnalysisResult
        """
        results = {}
        
        for model_id in model_ids:
            try:
                result = self.analyze_tokenization(text, model_id, **kwargs)
                if result:
                    results[model_id] = result
            except Exception as e:
                error(f"Failed to analyze tokenizer {model_id}", "tokenizer_summary_manager", e)
                continue
        
        self.performance_stats["algorithm_comparisons"] += 1
        return results
    
    def get_summary_info(self, model_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get comprehensive tokenizer summary information without loading."""
        summary_info = self.load_tokenizer_summary(model_id, **kwargs)
        if not summary_info:
            return None
        
        return {
            'model_id': summary_info.model_id,
            'tokenizer_type': summary_info.tokenizer_type,
            'algorithm': summary_info.algorithm,
            'vocabulary_size': summary_info.vocabulary_size,
            'base_vocabulary_size': summary_info.base_vocabulary_size,
            'merge_rules_count': summary_info.merge_rules_count,
            'language_support': summary_info.language_support,
            'pre_tokenizer': summary_info.pre_tokenizer,
            'special_tokens': summary_info.special_tokens,
            'supports_fast': summary_info.supports_fast,
            'supports_unicode': summary_info.supports_unicode,
            'supports_multilingual': summary_info.supports_multilingual,
            'byte_level': summary_info.byte_level,
            'sentence_piece': summary_info.sentence_piece,
            'wordpiece': summary_info.wordpiece,
            'bpe': summary_info.bpe,
            'unigram': summary_info.unigram,
            'load_time': summary_info.load_time,
            'cache_key': summary_info.cache_key
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
            'memory_cache_size': len(self.summary_cache),
            'disk_cache_size': len(list(self.cache_dir.glob("*.json"))),
            'average_tokens_per_analysis': (
                self.performance_stats["total_tokens_analyzed"] / 
                max(self.performance_stats["total_analyses"], 1)
            ),
            'unknown_token_rate': (
                self.performance_stats["total_unknown_tokens"] / 
                max(self.performance_stats["total_tokens_analyzed"], 1)
            )
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear tokenizer summary cache."""
        if memory_only:
            self.summary_cache.clear()
            info("Memory cache cleared", "tokenizer_summary_manager")
        else:
            # Clear both memory and disk cache
            self.summary_cache.clear()
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix != '.json':
                    cache_file.unlink()
            
            info("All caches cleared", "tokenizer_summary_manager")
    
    def list_cached_summaries(self) -> List[Dict[str, Any]]:
        """List all cached tokenizer summaries with metadata."""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    cached.append(metadata)
            except Exception as e:
                warning(f"Failed to read cache metadata", "tokenizer_summary_manager", e,
                       cache_file=cache_file.name)
        
        return cached


# Global instance for easy access
tokenizer_summary_manager = EnhancedTokenizerSummaryManager()
