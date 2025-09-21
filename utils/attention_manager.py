"""
Attention function management utilities for DurgasAI custom models.

This module provides tools for managing, registering, and testing custom attention
functions using HuggingFace's AttentionInterface and AttentionMaskInterface.
"""

import torch
import torch.nn.functional as F
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from functools import wraps

from .logger import debug, info, warning, error


@dataclass
class AttentionFunctionInfo:
    """Information about a registered attention function."""
    name: str
    function: Callable
    description: str
    performance_metrics: Dict[str, float]
    is_custom: bool = True
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AttentionMaskInfo:
    """Information about a registered attention mask function."""
    name: str
    function: Callable
    description: str
    compatible_attention_functions: List[str]
    is_custom: bool = True


class AttentionManager:
    """Manager for custom attention functions and masks."""
    
    def __init__(self):
        self.attention_functions: Dict[str, AttentionFunctionInfo] = {}
        self.attention_masks: Dict[str, AttentionMaskInfo] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Initialize with built-in functions
        self._initialize_builtin_functions()
        
        debug("AttentionManager initialized", "attention")
    
    def _initialize_builtin_functions(self):
        """Initialize information about built-in attention functions."""
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            
            builtin_functions = {
                "eager": "Simple matrix multiplication without optimization",
                "sdpa": "Scaled Dot-Product Attention (PyTorch optimized)",
                "flash_attention_2": "Memory-efficient attention implementation",
                "flex_attention": "Flexible attention with various optimizations"
            }
            
            for name, description in builtin_functions.items():
                if name in ALL_ATTENTION_FUNCTIONS:
                    self.attention_functions[name] = AttentionFunctionInfo(
                        name=name,
                        function=ALL_ATTENTION_FUNCTIONS[name],
                        description=description,
                        performance_metrics={},
                        is_custom=False
                    )
            
            info(f"Initialized {len(self.attention_functions)} built-in attention functions", "attention")
            
        except ImportError:
            warning("Could not import ALL_ATTENTION_FUNCTIONS", "attention")
    
    def register_attention_function(
        self, 
        name: str, 
        function: Callable, 
        description: str = "",
        dependencies: List[str] = None
    ) -> bool:
        """
        Register a custom attention function.
        
        Args:
            name: Name of the attention function
            function: The attention function implementation
            description: Description of the function
            dependencies: List of dependencies
            
        Returns:
            bool: True if registration successful
        """
        try:
            debug(f"Registering attention function: {name}", "attention")
            
            # Validate function signature
            if not self._validate_attention_function_signature(function):
                error(f"Invalid signature for attention function: {name}", "attention")
                return False
            
            # Register with HuggingFace AttentionInterface
            from transformers import AttentionInterface
            AttentionInterface.register(name, function)
            
            # Store information
            self.attention_functions[name] = AttentionFunctionInfo(
                name=name,
                function=function,
                description=description or f"Custom attention function: {name}",
                performance_metrics={},
                is_custom=True,
                dependencies=dependencies or []
            )
            
            info(f"Successfully registered attention function: {name}", "attention")
            return True
            
        except Exception as e:
            error(f"Error registering attention function {name}: {str(e)}", "attention", e)
            return False
    
    def register_attention_mask(
        self, 
        name: str, 
        function: Callable, 
        description: str = "",
        compatible_attention_functions: List[str] = None
    ) -> bool:
        """
        Register a custom attention mask function.
        
        Args:
            name: Name of the mask function
            function: The mask function implementation
            description: Description of the function
            compatible_attention_functions: List of compatible attention functions
            
        Returns:
            bool: True if registration successful
        """
        try:
            debug(f"Registering attention mask: {name}", "attention")
            
            # Validate function signature
            if not self._validate_attention_mask_signature(function):
                error(f"Invalid signature for attention mask: {name}", "attention")
                return False
            
            # Register with HuggingFace AttentionMaskInterface
            from transformers import AttentionMaskInterface
            AttentionMaskInterface.register(name, function)
            
            # Store information
            self.attention_masks[name] = AttentionMaskInfo(
                name=name,
                function=function,
                description=description or f"Custom attention mask: {name}",
                compatible_attention_functions=compatible_attention_functions or [],
                is_custom=True
            )
            
            info(f"Successfully registered attention mask: {name}", "attention")
            return True
            
        except Exception as e:
            error(f"Error registering attention mask {name}: {str(e)}", "attention", e)
            return False
    
    def _validate_attention_function_signature(self, function: Callable) -> bool:
        """Validate that function has correct attention function signature."""
        try:
            import inspect
            sig = inspect.signature(function)
            params = list(sig.parameters.keys())
            
            # Check for required parameters
            required_params = ['module', 'query', 'key', 'value']
            if not all(param in params for param in required_params):
                return False
            
            # Check for **kwargs
            if 'kwargs' not in params:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_attention_mask_signature(self, function: Callable) -> bool:
        """Validate that function has correct attention mask signature."""
        try:
            import inspect
            sig = inspect.signature(function)
            params = list(sig.parameters.keys())
            
            # Check for required parameters
            required_params = ['batch_size', 'cache_position', 'kv_length']
            if not all(param in params for param in required_params):
                return False
            
            # Check for **kwargs
            if 'kwargs' not in params:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_attention_functions(self) -> Dict[str, AttentionFunctionInfo]:
        """Get all registered attention functions."""
        return self.attention_functions.copy()
    
    def get_attention_masks(self) -> Dict[str, AttentionMaskInfo]:
        """Get all registered attention masks."""
        return self.attention_masks.copy()
    
    def get_attention_function(self, name: str) -> Optional[AttentionFunctionInfo]:
        """Get information about a specific attention function."""
        return self.attention_functions.get(name)
    
    def benchmark_attention_function(
        self, 
        name: str, 
        model, 
        input_ids: torch.Tensor,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark an attention function's performance.
        
        Args:
            name: Name of the attention function
            model: The model to test with
            input_ids: Input tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Dict containing performance metrics
        """
        try:
            debug(f"Starting benchmark for attention function: {name}", "attention",
                  input_shape=input_ids.shape,
                  num_runs=num_runs,
                  device=input_ids.device)
            
            # Validate attention function exists
            if name not in self.attention_functions:
                warning(f"Attention function {name} not found", "attention",
                       available_functions=list(self.attention_functions.keys()))
                return {}
            
            debug("Attention function found, preparing benchmark", "attention")
            
            # Store original attention implementation for restoration
            original_implementation = getattr(model.config, '_attn_implementation', 'eager')
            debug(f"Original attention implementation: {original_implementation}", "attention")
            
            # Switch to the target attention function
            debug(f"Switching to attention function: {name}", "attention")
            model.set_attn_implementation(name)
            
            # Perform warmup run to ensure model is ready and GPU memory is allocated
            debug("Performing warmup run", "attention")
            with torch.no_grad():
                _ = model(input_ids)
            debug("Warmup completed", "attention")
            
            # Benchmark the attention function across multiple runs
            debug(f"Starting {num_runs} benchmark runs", "attention")
            times = []
            
            for run_idx in range(num_runs):
                debug(f"Benchmark run {run_idx + 1}/{num_runs}", "attention")
                
                # Measure execution time for this run
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_ids)
                run_time = time.time() - start_time
                times.append(run_time)
                
                debug(f"Run {run_idx + 1} completed in {run_time:.4f}s", "attention")
            
            # Calculate comprehensive performance metrics
            debug("Calculating performance metrics", "attention")
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Calculate throughput (tokens per second)
            total_tokens = input_ids.numel()
            throughput = total_tokens / avg_time
            
            metrics = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'throughput': throughput,
                'total_tokens': total_tokens,
                'runs_completed': num_runs
            }
            
            debug("Performance metrics calculated", "attention", metrics=metrics)
            
            # Store metrics in the attention function info
            self.attention_functions[name].performance_metrics.update(metrics)
            debug("Metrics stored in attention function info", "attention")
            
            # Restore original attention implementation
            debug(f"Restoring original attention implementation: {original_implementation}", "attention")
            model.set_attn_implementation(original_implementation)
            
            info(f"Benchmark completed for {name}: {avg_time:.4f}s avg, {throughput:.2f} tokens/s", "attention",
                 function_name=name,
                 average_time=avg_time,
                 throughput=throughput,
                 num_runs=num_runs)
            
            return metrics
            
        except Exception as e:
            error(f"Error benchmarking attention function {name}: {str(e)}", "attention", e,
                  function_name=name,
                  input_shape=input_ids.shape if input_ids is not None else None,
                  num_runs=num_runs)
            return {}
    
    def compare_attention_functions(
        self, 
        function_names: List[str], 
        model, 
        input_ids: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of multiple attention functions.
        
        Args:
            function_names: List of attention function names to compare
            model: The model to test with
            input_ids: Input tensor
            
        Returns:
            Dict containing comparison results
        """
        try:
            debug(f"Comparing attention functions: {function_names}", "attention")
            
            results = {}
            for name in function_names:
                if name in self.attention_functions:
                    metrics = self.benchmark_attention_function(name, model, input_ids)
                    results[name] = metrics
                else:
                    warning(f"Attention function {name} not found", "attention")
            
            # Calculate relative performance
            if results:
                baseline_time = min(metrics['avg_time'] for metrics in results.values())
                for name, metrics in results.items():
                    metrics['speedup'] = baseline_time / metrics['avg_time']
            
            return results
            
        except Exception as e:
            error(f"Error comparing attention functions: {str(e)}", "attention", e)
            return {}
    
    def create_attention_function_template(self, name: str) -> str:
        """
        Create a template for a custom attention function.
        
        Args:
            name: Name for the attention function
            
        Returns:
            String template for the attention function
        """
        template = f'''"""
Custom attention function: {name}
"""

import torch
import torch.nn.functional as F
from typing import Optional


def {name}(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Custom attention function implementation.
    
    Args:
        module: The attention module
        query: Query tensor
        key: Key tensor
        value: Value tensor
        attention_mask: Optional attention mask
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    # TODO: Implement your custom attention logic here
    # 
    # Implementation Guidelines:
    # 1. Ensure compatibility with HuggingFace AttentionInterface
    # 2. Handle different input shapes and batch sizes
    # 3. Support optional attention masking
    # 4. Return both attention output and weights (if requested)
    # 5. Consider memory efficiency for large sequences
    # 6. Add proper error handling for edge cases
    # 7. Test with different model architectures
    
    # Example: Simple scaled dot-product attention implementation
    # This is a basic reference implementation - customize as needed
    
    # Calculate scaling factor for attention scores (prevents softmax saturation)
    scale_factor = 1.0 / (query.size(-1) ** 0.5)
    debug(f"Using scale factor: {scale_factor}", "attention")
    
    # Compute attention scores using matrix multiplication
    # Shape: [batch_size, num_heads, seq_len, seq_len]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    debug(f"Attention scores computed, shape: {attn_scores.shape}", "attention")
    
    # Apply attention mask if provided (prevents attention to padded tokens)
    if attention_mask is not None:
        debug("Applying attention mask", "attention", mask_shape=attention_mask.shape)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
    
    # Apply softmax to get attention weights (probabilities)
    attn_weights = F.softmax(attn_scores, dim=-1)
    debug("Softmax applied to attention scores", "attention", weights_shape=attn_weights.shape)
    
    # Apply attention weights to values to get final output
    attn_output = torch.matmul(attn_weights, value)
    debug("Attention output computed", "attention", output_shape=attn_output.shape)
    
    return attn_output, attn_weights
'''
        return template
    
    def create_attention_mask_template(self, name: str) -> str:
        """
        Create a template for a custom attention mask function.
        
        Args:
            name: Name for the attention mask function
            
        Returns:
            String template for the attention mask function
        """
        template = f'''"""
Custom attention mask function: {name}
"""

import torch
from typing import Optional, Callable


def {name}(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs
) -> Optional[torch.Tensor]:
    """
    Custom attention mask function implementation.
    
    Args:
        batch_size: Batch size
        cache_position: Cache position tensor
        kv_length: Key-value length
        kv_offset: Key-value offset
        mask_function: Mask function callable
        attention_mask: Optional attention mask
        **kwargs: Additional arguments
    
    Returns:
        Optional attention mask tensor
    """
    # TODO: Implement your custom mask logic here
    # 
    # Implementation Guidelines:
    # 1. Ensure compatibility with HuggingFace AttentionMaskInterface
    # 2. Handle different batch sizes and sequence lengths
    # 3. Support various masking patterns (causal, bidirectional, etc.)
    # 4. Consider memory efficiency for large sequences
    # 5. Return appropriate tensor types and shapes
    # 6. Add proper validation for input parameters
    # 7. Test with different model architectures and use cases
    
    # Example: Simple causal mask implementation
    # This creates a lower triangular mask for autoregressive models
    
    debug(f"Creating attention mask", "attention",
          batch_size=batch_size,
          kv_length=kv_length,
          kv_offset=kv_offset)
    
    if kv_length > 0:
        debug("Creating lower triangular causal mask", "attention")
        
        # Create lower triangular mask (allows attention to previous positions only)
        # Shape: [kv_length, kv_length]
        mask = torch.tril(torch.ones(kv_length, kv_length))
        debug(f"Base triangular mask created, shape: {mask.shape}", "attention")
        
        # Expand for batch dimension to support batched inputs
        # Shape: [batch_size, kv_length, kv_length]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        debug(f"Mask expanded for batch processing, final shape: {mask.shape}", "attention")
        
        return mask
    
    debug("No mask needed (kv_length <= 0)", "attention")
    return None
'''
        return template


# Global instance
attention_manager = AttentionManager()
