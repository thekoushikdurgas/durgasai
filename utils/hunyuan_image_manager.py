"""
HunyuanImage-2.1 Manager for DurgasAI Framework

This module provides comprehensive HunyuanImage-2.1 model management capabilities for
high-resolution text-to-image generation. It integrates with the existing DurgasAI 
architecture to support both API and local model usage.

Key Features:
- 2K (2048x2048) image generation
- Multiple aspect ratio support (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
- Multilingual prompt support (English and Chinese)
- Prompt enhancement and refiner model options
- Integration with existing model management system
- Comprehensive error handling and logging

Key Classes:
- HunyuanImageManager: Main interface for HunyuanImage-2.1 operations
- HunyuanImageResponse: Structured response format for generation tasks
"""

import os
import time
import torch
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

# HuggingFace imports
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

# Import existing configuration and logging
from .config import Config
from .logger import debug, info, warning, error, log_model_operation, time_operation, LoggedOperation


@dataclass
class HunyuanImageResponse:
    """
    Structured response format for HunyuanImage generation operations.
    
    This class standardizes the response format for HunyuanImage-2.1 operations.
    It includes success/failure status, generated image, metadata, and error information.
    
    Attributes:
        success (bool): Whether the generation was successful
        image (Optional[Image.Image]): Generated image if successful
        prompt (str): The prompt used for generation
        method (str): Generation method used ("api_hunyuan" or "local_hunyuan")
        error (Optional[str]): Error message if generation failed
        metadata (Optional[Dict[str, Any]]): Additional information like generation time, parameters, etc.
    """
    success: bool
    image: Optional[Image.Image] = None
    prompt: str = ""
    method: str = ""
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HunyuanImageManager:
    """
    Manages HunyuanImage-2.1 model for text-to-image generation.
    
    This class provides a unified interface for HunyuanImage-2.1 integration
    with the DurgasAI framework, supporting both API and local model usage.
    
    Supported Features:
    - High-resolution 2K image generation
    - Multiple aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
    - Multilingual prompt support
    - Prompt enhancement and refiner model options
    - Performance tracking and logging
    - Error handling and validation
    
    Architecture:
    - Uses HuggingFace InferenceClient for API calls
    - Supports local model loading for advanced users
    - Integrates with existing logging and configuration systems
    - Implements comprehensive error handling and validation
    """
    
    # Supported aspect ratios and their dimensions
    SUPPORTED_ASPECT_RATIOS = {
        "1:1": (2048, 2048),
        "16:9": (2560, 1536),
        "9:16": (1536, 2560),
        "4:3": (2304, 1792),
        "3:4": (1792, 2304),
        "3:2": (2048, 1365),
        "2:3": (1365, 2048)
    }
    
    def __init__(self, api_key: str = None, use_local: bool = False, device: str = "auto"):
        """
        Initialize the HunyuanImage manager.
        
        Args:
            api_key (str): HuggingFace API key for cloud inference
            use_local (bool): Whether to use local model loading
            device (str): Device to use for local inference ("auto", "cuda", "cpu")
        """
        debug("Initializing HunyuanImageManager", "hunyuan")
        
        self.api_key = api_key
        self.use_local = use_local
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.client = None
        self.pipeline = None
        
        # Performance tracking
        self.total_generations = 0
        self.successful_generations = 0
        self.failed_generations = 0
        self.average_generation_time = 0.0
        
        # Initialize based on configuration
        if use_local:
            self._initialize_local_model()
        else:
            self._initialize_api_client()
        
        info("HunyuanImageManager initialized successfully", "hunyuan",
             use_local=use_local, device=self.device)
        
        log_model_operation("hunyuan_manager_initialized")
    
    def _initialize_api_client(self):
        """Initialize HuggingFace API client."""
        try:
            if not HF_HUB_AVAILABLE:
                raise ImportError("huggingface_hub is required for API usage")
            
            if not self.api_key:
                # Try to get from config
                api_config = Config.get_api_config("huggingface")
                if api_config and "api_keys" in api_config:
                    self.api_key = api_config["api_keys"]
                else:
                    raise ValueError("No API key provided and none found in config")
            
            self.client = InferenceClient(
                provider="fal-ai",
                api_key=self.api_key,
            )
            
            info("HunyuanImage API client initialized", "hunyuan")
            
        except Exception as e:
            error(f"Failed to initialize API client: {str(e)}", "hunyuan", e)
            raise
    
    def _initialize_local_model(self):
        """Initialize local HunyuanImage pipeline."""
        try:
            # Set environment variables for CUDA optimization using centralized config
            try:
                from .config import Config
                
                # Setup environment variables from configuration
                Config.setup_environment_variables()
                
                # Ensure PyTorch specific variables are set
                Config.update_environment_variable('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True', save_to_config=False)
                
                debug("Environment variables configured from centralized config", "hunyuan")
                
            except Exception as e:
                debug(f"Could not load centralized config: {e}", "hunyuan")
                debug("Using fallback environment variable setup", "hunyuan")
                
                # Fallback to direct setting
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Try to import the HunyuanImage pipeline
            try:
                from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
            except ImportError:
                raise ImportError("hyimage package is required for local model usage. Install with: pip install hyimage")
            
            model_name = "hunyuanimage-v2.1"
            self.pipeline = HunyuanImagePipeline.from_pretrained(
                model_name=model_name, 
                use_fp8=True
            )
            self.pipeline = self.pipeline.to(self.device)
            
            info(f"HunyuanImage local model initialized on {self.device}", "hunyuan")
            
        except Exception as e:
            error(f"Failed to initialize local model: {str(e)}", "hunyuan", e)
            raise
    
    @time_operation("hunyuan_generation", "hunyuan")
    def generate_image(self, 
                      prompt: str,
                      width: int = 2048,
                      height: int = 2048,
                      aspect_ratio: str = "1:1",
                      use_refiner: bool = True,
                      use_prompt_enhancement: bool = False,
                      num_inference_steps: int = 50,
                      guidance_scale: float = 3.5,
                      seed: Optional[int] = None) -> HunyuanImageResponse:
        """
        Generate image using HunyuanImage-2.1.
        
        Args:
            prompt (str): Text prompt for image generation
            width (int): Image width (must be supported resolution)
            height (int): Image height (must be supported resolution)
            aspect_ratio (str): Aspect ratio preset ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3")
            use_refiner (bool): Whether to use the refiner model
            use_prompt_enhancement (bool): Whether to use prompt enhancement
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale for generation
            seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            HunyuanImageResponse: Structured response with generation results
        """
        # Initialize generation tracking and timing
        start_time = time.time()
        self.total_generations += 1
        
        debug(f"Starting HunyuanImage-2.1 generation #{self.total_generations}", "hunyuan",
              prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
              width=width, height=height, aspect_ratio=aspect_ratio,
              use_refiner=use_refiner, use_prompt_enhancement=use_prompt_enhancement,
              num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
              seed=seed, generation_method="local" if self.use_local else "api")
        
        try:
            with LoggedOperation("hunyuan_image_generation", "hunyuan",
                               extra_data={"prompt_length": len(prompt), "aspect_ratio": aspect_ratio,
                                         "generation_number": self.total_generations}):
                
                # Step 1: Validate dimensions and aspect ratio
                # Ensures the requested dimensions are supported by HunyuanImage-2.1
                debug("Validating image dimensions and aspect ratio", "hunyuan",
                      requested_width=width, requested_height=height,
                      requested_aspect_ratio=aspect_ratio)
                
                if not self._validate_dimensions(width, height, aspect_ratio):
                    error_msg = f"Invalid dimensions {width}x{height} for aspect ratio {aspect_ratio}"
                    error(error_msg, "hunyuan",
                          width=width, height=height, aspect_ratio=aspect_ratio,
                          supported_ratios=list(self.SUPPORTED_ASPECT_RATIOS.keys()))
                    raise ValueError(error_msg)
                
                debug("Dimension validation passed", "hunyuan")
                
                # Step 2: Choose generation method based on configuration
                # Local generation uses downloaded models, API uses HuggingFace Inference
                if self.use_local:
                    debug("Using local generation method", "hunyuan",
                          local_model_available=self.local_model is not None)
                    
                    result = self._generate_local(
                        prompt, width, height, use_refiner, use_prompt_enhancement,
                        num_inference_steps, guidance_scale, seed
                    )
                else:
                    debug("Using API generation method", "hunyuan",
                          api_client_available=self.api_client is not None)
                    result = self._generate_api(
                        prompt, width, height, use_refiner, use_prompt_enhancement,
                        num_inference_steps, guidance_scale, seed
                    )
                
                # Update metrics
                generation_time = time.time() - start_time
                self.successful_generations += 1
                self.average_generation_time = (
                    (self.average_generation_time * (self.total_generations - 1) + generation_time) 
                    / self.total_generations
                )
                
                # Add generation time to metadata
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["generation_time"] = generation_time
                
                info(f"HunyuanImage generation completed in {generation_time:.2f}s", "hunyuan",
                     generation_time=generation_time, success=True)
                
                log_model_operation("hunyuan_generation_completed",
                                  generation_time=generation_time,
                                  success=True)
                
                return result
                
        except Exception as e:
            generation_time = time.time() - start_time
            self.failed_generations += 1
            
            error_msg = f"HunyuanImage generation failed: {str(e)}"
            error(error_msg, "hunyuan", e, generation_time=generation_time)
            
            log_model_operation("hunyuan_generation_failed",
                              error=str(e),
                              generation_time=generation_time)
            
            return HunyuanImageResponse(
                success=False,
                prompt=prompt,
                method="hunyuan",
                error=error_msg,
                metadata={"generation_time": generation_time}
            )
    
    def _validate_dimensions(self, width: int, height: int, aspect_ratio: str) -> bool:
        """Validate image dimensions against supported aspect ratios."""
        if aspect_ratio in self.SUPPORTED_ASPECT_RATIOS:
            expected_width, expected_height = self.SUPPORTED_ASPECT_RATIOS[aspect_ratio]
            return width == expected_width and height == expected_height
        
        return False
    
    def _generate_local(self, prompt: str, width: int, height: int, 
                       use_refiner: bool, use_prompt_enhancement: bool,
                       num_inference_steps: int, guidance_scale: float, 
                       seed: Optional[int]) -> HunyuanImageResponse:
        """Generate image using local model."""
        if self.pipeline is None:
            raise RuntimeError("Local pipeline not initialized")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                use_reprompt=use_prompt_enhancement,
                use_refiner=use_refiner,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
        
        return HunyuanImageResponse(
            success=True,
            image=result.images[0],
            prompt=prompt,
            method="local_hunyuan",
            metadata={
                "width": width,
                "height": height,
                "use_refiner": use_refiner,
                "use_prompt_enhancement": use_prompt_enhancement,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "device": self.device
            }
        )
    
    def _generate_api(self, prompt: str, width: int, height: int,
                     use_refiner: bool, use_prompt_enhancement: bool,
                     num_inference_steps: int, guidance_scale: float,
                     seed: Optional[int]) -> HunyuanImageResponse:
        """Generate image using API."""
        if self.client is None:
            raise RuntimeError("API client not initialized")
        
        # For API, we use the text_to_image method
        # Note: API parameters may be limited compared to local model
        image = self.client.text_to_image(
            prompt,
            model="tencent/HunyuanImage-2.1",
        )
        
        return HunyuanImageResponse(
            success=True,
            image=image,
            prompt=prompt,
            method="api_hunyuan",
            metadata={
                "width": width,
                "height": height,
                "use_refiner": use_refiner,
                "use_prompt_enhancement": use_prompt_enhancement,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "provider": "fal-ai"
            }
        )
    
    def generate_with_aspect_ratio(self, prompt: str, aspect_ratio: str = "1:1", **kwargs) -> HunyuanImageResponse:
        """
        Generate image with predefined aspect ratio.
        
        Args:
            prompt (str): Text prompt for image generation
            aspect_ratio (str): Aspect ratio preset
            **kwargs: Additional generation parameters
            
        Returns:
            HunyuanImageResponse: Generation results
        """
        if aspect_ratio not in self.SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
        
        width, height = self.SUPPORTED_ASPECT_RATIOS[aspect_ratio]
        
        return self.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            **kwargs
        )
    
    def get_supported_aspect_ratios(self) -> Dict[str, Tuple[int, int]]:
        """Get supported aspect ratios and their dimensions."""
        return self.SUPPORTED_ASPECT_RATIOS.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the HunyuanImage manager."""
        success_rate = (self.successful_generations / self.total_generations * 100) if self.total_generations > 0 else 0
        
        return {
            "total_generations": self.total_generations,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "success_rate": success_rate,
            "average_generation_time": self.average_generation_time,
            "device": self.device,
            "use_local": self.use_local,
            "supported_aspect_ratios": list(self.SUPPORTED_ASPECT_RATIOS.keys())
        }
    
    def save_image(self, image: Image.Image, output_path: str, quality: int = 95) -> bool:
        """
        Save generated image to file.
        
        Args:
            image (Image.Image): Generated image
            output_path (str): Output file path
            quality (int): JPEG quality (1-100)
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            image.save(output_path, quality=quality, optimize=True)
            
            info(f"HunyuanImage saved to {output_path}", "hunyuan")
            return True
            
        except Exception as e:
            error(f"Failed to save HunyuanImage: {str(e)}", "hunyuan", e)
            return False


# Factory function for easy integration
def create_hunyuan_manager(api_key: str = None, use_local: bool = False, device: str = "auto") -> Optional[HunyuanImageManager]:
    """
    Factory function to create HunyuanImageManager instance.
    
    Args:
        api_key (str): HuggingFace API key. If None, tries to get from config.
        use_local (bool): Whether to use local model loading
        device (str): Device to use for local inference
        
    Returns:
        Optional[HunyuanImageManager]: HunyuanImage manager instance or None if failed
    """
    try:
        return HunyuanImageManager(api_key=api_key, use_local=use_local, device=device)
        
    except Exception as e:
        error(f"Failed to create HunyuanImage manager: {str(e)}", "hunyuan", e)
        return None
