"""
Generative Model Manager for AI Image Generation.

This module provides comprehensive generative AI model management capabilities for creating
AI-generated images from facial features. It integrates with the existing DurgasAI architecture
to support Stable Diffusion, ControlNet, and other generative models.

Key Features:
- Stable Diffusion integration with HuggingFace
- ControlNet support for guided generation
- Facial feature-based prompt generation
- Multiple generation styles and preferences
- Integration with existing model management system
- Comprehensive error handling and logging

Key Classes:
- GenerativeModelManager: Main generative model interface
- PromptGenerator: Converts facial features to text prompts
- ImagePostProcessor: Post-processing for generated images
"""

import os
import torch
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
import base64
from io import BytesIO

# HuggingFace Diffusers imports
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: Diffusers not available. Install with: pip install diffusers")

# OpenCV for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

# Import existing configuration and logging
from .config import Config
from .logger import debug, info, warning, error, log_model_operation, time_operation, LoggedOperation

# Import HunyuanImage manager
from .hunyuan_image_manager import HunyuanImageManager, create_hunyuan_manager


class PromptGenerator:
    """
    Generates detailed text prompts from facial features.
    
    This class converts extracted facial attributes into natural language prompts
    that can guide generative models to create personalized portraits.
    """
    
    def __init__(self):
        """Initialize the prompt generator."""
        self.style_templates = {
            "professional": {
                "description": "Professional headshot, studio lighting, clean background",
                "quality": "high quality, detailed, photorealistic",
                "style": "corporate, business, formal"
            },
            "artistic": {
                "description": "Artistic portrait, creative lighting, dramatic shadows",
                "quality": "high quality, detailed, artistic",
                "style": "creative, expressive, dramatic"
            },
            "casual": {
                "description": "Casual portrait, natural lighting, friendly expression",
                "quality": "high quality, detailed, natural",
                "style": "relaxed, friendly, approachable"
            },
            "fantasy": {
                "description": "Fantasy portrait, magical atmosphere, ethereal lighting",
                "quality": "high quality, detailed, magical",
                "style": "fantasy, mystical, enchanting"
            },
            "vintage": {
                "description": "Vintage portrait, classic lighting, retro style",
                "quality": "high quality, detailed, vintage",
                "style": "classic, timeless, elegant"
            }
        }
        
        debug("PromptGenerator initialized", "generative_models")
    
    def generate_prompt_from_attributes(self, attributes: Dict[str, Any], 
                                      style_preference: str = "professional",
                                      gender_preference: str = "neutral",
                                      age_preference: str = "adult") -> str:
        """
        Generate a detailed text prompt from facial attributes.
        
        Args:
            attributes (Dict[str, Any]): Facial attributes from feature extraction
            style_preference (str): Style preference for generation
            gender_preference (str): Gender preference ("male", "female", "neutral")
            age_preference (str): Age preference ("young", "adult", "mature", "neutral")
            
        Returns:
            str: Generated text prompt
        """
        try:
            # Extract facial features
            face_features = self._extract_face_features(attributes)
            
            # Get style template
            style_template = self.style_templates.get(style_preference, self.style_templates["professional"])
            
            # Build prompt components
            prompt_parts = []
            
            # Add person description
            person_desc = self._build_person_description(face_features, gender_preference, age_preference)
            prompt_parts.append(person_desc)
            
            # Add facial feature descriptions
            feature_desc = self._build_feature_description(face_features)
            prompt_parts.append(feature_desc)
            
            # Add style and quality
            prompt_parts.append(style_template["description"])
            prompt_parts.append(style_template["quality"])
            
            # Combine all parts
            full_prompt = ", ".join(filter(None, prompt_parts))
            
            debug(f"Generated prompt: {full_prompt[:100]}...", "generative_models")
            return full_prompt
            
        except Exception as e:
            error(f"Prompt generation failed: {str(e)}", "generative_models", e)
            return "A person, high quality, detailed"
    
    def _extract_face_features(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize face features from attributes."""
        features = {}
        
        # Face shape
        face_shape_data = attributes.get("face_shape", {})
        features["face_shape"] = face_shape_data.get("face_shape", "oval")
        
        # Eye features
        eye_features = attributes.get("eye_features", {})
        features["eye_shape"] = eye_features.get("eye_shape", "almond")
        
        # Nose features
        nose_features = attributes.get("nose_features", {})
        features["nose_type"] = nose_features.get("nose_type", "medium")
        
        # Mouth features
        mouth_features = attributes.get("mouth_features", {})
        features["lip_fullness"] = mouth_features.get("lip_fullness", "medium")
        
        return features
    
    def _build_person_description(self, features: Dict[str, Any], 
                                gender_preference: str, age_preference: str) -> str:
        """Build person description part of the prompt."""
        parts = []
        
        # Add age/gender if specified
        if age_preference != "neutral":
            parts.append(f"{age_preference}")
        
        if gender_preference != "neutral":
            parts.append(f"{gender_preference}")
        
        parts.append("person")
        
        return " ".join(parts)
    
    def _build_feature_description(self, features: Dict[str, Any]) -> str:
        """Build facial feature description part of the prompt."""
        parts = []
        
        # Face shape
        face_shape = features.get("face_shape", "oval")
        parts.append(f"{face_shape} face")
        
        # Eye shape
        eye_shape = features.get("eye_shape", "almond")
        parts.append(f"{eye_shape}-shaped eyes")
        
        # Nose type
        nose_type = features.get("nose_type", "medium")
        if nose_type != "medium":
            parts.append(f"{nose_type} nose")
        
        # Lip fullness
        lip_fullness = features.get("lip_fullness", "medium")
        if lip_fullness != "medium":
            parts.append(f"{lip_fullness} lips")
        
        return ", ".join(parts)


class ImagePostProcessor:
    """
    Post-processes generated images for quality enhancement.
    
    This class provides various post-processing techniques to enhance
    the quality and appearance of generated images.
    """
    
    def __init__(self):
        """Initialize the image post-processor."""
        debug("ImagePostProcessor initialized", "generative_models")
    
    def enhance_image(self, image: Image.Image, enhancement_type: str = "basic") -> Image.Image:
        """
        Enhance a generated image.
        
        Args:
            image (Image.Image): Input image
            enhancement_type (str): Type of enhancement ("basic", "advanced", "none")
            
        Returns:
            Image.Image: Enhanced image
        """
        if enhancement_type == "none":
            return image
        
        try:
            if enhancement_type == "basic":
                return self._basic_enhancement(image)
            elif enhancement_type == "advanced":
                return self._advanced_enhancement(image)
            else:
                return self._basic_enhancement(image)
                
        except Exception as e:
            error(f"Image enhancement failed: {str(e)}", "generative_models", e)
            return image
    
    def _basic_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply basic image enhancements."""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic sharpening
        image_array = np.array(image)
        
        # Apply slight contrast enhancement
        image_array = np.clip(image_array * 1.05, 0, 255).astype(np.uint8)
        
        return Image.fromarray(image_array)
    
    def _advanced_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply advanced image enhancements."""
        if not CV2_AVAILABLE:
            return self._basic_enhancement(image)
        
        try:
            # Convert PIL to OpenCV format
            image_array = np.array(image)
            if image_array.shape[2] == 3:  # RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply advanced enhancements
            # 1. Denoising
            image_array = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
            
            # 2. Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            image_array = cv2.filter2D(image_array, -1, kernel)
            
            # 3. Contrast enhancement
            lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            image_array = cv2.merge([l, a, b])
            image_array = cv2.cvtColor(image_array, cv2.COLOR_LAB2BGR)
            
            # Convert back to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            error(f"Advanced enhancement failed: {str(e)}", "generative_models", e)
            return self._basic_enhancement(image)
    
    def resize_image(self, image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image (Image.Image): Input image
            target_size (Tuple[int, int]): Target size (width, height)
            
        Returns:
            Image.Image: Resized image
        """
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def save_image(self, image: Image.Image, output_path: str, quality: int = 95) -> bool:
        """
        Save image to file.
        
        Args:
            image (Image.Image): Image to save
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
            
            info(f"Image saved to {output_path}", "generative_models")
            return True
            
        except Exception as e:
            error(f"Failed to save image: {str(e)}", "generative_models", e)
            return False


class GenerativeModelManager:
    """
    Manages generative AI models for image generation.
    
    This class provides a unified interface for various generative models
    including Stable Diffusion and ControlNet for creating AI-generated images.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the generative model manager.
        
        Args:
            device (str): Device to use for inference ("auto", "cuda", "cpu")
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers is required for generative models. Install with: pip install diffusers")
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        debug(f"Initializing GenerativeModelManager on device: {self.device}", "generative_models")
        
        # Initialize components
        self.prompt_generator = PromptGenerator()
        self.image_post_processor = ImagePostProcessor()
        
        # Model configurations
        self.model_configs = {
            "controlnet": {
                "model_id": "lllyasviel/sd-controlnet-canny",
                "pipeline_model_id": "runwayml/stable-diffusion-v1-5"
            },
            "stable_diffusion": {
                "model_id": "runwayml/stable-diffusion-v1-5"
            },
            "hunyuan_image": {
                "model_id": "tencent/HunyuanImage-2.1"
            }
        }
        
        # Initialize models (will be loaded on first use)
        self.controlnet_pipeline = None
        self.stable_diffusion_pipeline = None
        self.controlnet_model = None
        self.hunyuan_manager = None
        
        # Performance tracking
        self.total_generations = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        info("GenerativeModelManager initialized successfully", "generative_models")
        log_model_operation("generative_model_manager_initialized")
    
    @time_operation("load_controlnet_pipeline", "generative_models")
    def load_controlnet_pipeline(self) -> bool:
        """
        Load ControlNet pipeline for guided image generation.
        
        Returns:
            bool: Success status
        """
        try:
            debug("Loading ControlNet pipeline...", "generative_models")
            
            # Load ControlNet model
            self.controlnet_model = ControlNetModel.from_pretrained(
                self.model_configs["controlnet"]["model_id"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Load pipeline
            self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_configs["controlnet"]["pipeline_model_id"],
                controlnet=self.controlnet_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device and optimize
            self.controlnet_pipeline = self.controlnet_pipeline.to(self.device)
            
            # Use efficient scheduler
            self.controlnet_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.controlnet_pipeline.scheduler.config
            )
            
            # Enable memory efficient attention if available
            if hasattr(self.controlnet_pipeline, "enable_memory_efficient_attention"):
                self.controlnet_pipeline.enable_memory_efficient_attention()
            
            info("ControlNet pipeline loaded successfully", "generative_models")
            log_model_operation("controlnet_pipeline_loaded", success=True)
            return True
            
        except Exception as e:
            error(f"Error loading ControlNet pipeline: {e}", "generative_models", e)
            log_model_operation("controlnet_pipeline_load_failed", error=str(e))
            return False
    
    @time_operation("load_stable_diffusion_pipeline", "generative_models")
    def load_stable_diffusion_pipeline(self) -> bool:
        """
        Load standard Stable Diffusion pipeline.
        
        Returns:
            bool: Success status
        """
        try:
            debug("Loading Stable Diffusion pipeline...", "generative_models")
            
            self.stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_configs["stable_diffusion"]["model_id"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.stable_diffusion_pipeline = self.stable_diffusion_pipeline.to(self.device)
            
            # Use efficient scheduler
            self.stable_diffusion_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.stable_diffusion_pipeline.scheduler.config
            )
            
            info("Stable Diffusion pipeline loaded successfully", "generative_models")
            log_model_operation("stable_diffusion_pipeline_loaded", success=True)
            return True
            
        except Exception as e:
            error(f"Error loading Stable Diffusion pipeline: {e}", "generative_models", e)
            log_model_operation("stable_diffusion_pipeline_load_failed", error=str(e))
            return False
    
    @time_operation("generate_with_controlnet", "generative_models")
    def generate_with_controlnet(self, 
                               control_map: np.ndarray, 
                               facial_features: Dict[str, Any],
                               style_preference: str = "professional",
                               generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate image using ControlNet with facial feature guidance.
        
        Args:
            control_map (np.ndarray): Control map from facial feature extraction
            facial_features (Dict[str, Any]): Facial features and attributes
            style_preference (str): Style preference for generation
            generation_params (Dict[str, Any]): Additional generation parameters
            
        Returns:
            Dict[str, Any]: Generation results
        """
        start_time = time.time()
        self.total_generations += 1
        
        debug(f"Generating image with ControlNet", "generative_models",
              style_preference=style_preference)
        
        try:
            with LoggedOperation("controlnet_generation", "generative_models",
                               extra_data={"style_preference": style_preference}):
                
                # Load pipeline if not already loaded
                if self.controlnet_pipeline is None:
                    if not self.load_controlnet_pipeline():
                        self.failed_generations += 1
                        return {"success": False, "error": "Failed to load ControlNet pipeline"}
                
                # Generate prompt from facial features
                prompt = self.prompt_generator.generate_prompt_from_attributes(
                    facial_features, style_preference
                )
                
                # Preprocess control map
                control_image = self._preprocess_control_map(control_map)
                
                # Set generation parameters
                params = generation_params or {}
                num_inference_steps = params.get("num_inference_steps", 20)
                guidance_scale = params.get("guidance_scale", 7.5)
                controlnet_conditioning_scale = params.get("controlnet_conditioning_scale", 1.0)
                
                debug(f"ControlNet generation parameters: steps={num_inference_steps}, guidance={guidance_scale}",
                      "generative_models")
                
                # Generate image
                with torch.autocast(self.device):
                    result = self.controlnet_pipeline(
                        prompt=prompt,
                        image=control_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale
                    )
                
                generated_image = result.images[0]
                
                # Post-process image
                enhancement_type = params.get("enhancement_type", "basic")
                generated_image = self.image_post_processor.enhance_image(
                    generated_image, enhancement_type
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self.successful_generations += 1
                
                result = {
                    "success": True,
                    "image": generated_image,
                    "prompt": prompt,
                    "method": "controlnet",
                    "metadata": {
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "controlnet_conditioning_scale": controlnet_conditioning_scale,
                        "processing_time": processing_time,
                        "device": self.device,
                        "enhancement_type": enhancement_type
                    }
                }
                
                info(f"ControlNet generation completed successfully in {processing_time:.2f}s",
                     "generative_models",
                     processing_time=processing_time)
                
                log_model_operation("controlnet_generation_completed",
                                  processing_time=processing_time,
                                  success=True)
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_generations += 1
            
            error_msg = f"ControlNet generation failed: {str(e)}"
            error(error_msg, "generative_models", e,
                  processing_time=processing_time)
            
            log_model_operation("controlnet_generation_failed",
                              error=str(e),
                              processing_time=processing_time)
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    @time_operation("generate_with_stable_diffusion", "generative_models")
    def generate_with_stable_diffusion(self, 
                                     facial_features: Dict[str, Any],
                                     style_preference: str = "professional",
                                     generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate image using standard Stable Diffusion.
        
        Args:
            facial_features (Dict[str, Any]): Facial features and attributes
            style_preference (str): Style preference for generation
            generation_params (Dict[str, Any]): Additional generation parameters
            
        Returns:
            Dict[str, Any]: Generation results
        """
        start_time = time.time()
        self.total_generations += 1
        
        debug(f"Generating image with Stable Diffusion", "generative_models",
              style_preference=style_preference)
        
        try:
            with LoggedOperation("stable_diffusion_generation", "generative_models",
                               extra_data={"style_preference": style_preference}):
                
                # Load pipeline if not already loaded
                if self.stable_diffusion_pipeline is None:
                    if not self.load_stable_diffusion_pipeline():
                        self.failed_generations += 1
                        return {"success": False, "error": "Failed to load Stable Diffusion pipeline"}
                
                # Generate prompt from facial features
                prompt = self.prompt_generator.generate_prompt_from_attributes(
                    facial_features, style_preference
                )
                
                # Set generation parameters
                params = generation_params or {}
                num_inference_steps = params.get("num_inference_steps", 20)
                guidance_scale = params.get("guidance_scale", 7.5)
                
                debug(f"Stable Diffusion generation parameters: steps={num_inference_steps}, guidance={guidance_scale}",
                      "generative_models")
                
                # Generate image
                with torch.autocast(self.device):
                    result = self.stable_diffusion_pipeline(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )
                
                generated_image = result.images[0]
                
                # Post-process image
                enhancement_type = params.get("enhancement_type", "basic")
                generated_image = self.image_post_processor.enhance_image(
                    generated_image, enhancement_type
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self.successful_generations += 1
                
                result = {
                    "success": True,
                    "image": generated_image,
                    "prompt": prompt,
                    "method": "stable_diffusion",
                    "metadata": {
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "processing_time": processing_time,
                        "device": self.device,
                        "enhancement_type": enhancement_type
                    }
                }
                
                info(f"Stable Diffusion generation completed successfully in {processing_time:.2f}s",
                     "generative_models",
                     processing_time=processing_time)
                
                log_model_operation("stable_diffusion_generation_completed",
                                  processing_time=processing_time,
                                  success=True)
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_generations += 1
            
            error_msg = f"Stable Diffusion generation failed: {str(e)}"
            error(error_msg, "generative_models", e,
                  processing_time=processing_time)
            
            log_model_operation("stable_diffusion_generation_failed",
                              error=str(e),
                              processing_time=processing_time)
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    @time_operation("load_hunyuan_manager", "generative_models")
    def load_hunyuan_manager(self, use_local: bool = False) -> bool:
        """
        Load HunyuanImage manager for high-resolution image generation.
        
        Args:
            use_local (bool): Whether to use local model loading
            
        Returns:
            bool: Success status
        """
        try:
            debug("Loading HunyuanImage manager...", "generative_models")
            
            self.hunyuan_manager = create_hunyuan_manager(use_local=use_local, device=self.device)
            
            if self.hunyuan_manager is None:
                raise RuntimeError("Failed to create HunyuanImage manager")
            
            info("HunyuanImage manager loaded successfully", "generative_models")
            log_model_operation("hunyuan_manager_loaded", success=True)
            return True
            
        except Exception as e:
            error(f"Error loading HunyuanImage manager: {e}", "generative_models", e)
            log_model_operation("hunyuan_manager_load_failed", error=str(e))
            return False
    
    @time_operation("generate_with_hunyuan", "generative_models")
    def generate_with_hunyuan(self, 
                             prompt: str,
                             facial_features: Dict[str, Any] = None,
                             aspect_ratio: str = "1:1",
                             use_refiner: bool = True,
                             use_prompt_enhancement: bool = False,
                             generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate high-resolution image using HunyuanImage-2.1.
        
        Args:
            prompt (str): Text prompt for image generation
            facial_features (Dict[str, Any]): Facial features for enhanced prompts
            aspect_ratio (str): Aspect ratio preset
            use_refiner (bool): Whether to use the refiner model
            use_prompt_enhancement (bool): Whether to use prompt enhancement
            generation_params (Dict[str, Any]): Additional generation parameters
            
        Returns:
            Dict[str, Any]: Generation results
        """
        start_time = time.time()
        self.total_generations += 1
        
        debug(f"Generating image with HunyuanImage-2.1", "generative_models",
              prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
              aspect_ratio=aspect_ratio)
        
        try:
            with LoggedOperation("hunyuan_generation", "generative_models",
                               extra_data={"aspect_ratio": aspect_ratio, "has_facial_features": bool(facial_features)}):
                
                # Load HunyuanImage manager if not already loaded
                if self.hunyuan_manager is None:
                    if not self.load_hunyuan_manager():
                        self.failed_generations += 1
                        return {"success": False, "error": "Failed to load HunyuanImage manager"}
                
                # Enhance prompt with facial features if provided
                if facial_features:
                    enhanced_prompt = self._enhance_prompt_with_facial_features(prompt, facial_features)
                else:
                    enhanced_prompt = prompt
                
                # Set generation parameters
                params = generation_params or {}
                num_inference_steps = params.get("num_inference_steps", 50)
                guidance_scale = params.get("guidance_scale", 3.5)
                seed = params.get("seed")
                
                debug(f"HunyuanImage generation parameters: steps={num_inference_steps}, guidance={guidance_scale}",
                      "generative_models")
                
                # Generate image
                result = self.hunyuan_manager.generate_with_aspect_ratio(
                    prompt=enhanced_prompt,
                    aspect_ratio=aspect_ratio,
                    use_refiner=use_refiner,
                    use_prompt_enhancement=use_prompt_enhancement,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                if not result.success:
                    self.failed_generations += 1
                    return {"success": False, "error": result.error}
                
                generated_image = result.image
                
                # Post-process image
                enhancement_type = params.get("enhancement_type", "basic")
                generated_image = self.image_post_processor.enhance_image(
                    generated_image, enhancement_type
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self.successful_generations += 1
                
                result_dict = {
                    "success": True,
                    "image": generated_image,
                    "prompt": enhanced_prompt,
                    "original_prompt": prompt,
                    "method": "hunyuan_image",
                    "metadata": {
                        "aspect_ratio": aspect_ratio,
                        "use_refiner": use_refiner,
                        "use_prompt_enhancement": use_prompt_enhancement,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "processing_time": processing_time,
                        "device": self.device,
                        "enhancement_type": enhancement_type,
                        "seed": seed,
                        "has_facial_features": bool(facial_features)
                    }
                }
                
                info(f"HunyuanImage generation completed successfully in {processing_time:.2f}s",
                     "generative_models",
                     processing_time=processing_time)
                
                log_model_operation("hunyuan_generation_completed",
                                  processing_time=processing_time,
                                  success=True)
                
                return result_dict
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_generations += 1
            
            error_msg = f"HunyuanImage generation failed: {str(e)}"
            error(error_msg, "generative_models", e,
                  processing_time=processing_time)
            
            log_model_operation("hunyuan_generation_failed",
                              error=str(e),
                              processing_time=processing_time)
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def _enhance_prompt_with_facial_features(self, prompt: str, facial_features: Dict[str, Any]) -> str:
        """
        Enhance prompt with facial feature information.
        
        Args:
            prompt (str): Original prompt
            facial_features (Dict[str, Any]): Facial features from extraction
            
        Returns:
            str: Enhanced prompt
        """
        try:
            # Generate facial feature description
            facial_desc = self.prompt_generator.generate_prompt_from_attributes(
                facial_features, style_preference="neutral"
            )
            
            # Combine with original prompt
            enhanced_prompt = f"{prompt}, featuring {facial_desc}"
            
            debug(f"Enhanced prompt with facial features", "generative_models",
                  original_length=len(prompt), enhanced_length=len(enhanced_prompt))
            
            return enhanced_prompt
            
        except Exception as e:
            error(f"Failed to enhance prompt with facial features: {str(e)}", "generative_models", e)
            return prompt
    
    def generate_portrait(self, 
                         facial_features: Dict[str, Any], 
                         method: str = "controlnet",
                         style_preference: str = "professional",
                         generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete portrait generation pipeline.
        
        Args:
            facial_features (Dict[str, Any]): Complete facial feature extraction results
            method (str): Generation method ("controlnet" or "stable_diffusion")
            style_preference (str): Style preference for generation
            generation_params (Dict[str, Any]): Additional generation parameters
            
        Returns:
            Dict[str, Any]: Complete generation results
        """
        if not facial_features.get("success", False):
            return {"success": False, "error": "Invalid facial features provided"}
        
        attributes = facial_features.get("attributes", {})
        control_maps = facial_features.get("control_maps", {})
        
        # Choose generation method
        if method == "controlnet":
            control_map = control_maps.get("contours") or control_maps.get("points")
            if control_map is not None:
                return self.generate_with_controlnet(
                    control_map=control_map,
                    facial_features=attributes,
                    style_preference=style_preference,
                    generation_params=generation_params
                )
            else:
                return {"success": False, "error": "ControlNet requires control map but none provided"}
        
        elif method == "stable_diffusion":
            return self.generate_with_stable_diffusion(
                facial_features=attributes,
                style_preference=style_preference,
                generation_params=generation_params
            )
        

        elif method == "hunyuan_image":
            # Generate a prompt from facial features
            prompt = self.prompt_generator.generate_prompt_from_attributes(
                attributes, style_preference
            )
            
            # Extract HunyuanImage-specific parameters
            hunyuan_params = generation_params or {}
            aspect_ratio = hunyuan_params.get("aspect_ratio", "1:1")
            use_refiner = hunyuan_params.get("use_refiner", True)
            use_prompt_enhancement = hunyuan_params.get("use_prompt_enhancement", False)
            
            return self.generate_with_hunyuan(
                prompt=prompt,
                facial_features=attributes,
                aspect_ratio=aspect_ratio,
                use_refiner=use_refiner,
                use_prompt_enhancement=use_prompt_enhancement,
                generation_params=generation_params
            )
        
        else:
            return {"success": False, "error": f"Invalid method '{method}'. Supported methods: controlnet, stable_diffusion, hunyuan_image"}
    
    def _preprocess_control_map(self, control_map: np.ndarray, 
                              target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Preprocess control map for ControlNet.
        
        Args:
            control_map (np.ndarray): Control map from facial feature extraction
            target_size (Tuple[int, int]): Target size for the control map
            
        Returns:
            Image.Image: Preprocessed control map
        """
        # Resize to target size
        resized_map = cv2.resize(control_map, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to PIL Image
        control_image = Image.fromarray(resized_map).convert("RGB")
        
        return control_image
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the generative model manager.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        success_rate = (self.successful_generations / self.total_generations * 100) if self.total_generations > 0 else 0
        
        return {
            "total_generations": self.total_generations,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "success_rate": success_rate,
            "device": self.device
        }
    
    def save_generated_image(self, image: Image.Image, output_path: str, 
                           quality: int = 95) -> bool:
        """
        Save generated image to file.
        
        Args:
            image (Image.Image): Generated image
            output_path (str): Output file path
            quality (int): JPEG quality (1-100)
            
        Returns:
            bool: Success status
        """
        return self.image_post_processor.save_image(image, output_path, quality)


# Factory function for easy integration
def create_generative_manager(device: str = "auto") -> Optional[GenerativeModelManager]:
    """
    Factory function to create GenerativeModelManager instance.
    
    Args:
        device (str): Device to use for inference
        
    Returns:
        Optional[GenerativeModelManager]: Generative model manager instance or None if failed
    """
    try:
        if not DIFFUSERS_AVAILABLE:
            error("Diffusers not available for generative models", "generative_models")
            return None
        
        return GenerativeModelManager(device)
        
    except Exception as e:
        error(f"Failed to create generative model manager: {str(e)}", "generative_models", e)
        return None
