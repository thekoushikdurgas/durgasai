"""
Vision Model Manager for Hugging Face Image-Text-to-Text Models.

This module provides comprehensive vision AI model management capabilities including:
- HuggingFace API integration for vision models
- Image analysis and captioning
- Visual question answering
- Multi-modal conversation support
- Error handling and performance monitoring

Key Classes:
- VisionModelManager: Main interface for vision model operations
- VisionModelResponse: Structured response format for vision tasks
- ImageProcessor: Helper class for image preprocessing

The module integrates with the existing ModelManager architecture and supports
both chat-based and direct API approaches for vision tasks.
"""

import os
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from PIL import Image
import io

# HuggingFace imports
from huggingface_hub import InferenceClient

# Import existing configuration and logging
from .config import Config, ModelConfig, ModelProvider
from .logger import debug, info, warning, error, log_model_operation, time_operation, LoggedOperation


@dataclass
class VisionModelResponse:
    """
    Structured response format for vision model operations.
    
    This class standardizes the response format across different vision model types.
    It includes success/failure status, content, error information, and metadata for debugging.
    
    Attributes:
        content (str): The generated text response from the vision model
        success (bool): Whether the operation was successful
        error (Optional[str]): Error message if operation failed
        metadata (Optional[Dict[str, Any]]): Additional information like model name, timing, etc.
        image_processed (bool): Whether an image was successfully processed
        model_used (str): The model ID that was used for the operation
    """
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    image_processed: bool = False
    model_used: str = ""


class ImageProcessor:
    """
    Helper class for image preprocessing and validation.
    
    This class handles image loading, format conversion, size optimization,
    and validation before sending to vision models.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    OPTIMAL_SIZE = (1024, 1024)
    
    @staticmethod
    def validate_image(image_path: str) -> Dict[str, Any]:
        """
        Validate image file for processing.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict[str, Any]: Validation result with success status and details
        """
        try:
            if not os.path.exists(image_path):
                return {"success": False, "error": "Image file not found"}
            
            # Check file extension
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in ImageProcessor.SUPPORTED_FORMATS:
                return {
                    "success": False, 
                    "error": f"Unsupported image format: {file_ext}. Supported: {ImageProcessor.SUPPORTED_FORMATS}"
                }
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > ImageProcessor.MAX_IMAGE_SIZE:
                return {
                    "success": False,
                    "error": f"Image too large: {file_size / 1024 / 1024:.1f}MB. Max: {ImageProcessor.MAX_IMAGE_SIZE / 1024 / 1024}MB"
                }
            
            # Try to open and validate image
            with Image.open(image_path) as img:
                width, height = img.size
                format_info = img.format
                
                return {
                    "success": True,
                    "width": width,
                    "height": height,
                    "format": format_info,
                    "size_mb": file_size / 1024 / 1024
                }
                
        except Exception as e:
            return {"success": False, "error": f"Image validation failed: {str(e)}"}
    
    @staticmethod
    def optimize_image(image_path: str, target_size: tuple = None) -> str:
        """
        Optimize image for API processing.
        
        Args:
            image_path (str): Path to the input image
            target_size (tuple): Target size (width, height). If None, uses optimal size.
            
        Returns:
            str: Path to the optimized image (may be same as input if no optimization needed)
        """
        try:
            target_size = target_size or ImageProcessor.OPTIMAL_SIZE
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if significantly larger than target
                if img.size[0] > target_size[0] * 1.5 or img.size[1] > target_size[1] * 1.5:
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Save optimized version if changes were made
                if img.size != Image.open(image_path).size or img.mode != Image.open(image_path).mode:
                    optimized_path = str(Path(image_path).with_suffix('.optimized.jpg'))
                    img.save(optimized_path, 'JPEG', quality=85, optimize=True)
                    return optimized_path
            
            return image_path
            
        except Exception as e:
            warning(f"Image optimization failed: {str(e)}", "vision")
            return image_path


class VisionModelManager:
    """
    Manages vision AI models and provides unified interface for vision operations.
    
    This class extends the existing ModelManager architecture to support vision tasks:
    - Image analysis and description
    - Visual question answering
    - Multi-modal conversations
    - Image captioning
    - Error handling and performance monitoring
    
    Supported Model Types:
    - HuggingFace API vision models (cloud-based, fast)
    - Chat-based vision models (Llama, GLM-4.5V)
    - Specialized models (BLIP, CLIP)
    
    Architecture:
    - Uses HuggingFace InferenceClient for API calls
    - Integrates with existing logging and configuration systems
    - Implements comprehensive error handling and validation
    """
    
    def __init__(self, api_token: str):
        """
        Initialize the VisionModelManager.
        
        Args:
            api_token (str): HuggingFace API token for authentication
        """
        debug("Initializing VisionModelManager", "vision")
        
        if not api_token or not api_token.startswith('hf_'):
            raise ValueError("Invalid HuggingFace API token format")
        
        self.api_token = api_token
        self.client = InferenceClient(api_key=api_token)
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        
        info("VisionModelManager initialized successfully", "vision",
             api_token_present=bool(api_token))
        
        log_model_operation("vision_manager_initialized")
    
    @time_operation("vision_analysis", "vision")
    def analyze_image(self, image_path: str, text_prompt: str, 
                     model_id: str = "HuggingFaceM4/idefics2-8b") -> VisionModelResponse:
        """
        Analyze image with text prompt using vision model.
        
        This method performs comprehensive image analysis:
        1. Validates and optimizes the input image
        2. Sends request to HuggingFace vision API
        3. Processes and formats the response
        4. Logs performance metrics
        
        Args:
            image_path (str): Path to the image file or URL
            text_prompt (str): Text prompt describing what to analyze
            model_id (str): HuggingFace model ID to use
            
        Returns:
            VisionModelResponse: Structured response with analysis results
        """
        start_time = time.time()
        
        debug(f"Analyzing image with vision model", "vision",
              image_path=image_path[:50] + "..." if len(image_path) > 50 else image_path,
              text_prompt=text_prompt[:100] + "..." if len(text_prompt) > 100 else text_prompt,
              model_id=model_id)
        
        try:
            with LoggedOperation(f"analyze_image_{model_id}", "vision", 
                               extra_data={"model_id": model_id, "prompt_length": len(text_prompt)}):
                
                # Step 1: Validate image if it's a local file
                if not image_path.startswith(('http://', 'https://', 'data:')):
                    validation = ImageProcessor.validate_image(image_path)
                    if not validation["success"]:
                        return VisionModelResponse(
                            content="",
                            success=False,
                            error=f"Image validation failed: {validation['error']}",
                            image_processed=False,
                            model_used=model_id
                        )
                    
                    # Optimize image for processing
                    image_path = ImageProcessor.optimize_image(image_path)
                    debug(f"Image validated and optimized", "vision", 
                         validation_details=validation)
                
                # Step 2: Make API request
                completion = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_prompt},
                                {"type": "image_url", "image_url": {"url": image_path}}
                            ]
                        }
                    ]
                )
                
                # Step 3: Process response
                content = completion.choices[0].message.content
                response_time = time.time() - start_time
                
                # Update performance metrics
                self.total_requests += 1
                self.successful_requests += 1
                self.average_response_time = (
                    (self.average_response_time * (self.total_requests - 1) + response_time) 
                    / self.total_requests
                )
                
                info(f"Image analysis completed successfully in {response_time:.2f}s", "vision",
                    model_id=model_id,
                    response_length=len(content),
                    response_time=response_time)
                
                log_model_operation("vision_analysis_completed", model_id,
                    response_time=response_time,
                    success=True)
                
                return VisionModelResponse(
                    content=content,
                    success=True,
                    metadata={
                        "model_id": model_id,
                        "response_time": response_time,
                        "timestamp": time.time(),
                        "total_requests": self.total_requests
                    },
                    image_processed=True,
                    model_used=model_id
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            self.total_requests += 1
            self.failed_requests += 1
            
            error_msg = f"Error analyzing image with {model_id}: {str(e)}"
            error(error_msg, "vision", e,
                  model_id=model_id,
                  response_time=response_time)
            
            log_model_operation("vision_analysis_failed", model_id,
                error=str(e),
                response_time=response_time)
            
            return VisionModelResponse(
                content="",
                success=False,
                error=error_msg,
                image_processed=False,
                model_used=model_id
            )
    
    def generate_caption(self, image_path: str, 
                        model_id: str = "Salesforce/blip-image-captioning-base") -> VisionModelResponse:
        """
        Generate caption for image using specialized captioning model.
        
        Args:
            image_path (str): Path to the image file or URL
            model_id (str): HuggingFace model ID for captioning
            
        Returns:
            VisionModelResponse: Structured response with caption
        """
        start_time = time.time()
        
        debug(f"Generating caption for image", "vision",
              image_path=image_path[:50] + "..." if len(image_path) > 50 else image_path,
              model_id=model_id)
        
        try:
            with LoggedOperation(f"generate_caption_{model_id}", "vision",
                               extra_data={"model_id": model_id}):
                
                # Validate image if local file
                if not image_path.startswith(('http://', 'https://', 'data:')):
                    validation = ImageProcessor.validate_image(image_path)
                    if not validation["success"]:
                        return VisionModelResponse(
                            content="",
                            success=False,
                            error=f"Image validation failed: {validation['error']}",
                            image_processed=False,
                            model_used=model_id
                        )
                
                # Generate caption
                response = self.client.image_to_text(image_path)
                
                response_time = time.time() - start_time
                
                # Update metrics
                self.total_requests += 1
                self.successful_requests += 1
                self.average_response_time = (
                    (self.average_response_time * (self.total_requests - 1) + response_time) 
                    / self.total_requests
                )
                
                info(f"Caption generated successfully in {response_time:.2f}s", "vision",
                    model_id=model_id,
                    caption_length=len(str(response)),
                    response_time=response_time)
                
                log_model_operation("caption_generated", model_id,
                    response_time=response_time,
                    success=True)
                
                return VisionModelResponse(
                    content=str(response),
                    success=True,
                    metadata={
                        "model_id": model_id,
                        "response_time": response_time,
                        "timestamp": time.time(),
                        "total_requests": self.total_requests
                    },
                    image_processed=True,
                    model_used=model_id
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            self.total_requests += 1
            self.failed_requests += 1
            
            error_msg = f"Error generating caption with {model_id}: {str(e)}"
            error(error_msg, "vision", e,
                  model_id=model_id,
                  response_time=response_time)
            
            log_model_operation("caption_generation_failed", model_id,
                error=str(e),
                response_time=response_time)
            
            return VisionModelResponse(
                content="",
                success=False,
                error=error_msg,
                image_processed=False,
                model_used=model_id
            )
    
    def visual_question_answering(self, image_path: str, question: str,
                                 model_id: str = "Salesforce/blip-vqa-base") -> VisionModelResponse:
        """
        Answer questions about images using VQA model.
        
        Args:
            image_path (str): Path to the image file or URL
            question (str): Question to ask about the image
            model_id (str): HuggingFace model ID for VQA
            
        Returns:
            VisionModelResponse: Structured response with answer
        """
        start_time = time.time()
        
        debug(f"Answering visual question", "vision",
              image_path=image_path[:50] + "..." if len(image_path) > 50 else image_path,
              question=question[:100] + "..." if len(question) > 100 else question,
              model_id=model_id)
        
        try:
            with LoggedOperation(f"visual_qa_{model_id}", "vision",
                               extra_data={"model_id": model_id, "question_length": len(question)}):
                
                # Validate image if local file
                if not image_path.startswith(('http://', 'https://', 'data:')):
                    validation = ImageProcessor.validate_image(image_path)
                    if not validation["success"]:
                        return VisionModelResponse(
                            content="",
                            success=False,
                            error=f"Image validation failed: {validation['error']}",
                            image_processed=False,
                            model_used=model_id
                        )
                
                # Answer visual question
                response = self.client.image_to_text(image_path, text=question)
                
                response_time = time.time() - start_time
                
                # Update metrics
                self.total_requests += 1
                self.successful_requests += 1
                self.average_response_time = (
                    (self.average_response_time * (self.total_requests - 1) + response_time) 
                    / self.total_requests
                )
                
                info(f"Visual question answered successfully in {response_time:.2f}s", "vision",
                    model_id=model_id,
                    answer_length=len(str(response)),
                    response_time=response_time)
                
                log_model_operation("visual_qa_completed", model_id,
                    response_time=response_time,
                    success=True)
                
                return VisionModelResponse(
                    content=str(response),
                    success=True,
                    metadata={
                        "model_id": model_id,
                        "response_time": response_time,
                        "timestamp": time.time(),
                        "total_requests": self.total_requests,
                        "question": question
                    },
                    image_processed=True,
                    model_used=model_id
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            self.total_requests += 1
            self.failed_requests += 1
            
            error_msg = f"Error answering visual question with {model_id}: {str(e)}"
            error(error_msg, "vision", e,
                  model_id=model_id,
                  response_time=response_time)
            
            log_model_operation("visual_qa_failed", model_id,
                error=str(e),
                response_time=response_time)
            
            return VisionModelResponse(
                content="",
                success=False,
                error=error_msg,
                image_processed=False,
                model_used=model_id
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the vision model manager.
        
        Returns:
            Dict[str, Any]: Performance metrics including success rate, average response time, etc.
        """
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "average_response_time": self.average_response_time
        }


# Factory function for easy integration
def create_vision_manager(api_token: str = None) -> Optional[VisionModelManager]:
    """
    Factory function to create VisionModelManager instance.
    
    Args:
        api_token (str): HuggingFace API token. If None, tries to get from config.
        
    Returns:
        Optional[VisionModelManager]: Vision model manager instance or None if failed
    """
    try:
        if not api_token:
            # Try to get from config
            api_config = Config.get_api_config("huggingface")
            if api_config and "api_keys" in api_config:
                api_token = api_config["api_keys"]
            else:
                error("No HuggingFace API token provided and none found in config", "vision")
                return None
        
        return VisionModelManager(api_token)
        
    except Exception as e:
        error(f"Failed to create vision manager: {str(e)}", "vision", e)
        return None
