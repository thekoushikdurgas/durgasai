"""
Facial AI Pipeline - Complete end-to-end facial feature to AI image generation.

This module provides a unified pipeline that combines facial feature extraction
with generative AI models to create personalized AI-generated portraits from
detailed facial input.

Key Features:
- Complete end-to-end pipeline from image input to AI-generated portrait
- Integration of facial feature extraction and generative models
- Multiple generation methods and style preferences
- Batch processing capabilities
- Comprehensive error handling and logging
- Integration with existing DurgasAI architecture

Key Classes:
- FacialAIPipeline: Main pipeline interface
- PipelineConfig: Configuration management for the pipeline
- PipelineResult: Structured result format for pipeline operations
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np

# Import existing utilities
from .config import Config
from .logger import debug, info, warning, error, log_model_operation, time_operation, LoggedOperation

# Import facial feature extraction and generative model components
try:
    from .facial_feature_extractor import FacialFeatureExtractor, create_facial_extractor
    FACIAL_EXTRACTION_AVAILABLE = True
except ImportError:
    FACIAL_EXTRACTION_AVAILABLE = False
    warning("Facial feature extraction not available", "facial_ai_pipeline")

try:
    from .generative_model_manager import GenerativeModelManager, create_generative_manager
    GENERATIVE_MODELS_AVAILABLE = True
except ImportError:
    GENERATIVE_MODELS_AVAILABLE = False
    warning("Generative models not available", "facial_ai_pipeline")


@dataclass
class PipelineConfig:
    """
    Configuration for the facial AI pipeline.
    
    This class contains all configurable parameters for the pipeline including
    model settings, generation parameters, and output preferences.
    """
    # Generation method
    generation_method: str = "controlnet"  # "controlnet", "stable_diffusion", or "hunyuan_image"
    
    # Style preferences
    style_preference: str = "professional"  # "professional", "artistic", "casual", "fantasy", "vintage"
    
    # Generation parameters
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    
    # Image processing
    enhancement_type: str = "basic"  # "basic", "advanced", "none"
    output_size: Tuple[int, int] = (512, 512)
    output_quality: int = 95
    
    # Pipeline settings
    save_intermediate_results: bool = False
    save_features: bool = True
    save_control_maps: bool = True
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # HunyuanImage-specific parameters
    hunyuan_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class PipelineResult:
    """
    Structured result format for pipeline operations.
    
    This class standardizes the output format for all pipeline operations
    including success/failure status, generated images, metadata, and error information.
    """
    success: bool
    generated_image: Optional[Image.Image] = None
    facial_features: Optional[Dict[str, Any]] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    pipeline_config: Optional[PipelineConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = {
            "success": self.success,
            "error": self.error,
            "processing_time": self.processing_time,
            "generation_metadata": self.generation_metadata,
            "facial_features": self.facial_features,
            "pipeline_config": self.pipeline_config.to_dict() if self.pipeline_config else None
        }
        
        # Handle PIL Image serialization
        if self.generated_image:
            result_dict["generated_image_size"] = self.generated_image.size
            result_dict["generated_image_mode"] = self.generated_image.mode
        
        return result_dict


class FacialAIPipeline:
    """
    Complete facial AI pipeline for generating personalized portraits.
    
    This class orchestrates the entire pipeline from facial feature extraction
    to AI-generated portrait creation, providing a unified interface for all
    facial AI operations.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the facial AI pipeline.
        
        Args:
            config (Optional[PipelineConfig]): Pipeline configuration
        """
        debug("Initializing FacialAIPipeline", "facial_ai_pipeline")
        
        # Set configuration
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.facial_extractor = None
        self.generative_manager = None
        
        # Performance tracking
        self.total_pipelines = 0
        self.successful_pipelines = 0
        self.failed_pipelines = 0
        self.total_processing_time = 0.0
        
        # Initialize components based on availability
        self._initialize_components()
        
        info("FacialAIPipeline initialized successfully", "facial_ai_pipeline",
             config=self.config.to_dict())
        log_model_operation("facial_ai_pipeline_initialized")
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize facial feature extractor
            if FACIAL_EXTRACTION_AVAILABLE:
                self.facial_extractor = create_facial_extractor()
                if self.facial_extractor:
                    debug("Facial feature extractor initialized", "facial_ai_pipeline")
                else:
                    warning("Failed to initialize facial feature extractor", "facial_ai_pipeline")
            else:
                warning("Facial feature extraction not available", "facial_ai_pipeline")
            
            # Initialize generative model manager
            if GENERATIVE_MODELS_AVAILABLE:
                self.generative_manager = create_generative_manager(self.config.device)
                if self.generative_manager:
                    debug("Generative model manager initialized", "facial_ai_pipeline")
                else:
                    warning("Failed to initialize generative model manager", "facial_ai_pipeline")
            else:
                warning("Generative models not available", "facial_ai_pipeline")
                
        except Exception as e:
            error(f"Error initializing pipeline components: {str(e)}", "facial_ai_pipeline", e)
    
    @time_operation("facial_ai_pipeline", "facial_ai_pipeline")
    def generate_portrait(self, image_path: str, 
                         custom_config: Optional[PipelineConfig] = None) -> PipelineResult:
        """
        Generate AI portrait from facial image.
        
        Args:
            image_path (str): Path to the input facial image
            custom_config (Optional[PipelineConfig]): Custom configuration for this generation
            
        Returns:
            PipelineResult: Complete pipeline result with generated image
        """
        start_time = time.time()
        self.total_pipelines += 1
        
        # Use custom config if provided, otherwise use default
        config = custom_config or self.config
        
        debug(f"Starting facial AI pipeline", "facial_ai_pipeline",
              image_path=image_path[:50] + "..." if len(image_path) > 50 else image_path,
              config=config.to_dict())
        
        try:
            with LoggedOperation("facial_ai_pipeline_generation", "facial_ai_pipeline",
                               extra_data={"image_path": image_path, "config": config.to_dict()}):
                
                # Step 1: Extract facial features
                if not self.facial_extractor:
                    self.failed_pipelines += 1
                    return PipelineResult(
                        success=False,
                        error="Facial feature extractor not available",
                        processing_time=time.time() - start_time,
                        pipeline_config=config
                    )
                
                debug("Step 1: Extracting facial features", "facial_ai_pipeline")
                facial_features = self.facial_extractor.extract_features(image_path)
                
                if not facial_features.get("success", False):
                    self.failed_pipelines += 1
                    return PipelineResult(
                        success=False,
                        error=f"Facial feature extraction failed: {facial_features.get('error', 'Unknown error')}",
                        processing_time=time.time() - start_time,
                        pipeline_config=config
                    )
                
                # Step 2: Generate AI portrait
                if not self.generative_manager:
                    self.failed_pipelines += 1
                    return PipelineResult(
                        success=False,
                        error="Generative model manager not available",
                        facial_features=facial_features,
                        processing_time=time.time() - start_time,
                        pipeline_config=config
                    )
                
                debug("Step 2: Generating AI portrait", "facial_ai_pipeline")
                
                # Prepare generation parameters
                generation_params = {
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
                    "enhancement_type": config.enhancement_type
                }
                
                # Add HunyuanImage-specific parameters if available
                if config.hunyuan_params:
                    generation_params.update(config.hunyuan_params)
                
                # Generate portrait
                generation_result = self.generative_manager.generate_portrait(
                    facial_features=facial_features,
                    method=config.generation_method,
                    style_preference=config.style_preference,
                    generation_params=generation_params
                )
                
                if not generation_result.get("success", False):
                    self.failed_pipelines += 1
                    return PipelineResult(
                        success=False,
                        error=f"Portrait generation failed: {generation_result.get('error', 'Unknown error')}",
                        facial_features=facial_features,
                        processing_time=time.time() - start_time,
                        pipeline_config=config
                    )
                
                # Step 3: Post-process and finalize
                generated_image = generation_result["image"]
                
                # Resize if needed
                if config.output_size != (512, 512):
                    generated_image = generated_image.resize(config.output_size, Image.Resampling.LANCZOS)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self.successful_pipelines += 1
                self.total_processing_time += processing_time
                
                # Create result
                result = PipelineResult(
                    success=True,
                    generated_image=generated_image,
                    facial_features=facial_features,
                    generation_metadata=generation_result.get("metadata", {}),
                    processing_time=processing_time,
                    pipeline_config=config
                )
                
                info(f"Facial AI pipeline completed successfully in {processing_time:.2f}s",
                     "facial_ai_pipeline",
                     processing_time=processing_time)
                
                log_model_operation("facial_ai_pipeline_completed",
                                  processing_time=processing_time,
                                  success=True)
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_pipelines += 1
            
            error_msg = f"Facial AI pipeline failed: {str(e)}"
            error(error_msg, "facial_ai_pipeline", e,
                  processing_time=processing_time)
            
            log_model_operation("facial_ai_pipeline_failed",
                              error=str(e),
                              processing_time=processing_time)
            
            return PipelineResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
                pipeline_config=config
            )
    
    def generate_portraits_batch(self, image_paths: List[str],
                               custom_config: Optional[PipelineConfig] = None) -> List[PipelineResult]:
        """
        Generate AI portraits for multiple images in batch.
        
        Args:
            image_paths (List[str]): List of paths to input facial images
            custom_config (Optional[PipelineConfig]): Custom configuration for all generations
            
        Returns:
            List[PipelineResult]: List of pipeline results for each image
        """
        debug(f"Starting batch generation for {len(image_paths)} images", "facial_ai_pipeline")
        
        results = []
        for i, image_path in enumerate(image_paths):
            info(f"Processing image {i+1}/{len(image_paths)}: {image_path}", "facial_ai_pipeline")
            
            result = self.generate_portrait(image_path, custom_config)
            results.append(result)
            
            # Log progress
            if result.success:
                debug(f"Successfully processed image {i+1}/{len(image_paths)}", "facial_ai_pipeline")
            else:
                warning(f"Failed to process image {i+1}/{len(image_paths)}: {result.error}", "facial_ai_pipeline")
        
        # Log batch summary
        successful = sum(1 for r in results if r.success)
        info(f"Batch generation completed: {successful}/{len(image_paths)} successful", "facial_ai_pipeline")
        
        return results
    
    def save_portrait(self, result: PipelineResult, output_path: str) -> bool:
        """
        Save generated portrait to file.
        
        Args:
            result (PipelineResult): Pipeline result containing generated image
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        if not result.success or not result.generated_image:
            error("Cannot save portrait: result not successful or no image", "facial_ai_pipeline")
            return False
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            quality = result.pipeline_config.output_quality if result.pipeline_config else 95
            result.generated_image.save(output_path, quality=quality, optimize=True)
            
            info(f"Portrait saved to {output_path}", "facial_ai_pipeline")
            return True
            
        except Exception as e:
            error(f"Failed to save portrait: {str(e)}", "facial_ai_pipeline", e)
            return False
    
    def save_pipeline_result(self, result: PipelineResult, output_dir: str) -> bool:
        """
        Save complete pipeline result including image, features, and metadata.
        
        Args:
            result (PipelineResult): Pipeline result to save
            output_dir (str): Output directory path
            
        Returns:
            bool: Success status
        """
        if not result.success:
            error("Cannot save pipeline result: result not successful", "facial_ai_pipeline")
            return False
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save generated image
            image_path = os.path.join(output_dir, "generated_portrait.jpg")
            if not self.save_portrait(result, image_path):
                return False
            
            # Save facial features if available
            if result.facial_features and result.pipeline_config.save_features:
                features_path = os.path.join(output_dir, "facial_features.json")
                with open(features_path, 'w') as f:
                    json.dump(result.facial_features, f, indent=2)
            
            # Save pipeline metadata
            metadata_path = os.path.join(output_dir, "pipeline_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # Save control maps if available
            if (result.facial_features and 
                result.pipeline_config.save_control_maps and 
                "control_maps" in result.facial_features):
                
                control_maps_dir = os.path.join(output_dir, "control_maps")
                os.makedirs(control_maps_dir, exist_ok=True)
                
                control_maps = result.facial_features["control_maps"]
                for map_type, control_map in control_maps.items():
                    if isinstance(control_map, np.ndarray):
                        map_path = os.path.join(control_maps_dir, f"{map_type}.jpg")
                        map_image = Image.fromarray(control_map)
                        map_image.save(map_path)
            
            info(f"Pipeline result saved to {output_dir}", "facial_ai_pipeline")
            return True
            
        except Exception as e:
            error(f"Failed to save pipeline result: {str(e)}", "facial_ai_pipeline", e)
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and availability.
        
        Returns:
            Dict[str, Any]: Pipeline status information
        """
        return {
            "facial_extraction_available": self.facial_extractor is not None,
            "generative_models_available": self.generative_manager is not None,
            "total_pipelines": self.total_pipelines,
            "successful_pipelines": self.successful_pipelines,
            "failed_pipelines": self.failed_pipelines,
            "success_rate": (self.successful_pipelines / self.total_pipelines * 100) if self.total_pipelines > 0 else 0,
            "average_processing_time": self.total_processing_time / self.successful_pipelines if self.successful_pipelines > 0 else 0,
            "current_config": self.config.to_dict()
        }
    
    def update_config(self, new_config: PipelineConfig):
        """
        Update pipeline configuration.
        
        Args:
            new_config (PipelineConfig): New configuration
        """
        self.config = new_config
        
        # Reinitialize generative manager if device changed
        if (self.generative_manager and 
            new_config.device != "auto" and 
            hasattr(self.generative_manager, 'device') and
            self.generative_manager.device != new_config.device):
            
            debug("Device changed, reinitializing generative manager", "facial_ai_pipeline")
            self.generative_manager = create_generative_manager(new_config.device)
        
        info("Pipeline configuration updated", "facial_ai_pipeline",
             new_config=new_config.to_dict())


# Factory function for easy integration
def create_facial_ai_pipeline(config: Optional[PipelineConfig] = None) -> Optional[FacialAIPipeline]:
    """
    Factory function to create FacialAIPipeline instance.
    
    Args:
        config (Optional[PipelineConfig]): Pipeline configuration
        
    Returns:
        Optional[FacialAIPipeline]: Facial AI pipeline instance or None if failed
    """
    try:
        return FacialAIPipeline(config)
    except Exception as e:
        error(f"Failed to create facial AI pipeline: {str(e)}", "facial_ai_pipeline", e)
        return None


# Convenience function for quick portrait generation
def generate_portrait_from_image(image_path: str, 
                               style: str = "professional",
                               method: str = "controlnet",
                               output_path: Optional[str] = None) -> Optional[PipelineResult]:
    """
    Convenience function to generate a portrait from an image.
    
    Args:
        image_path (str): Path to input image
        style (str): Style preference
        method (str): Generation method
        output_path (Optional[str]): Output path for saving
        
    Returns:
        Optional[PipelineResult]: Generated portrait result
    """
    try:
        # Create pipeline with default config
        config = PipelineConfig(
            generation_method=method,
            style_preference=style
        )
        
        pipeline = create_facial_ai_pipeline(config)
        if not pipeline:
            return None
        
        # Generate portrait
        result = pipeline.generate_portrait(image_path)
        
        # Save if output path provided
        if result.success and output_path:
            pipeline.save_portrait(result, output_path)
        
        return result
        
    except Exception as e:
        error(f"Quick portrait generation failed: {str(e)}", "facial_ai_pipeline", e)
        return None
