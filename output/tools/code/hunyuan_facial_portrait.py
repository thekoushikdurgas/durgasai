import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

# Import required components
try:
    from utils.hunyuan_image_manager import create_hunyuan_manager
    from utils.generative_model_manager import create_generative_manager
    from utils.facial_feature_extractor import create_facial_extractor
    from utils.logger import info, error, debug
    HUNYUAN_FACIAL_AVAILABLE = True
except ImportError as e:
    HUNYUAN_FACIAL_AVAILABLE = False
    print(f"Warning: HunyuanImage facial components not available: {e}")

def generate_hunyuan_facial_portrait(
    image_path: str,
    style_preference: str = "professional",
    aspect_ratio: str = "1:1",
    use_refiner: bool = True,
    use_prompt_enhancement: bool = True,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    additional_prompt: Optional[str] = None,
    use_local: bool = False
) -> Dict[str, Any]:
    """
    Generate high-resolution 2K facial portrait using HunyuanImage-2.1 with facial feature enhancement.
    
    Args:
        image_path (str): Path to the input facial image
        style_preference (str): Artistic style for the generated portrait
        aspect_ratio (str): Aspect ratio for the generated portrait
        use_refiner (bool): Enable refiner model for higher quality
        use_prompt_enhancement (bool): Automatically enhance prompts based on facial features
        num_inference_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        seed (Optional[int]): Random seed for reproducible generation
        additional_prompt (Optional[str]): Additional text to add to the prompt
        use_local (bool): Use local model instead of API
        
    Returns:
        Dict[str, Any]: Generation results with portrait image and facial analysis
    """
    
    if not HUNYUAN_FACIAL_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage facial components not available. Please check installation."
        }
    
    try:
        debug(f"Starting HunyuanImage facial portrait generation", "hunyuan_facial_tool",
              image_path=image_path, style_preference=style_preference, aspect_ratio=aspect_ratio)
        
        # Step 1: Extract facial features
        info("Extracting facial features from input image", "hunyuan_facial_tool")
        
        facial_extractor = create_facial_extractor()
        if not facial_extractor:
            return {
                "success": False,
                "error": "Facial feature extractor not available"
            }
        
        facial_features = facial_extractor.extract_features(image_path)
        
        if not facial_features.get("success", False):
            return {
                "success": False,
                "error": f"Facial feature extraction failed: {facial_features.get('error', 'Unknown error')}"
            }
        
        debug("Facial features extracted successfully", "hunyuan_facial_tool")
        
        # Step 2: Generate prompt from facial features
        generative_manager = create_generative_manager()
        if not generative_manager:
            return {
                "success": False,
                "error": "Generative model manager not available"
            }
        
        # Generate base prompt from facial features
        base_prompt = generative_manager.prompt_generator.generate_prompt_from_attributes(
            facial_features.get("attributes", {}), 
            style_preference
        )
        
        # Add additional prompt if provided
        if additional_prompt:
            enhanced_prompt = f"{base_prompt}, {additional_prompt}"
        else:
            enhanced_prompt = base_prompt
        
        debug(f"Generated prompt from facial features: {enhanced_prompt[:100]}...", "hunyuan_facial_tool")
        
        # Step 3: Create HunyuanImage manager and generate
        hunyuan_manager = create_hunyuan_manager(use_local=use_local)
        
        if not hunyuan_manager:
            return {
                "success": False,
                "error": "Failed to create HunyuanImage manager. Check API key and dependencies."
            }
        
        # Generate portrait using HunyuanImage
        info("Generating portrait with HunyuanImage-2.1", "hunyuan_facial_tool")
        
        result = hunyuan_manager.generate_with_aspect_ratio(
            prompt=enhanced_prompt,
            aspect_ratio=aspect_ratio,
            use_refiner=use_refiner,
            use_prompt_enhancement=use_prompt_enhancement,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        if not result.success:
            return {
                "success": False,
                "error": result.error
            }
        
        # Step 4: Save generated portrait
        output_dir = Path("output/media/hunyuan/portraits")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        import time
        timestamp = int(time.time())
        filename = f"hunyuan_portrait_{timestamp}_{aspect_ratio.replace(':', '_')}.png"
        output_path = output_dir / filename
        
        # Save image
        result.image.save(output_path, quality=95, optimize=True)
        
        info(f"HunyuanImage facial portrait generated successfully: {output_path}", "hunyuan_facial_tool")
        
        # Prepare generation info
        generation_info = {
            "prompt": base_prompt,
            "enhanced_prompt": enhanced_prompt,
            "aspect_ratio": aspect_ratio,
            "width": result.metadata.get("width", 2048),
            "height": result.metadata.get("height", 2048),
            "style_preference": style_preference,
            "generation_time": result.metadata.get("generation_time", 0),
            "method": result.method
        }
        
        # Extract facial analysis for return
        facial_analysis = {
            "face_shape": facial_features.get("attributes", {}).get("face_shape", {}),
            "eye_features": facial_features.get("attributes", {}).get("eye_features", {}),
            "nose_features": facial_features.get("attributes", {}).get("nose_features", {}),
            "mouth_features": facial_features.get("attributes", {}).get("mouth_features", {})
        }
        
        return {
            "success": True,
            "image_path": str(output_path),
            "facial_analysis": facial_analysis,
            "generation_info": generation_info
        }
        
    except Exception as e:
        error(f"Error in HunyuanImage facial portrait generation: {str(e)}", "hunyuan_facial_tool", e)
        return {
            "success": False,
            "error": f"Error generating facial portrait: {str(e)}"
        }

def batch_generate_hunyuan_portraits(
    image_paths: list,
    style_preference: str = "professional",
    aspect_ratio: str = "1:1",
    use_refiner: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate multiple HunyuanImage portraits in batch.
    
    Args:
        image_paths (list): List of paths to facial images
        style_preference (str): Artistic style for all portraits
        aspect_ratio (str): Aspect ratio for all portraits
        use_refiner (bool): Enable refiner model
        **kwargs: Additional parameters for generation
        
    Returns:
        Dict[str, Any]: Batch generation results
    """
    
    if not HUNYUAN_FACIAL_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage facial components not available"
        }
    
    results = []
    successful = 0
    failed = 0
    
    info(f"Starting batch generation of {len(image_paths)} portraits", "hunyuan_facial_tool")
    
    for i, image_path in enumerate(image_paths):
        try:
            info(f"Processing image {i+1}/{len(image_paths)}: {image_path}", "hunyuan_facial_tool")
            
            result = generate_hunyuan_facial_portrait(
                image_path=image_path,
                style_preference=style_preference,
                aspect_ratio=aspect_ratio,
                use_refiner=use_refiner,
                **kwargs
            )
            
            results.append({
                "image_path": image_path,
                "result": result
            })
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            error(f"Error processing {image_path}: {str(e)}", "hunyuan_facial_tool", e)
            results.append({
                "image_path": image_path,
                "result": {
                    "success": False,
                    "error": str(e)
                }
            })
            failed += 1
    
    info(f"Batch generation completed: {successful} successful, {failed} failed", "hunyuan_facial_tool")
    
    return {
        "success": successful > 0,
        "total_images": len(image_paths),
        "successful": successful,
        "failed": failed,
        "results": results
    }
