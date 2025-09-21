import os
import base64
import tempfile
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional

# Import HunyuanImage manager
try:
    from utils.hunyuan_image_manager import create_hunyuan_manager
    from utils.logger import info, error, debug
    HUNYUAN_AVAILABLE = True
except ImportError as e:
    HUNYUAN_AVAILABLE = False
    print(f"Warning: HunyuanImage components not available: {e}")

def generate_hunyuan_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    use_refiner: bool = True,
    use_prompt_enhancement: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    use_local: bool = False,
    facial_features: Optional[Dict[str, Any]] = None,
    style_preference: str = "professional"
) -> Dict[str, Any]:
    """
    Generate ultra-high-definition 2K image using HunyuanImage-2.1.
    
    Args:
        prompt (str): Text description of the image to generate
        aspect_ratio (str): Aspect ratio for the generated image
        use_refiner (bool): Enable refiner model for higher quality
        use_prompt_enhancement (bool): Automatically enhance prompts
        num_inference_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        seed (Optional[int]): Random seed for reproducible generation
        use_local (bool): Use local model instead of API
        facial_features (Optional[Dict[str, Any]]): Facial features to enhance prompt
        style_preference (str): Artistic style preference
        
    Returns:
        Dict[str, Any]: Generation results with image path and metadata
    """
    
    if not HUNYUAN_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage components not available. Please check installation."
        }
    
    try:
        debug(f"Starting HunyuanImage generation", "hunyuan_tool",
              prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
              aspect_ratio=aspect_ratio, use_local=use_local)
        
        # Create HunyuanImage manager
        hunyuan_manager = create_hunyuan_manager(use_local=use_local)
        
        if not hunyuan_manager:
            return {
                "success": False,
                "error": "Failed to create HunyuanImage manager. Check API key and dependencies."
            }
        
        # Enhance prompt with facial features if provided
        enhanced_prompt = prompt
        if facial_features:
            try:
                # Import generative model manager for prompt enhancement
                from utils.generative_model_manager import GenerativeModelManager
                gen_manager = GenerativeModelManager()
                enhanced_prompt = gen_manager._enhance_prompt_with_facial_features(
                    prompt, facial_features
                )
                debug(f"Enhanced prompt with facial features", "hunyuan_tool",
                      original_length=len(prompt), enhanced_length=len(enhanced_prompt))
            except Exception as e:
                error(f"Failed to enhance prompt with facial features: {str(e)}", "hunyuan_tool", e)
                # Continue with original prompt
        
        # Generate image
        info(f"Generating HunyuanImage with prompt: {prompt[:50]}...", "hunyuan_tool")
        
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
        
        # Save image to output directory
        output_dir = Path("output/media/hunyuan")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        import time
        timestamp = int(time.time())
        filename = f"hunyuan_{timestamp}_{aspect_ratio.replace(':', '_')}.png"
        output_path = output_dir / filename
        
        # Save image
        result.image.save(output_path, quality=95, optimize=True)
        
        info(f"HunyuanImage generated successfully: {output_path}", "hunyuan_tool")
        
        # Prepare generation info
        generation_info = {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt if enhanced_prompt != prompt else None,
            "aspect_ratio": aspect_ratio,
            "width": result.metadata.get("width", 2048),
            "height": result.metadata.get("height", 2048),
            "use_refiner": use_refiner,
            "use_prompt_enhancement": use_prompt_enhancement,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "method": result.method,
            "generation_time": result.metadata.get("generation_time", 0),
            "style_preference": style_preference
        }
        
        return {
            "success": True,
            "image_path": str(output_path),
            "generation_info": generation_info
        }
        
    except Exception as e:
        error(f"Error in HunyuanImage generation: {str(e)}", "hunyuan_tool", e)
        return {
            "success": False,
            "error": f"Error generating image: {str(e)}"
        }

def get_hunyuan_image_aspect_ratios() -> Dict[str, Any]:
    """
    Get supported aspect ratios for HunyuanImage-2.1.
    
    Returns:
        Dict[str, Any]: Supported aspect ratios and their dimensions
    """
    if not HUNYUAN_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage components not available"
        }
    
    try:
        from utils.hunyuan_image_manager import HunyuanImageManager
        ratios = HunyuanImageManager.SUPPORTED_ASPECT_RATIOS
        
        return {
            "success": True,
            "aspect_ratios": ratios,
            "supported_ratios": list(ratios.keys())
        }
        
    except Exception as e:
        error(f"Error getting aspect ratios: {str(e)}", "hunyuan_tool", e)
        return {
            "success": False,
            "error": f"Error getting aspect ratios: {str(e)}"
        }

def get_hunyuan_image_info() -> Dict[str, Any]:
    """
    Get information about HunyuanImage-2.1 model.
    
    Returns:
        Dict[str, Any]: Model information and capabilities
    """
    return {
        "success": True,
        "model_info": {
            "name": "HunyuanImage-2.1",
            "description": "Highly efficient text-to-image model capable of generating 2K (2048 × 2048) resolution images",
            "features": [
                "2K high-resolution generation",
                "Multilingual support (English and Chinese)",
                "Multiple aspect ratios",
                "Prompt enhancement",
                "Refiner model",
                "Advanced DiT architecture"
            ],
            "supported_resolutions": [
                "2048x2048 (1:1)",
                "2560x1536 (16:9)",
                "1536x2560 (9:16)",
                "2304x1792 (4:3)",
                "1792x2304 (3:4)",
                "2048x1365 (3:2)",
                "1365x2048 (2:3)"
            ],
            "system_requirements": {
                "api_usage": "HuggingFace API key, internet connection",
                "local_usage": "24GB+ GPU memory, CUDA support, Linux OS"
            },
            "generation_params": {
                "default_steps": 50,
                "distilled_steps": 8,
                "default_guidance_scale": 3.5,
                "distilled_guidance_scale": 3.25
            }
        }
    }
