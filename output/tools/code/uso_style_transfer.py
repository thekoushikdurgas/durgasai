import os
from PIL import Image
from utils.uso_service import get_uso_service
from utils.logging_utils import log_info, log_error

def apply_uso_style(
    prompt: str,
    style_description: str = "artistic style",
    style_image_path: str = None,
    width: int = 1024,
    height: int = 1024,
    style_strength: float = 4.0
) -> dict:
    """Apply artistic style using USO pipeline"""
    
    try:
        uso_service = get_uso_service()
        
        # Check if pipeline is initialized
        if not hasattr(uso_service, 'pipeline') or not uso_service.pipeline:
            log_info("Initializing USO pipeline for style transfer", "uso_style")
            success = uso_service.initialize_pipeline(
                model_type="flux-dev",
                offload=True,
                only_lora=True,
                lora_rank=128
            )
            if not success:
                return {
                    "success": False,
                    "error": "Failed to initialize USO pipeline"
                }
        
        # Load style image if provided
        style_image = None
        if style_image_path and os.path.exists(style_image_path):
            try:
                style_image = Image.open(style_image_path)
                log_info(f"Loaded style reference: {style_image_path}", "uso_style")
            except Exception as e:
                log_error(f"Error loading style image: {e}", "uso_style")
        
        # Enhance prompt with style description
        styled_prompt = f"{prompt}, {style_description}"
        
        log_info(f"Applying style transfer: {styled_prompt[:50]}...", "uso_style")
        
        # Generate styled image
        result_image, filename = uso_service.generate_image(
            prompt=styled_prompt,
            content_image=None,
            style_image=style_image,
            extra_style_image=None,
            width=width,
            height=height,
            guidance=style_strength,
            num_steps=25,
            seed=-1,
            keep_size=False,
            content_long_size=512
        )
        
        if result_image and filename:
            log_info(f"Style transfer completed: {filename}", "uso_style")
            
            return {
                "success": True,
                "image_path": filename,
                "style_info": {
                    "prompt": prompt,
                    "style_description": style_description,
                    "style_strength": style_strength
                }
            }
        else:
            return {
                "success": False,
                "error": "Failed to apply style transfer"
            }
    
    except Exception as e:
        log_error(f"Error in style transfer: {e}", "uso_style", e)
        return {
            "success": False,
            "error": f"Style transfer failed: {str(e)}"
        }