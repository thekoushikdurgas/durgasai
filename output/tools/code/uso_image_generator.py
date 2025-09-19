import os
import torch
from PIL import Image
from utils.uso_service import get_uso_service
from utils.logging_utils import log_info, log_error

def generate_uso_image(
    prompt: str,
    generation_mode: str = "Text-to-image",
    width: int = 1024,
    height: int = 1024,
    guidance: float = 4.0,
    num_steps: int = 25,
    seed: int = -1,
    content_image_path: str = None,
    style_image_path: str = None,
    extra_style_image_path: str = None
) -> dict:
    """Generate image using USO pipeline"""
    
    try:
        # Get USO service
        uso_service = get_uso_service()
        
        # Check if pipeline is initialized
        if not hasattr(uso_service, 'pipeline') or not uso_service.pipeline:
            # Try to initialize with default settings
            log_info("Initializing USO pipeline for tool usage", "uso_tool")
            success = uso_service.initialize_pipeline(
                model_type="flux-dev",
                offload=True,  # Use offloading for memory efficiency
                only_lora=True,
                lora_rank=128
            )
            if not success:
                return {
                    "success": False,
                    "error": "Failed to initialize USO pipeline. Please initialize it manually in the USO page first."
                }
        
        # Load reference images if provided
        content_image = None
        style_image = None
        extra_style_image = None
        
        if content_image_path and os.path.exists(content_image_path):
            try:
                content_image = Image.open(content_image_path)
                log_info(f"Loaded content image: {content_image_path}", "uso_tool")
            except Exception as e:
                log_error(f"Error loading content image: {e}", "uso_tool")
        
        if style_image_path and os.path.exists(style_image_path):
            try:
                style_image = Image.open(style_image_path)
                log_info(f"Loaded style image: {style_image_path}", "uso_tool")
            except Exception as e:
                log_error(f"Error loading style image: {e}", "uso_tool")
        
        if extra_style_image_path and os.path.exists(extra_style_image_path):
            try:
                extra_style_image = Image.open(extra_style_image_path)
                log_info(f"Loaded extra style image: {extra_style_image_path}", "uso_tool")
            except Exception as e:
                log_error(f"Error loading extra style image: {e}", "uso_tool")
        
        # Generate image
        log_info(f"Generating USO image with prompt: {prompt[:50]}...", "uso_tool")
        
        result_image, filename = uso_service.generate_image(
            prompt=prompt,
            content_image=content_image,
            style_image=style_image,
            extra_style_image=extra_style_image,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            keep_size=False,
            content_long_size=512
        )
        
        if result_image and filename:
            log_info(f"USO image generated successfully: {filename}", "uso_tool")
            
            return {
                "success": True,
                "image_path": filename,
                "generation_info": {
                    "prompt": prompt,
                    "mode": generation_mode,
                    "seed": seed if seed != -1 else "random",
                    "width": width,
                    "height": height,
                    "guidance": guidance,
                    "num_steps": num_steps
                }
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate image. Check logs for details."
            }
    
    except Exception as e:
        log_error(f"Error in USO image generation: {e}", "uso_tool", e)
        return {
            "success": False,
            "error": f"Error generating image: {str(e)}"
        }