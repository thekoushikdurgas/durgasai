"""
DurgasAI Image Outpainting Tool
Extends images intelligently using Stable Diffusion XL with ControlNet
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import base64
import io

from utils.outpaint_service import get_outpaint_service
from utils.logging_utils import log_info, log_error
from models.outpaint_models import OutpaintRequest, OutpaintResult


def outpaint_image(
    image_path: str,
    prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    alignment: str = "Middle",
    num_inference_steps: int = 8,
    overlap_percentage: int = 10,
    resize_option: str = "Full",
    custom_resize_percentage: int = 50,
    overlap_left: bool = True,
    overlap_right: bool = True,
    overlap_top: bool = True,
    overlap_bottom: bool = True
) -> Dict[str, Any]:
    """
    Outpaint an image to extend its boundaries with AI-generated content.
    
    Args:
        image_path (str): Path to the input image file
        prompt (str): Description of content to generate in extended areas
        width (int): Target width (720-1536, divisible by 8)
        height (int): Target height (720-1536, divisible by 8)
        alignment (str): Image alignment - "Middle", "Left", "Right", "Top", "Bottom"
        num_inference_steps (int): Number of generation steps (4-12)
        overlap_percentage (int): Overlap between original and generated areas (1-50%)
        resize_option (str): How to resize input - "Full", "50%", "33%", "25%", "Custom"
        custom_resize_percentage (int): Custom resize percentage if resize_option is "Custom"
        overlap_left (bool): Allow overlap on left edge
        overlap_right (bool): Allow overlap on right edge
        overlap_top (bool): Allow overlap on top edge
        overlap_bottom (bool): Allow overlap on bottom edge
    
    Returns:
        Dict containing success status, output paths, and metadata
    """
    try:
        log_info(f"Starting outpaint for image: {image_path}", "outpaint_tool")
        
        # Validate input image
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image file not found: {image_path}"}
        
        # Load image
        try:
            input_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)}"}
        
        # Validate dimensions
        if width % 8 != 0 or height % 8 != 0:
            return {"success": False, "error": "Width and height must be divisible by 8"}
        
        if not (720 <= width <= 1536) or not (720 <= height <= 1536):
            return {"success": False, "error": "Dimensions must be between 720 and 1536"}
        
        # Get outpaint service
        outpaint_service = get_outpaint_service()
        
        # Initialize pipeline if not already done
        if not outpaint_service.model_loaded:
            log_info("Initializing outpaint pipeline...", "outpaint_tool")
            success = outpaint_service.initialize_pipeline()
            if not success:
                return {"success": False, "error": "Failed to initialize outpaint pipeline"}
        
        start_time = time.time()
        
        # Generate outpainted image
        final_control_image = None
        final_result_image = None
        
        for control_image, result_image in outpaint_service.outpaint_image(
            input_image, width, height, overlap_percentage, num_inference_steps,
            resize_option, custom_resize_percentage, prompt, alignment,
            overlap_left, overlap_right, overlap_top, overlap_bottom
        ):
            final_control_image = control_image
            final_result_image = result_image
        
        processing_time = time.time() - start_time
        
        # Save results
        output_dir = Path("output/media/outpaint")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        control_path = output_dir / f"control_{timestamp}.png"
        result_path = output_dir / f"result_{timestamp}.png"
        
        final_control_image.save(control_path, format="PNG")
        final_result_image.save(result_path, format="PNG")
        
        log_info(f"Outpaint completed in {processing_time:.1f}s", "outpaint_tool")
        
        return {
            "success": True,
            "control_image_path": str(control_path),
            "result_image_path": str(result_path),
            "processing_time": processing_time,
            "settings": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "alignment": alignment,
                "num_inference_steps": num_inference_steps,
                "overlap_percentage": overlap_percentage,
                "resize_option": resize_option,
                "custom_resize_percentage": custom_resize_percentage,
                "overlap_edges": {
                    "left": overlap_left,
                    "right": overlap_right,
                    "top": overlap_top,
                    "bottom": overlap_bottom
                }
            },
            "input_image_info": {
                "original_size": f"{input_image.width}x{input_image.height}",
                "output_size": f"{width}x{height}"
            }
        }
        
    except Exception as e:
        log_error(f"Outpaint tool error: {str(e)}", "outpaint_tool")
        return {"success": False, "error": str(e)}


def preview_outpaint_mask(
    image_path: str,
    width: int = 1024,
    height: int = 1024,
    alignment: str = "Middle",
    overlap_percentage: int = 10,
    resize_option: str = "Full",
    custom_resize_percentage: int = 50,
    overlap_left: bool = True,
    overlap_right: bool = True,
    overlap_top: bool = True,
    overlap_bottom: bool = True
) -> Dict[str, Any]:
    """
    Preview the outpainting mask and alignment without generating content.
    
    Args:
        image_path (str): Path to the input image file
        width (int): Target width
        height (int): Target height
        alignment (str): Image alignment
        overlap_percentage (int): Overlap percentage
        resize_option (str): Resize option
        custom_resize_percentage (int): Custom resize percentage
        overlap_left/right/top/bottom (bool): Edge overlap settings
    
    Returns:
        Dict containing preview image path and settings
    """
    try:
        # Validate input image
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image file not found: {image_path}"}
        
        # Load image
        input_image = Image.open(image_path).convert('RGB')
        
        # Get outpaint service
        outpaint_service = get_outpaint_service()
        
        # Generate preview
        preview_image = outpaint_service.preview_image_and_mask(
            input_image, width, height, overlap_percentage,
            resize_option, custom_resize_percentage, alignment,
            overlap_left, overlap_right, overlap_top, overlap_bottom
        )
        
        # Save preview
        output_dir = Path("output/media/outpaint")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        preview_path = output_dir / f"preview_{timestamp}.png"
        preview_image.save(preview_path, format="PNG")
        
        return {
            "success": True,
            "preview_image_path": str(preview_path),
            "settings": {
                "width": width,
                "height": height,
                "alignment": alignment,
                "overlap_percentage": overlap_percentage,
                "resize_option": resize_option,
                "custom_resize_percentage": custom_resize_percentage
            }
        }
        
    except Exception as e:
        log_error(f"Preview tool error: {str(e)}", "outpaint_tool")
        return {"success": False, "error": str(e)}


# Tool metadata for DurgasAI tool manager
TOOL_METADATA = {
    "name": "outpaint_image",
    "description": "Extend images intelligently using AI outpainting",
    "category": "image_generation",
    "version": "1.0.0",
    "author": "DurgasAI Team",
    "parameters": {
        "image_path": {
            "type": "string",
            "description": "Path to input image file",
            "required": True
        },
        "prompt": {
            "type": "string", 
            "description": "Description of content to generate",
            "required": False,
            "default": ""
        },
        "width": {
            "type": "integer",
            "description": "Target width (720-1536, divisible by 8)",
            "required": False,
            "default": 1024,
            "minimum": 720,
            "maximum": 1536
        },
        "height": {
            "type": "integer",
            "description": "Target height (720-1536, divisible by 8)",
            "required": False,
            "default": 1024,
            "minimum": 720,
            "maximum": 1536
        },
        "alignment": {
            "type": "string",
            "description": "Image alignment within canvas",
            "required": False,
            "default": "Middle",
            "enum": ["Middle", "Left", "Right", "Top", "Bottom"]
        },
        "num_inference_steps": {
            "type": "integer",
            "description": "Number of generation steps",
            "required": False,
            "default": 8,
            "minimum": 4,
            "maximum": 12
        }
    },
    "examples": [
        {
            "description": "Basic outpainting to 16:9 aspect ratio",
            "parameters": {
                "image_path": "input.jpg",
                "prompt": "beautiful landscape, mountains, clear sky",
                "width": 1280,
                "height": 720,
                "alignment": "Middle"
            }
        },
        {
            "description": "Portrait extension with custom prompt",
            "parameters": {
                "image_path": "portrait.jpg", 
                "prompt": "professional studio background, soft lighting",
                "width": 1024,
                "height": 1024,
                "alignment": "Bottom",
                "num_inference_steps": 10
            }
        }
    ]
}
