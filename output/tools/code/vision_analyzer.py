"""
DurgasAI Vision Analysis Tool
Analyzes images using Hugging Face vision models with comprehensive error handling
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Import vision model manager
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.vision_model_manager import create_vision_manager, VisionModelManager
from utils.logging_utils import log_info, log_error


def analyze_image_with_vision_model(
    image_path: str,
    analysis_prompt: str = "Describe this image in detail",
    model_name: str = "HuggingFaceM4/idefics2-8b",
    api_token: str = None
) -> Dict[str, Any]:
    """
    Analyze an image using Hugging Face vision models.
    
    Args:
        image_path (str): Path to the image file to analyze
        analysis_prompt (str): Text prompt describing what to analyze
        model_name (str): HuggingFace model ID to use for analysis
        api_token (str): HuggingFace API token (optional, will try to get from config)
    
    Returns:
        Dict containing success status, analysis results, and metadata
    """
    try:
        log_info(f"Starting vision analysis for image: {image_path}", "vision_tool")
        
        # Validate image file
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }
        
        # Create vision manager
        vision_manager = create_vision_manager(api_token)
        if not vision_manager:
            return {
                "success": False,
                "error": "Failed to initialize vision model manager. Please check your API token."
            }
        
        start_time = time.time()
        
        # Analyze image
        result = vision_manager.analyze_image(
            image_path=image_path,
            text_prompt=analysis_prompt,
            model_id=model_name
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            log_info(f"Vision analysis completed successfully in {processing_time:.2f}s", "vision_tool")
            
            return {
                "success": True,
                "analysis": result.content,
                "model_used": result.model_used,
                "processing_time": processing_time,
                "image_processed": result.image_processed,
                "metadata": result.metadata
            }
        else:
            log_error(f"Vision analysis failed: {result.error}", "vision_tool")
            
            return {
                "success": False,
                "error": result.error,
                "model_used": result.model_used,
                "processing_time": processing_time
            }
            
    except Exception as e:
        log_error(f"Error in vision analysis: {str(e)}", "vision_tool")
        
        return {
            "success": False,
            "error": f"Vision analysis failed: {str(e)}"
        }


def generate_image_caption(
    image_path: str,
    model_name: str = "Salesforce/blip-image-captioning-base",
    api_token: str = None
) -> Dict[str, Any]:
    """
    Generate a caption for an image using specialized captioning models.
    
    Args:
        image_path (str): Path to the image file
        model_name (str): HuggingFace model ID for captioning
        api_token (str): HuggingFace API token (optional)
    
    Returns:
        Dict containing success status, caption, and metadata
    """
    try:
        log_info(f"Generating caption for image: {image_path}", "vision_tool")
        
        # Validate image file
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }
        
        # Create vision manager
        vision_manager = create_vision_manager(api_token)
        if not vision_manager:
            return {
                "success": False,
                "error": "Failed to initialize vision model manager. Please check your API token."
            }
        
        start_time = time.time()
        
        # Generate caption
        result = vision_manager.generate_caption(
            image_path=image_path,
            model_id=model_name
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            log_info(f"Caption generated successfully in {processing_time:.2f}s", "vision_tool")
            
            return {
                "success": True,
                "caption": result.content,
                "model_used": result.model_used,
                "processing_time": processing_time,
                "image_processed": result.image_processed,
                "metadata": result.metadata
            }
        else:
            log_error(f"Caption generation failed: {result.error}", "vision_tool")
            
            return {
                "success": False,
                "error": result.error,
                "model_used": result.model_used,
                "processing_time": processing_time
            }
            
    except Exception as e:
        log_error(f"Error in caption generation: {str(e)}", "vision_tool")
        
        return {
            "success": False,
            "error": f"Caption generation failed: {str(e)}"
        }


def answer_visual_question(
    image_path: str,
    question: str,
    model_name: str = "Salesforce/blip-vqa-base",
    api_token: str = None
) -> Dict[str, Any]:
    """
    Answer a question about an image using visual question answering models.
    
    Args:
        image_path (str): Path to the image file
        question (str): Question to ask about the image
        model_name (str): HuggingFace model ID for VQA
        api_token (str): HuggingFace API token (optional)
    
    Returns:
        Dict containing success status, answer, and metadata
    """
    try:
        log_info(f"Answering visual question: {question[:50]}...", "vision_tool")
        
        # Validate image file
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }
        
        # Create vision manager
        vision_manager = create_vision_manager(api_token)
        if not vision_manager:
            return {
                "success": False,
                "error": "Failed to initialize vision model manager. Please check your API token."
            }
        
        start_time = time.time()
        
        # Answer visual question
        result = vision_manager.visual_question_answering(
            image_path=image_path,
            question=question,
            model_id=model_name
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            log_info(f"Visual question answered successfully in {processing_time:.2f}s", "vision_tool")
            
            return {
                "success": True,
                "answer": result.content,
                "question": question,
                "model_used": result.model_used,
                "processing_time": processing_time,
                "image_processed": result.image_processed,
                "metadata": result.metadata
            }
        else:
            log_error(f"Visual question answering failed: {result.error}", "vision_tool")
            
            return {
                "success": False,
                "error": result.error,
                "question": question,
                "model_used": result.model_used,
                "processing_time": processing_time
            }
            
    except Exception as e:
        log_error(f"Error in visual question answering: {str(e)}", "vision_tool")
        
        return {
            "success": False,
            "error": f"Visual question answering failed: {str(e)}"
        }


def get_vision_model_performance(api_token: str = None) -> Dict[str, Any]:
    """
    Get performance metrics for the vision model manager.
    
    Args:
        api_token (str): HuggingFace API token (optional)
    
    Returns:
        Dict containing performance metrics
    """
    try:
        vision_manager = create_vision_manager(api_token)
        if not vision_manager:
            return {
                "success": False,
                "error": "Failed to initialize vision model manager"
            }
        
        metrics = vision_manager.get_performance_metrics()
        
        return {
            "success": True,
            "performance_metrics": metrics
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get performance metrics: {str(e)}"
        }
