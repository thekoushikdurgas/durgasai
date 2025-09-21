import os
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

# Import required components
try:
    from utils.hunyuan_image_manager import create_hunyuan_manager
    from utils.logger import info, error, debug
    HUNYUAN_BATCH_AVAILABLE = True
except ImportError as e:
    HUNYUAN_BATCH_AVAILABLE = False
    print(f"Warning: HunyuanImage batch components not available: {e}")

def generate_hunyuan_batch(
    prompts: List[str],
    aspect_ratio: str = "1:1",
    use_refiner: bool = True,
    use_prompt_enhancement: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    use_local: bool = False,
    parallel_processing: bool = False,
    output_format: str = "png",
    create_zip: bool = False
) -> Dict[str, Any]:
    """
    Generate multiple high-resolution 2K images in batch using HunyuanImage-2.1.
    
    Args:
        prompts (List[str]): List of text prompts for image generation
        aspect_ratio (str): Aspect ratio for all generated images
        use_refiner (bool): Enable refiner model for higher quality
        use_prompt_enhancement (bool): Automatically enhance prompts
        num_inference_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for generation
        seed (Optional[int]): Random seed for reproducible generation
        use_local (bool): Use local model instead of API
        parallel_processing (bool): Process multiple images in parallel (experimental)
        output_format (str): Output format for generated images
        create_zip (bool): Create a ZIP file containing all generated images
        
    Returns:
        Dict[str, Any]: Batch generation results
    """
    
    if not HUNYUAN_BATCH_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage batch components not available. Please check installation."
        }
    
    if not prompts or len(prompts) == 0:
        return {
            "success": False,
            "error": "No prompts provided for batch generation"
        }
    
    start_time = time.time()
    
    try:
        debug(f"Starting HunyuanImage batch generation", "hunyuan_batch_tool",
              prompt_count=len(prompts), aspect_ratio=aspect_ratio, use_local=use_local)
        
        # Create output directory
        output_dir = Path("output/media/hunyuan/batch")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory for this batch
        timestamp = int(time.time())
        batch_dir = output_dir / f"batch_{timestamp}"
        batch_dir.mkdir(exist_ok=True)
        
        # Create HunyuanImage manager
        hunyuan_manager = create_hunyuan_manager(use_local=use_local)
        
        if not hunyuan_manager:
            return {
                "success": False,
                "error": "Failed to create HunyuanImage manager. Check API key and dependencies."
            }
        
        info(f"Processing {len(prompts)} prompts for batch generation", "hunyuan_batch_tool")
        
        # Process images
        generated_images = []
        failed_images = []
        
        for i, prompt in enumerate(prompts):
            try:
                info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...", "hunyuan_batch_tool")
                
                prompt_start_time = time.time()
                
                # Generate image
                result = hunyuan_manager.generate_with_aspect_ratio(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    use_refiner=use_refiner,
                    use_prompt_enhancement=use_prompt_enhancement,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                if result.success:
                    # Save image
                    filename = f"hunyuan_batch_{timestamp}_{i+1:03d}.{output_format}"
                    output_path = batch_dir / filename
                    
                    # Convert and save with specified format
                    if output_format.lower() in ['jpg', 'jpeg']:
                        # Convert to RGB if saving as JPEG
                        if result.image.mode != 'RGB':
                            result.image = result.image.convert('RGB')
                        result.image.save(output_path, format='JPEG', quality=95, optimize=True)
                    else:
                        result.image.save(output_path, format='PNG', optimize=True)
                    
                    prompt_generation_time = time.time() - prompt_start_time
                    
                    generated_images.append({
                        "prompt": prompt,
                        "image_path": str(output_path),
                        "generation_time": prompt_generation_time,
                        "index": i + 1
                    })
                    
                    info(f"Successfully generated image {i+1}/{len(prompts)}: {output_path}", "hunyuan_batch_tool")
                else:
                    failed_images.append({
                        "prompt": prompt,
                        "error": result.error,
                        "index": i + 1
                    })
                    
                    error(f"Failed to generate image {i+1}/{len(prompts)}: {result.error}", "hunyuan_batch_tool")
                
            except Exception as e:
                failed_images.append({
                    "prompt": prompt,
                    "error": str(e),
                    "index": i + 1
                })
                
                error(f"Error processing prompt {i+1}/{len(prompts)}: {str(e)}", "hunyuan_batch_tool", e)
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        successful_count = len(generated_images)
        failed_count = len(failed_images)
        
        # Create ZIP file if requested
        zip_file_path = None
        if create_zip and successful_count > 0:
            try:
                zip_filename = f"hunyuan_batch_{timestamp}.zip"
                zip_file_path = output_dir / zip_filename
                
                with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for img_info in generated_images:
                        img_path = Path(img_info["image_path"])
                        zipf.write(img_path, img_path.name)
                
                info(f"Created ZIP file: {zip_file_path}", "hunyuan_batch_tool")
                
            except Exception as e:
                error(f"Failed to create ZIP file: {str(e)}", "hunyuan_batch_tool", e)
        
        # Prepare batch info
        batch_info = {
            "total_time": total_time,
            "average_time_per_image": total_time / len(prompts) if prompts else 0,
            "aspect_ratio": aspect_ratio,
            "use_refiner": use_refiner,
            "zip_file_path": str(zip_file_path) if zip_file_path else None
        }
        
        info(f"Batch generation completed: {successful_count} successful, {failed_count} failed in {total_time:.2f}s", 
             "hunyuan_batch_tool")
        
        return {
            "success": successful_count > 0,
            "total_prompts": len(prompts),
            "successful_generations": successful_count,
            "failed_generations": failed_count,
            "generated_images": generated_images,
            "failed_images": failed_images,
            "batch_info": batch_info
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        error(f"Error in HunyuanImage batch generation: {str(e)}", "hunyuan_batch_tool", e)
        
        return {
            "success": False,
            "total_prompts": len(prompts),
            "successful_generations": 0,
            "failed_generations": len(prompts),
            "generated_images": [],
            "failed_images": [{"prompt": prompt, "error": str(e), "index": i+1} for i, prompt in enumerate(prompts)],
            "batch_info": {
                "total_time": total_time,
                "average_time_per_image": 0,
                "aspect_ratio": aspect_ratio,
                "use_refiner": use_refiner,
                "zip_file_path": None
            },
            "error": f"Batch generation failed: {str(e)}"
        }

def get_batch_generation_status() -> Dict[str, Any]:
    """
    Get status of batch generation capabilities.
    
    Returns:
        Dict[str, Any]: Batch generation status and capabilities
    """
    
    if not HUNYUAN_BATCH_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage batch components not available"
        }
    
    try:
        # Check if manager can be created
        manager = create_hunyuan_manager(use_local=False)
        
        capabilities = {
            "batch_processing": True,
            "parallel_processing": False,  # Not implemented yet
            "supported_formats": ["png", "jpg", "jpeg"],
            "zip_creation": True,
            "max_batch_size": 50,  # Reasonable limit
            "supported_aspect_ratios": [
                "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
            ],
            "features": [
                "High-resolution 2K generation",
                "Batch processing",
                "Multiple output formats",
                "ZIP file creation",
                "Progress tracking",
                "Error handling"
            ]
        }
        
        return {
            "success": True,
            "result": capabilities
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get batch generation status: {str(e)}"
        }

def cleanup_batch_files(older_than_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up old batch generation files.
    
    Args:
        older_than_hours (int): Remove files older than this many hours
        
    Returns:
        Dict[str, Any]: Cleanup results
    """
    
    try:
        batch_dir = Path("output/media/hunyuan/batch")
        
        if not batch_dir.exists():
            return {
                "success": True,
                "result": {
                    "cleaned_files": 0,
                    "freed_space": 0,
                    "message": "No batch directory found"
                }
            }
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        cleaned_files = 0
        freed_space = 0
        
        # Remove old batch directories
        for item in batch_dir.iterdir():
            if item.is_dir() and item.stat().st_mtime < cutoff_time:
                # Calculate size before deletion
                for file in item.rglob('*'):
                    if file.is_file():
                        freed_space += file.stat().st_size
                        cleaned_files += 1
                
                # Remove directory
                import shutil
                shutil.rmtree(item)
                info(f"Removed old batch directory: {item}", "hunyuan_batch_tool")
        
        return {
            "success": True,
            "result": {
                "cleaned_files": cleaned_files,
                "freed_space": freed_space,
                "freed_space_mb": round(freed_space / 1024 / 1024, 2),
                "message": f"Cleaned {cleaned_files} files older than {older_than_hours} hours"
            }
        }
        
    except Exception as e:
        error(f"Error cleaning up batch files: {str(e)}", "hunyuan_batch_tool", e)
        return {
            "success": False,
            "error": f"Failed to clean up batch files: {str(e)}"
        }
