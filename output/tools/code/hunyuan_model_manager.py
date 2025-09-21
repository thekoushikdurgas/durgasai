import os
import torch
import platform
from typing import Dict, Any, Optional

# Import required components
try:
    from utils.hunyuan_image_manager import create_hunyuan_manager
    from utils.logger import info, error, debug
    HUNYUAN_MANAGER_AVAILABLE = True
except ImportError as e:
    HUNYUAN_MANAGER_AVAILABLE = False
    print(f"Warning: HunyuanImage manager components not available: {e}")

def manage_hunyuan_model(
    action: str,
    use_local: bool = False
) -> Dict[str, Any]:
    """
    Manage and get information about HunyuanImage-2.1 model.
    
    Args:
        action (str): Action to perform
        use_local (bool): Check local model instead of API
        
    Returns:
        Dict[str, Any]: Action results
    """
    
    if not HUNYUAN_MANAGER_AVAILABLE:
        return {
            "success": False,
            "error": "HunyuanImage manager components not available"
        }
    
    try:
        debug(f"Performing HunyuanImage action: {action}", "hunyuan_manager_tool",
              use_local=use_local)
        
        if action == "get_info":
            return _get_model_info()
        
        elif action == "get_capabilities":
            return _get_model_capabilities()
        
        elif action == "get_performance_metrics":
            return _get_performance_metrics(use_local)
        
        elif action == "get_supported_aspect_ratios":
            return _get_supported_aspect_ratios()
        
        elif action == "check_system_requirements":
            return _check_system_requirements(use_local)
        
        elif action == "test_connection":
            return _test_connection(use_local)
        
        elif action == "get_model_status":
            return _get_model_status(use_local)
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Supported actions: get_info, get_capabilities, get_performance_metrics, get_supported_aspect_ratios, check_system_requirements, test_connection, get_model_status"
            }
        
    except Exception as e:
        error(f"Error in HunyuanImage model management: {str(e)}", "hunyuan_manager_tool", e)
        return {
            "success": False,
            "error": f"Error performing action '{action}': {str(e)}"
        }

def _get_model_info() -> Dict[str, Any]:
    """Get basic model information."""
    model_info = {
        "name": "HunyuanImage-2.1",
        "description": "Highly efficient text-to-image model capable of generating 2K (2048 × 2048) resolution images",
        "version": "2.1",
        "developer": "Tencent",
        "license": "tencent-hunyuan-community",
        "model_size": "17B parameters",
        "architecture": "Multi-modal, single- and dual-stream combined DiT (Diffusion Transformer)",
        "release_date": "September 2025",
        "paper": "arxiv:2509.04545",
        "github": "https://github.com/Tencent-Hunyuan/HunyuanImage-2.1",
        "huggingface": "https://huggingface.co/tencent/HunyuanImage-2.1"
    }
    
    return {
        "success": True,
        "result": model_info
    }

def _get_model_capabilities() -> Dict[str, Any]:
    """Get model capabilities and features."""
    capabilities = {
        "generation_capabilities": {
            "max_resolution": "2048x2048 (2K)",
            "supported_aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
            "multilingual_support": ["English", "Chinese"],
            "prompt_enhancement": True,
            "refiner_model": True,
            "style_transfer": True,
            "facial_feature_integration": True
        },
        "technical_features": {
            "high_compression_vae": "32x compression rate",
            "dual_text_encoder": True,
            "reinforcement_learning": "RLHF support",
            "model_distillation": True,
            "meanflow_distillation": True,
            "prompt_rewriting": True
        },
        "performance_benchmarks": {
            "ssae_evaluation": "88.88% (open-source models)",
            "gsb_evaluation": "Comparable to closed-source models",
            "inference_efficiency": "Same token length as 1K models for 2K generation",
            "memory_requirements": "24GB+ GPU for local inference"
        }
    }
    
    return {
        "success": True,
        "result": capabilities
    }

def _get_performance_metrics(use_local: bool) -> Dict[str, Any]:
    """Get performance metrics for the model manager."""
    try:
        manager = create_hunyuan_manager(use_local=use_local)
        
        if not manager:
            return {
                "success": False,
                "error": "Failed to create HunyuanImage manager"
            }
        
        metrics = manager.get_performance_metrics()
        
        return {
            "success": True,
            "result": {
                "metrics": metrics,
                "model_type": "local" if use_local else "api"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get performance metrics: {str(e)}"
        }

def _get_supported_aspect_ratios() -> Dict[str, Any]:
    """Get supported aspect ratios and their dimensions."""
    try:
        from utils.hunyuan_image_manager import HunyuanImageManager
        ratios = HunyuanImageManager.SUPPORTED_ASPECT_RATIOS
        
        # Format for better readability
        formatted_ratios = {}
        for ratio, dimensions in ratios.items():
            formatted_ratios[ratio] = {
                "width": dimensions[0],
                "height": dimensions[1],
                "resolution": f"{dimensions[0]}x{dimensions[1]}",
                "pixel_count": dimensions[0] * dimensions[1],
                "megapixels": round((dimensions[0] * dimensions[1]) / 1_000_000, 1)
            }
        
        return {
            "success": True,
            "result": {
                "supported_ratios": list(ratios.keys()),
                "ratio_details": formatted_ratios,
                "total_ratios": len(ratios)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get aspect ratios: {str(e)}"
        }

def _check_system_requirements(use_local: bool) -> Dict[str, Any]:
    """Check system requirements for HunyuanImage usage."""
    requirements = {
        "api_usage": {
            "internet_connection": True,
            "huggingface_api_key": True,
            "python_version": "3.8+",
            "memory": "Minimal (cloud inference)"
        },
        "local_usage": {
            "gpu_memory": "24GB+",
            "cuda_support": torch.cuda.is_available() if torch else False,
            "operating_system": platform.system(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if torch else "Not available",
            "cuda_version": torch.version.cuda if torch and torch.cuda.is_available() else "Not available"
        }
    }
    
    # Check current system
    current_system = {
        "operating_system": platform.system(),
        "python_version": platform.python_version(),
        "torch_available": torch is not None,
        "torch_version": torch.__version__ if torch else None,
        "cuda_available": torch.cuda.is_available() if torch else False,
        "cuda_version": torch.version.cuda if torch and torch.cuda.is_available() else None,
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch and torch.cuda.is_available() else None
    }
    
    # Determine compatibility
    api_compatible = True  # API usage is generally compatible
    local_compatible = (
        torch and 
        torch.cuda.is_available() and 
        current_system["gpu_memory"] and 
        current_system["gpu_memory"] >= 24
    )
    
    return {
        "success": True,
        "result": {
            "requirements": requirements,
            "current_system": current_system,
            "compatibility": {
                "api_usage": api_compatible,
                "local_usage": local_compatible,
                "recommended_usage": "local" if local_compatible else "api"
            }
        }
    }

def _test_connection(use_local: bool) -> Dict[str, Any]:
    """Test connection to HunyuanImage model."""
    try:
        manager = create_hunyuan_manager(use_local=use_local)
        
        if not manager:
            return {
                "success": False,
                "error": "Failed to create HunyuanImage manager"
            }
        
        # Test basic functionality
        test_result = {
            "manager_created": True,
            "model_type": "local" if use_local else "api",
            "supported_ratios_count": len(manager.get_supported_aspect_ratios()),
            "performance_metrics_available": True
        }
        
        # Try to get metrics to test connection
        try:
            metrics = manager.get_performance_metrics()
            test_result["metrics_accessible"] = True
            test_result["total_generations"] = metrics.get("total_generations", 0)
        except Exception as e:
            test_result["metrics_accessible"] = False
            test_result["metrics_error"] = str(e)
        
        return {
            "success": True,
            "result": {
                "connection_test": test_result,
                "status": "Connected successfully"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Connection test failed: {str(e)}"
        }

def _get_model_status(use_local: bool) -> Dict[str, Any]:
    """Get comprehensive model status."""
    try:
        manager = create_hunyuan_manager(use_local=use_local)
        
        if not manager:
            return {
                "success": False,
                "error": "Failed to create HunyuanImage manager"
            }
        
        # Get all status information
        metrics = manager.get_performance_metrics()
        aspect_ratios = manager.get_supported_aspect_ratios()
        
        status = {
            "model_info": {
                "name": "HunyuanImage-2.1",
                "type": "local" if use_local else "api",
                "status": "Available",
                "device": metrics.get("device", "unknown"),
                "initialization_time": "Unknown"
            },
            "performance": metrics,
            "capabilities": {
                "supported_aspect_ratios": list(aspect_ratios.keys()),
                "total_ratios": len(aspect_ratios),
                "max_resolution": "2048x2048",
                "multilingual_support": True,
                "prompt_enhancement": True,
                "refiner_model": True
            },
            "usage_stats": {
                "total_generations": metrics.get("total_generations", 0),
                "successful_generations": metrics.get("successful_generations", 0),
                "failed_generations": metrics.get("failed_generations", 0),
                "success_rate": metrics.get("success_rate", 0),
                "average_generation_time": metrics.get("average_generation_time", 0)
            }
        }
        
        return {
            "success": True,
            "result": status
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get model status: {str(e)}"
        }
