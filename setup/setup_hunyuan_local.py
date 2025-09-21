#!/usr/bin/env python3
"""
HunyuanImage-2.1 Local Setup Script
This script sets up and tests HunyuanImage-2.1 for local execution.
"""

import os
import sys
import subprocess
import torch
import time
from pathlib import Path

def check_system_requirements():
    """Check if system meets requirements for local execution"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("❌ CUDA not available - GPU required")
        return False
    
    print(f"✅ CUDA {torch.version.cuda} available")
    
    # Check GPU memory
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 24:
            print(f"⚠️  Warning: GPU {i} has less than 24GB VRAM")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n🚀 Installing dependencies...")
    
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "transformers>=4.35.0",
        "diffusers>=0.21.0",
        "accelerate>=0.24.0",
        "xformers>=0.0.20",
        "hyimage>=0.0.2",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "omegaconf>=2.3.0",
        "safetensors>=0.3.0",
        "einops>=0.6.0",
        "loguru>=0.7.0",
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_hunyuan_loading():
    """Test loading HunyuanImage-2.1 model"""
    print("\n🧪 Testing HunyuanImage-2.1 loading...")
    
    try:
        from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
        
        # Set optimization environment variables using centralized config
        try:
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from utils.config import Config
            
            # Setup environment variables from configuration
            Config.setup_environment_variables()
            
            # Ensure PyTorch specific variables are set
            Config.update_environment_variable('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True', save_to_config=False)
            Config.update_environment_variable('TOKENIZERS_PARALLELISM', 'false', save_to_config=False)
            
            print("✅ Environment variables configured from centralized config")
            
        except Exception as e:
            print(f"⚠️ Could not load centralized config: {e}")
            print("🔧 Using fallback environment variable setup")
            
            # Fallback to direct setting
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("Loading HunyuanImage-2.1 (this may take several minutes)...")
        start_time = time.time()
        
        # Load model with optimizations
        pipe = HunyuanImagePipeline.from_pretrained(
            model_name="hunyuanimage-v2.1",
            use_fp8=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipe = pipe.to("cuda")
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        return pipe
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"❌ Loading error: {e}")
        return None

def test_image_generation(pipe):
    """Test image generation"""
    print("\n🎨 Testing image generation...")
    
    try:
        prompt = "A beautiful mountain landscape at sunset, photorealistic, high quality"
        
        print(f"Generating image with prompt: {prompt}")
        start_time = time.time()
        
        # Generate image
        with torch.autocast("cuda"):
            result = pipe(
                prompt=prompt,
                width=1024,  # Start with lower resolution for testing
                height=1024,
                use_reprompt=False,
                use_refiner=False,  # Disable refiner for faster test
                num_inference_steps=25,
                guidance_scale=3.5,
                seed=42,
            )
        
        generation_time = time.time() - start_time
        
        # Save image
        image = result.images[0]
        output_path = Path("test_generation.png")
        image.save(output_path)
        
        print(f"✅ Image generated successfully!")
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        print(f"💾 Image saved as: {output_path.absolute()}")
        print(f"📏 Image size: {image.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 HunyuanImage-2.1 Local Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please check your setup.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        return False
    
    # Test model loading
    pipe = test_hunyuan_loading()
    if pipe is None:
        print("\n❌ Failed to load model.")
        return False
    
    # Test image generation
    if not test_image_generation(pipe):
        print("\n❌ Failed to generate image.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("HunyuanImage-2.1 is ready for local use.")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
