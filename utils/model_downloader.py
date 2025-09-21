"""
Model Downloader Utility for DurgasAI.

This module provides comprehensive functionality for downloading, managing,
and configuring Hugging Face models locally with system requirements validation.
"""

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
import streamlit as st
from datetime import datetime

# Import logging
try:
    from .logger import debug, info, warning, error, log_user_action
except ImportError:
    def debug(msg, component="model_downloader", **kwargs): pass
    def info(msg, component="model_downloader", **kwargs): pass
    def warning(msg, component="model_downloader", **kwargs): pass
    def error(msg, component="model_downloader", **kwargs): pass
    def log_user_action(action, **kwargs): pass


class ModelDownloader:
    """Handles downloading and managing Hugging Face models locally."""
    
    def __init__(self, base_dir: str = "output/downloaded_models"):
        """
        Initialize the model downloader.
        
        Args:
            base_dir (str): Base directory for storing downloaded models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        self.scripts_dir = self.base_dir / "scripts"
        self.configs_dir = self.base_dir / "configs"
        
        for dir_path in [self.models_dir, self.cache_dir, self.scripts_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        info(f"Model downloader initialized with base directory: {self.base_dir}")
    
    def check_system_requirements(self, model_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if the system meets the requirements for a model.
        
        Args:
            model_info (Dict[str, Any]): Model information including system requirements
            
        Returns:
            Tuple[bool, List[str]]: (meets_requirements, warnings_list)
        """
        warnings = []
        meets_requirements = True
        
        reqs = model_info.get('system_requirements', {})
        
        # Check GPU memory
        if reqs.get('gpu_memory'):
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    required_gpu = float(reqs['gpu_memory'].replace('GB+', '').replace('GB', ''))
                    
                    if gpu_memory < required_gpu:
                        warnings.append(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for {reqs['gpu_memory']}")
                        meets_requirements = False
                else:
                    if reqs.get('cuda', False):
                        warnings.append("CUDA is required but not available")
                        meets_requirements = False
            except ImportError:
                warnings.append("PyTorch not installed - cannot check GPU requirements")
        
        # Check available disk space
        if reqs.get('storage'):
            try:
                required_storage = float(reqs['storage'].replace('GB', ''))
                available_space = shutil.disk_usage(self.base_dir).free / (1024**3)  # GB
                
                if available_space < required_storage * 2:  # Require 2x space for safety
                    warnings.append(f"Insufficient disk space. Required: {reqs['storage']}, Available: {available_space:.1f}GB")
                    meets_requirements = False
            except Exception as e:
                warnings.append(f"Could not check disk space: {e}")
        
        # Check RAM (approximate)
        try:
            import psutil
            available_ram = psutil.virtual_memory().total / (1024**3)  # GB
            if reqs.get('ram'):
                required_ram = float(reqs['ram'].replace('GB+', '').replace('GB', ''))
                if available_ram < required_ram:
                    warnings.append(f"RAM ({available_ram:.1f}GB) may be insufficient for {reqs['ram']}")
        except ImportError:
            warnings.append("psutil not available - cannot check RAM requirements")
        
        return meets_requirements, warnings
    
    def download_model(self, model_info: Dict[str, Any], progress_callback=None) -> bool:
        """
        Download a model with progress tracking.
        
        Args:
            model_info (Dict[str, Any]): Model information
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if download successful, False otherwise
        """
        model_id = model_info['id']
        model_name = model_info['name']
        
        info(f"Starting download of {model_name} ({model_id})")
        log_user_action("model_download_started", model_id=model_id)
        
        try:
            # Check system requirements
            meets_requirements, warnings = self.check_system_requirements(model_info)
            
            if warnings:
                warning(f"System requirement warnings for {model_id}: {warnings}")
                if not meets_requirements:
                    error(f"System requirements not met for {model_id}")
                    return False
            
            # Create model directory
            model_dir = self.models_dir / model_id.replace('/', '_')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model info
            info_file = model_dir / "model_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            # Generate download script
            download_script = self._generate_download_script(model_info)
            script_file = self.scripts_dir / f"{model_id.replace('/', '_')}_download.py"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(download_script)
            
            # Generate configuration file
            config = self._generate_model_config(model_info)
            config_file = self.configs_dir / f"{model_id.replace('/', '_')}_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Create requirements file
            requirements = self._generate_requirements(model_info)
            req_file = model_dir / "requirements.txt"
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(requirements))
            
            info(f"Model {model_id} prepared for download successfully")
            log_user_action("model_download_prepared", model_id=model_id)
            
            return True
            
        except Exception as e:
            error(f"Error preparing download for {model_id}: {str(e)}")
            log_user_action("model_download_failed", model_id=model_id, error=str(e))
            return False
    
    def _generate_download_script(self, model_info: Dict[str, Any]) -> str:
        """Generate a Python script for downloading and using the model."""
        model_id = model_info['id']
        model_name = model_info['name']
        
        script_template = f'''"""
Auto-generated download script for {model_name}
Model ID: {model_id}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

System Requirements:
- GPU Memory: {model_info.get('system_requirements', {}).get('gpu_memory', 'N/A')}
- RAM: {model_info.get('system_requirements', {}).get('ram', 'N/A')}
- Storage: {model_info.get('system_requirements', {}).get('storage', 'N/A')}
- CUDA Required: {model_info.get('system_requirements', {}).get('cuda', False)}
"""

import os
import sys
import torch
from pathlib import Path

def check_requirements():
    """Check if system meets model requirements."""
    print("🔍 Checking system requirements...")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch version: {{torch.__version__}}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {{torch.cuda.get_device_name(0)}}")
            print(f"✅ GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}}GB")
        else:
            print("⚠️  CUDA not available - will use CPU")
            
    except ImportError:
        print("❌ PyTorch not installed!")
        return False
    
    # Check Transformers
    try:
        import transformers
        print(f"✅ Transformers version: {{transformers.__version__}}")
    except ImportError:
        print("❌ Transformers not installed!")
        return False
    
    return True

def download_model():
    """Download the model with proper configuration."""
    if not check_requirements():
        print("❌ Requirements not met. Please install required packages.")
        return None, None
    
    model_name = "{model_id}"
    print(f"\\n📥 Downloading {{model_name}}...")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # Download configuration
        print("📋 Downloading model configuration...")
        config = AutoConfig.from_pretrained(model_name)
        
        # Download tokenizer
        print("🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model with appropriate settings
        print("🧠 Downloading model...")
        
        # Configure model loading parameters
        model_kwargs = {{
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }}
        
        # Add device mapping for GPU
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        
        print("✅ Model downloaded successfully!")
        print(f"Model: {{model_name}}")
        print(f"Device: {{'CUDA' if torch.cuda.is_available() else 'CPU'}}")
        
        # Save model info
        model_info = {{
            "model_name": "{model_name}",
            "model_id": "{model_id}",
            "download_time": "{datetime.now().isoformat()}",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }}
        
        import json
        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error downloading model: {{e}}")
        return None, None

def test_model(model, tokenizer):
    """Test the downloaded model with a simple example."""
    if not model or not tokenizer:
        print("❌ No model or tokenizer available for testing")
        return
    
    print("\\n🧪 Testing model...")
    
    try:
        # Example test based on model type
        if hasattr(model, 'generate'):
            # Text generation model
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {{k: v.cuda() for k, v in inputs.items()}}
                model = model.cuda()
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50, do_sample=True)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Test input: {{test_input}}")
            print(f"Model output: {{response}}")
        
        print("✅ Model test completed successfully!")
        
    except Exception as e:
        print(f"⚠️  Model test failed: {{e}}")

def main():
    """Main function to download and test the model."""
    print("🚀 Starting model download process...")
    print(f"Model: {model_name}")
    print(f"ID: {model_id}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Download model
    model, tokenizer = download_model()
    
    if model and tokenizer:
        print("\\n🎉 Model is ready to use!")
        
        # Test the model
        test_model(model, tokenizer)
        
        print("\\n📋 Usage example:")
        print("```python")
        print("from transformers import AutoModel, AutoTokenizer")
        print(f"model = AutoModel.from_pretrained('{model_id}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{model_id}')")
        print("```")
    else:
        print("❌ Failed to download model")

if __name__ == "__main__":
    main()
'''
        return script_template
    
    def _generate_model_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration file for the model."""
        return {
            "model_info": model_info,
            "download_config": {
                "timestamp": datetime.now().isoformat(),
                "auto_generated": True,
                "version": "1.0"
            },
            "usage_examples": [
                {
                    "description": "Basic usage",
                    "code": f"from transformers import AutoModel, AutoTokenizer\nmodel = AutoModel.from_pretrained('{model_info['id']}')\ntokenizer = AutoTokenizer.from_pretrained('{model_info['id']}')"
                }
            ],
            "system_requirements": model_info.get('system_requirements', {}),
            "recommended_settings": {
                "batch_size": 1,
                "max_length": 512,
                "temperature": 0.7,
                "do_sample": True
            }
        }
    
    def _generate_requirements(self, model_info: Dict[str, Any]) -> List[str]:
        """Generate requirements.txt for the model."""
        base_requirements = [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "tokenizers>=0.12.0",
            "numpy>=1.21.0",
            "requests>=2.25.0"
        ]
        
        # Add model-specific requirements
        if model_info.get('system_requirements', {}).get('cuda', False):
            base_requirements.append("torch[cuda]")
        
        # Add vision model requirements
        if 'vision' in model_info.get('category', '').lower():
            base_requirements.extend([
                "pillow>=8.0.0",
                "torchvision>=0.10.0"
            ])
        
        # Add audio model requirements
        if 'audio' in model_info.get('category', '').lower():
            base_requirements.extend([
                "librosa>=0.8.0",
                "soundfile>=0.10.0"
            ])
        
        return base_requirements
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """List all downloaded models."""
        downloaded_models = []
        
        for info_file in self.base_dir.rglob("model_info.json"):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                downloaded_models.append(model_info)
            except Exception as e:
                warning(f"Error reading model info from {info_file}: {e}")
        
        return downloaded_models
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a downloaded model."""
        model_dir = self.models_dir / model_id.replace('/', '_')
        info_file = model_dir / "model_info.json"
        
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                error(f"Error reading model info: {e}")
        
        return None
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model."""
        try:
            model_dir = self.models_dir / model_id.replace('/', '_')
            
            if model_dir.exists():
                shutil.rmtree(model_dir)
                info(f"Removed model directory: {model_dir}")
            
            # Remove related files
            script_file = self.scripts_dir / f"{model_id.replace('/', '_')}_download.py"
            config_file = self.configs_dir / f"{model_id.replace('/', '_')}_config.json"
            
            for file_path in [script_file, config_file]:
                if file_path.exists():
                    file_path.unlink()
                    info(f"Removed file: {file_path}")
            
            log_user_action("model_removed", model_id=model_id)
            return True
            
        except Exception as e:
            error(f"Error removing model {model_id}: {e}")
            return False
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        downloaded_models = self.list_downloaded_models()
        
        total_size = 0
        categories = {}
        
        for model in downloaded_models:
            # Estimate model size
            model_size = model.get('system_requirements', {}).get('storage', '1GB')
            try:
                size_gb = float(model_size.replace('GB', ''))
                total_size += size_gb
            except:
                total_size += 1  # Default to 1GB
            
            # Count by category
            category = model.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_models": len(downloaded_models),
            "total_size_gb": round(total_size, 2),
            "categories": categories,
            "download_dir": str(self.base_dir)
        }


def create_model_downloader(base_dir: str = "output/downloaded_models") -> ModelDownloader:
    """Create and return a ModelDownloader instance."""
    return ModelDownloader(base_dir)
