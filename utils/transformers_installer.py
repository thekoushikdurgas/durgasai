"""
Transformers Installation and Setup Tools.

This module provides automated installation and setup tools for HuggingFace Transformers,
including dependency checking, environment setup, and installation verification.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Import logging with fallback
try:
    from .logger import debug, info, warning, error
except ImportError:
    def debug(msg, component="installer", **kwargs): print(f"[DEBUG] {msg}")
    def info(msg, component="installer", **kwargs): print(f"[INFO] {msg}")
    def warning(msg, component="installer", **kwargs): print(f"[WARNING] {msg}")
    def error(msg, component="installer", **kwargs): print(f"[ERROR] {msg}")


class TransformersInstaller:
    """
    Automated installer for HuggingFace Transformers and related dependencies.
    
    This class provides:
    - System compatibility checking
    - Dependency installation
    - Virtual environment setup
    - Installation verification
    - Configuration management
    """
    
    def __init__(self):
        """Initialize the Transformers installer."""
        self.system_info = self._get_system_info()
        self.requirements_files = {
            "basic": "requirements-minimal.txt",
            "full": "requirements.txt", 
            "transformers": "requirements-transformers.txt"
        }
        info("TransformersInstaller initialized", "installer", 
             system=platform.system(), 
             python_version=sys.version)
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for compatibility checking."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "python_version": sys.version,
            "python_executable": sys.executable,
            "cwd": os.getcwd()
        }
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """
        Check if the system meets the requirements for Transformers.
        
        Returns:
            Dict[str, bool]: System requirement check results
        """
        info("Checking system requirements", "installer")
        
        checks = {
            "python_version": self._check_python_version(),
            "pip_available": self._check_pip_available(),
            "virtual_env_support": self._check_venv_support(),
            "gpu_available": self._check_gpu_availability(),
            "sufficient_memory": self._check_memory_requirements(),
            "disk_space": self._check_disk_space()
        }
        
        # Log results
        for check, result in checks.items():
            status = "✅" if result else "❌"
            info(f"{status} {check}: {result}", "installer")
        
        return checks
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        debug("Starting Python version compatibility check", "installer")
        
        version_info = sys.version_info
        required_version = (3, 9)
        compatible = version_info >= required_version
        
        debug(f"Python version check details", "installer",
              current_version=f"{version_info.major}.{version_info.minor}.{version_info.micro}",
              required_version=f"{required_version[0]}.{required_version[1]}",
              compatible=compatible)
        
        info(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}", "installer")
        
        if not compatible:
            warning(f"Python version {version_info.major}.{version_info.minor} is below required {required_version[0]}.{required_version[1]}", "installer")
        
        return compatible
    
    def _check_pip_available(self) -> bool:
        """Check if pip is available."""
        debug("Starting pip availability check", "installer")
        
        try:
            # First try importing pip directly
            debug("Attempting to import pip module", "installer")
            import pip
            debug("pip module imported successfully", "installer", pip_version=getattr(pip, '__version__', 'unknown'))
            return True
        except ImportError:
            debug("pip module import failed, trying command line", "installer")
            
            try:
                # Try running pip via command line
                debug("Executing pip --version command", "installer")
                result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                     check=True, capture_output=True, text=True)
                debug("pip command executed successfully", "installer",
                      output=result.stdout.strip() if result.stdout else "no output")
                return True
            except subprocess.CalledProcessError as e:
                debug("pip command line execution failed", "installer",
                      return_code=e.returncode,
                      error_output=e.stderr if e.stderr else "no error output")
                return False
            except Exception as e:
                debug(f"Unexpected error checking pip: {e}", "installer")
                return False
    
    def _check_venv_support(self) -> bool:
        """Check if venv module is available."""
        try:
            import venv
            return True
        except ImportError:
            return False
    
    def _check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_memory_requirements(self) -> bool:
        """Check if system has sufficient memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Check if at least 4GB available
            return memory.total >= 4 * 1024**3
        except ImportError:
            # If psutil not available, assume sufficient memory
            warning("psutil not available, cannot check memory", "installer")
            return True
    
    def _check_disk_space(self) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            # Check if at least 2GB free space
            free_space = shutil.disk_usage('.').free
            return free_space >= 2 * 1024**3
        except Exception:
            # If check fails, assume sufficient space
            warning("Could not check disk space", "installer")
            return True
    
    def create_virtual_environment(self, env_name: str = "transformers_env", 
                                 force: bool = False) -> Tuple[bool, str]:
        """
        Create a virtual environment for Transformers installation.
        
        Args:
            env_name (str): Name of the virtual environment
            force (bool): Whether to recreate if it already exists
            
        Returns:
            Tuple[bool, str]: (success, path_to_environment)
        """
        info(f"Creating virtual environment: {env_name}", "installer")
        
        env_path = Path(env_name)
        
        # Check if environment already exists
        if env_path.exists():
            if force:
                info(f"Removing existing environment: {env_name}", "installer")
                import shutil
                shutil.rmtree(env_path)
            else:
                warning(f"Virtual environment already exists: {env_name}", "installer")
                return True, str(env_path.absolute())
        
        try:
            # Create virtual environment
            subprocess.run([
                sys.executable, "-m", "venv", str(env_path)
            ], check=True)
            
            info(f"Virtual environment created successfully: {env_path.absolute()}", "installer")
            
            # Provide activation instructions
            if platform.system() == "Windows":
                activation_cmd = f"{env_name}\\Scripts\\activate"
            else:
                activation_cmd = f"source {env_name}/bin/activate"
            
            info(f"To activate: {activation_cmd}", "installer")
            
            return True, str(env_path.absolute())
            
        except subprocess.CalledProcessError as e:
            error(f"Failed to create virtual environment: {e}", "installer")
            return False, ""
    
    def install_requirements(self, requirements_file: str = "requirements-transformers.txt",
                           upgrade: bool = True) -> bool:
        """
        Install requirements from a requirements file.
        
        Args:
            requirements_file (str): Path to requirements file
            upgrade (bool): Whether to upgrade existing packages
            
        Returns:
            bool: True if installation successful
        """
        info(f"Installing requirements from: {requirements_file}", "installer")
        
        if not Path(requirements_file).exists():
            error(f"Requirements file not found: {requirements_file}", "installer")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
            
            if upgrade:
                cmd.append("--upgrade")
            
            info(f"Running command: {' '.join(cmd)}", "installer")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            info("Requirements installed successfully", "installer")
            debug(f"Installation output: {result.stdout}", "installer")
            
            return True
            
        except subprocess.CalledProcessError as e:
            error(f"Failed to install requirements: {e}", "installer")
            debug(f"Error output: {e.stderr}", "installer")
            return False
    
    def install_transformers_basic(self) -> bool:
        """Install basic Transformers package."""
        info("Installing basic Transformers package", "installer")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "transformers"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            info("Basic Transformers package installed successfully", "installer")
            return True
            
        except subprocess.CalledProcessError as e:
            error(f"Failed to install basic Transformers: {e}", "installer")
            return False
    
    def install_transformers_with_torch(self) -> bool:
        """Install Transformers with PyTorch support."""
        info("Installing Transformers with PyTorch support", "installer")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "transformers[torch]"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            info("Transformers with PyTorch installed successfully", "installer")
            return True
            
        except subprocess.CalledProcessError as e:
            error(f"Failed to install Transformers with PyTorch: {e}", "installer")
            return False
    
    def verify_installation(self) -> Dict[str, bool]:
        """
        Verify that Transformers and related packages are properly installed.
        
        Returns:
            Dict[str, bool]: Verification results for each package
        """
        info("Verifying Transformers installation", "installer")
        
        verification_results = {}
        
        # Test basic import
        try:
            import transformers
            verification_results["transformers_import"] = True
            info(f"✅ Transformers imported successfully (v{transformers.__version__})", "installer")
        except ImportError as e:
            verification_results["transformers_import"] = False
            error(f"❌ Failed to import Transformers: {e}", "installer")
        
        # Test pipeline creation
        try:
            from transformers import pipeline
            classifier = pipeline('sentiment-analysis')
            result = classifier('This is a test.')
            verification_results["pipeline_test"] = True
            info("✅ Pipeline test successful", "installer")
        except Exception as e:
            verification_results["pipeline_test"] = False
            error(f"❌ Pipeline test failed: {e}", "installer")
        
        # Test PyTorch availability
        try:
            import torch
            verification_results["torch_available"] = True
            info(f"✅ PyTorch available (v{torch.__version__})", "installer")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                verification_results["cuda_available"] = True
                info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}", "installer")
            else:
                verification_results["cuda_available"] = False
                info("ℹ️ CUDA not available (CPU-only mode)", "installer")
                
        except ImportError:
            verification_results["torch_available"] = False
            verification_results["cuda_available"] = False
            warning("⚠️ PyTorch not available", "installer")
        
        # Test tokenizers
        try:
            from transformers import AutoTokenizer
            verification_results["tokenizers_available"] = True
            info("✅ Tokenizers available", "installer")
        except ImportError:
            verification_results["tokenizers_available"] = False
            error("❌ Tokenizers not available", "installer")
        
        # Overall success
        critical_tests = ["transformers_import", "pipeline_test"]
        verification_results["overall_success"] = all(
            verification_results.get(test, False) for test in critical_tests
        )
        
        if verification_results["overall_success"]:
            info("🎉 Transformers installation verified successfully!", "installer")
        else:
            error("❌ Transformers installation verification failed", "installer")
        
        return verification_results
    
    def setup_cache_directories(self) -> bool:
        """Set up cache directories for HuggingFace models."""
        info("Setting up cache directories", "installer")
        
        cache_dirs = [
            "output/cache/huggingface",
            "output/cache/transformers", 
            "output/cache/torch"
        ]
        
        try:
            for cache_dir in cache_dirs:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                info(f"Created cache directory: {cache_dir}", "installer")
            
            # Set environment variables using centralized config
            try:
                from .config import Config
                
                # Setup environment variables from configuration
                Config.setup_environment_variables()
                
                # Get current environment variables to verify setup
                env_vars = Config.get_environment_variables()
                
                info("Environment variables configured from centralized config", "installer")
                info(f"HF_HOME = {env_vars.get('HF_HOME')}", "installer")
                info(f"TRANSFORMERS_CACHE = {env_vars.get('TRANSFORMERS_CACHE')}", "installer")
                info(f"TORCH_HOME = {env_vars.get('TORCH_HOME')}", "installer")
                
            except Exception as e:
                warning(f"Could not load centralized config: {e}", "installer")
                info("Using fallback environment variable setup", "installer")
                
                # Fallback to direct setting
                os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path("output/cache/huggingface").absolute())
                os.environ["HF_HOME"] = str(Path("output/cache/huggingface").absolute())
                os.environ["TRANSFORMERS_CACHE"] = str(Path("output/cache/transformers").absolute())
                os.environ["TORCH_HOME"] = str(Path("output/cache/torch").absolute())
            
            info("Cache directories and environment variables set up successfully", "installer")
            return True
            
        except Exception as e:
            error(f"Failed to set up cache directories: {e}", "installer")
            return False
    
    def generate_installation_report(self, verification_results: Dict[str, bool]) -> str:
        """Generate a detailed installation report."""
        report = []
        report.append("# 🤗 HuggingFace Transformers Installation Report")
        report.append("")
        report.append(f"**Installation Date:** {os.popen('date').read().strip()}")
        report.append(f"**System:** {self.system_info['platform']} {self.system_info['architecture']}")
        report.append(f"**Python:** {self.system_info['python_version']}")
        report.append("")
        
        report.append("## ✅ Verification Results")
        for test, result in verification_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report.append(f"- **{test}**: {status}")
        
        report.append("")
        report.append("## 🚀 Next Steps")
        
        if verification_results.get("overall_success", False):
            report.append("- ✅ Transformers is ready to use!")
            report.append("- 🧪 Try the quick start example in the notebook")
            report.append("- 📚 Check out the HuggingFace documentation")
        else:
            report.append("- ❌ Some components failed to install")
            report.append("- 🔧 Check the error messages above")
            report.append("- 📖 Refer to the troubleshooting guide")
        
        return "\n".join(report)
    
    def full_installation(self, env_name: str = "transformers_env",
                         requirements_file: str = "requirements-transformers.txt") -> bool:
        """
        Perform a complete Transformers installation.
        
        Args:
            env_name (str): Virtual environment name
            requirements_file (str): Requirements file to use
            
        Returns:
            bool: True if installation successful
        """
        info("Starting full Transformers installation", "installer")
        
        # Step 1: Check system requirements
        info("Step 1: Checking system requirements", "installer")
        requirements_check = self.check_system_requirements()
        
        if not requirements_check.get("python_version", False):
            error("Python version not compatible (requires 3.9+)", "installer")
            return False
        
        if not requirements_check.get("pip_available", False):
            error("pip not available", "installer")
            return False
        
        # Step 2: Set up cache directories
        info("Step 2: Setting up cache directories", "installer")
        if not self.setup_cache_directories():
            warning("Failed to set up cache directories, continuing anyway", "installer")
        
        # Step 3: Install requirements
        info("Step 3: Installing requirements", "installer")
        if Path(requirements_file).exists():
            if not self.install_requirements(requirements_file):
                error("Failed to install from requirements file", "installer")
                return False
        else:
            warning(f"Requirements file not found: {requirements_file}", "installer")
            if not self.install_transformers_with_torch():
                error("Failed to install Transformers with PyTorch", "installer")
                return False
        
        # Step 4: Verify installation
        info("Step 4: Verifying installation", "installer")
        verification_results = self.verify_installation()
        
        # Step 5: Generate report
        report = self.generate_installation_report(verification_results)
        
        # Save report
        report_path = "installation_report.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            info(f"Installation report saved to: {report_path}", "installer")
        except Exception as e:
            warning(f"Failed to save installation report: {e}", "installer")
        
        success = verification_results.get("overall_success", False)
        
        if success:
            info("🎉 Full installation completed successfully!", "installer")
        else:
            error("❌ Full installation completed with errors", "installer")
        
        return success


def install_transformers_cli():
    """Command-line interface for Transformers installation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install HuggingFace Transformers")
    parser.add_argument("--env-name", default="transformers_env", 
                       help="Virtual environment name")
    parser.add_argument("--requirements", default="requirements-transformers.txt",
                       help="Requirements file to use")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check system requirements")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing installation")
    
    args = parser.parse_args()
    
    installer = TransformersInstaller()
    
    if args.check_only:
        installer.check_system_requirements()
    elif args.verify_only:
        installer.verify_installation()
    else:
        installer.full_installation(args.env_name, args.requirements)


if __name__ == "__main__":
    install_transformers_cli()
