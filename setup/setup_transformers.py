#!/usr/bin/env python3
"""
HuggingFace Transformers Setup Script for DurgasAI.

This script provides an easy way to set up HuggingFace Transformers
for the DurgasAI application with proper dependencies and configuration.
"""

import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from transformers_installer import TransformersInstaller


def main():
    """Main setup function."""
    print("🤗 HuggingFace Transformers Setup for DurgasAI")
    print("=" * 50)
    
    # Initialize installer
    installer = TransformersInstaller()
    
    print("\n🔍 Checking system requirements...")
    requirements_check = installer.check_system_requirements()
    
    # Check critical requirements
    if not requirements_check.get("python_version", False):
        print("❌ Python version not compatible (requires 3.9+)")
        return False
    
    if not requirements_check.get("pip_available", False):
        print("❌ pip not available")
        return False
    
    print("\n✅ System requirements check passed!")
    
    # Ask user for installation options
    print("\n📋 Installation Options:")
    print("1. Quick install (basic Transformers)")
    print("2. Full install (with PyTorch and all dependencies)")
    print("3. Install from requirements file")
    print("4. Check system only")
    print("5. Verify existing installation")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Installing basic Transformers...")
        success = installer.install_transformers_basic()
        
    elif choice == "2":
        print("\n🚀 Installing Transformers with PyTorch...")
        success = installer.install_transformers_with_torch()
        
    elif choice == "3":
        requirements_file = input("Enter requirements file path (default: requirements-transformers.txt): ").strip()
        if not requirements_file:
            requirements_file = "requirements-transformers.txt"
        
        print(f"\n🚀 Installing from {requirements_file}...")
        success = installer.install_requirements(requirements_file)
        
    elif choice == "4":
        print("\n✅ System check complete!")
        return True
        
    elif choice == "5":
        print("\n🔍 Verifying existing installation...")
        verification_results = installer.verify_installation()
        success = verification_results.get("overall_success", False)
        
    else:
        print("❌ Invalid choice")
        return False
    
    if success:
        print("\n✅ Installation completed successfully!")
        
        # Verify installation
        print("\n🔍 Verifying installation...")
        verification_results = installer.verify_installation()
        
        if verification_results.get("overall_success", False):
            print("\n🎉 Transformers is ready to use!")
            print("\n📚 Next steps:")
            print("- Open the HuggingFace page in DurgasAI")
            print("- Check out the installation guide")
            print("- Try the quick start examples")
        else:
            print("\n⚠️ Installation completed but verification failed")
            print("Check the error messages above for details")
            
        # Generate report
        report = installer.generate_installation_report(verification_results)
        print(f"\n📄 Installation report:\n{report}")
        
    else:
        print("\n❌ Installation failed!")
        print("Check the error messages above for details")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
