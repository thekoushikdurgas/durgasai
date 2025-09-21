#!/usr/bin/env python3
"""
DurgasAI Installation Script

This script provides different installation options for DurgasAI:
1. Minimal installation - Basic functionality only
2. Full installation - All features and dependencies
3. Development installation - Includes development tools
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_minimal():
    """Install minimal dependencies."""
    print("\n🚀 Installing minimal dependencies...")
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements-minimal.txt", "Installing minimal requirements"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    print("\n✅ Minimal installation completed!")
    print("📋 Installed packages:")
    print("   - Streamlit for web interface")
    print("   - LangChain for AI model integration")
    print("   - HuggingFace transformers")
    print("   - Basic PyTorch (CPU version)")
    print("   - Essential utilities")
    
    return True


def install_full():
    """Install full dependencies."""
    print("\n🚀 Installing full dependencies...")
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing full requirements"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    print("\n✅ Full installation completed!")
    print("📋 Installed packages:")
    print("   - All minimal dependencies")
    print("   - Advanced HuggingFace features")
    print("   - Vision model support")
    print("   - Development tools")
    print("   - Performance optimization libraries")
    
    return True


def install_development():
    """Install development dependencies."""
    print("\n🚀 Installing development dependencies...")
    
    # First install full requirements
    if not install_full():
        return False
    
    # Additional development tools
    dev_packages = [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.11.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0"
    ]
    
    for package in dev_packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('>=')[0]}"):
            print(f"⚠️ Warning: Failed to install {package}")
    
    print("\n✅ Development installation completed!")
    print("📋 Additional development tools:")
    print("   - pytest for testing")
    print("   - black for code formatting")
    print("   - flake8 for linting")
    print("   - mypy for type checking")
    print("   - pre-commit for git hooks")
    
    return True


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "output/cache/huggingface",
        "output/cache/transformers", 
        "output/cache/torch",
        "output/logs",
        "output/sessions",
        "output/temp",
        "logs/debug",
        "logs/performance",
        "logs/sessions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def verify_installation():
    """Verify that key packages are installed."""
    print("\n🔍 Verifying installation...")
    
    key_packages = [
        "streamlit",
        "langchain",
        "transformers",
        "torch",
        "requests",
        "pandas",
        "numpy"
    ]
    
    failed_packages = []
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - FAILED")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️ Warning: {len(failed_packages)} packages failed to import:")
        for package in failed_packages:
            print(f"   - {package}")
        return False
    
    print("\n✅ All key packages verified successfully!")
    return True


def main():
    """Main installation function."""
    print("🤖 DurgasAI Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Show installation options
    print("\n📋 Installation Options:")
    print("1. Minimal - Basic functionality only (recommended for beginners)")
    print("2. Full - All features and dependencies (recommended for most users)")
    print("3. Development - Full + development tools (for contributors)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\n🎯 Select installation type (1-4): ").strip()
            
            if choice == "1":
                install_type = "minimal"
                break
            elif choice == "2":
                install_type = "full"
                break
            elif choice == "3":
                install_type = "development"
                break
            elif choice == "4":
                print("👋 Installation cancelled")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Installation cancelled")
            sys.exit(0)
    
    # Setup directories
    if not setup_directories():
        print("❌ Failed to setup directories")
        sys.exit(1)
    
    # Install based on choice
    success = False
    if install_type == "minimal":
        success = install_minimal()
    elif install_type == "full":
        success = install_full()
    elif install_type == "development":
        success = install_development()
    
    if not success:
        print("\n❌ Installation failed!")
        print("💡 Try running the installation again or check the error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️ Installation completed but some packages may have issues.")
        print("💡 Try running the app to see if everything works correctly.")
    else:
        print("\n🎉 Installation completed successfully!")
    
    # Show next steps
    print("\n🚀 Next Steps:")
    print("1. Get your HuggingFace API token from: https://huggingface.co/settings/tokens")
    print("2. Run the application:")
    print("   streamlit run app.py")
    print("3. Navigate to '🤗 HuggingFace Chat' to start using the new feature!")
    
    print("\n📚 Documentation:")
    print("- HuggingFace Page Guide: docs/HUGGINGFACE_PAGE_GUIDE.md")
    print("- Model Catalog Guide: docs/MODEL_CATALOG_GUIDE.md")
    
    print("\n🆘 Need Help?")
    print("- Check the logs in the logs/ directory")
    print("- Review the documentation in the docs/ directory")
    print("- Run with --help for more options")


if __name__ == "__main__":
    main()
