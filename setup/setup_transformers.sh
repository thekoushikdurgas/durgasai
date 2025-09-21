#!/bin/bash
# HuggingFace Transformers Setup Script for Linux/macOS
# This script provides an easy way to set up HuggingFace Transformers

echo ""
echo "========================================"
echo "  HuggingFace Transformers Setup"
echo "  for DurgasAI Application"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.9+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.9+ is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Python version check passed! ($PYTHON_VERSION)"
echo ""

# Make the setup script executable
chmod +x setup_transformers.py

# Run the setup script
echo "Running Transformers setup..."
$PYTHON_CMD setup_transformers.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Setup completed successfully!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Start DurgasAI application"
    echo "2. Go to HuggingFace page"
    echo "3. Try the installation guide"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  Setup failed!"
    echo "========================================"
    echo ""
    echo "Check the error messages above for details"
    echo ""
fi
