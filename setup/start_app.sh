#!/bin/bash
# ============================================================================
# DurgasAI Application Startup Script (Linux/macOS)
# 
# This script handles:
# - Environment variable configuration
# - Dependency verification
# - Error handling and logging
# - Application startup with monitoring
# 
# Usage: ./start_app.sh
# ============================================================================

echo ""
echo "========================================"
echo "🤖 DurgasAI Application Startup"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/{debug,performance,sessions}

# Log startup attempt
echo "$(date): Starting DurgasAI application..." >> logs/startup.log

echo "Prerequisites checklist:"
echo "✓ 1. Install dependencies: pip install -r requirements.txt"
echo "✓ 2. Have your HuggingFace API token ready"
echo "✓ 3. Ensure Python 3.8+ is installed"
echo ""

echo "⏳ Initializing environment..."

# Set environment variables to suppress TensorFlow warnings
echo "$(date): Setting TensorFlow environment variables" >> logs/startup.log
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export TF2_BEHAVIOR=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONWARNINGS=ignore

echo "✅ TensorFlow warnings suppression configured"

# Run environment setup with error handling
echo "⏳ Running environment setup..."
python env_setup.py
if [ $? -ne 0 ]; then
    echo "❌ Environment setup failed with exit code $?"
    echo "$(date): Environment setup failed: $?" >> logs/startup.log
    echo ""
    echo "Troubleshooting:"
    echo "- Check if Python is installed and in PATH"
    echo "- Verify requirements.txt dependencies are installed"
    echo "- Check logs/startup.log for details"
    echo "- Try running: pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "✅ Environment setup completed"

# Start the application with error handling
echo ""
echo "🚀 Starting DurgasAI application..."
echo "The application will open in your default browser."
echo "Press Ctrl+C to stop the application."
echo ""
echo "$(date): Starting Streamlit application" >> logs/startup.log

# Start Streamlit with error handling
streamlit run app.py
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ Application failed to start with exit code $exit_code"
    echo "$(date): Streamlit startup failed: $exit_code" >> logs/startup.log
    echo ""
    echo "Troubleshooting:"
    echo "- Check if all dependencies are installed"
    echo "- Verify Python and Streamlit are working"
    echo "- Check logs/startup.log and logs/errors.log for details"
    echo "- Try running: pip install -r requirements.txt"
    echo ""
    exit $exit_code
fi

echo "$(date): Application shutdown" >> logs/startup.log
