@echo off
echo 🤖 Starting DurgasAI Application...
echo.
echo Make sure you have:
echo 1. Installed dependencies: pip install -r requirements-minimal.txt
echo 2. Your HuggingFace API token ready
echo.
echo The application will open in your default browser.
echo Press Ctrl+C to stop the application.
echo.

REM Set environment variables to suppress TensorFlow warnings
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set TF2_BEHAVIOR=1
set CUDA_VISIBLE_DEVICES=

REM Run environment setup (optional)
python env_setup.py

streamlit run app.py
