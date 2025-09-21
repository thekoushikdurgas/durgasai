@echo off
REM ============================================================================
REM DurgasAI Application Startup Script (Windows)
REM 
REM This script handles:
REM - Environment variable configuration
REM - Dependency verification
REM - Error handling and logging
REM - Application startup with monitoring
REM 
REM Usage: start_app.bat
REM ============================================================================

echo.
echo ========================================
echo 🤖 DurgasAI Application Startup
echo ========================================
echo.

REM Log startup attempt
echo [%date% %time%] Starting DurgasAI application... >> logs\startup.log

echo Prerequisites checklist:
echo ✓ 1. Install dependencies: pip install -r requirements.txt
echo ✓ 2. Have your HuggingFace API token ready
echo ✓ 3. Ensure Python 3.8+ is installed
echo.

echo ⏳ Initializing environment...

REM Set environment variables to suppress TensorFlow warnings
echo [%date% %time%] Setting TensorFlow environment variables >> logs\startup.log
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
set TF2_BEHAVIOR=1
set CUDA_VISIBLE_DEVICES=
set PYTHONWARNINGS=ignore

echo ✅ TensorFlow warnings suppression configured

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs
if not exist "logs\debug" mkdir logs\debug
if not exist "logs\performance" mkdir logs\performance
if not exist "logs\sessions" mkdir logs\sessions

echo ✅ Log directories created

REM Run environment setup with error handling
echo ⏳ Running environment setup...
python env_setup.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Environment setup failed with error level %ERRORLEVEL%
    echo [%date% %time%] Environment setup failed: %ERRORLEVEL% >> logs\startup.log
    echo.
    echo Troubleshooting:
    echo - Check if Python is installed and in PATH
    echo - Verify requirements.txt dependencies are installed
    echo - Check logs\startup.log for details
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo ✅ Environment setup completed

REM Start the application with error handling
echo.
echo 🚀 Starting DurgasAI application...
echo The application will open in your default browser.
echo Press Ctrl+C to stop the application.
echo.
echo [%date% %time%] Starting Streamlit application >> logs\startup.log

streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Application failed to start with error level %ERRORLEVEL%
    echo [%date% %time%] Streamlit startup failed: %ERRORLEVEL% >> logs\startup.log
    echo.
    echo Troubleshooting:
    echo - Check if all dependencies are installed
    echo - Verify Python and Streamlit are working
    echo - Check logs\startup.log and logs\errors.log for details
    echo - Try running: pip install -r requirements.txt
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo [%date% %time%] Application shutdown >> logs\startup.log
