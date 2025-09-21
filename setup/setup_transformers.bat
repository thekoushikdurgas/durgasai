@echo off
REM HuggingFace Transformers Setup Script for Windows
REM This batch file provides an easy way to set up HuggingFace Transformers

echo.
echo ========================================
echo   HuggingFace Transformers Setup
echo   for DurgasAI Application
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.9+ is required
    echo Current version:
    python --version
    pause
    exit /b 1
)

echo Python version check passed!
echo.

REM Run the setup script
echo Running Transformers setup...
python setup_transformers.py

REM Check if setup was successful
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   Setup completed successfully!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Start DurgasAI application
    echo 2. Go to HuggingFace page
    echo 3. Try the installation guide
    echo.
) else (
    echo.
    echo ========================================
    echo   Setup failed!
    echo ========================================
    echo.
    echo Check the error messages above for details
    echo.
)

pause
