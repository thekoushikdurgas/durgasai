@echo off
echo ========================================
echo HunyuanImage-2.1 Local Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
python setup_hunyuan_local.py

echo.
echo Setup completed! You can now run:
echo   python test_hunyuan_local.py
echo.
pause
