@echo off
echo 🤖 DurgasAI Setup Script
echo ========================
echo.
echo Installing required dependencies...
pip install -r requirements-minimal.txt
echo.
echo Setup complete! 
echo.
echo To start the application, run:
echo   start_app.bat
echo.
echo Or manually run:
echo   streamlit run app.py
echo.
pause
