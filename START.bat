@echo off
rem Launches the poly-framepack-studio server. Extra arguments are passed to studio.py.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv. Run install.bat first, or see README.md.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python studio.py --port 7860 %*
pause
