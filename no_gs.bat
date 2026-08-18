@echo off
cd /d "%~dp0"

call venv\Scripts\activate

python dummy_gesture_control.py

pause
