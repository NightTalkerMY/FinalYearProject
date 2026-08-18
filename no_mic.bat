@echo off
cd /d "%~dp0"

call venv\Scripts\activate

python dummy_no_mic.py

pause
