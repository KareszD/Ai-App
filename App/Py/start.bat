@echo off
cd /d "%~dp0"
echo Current Working Directory: %cd%
call .venv\Scripts\activate
python api.py
pause
