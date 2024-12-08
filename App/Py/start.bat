@echo off
cd /d "%~dp0"
echo Current Working Directory: %cd%
call .venvmain\Scripts\activate
python api.py
pause
