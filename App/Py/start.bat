@echo off
cd /d "%~dp0"
call .venvmain\Scripts\activate
python api.py
