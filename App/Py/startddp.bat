@echo off
CALL .venvmain\Scripts\activate
set USE_LIBUV=0
python ddp.py
