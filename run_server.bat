@echo off
title Webtoon Translator Server
echo ===================================================
echo Starting Webtoon Translator Local Backend Server...
echo http://127.0.0.1:8000
echo ===================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m server.main
pause
