@echo off
chcp 65001 >nul
title VLM-Game-Agent

cd /d "%~dp0" || exit /b 1

"venv\Scripts\python.exe" examples\agent_demo.py

echo.
echo Agent 已退出，按任意键关闭...
pause >nul
