@echo off
rem Launches the Polymarket paper-trading bot and its live dashboard,
rem each in its own console window. Close a window (or Ctrl+C) to stop that piece.

cd /d "%~dp0"

start "Polymarket Bot" cmd /k "chcp 65001 >nul && python main.py"
start "Paper Trading Dashboard" cmd /k "chcp 65001 >nul && python paper_stats.py"
