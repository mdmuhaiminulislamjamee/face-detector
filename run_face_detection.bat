@echo off
setlocal EnableDelayedExpansion
title Live Face Detector and Tracker
color 0b

echo ======================================================================
echo                 LIVE FACE DETECTOR ^& TRACKER
echo ======================================================================
echo.

cd /d "%~dp0python_face_detector"

:: Check if local virtual environment exists
if exist ".venv\Scripts\python.exe" goto :run_detector

:: Check if Python is installed on system
python --version >nul 2>&1
if errorlevel 1 goto :no_python

echo [INFO] Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 goto :venv_fail

echo [INFO] Installing required dependencies...
.venv\Scripts\pip.exe install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 goto :pip_fail

:run_detector
echo [INFO] Launching Live Face Detector...
echo [INFO] Source: webcam by default; use --ncm for the USB-NCM board camera.
echo [INFO] Webcam uses MJPEG; NCM receives JLIP JPEG packets. Latest frames only.
echo [INFO] Detection and display limit: 10 FPS.
echo [INFO] Controls:
echo          [Q] or [ESC] - Exit
echo          [S]           - Save screenshot snapshot
echo          [T]           - Toggle Tracking Stabilizer
echo          [M]           - Toggle Facial Landmarks
echo          [D]           - Toggle Distance Estimation
echo          [C]           - Switch Webcam / Reconnect NCM
echo.

.venv\Scripts\python.exe live_face_detector.py %*
if errorlevel 1 (
    echo.
    echo [NOTICE] Camera stream closed with code %ERRORLEVEL%.
    pause
)
goto :eof

:no_python
color 0c
echo [ERROR] Python is not found on your system PATH!
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo Make sure to check Add python.exe to PATH during installation.
echo.
pause
exit /b 1

:venv_fail
color 0c
echo [ERROR] Failed to create virtual environment!
pause
exit /b 1

:pip_fail
color 0c
echo [ERROR] Failed to install dependencies!
pause
exit /b 1
