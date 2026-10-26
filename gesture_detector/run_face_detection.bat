@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Live Face Detector and Tracker
color 0b

echo ======================================================================
echo                 LIVE FACE DETECTOR ^& TRACKER
echo ======================================================================
echo.

set "CAMERA_SOURCE_ARGS="
set "CAMERA_SOURCE_LABEL=Using supplied command-line arguments."
:: Explicit arguments keep working without an interactive prompt (including --help).
if not "%~1"=="" goto :prepare_environment

:choose_source
echo Choose the camera source:
echo   [1] Webcam - built-in camera or standard USB webcam
echo   [2] USB-NCM camera - network JPEG stream
echo   [Q] Quit
echo.
choice /c 12Q /n /m "Select camera [1/2] or Q to quit: "
if errorlevel 4 goto :menu_fail
if errorlevel 3 exit /b 0
if errorlevel 2 goto :select_ncm
if errorlevel 1 goto :select_webcam
goto :choose_source

:select_webcam
set "CAMERA_SOURCE_ARGS=--source webcam"
set "CAMERA_SOURCE_LABEL=Webcam - camera index 0."
goto :prepare_environment

:select_ncm
set "CAMERA_SOURCE_ARGS=--ncm"
set "CAMERA_SOURCE_LABEL=USB-NCM camera - TCP/JLIP JPEG stream."
echo.
echo [INFO] Default NCM connection: PC 192.168.50.1 to board 192.168.50.2, TCP 5000.
echo [INFO] The USB-NCM network adapter must have the matching host IP configured.
goto :prepare_environment

:prepare_environment
echo.
cd /d "%~dp0python_face_detector"
if errorlevel 1 goto :missing_detector

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
echo [INFO] Source: %CAMERA_SOURCE_LABEL%
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

.venv\Scripts\python.exe live_face_detector.py %CAMERA_SOURCE_ARGS% %*
set "DETECTOR_EXIT_CODE=%ERRORLEVEL%"
if not "%DETECTOR_EXIT_CODE%"=="0" (
    echo.
    echo [NOTICE] Camera stream closed with code %DETECTOR_EXIT_CODE%.
    pause
)
exit /b %DETECTOR_EXIT_CODE%

:menu_fail
echo [ERROR] Camera selection failed. Run this launcher in an interactive terminal.
echo Or run run_face_detection.bat --source webcam / run_face_detection.bat --ncm.
pause
exit /b 1

:missing_detector
echo [ERROR] The python_face_detector folder is missing beside this launcher.
pause
exit /b 1

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
