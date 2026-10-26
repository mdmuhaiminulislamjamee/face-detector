@echo off
call "%~dp0run_face_detection.bat" --ncm %*
exit /b %errorlevel%
