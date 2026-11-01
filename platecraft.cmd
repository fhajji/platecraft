@echo off
rem PlateCraft launcher: uses the engine's core Python, falls back to the system one.
setlocal
set "PC_ROOT=%~dp0"
set "PC_PY=%PC_ROOT%engine\envs\core\Scripts\python.exe"
if not exist "%PC_PY%" (
    where py >nul 2>nul && (set "PC_PY=py -3") || (set "PC_PY=python")
)
%PC_PY% "%PC_ROOT%platecraft.py" %*
exit /b %ERRORLEVEL%
