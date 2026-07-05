@echo off
setlocal

set APP_NAME=Sattmal
set SPEC_FILE=sct-sattmal-windows.spec
set VENV_PY=.venv\Scripts\python.exe
set VENV_PIP=.venv\Scripts\pip.exe
set VENV_PYINSTALLER=.venv\Scripts\pyinstaller.exe

echo Starting Windows build for %APP_NAME%...

if not exist "%VENV_PY%" (
    echo Virtual environment .venv not found. Create it first with:
    echo   python -m venv .venv
    exit /b 1
)

echo Bumping build version...
"%VENV_PY%" scripts\bump_version.py
if errorlevel 1 exit /b 1

echo Installing dependencies...
"%VENV_PIP%" install -r requirements.txt
if errorlevel 1 exit /b 1
"%VENV_PIP%" install pyinstaller pywebview
if errorlevel 1 exit /b 1

echo Cleaning previous Windows build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building Windows app bundle...
"%VENV_PYINSTALLER%" "%SPEC_FILE%" --noconfirm
if errorlevel 1 exit /b 1

echo.
echo Build complete.
echo App folder: dist\%APP_NAME%
echo Run: dist\%APP_NAME%\%APP_NAME%.exe

endlocal
