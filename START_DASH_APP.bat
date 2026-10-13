@echo off
REM Start the Dash application from the project directory.
REM No machine-specific absolute paths are used.

setlocal
set "PROJECT_DIR=%~dp0"

echo.
echo ========================================
echo   STARTING SURVIVAL ANALYSIS DASHBOARD
echo ========================================
echo.

cd /d "%PROJECT_DIR%"

if not exist "cargaDataset.py" (
    echo ERROR: cargaDataset.py was not found in:
    echo %PROJECT_DIR%
    pause
    exit /b 1
)

echo [OK] Project found: %PROJECT_DIR%
echo.
echo ========================================
echo   STARTING DASH SERVER
echo ========================================
echo.
echo Open: http://127.0.0.1:8050
echo.
echo To stop: Press Ctrl+C
echo.

python -u cargaDataset.py

pause
