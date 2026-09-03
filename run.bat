@echo off
setlocal
cd /d "%~dp0"

REM Double-click this file, or run `run.bat` from a terminal.
REM `run.bat fresh` retrains the models and regenerates all outputs.

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo First run. Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto nopython
    echo Installing dependencies. This takes a minute.
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    echo.
)

if /i "%~1"=="fresh" (
    echo Retraining from scratch, about 95 seconds...
    "%PY%" run_pipeline.py
    if errorlevel 1 goto failed
    echo.
) else (
    if not exist "outputs\decisions.csv" (
        echo No outputs found. Generating data and training models, about 95 seconds...
        "%PY%" run_pipeline.py
        if errorlevel 1 goto failed
        echo.
    )
)

echo Starting the dashboard at http://localhost:8501
echo Press Ctrl+C in this window to stop it.
echo.
"%PY%" -m streamlit run app.py
goto end

:nopython
echo.
echo Could not create the virtual environment.
echo Python may not be installed or not on PATH. Check with:  python --version
echo Install from python.org and tick "Add Python to PATH".
pause
goto end

:failed
echo.
echo Setup failed. Scroll up for the error.
pause

:end
endlocal
