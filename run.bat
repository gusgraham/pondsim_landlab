@echo off
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src"

if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

set "COMMAND=%~1"

if "%COMMAND%"=="" set "COMMAND=gui"
if "%COMMAND%"=="-h" goto :help
if "%COMMAND%"=="--help" goto :help

if "%COMMAND%"=="gui" (
    echo Starting Swesim GUI...
    "%PYTHON%" -m swesim.app %*
    goto :eof
)

if "%COMMAND%"=="cli" (
    echo Starting Swesim CLI...
    "%PYTHON%" -m swesim.cli %*
    goto :eof
)

if "%COMMAND%"=="test" (
    echo Running tests...
    "%PYTHON%" -m pytest %*
    goto :eof
)

REM Default fallback
echo Starting Swesim GUI...
"%PYTHON%" -m swesim.app %*
goto :eof

:help
echo Swesim — Overland Flow Simulation
echo.
echo Usage: run.bat [gui^|cli^|test] [args...]
echo.
echo Commands:
echo   gui          Start the Swesim Qt Graphical Interface (default)
echo   cli          Run the Headless CLI for automated pipelines
echo   test         Run the unit test suite
echo.
echo Examples:
echo   run.bat gui
echo   run.bat cli --dem path/to/dem.tif
echo   run.bat test
goto :eof
