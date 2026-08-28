@echo off
rem Double-click this to open the segmentation interface.
cd /d "%~dp0"

rem Conda is the documented way to install this, but nothing here requires it - any
rem python that can import totalsegmentator will do. Try the conda env first, then a
rem local venv, then whatever is on PATH. "conda activate" is not used: it fails
rem silently inside a double-clicked batch file and then runs the wrong python.
rem A python that cannot import totalsegmentator is no use - the interface would start
rem and then fail on the first segmentation - so each candidate is tested, not assumed.
setlocal enabledelayedexpansion
set PY=

for %%P in (
  "%USERPROFILE%\anaconda3\envs\segmentator\python.exe"
  "%USERPROFILE%\miniconda3\envs\segmentator\python.exe"
  "%LOCALAPPDATA%\anaconda3\envs\segmentator\python.exe"
  "%LOCALAPPDATA%\miniconda3\envs\segmentator\python.exe"
  "C:\ProgramData\anaconda3\envs\segmentator\python.exe"
  "C:\ProgramData\miniconda3\envs\segmentator\python.exe"
  ".venv\Scripts\python.exe"
) do if not defined PY if exist %%P (
  %%P -c "import totalsegmentator" >nul 2>&1 && set PY=%%P
)

if not defined PY (
  for /f "delims=" %%W in ('where python 2^>nul') do if not defined PY (
    "%%W" -c "import totalsegmentator" >nul 2>&1 && set PY="%%W"
  )
)

if not defined PY (
  echo.
  echo   Could not find a Python with TotalSegmentator installed.
  echo.
  echo   If you followed the setup guide, open Anaconda Prompt and run:
  echo       conda activate segmentator
  echo       cd /d "%~dp0"
  echo       python ct_gui.py --open
  echo.
  echo   If you have not installed it yet, see the README - step 2.
  echo.
  pause
  exit /b 1
)

echo Starting the interface... a browser window will open.
%PY% ct_gui.py --open
echo.
echo The server has stopped.
pause
