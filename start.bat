@echo off
rem Double-click this to open the segmentation interface.
cd /d "%~dp0"

rem The pipeline lives in the "segmentator" conda environment. Find its python
rem directly rather than relying on "conda activate", which fails silently inside a
rem double-clicked batch file and then runs the wrong python.
set PY=
for %%P in (
  "%USERPROFILE%\anaconda3\envs\segmentator\python.exe"
  "%USERPROFILE%\miniconda3\envs\segmentator\python.exe"
  "%LOCALAPPDATA%\anaconda3\envs\segmentator\python.exe"
  "%LOCALAPPDATA%\miniconda3\envs\segmentator\python.exe"
  "C:\ProgramData\anaconda3\envs\segmentator\python.exe"
  "C:\ProgramData\miniconda3\envs\segmentator\python.exe"
) do if not defined PY if exist %%P set PY=%%P

if not defined PY (
  echo.
  echo   Could not find the "segmentator" environment.
  echo   Open Anaconda Prompt and run:  conda activate segmentator
  echo   then:  python ct_gui.py --open
  echo.
  pause
  exit /b 1
)

echo Starting the interface... a browser window will open.
%PY% ct_gui.py --open
echo.
echo The server has stopped.
pause
