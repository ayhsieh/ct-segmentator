@echo off
rem Double-click this to open the segmentation interface.
cd /d "%~dp0"

rem Nothing here requires conda - any python with the packages will do. Try the conda
rem env first, then a local venv, then 3D Slicer's bundled python, then PATH.
rem "conda activate" is not used: it fails silently inside a double-clicked batch file
rem and then runs the wrong python. Each candidate is tested against the whole set the
rem pipeline imports, not just totalsegmentator - Slicer ships TotalSegmentator but not
rem dicom2nifti or pynrrd, so a narrower test would pick a python that fails later.
setlocal enabledelayedexpansion
set PY=
set NEEDS=import totalsegmentator,pydicom,dicom2nifti,nibabel,scipy,matplotlib,pandas,nrrd

for %%P in (
  "miniconda\python.exe"
  "%USERPROFILE%\anaconda3\envs\segmentator\python.exe"
  "%USERPROFILE%\miniconda3\envs\segmentator\python.exe"
  "%LOCALAPPDATA%\anaconda3\envs\segmentator\python.exe"
  "%LOCALAPPDATA%\miniconda3\envs\segmentator\python.exe"
  "C:\ProgramData\anaconda3\envs\segmentator\python.exe"
  "C:\ProgramData\miniconda3\envs\segmentator\python.exe"
  ".venv\Scripts\python.exe"
) do if not defined PY if exist %%P (
  %%P -c "!NEEDS!" >nul 2>&1 && set PY=%%P
)

for /d %%D in ("%LOCALAPPDATA%\slicer.org\Slicer *") do if not defined PY (
  if exist "%%D\bin\PythonSlicer.exe" (
    "%%D\bin\PythonSlicer.exe" -c "!NEEDS!" >nul 2>&1 && set PY="%%D\bin\PythonSlicer.exe"
  )
)

if not defined PY (
  for /f "delims=" %%W in ('where python 2^>nul') do if not defined PY (
    "%%W" -c "!NEEDS!" >nul 2>&1 && set PY="%%W"
  )
)

if not defined PY (
  echo.
  echo   Could not find a Python with the required packages.
  echo.
  echo   Open Anaconda Prompt and run:
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
