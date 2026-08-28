@echo off
rem Double-click this to open the segmentation interface. If this computer already has
rem a Python with the segmentation packages it uses that; if one is close, it offers to
rem add the few packages it lacks.
cd /d "%~dp0"
setlocal

echo   Looking for the Python this tool needs...
call "%~dp0find_python.bat"

if defined FOUND_PY goto :run

rem An environment that already has TotalSegmentator is missing only small packages -
rem seconds of download, against a fresh multi-gigabyte install of everything. Offered
rem rather than done, because it writes into an environment used for other work.
if not defined PARTIAL_PY goto :nothing

rem pip names differ from import names in one place, so the missing module list cannot
rem simply be handed to pip.
set "ADD=%PARTIAL_MISSING:nrrd=pynrrd%"

echo.
echo   This computer already has most of what is needed:
echo     %PARTIAL_PY%
echo   It only needs: %ADD%
echo.
echo   Adding those takes about a minute.
echo.
set "ANSWER="
set /p "ANSWER=  Add the missing packages to that Python? [Y/n] "
if /i "%ANSWER%"=="n" goto :nothing
if /i "%ANSWER%"=="no" goto :nothing

echo.
echo   Installing: %ADD%
"%PARTIAL_PY%" -m pip install %ADD%
if errorlevel 1 (
  echo.
  echo   Those packages would not install. Nothing else was changed.
  echo.
  pause
  exit /b 1
)
set "FOUND_PY=%PARTIAL_PY%"

:run
echo   Using: %FOUND_PY%
echo   Starting the interface... a browser window will open.
"%FOUND_PY%" ct_gui.py --open
echo.
echo The server has stopped.
pause
exit /b 0

:nothing
echo.
echo   Could not find a Python with the required packages.
echo.
echo   Open Anaconda Prompt and run:
echo       conda create -n segmentator python=3.10
echo       conda activate segmentator
echo       pip install pydicom dicom2nifti nibabel numpy scipy matplotlib pandas pynrrd totalsegmentator torch torchvision torchaudio
echo.
echo   Then double-click this file again - it will find that environment on its own.
echo.
pause
exit /b 1
