@echo off
rem Called by start.bat; not meant to be run on its own. Sets three variables:
rem   FOUND_PY         a python with every package, or empty
rem   PARTIAL_PY       a python with TotalSegmentator but missing a few small packages
rem   PARTIAL_MISSING  which packages that one is missing
rem
rem No setlocal on purpose - the caller needs these back.
rem
rem "conda activate" is never used: it fails silently inside a double-clicked batch
rem file and then runs the wrong python. Candidates are tested instead, and against the
rem whole set the pipeline imports rather than totalsegmentator alone - Slicer ships
rem TotalSegmentator but not dicom2nifti or pynrrd, so a narrower test would pick a
rem python that fails later, mid-run.

set "HERE=%~dp0"
set "NEEDS_MODULES=totalsegmentator pydicom dicom2nifti nibabel numpy scipy matplotlib pandas nrrd"
set "FOUND_PY="
set "PARTIAL_PY="
set "PARTIAL_MISSING="

rem The private copy, if some earlier setup made one, then whatever env is active.
call :try "%HERE%miniconda\python.exe"
if defined CONDA_PREFIX call :try "%CONDA_PREFIX%\python.exe"

rem Anaconda, Miniconda, miniforge and mambaforge each pick a different install
rem location, and the choice is per-installer and per-machine. Someone who has one of
rem these has no reason to know which, so all the usual ones are scanned.
for %%R in (
  "%USERPROFILE%\anaconda3"
  "%USERPROFILE%\miniconda3"
  "%USERPROFILE%\miniforge3"
  "%USERPROFILE%\mambaforge"
  "%LOCALAPPDATA%\anaconda3"
  "%LOCALAPPDATA%\miniconda3"
  "%LOCALAPPDATA%\miniforge3"
  "%LOCALAPPDATA%\Continuum\anaconda3"
  "C:\ProgramData\anaconda3"
  "C:\ProgramData\miniconda3"
  "C:\anaconda3"
  "C:\miniconda3"
) do call :scan_root "%%~R"

call :try "%HERE%.venv\Scripts\python.exe"
call :try "%USERPROFILE%\.venv\Scripts\python.exe"

for /d %%D in ("%LOCALAPPDATA%\slicer.org\Slicer *") do call :try "%%~D\bin\PythonSlicer.exe"
for /d %%D in ("C:\Program Files\Slicer *") do call :try "%%~D\bin\PythonSlicer.exe"

for /f "delims=" %%W in ('where python 2^>nul') do call :try "%%~W"
for /f "delims=" %%W in ('where python3 2^>nul') do call :try "%%~W"
exit /b 0

rem Named environments before the base environment: someone who made an env for this
rem put the packages there, and base is the one you least want to write into.
:scan_root
if not exist "%~1" exit /b 0
for /d %%E in ("%~1\envs\*") do call :try "%%~E\python.exe"
call :try "%~1\python.exe"
exit /b 0

:try
if defined FOUND_PY exit /b 0
if not exist "%~1" exit /b 0
set "MISS="
rem check_env.py prints nothing when the interpreter has everything, and for /f skips
rem that blank line, so MISS staying empty is the success case.
for /f "delims=" %%M in ('""%~1" "%HERE%check_env.py" %NEEDS_MODULES%" 2^>nul') do set "MISS=%%M"
if not defined MISS (
  set "FOUND_PY=%~1"
  exit /b 0
)
if defined PARTIAL_PY exit /b 0
rem torch is the multi-gigabyte one and it arrives with totalsegmentator, so an env
rem that lacks totalsegmentator is not worth topping up - that is a full install.
echo %MISS% | findstr /C:"totalsegmentator" >nul && exit /b 0
rem Slicer's bundled python qualifies on packages, but pip-installing into an
rem application's private interpreter can break that application, so it is only ever
rem used as-is, never modified.
echo %~1 | findstr /I /C:"Slicer" >nul && exit /b 0
set "PARTIAL_PY=%~1"
set "PARTIAL_MISSING=%MISS%"
exit /b 0
