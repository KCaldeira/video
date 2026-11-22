@echo off
rem Level audio: drag audio files onto this icon (or use Send to).
rem Runs level_dropped.sh in WSL for each file; the leveled WAV, a .dawproject
rem and a plot appear beside the source file.
rem For other settings, copy this file under another name and change LEVEL_ARGS
rem (any audio_level_curve.py flags, e.g. --target-fraction 0.25 --mean-distance 2).
set LEVEL_ARGS=--target-fraction 0.5
for %%F in (%*) do (
    echo ============================================================
    echo %%~F
    wsl.exe -d Ubuntu-22.04 --cd /home/kcaldeira/video -e ./level_dropped.sh "%%~F" %LEVEL_ARGS%
    if errorlevel 1 echo *** FAILED: %%~F ***
)
pause
