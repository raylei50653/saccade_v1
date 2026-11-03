@echo off
set VSCMD_SKIP_SENDTELEMETRY=1
set CUDA=%LOCALAPPDATA%\Temp\saccade-f01-native-20261031T111232\cuda
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
"%CUDA%\bin\nvcc.exe" --version
"%CUDA%\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_120 -I"%CUDA%\include" "%~dp0repro_order.cu" -L"%CUDA%\lib\x64" -lnvjpeg -o "%~dp0repro_order.exe"
exit /b %errorlevel%
