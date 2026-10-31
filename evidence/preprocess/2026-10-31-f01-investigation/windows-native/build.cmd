@echo off
set VSCMD_SKIP_SENDTELEMETRY=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"%~dp0cuda\bin\nvcc.exe" -std=c++20 -O2 -lineinfo -arch=sm_120 -I"%~dp0cuda\include" "%~dp0repro.cu" -L"%~dp0cuda\lib\x64" -lnvjpeg -o "%~dp0repro.exe"
exit /b %errorlevel%