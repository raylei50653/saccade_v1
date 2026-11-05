@echo off
rem Builds repro_getplane.cpp as a native Windows executable and records every
rem configuration in .\capture. Run from a directory holding the source and headers.
setlocal
cd /d "%~dp0"
set VSCMD_SKIP_SENDTELEMETRY=1
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "SAN=%CUDA%\compute-sanitizer\compute-sanitizer.exe"
set "MEMCHECK=--tool memcheck --print-limit 0 --error-exitcode 99"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
if exist capture rmdir /s /q capture
mkdir capture
cl /nologo /std:c++17 /O2 /EHsc /W3 /I. /I"%CUDA%\include" repro_getplane.cpp /link /LIBPATH:"%CUDA%\lib\x64" cuda.lib > build.log 2>&1
if errorlevel 1 (type build.log & exit /b 1)

repro_getplane.exe 2 > capture\win-plain-execute-2.log 2>&1
>> capture\win-plain-execute-2.log echo exit=%errorlevel%
"%SAN%" %MEMCHECK% --report-api-errors explicit repro_getplane.exe 0 > capture\win-explicit-init-only.log 2>&1
>> capture\win-explicit-init-only.log echo exit=%errorlevel%
"%SAN%" %MEMCHECK% --report-api-errors explicit repro_getplane.exe 1 > capture\win-explicit-execute-1.log 2>&1
>> capture\win-explicit-execute-1.log echo exit=%errorlevel%
"%SAN%" %MEMCHECK% --report-api-errors explicit repro_getplane.exe 2 > capture\win-explicit-execute-2.log 2>&1
>> capture\win-explicit-execute-2.log echo exit=%errorlevel%
"%SAN%" %MEMCHECK% --report-api-errors explicit repro_getplane.exe 30 > capture\win-explicit-execute-30.log 2>&1
>> capture\win-explicit-execute-30.log echo exit=%errorlevel%
"%SAN%" %MEMCHECK% --report-api-errors extended repro_getplane.exe 2 > capture\win-extended-execute-2.log 2>&1
>> capture\win-extended-execute-2.log echo exit=%errorlevel%
"%SAN%" %MEMCHECK% --report-api-errors no repro_getplane.exe 2 > capture\win-no-report-execute-2.log 2>&1
>> capture\win-no-report-execute-2.log echo exit=%errorlevel%

> capture\windows-env.log (
  nvidia-smi --query-gpu=name,driver_version --format=csv
  "%SAN%" --version
  cl 2>&1 | findstr /c:"Version"
  powershell -NoProfile -Command "[Environment]::OSVersion.VersionString; foreach($f in 'nvofapi64.dll','nvcuvid.dll','nvcuda.dll'){ (Get-FileHash -Algorithm SHA256 (Join-Path $env:SystemRoot ('System32\' + $f))).Hash.ToLower() + '  ' + $f }; (Get-FileHash -Algorithm SHA256 repro_getplane.cpp).Hash.ToLower() + '  repro_getplane.cpp'"
)
exit /b 0
