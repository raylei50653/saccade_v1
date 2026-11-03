@echo off
cd /d "%~dp0"
for %%C in ("hw default order 20" "hw no-pipeline order 20" "hw default control 20" "hybrid default order 20" "hw default order 1") do call :one %%~C
exit /b 0
:one
set NAME=%1-%2-%3-%4ms
.\repro_order.exe %1 %2 %3 %4 20 a.jpg b.jpg > capture\win-%NAME%.log 2>&1
>> capture\win-%NAME%.log echo exit=%errorlevel%
echo %NAME% exit=%errorlevel%
exit /b 0
