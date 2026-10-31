param([string]$Phase = 'build')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
Set-Location -LiteralPath $PSScriptRoot
$env:PATH = "$PSScriptRoot\cuda\bin;$PSScriptRoot\cuda\bin\x64;$PSScriptRoot\cuda\compute-sanitizer;" + $env:PATH
$Sanitizer = "$PSScriptRoot\cuda\compute-sanitizer\compute-sanitizer.exe"
$Nvcc = "$PSScriptRoot\cuda\bin\nvcc.exe"
$Program = "$PSScriptRoot\repro.exe"
function Invoke-Captured {
 param([string]$Name, [string]$Executable, [string[]]$NativeArgs = @(), [int]$LimitSeconds = 90)
 $base = Join-Path "$PSScriptRoot\capture" $Name
 if (Test-Path "$base.command.json") { throw "Refusing to overwrite $Name" }
 $available = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory
 if ($available -lt 4194304) { throw 'Need at least 4 GiB free physical memory' }
 $command = [ordered]@{file=$Executable;arguments=$NativeArgs;working_directory=$PSScriptRoot;path_prefix="$PSScriptRoot\cuda\bin;$PSScriptRoot\cuda\bin\x64;$PSScriptRoot\cuda\compute-sanitizer";started=(Get-Date).ToString('o');available_kib=$available;timeout_seconds=$LimitSeconds}
 $command | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "$base.command.json"
 $quoted = ($NativeArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
 $timer = [System.Diagnostics.Stopwatch]::StartNew()
 $proc = New-Object System.Diagnostics.Process
 $proc.StartInfo.FileName = $Executable
 $proc.StartInfo.Arguments = $quoted
 $proc.StartInfo.WorkingDirectory = $PSScriptRoot
 $proc.StartInfo.UseShellExecute = $false
 $proc.StartInfo.CreateNoWindow = $true
 $proc.StartInfo.RedirectStandardOutput = $true
 $proc.StartInfo.RedirectStandardError = $true
 [void]$proc.Start()
 $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
 $stderrTask = $proc.StandardError.ReadToEndAsync()
 $timedOut = -not $proc.WaitForExit($LimitSeconds*1000)
 if ($timedOut) { & taskkill.exe /PID $proc.Id /T /F | Out-Null; $proc.WaitForExit(); $status=124 } else { $proc.WaitForExit(); $status=$proc.ExitCode }
 [IO.File]::WriteAllText("$base.stdout.log",$stdoutTask.Result)
 [IO.File]::WriteAllText("$base.stderr.log",$stderrTask.Result)
 $timer.Stop()
 [IO.File]::WriteAllText("$base.exit", "$status`n")
 [ordered]@{exit=$status;timed_out=$timedOut;process_id=$proc.Id;elapsed_seconds=$timer.Elapsed.TotalSeconds;finished=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -Encoding utf8 "$base.status.json"
 # Keep original streams separately. This convenience log is not chronological.
 [IO.File]::WriteAllText("$base.log", "--- STDOUT ---`n"+[IO.File]::ReadAllText("$base.stdout.log")+"`n--- STDERR ---`n"+[IO.File]::ReadAllText("$base.stderr.log"))
 Write-Output "$Name exit=$status elapsed=$($timer.Elapsed.TotalSeconds)"
 Get-Content "$base.log" | Where-Object { $_ -match '^(nvjpeg=|result=)|ERROR SUMMARY' } | Write-Output
 return $status
}
function Run-Case {
 param([string]$Name,[string]$Backend='hw',[string]$Format='rgb',[string]$Flags='default',[string]$Preallocation='preallocate',[string]$Fill='uninit',[int]$Repeats=2,[string[]]$Tools=@('normal','memcheck','initcheck'))
 foreach ($tool in $Tools) {
  $caseName="$Name-$tool"
  $programArgs=@($Backend,$Format,$Flags,$Preallocation,$Fill,[string]$Repeats,"$PSScriptRoot\baseline.jpg","$PSScriptRoot\baseline.rgb","$PSScriptRoot\$caseName.bin")
  if ($tool -eq 'normal') { Invoke-Captured -Name $caseName -Executable $Program -NativeArgs $programArgs | Out-Host }
  else { Invoke-Captured -Name $caseName -Executable $Sanitizer -NativeArgs (@('--tool',$tool,'--error-exitcode','99','--print-limit','10',$Program)+$programArgs) | Out-Host }
 }
}
if ($Phase -eq 'build') {
 $build = @'
@echo off
set VSCMD_SKIP_SENDTELEMETRY=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"%~dp0cuda\bin\nvcc.exe" -std=c++20 -O2 -lineinfo -arch=sm_120 -I"%~dp0cuda\include" "%~dp0repro.cu" -L"%~dp0cuda\lib\x64" -lnvjpeg -o "%~dp0repro.exe"
exit /b %errorlevel%
'@
 [IO.File]::WriteAllText("$PSScriptRoot\build.cmd",$build,[Text.Encoding]::ASCII)
 Invoke-Captured -Name 'compile' -Executable $env:ComSpec -NativeArgs @('/d','/c',"$PSScriptRoot\build.cmd") -LimitSeconds 120 | Out-Host
 if ((Get-Content capture\compile.exit).Trim() -ne '0') { throw 'Compilation failed; see capture logs' }
}
elseif ($Phase -eq 'identity') {
 Invoke-Captured -Name 'nvcc-version' -Executable $Nvcc -NativeArgs @('--version') | Out-Host
 Invoke-Captured -Name 'sanitizer-version' -Executable $Sanitizer -NativeArgs @('--version') | Out-Host
 Invoke-Captured -Name 'gpu-identity' -Executable (Get-Command nvidia-smi.exe).Source -NativeArgs @('--query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap','--format=csv') | Out-Host
 $paths=@('repro.cu','repro.exe','baseline.jpg','baseline.rgb','nvjpeg64_13.dll','cudart64_13.dll','cuda\bin\nvcc.exe','cuda\compute-sanitizer\compute-sanitizer.exe','cuda\lib\x64\cudart_static.lib',"$env:windir\System32\nvcuda.dll")
 $items = foreach ($path in $paths) { $item=Get-Item -LiteralPath $path; [ordered]@{path=$item.FullName;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLower();file_version=$item.VersionInfo.FileVersion;product_version=$item.VersionInfo.ProductVersion} }
 $items | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 capture\artifact-identity.json
}
elseif ($Phase -eq 'baseline') { Run-Case -Name 'windows-rgb' }
elseif ($Phase -eq 'controls') {
 Run-Case -Name 'windows-skip-uninit' -Backend 'skip' -Repeats 1 -Tools @('memcheck','initcheck')
 Run-Case -Name 'windows-skip-zero' -Backend 'skip' -Fill 'zero' -Repeats 1 -Tools @('initcheck')
 Run-Case -Name 'windows-rgb-no-preallocate' -Preallocation 'no-preallocate' -Tools @('initcheck')
 Run-Case -Name 'windows-rgb-no-pipeline' -Flags 'no-pipeline' -Tools @('initcheck')
 Run-Case -Name 'windows-rgbi' -Format 'rgbi' -Tools @('initcheck')
 Run-Case -Name 'windows-y' -Format 'y' -Tools @('initcheck')
 Run-Case -Name 'windows-hybrid' -Backend 'hybrid' -Tools @('initcheck')
}
else { throw "Unknown phase $Phase" }
