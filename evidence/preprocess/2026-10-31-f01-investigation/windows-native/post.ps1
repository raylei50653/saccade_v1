$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
Set-Location -LiteralPath $PSScriptRoot
$paths=@('repro.cu','repro.exe','run.ps1','baseline.jpg','baseline.rgb','nvjpeg64_13.dll','cudart64_13.dll')
$items=foreach($path in $paths){$item=Get-Item -LiteralPath $path;[ordered]@{path=$item.FullName;bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLower()}}
$items | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 capture\post-artifact-identity.json
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture | ConvertTo-Json | Set-Content -Encoding utf8 capture\windows-os.json
$dumpbin=Get-ChildItem -Path 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe' | Sort-Object FullName -Descending | Select-Object -First 1
@{file=$dumpbin.FullName;arguments=@('/DEPENDENTS',"$PSScriptRoot\repro.exe")} | ConvertTo-Json | Set-Content -Encoding utf8 capture\pe-dependencies.command.json
& $dumpbin.FullName /DEPENDENTS .\repro.exe 2>&1 | Out-File -Encoding utf8 capture\pe-dependencies.log
$code=$LASTEXITCODE
[IO.File]::WriteAllText("$PSScriptRoot\capture\pe-dependencies.exit","$code`n")
if($code -ne 0){throw 'dumpbin failed'}
Get-Date -Format o | Out-File -Encoding utf8 capture\finished.txt
Write-Output 'Post-run identities and PE dependencies recorded'
