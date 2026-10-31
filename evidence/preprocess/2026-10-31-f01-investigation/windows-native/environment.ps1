$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
Set-Location -LiteralPath $PSScriptRoot
$info = [ordered]@{
 time = (Get-Date).ToString('o')
 os = [Environment]::OSVersion.VersionString
 powershell = $PSVersionTable.PSVersion.ToString()
 userProfile = $env:USERPROFILE
 commands = @(Get-Command python.exe,nvidia-smi.exe,nvcc.exe,compute-sanitizer.exe -ErrorAction SilentlyContinue | Select-Object Name,Source)
 vs = (& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -all -products '*' -format json | Out-String)
}
$info | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 capture\environment-before.json
& nvidia-smi.exe | Out-File -Encoding utf8 capture\nvidia-smi-before.log
$LASTEXITCODE | Out-File -Encoding ascii capture\nvidia-smi-before.exit
Get-ChildItem -LiteralPath . -File | Get-FileHash -Algorithm SHA256 | Format-List | Out-File -Encoding utf8 capture\input-hashes.log
Write-Output ('Native root: ' + $PSScriptRoot)
Get-Content capture\nvidia-smi-before.log
