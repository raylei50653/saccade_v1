$ErrorActionPreference = 'Stop'
$payload = @'
$ErrorActionPreference = 'Stop'
try {
    $key = 'HKLM:\SOFTWARE\NVIDIA Corporation\GPUDebugger'
    if (-not (Test-Path -LiteralPath $key)) {
        New-Item -Path $key | Out-Null
    }
    New-ItemProperty -Path $key -Name EnableInterface -PropertyType DWord -Value 1 -Force | Out-Null
    exit 0
} catch {
    Write-Error $_
    exit 1
}
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
try {
    $process = Start-Process -FilePath "$PSHOME\powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList @('-NoProfile', '-EncodedCommand', $encoded)
    Write-Output "elevated_registry_update_exit=$($process.ExitCode)"
    exit $process.ExitCode
} catch {
    Write-Error $_
    exit 1
}
