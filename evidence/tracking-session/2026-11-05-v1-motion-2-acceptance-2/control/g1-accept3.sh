#!/usr/bin/env bash
# Formal §20.8 acceptance from the clean committed branch, with crash-safe
# telemetry: each sample is fsynced so a machine reset cannot lose it.
set -u
W=/home/ray/developer/ai/saccade_v1-worktrees/g1-ofa
C=$W/build/acceptance-3-control
cd "$W" || exit 2
mkdir -p "$C"
git rev-parse HEAD > "$C/source.txt"
git status --porcelain > "$C/source-status.txt"
date -Is > "$C/started.txt"

gpu_log() {
  while :; do
    nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.gr,clocks.mem,utilization.gpu,memory.used,clocks_event_reasons.active,pstate \
      --format=csv,noheader >> "$C/gpu-telemetry.csv" 2>&1
    sync "$C/gpu-telemetry.csv"
    sleep 5
  done
}
# One long-lived PowerShell, bounded so it cannot outlive the run by much.
power_log() {
  powershell.exe -NoProfile -Command 'for($i=0;$i -lt 1200;$i++){ $b=Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus; "{0:s},{1},{2},{3},{4},{5}" -f (Get-Date),$b.PowerOnline,$b.Charging,$b.Discharging,$b.DischargeRate,$b.RemainingCapacity; Start-Sleep 15 }' 2>/dev/null |
    while IFS= read -r line; do
      printf '%s\n' "${line%$'\r'}" >> "$C/power-telemetry.csv"
      sync "$C/power-telemetry.csv"
    done
}
gpu_log & gpu_pid=$!
power_log & power_pid=$!

flock ~/.cache/saccade-v1/gpu.lock timeout -k 10s 14400s \
  scripts/run_tracking_session_acceptance.sh build/v1-motion-2-acceptance-3 > "$C/run.log" 2>&1
status=$?

kill "$gpu_pid" "$power_pid" 2>/dev/null
pkill -f 'root/wmi -ClassName BatteryStatus' 2>/dev/null
printf 'exit=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$C/status.txt"
sync "$C/status.txt"
exit "$status"
