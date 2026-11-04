#!/usr/bin/env bash
set -uo pipefail
name=$1
shift
cap=/home/ray/developer/ai/saccade_v1-worktrees/f04-parallel/build/f04-logs
printf '%q ' "$@" > "$cap/$name.command"
printf '\n' >> "$cap/$name.command"
date -Is > "$cap/$name.started"
"$@" > "$cap/$name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$cap/$name.exit"
date -Is > "$cap/$name.finished"
printf '%s exit=%s\n' "$name" "$status"
tail -n 20 "$cap/$name.log"
exit "$status"
