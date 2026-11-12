#!/usr/bin/env bash
# Functions below are invoked by signal/EXIT traps.
# shellcheck disable=SC2329
set -euo pipefail
# Give asynchronous jobs normal SIGINT handling and their own process groups.
set -m

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 SOURCE_VIDEO [ENGINE]" >&2
    exit 1
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_video=$(realpath -- "$1")
engine=${2:-/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine}
track="$root/build/stream-view/track_stream"
listen=${RTSP_LISTEN:-127.0.0.1:8554}
# A wildcard listen address is not a client destination.
host=$listen
case $listen in
    0.0.0.0:*) host="127.0.0.1:${listen##*:}" ;;
    :*) host="127.0.0.1$listen" ;;
esac
base="rtsp://$host"
port_open() {
    # The child bash expands its own positional parameters.
    # shellcheck disable=SC2016
    timeout --signal=TERM --kill-after=1s 1s bash -c 'exec 3<>/dev/tcp/"$1"/"$2"' _ \
        "${host%:*}" "${host##*:}" >/dev/null 2>&1
}
[[ -f "$source_video" && -f "$engine" && -x "$track" ]]
for tool in mediamtx ffmpeg ffprobe timeout; do command -v "$tool" >/dev/null; done
if port_open; then echo "RTSP listen address is already in use" >&2; exit 1; fi
run="$root/build/stream-view-demo/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p -- "$(dirname -- "$run")"
mkdir -- "$run"
cp -- "$root/scripts/stream_view_demo.mediamtx.yml" "$run/mediamtx.yml"
tracker='' publisher='' server=''
tracker_rc=0
cleanup() {
    trap '' INT TERM
    # Leave input and RTSP server alive while the tracker drains and closes view.
    if [[ -n $tracker ]]; then
        kill -INT "$tracker" 2>/dev/null || true
        if wait "$tracker"; then tracker_rc=0; else tracker_rc=$?; fi
        printf '%s\n' "$tracker_rc" > "$run/track_stream.exit"
        tracker=
    fi
    for pid in "$publisher" "$server"; do
        [[ -z $pid ]] && continue
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    publisher='' server=''
}
on_signal() { cleanup; exit "$1"; }
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap cleanup EXIT

MTX_RTSPADDRESS="$listen" timeout --foreground --signal=TERM --kill-after=10s 12h \
    mediamtx "$run/mediamtx.yml" > "$run/mediamtx.log" 2>&1 &
server=$!
for ((attempt=0; attempt<50; ++attempt)); do
    kill -0 "$server" 2>/dev/null || break
    if port_open; then break; fi
    sleep 0.1
done
if ! kill -0 "$server" 2>/dev/null || ! port_open; then
    echo "MediaMTX did not start; logs: $run" >&2
    exit 1
fi
timeout --foreground --signal=TERM --kill-after=10s 12h \
    ffmpeg -hide_banner -loglevel warning -nostdin -re -stream_loop -1 -i "$source_video" \
    -c copy -an -f rtsp -rtsp_transport tcp "$base/in" > "$run/publisher.log" 2>&1 &
publisher=$!
ready=0
for ((attempt=0; attempt<60; ++attempt)); do
    if ! kill -0 "$server" 2>/dev/null || ! kill -0 "$publisher" 2>/dev/null; then break; fi
    if timeout --signal=TERM --kill-after=1s 2s ffprobe -v error -rtsp_transport tcp \
        -nofind_stream_info -show_entries stream=codec_name \
        -of csv=p=0 "$base/in" > "$run/probe.log" 2>&1; then
        ready=1
        break
    fi
    sleep 0.2
done
if [[ $ready != 1 ]]; then echo "demo input did not become readable; logs: $run" >&2; exit 1; fi
printf 'Outputs: %s\n' "$run"
printf 'WSLg: ffplay -rtsp_transport tcp %s/tracked\n' "$base"
printf 'Windows VLC (WSL2 NAT localhost forwarding): rtsp://localhost:%s/tracked\n' "${listen##*:}"
timeout --foreground --signal=INT --kill-after=15s 12h \
    "$track" "$engine" "$base/in" "$run/tracks.txt" --packets "$run/packets.bin" \
    --timeout-ms 15000 --view "$base/tracked" > "$run/track_stream.stdout" 2> "$run/track_stream.stderr" &
tracker=$!
if wait "$tracker"; then tracker_rc=0; else tracker_rc=$?; fi
tracker=
printf '%s\n' "$tracker_rc" > "$run/track_stream.exit"
exit "$tracker_rc"
