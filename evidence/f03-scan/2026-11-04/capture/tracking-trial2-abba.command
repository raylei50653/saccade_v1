flock -w 60 /home/ray/.cache/saccade-v1/gpu.lock timeout --signal=TERM --kill-after=10s 240 python3 build/f03-scan-20261104/run_tracking_abba.py
