flock -w 300 /home/ray/.cache/saccade-v1/gpu.lock timeout --signal=TERM --kill-after=10s 180 build/f03-llama-20261104/scan_benchmark build/f03-llama-20261104/primitive-verify --verify-only
