flock -w 60 /home/ray/.cache/saccade-v1/gpu.lock timeout --signal=TERM --kill-after=10s 180 python3 build/f03-scan-20261104/run_model_abba.py trial2
