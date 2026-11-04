flock -w 600 /home/ray/.cache/saccade-v1/gpu.lock timeout --signal=TERM --kill-after=10s 180 python3 build/f03-backbone-head-20261104/context_confirm.py
