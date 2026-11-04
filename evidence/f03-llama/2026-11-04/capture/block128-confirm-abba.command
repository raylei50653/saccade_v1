flock -w 300 /home/ray/.cache/saccade-v1/gpu.lock timeout --signal=TERM --kill-after=10s 180 python3 build/f03-llama-20261104/run_model_confirmation.py
