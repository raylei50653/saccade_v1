flock -w 60 /home/ray/.cache/saccade-v1/gpu.lock timeout --signal=TERM --kill-after=10s 600 ctest --test-dir build/f03-scan-20261104/candidate --output-on-failure -j 1
