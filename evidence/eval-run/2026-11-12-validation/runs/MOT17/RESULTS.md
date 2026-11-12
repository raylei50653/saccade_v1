# MOT17 evaluation

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-10-SDP | 654 | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 | 6.697 | 6.672 | 6.984 | 7.145 | 5.092 | 196.387 |  |
| MOT17-02-SDP | 600 | 40.565 | 40.896 | 40.566 | 49.154 | 46.784 | 102 | 6.768 | 6.720 | 7.045 | 7.442 | 5.149 | 194.209 |  |
| MOT17-04-SDP | 1050 | 74.802 | 83.169 | 67.376 | 80.578 | 92.464 | 71 | 6.840 | 6.823 | 7.065 | 7.288 | 5.113 | 195.563 |  |
| COMBINED | 2304 | 64.256 | 67.382 | 61.768 | 71.516 | 77.559 | 229 |  |  |  |  |  |  |  |

FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.
It is neither per-frame latency nor live-stream or camera FPS.
Clocks are not locked.
