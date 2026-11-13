# MOT17 evaluation

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-10-SDP | 654 | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 | 6.633 | 6.621 | 6.870 | 7.049 | 5.104 | 195.939 |  |
| MOT17-02-SDP | 600 | 40.565 | 40.896 | 40.566 | 49.154 | 46.784 | 102 | 6.696 | 6.678 | 6.921 | 7.198 | 5.146 | 194.338 |  |
| MOT17-04-SDP | 1050 | 74.802 | 83.169 | 67.376 | 80.578 | 92.464 | 71 | 6.803 | 6.769 | 7.069 | 7.293 | 5.133 | 194.829 |  |
| COMBINED | 2304 | 64.256 | 67.382 | 61.768 | 71.516 | 77.559 | 229 |  |  |  |  |  |  |  |

FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.
It is neither per-frame latency nor live-stream or camera FPS.
Clocks are not locked.
