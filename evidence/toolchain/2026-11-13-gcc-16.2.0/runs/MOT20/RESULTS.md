# MOT20 evaluation

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT20-01 | 429 | 50.586 | 56.904 | 45.413 | 63.577 | 70.891 | 147 | 6.872 | 6.859 | 7.103 | 7.299 | 5.142 | 194.463 |  |
| COMBINED | 429 | 50.586 | 56.904 | 45.413 | 63.577 | 70.891 | 147 |  |  |  |  |  |  |  |

FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.
It is neither per-frame latency nor live-stream or camera FPS.
Clocks are not locked.
