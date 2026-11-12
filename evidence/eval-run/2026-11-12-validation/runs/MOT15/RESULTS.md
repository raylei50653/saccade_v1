# MOT15 evaluation

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADL-Rundle-6 | 525 | 49.640 | 60.377 | 40.907 | 61.868 | 71.252 | 21 | 6.510 | 6.485 | 6.717 | 7.053 | 5.053 | 197.888 |  |
| COMBINED | 525 | 49.640 | 60.377 | 40.907 | 61.868 | 71.252 | 21 |  |  |  |  |  |  |  |

FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.
It is neither per-frame latency nor live-stream or camera FPS.
Clocks are not locked.
