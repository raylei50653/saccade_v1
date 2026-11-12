# DanceTrack evaluation

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 1203 | 29.921 | 59.442 | 15.164 | 32.296 | 74.335 | 44 | 6.518 | 6.457 | 6.758 | 8.489 | 5.116 | 195.465 |  |
| COMBINED | 1203 | 29.921 | 59.442 | 15.164 | 32.296 | 74.335 | 44 |  |  |  |  |  |  |  |

FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.
It is neither per-frame latency nor live-stream or camera FPS.
Clocks are not locked.
