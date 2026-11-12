# PP22 evaluation

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uid_vid_00008.mp4 | 798 | 31.732 | 50.506 | 20.413 | 40.228 | 50.497 | 146 | 6.815 | 6.791 | 7.031 | 7.358 | 5.162 | 193.739 |  |
| COMBINED | 798 | 31.732 | 50.506 | 20.413 | 40.228 | 50.497 | 146 |  |  |  |  |  |  |  |

FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.
It is neither per-frame latency nor live-stream or camera FPS.
Clocks are not locked.
