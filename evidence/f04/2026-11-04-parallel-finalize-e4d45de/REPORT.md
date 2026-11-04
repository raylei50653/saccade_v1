# 平行 finalize 驗收（第二輪）：e4d45de

日期：2026-11-04。規格：[POSTPROCESS §5](../../../docs/POSTPROCESS.md#5-平行-finalize2026-11-04-實作前固定)。受測 source：`e4d45de95b844d339dec63e8e6658e58077fc2cd`，乾淨工作樹。GPU：RTX 5070 Ti Laptop（WSL2，driver 617.42）。所有 GPU 步驟在 `flock ~/.cache/saccade-v1/gpu.lock` 下依序執行。

**結論：§5.7 全部 PASS。** finalize 在 HR MOT17-10 trace 平均 50.4 µs，低於 100 µs 門檻；所有輸出與改動前逐位元相同。

第一輪（`c050c84`）正確性全過但速度門檻 FAIL（116.4 µs），見 [第一輪報告](../2026-11-04-parallel-finalize-c050c84/REPORT.md)。本輪的改動（`e4d45de`）只換 finalize 的執行方式：CTA 改為 1,024 threads，每個 warp 以兩次 ballot 組出一個 64-bit 的 `fail`／`over` word；greedy 判定改用五個放在暫存器的 kept words，每列的 mask 載入不再依賴 kept 的前後鏈。`finalize_strategy=cta-bitmask-greedy-v2`；56 registers、0 local bytes、39,072 bytes static shared memory；workspace 仍為 698,432 bytes，每次 enqueue 仍是五個 kernels。

## 結果

| §5.7 步驟 | 結果 | 證據 |
| --- | --- | --- |
| 1. CPU CTest | PASS，16/16 | [logs/1-cpu.log](logs/1-cpu.log) |
| 2. F04 驗收 | PASS：F04 CTest 5/5（含 HR 與六份 replay）；完整 suite 37,382 checks；memcheck／initcheck／synccheck 0 errors，racecheck 0 hazards，各 37,077 checks；四個正控制 exit 99 並有預期診斷 | [f04-acceptance/](f04-acceptance/) |
| 3. Mutations | 15/15 DETECTED，診斷皆為預期 | [mutations/results.tsv](mutations/results.tsv) |
| 4. Tracking session 驗收 | PASS：CPU 16/16、CUDA 59/59；§14.5 預期 hash 不變；kernel／session sanitizer 與四個正控制通過 | [tracking-acceptance/](tracking-acceptance/) |
| 5. MOT17 7 序列 | 守門 PASS（身分鏈、`track_offline` == `track_sequence`、既有 hash、stream 位置）；對 [M4 outputs.sha256](../../m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256) 21/21 OK（7 個 SCDR、7 個 MOT、7 個 offline MOT）。未執行計分，未讀 `gt/` | [mot17-run/](mot17-run/)、[logs/5-m4-hashes.log](logs/5-m4-hashes.log) |
| 6. 速度門檻 | **PASS**：finalize 平均 50.4 µs（46.6–51.9，30 次）≤ 100 µs | [nsys/f04-kernels.txt](nsys/f04-kernels.txt) |

## 速度（記錄）

nsys：`run_mot17_evaluation.sh` 對 MOT17-10 前 30 幀的 trace，命令與 M4 相同。

| 項目 | M4（cad5298） | 第一輪（c050c84） | 本輪（e4d45de） |
| --- | ---: | ---: | ---: |
| F04 finalize | 1,539 µs | 116.4 µs | 50.4 µs |
| F04 五個 kernels 合計 | 約 1,589 µs | 約 170 µs | 約 105 µs |
| D stream GPU 工作（整段） | 205.56 ms | 172.79 ms | 171.33 ms |
| D stream 每幀（÷31 次 enqueue） | 6.63 ms | 5.57 ms | 5.53 ms |

`postprocess_benchmark`（六份 F03 dumps、32 × 7、round 平均延遲中位數，µs），三個 binary 在同一時段先後執行；三者的輸出 SHA-256 全同：

| 輸入 | rows | 原版 e14c94c graph／plain | 第一輪 graph／plain | 本輪 graph／plain |
| --- | ---: | ---: | ---: | ---: |
| f03-0 | 237 | 943.6／956.0 | 120.9／133.0 | 80.8／96.5 |
| f03-1 | 246 | 922.6／931.8 | 123.5／135.0 | 82.5／93.4 |
| f03-2 | 285 | 1,215.6／1,226.6 | 139.2／149.7 | 90.2／101.3 |
| f03-3 | 7 | 40.7／76.1 | 39.1／63.1 | 41.3／74.1 |
| f03-4 | 5 | 39.5／69.1 | 39.0／72.7 | 41.0／72.3 |
| f03-5 | 3 | 37.2／61.6 | 37.5／66.9 | 40.8／73.0 |

真實輸入（237–285 rows）的完整 F04 快 11.2–13.5 倍（graph）。只有 3–7 個框的合成輸入（graph）比原版慢 0.6–3.6 µs，是 1,024-thread CTA 的固定成本。

`track_sequence` pipelined（ms／幀，沒有控制 clock）：

| 序列 | M4 | 第一輪 | 本輪 |
| --- | ---: | ---: | ---: |
| MOT17-02 | 8.509 | 7.152 | 6.987 |
| MOT17-04 | 8.329 | 7.124 | 6.914 |
| MOT17-05 | 7.363 | 7.077 | 6.789 |
| MOT17-09 | 7.238 | 7.530 | 6.940 |
| MOT17-10 | 8.459 | 7.472 | 6.957 |
| MOT17-11 | 8.125 | 7.002 | 6.910 |
| MOT17-13 | 8.340 | 6.976 | 6.918 |

## 範圍

本驗收涵蓋 raw predictions → `DetectionCandidates`（F04 API、六份 T01 replay）、部署中的 HR FP16 tracking session（§14.5 項目），以及 7 個序列的完整輸出 bytes。輸出與 M4 相同，所以 M4 的 MOT17-02 指標不受影響，未重新計分。跨 GPU、driver 或 toolkit 的結果不在範圍內。

Codex 的開發紀錄：停在 workspace 規格錯誤的那次見 [2026-11-04-parallel-finalize-6699091](../2026-11-04-parallel-finalize-6699091/REPORT.md)（含改動前的 benchmark 與新 cases 在原 finalize 上的通過紀錄）。第二輪的分段計時（暫時加入的 instrumentation，未 commit）：pair bitmasks 60.5 → 19.8 µs、greedy 35.1 → 24.5 µs，其餘階段合計約 3 µs。

## 未保存

nsys 的 `track.nsys-rep` 與 `track.sqlite` 只記 SHA-256（[nsys/large-files.sha256](nsys/large-files.sha256)），檔案留在 worktree 的 `build/f04-eval2/nsys/`。

從本目錄執行 `sha256sum -c SHA256SUMS` 可核對所有封存檔案。
