# 平行 finalize 驗收（第一輪）：c050c84

日期：2026-11-04。規格：[POSTPROCESS §5](../../../docs/POSTPROCESS.md#5-平行-finalize2026-11-04-實作前固定)。受測 source：`c050c847b617d6832cb65ece38105957e1d23d8b`，乾淨工作樹。GPU：RTX 5070 Ti Laptop（WSL2，driver 617.42）。所有 GPU 步驟在 `flock ~/.cache/saccade-v1/gpu.lock` 下依序執行。

**結論：正確性全部 PASS；§5.7 第 6 步的速度門檻 FAIL**（finalize 平均 116.4 µs，門檻 ≤ 100 µs）。依規格不調整門檻；下一輪在同一 branch 繼續優化 finalize，並重跑整個 §5.7。

## 結果

| §5.7 步驟 | 結果 | 證據 |
| --- | --- | --- |
| 1. CPU CTest | PASS，16/16 | [logs/1-cpu.log](logs/1-cpu.log) |
| 2. F04 驗收 | PASS：F04 CTest 5/5（含 HR 與六份 replay）；完整 suite 37,382 checks；memcheck／initcheck／synccheck 0 errors，racecheck 0 hazards，各 37,077 checks；四個正控制 exit 99 並有預期診斷 | [f04-acceptance/](f04-acceptance/) |
| 3. Mutations | 15/15 DETECTED，診斷皆為預期 | [mutations/results.tsv](mutations/results.tsv) |
| 4. Tracking session 驗收 | PASS：CPU 16/16、CUDA 59/59；§14.5 預期 hash 不變；kernel／session sanitizer 與四個正控制通過 | [tracking-acceptance/](tracking-acceptance/) |
| 5. MOT17 7 序列 | 守門 PASS（身分鏈、`track_offline` == `track_sequence`、既有 hash、stream 位置）；對 [M4 outputs.sha256](../../m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256) 21/21 OK（7 個 SCDR、7 個 MOT、7 個 offline MOT）。未執行計分，未讀 `gt/` | [mot17-run/](mot17-run/)、[logs/5-m4-hashes.log](logs/5-m4-hashes.log) |
| 6. 速度門檻 | **FAIL**：finalize 平均 116.4 µs（108.5–118.9，30 次）＞ 100 µs | [nsys/f04-kernels.txt](nsys/f04-kernels.txt) |

## 速度（記錄）

nsys：`run_mot17_evaluation.sh` 對 MOT17-10 前 30 幀的 trace，命令與 M4 相同。

| 項目 | M4（cad5298） | 本輪（c050c84） |
| --- | ---: | ---: |
| F04 finalize（grid 1） | 1,539 µs（block 1） | 116.4 µs（block 256） |
| F04 五個 kernels 合計 | 約 1,589 µs | 約 170 µs |
| D stream GPU 工作（整段） | 205.56 ms | 172.79 ms |
| D stream 每幀（÷31 次 enqueue） | 6.63 ms | 5.57 ms |

trace 中其他名為 `finalize` 的 kernel（grid ≥ 30）是 selective scan 的，不屬於 F04。

`postprocess_benchmark`（六份 F03 dumps、32 × 7、round 平均延遲中位數），舊 binary（e14c94c）與新 binary 同一時段先後執行；輸出 SHA-256 兩者全同：

| 輸入 | rows | 舊 graph／plain（µs） | 新 graph／plain（µs） |
| --- | ---: | ---: | ---: |
| f03-0 | 237 | 946.1／957.4 | 134.4／144.9 |
| f03-1 | 246 | 922.1／933.7 | 137.1／148.3 |
| f03-2 | 285 | 1,215.5／1,226.8 | 154.5／164.8 |
| f03-3 | 7 | 40.9／72.0 | 42.9／67.7 |
| f03-4 | 5 | 39.4／63.6 | 42.3／80.2 |
| f03-5 | 3 | 37.2／63.2 | 41.0／72.4 |

真實輸入快 6.6–7.9 倍；只有 3–7 個框的合成輸入（graph 模式）慢 2–4 µs。

`track_sequence` pipelined（ms／幀，沒有控制 clock）：

| 序列 | M4 | 本輪 |
| --- | ---: | ---: |
| MOT17-02 | 8.509 | 7.152 |
| MOT17-04 | 8.329 | 7.124 |
| MOT17-05 | 7.363 | 7.077 |
| MOT17-09 | 7.238 | 7.530 |
| MOT17-10 | 8.459 | 7.472 |
| MOT17-11 | 8.125 | 7.002 |
| MOT17-13 | 8.340 | 6.976 |

## 本輪修補的驗收工具

`c050c84` 修正了兩個加入 HR profile（`96ade04`）後就失效、本輪才發現的問題：規格中的 workspace bytes（實際 698,432），以及 `run_postprocess_acceptance.sh` 的 sanitizer 總結格式檢查。三個失效的 mutation needle 由 `6699091` 修好。Codex 停下時的紀錄見 [2026-11-04-parallel-finalize-6699091](../2026-11-04-parallel-finalize-6699091/REPORT.md)，含改動前的 benchmark 與新 cases 在原 finalize 上的通過紀錄。

## 未保存

nsys 的 `track.nsys-rep` 與 `track.sqlite` 只記 SHA-256（[nsys/large-files.sha256](nsys/large-files.sha256)），檔案留在 worktree 的 `build/f04-eval/nsys/`。

從本目錄執行 `sha256sum -c SHA256SUMS` 可核對所有封存檔案。
