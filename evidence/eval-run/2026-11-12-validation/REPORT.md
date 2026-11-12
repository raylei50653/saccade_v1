# 單指令品質與速度 runner 驗證（2026-11-12，run 2）

**結果：PASS；五個 benchmark、七個序列完成 G1–G5 與 DanceTrack link gate。** 本報告的結果、gates、速度與 telemetry 均只使用 run 2；程式提交為 `b32abf338c1573e25b45b183d3734de5e76de52f`，五組資料皆由同一組重建 binaries 產生。

## 範圍與身分

- GPU 驗證（UTC）：`2026-11-12T15:32:32.490435+00:00` 至完整驗證結束 `2026-11-12T15:35:01.042528+00:00`。
- 前置輸入檢查 19/19 通過：engine hash、driver 621.14、七組 GT／seqinfo 共十四個 hashes、frozen per-sequence CSV hash，以及 eval interpreter 的 NumPy／SciPy imports。詳見 `receipts/preflight.json`。
- 以 run 1 的完整 configure 參數重新 configure／build `build/eval-run`，Release、SM 120、`--parallel 2`，只建 `track_sequence`、`tracking_benchmark`。相對 `9b0d14f8` 的 C++／CUDA sources 及指定凍結目錄差異為空（`receipts/source-control.json`）。
- 五次 runner 的 git porcelain 均為空；七個 capture、五十六個 speed sessions，共 63 份 native identity records，全部 `source_commit` 等於程式提交且 `source_dirty=false`。五份 RESULTS 沒有 dirty／mismatch warning；五組 binary／engine hashes 完全相同，見 `parity.json` 的 `build_identity`。
- GPU 為 NVIDIA GeForce RTX 5070 Ti Laptop GPU，driver 621.14；engine SHA-256 為 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`，未重建 engine。
- 固定 `hr-1280x768`、NSA／measurement box OFF；capture 不輸出 packets／motion。五組 GPU jobs 循序執行，GPU phase 持有 gpu.lock、每個 native 呼叫經 `timeout -k 30s 3600s`、每秒 telemetry；評分在釋放 lock 後執行。

## Gates（僅 run 2）

| Gate | 結果 | 數量與規則 |
| --- | --- | --- |
| G1 MOT bytes | PASS | 7/7 MOT SHA-256 等於 primary156 main |
| G2 逐序列分數 | PASS | 7/7 序列、42/42 指標完全相同；HOTA／DetA／AssA／IDF1／MOTA parsed floats exact equality，IDSW integer equality |
| G3 既有 scorer parity | PASS | 2/2 組、12/12 指標；eval_mot17.sh 評分三份 MOT17 複本，eval_pp22.sh 評分 PP22 複本；COMBINED 依既有輸出精度（品質五位有效數字、IDSW 整數）完全相同 |
| G4 速度 | PASS | 7/7 summary status PASS；56/56 fresh sessions、56/56 tracker config checks PASS；每序列 480 single samples，合計 3360；沒有 short-sequence skip |
| G5 CPU 測試 | PASS | eval_run_tests.py 12/12、evaluation_tools_tests.py 12/12；stub GPU 工具＋真實 repository TrackEval worker，沒有 GPU 推論 |
| DanceTrack link gate | PASS | 1203/1203 symlinks，以雙方逐 frame 的 readlink -f 結果比對，全部指向相同實體影像 |

DanceTrack G1 重現 `503f5596d5b17ea335bfae78027e87e1a14cea76de14916aaf08e49578c94c59`。原始八位檔名影像透過 `build/eval-run-val2-DanceTrack/inputs/dancetrack0004` 的六位 symlink view 供兩個 native 工具讀取，`seqinfo.ini` 複製原檔；GT、評分 seqinfo 與 shared-parent check 仍使用原始序列。`environment.json` 記錄 original path、view path、source_digits 8、link_count 1203。逐 frame 雙方連結與解析結果保留於 `receipts/dancetrack-links.tsv`；primary156 view 僅讀取、未修改。

所有 COMBINED 均直接來自該組 TrackEval detailed CSV，沒有平均各序列品質。G3 控制工具透過 `SACCADE_DATASET_ROOT=/home/ray/developer/ai/dataset` 指定資料位置，既有 scripts byte-unchanged；控制輸出見 `controls/` 與 `receipts/parity-*.log.gz`。

## 五組 RESULTS

品質為百分比（IDSW 除外）；CSV／JSON 保留完整精度，以下為 runner 的三位小數表格。

### MOT17

[RESULTS](runs/MOT17/RESULTS.md) · [CSV](runs/MOT17/results.csv) · [JSON](runs/MOT17/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-10-SDP | 654 | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 | 6.697 | 6.672 | 6.984 | 7.145 | 5.092 | 196.387 |  |
| MOT17-02-SDP | 600 | 40.565 | 40.896 | 40.566 | 49.154 | 46.784 | 102 | 6.768 | 6.720 | 7.045 | 7.442 | 5.149 | 194.209 |  |
| MOT17-04-SDP | 1050 | 74.802 | 83.169 | 67.376 | 80.578 | 92.464 | 71 | 6.840 | 6.823 | 7.065 | 7.288 | 5.113 | 195.563 |  |
| COMBINED | 2304 | 64.256 | 67.382 | 61.768 | 71.516 | 77.559 | 229 |  |  |  |  |  |  |  |

### MOT20

[RESULTS](runs/MOT20/RESULTS.md) · [CSV](runs/MOT20/results.csv) · [JSON](runs/MOT20/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT20-01 | 429 | 50.586 | 56.904 | 45.413 | 63.577 | 70.891 | 147 | 6.887 | 6.865 | 7.122 | 7.346 | 5.133 | 194.811 |  |
| COMBINED | 429 | 50.586 | 56.904 | 45.413 | 63.577 | 70.891 | 147 |  |  |  |  |  |  |  |

### MOT15

[RESULTS](runs/MOT15/RESULTS.md) · [CSV](runs/MOT15/results.csv) · [JSON](runs/MOT15/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADL-Rundle-6 | 525 | 49.640 | 60.377 | 40.907 | 61.868 | 71.252 | 21 | 6.510 | 6.485 | 6.717 | 7.053 | 5.053 | 197.888 |  |
| COMBINED | 525 | 49.640 | 60.377 | 40.907 | 61.868 | 71.252 | 21 |  |  |  |  |  |  |  |

### DanceTrack

[RESULTS](runs/DanceTrack/RESULTS.md) · [CSV](runs/DanceTrack/results.csv) · [JSON](runs/DanceTrack/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 1203 | 29.921 | 59.442 | 15.164 | 32.296 | 74.335 | 44 | 6.518 | 6.457 | 6.758 | 8.489 | 5.116 | 195.465 |  |
| COMBINED | 1203 | 29.921 | 59.442 | 15.164 | 32.296 | 74.335 | 44 |  |  |  |  |  |  |  |

### PP22

[RESULTS](runs/PP22/RESULTS.md) · [CSV](runs/PP22/results.csv) · [JSON](runs/PP22/results.json)

| sequence | frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | single_mean_ms | single_p50_ms | single_p95_ms | single_p99_ms | queued_ms_per_frame | queued_fps | speed_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uid_vid_00008.mp4 | 798 | 31.732 | 50.506 | 20.413 | 40.228 | 50.497 | 146 | 6.815 | 6.791 | 7.031 | 7.358 | 5.162 | 193.739 |  |
| COMBINED | 798 | 31.732 | 50.506 | 20.413 | 40.228 | 50.497 | 146 |  |  |  |  |  |  |  |

## MOT17-10 速度對照

百分比為 `(run 2 / 基準 − 1) × 100`；此項為資訊對照，不作效能 gate，也沒有為追逐速度差異重跑。

| 指標 | g1-live | attempt 1 | run 2 | 相對 g1-live | 相對 attempt 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_mean_ms | 6.712865 | 6.696337 | 6.696704 | -0.241% | +0.005% |
| single_p95_ms | 6.981673 | 6.903525 | 6.984476 | +0.040% | +1.173% |
| single_p99_ms | 7.228440 | 7.157335 | 7.145472 | -1.148% | -0.166% |
| queued_fps | 194.620666 | 195.850553 | 196.386685 | +0.907% | +0.274% |

本輪與 attempt 1 均取 native benchmark summary；g1-live 的指定 single 基準採 host timestamp instrumentation 的 enqueue→receive 返回時間，與 summary 有少量計時邊界差異。clocks 未鎖定，此對照不代表受控加速。

## Telemetry

一秒取樣包含初始化、warmup、capture、處理與清理，不能當作穩態 kernel／Tensor Core／OFA utilization。

| benchmark | samples | GPU utilization mean / max (%) | GPU memory max (MiB) | temperature max (°C) | SM clock min / max (MHz) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOT17 | 52 | 58.4 / 100 | 495 | 73 | 0 / 2565 |
| MOT20 | 16 | 50.3 / 100 | 497 | 73 | 180 / 8895 |
| MOT15 | 16 | 48.4 / 100 | 495 | 73 | 1080 / 2640 |
| DanceTrack | 19 | 59.5 / 100 | 495 | 75 | 0 / 2580 |
| PP22 | 18 | 54.7 / 100 | 495 | 75 | 180 / 2880 |

## Attempt 1 歷史與規格原因

Run 1 在指定 DanceTrack 原始目錄只有八位檔名、而 native 工具要求六位檔名時依規則停止；當時規格沒有授權 image view，這是規格缺口，不是 runner 缺陷。當時只完成五個序列的 G1／G2／G4，G3 與 PP22 未執行。本輪依補充規格新增與 primary156 相同的 symlink view，沒有變更 native sources。Run 1 的全部 run records、失敗／停止收據、報告、parity 與原始 SHA256SUMS 均原樣移到 `attempt-1/`；不併入本輪 gates 或結果。

## 自有缺陷與修正

本輪 runner、worker、測試沒有觸發需修正的程式失敗，沒有 GPU 重跑或放寬比較。準備階段一次本機測試命令將 log 目錄建立與另一個測試輸出平行啟動，造成 shell redirection 先於目錄建立而失敗，該次測試尚未啟動；改為先建立目錄，再執行測試後通過，乾淨 HEAD 的正式 G5 也再次通過。此為本機命令編排錯誤，不影響 GPU 結果。封裝檢查另發現 Notes 原先只列 detailed CSV，但原生 summary TXT 也保留 CRLF；已補正文字，未改動工具輸出。Run 1 已修正的證據文字錯誤保留於其原始報告。

## 保管與限制

- `runs/` 的五組結果皆來自 `build/eval-run-val2-<benchmark>`；保留完整精度結果、identity、commands、telemetry、MOT gzip、resolved configs、benchmark summary／runs.csv／single CSV／session identities、scoring detailed CSV 與 gzip logs。影像連結以逐 frame TSV 和 environment 記錄保管，沒有把外部影像或 symlinks 複製入 git。
- 全部 gates 通過後，已刪除 run 1 的 `build/eval-run-val-*` 及其暫存 `build/eval-run-work`；清單見 `receipts/retired-run1-outputs.json`。run 2 build／輸出留在 worktree 的 build/，沒有提交 build trees 或 packet .bin。
- 速度只量每序列前 140 frames，20 warmup 後量 frames 21–140；single 為 480 pooled samples，queued FPS 是預載 JPEG 的完成吞吐，排除磁碟 I/O、constructor、warmup 與輸出檔案；它不是逐 frame latency、live-stream 或 camera FPS。clocks 未鎖定。
- 只驗證指定七個序列，沒有重跑 primary156 全集，也沒有進行 sanitizer、nsys、GPU CTest、training 或其他 GPU 工作。短序列 skip 另由 CPU fixture 驗證。
- evidence commit 只更動證據；binary 身分對應程式提交，量測時 HEAD 與 binary 相同。未 push、未 merge；沒有修改 main checkout 或資料集。
- `SHA256SUMS` 涵蓋本目錄除自身以外的所有檔案，包含 attempt-1 的原始 checksum；在 attempt-1/ 內另可驗證原有清單。

## Notes

1. 三個 MOT17 序列的 primary156 `command.json` 未包含於原封存 checkout；七個 capture hashes 均存在且本輪 G1 全部相同，沒有觸發 mismatch argv diff。
2. TrackEval 原生 detailed CSV 與 summary TXT 保留 CRLF bytes；一般 git diff --check 會列出行尾空白提示。程式與 Markdown whitespace check 通過，沒有為消除提示改寫原始 CSV。
