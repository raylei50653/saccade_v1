# v1-motion/2 正式驗收（2026-11-05）

**結果：未完全通過。** 依 `docs/TRACKING.md` §20.8，第 6 步完整 session memcheck 的 exit 99 記為 OPEN（[NVIDIA Bug 7048273](https://developer.nvidia.com/bugs/7048273)），其餘八步全部通過。Capture 的 [acceptance.txt](capture/acceptance.txt) 寫的是 `NOT A FULL PASS`。

G1 的輸出和實驗逐位元相同：MOT17 7/7、PersonPath22 69/69，分數也完全相等。吞吐比 main `23ade53` 慢 2.70%，上限 3%。

## Source 與環境

| 項目 | 紀錄 |
| --- | --- |
| Source | `fae5cb7c56388195b5dda0c356383b3bd9963c4f`，乾淨 worktree，[source.txt](capture/source.txt) |
| 對照組 | detached worktree `../g1-ofa-base`，main `23ade53` |
| 時間 | 2026-11-05 13:26:43 到 14:07:37（+08:00），約 41 分鐘；353 個命令的 exit 與時間都在 [capture/](capture/) |
| 執行方式 | [control/g1-accept3.sh](control/g1-accept3.sh)：在 GPU lock 與 4 小時 timeout 內跑 [驗收腳本](../../../scripts/run_tracking_session_acceptance.sh)，一次跑完 |
| 環境 | RTX 5070 Ti Laptop，driver 617.42，WSL2，CUDA 13.4.92，Compute Sanitizer 2026.3.0.0，Nsight Systems 2026.3.2，[environment.log](capture/environment.log) |

Payload 留在 `build/v1-motion-2-acceptance-3/`（Git 忽略），這裡只記 SHA-256：[hr-outputs](capture/hr-outputs.sha256)、[pp22-outputs](capture/pp22-outputs.sha256)、[outputs](capture/outputs.sha256)。

## 各步結果

| 步驟 | 結果 | 內容 |
| --- | --- | --- |
| 1. Clean source 與輸入 | PASS | 兩個引擎、44 個凍結語料、NVOF headers 與載入路徑都相符 |
| 2. Build 與測試 | PASS | CPU 18/18，CUDA 65/65；7 序列 packet 比對；§20.6 A1–A4 |
| 3. MOT17 HR 路徑 | PASS | 全部逐位元相同（見下） |
| 4. PersonPath22 與 4K | PASS | 69/69 等於 G1；5 個 4K 序列跑完 |
| 5. 計分 | PASS | 6 組都等於實驗值；4K 只記錄 |
| 6. Sanitizer | **OPEN** | 完整 session memcheck exit 99，來自 NVOF 內部；其餘全部乾淨 |
| 7. Stream 位置 | PASS | F05 在 D，F06 和 OFA 在 M；OFA 與 F03 重疊 28/29 幀 |
| 8. 吞吐 | PASS | 總計 +2.70%，上限 3% |
| 9. 證據保管 | PASS | binary 與輸入雜湊在跑前跑後相同，source 全程未變 |

逐步紀錄見 [steps.tsv](capture/steps.tsv)。353 個命令中只有 6 個非零：4 個 positive control 和 2 個完整 session memcheck，都是 exit 99，符合預期。

### 2. Build 與測試

- [CPU CTest](capture/cpu-ctest.log) 18/18，[CUDA CTest](capture/ctest.log) 65/65，後者耗時 632.63 秒。
- 7 序列 T01 的 session packet 都和 CPU reference 逐位元相同，包含 serial、pipelined 和兩個擾動 seed（[LastTest.log](capture/LastTest.log)）。
- flow kernel parity 在 s=1 和 s=2 都通過。

**§20.6 OFA 審查：** A1–A4 各 60/60，共 240 筆，違反 0 次（[ofa-trials.csv](capture/ofa-trials.csv)）。A4 重跑 `flow_sequence` 的 MOT17-10 SCFL 與實驗檔案逐位元相同（[a4-scfl.sha256](capture/a4-scfl.sha256)）。A5：host 端 Execute 中位數 0.709 ms，p95 0.931 ms。

### 3. MOT17 HR 路徑

我另外不經腳本重算雜湊比對，結果一致：

| 比對 | 相同序列數 |
| --- | --- |
| SCDR 對 M4 | 7/7 |
| 即時 MotionContext 對 `gmc_records` | 7/7 |
| pipelined MOT 對實驗 G1 | 7/7 |
| serial MOT 對實驗 G1 | 7/7 |
| `track_offline` MOT 對實驗 G1 | 7/7 |
| pipelined packets 對 serial packets | 7/7 |

### 4. PersonPath22 與 4K

69 個序列的 MOT 都等於第二輪 G1 臂（獨立重算 69/69）。5 個 4K 序列全部跑完，分數只記錄，不設門檻：

| 序列 | 幀數 | ms/frame | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uid_vid_00179` | 977 | 5.98 | 33.44 | 45.73 | 24.78 | 38.25 | 39.43 | 141 |
| `uid_vid_00189` | 527 | 6.02 | 41.64 | 42.89 | 40.89 | 47.61 | 18.14 | 40 |
| `uid_vid_00218` | 446 | 6.00 | 45.59 | 51.24 | 40.91 | 54.27 | 52.27 | 61 |
| `uid_vid_00226` | 741 | 5.95 | 65.97 | 65.15 | 66.85 | 81.95 | 72.13 | 5 |
| `uid_vid_00230` | 626 | 5.98 | 42.09 | 43.44 | 41.00 | 49.56 | 41.97 | 55 |

### 5. 計分

未四捨五入的值都和實驗參考逐位相等，完整數值在 [scores.csv](capture/scores.csv)：

| 組別 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOT17 val（MOT17-10） | 50.87 | 54.41 | 47.80 | 64.97 | 66.89 | 56 |
| MOT17 in-domain | 68.93 | 74.70 | 63.85 | 76.91 | 84.46 | 227 |
| MOT17-02 | 40.57 | 40.90 | 40.57 | 49.15 | 46.78 | 102 |
| PP-moving（9 序列） | 42.61 | 43.85 | 41.86 | 51.25 | 36.23 | 372 |
| PP-static（60 序列） | 41.58 | 44.12 | 39.78 | 47.72 | 39.33 | 4766 |
| PP-all（69 序列） | 41.65 | 44.10 | 39.91 | 47.95 | 39.14 | 5138 |

### 6. Sanitizer

| 項目 | memcheck | initcheck | racecheck | synccheck |
| --- | --- | --- | --- | --- |
| `tracking_packet_tests` | 0 | 0 | 0 | 0 |
| `tracking_cuda_tests` | 0 | 0 | 0 | 0 |
| `flow_motion_cuda_tests` | 0 | 0 | 0 | 0 |
| Positive control | exit 99，有診斷 | exit 99，有診斷 | exit 99，有診斷 | exit 99，有診斷 |

- **`session-racecheck`、`session-synccheck`：** T01 MOT17-10 前 10 幀，只檢查 v1 kernels，都是 0 個錯誤，各 222 checks。
- **完整 session memcheck（OPEN）：**

| 項目 | exit | 錯誤數 | 其他 API 或 device 錯誤 | 測試本身 |
| --- | ---: | ---: | ---: | --- |
| [hr-memcheck](capture/hr-memcheck.log) | 99 | 86 | 0 | 30 幀跑完 |
| [lifecycle-memcheck](capture/lifecycle-memcheck.log) | 99 | 1008 | 0 | 180 checks 通過 |

這些全部是 NVOF 函式庫內部呼叫 `cuArrayGetPlane` 失敗。呼叫端無法避免，但 NVIDIA 尚未確認它是否無害，所以這一項不算通過。Gate 只證明除此之外沒有別的錯誤（[hr](capture/hr-memcheck-classify.log)、[lifecycle](capture/lifecycle-memcheck-classify.log)）。重現與分析見 [evidence/nvof-getplane/2026-11-05](../../nvof-getplane/2026-11-05/README.md)。

### 7. Stream 位置

[stream-placement.txt](capture/stream-placement.txt) 判定 PASS：D 是 stream 13，M 是 stream 14。F05 kernels 在 D，F06 kernels 和 203 個 NVOF kernels 在 M。

**OFA 與 F03 重疊：** 29 幀中有 28 幀成立（[ofa-overlap.csv](capture/ofa-overlap.csv)）。nsys 看不到 OFA 本身，所以這是用界限證明的：OFA 一定發生在輸入複製完成和 cells 開始之間，而這整段（中位數 2.22 ms）落在同一幀的 F03 內。這個界限嚴格但間接。沒成立的是第 1 幀，因為第一次 Execute 的界限特別長；第一次 Execute 也會多出 12 個 `cuArrayGetPlane` 呼叫，但這次沒有驗證兩者的關係。

### 8. 吞吐

HR pipelined，無 profiler，7 序列各跑 3 次，新舊版本交替：

| 序列 | 每組幀數 | main（ms/frame） | G1（ms/frame） | 變化 |
| --- | ---: | ---: | ---: | ---: |
| MOT17-02 | 1800 | 5.112 | 5.263 | +2.95% |
| MOT17-04 | 3150 | 5.386 | 5.573 | +3.46% |
| MOT17-05 | 2511 | 5.297 | 5.314 | +0.33% |
| MOT17-09 | 1575 | 5.268 | 5.341 | +1.39% |
| MOT17-10 | 1962 | 5.264 | 5.566 | +5.75% |
| MOT17-11 | 2700 | 5.320 | 5.488 | +3.17% |
| MOT17-13 | 2250 | 5.365 | 5.460 | +1.77% |
| **總計** | **15948** | **5.300** | **5.443** | **+2.70%** |

§20.8 的門檻看總計，所以通過，但有幾點要照實說：

- **餘裕很小：** 只差 0.30 個百分點。三輪分開算是 +2.66%、+2.26%、+3.17%，第三輪單獨看超過 3%。
- **有離群值：** MOT17-10 第三輪是 5.92 ms，同序列其他兩輪是 5.38 和 5.40。沒有排除它，它占 MOT17-10 那個 +5.75% 的一大部分。
- **測試條件：**
  - 每組都是 main 先跑，G1 後跑，順序沒有輪換。
  - 遙測程式在整段期間都有執行。它不是 profiler，對兩個版本一視同仁，但確實不是完全沒有額外負載。

Serial 單幀延遲只記錄，6.18 到 7.16 ms/frame（[serial-latency.csv](capture/serial-latency.csv)）。

## 這份結果之前發生的事

- **2026-11-04：** s=2 parity 失敗，是測試程式的競態，kernel 沒錯。見[失敗報告](../2026-11-04-v1-motion-2/REPORT.md)，修正在 `0cae6af`。
- **2026-11-05 第二輪：** 第 42 個測試找不到輸入，因為 CMake 快取留著舊路徑。見[受阻報告](../2026-11-05-v1-motion-2-acceptance/REPORT.md)。
- **2026-11-05 第三輪：** [輸入檢查](preflight/)和[開發階段](development/)都通過後，機器在 10:18 硬重置，正式驗收中斷。保留的紀錄在 [interrupted-formal/](interrupted-formal/)，該次輸出未使用。

### 看到失敗後才改的 gate

這輪正式驗收前，我用開發 build 試跑了從未對真實資料執行過的項目，發現原本的 gate 過不了。以下三項改動都在 capture 前 commit（`fae5cb7`），前兩項也寫進了 §20.8：

| 改動 | 原因 | 方向 |
| --- | --- | --- |
| 三個 NVOF 內部 kernel 計為 OFA，必須在 M | 串流檢查把它們判成未分類 | 補規則，§20.8 本來就要求 OFA 在 M |
| 完整 session memcheck 只容許 NVOF 內部的錯誤，次數要完全相符，結果記為 OPEN | 這些錯誤呼叫端避不開 | **這是放寬**，所以不算 PASS |
| Positive control 的 exit code 真的會擋下錯誤 | 舊寫法在 `set -e` 下不會停，`main` 也一樣 | 變嚴格 |

試跑用的是開發 build，不算正式證據。

## 機器狀態

上一次中斷的 10:18 硬重置沒有藍屏代碼、沒有 dump，原因未查明。這一輪沒有再發生：

- **同樣的 TDR 又出現了：** `nvlddmkm #153` 在 13:30:34，正好是 `scan_trt_async_fault` 結束的時間（CTest 開始後 67 秒）。上次重置時正在跑的 `head_fp16_trt` 這次 38.54 秒通過。
- **遙測（[GPU](control/gpu-telemetry.csv)、[電源](control/power-telemetry.csv)）：** GPU 最高 75°C，功耗最高 143.52 W，記憶體最高 965 MiB。全程接著 AC，165 筆電源紀錄都沒有在放電。
- **clock event reason 原始值：** `0x400` 322 筆、`0x20` 123 筆、`0x4` 22 筆。這份報告沒有解讀它們的意義。

只跑一輪沒事不代表機器沒問題，只能說這次沒有重現。

## 未解決與不涵蓋

- **未解決：** Bug 7048273，NVIDIA 尚未回覆；10:18 的硬重置原因不明。
- **不涵蓋（§20.8）：** OFA 內部的 sanitizer、MEDIUM／FAST 的品質、局部光流臂。
- **未測試：** 原生 Linux driver、其他 GPU、其他 driver 版本。

## 合併前要處理

這些不影響這次的結果，但合併進 `main` 前必須解決：

- [`scripts/run_mot17_evaluation.sh:32`](../../../scripts/run_mot17_evaluation.sh) 要求 `acceptance.txt` 必須是 `PASS`，現在會拒絕執行。
- 驗收腳本用相對於 worktree 的路徑找 `../m4-formal`、`../gmc-flow` 和 `../g1-ofa-base`，從 `main` 的目錄跑會找不到。
- `m1-data-v2` 是連到這個 worktree `build/records/t01` 的符號連結，HR 的 SCMO 也在 `build/records/hr`。刪掉這個 worktree 或它的 build，這些 records 就會失效。

## 重跑

從乾淨的 commit，在這個 worktree 裡：

```sh
flock ~/.cache/saccade-v1/gpu.lock timeout -k 10s 14400s \
  scripts/run_tracking_session_acceptance.sh build/NEW-OUTPUT
```
