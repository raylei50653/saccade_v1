# M4 正式評估與 P01 收尾（2026-11-03）

**判定**
- [TRACKING §17.3](../../../docs/TRACKING.md) 的正式執行 **PASS**：完整正式驗收、四項守門與計分全部完成。
- P01 依 §17.4 記為 **CLOSED（有範圍）**。

MOT17-02 的 GT 在本次執行中第一次、也是唯一一次對 HR 偵測器開啟（20:56:50），使用的設定在 [e6bc242](../../../docs/TRACKING.md) 提交時已凍結。

## Source 與環境

| 項目 | 值 |
| --- | --- |
| Commit | `cad52981c7ecf40e13c9c016f351f6137b2b61f3`，乾淨的 detached worktree（兩段 capture 的 `source-status.txt` 都為空，`allow_dirty=false`） |
| Build inputs | 驗收：165 個檔案，digest `fc85bdd2…`（[acceptance/source-files.sha256](acceptance/source-files.sha256)）。評估：另含 M4 腳本、`eval_scdr.py`、`t01_eval.py` 與兩份 §16 紀錄，digest `d954c74f…`（[evaluation/source-files.sha256](evaluation/source-files.sha256)）。每一步前後都核對 |
| GPU／driver | RTX 5070 Ti Laptop（sm_120，最高 SM clock 3090 MHz）、617.42、WSL2 |
| Toolchain | GCC 16.2.1、CMake 4.4.5、nvcc 13.4.92、TensorRT 10.16.1.11、Compute Sanitizer 2026.3.0.0、Nsight Systems 2026.3.2 |
| 部署 engine | HR FP16 `model-hr-fp16.engine`，SHA-256 `8292cc23…`（§16） |
| 時間（+08:00） | 驗收 20:36–20:53；評估與守門 20:53–20:56；計分 20:56–20:57 |

## 結果

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | 完整正式驗收（§14.5，M4 結案時） | PASS：CPU CTest **16/16**，CUDA CTest **59/59**；部署路徑 hash 與 §16 相同；17 次 sanitizer 執行全部 0 errors／0 hazards，4 個正控制全部 exit 99 |
| 2a | 身分鏈 | PASS：39 個檔案與 deploy manifest、build report 一致 |
| 2b | CPU reference 對照 | PASS：7 個序列的 `track_offline` 與 `track_sequence` MOT 檔逐位元相同 |
| 2c | 與既有紀錄一致 | PASS：MOT17-10 的 SCDR `98e2554f…` 與 MOT `1fb713f3…`；5 個 in-domain 序列的 MOT 等於 §16 |
| 2d | Stream 位置 | PASS（下表） |
| 3 | 計分 | 完成，結果見〈指標〉 |

### 1. 完整正式驗收

- 用的是 M3 的 [`run_tracking_session_acceptance.sh`](../../../scripts/run_tracking_session_acceptance.sh)，標準與預期 hash 都沒有改，capture 在 [acceptance/](acceptance/)。
- capture 前後的 hash 都相同：77 個 binaries 與 library，以及 67 項 artifacts。artifact 清單與 M3 正式驗收（e021bd1）的 [artifacts.sha256](../../tracking-session/2026-11-03-formal-e021bd1/capture/artifacts.sha256) 逐位元相同。
- 部署路徑（MOT17-10）：
  - SCDR 為 `98e2554f…`，SCMO 為 `015ee1a1…`（等於 M1 records）。
  - CPU reference、pipelined、serial 三者的 MOT 都是 `1fb713f3…`。
  - pipelined 與 serial 的 packet stream 都是 `f5cb1384…`。
  - `track_sequence`：pipelined 8.41 ms／幀，serial 10.77 ms／幀。
- Sanitizer：
  - 3 個 kernel suites × 4 種工具。
  - 完整 session 的 memcheck：T01 前 30 幀、lifecycle、HR FP16 前 30 幀。
  - 完整 session 的 racecheck／synccheck：T01 前 10 幀，只檢查 v1 的 kernel。
  - 正控制：memcheck、initcheck、racecheck、synccheck 各一個。
  - 範圍與 M3 相同，見 M3 [REPORT](../../tracking-session/2026-11-03-formal-e021bd1/REPORT.md) §4。

### 2. 評估與守門（[evaluation/](evaluation/)）

[`run_mot17_evaluation.sh`](../../../scripts/run_mot17_evaluation.sh) 只使用第 1 步 build 出的 `detect_sequence`、`track_sequence` 與 `track_offline`。它先核對這些 binaries 的 hash 等於驗收時的 `binaries.sha256`，以及驗收的 revision 等於 HEAD。守門全部在讀 GT 之前完成（[gates.log](evaluation/gates.log)）。

**身分鏈**（[identity.txt](evaluation/identity.txt)；[deploy manifest 副本](evaluation/hr-deploy-manifest.json)）
- student checkpoint `53bccffe…`（HR01-MS-FULL25-20261103 mixed e10）。
- teacher checkpoint `08be5aba…`。
- base `yolo26m.pt` `401cea9a…`。
- manifest 列出的 35 個 artifact，bytes 與 SHA-256 都相同；其中 backbone ONNX 為 `cc1fa6d3…`，head bundle 為 `c6f3416f…`。
- build report 的 `onnx_sha256`、`bundle_sha256`、`engine_sha256` 分別等於上述 ONNX、bundle 與部署 engine。

**Stream 位置**（[stream-placement.txt](evaluation/stream-placement.txt)；trace 為 [nsys/track.nsys-rep](evaluation/nsys/track.nsys-rep)）

HR FP16 `track_sequence`，MOT17-10 前 30 幀。從第一個 F02 kernel 起共 10,289 個 kernel，每個都恰好歸入一類。TensorRT 的 kernel 以 31 個 `ExecutionContext::enqueueV3` NVTX range 判定：每幀一次，加上建構時一次 warm-up。

| 類別 | 要求 | stream | kernels | 不同名稱 |
| --- | --- | ---: | ---: | ---: |
| TensorRT engine（含 scan plugin） | D | 13 | 9,060 | 153 |
| F02 | D | 13 | 60 | 2 |
| F05 | D | 13 | 60 | 2 |
| producer | D | 13 | 90 | 3 |
| F04 | D | 13 | 150 | 5 |
| cuFFT | M | 14 | 120 | 3 |
| F06 | M | 14 | 300 | 6 |
| tracking | M | 14 | 150 | 3 |
| session control | M | 14 | 240 | 4 |
| packet | M | 14 | 30 | 1 |
| nvJPEG `rgba2rgb` | 只記錄 | 25 | 29 | 1 |

- TensorRT 沒有使用 aux stream，cuFFT 也只跑在 M 上，對應 [ASYNC_CONTROL](../../../docs/ASYNC_CONTROL.md) 列入 P01／P05 的兩項。
- nvJPEG HARDWARE backend 的色彩轉換 kernel 跑在 nvJPEG 自己的 stream 25 上，不在呼叫端傳入的 D。這是 [nvJPEG stream 順序問題](../../preprocess/2026-11-03-nvjpeg-stream-order/README.md)（NVIDIA Bug 7038860）在完整 session 裡的直接觀測。session 用 host 等待 `rgb_read_done` 處理（§14.1）。

## 指標

計分由 [`score_mot17_evaluation.sh`](../../../scripts/score_mot17_evaluation.sh) 執行，MOT17-02 排在最後，彙整在 [metrics.tsv](evaluation/metrics.tsv)。MOT 檔在 [mot/](mot/)，TrackEval 的 summary 與 detailed 表在 [evaluation/trackeval/](evaluation/trackeval/)。SCDR 只記 hash（[outputs.sha256](evaluation/outputs.sha256)），不封存。依 §17.2，這些指標不設門檻。

### 偵測 AP50（`eval_scdr.py`，kept rows）

| 序列 | 組別 | HR FP16 | T01 native | recall@0.25 HR／T01 | FP@0.25 HR／T01 |
| --- | --- | ---: | ---: | --- | --- |
| MOT17-10 | val | 86.02% | 67.59% | 83.5%／62.4% | 1,972／1,459 |
| **MOT17-02** | **eval** | **72.91%** | 54.40% | 77.8%／47.8% | 4,407／1,580 |
| MOT17-04 | in-domain | 93.07% | 95.05% | | |
| MOT17-05 | in-domain | 97.36% | 98.70% | | |
| MOT17-09 | in-domain | 97.36% | 96.74% | | |
| MOT17-11 | in-domain | 97.33% | 98.86% | | |
| MOT17-13 | in-domain | 92.70% | 94.54% | | |

T01 的 MOT17-02 native AP50 為 54.40%，與 T01 訓練時的紀錄 0.544 一致。

### 追蹤（TrackEval）

| 組別 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | ms／幀 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val（MOT17-10） | 49.676 | 54.056 | 45.894 | 62.962 | 65.044 | 66 | 8.46 |
| **eval（MOT17-02）** | **40.625** | 40.853 | 40.723 | 49.290 | 46.752 | 103 | 8.51 |
| in-domain（04／05／09／11／13） | 68.423 | 74.454 | 63.148 | 76.136 | 84.284 | 231 | |
| MOT17-04 | 74.802 | 83.170 | 67.376 | 80.578 | 92.464 | 71 | 8.33 |
| MOT17-05 | 57.999 | 58.981 | 57.559 | 69.842 | 66.329 | 48 | 7.36 |
| MOT17-09 | 57.724 | 69.782 | 47.909 | 65.467 | 81.822 | 24 | 7.24 |
| MOT17-11 | 65.821 | 70.249 | 61.768 | 74.569 | 78.317 | 19 | 8.13 |
| MOT17-13 | 54.361 | 59.051 | 50.267 | 67.848 | 67.497 | 69 | 8.34 |

ms／幀是 `track_sequence` pipelined 自己回報的時間：從第一張 JPEG 讀檔到最後一個封包收回，不含 engine 載入與 MOT 寫檔。7 個序列為 7.2–8.5 ms，約 118–138 FPS。

### 與 T01（§13）比較，MOT17-02

| | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T01 | 31.6 | 23.7 | 42.2 | 37.2 | 23.5 | 31 |
| HR FP16 | 40.6 | 40.9 | 40.7 | 49.3 | 46.8 | 103 |

- HOTA 的提升全部來自偵測：DetA +17.2，AssA 則是 −1.5。
- IDSW 從 31 增加到 103。
- val 與 eval 的 AP50 差距，HR 為 13.1 pp（86.0 → 72.9），T01 為 13.2 pp（67.6 → 54.4）。

## P01 判定（§17.4）

| 項目 | 證據 | 狀態 |
| --- | --- | --- |
| 來源／版本 | HR deploy manifest（`saccade-hr-deploy/1`）與本次身分鏈；訓練 split 的 eval 固定為 MOT17-02，manifest 記為 `held_out_evaluated=false`，直到本次才開啟 | 已驗 |
| 權重 native 交付與 component I/O | D1（TRACKING §15）：head bundle 由 native converter 產生；backbone 特徵與訓練 cache 的 51,118,080 個元素逐位元相同；native AP50 與訓練紀錄差 −0.0001 pp | 已驗 |
| Engine 精度與 shape binding | D2 build report 的讀回值與 inspector（TRACKING §16 第 1 項） | 已驗 |
| Engine 與 runtime 身分一致 | 本次身分鏈；兩次正式驗收（e021bd1、cad5298）的 SCDR 與 MOT hash | 已驗 |
| 函式庫的 stream 行為 | 本次 stream 位置 | 已驗 |
| 模型品質 | 上述 AP50 與追蹤指標 | 只記錄 |

**P01：CLOSED（有範圍）。**

## 不涵蓋

- TensorRT 內部 kernel 的 racecheck／synccheck：沿用 S04 起的決定，未執行。v1 自己的 scan plugin、producer、F04 與 tracking kernel 有四種 sanitizer 的證據；TensorRT kernel 有完整 session 的 memcheck。
- 完整 session 的 initcheck：nvJPEG 的既知報告（Bug 7021871）依 PREPROCESS_USAGE_POLICY 處理。
- FP16 engine 是在 761889c 加上未提交的修改（之後提交為 04912e2）下 build 的（`build_dirty=true`）。部署身分以 engine SHA-256 固定，沒有重新 build。
- v1 的 native manifest 工具（schema 1–3）不描述 HR profile 或 FP16。HR 的身分紀錄是 deploy manifest，不另加 schema。
- 跨 GPU、driver 或 TensorRT 版本的支援（P07）。
- 速度只在一台筆電 GPU、WSL2 上量過一次，沒有控制 clock 或溫度。
- 本次之後若再改偵測器、tracker 或參數，MOT17-02 的結果要標為「M4 之後」，不再算獨立 eval（§17.1）。

## 重跑

```sh
SACCADE_TRACKING_ENGINE=/abs/t01/model.engine SACCADE_DETECTION_REFERENCES=/abs/refs \
SACCADE_TRACKING_SEQUENCES=/abs/MOT17/train SACCADE_TRACKING_RECORDS=/abs/m1-data \
  flock ~/.cache/saccade-v1/gpu.lock bash scripts/run_tracking_session_acceptance.sh build-ACCEPT
SACCADE_TRACKING_SEQUENCES=/abs/MOT17/train SACCADE_TRACKING_RECORDS=/abs/m1-data \
  flock ~/.cache/saccade-v1/gpu.lock scripts/run_mot17_evaluation.sh build-ACCEPT build-EVAL
scripts/score_mot17_evaluation.sh build-EVAL
```

從本目錄執行 `sha256sum -c SHA256SUMS` 可核對封存的檔案。
