# M3 正式驗收：TrackingSession（2026-11-03）

**判定：PASS**。依 [TRACKING §14.5](../../../docs/TRACKING.md) 在 capture 前固定的標準，五項全部通過。capture 前的 sanitizer 探測找到兩個 session 的 cross-stream 順序錯誤，以及一個 initcheck 報告來源，三者都已在 1c03e22 修正，修正過程見下文〈Capture 前的探測〉。

## Source 與環境

| 項目 | 值 |
| --- | --- |
| Commit | `e021bd15c0f51ef800350d22c18fa5b4a31fe514`，乾淨的 detached worktree（`source-status.txt` 為空，`allow_dirty=false`） |
| Build inputs | 165 個檔案，digest `92157b09…`（[source-files.sha256](capture/source-files.sha256)）；每一步前後核對 |
| GPU／driver | RTX 5070 Ti Laptop（sm_120）、617.42、WSL2 |
| Toolchain | GCC 16.2.1、CMake 4.4.5、nvcc 13.4.92、TensorRT 10.16.1.11、Compute Sanitizer 2026.3.0.0 |
| 部署 engine | HR FP16 `model-hr-fp16.engine`，SHA-256 `8292cc23…`（§16），capture 開始時先核對 |
| CPU reference 用 engine | T01 `c6cff193…`，即錄製 M1 records 的 engine |
| 時間 | 18:56–19:13（+08:00），約 18 分鐘 |

capture 前後的 hash 都相同：77 個 binaries 與 library，以及 67 項 artifacts。artifacts 包括兩個 engine、detection reference corpus、14 個 M1 records，與 7 個 MOT17 SDP 序列，每個序列以 seqinfo 加全部影像算一個 digest。

## 結果

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | Clean source | PASS：HEAD 與 build inputs 全程不變；binaries 與 artifacts 前後相同 |
| 2 | Build 與測試 | PASS：CPU CTest **16/16**；CUDA CTest **59/59**（8 分 17 秒） |
| 3 | 部署路徑（HR FP16，MOT17-10） | PASS：以下四項逐位元相同 |
| 4 | Sanitizer | PASS：17 次執行全部 0 errors／0 hazards；4 個正控制都偵測到 |
| 5 | Evidence report | 本文件與 [capture/](capture/) |

### 2. 7 序列 TrackingSession 對 CPU reference

T01 engine 加上 M1 records：每個序列的 serial 與 pipelined packet stream 都和 `TrackerReference`＋`encode_packet` 逐位元相同，MOT17-10 與 MOT17-05 另加兩個擾動 seed。這些擾動 run 現在含有兩段固定長度的 spin，正是 1c03e22 修正的兩個 race 的回歸測試。

| 序列 | 幀數 | checks | 時間 |
| --- | ---: | ---: | ---: |
| MOT17-02 | 600 | 4,212 | 19.6 s |
| MOT17-04 | 1,050 | 7,362 | 33.4 s |
| MOT17-05 | 837 | 10,903 | 52.3 s |
| MOT17-09 | 525 | 3,687 | 16.5 s |
| MOT17-10 | 654 | 8,524 | 43.9 s |
| MOT17-11 | 900 | 6,312 | 27.0 s |
| MOT17-13 | 750 | 5,262 | 23.5 s |
| lifecycle（MOT17-10） | — | 86 | 4.9 s |

MOT17-02 只比較 packet bytes，不讀 GT，GT 評估留給 M4。其餘 tracking 相關測試：`tracking_cuda_mot17` 檢查 7 序列共 5,316 幀的 M2 kernel 與 CPU 逐位元相同；另有 `tracking_cuda`、`tracking_packet`、`motion_cuda`、`motion_reference`、`tracking_reference`。完整輸出見 [LastTest.log](capture/LastTest.log)。

### 3. 部署路徑

| 輸出 | SHA-256 | 判定 |
| --- | --- | --- |
| `detect_sequence --profile hr-1280x768` SCDR | `98e2554f…` | 等於 §16 |
| `motion_sequence` SCMO | `015ee1a1…` | 等於 M1 records |
| `track_offline`（CPU reference）MOT 檔 | `1fb713f3…` | 等於 §16 |
| `track_sequence` pipelined／`--serial` MOT 檔 | `1fb713f3…` | 與 CPU reference 相同 |
| pipelined／serial packet stream（654 × 12,368 bytes） | `f5cb1384…` | 兩者相同 |

修正後的 session 在部署設定下輸出不變：兩個 race 在一般時序下從未發生。`track_sequence` 在本次 capture 中為 pipelined 8.55 ms／幀、serial 10.64 ms／幀，只作紀錄。

### 4. Sanitizer

| 執行 | 範圍 | 結果 |
| --- | --- | --- |
| memcheck／initcheck／racecheck／synccheck × 3 個 kernel suites | `tracking_packet_tests`（804 checks）、`tracking_cuda_tests`（7,404,862）、`motion_cuda_tests`（537；initcheck 加 `--synthetic` 為 523） | 12 次全部 0 errors；racecheck 為 0 hazards |
| 完整 session memcheck，所有 kernel | T01 MOT17-10 前 30 幀（412 checks，含擾動 run）、lifecycle（86）、HR FP16 `track_sequence` 前 30 幀 | 0 errors |
| 完整 session racecheck／synccheck，只檢查 tracking、motion 與 session kernels | T01 MOT17-10 前 10 幀（152 checks，含 serial、pipelined 與兩個擾動 seed） | 0 hazards／0 errors |
| 正控制 `postprocess_sanitizer_control` | 四種工具 | 全部 exit 99：memcheck `Invalid __global__ write`、initcheck `Uninitialized __global__ memory read`、racecheck `Race reported between`、synccheck `Barrier error detected` |

initcheck 的範圍依 §14.5：
- 不檢查 host API 複製：harness 會把整個固定容量結構複製回 host。
- motion suite 不解碼 nvJPEG 的兩張真實影像：nvJPEG 的既知報告依 [PREPROCESS_USAGE_POLICY](../../../docs/PREPROCESS_USAGE_POLICY.md) §3 是診斷項。

## Capture 前的探測（[probe/](probe/)）

探測用的是 04912e2 的開發 build。它找到下面三件事，都在 1c03e22 修正。實作由 Codex 完成，見 [part 1](probe/codex-part1-report.md) 與 [part 2](probe/codex-part2-report.md) 的回報。

1. **Tracker config 被清零（正式程式碼的 race）**
   - 原因：建構時 tracker storage 用 `cudaMemsetAsync` 在 non-blocking 的 M stream 清零，config 卻用同步 `cudaMemcpy` 在 legacy stream 上傳，兩者沒有先後保證。
   - 症狀：racecheck 下 pipelined 第 0 幀輸出 28 筆而非 0 筆，之後每幀都是新 uid，符合 config 全為 0。紀錄見 [config-race-racecheck.log](probe/config-race-racecheck.log)，3 次中 3 次重現；同一 binary 不加 sanitizer 時通過，見 [config-race-plain.log](probe/config-race-plain.log)。
   - 修正：config 改在 M 上、清零之後上傳。
   - 回歸測試：擾動模式在建構時插入 20M cycles 的 spin；修正前失敗，修正後通過。
2. **F05 讀到下一幀的 RGB（正式程式碼的 race）**
   - 原因：nvJPEG HARDWARE decode 寫 RGB buffer 時，不排在 detect stream 上先前已排入的工作之後。preprocessor 在下一幀解碼前只等 F02，F05 卻排在 F02 之後才讀同一個 buffer。
   - 症狀：racecheck 下擾動 run 的 MotionContext 不同。第 1 幀 pcr 小幅不同，第 2 幀 tx 為 0 而非 −8；F04 candidates 完全相同。
   - 驗證：實驗版在 F05 之後記 event、下一幀解碼前於 host 等待，結果 3 次全部相同（[實驗程式](probe/rgb-race-experiment.cpp)，[1](probe/rgb-race-experiment-1.log)、[2](probe/rgb-race-experiment-2.log)、[3](probe/rgb-race-experiment-3.log)）；未修改的對照組仍失敗（[control](probe/rgb-race-control.log)）。
   - 修正：正式版採同樣做法，事件名為 `rgb_read_done`。
   - 回歸測試：擾動模式每幀在 F05 前插入 10M cycles 的 spin；修正前失敗，修正後通過。
   - 文件：[PREPROCESS.md](../../../docs/PREPROCESS.md) 的 RGB 借用規則已改為 consumer 必須在下一次 `enqueue` 前執行完成。
   - 影響範圍：`motion_sequence` 每幀都在 host 等 F05 完成，所以 M1 records 不受影響；DetectionSession 在 F02 之後不讀 RGB，也不受影響。
3. **`DeviceTrack` padding 的 initcheck 報告**
   - 現象：`tracking_cuda_tests` 有 35,841 筆報告，全部是 `tracking.cu:216` 的 struct 複製，讀到 offset 332–335 從未寫過的 4 bytes padding（[summary](probe/initcheck-padding-summary.txt)）。
   - 正式路徑：session 會把所有 storage 清零，結果仍是確定的。
   - 修正：padding 改為明確且恆為 0 的 `reserved` 欄位，`sizeof` 仍為 408；修正後 0 errors。

兩個 race 在一般時序下都沒有發生：先前所有錄製與逐位元比較的結果，以及本次部署路徑的 hash，都和修正前相同。

## 不涵蓋

- F03（TensorRT）kernels 的 racecheck／synccheck 仍未執行，沿用 S04 的狀態。session 的 racecheck／synccheck 只檢查 v1 的 tracking、motion 與 session kernels。
- 完整 session 的 initcheck 未執行：nvJPEG 的既知報告（Bug 7021871）依政策只作診斷。
- 效能與追蹤品質不設門檻。MOT17-02 的 GT 評估留給 M4。
- 有限的序列、幀數與擾動 seed 通過，不等於所有時序都已證明。

## 重跑

```sh
SACCADE_TRACKING_ENGINE=/abs/t01/model.engine SACCADE_DETECTION_REFERENCES=/abs/refs \
SACCADE_TRACKING_SEQUENCES=/abs/MOT17/train SACCADE_TRACKING_RECORDS=/abs/m1-data \
  flock ~/.cache/saccade-v1/gpu.lock bash scripts/run_tracking_session_acceptance.sh build-NEW
```

從本目錄執行 `sha256sum -c SHA256SUMS` 可核對封存的檔案。
