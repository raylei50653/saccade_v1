# T01 native head 驗證報告（2026-10-30）

**結論：PASS。** T01 選定的 T1 head（epoch 1，`070580f9…6e6962b`）通過 S04 的完整 FP32 TensorRT head 驗證：

- 三組固定輸入與三幀真實 T01 features，六個輸出的每個元素都在 `s04-head-fp32/1` 容差內；
- 執行、失敗傳遞與 slot／graph 隔離檢查全部通過；
- head memcheck 與 initcheck 都是 0 errors。

範圍只到 head：P3／P4／P5 features → 六個 raw outputs。backbone／neck 的 native engine、native 前處理、decode／NMS 不在本報告範圍，**NATIVE ARTIFACT READY、F03、P01 都未關閉**。規格見 [HEAD_TRT §10](../../../../docs/HEAD_TRT.md#10-t01-候選的延續驗證2026-10-30)。

## 身分

| 項目 | 值 |
| --- | --- |
| 受測 source | `401ec6943f546743c4baa0e052a8c6c437949dba`（乾淨工作樹；archive SHA-256 `9a168ce6…7b681ea44`）；capture 期間每一步前後核對 HEAD 與工作樹，沒有變動 |
| 與 S04 的差異 | 只有 `tools/head_trt.cpp`（真實 features 輸入與 dump）與 `CMakeLists.txt`（可設定的候選 hash）。GPU kernels、plugins、建圖、frame session、reference 與 HEAD_POLICY 自 S04 受測 source `3857e24` 與政策 commit `4d38bcc` 起都沒有改動；S04 以前的證據與 manifest 也沒有改動。三項都在 capture 中以 `git diff --exit-code` 核對 |
| Checkpoint | `runs/t1/attempt-01/run/epoch_0001.ckpt`，126,818,317 bytes，SHA-256 `070580f9f5755e57acd9ee7ed9b4307672f4d9996332b40e6b61ee7cc6e6962b`；與 T01 選定紀錄相符 |
| Bundle | SCWB 1，40,711,616 bytes，SHA-256 `4fccfda6342b88d43fc2bf676228810b974b9e9956f45fa53352124098f046a3`；69 active＋39 excluded（mapping 見 `capture/t1-head-mapping.tsv`） |
| Engine | 41,471,380 bytes，SHA-256 `5b6d30fab444a6a6f649b187f3bc7d6f315a3a9256b48374872df4b35276024a`；重建得到相同 plan bytes；與開發建置也相同。builder level 0，沒有 Winograd、FFT 或 TF32 tactic |
| 真實 features | 3 幀（MOT17-02/1、MOT17-10/1、MOT17-13/400），31,948,800 bytes，SHA-256 `f8224da5…ba52e1`，由 T01 teacher e12 以 `fp32-strict` 算出 |
| 環境 | RTX 5070 Ti Laptop（sm_120，driver 617.42）、CUDA 13.4.92、TensorRT 10.16.1.11、GCC 16.2.1、WSL2；全程 6 min 50 s，可用記憶體最低 40.7 GiB（低於 4 GiB 會停止 capture） |

## 結果

| 項目 | 結果 |
| --- | --- |
| CPU CTest | 13/13，含以 T01 checkpoint 轉換 bundle，以及檢查 checkpoint 與 bundle 的 hash（錯誤 hash 須被拒絕） |
| Reference（binary64） | 228 checks；以 T01 權重檢查 13 個語義變體，**全部被偵測到**（`missed=none`） |
| CUDA＋TRT CTest | 31/31：S01–S03 全部；head engine 建置、執行、未註冊 plugin 被拒絕；`head_trt` suite 824 checks；api_trace |
| 固定輸入（asymmetric/1、random/1、random/2） | 18 個輸出全部 PASS，最差 誤差／容差 0.00498 |
| 真實 T01 features（3 幀） | 18 個輸出全部 PASS，最差 誤差／容差 0.0130（`reg_p3`）；非有限值 0 |
| 執行性質 | 六個輸入各自重跑逐位元相同；失敗序列 OK／FAILED(8)／SKIPPED，committed 不變；CUDA graph 重播與 enqueueV3 逐位元相同；未註冊 plugin 無法建立 execution context |
| Head memcheck | 640 checks，0 errors（explicit API 報告，排除 intermediates） |
| Head initcheck | 593 checks，0 errors（另加 `--no-held-stream`） |
| Model manifest | [manifests/t01/yolo26m-head-t1.json](../../../../manifests/t01/yolo26m-head-t1.json)：宣告與 artifact 檔案皆 verified；`real_model_ready=false`（缺 backbone） |
| 身分 | build、run、真實 features run 與 suite 四份報告都是受測 commit 且 `build_dirty=false`；capture 結束時工作樹仍乾淨 |

**診斷（不是驗收）：訓練側 PyTorch T1 head vs native。** 同三幀、PyTorch `fp32-strict`：
- 六個輸出的相對 L2 ≤ 6.5e-7，最大絕對差 ≤ 5.7e-5。
- 經 T01 decode（conf 0.001、top-300、NMS 0.5）後，偵測數相同（32／28／39），且次序相同；框最大差 3.1e-4 px，score 最大差 3.7e-6。

**未執行或不在範圍：**
- head racecheck／synccheck 未執行，原因同 S04；
- S03 scan suite 的 sanitizer 不重跑，因為其 code 自 S04 起未改動；
- 沒有跑 mutation 檢查：GPU 實作與 reference 未改；
- backbone／neck 的 native engine、native 前處理、decode／NMS、tracking 品質。

## 檔案

`capture/` 是 [capture.sh](capture.sh) 每一步的 `.command`／`.log`／`.exit`，以及 `memory.log`、報告與 mapping。其中：
- `head-run-real-features-report.txt`：六個輸入的逐輸出判定；
- `pytorch-vs-native.json`：上述診斷；
- `sanitizer-skips.txt`：省略項目。

大型檔案（bundle、engine、dumps、features）留在 `~/.cache/saccade-v1/t01/` 與本機 `build-t01-head-evidence/`，hash 見 `capture/binary-hashes.log`。本目錄全部檔案的 SHA-256 見 [SHA256SUMS](SHA256SUMS)。
