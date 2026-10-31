# T01 計算精度證據（2026-10-30）

規格見 [T01_TRAINING §2.2](../../../../docs/T01_TRAINING.md#22-計算精度與儲存-t01-precision1)。此處全部是 smoke 證據（MOT17-09 訓練、MOT17-10 val、MOT17-02 final eval），不是正式訓練結果。大型檔案（checkpoint、cache、ONNX）留在 `~/.cache/saccade-v1/t01/`，此處只記錄 hash。

## 檔案

| 檔案 | 內容 |
| --- | --- |
| `tf32-finding/input_paths_check.py`、`.json` | 發現 TF32 差距的分解，對象是加入精度控制前的舊 smoke（`smoke-tf32-20261030`，舊 cache 與其 teacher）。各輸入變體與 TF32 對特徵的影響；舊 cache ＝ TF32 重算後轉 fp16，逐位元相同。最初的報告在暫存目錄遺失，以相同分析重跑（腳本改為以參數指定舊 smoke root）；數字與最初相同 |
| `tf32-finding/tf32_probe.py` | 以 conv2d／matmul 對 float64 量測實際運算；也驗證 `NVIDIA_TF32_OVERRIDE=0` 會在函式庫層強制 FP32 |
| `tf32-finding/tf32_timing.py`、`.txt` | backbone 前向時間，TF32 開／關：batch 1 為 8.2／14.0 ms，batch 16 為 5.3／8.4 ms |
| `guard-test/` | `t01_exec` 的程序內 guard：`fp32-strict` 與 `torch-default` 的讀回（`rec-*.json`）；腳本中途打開 cuDNN TF32 後 10 s 內被 SIGTERM（exit 143），偏離記在 `rec-flip.json` |
| `smoke-summary.json` | `t01_run.py smoke` 全鏈各步驟與時間；teacher 完整 resume 的 epoch 2 loss 與不中斷時相同（5.76607591455633），distill resume 不同（17.6437 vs 17.5802，預期內，不恢復 RNG） |
| `smoke-attempts.json`（`collect_attempts.py` 產生） | 每個 attempt 的 policy、環境變數、程序內開始與結束讀回、monitor 取樣、偏離、cache manifest 的儲存 dtype、每個 epoch checkpoint 的 tensor dtype 與存檔時精度；以及每個評估結果的開始與結束讀回 |
| `smoke-backbone-export.json`、`smoke-backbone-verify.json` | smoke teacher 的 backbone／neck ONNX（SHA-256 `e96e5e58…0174fc03`，440 個 initializer 全部逐位元對應到 teacher tensor）；ONNX Runtime vs PyTorch vs cache，未訓練權重為負向對照 |
| `smoke-feature-precision.json` | `t01_verify_features.py` 的驗收報告（容差事先固定）：**PASS** |
| `smoke-feature-precision-run1-harness-bug.json` | 同一驗收的第一次執行：除一項外全過。失敗的是「trainer 拒用舊 cache」，原因是腳本取了 traceback 中的 `raise` 原始碼行，而不是例外訊息（trainer 實際上正確拒絕，mismatch 只有 `storage_dtype` 與 `compute_precision`）。修正解析後整份驗收重跑，容差未變 |
| `smoke-final-eval.json` | smoke 的 final eval，含評估程序的精度讀回 |

## 結果摘要

所有 attempt 的程序內紀錄都相符，沒有偏離：

- cache_train／cache_val／cache_eval 與 GT1 為 `fp32-strict`，開始與結束時的探針都是 conv／matmul FP32；
- teacher、distill、T3、T1 為 `torch-default`（TF32 卷積、FP32 matmul）；
- 12 個評估紀錄開始與結束都是 `fp32-strict`；
- cache 的 `storage_dtype` 為 float16，checkpoint 權重全部 float32。

驗收（8 張 frame：MOT17-09 與 MOT17-10 各取第 1、1/3、2/3、最後一幀；相對 L2 取最差的 frame 與尺度）：

| 檢查 | 最差值 | 容差 |
| --- | ---: | ---: |
| FP32 實作差：ONNX Runtime CPU vs PyTorch `fp32-strict`，batch 1 | 1.81e-5 | 1e-4 |
| Batch 4 vs batch 1 | 1.18e-5 | 1e-4 |
| Batch 8 vs batch 1 | 1.22e-5 | 1e-4 |
| FP16 儲存：cache vs batch-1 FP32 參考 | 2.11e-4 | 4.88e-4 |
| cache vs batch 4／batch 8 | 2.12e-4／2.12e-4 | 5.88e-4 |
| 訓練 vs 部署參考：cache vs ONNX Runtime | 2.12e-4 | 5.88e-4 |

- **FP16 儲存誤差只來自捨入**：cache 與「FP32 參考轉 fp16」逐元素相同的比例是 1.0（全部元素）。
- **批次解碼**：批次與單張 nvJPEG 解碼逐位元相同。
- **ONNX Runtime**：用 `CPUExecutionProvider`（安裝的套件只提供 Azure 與 CPU provider），不涉及 TF32。
- **負向對照：**
  - 舊 TF32 cache 對同一 teacher 的 FP32 重算，最小差 3.4e-3，超過 FP16 儲存容差；
  - runner 拒絕舊 cache；
  - distill 拒絕舊 cache，mismatch 只有 `storage_dtype` 與 `compute_precision`；
  - T3／T1 trainer 也拒絕舊 cache。
- **TF32 對照**：舊 TF32 smoke 的 cache 與 FP32 參考差 0.81%／1.58%／0.95%；新 cache 與 ONNX Runtime 差 2.1e-4。

驗收程序本身在 `fp32-strict` 下執行，開始與結束都讀回確認。執行時 v1 為 `3134f0a`＋本目錄與文件的未提交更新。
