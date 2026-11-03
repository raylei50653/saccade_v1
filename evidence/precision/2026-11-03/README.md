# Full-model FP16／BF16 精度實驗（2026-11-03）

- 範圍：完整模型（backbone/neck + Mamba-FPN head），batch 1，T=1。
- 序列：只用 MOT17-10-SDP；MOT17-02 沒有開啟。
- 本實驗只用來選精度。正式驗收見 [TRACKING §16](../../../docs/TRACKING.md)。

## 來源

**第一輪（使用者，opt0）**
- 目錄：`/home/ray/developer/ai/saccade_v1_dev/build-precision/20261103-hr-t01`，報告為該目錄下的 `REPORT.md`。
- 內容：FP32 部署 engine 當對照組，另有 FP16、BF16 兩組 weakly typed 候選。
- Builder 設定：optimization level 0，TF32 關閉。

**第二輪（本目錄，opt3）**
- 用相同的 patch，再加 `SACCADE_EXPERIMENT_OPT_LEVEL`。這個參數只作用在 FP16／BF16 實驗路徑，見 [experiment.patch](experiment.patch)。
- 以 optimization level 3 重新 build 四個候選。
- 每個候選的步驟：
  1. 用 `precision_probe` 計時（100 次 warmup、500 次計時、一輪）。
  2. 跑完整 654 幀 `detect_sequence`。
  3. 用 `training/hr_deploy/eval_scdr.py` 計算 AP50。

逐項數值、engine SHA-256 與卷積／GEMM kernel 統計見 [candidates.json](candidates.json)。

## 結果

| 模型 | 候選 | AP50 | 與 FP32 差（pp） | enqueue ms | CUDA graph ms |
| --- | --- | ---: | ---: | ---: | ---: |
| HR | FP32（部署中） | 86.0092 | — | 25.67 | 25.26 |
| HR | FP16 opt0 | 86.0265 | +0.017 | 18.24 | 17.34 |
| HR | BF16 opt0 | 85.5460 | −0.463 | 8.59 | 7.42 |
| HR | **FP16 opt3** | 85.9840 | −0.025 | **6.50** | **5.10** |
| HR | BF16 opt3 | 85.4614 | −0.548 | 7.58 | 6.36 |
| T01 | FP32 | 67.5850 | — | 12.36 | 11.65 |
| T01 | FP16 opt0 | 67.6029 | +0.018 | 11.92 | 10.83 |
| T01 | BF16 opt0 | 67.5578 | −0.027 | 5.33 | 3.73 |
| T01 | **FP16 opt3** | 67.5992 | +0.014 | **4.49** | **2.60** |
| T01 | BF16 opt3 | 67.6005 | +0.016 | 5.05 | 3.10 |

**六個原始輸出的最大 relative L2 誤差**

| 模型 | 對照基準 | FP16 | BF16 |
| --- | --- | ---: | ---: |
| HR | PyTorch strict FP32 | 約 0.0096 | 約 0.066 |
| T01 | 同輸入的 TensorRT FP32 | 約 0.021 | 約 0.149 |

**每個 engine 都通過的檢查**
- 輸出全為有限值。
- Plugin statuses 為 PASS。
- 同一 engine 重跑逐位元相同。
- CUDA graph 重播逐位元相同。

## 判讀

**opt0 時 FP16 慢、BF16 快，原因是 kernel 選擇，不是數值格式。**
- opt0 的 FP16 engine：多數卷積用的是 `implicit_gemm_indexed_wo_smem`／`implicit_gemm_indexed` 這類通用 kernel。
- opt0 的 BF16 engine：多數用一般的 `implicit_gemm`。
- 在 SM 12.0 上，FP16 與 BF16 的 tensor core 吞吐量相同。
- 換到 opt3 後，FP16 改用一般的 `implicit_gemm`，兩個模型都變成最快。原始輸出誤差約為 BF16 的七分之一，AP50 差異在 ±0.03 pp 內。

**決定（使用者，2026-11-03）**：HR 部署改用 FP16 opt3。

## 注意事項

- **重新 build 可能不同**：opt3 依實測時間挑 tactic，重新 build 可能產生不同的 engine。部署 engine 要用 SHA-256 固定，每次重新 build 都要重做 §16 驗收。
- **FP16 累加**：多數 FP16 卷積 kernel 以 FP16 累加（kernel 名稱結尾 `_f16`）。這正是每個 engine 都要通過 AP50 門檻的原因。
- **計時範圍**：只量模型本身（輸入已在 GPU 上），不含解碼、前處理與後處理。opt3 候選只量了一輪；它和 opt0 的差距遠大於第一輪三次重複之間的變動。
