# Tracking 排序優化：組裝 kernel 改善，整體延遲差異很小

2026-11-04。開發來源驗證；基底 commit `865bb105b2b04e5e51a09c7106952184f5b2a9e1` 加上既有 CUDA Graph 與本次未提交修改。排序前 kernel SHA-256 為 `5a582b15ed42b04efa7d919191daba3ff02622277fb036b27e08f41573ea0b0e`，baseline binary 在修改 kernel 前獨立建置、凍結，不覆寫既有 CUDA Graph 證據。

## 結論

可以減少排序工作，且不改變原有順序或數值結果。F04b 整段組裝 kernel 的獨立 graph replay 平均 **143.57→102.81 µs，降低 28.39%，1.396×**；update kernel 為 **77.95→74.58 µs**。這些時間包含該 kernel 的其他工作，並非單獨排序時間。

同條件完整 HR tracking 的單幀平均延遲只有 **7.037→6.978 ms（−0.85%）**，p95 為 **7.383→7.369 ms**；queued wall time/frame 為 **5.325→5.323 ms**，吞吐幾乎不變。這些整體差異很小，**本次不宣稱穩定的端到端加速**，不能把 F04b 的 1.396× 推廣為整條路徑的加速。

## 改了什麼

1. **驗證已有順序，避免重排。** Native F04 的完整 pool 已按 score 降序、candidate ID 升序輸出。F04b 在 device 檢查全部相鄰 keys；若有序，以穩定 block scan 壓實 suppressed 子序列。不是假設任意 caller 都有序：發現 inversion 便改走 stable radix。
2. **一般路徑改 stable block radix。** suppressed 使用 64-bit `(score-desc32, full-uint32-ID)` key、原 input index payload；signed zero 正規化，支援負有限分數。blocked load 與 CUB stable sort 保留相同 score/ID 的原 index 順序。沒有截短 candidate ID。
3. **batch ID 排序分級。** 0/1 筆直接返回，2..32 筆保留原有低固定成本的 rank，較大 batch 改 32-bit stable radix。合法 `UINT32_MAX` ID 與尾端 padding 相同 key 時，stable blocked order 保證合法 rows 在前。
4. **birth 不再重排已排序 batch。** device 檢查 batch ID 是否非遞減；session 正常 batch 直接按原 index 配 UID。public update 仍接受未排序 batch，必要時沿用原有穩定 insertion，只排序 index permutation，不改 association rows 或 `used[]` 映射。

全候選 finite/header 檢查與原有 deferred NUMERIC gates 保留。到達 continuation 上限後，未實際訪問候選的預算 IoU failure 不會因此升級為錯誤。沒有改 tracker thresholds、浮點算術順序、ABI、scratch allocation 大小、CUDA Graph 的 frame 更新或 ready/consumed 協定。

## 怎麼量

相同 RTX 5070 Ti Laptop GPU / SM 12.0、CUDA 13.4.92、TensorRT 10.16.1.11 / WSL2，沒有鎖 clock。HR FP16 部署 engine hash 仍為 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`。

固定 MOT17-10-SDP 的 HR FP16 SCDR（`98e2554f1b1db179bfd756fb1dbd37ee694e912d6debb2c7a11c3a610a433a6f`）與 SCMO（`015ee1a1684c4b479dd273b4083722cf8721c14ff3c87d99c4a44ae2bcacb5e2`）。CPU 以相同 tracker config 依序產生前 140 幀的 predicted / batch / update 預期值，固定 replay inputs，再用 baseline→optimized→optimized→baseline 順序比較兩個獨立 binary。

Kernel replay 每幀、每模式 8 次呼叫/輪、3 輪，以 CUDA events 測 batch 平均；ordinary submission 與 graph replay 分開。upload、capture、instantiate、CPU 比較不在 timing 內，沒有宣稱單幀 p95。受測 pool 固定最多 300 筆、274 suppressed、24 priors。所有 frame / mode / round 都對完整邏輯 batch、track state 與 output rows 做 CPU float-bit 比較；未用容量及 device struct padding 不是這個 ABI 的比較範圍。兩個版本的固定 predicted bytes 與 expected semantic hashes 全部相同。

| 獨立 kernel graph replay | baseline | optimized |
| --- | ---: | ---: |
| F04b | 143.567 µs | 102.811 µs |
| update | 77.954 µs | 74.578 µs |

端到端比較兩個 binary 都啟用 CUDA Graph，預載完全相同 JPEG。每個 fresh session 前 20 幀暖機，再量接續 120 幀；ABBA 四次程序，各自包含兩個 single-frame sessions 與兩個 queued sessions。每版本 **480 個單幀樣本**，p95/p99 對實際單幀樣本採 nearest-rank。全部 16 個 sessions 的完整 140-frame packets hash 都是 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`。

既有 Nsight trace 的 batch 平均 642.6 µs 來自整個 pipelined session，31 次呼叫中 26 次與其他 stream 的 TensorRT kernels 重疊；不能將它與獨立 replay 的 143.3 µs 當成同一條件，更不能把兩者差額視為可直接扣除的端到端時間。原始 overlap 查詢結果與 trace hash 保留於本目錄。

只換 CUB radix 的中間版為 F04b 約 112.48 µs；最後才加入已排序壓實及小 batch 分支。另發現 birth 即使不進 radix 分支，中間版 kernel 的 Nsight static shared storage 仍為 9,284 bytes；最終移除該 collective，source 只保留 4-byte 已排序 flag，未排序 caller 沿用原 insertion。相同 `cuobjdump` 口徑的 update SHARED 分別為 baseline **0**、中間版 **10,308**、最終版 **1,028 bytes**，registers 維持 128、local memory 為 0。source 的宣告大小與工具回報分列，不推定額外配置的來源，也不據此獨立推定端到端因果。`profile/` 的 Nsight CSV 屬中間 CUB-birth 版；最終資源以 `final-resources.txt` 為準。

中間版 source / binary / replay 結果保留在 `build/tracking-sort-evidence/cub-variant/`、`adaptive-cub-birth/`、`replay-*`、`adaptive-replay-*`，不是最終 baseline。最終比較使用 `final-replay-*` 與 `final-latency-*`；兩版都啟用 CUDA Graph。

## 驗證與保留邊界

- 新增 counts 0/1/31/32/33/127/128/129/255/256/257/300、full uint32 ID / `UINT32_MAX`、同分、±0 / 負分、未排序 birth、混合 flags、跨 255/256 inversion、cap 截斷及 sorted→unsorted→numeric→sorted 重用。
- 相關 CTest 5/5 PASS：tracking CUDA、七序列 recorded CPU/CUDA 對照、packet、MOT17-10 完整 T01 session、ordinary/graph lifecycle。
- memcheck / initcheck / racecheck / synccheck：各 7,599,964 checks，0 errors / 0 hazards；四個正控制均以退出碼 99 與對應預期診斷確認工具有作用。initcheck 關閉 API copy memory-access 檢查，因 harness 會下載固定容量結構的非契約 tails；kernel 讀取仍完整插樁。
- HR MOT17-10 全 654 幀的 graph pipelined / graph serial / ordinary pipelined，完整 packet 與 MOT bytes 均與原排序 binary 相同。packet hash `f5cb13847999e4bdbb2ce42457884ce9d1e75974a795d6b026e57474a6ee4216`，MOT hash `1fb713f30444d2c6e3e9ae30d34621f3155b0967b3a52a9244f107e28547fba7`。

沒有重新選模型、調 tracker 或重算品質分數。本次不是 clean-source 正式驗收，也沒有宣稱任意輸入、其他 GPU/SDK 或完整 TensorRT/nvJPEG 的 sanitizer 驗收。

## 重跑與檔案

`tracking_batch_benchmark` 隨 CUDA build 提供：

```sh
flock ~/.cache/saccade-v1/gpu.lock build/tracking-sort/tracking_batch_benchmark \
  /absolute/HR/MOT17-10-SDP.scdr /absolute/MOT17-10-SDP.scmo \
  build/NEW-REPLAY --frames 140 --iterations 8 --rounds 3
```

固定 replay bins、完整 packet bins、baseline / optimized binaries 與原始 Nsight 檔留在 `build/tracking-sort-evidence/`、`build/tracking-sort-baseline/`、`build/tracking-sort/`。本目錄封存比較 JSON、逐幀/逐輪 CSV、命令腳本、source snapshot / hashes、binary hashes 與驗證 logs；以 `SHA256SUMS` 檢查。
