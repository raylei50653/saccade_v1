# OAO duration-ramp / front-occluder：固定輸入消融

登記日期：2026-11-05；在本次四臂結果產生前固定。開發來源實驗，非正式驗收。

## 問題與範圍

在目前 HR detector、G1 NVOFA GMC、continuation、bridge 與預設 tracker 上，
恢復舊專案的兩個遮擋懲罰是否改善 MOT17-10 的關聯品質？
只跑 MOT17-10-SDP（val，654 幀）；不讀 MOT17-02、in-domain 或 PersonPath22 GT。
不改 detector、GMC、birth、bridge、interpolation 或其參數。

輸入固定為 `../saccade_v1-worktrees/hr-records-v2/MOT17-10-SDP.{scdr,scmo}`。
SCMO 是目前 v1-motion/2 的 G1；不使用 PCR 時期的 M4 評估腳本。
baseline MOT 必須與該目錄的同名 `.txt` 完全相同；否則停止。

## 四個實驗臂

| 臂 | OAO | front |
| --- | --- | --- |
| baseline | off | off |
| oao | on | off |
| front | off | on |
| both | on | on |

使用舊 `mamba_whole_graph_m` preset 的固定值，不以本次結果掃描參數：
OAO tau 0.50、ramp 25；front IoU 0.45、normalized foot gap 0.15、TTL 4、weight 0.50。
來源為舊 `src/tracking/tracker_gpu.cu` 的 `compute_track_occlusion_kernel` 與
`stage1_cost_fused_kernel`，設定為 `configs/presets/mamba_whole_graph_m.yaml`。

每幀在 GMC＋predict／移除之後、association 之前，以所有存活 predicted tracks
（含 tentative／lost）計算每個 track 與其他 track 的最大 IoU，最大值同分取較小 UID。
重疊大於零就增加 consecutive duration，無重疊歸零：
`OAO = 0.50 * max_iou * min(1, duration/25)`。
最大 overlap partner 的腳點較高，且本 track 的腳點差／平均高度在 `(0,0.15]`，
最大 IoU ≥ 0.45 時，本 track front TTL 設為 4；否則每幀減一至零。
TTL > 0 時，`front = 0.50 * max(0, (predicted_foot - detection_y2)/max(track_h,1e-3))`。

沿用 v1 的 FP32 共用數學與原 stability reward R：
`cost = clamp(1 - IoU * det_expf(-(OAO + front - R)), 0, 1)`。
懲罰為零時直接使用原 association_cost，維持 baseline bits。
不加入舊專案的 score weighting、contention、union coverage、velocity 或其他 gate。

duration／TTL 按 UID 保存，birth 為零、移除即丟棄；bridge 不繼承 lost UID 的遮擋狀態。
這是 v1 生命週期下的明確選擇，不宣稱整個舊 tracker 的逐位元重現。

## 驗證與決策

- 單元測試覆蓋 duration reset／ramp、front direction／latch、cost integration 及 baseline。
- 全新 CPU build／CTest；baseline 的每幀完整 packet 亦比對原 TrackerReference。
- 四臂使用同一 binary、相同輸入，各自保存 MOT、診斷、TrackEval 原始表與 log。
- 指標完整記錄 HOTA、DetA、AssA、IDF1、MOTA、IDSW；以 HOTA、AssA、IDF1
  與 IDSW 的方向評估，混合結果如實列出，不宣稱跨場景收益。
- 本輪不自動改部署預設。若 val 顯示值得跟進，另做 CUDA parity、完整 session、
  held-out 範圍與端到端效能驗證。
- CPU replay 耗時只屬該程式執行，不當作 GPU／產品 FPS。

保留 source commit／dirty diff、source／binary／input／output hashes、canonical tracker
config、實驗參數及命令。既有正式證書與 NVOF memcheck OPEN 不因本輪改寫。
