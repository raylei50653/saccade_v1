# Tracking CUDA Graph 與延遲量測

2026-11-04。`TrackingSessionConfig::cuda_graph` 預設為 `true`；
`track_sequence --no-cuda-graph` 可選原有逐次 enqueue 路徑。

文件狀態：實作與 [clean-source 正式驗收](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) 均已完成（受測範圍）。本頁的執行契約與重跑方法配合 [TRACKING](TRACKING.md)；目前部署與未涵蓋項目見 [STATUS](STATUS.md)。

## 執行範圍

- D stream：TensorRT F03 → producer finite/status checks → F04，依 candidate slot 各建立一張 graph。HR 部署 engine 的每張 graph 有 319 個 nodes。
- M stream：候選檢查 → F04b batch → association/update → failure/commit → packet，依 candidate slot 各建立一張 graph，每張有 9 個 kernel nodes。
- nvJPEG/F02、F05、GMC/cuFFT、predict、跨 stream ready/consumed、結果 D2H 與 host ring 仍由原有 session 排程。這是兩段計算路徑的 graph 化，沒有將完整 JPEG session 包成單張 graph。

M graph 在建構時 capture，不執行或提交 frame。D 的實際 FP32 input 地址在首次 preprocessing enqueue 後取得；先 warmup 相同 TensorRT context/address，再 capture/instantiate，之後固定地址並檢查其未變。首幀額外 setup 成本必須與 steady-state 延遲分列。

epoch/frame/mapping/capacity 在 F04 initialize 與 finalize 中按值傳入，故每次 replay 前必須更新**兩個** kernel nodes；不能直接重播首次 frame 的 request。固定 shape、allocation extent、raw pointers 等驗證在 capture 時完成。M 的 frame、failure gate 與 working slot 直接讀 device Control。候選 consumed 在 M tail graph 完成後 record；D replay 之前 wait consumed，仍保留完整 lease。checked close 先 drain 兩條 streams，再銷毀 graphs，最後釋放借用的 context 與 buffers。

TensorRT 含 driver API 建立的 kernel nodes；對全圖逐一呼叫 runtime `cudaGraphKernelNodeGetParams` 曾回報 `invalid device function`。現在沿唯一 terminal node 的 F04 五節點尾鏈定位 initialize/finalize，只查詢本專案擁有的兩個 nodes。找不到預期結構會明確失敗，不猜測 node 順序。

測試專用 `perturb_seed != 0` 保留各原始擾動邊界，使用 ordinary 路徑；producer flag 注入當幀的 D 亦使用 ordinary，M tail 仍可 replay。這些不能混列為完整 graph 擾動覆蓋；另有無擾動 graph ring/lifecycle 與 F04 graph failure/recovery 測試。

## 重跑量測

啟用 detection session 的 Release CUDA/TensorRT build 會提供 `tracking_benchmark`：

```sh
flock ~/.cache/saccade-v1/gpu.lock build/cuda-graph/tracking_benchmark \
  ~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  /absolute/path/MOT17/train/MOT17-10-SDP build/NEW-LATENCY \
  --profile hr-1280x768 --frames 120 --warmup 20 --rounds 2
```

`--warmup` 必須為正，確保首次 capture 不混入暖機後的延遲樣本。

工具預載相同 JPEG，在 fresh sessions 以 ABBA 順序執行 ordinary/graph，分開量單幀 enqueue→blocking receive 延遲與 queued wall time/frame。後者是吞吐換算，不是單幀延遲。每模式 480 個單幀樣本；p50/p95/p99 對單幀採 nearest-rank，包含所有 outliers，不以輪平均取分位數。constructor、首幀、warmup 另列；steady-state 不包含磁碟輸入、輸出檔案寫入、constructor 或 warmup。

所有 sessions 的完整固定容量 packet bytes（含 warmup）必須相同。輸出包含逐幀 CSV、run CSV、JSON 摘要、來源影像/engine hashes、session identities、packet bins 與 PASS/FAIL status。`--only ordinary|graph` 可供獨立診斷，但單一模式結果不能宣稱跨模式等價。

原始效能數字、kernel 審查、測試與限制見 [開發量測報告](../evidence/cuda-graph/2026-11-04/REPORT.md)；該次為未提交開發來源，保留原身分，不改作正式驗收測時。

後續 [clean-source 正式驗收](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) 已補完：步驟 1–3 為乾淨 `4329ed7`，步驟 4–5 為乾淨 `d21a144`，CPU／F04／mutations／tracking 與 stream 守門全部 PASS；HR 七序列的 21 份輸出與 M4 逐位元相同，未讀 GT、未重新計分。Graph 的實際 launch stream 以 node／graph 雙 capture 配對檢查，規則與前提見 [TRACKING §17.3](TRACKING.md#173-正式執行)。完整 TensorRT 內部 racecheck／synccheck、完整 session initcheck 與完整 HR racecheck 等限制仍依 [STATUS](STATUS.md#未涵蓋與未完成邊界) 保留。
