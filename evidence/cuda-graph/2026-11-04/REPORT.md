# CUDA Graph：tracking 延遲與 kernel 結論

2026-11-04；開發來源驗證。來源基底 `865bb105b2b04e5e51a09c7106952184f5b2a9e1` 加本次未提交修改；不是 clean-source 正式驗收。HR 部署 engine 保持 SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`。

## 結果

F03→producer→F04 與 M stream tracking tail 已接上 CUDA Graph，`TrackingSession` 預設啟用。D 每 candidate slot 一張 319-node graph，M 每 slot 一張 9-kernel graph；每幀兩次 replay。nvJPEG/F02、F05、GMC/cuFFT、predict、跨 stream leases 與結果 D2H 保留原有排程，沒有宣稱完整 session 是單張 graph。設定、首幀 setup 與重跑入口見 [CUDA_GRAPH](../../../docs/CUDA_GRAPH.md)。

受測環境是 RTX 5070 Ti Laptop GPU（SM 12.0）、CUDA 13.4.92、TensorRT 10.16.1.11、WSL2；HR 1280×768 FP16，MOT17-10-SDP 的同一批 JPEG。未鎖 clock；數字限本機本次測量。

| 指標 | ordinary | CUDA Graph | 變化 |
| --- | ---: | ---: | ---: |
| 單幀 enqueue→receive 平均 | 8.591 ms | 7.059 ms | −17.83%，1.217× |
| 單幀 p50 | 8.585 ms | 7.001 ms | |
| 單幀 p95 | 9.083 ms | 7.443 ms | −18.05% |
| 單幀 p99 | 9.268 ms | 7.799 ms | |
| queued wall time/frame | 6.810 ms | 5.348 ms | 吞吐 1.273× |
| 首幀 enqueue→receive（含 lazy capture） | 9.957 ms | 16.520 ms | 首幀成本增加，另列 |

JPEG 預先載入，量測包含 JPEG host 提交、native preprocessing、模型、tracker、完整 packet D2H 與 receive；不含磁碟 I/O、MOT 檔案輸出、constructor 或 warmup。各 fresh session 先以序列前 20 幀暖機，再量接續 120 幀；兩輪 ABBA，ordinary/graph 各 **480 個單幀樣本**。同一模式的四次 queued sessions 另量吞吐，不能將其 wall time/frame 當成單幀延遲。p95/p99 是單幀 nearest-rank，沒有刪除 outliers。constructor 數值保留在 CSV；首個 ordinary constructor 包含程序初次 CUDA/TensorRT 載入，不能把兩組 constructor 平均差異當作 graph 初始化加速。

全部 16 個 sessions 的 140 幀完整固定容量 packet bytes（含 warmup）相同，SHA-256 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`。原始摘要、每幀樣本、命令與來源影像 hashes 在 [latency](latency/)。

表中採最後交付 binary 的結果。較早的同條件完整 ABBA 量測為 8.633→6.983 ms（−19.11%），保留於 [latency-initial](latency-initial/)；兩次均保留完整樣本，沒有按較快的一次選報。最後 rebuild 後另重跑 HR 654 幀，完整 packets/MOT 仍與既有 hash 相同。

## Kernel 與實作結論

1. **剩餘自訂 kernel 瓶頸是 scan 與 F04b batch。** 新 graph trace 的 `scan_kernel` 平均 388.5 µs/次，占被記錄 GPU kernel time 17.7%；`batch_kernel` 平均 642.6 µs/次，占 9.5%；F04 finalize 為 51.5 µs。trace 是 30 幀 prefix，包含建構/首幀 warmup，不能當成純 steady-state 的 kernel 分位數，也不能直接和沒有 profiler 的 wall latency 相減。
2. `cuda/tracking.cu` 的 batch 對 suppressed pool 做平方次 rank、對 priors/kept 做 IoU 掃描；thread 0 串行處理 continuation dedup，最後再做平方次 candidate-ID rank-sort。這些是已確認的實作工作，CUDA Graph 沒有消除它們。
3. `cuda/selective_scan.cu` 沿 token length 串行遞推，每步執行 softplus/exp、縮放與 state reduction。predict/update 也仍有單 thread 的大結構壓實；update 對已按 candidate ID 排好的 batch 再做 insertion sort。這些保留為後續有界優化項，本次沒有改變數值運算順序或 tracker 政策。
4. **已修正 graph 整合時實際遇到的 API 問題。** TensorRT 的 driver kernel nodes 無法全部用 runtime `cudaGraphKernelNodeGetParams` 查詢，初次實跑回報 `invalid device function`。現在只從圖的唯一 sink 回溯 F04 五節點尾鏈，並核對 initialize/finalize 的函式，避免查詢 TensorRT nodes。
5. F04 的 request 以值捕捉，直接 replay 會重用舊 frame；現在每幀更新 initialize 與 finalize 兩處，M graph 則讀 device Control。八階段測試覆蓋 epoch/frame/mapping/capacity 變更、成功、overflow、numeric、upstream、protocol、空輸出與恢復。未找到可由本次測試確立的新 kernel 數值正確性錯誤；不能將有限測試解讀為任意輸入的證明。

graph 加速包含減少 host 提交與省下每幀重複的 immutable pointer/shape 驗證，不能全部歸因於 kernel launch overhead，也沒有宣稱 kernel 演算法本身變快。

## 驗證與範圍

- CPU CTest：16/16 PASS。
- 本次相關 CTest：22/22 PASS；包含七個 MOT17 SDP 序列的 T01 CPU/ordinary serial/graph pipelined 完整 packet 比對、graph ring `1/1/8` 與 `3/3/5`、ordinary/graph sticky failure、committed state 凍結、backpressure、cancel、D-only tail、destructor、checked repeat close。
- HR graph 30 幀完整 session 的 compute-sanitizer memcheck：0 errors。
- 新增 F04 graph metadata tests：T01/HR/LAZY CTest 3/3 PASS；HR memcheck 44,127 checks、0 errors，held-stream callbacks 在此 sanitizer run 為 UNEXECUTED。
- HR MOT17-10 全部 654 幀：ordinary pipelined、graph pipelined、graph serial 的完整 packet 與 MOT bytes 相同；MOT SHA-256 保持既有 `1fb713f30444d2c6e3e9ae30d34621f3155b0967b3a52a9244f107e28547fba7`，packet SHA-256 為 `f5cb13847999e4bdbb2ce42457884ce9d1e75974a795d6b026e57474a6ee4216`。
- Nsight 30 幀 prefix：graph/ordinary 的 MOT bytes 相同；graph trace 確認有 60 次 `cudaGraphLaunch` 與 60 次 metadata node 更新，對應每幀兩張 graph 與兩個 request nodes。此 trace 含 setup，不拿總 API calls 直接除以 frame 作 steady-state 提交成本。
- CLI `--warmup 0` 明確拒絕，避免將 cold capture 誤標為暖機後延遲。

`perturb_seed != 0` 為保留原始擾動邊界而執行 ordinary；producer flag 注入的 D 當幀亦為 ordinary，後續 M graph 的 sticky failure 另外實測。沒有將這些擾動測試冒充 graph 擾動覆蓋。完整 session 的 initcheck/racecheck/synccheck 未在本次重新執行，nvJPEG 既知診斷仍依既有使用政策處理。未重新選模型、調 tracker、計算品質分數或宣稱跨裝置支援。

完整本地工作目錄為 `build/cuda-graph/`，保留原始 packet bins、兩份 Nsight `.nsys-rep`/SQLite、編譯與初次失敗嘗試；本目錄封存可攜的摘要、samples、trace CSV、source/binary hashes 與驗證結果。測量時的 binaries 另保留於 `build/cuda-graph/measured-binaries/`。檔案清單以 `SHA256SUMS` 核對。
