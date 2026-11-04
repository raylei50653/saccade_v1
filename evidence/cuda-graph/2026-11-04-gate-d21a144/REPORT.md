# CUDA Graph 分支的 clean-source 驗收：d21a144

日期：2026-11-04。分支 `cuda-graph-resource-schedule`。受測 source：步驟 1–3 為 `4329ed7`，步驟 4–5 為 `d21a144`，皆為乾淨工作樹。`4329ed7..d21a144` 只改 `scripts/run_tracking_session_acceptance.sh`、`scripts/run_mot17_evaluation.sh`、`scripts/check_stream_placement.py`，步驟 1–3 不使用這三個檔案。GPU：RTX 5070 Ti Laptop（WSL2，driver 617.42）；Compute Sanitizer 2026.3.0、Nsight Systems 2026.3.2。所有 GPU 步驟在 `flock ~/.cache/saccade-v1/gpu.lock` 下依序執行，整個 runner 在 `systemd-run --user --scope -p MemoryMax=36G -p MemorySwapMax=0` 內。

受測內容是 `4329ed7` 的三批開發成果：CUDA Graph（[CUDA_GRAPH](../../../docs/CUDA_GRAPH.md)、[開發報告](../2026-11-04/REPORT.md)）、tracking 排序（[報告](../../tracking-sort/2026-11-04/REPORT.md)）、資源與排程報告（[報告](../../resource-schedule/2026-11-04/REPORT.md)）。三份開發報告都標明自己不是 clean-source 驗收，也沒有跑 HR 7 序列對 M4 的逐位元比對；本次補上這兩項。

**結論：全部 PASS。** HR FP16 在 7 個 MOT17 SDP 序列的 21 個輸出檔與 M4 逐位元相同。過程中有三次驗證腳本跟不上 CUDA Graph 而中止，GPU 程式沒有改動；三次的原始 capture 保留於 [failed-attempts/](failed-attempts/)。

## 結果

| 步驟 | 結果 | 證據 |
| --- | --- | --- |
| 1. CPU CTest（`4329ed7`） | PASS，16/16 | [logs/run1/1-cpu.log](logs/run1/1-cpu.log) |
| 2. F04 驗收（`4329ed7`） | PASS：F04 CTest 5/5；postprocess suite 37,094 checks，memcheck／initcheck／synccheck 0 errors，racecheck 0 hazards；四個正控制 exit 99 | [f04-acceptance/](f04-acceptance/) |
| 3. Mutations（`4329ed7`） | 15/15 DETECTED，診斷皆為預期 | [mutations/results.tsv](mutations/results.tsv) |
| 4. Tracking session 驗收（`d21a144`） | PASS：CPU 16/16、CUDA 59/59，其中 7 個序列的 CPU／ordinary serial／graph pipelined packet bytes 相同；HR MOT17-10 路徑 PASS；tracking／motion／packet kernel suites 四種 sanitizer 皆 0 errors／0 hazards；完整 session memcheck 602 checks、lifecycle memcheck 180 checks、HR memcheck、session racecheck／synccheck 222 checks 皆通過；四個正控制 exit 99 | [tracking-acceptance/](tracking-acceptance/) |
| 5. MOT17 7 序列（`d21a144`） | 守門 PASS：身分鏈、`track_offline` == `track_sequence`、既有 hash、stream 位置；對 [M4 outputs.sha256](../../m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256) 21/21 OK（7 個 SCDR、7 個 MOT、7 個 offline MOT）。未執行計分，未讀 `gt/` | [mot17-run/](mot17-run/)、[logs/run4/5-m4-hashes.log](logs/run4/5-m4-hashes.log) |

## 三次中止與修正

| 次數 | 受測 | 停在 | 原因 | 修正 |
| --- | --- | --- | --- | --- |
| 1 | `4329ed7` | 步驟 4，CTest 之後 | CTest 59/59 通過，但 `tracking_session_tests` 的輸出行已改為 `CPU/ordinary serial/graph pipelined packet bytes identical; graph ring controls=N frames`，acceptance 仍比對舊字串 | `ea877a7`：改比對新的完整行 |
| 2 | `ea877a7` | 步驟 4，session memcheck | 602 checks 通過、沒有記憶體錯誤，但預設 extended API 報告把 TensorRT 在 `TrackingSession` 建構時載入 kernel module 的 driver 提示 `Selective device code recompilation in progress` 計為 1 error，exit 99 | `e7027c3`：三個完整 session memcheck 改用 `--report-api-errors explicit`，與 S04 相同（[S04 第三次 capture](../../s04/2026-10-30/REPORT.md)）；kernel suite 的 sanitizer 不變 |
| 3 | `e7027c3` | 步驟 5，stream 位置 | 21 個輸出已與 M4 相同。`check_stream_placement.py` 寫於 CUDA Graph 之前：node 追蹤把 graph kernels 放在合成的 profiler streams 28–31，重播的 TensorRT kernels 也不在 `enqueueV3` NVTX 範圍內 | `d21a144`：stream 位置改為兩次 capture，見下節 |

第 2 次的修正是在看到失敗後才改 sanitizer 選項。依據是 S04 對同一則 driver 提示的既有處理，且 explicit 模式仍回報所有記憶體錯誤與應用程式收到的 API 錯誤；第 2 次的原始 log 保留於 [failed-attempts/tracking-ea877a7/session-memcheck.log](failed-attempts/tracking-ea877a7/session-memcheck.log)。

## Stream 位置的新判定

node 追蹤的 sqlite 不記錄 `cudaGraphLaunch` 使用的 stream。`run_mot17_evaluation.sh` 因此對同一段 MOT17-10 前 30 幀再以 `--cuda-graph-trace=graph` 擷取一次（每次 launch 一列，帶真實 stream），並要求兩次 capture 寫出相同的 MOT 檔。

- graph 外的 kernels 沿用原規則：`enqueueV3` 範圍內為 engine，其餘依名稱分類；D 為 F02 的 stream，M 為 graph 外 tracking kernels（predict）的 stream。
- 每張 graph 依內容只能是 D（TensorRT、producer、F04）或 M（control、tracking、packet）；D graph 內不符合任何名稱規則的 kernel 才算 engine。
- 兩次 capture 的 launch 依 host 呼叫順序逐一配對，graph ID 必須一對一；D graph 的每次 launch 必須在 D stream，M graph 在 M stream。

本次結果：graph 8、11 為 D，launch 在 stream 13；graph 2、5 為 M，launch 在 stream 14；各 15 次，共 60 次（[stream-placement.txt](mot17-run/stream-placement.txt)）。

負控制以本次兩份 sqlite 的副本執行，[mutations.sh](placement-checker/mutations.sh) 可重跑：8 個改壞的副本（graph 改到另一條 stream、異類 kernel 放進 graph、D／M 混合的 graph、launch 對調或缺少、F02 或 predict 搬到另一條 stream）全部 exit 1，未改動的副本 PASS（[results.tsv](placement-checker/results.tsv)）。

這項判定仍以「兩次執行 launch 順序相同」為前提，由相同 binary、輸入與相同輸出支持；[TRACKING §17.3](../../../docs/TRACKING.md) 仍描述單次 capture 的舊規則，留待文件整理。

## 速度（記錄）

步驟 5 的 7 個 `track_sequence` pipelined 執行未開 profiler，wall time 含 JPEG 讀檔與 MOT 寫出；每序列各跑一次，不是 ABBA。與 F04 第二輪驗收（`e4d45de`，同一腳本與條件）並列：

| 序列 | `e4d45de` ms/幀 | `d21a144` ms/幀 |
| --- | ---: | ---: |
| MOT17-02 | 6.987 | 5.534 |
| MOT17-04 | 6.914 | 5.527 |
| MOT17-05 | 6.789 | 5.425 |
| MOT17-09 | 6.940 | 5.507 |
| MOT17-10 | 6.957 | 5.808 |
| MOT17-11 | 6.910 | 5.500 |
| MOT17-13 | 6.918 | 5.525 |

這些是單次觀測，不作速度門檻，也不取代開發報告的 ABBA 延遲量測。

## 檔案

- [logs/](logs/)：四次 runner（`cg-accept*.sh`）、終端輸出與各步驟 log；run1 含步驟 1–3 與第 1 次中止。
- [f04-acceptance/](f04-acceptance/)、[mutations/](mutations/)、[tracking-acceptance/](tracking-acceptance/)、[mot17-run/](mot17-run/)：通過的 capture。
- [failed-attempts/](failed-attempts/)：三次中止的原始 capture。
- [placement-checker/](placement-checker/)：stream 位置負控制。
- 完整工作目錄（binaries、packets、輸出、nsys `.nsys-rep`／sqlite）在本機 `build/cg-*`，不入 Git。

檔案清單以 [SHA256SUMS](SHA256SUMS) 核對。
