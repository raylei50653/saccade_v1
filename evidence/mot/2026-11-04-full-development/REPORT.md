# MOT17 完整開發來源測試

日期：2026-11-04。現行工作樹的 HR 1280×768 FP16 模型完成七個 MOT17 SDP 序列共 **5,316 幀**，tracking acceptance、輸出守門與品質計分均 PASS。HR 七序列的 SCDR、native MOT 與 CPU offline MOT 共 21/21 份與 M4 逐位元相同。七序列合計的處理吞吐為 **195.85 FPS**（加權 5.10608 ms／幀）。

受測 HEAD 為 `f4865cdb27f0e624b8cc26219e78f4aa938690df`，`source_dirty=true`、`allow_dirty=true`、`formal_acceptance=false`。來源包含現行 F02 tile fusion 與 F03 scan 修改；這是保留未提交內容的開發來源測試。MOT17-02 已在 M4 開啟，本次記為 **M4 之後的重跑**。部署 engine 與 tracker 設定沿用現行版本。

## 執行結果

| 項目 | 結果 | 證據 |
| --- | --- | --- |
| CPU CTest | PASS 16/16 | [acceptance/cpu-ctest.log](acceptance/cpu-ctest.log) |
| CUDA CTest | PASS 59/59 | [acceptance/ctest.log](acceptance/ctest.log) |
| HR MOT17 10 | CPU offline、serial、pipelined MOT 相同；serial／pipelined packet bytes 相同 | [acceptance/hr-outputs.sha256](acceptance/hr-outputs.sha256) |
| HR 七序列守門 | 身分鏈、offline／sequence、既有輸出 hash、兩次 nsys stream placement PASS | [evaluation/gates.log](evaluation/gates.log) |
| M4 輸出比對 | PASS 21/21 | [summary.json](summary.json) |
| Sanitizer | 17 次 0 errors／0 hazards；4 個正控制皆 exit 99 且診斷符合 | [acceptance/](acceptance/) |
| 品質計分 | 三組與七序列完成；MOT17 02 最後計分 | [evaluation/metrics.tsv](evaluation/metrics.tsv) |
| Stream checker controls | 14/14 PASS | [driver/stream-checker-tests/results.tsv](driver/stream-checker-tests/results.tsv) |

driver 實際開始於 `2026-11-04T20:14:00+08:00`，結束於 `2026-11-04T20:38:40+08:00`，牆鐘歷時 **1480 秒**（24.67 分鐘）。這段時間涵蓋 build、CTest、sanitizer、MOT 產生、profiler 守門與計分，與下方 pipeline 處理時間分開。原始狀態見 [driver/status.txt](driver/status.txt) 與 [driver/run.log](driver/run.log)。

## 品質

依 [TRACKING 第 17 節](../../../docs/TRACKING.md) 分組，不設品質門檻。AP50 為 HR FP16 kept detections 的百分比；其他品質分數維持 TrackEval 的 0–100 尺度，IDSW 為次數。MOT17-02 在其他序列之後計分。

| 組別 | HR AP50 % | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val MOT17 10 | 86.0156 | 49.676 | 54.056 | 45.894 | 62.962 | 65.044 | 66 |
| eval MOT17 02 M4 之後 | 72.9086 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| in domain 04 05 09 11 13 | — | 68.423 | 74.454 | 63.148 | 76.136 | 84.284 | 231 |

| 序列 | HR AP50 % | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-SDP | 72.9086 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| MOT17-04-SDP | 93.0675 | 74.802 | 83.170 | 67.376 | 80.578 | 92.464 | 71 |
| MOT17-05-SDP | 97.3619 | 57.999 | 58.981 | 57.559 | 69.842 | 66.329 | 48 |
| MOT17-09-SDP | 97.3628 | 57.724 | 69.782 | 47.909 | 65.467 | 81.822 | 24 |
| MOT17-10-SDP | 86.0156 | 49.676 | 54.056 | 45.894 | 62.962 | 65.044 | 66 |
| MOT17-11-SDP | 97.3267 | 65.821 | 70.249 | 61.768 | 74.569 | 78.317 | 19 |
| MOT17-13-SDP | 92.6990 | 54.361 | 59.051 | 50.267 | 67.848 | 67.497 | 69 |

本次八列品質彙整與 M4 相同，AP50／HOTA／DetA／AssA／IDF1／MOTA／IDSW 的差值均為 0。精確 AP50、T01 AP50 對照與逐列數值見 [metrics.tsv](evaluation/metrics.tsv)；FP／FN、identity counts 與完整表見 [TrackEval](evaluation/trackeval/) 及 [summary.json](summary.json)。M4 原始對照為 [metrics.tsv](../../m4/2026-11-03-formal-cad5298/evaluation/metrics.tsv)。

## 處理速度

每個序列各跑一次，沒有開 profiler。`track_sequence` 在 session 建構後、第一張 JPEG 讀取前開始計時，在 `finish()` 與全部 ResultPacket 收回後停止；包含讀圖、host submission 與 pipeline，排除 engine 初始化、`session.close()`、MOT／packet 檔寫入。實作見 [track_sequence.cpp](../../../tools/track_sequence.cpp)。

| 序列 | 幀數 | 本次 ms／幀 | 本次 FPS | d21a144 ms／幀 |
| --- | --- | --- | --- | --- |
| MOT17-02-SDP | 600 | 5.10690 | 195.81 | 5.53377 |
| MOT17-04-SDP | 1050 | 5.12717 | 195.04 | 5.52718 |
| MOT17-05-SDP | 837 | 5.06663 | 197.37 | 5.42516 |
| MOT17-09-SDP | 525 | 5.11406 | 195.54 | 5.50672 |
| MOT17-10-SDP | 654 | 5.12477 | 195.13 | 5.80823 |
| MOT17-11-SDP | 900 | 5.09016 | 196.46 | 5.49970 |
| MOT17-13-SDP | 750 | 5.11711 | 195.42 | 5.52472 |

本次七序列回報的處理時間合計 `27143.90 ms`；以總幀數除以總時間得到 **195.85 FPS**。這是 pipelined 吞吐，並非單幀交付 latency。原值見 [timing.tsv](evaluation/timing.tsv)。d21a144 是先前獨立 capture 的單次觀察，見 [歷史 timing.tsv](../../cuda-graph/2026-11-04-gate-d21a144/mot17-run/timing.tsv)；兩次沒有控制 clock／溫度或做 ABBA，並列數值不作為可歸因的加速比。歷史報告寫入 MOT 寫檔的措辭，以現行 executable 的上述計時邊界為準。

## 輸出與驗證範圍

T01 640×640 reference engine 的七序列測試涵蓋全部 5,316 幀，CPU reference、ordinary serial 與 graph pipelined 的固定容量 packet bytes 相同，另檢查 ring controls；完整行保存在 [LastTest.log](acceptance/LastTest.log)。這項 packet 測試使用 T01 engine。

HR FP16 的 MOT17-10 全部 654 幀另比較 serial／pipelined packet bytes，SHA-256 為 `f5cb13847999e4bdbb2ce42457884ce9d1e75974a795d6b026e57474a6ee4216`；CPU offline、serial 與 pipelined MOT 相同。HR 的七個完整序列均另外產生 detection SCDR、native MOT 與 CPU offline MOT，七序列 offline／native MOT 逐位元一致；全七序列的 HR serial／pipelined packet 對照不在這次範圍。

HR 七序列的 SCDR、native MOT 與 CPU offline MOT 共 21/21 份與 M4 逐位元相同。全部 21 份本機檔案在生成本報告前再次核對本次 [outputs.sha256](evaluation/outputs.sha256)；M4 比較依據 [M4 outputs.sha256](../../m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256)。

Sanitizer 的 17 次成功執行分為 12 個 kernel suites、3 個 bounded session memcheck、2 個 filtered session racecheck／synccheck。kernel suites 為 tracking packet、tracking CUDA、motion CUDA，各跑四種工具；initcheck 關閉 API memory copy tail 檢查，motion 使用 synthetic 案例避開既有 nvJPEG 診斷。三個 session memcheck 分別是 T01 前 30 幀、T01 lifecycle、HR FP16 前 30 幀，包含該 bounded run 的 TensorRT／nvJPEG／cuFFT kernels。session racecheck／synccheck 僅對 T01 前 10 幀的 tracking／motion／session control kernels 插樁；不含 TensorRT 內部、nvJPEG、cuFFT、F02 resize 或 scan kernels。四個植入缺陷的正控制均 exit 99，且各有預期診斷，原始命令與 log 在 [acceptance](acceptance/)。

本次沒有執行完整 session initcheck、TensorRT 內部 racecheck／synccheck、全幀 HR sanitizer 或新的 semantic mutation suite。既有 nvJPEG initcheck 診斷維持原狀，見 [PREPROCESS](../../../docs/PREPROCESS.md)。

## Stream 判定與執行問題

在評估前發現 stream checker 只辨識舊 F02 `horizontal`／`vertical`，而現行 1920×1080→1280×768 路徑改用 `tiled_hr`。本次只在 F02 的名稱規則加入 `tiled_hr`，仍要求 F02 位於 D stream；[原始 diff](source/checker.diff) 與 [受測 checker](source/check_stream_placement.py) 保留。其餘 D／M、graph content、launch 配對與 unknown-kernel 規則不變。

checker replay 的 14 案全部符合預期：原始 real-capture 的八個負控制與 unchanged 正例，加上受控 fused fixture 正例、node／graph 的錯 stream、缺少 F02、unknown kernel 四個負控制。fused fixture 是刪除舊 horizontal、將 vertical 改為精確 `tiled_hr` 名稱的 sqlite 副本；這些 fixture 只驗證 checker。實際新 kernel 的 stream 判定由本次 node／graph 兩個真實 nsys capture 通過，見 [stream-placement.txt](evaluation/stream-placement.txt)。重播腳本與全部 verdict 在 [stream-checker-tests](driver/stream-checker-tests/)，synthetic sqlite 不封存。

capture 中除四個預期 exit 99 的 sanitizer 正控制外，沒有非預期非零退出。checker 相容性在完整 MOT 評估前修正，沒有將未執行的舊 checker 失敗記為一次 capture 失敗。

第一次報告生成因 parser 只接受舊 CTest summary 格式而 REFUSED／exit 1；修正為支援新舊格式，並核對 CPU 16 與 CUDA 59 個 Passed 結果行後重新生成。這是報告工具問題，沒有重跑 GPU 測試或計分，紀錄見 [report-first-attempt.json](driver/report-first-attempt.json)。

## 來源與產物身分

環境：`Linux 6.18.33.1-microsoft-standard-WSL2 x86_64 GNU/Linux`；GPU capture：`NVIDIA GeForce RTX 5070 Ti Laptop GPU, GPU-825ae551-ba16-539a-e444-22726b59ee9a, 617.42, 11944 MiB`。工具版本與完整環境見 [environment.log](acceptance/environment.log)。engine 為 60,753,004 bytes，profile `hr-1280x768`；model input／raw outputs 與 scan 邊界維持 FP32，backbone／head tactics 為 FP16，aux streams 為 0。engine 最初建置仍保留既有 `build_dirty=true` 身分，並以固定 engine SHA-256 識別。

| 對象 | SHA256 或 revision |
| --- | --- |
| HEAD | `f4865cdb27f0e624b8cc26219e78f4aa938690df` |
| Acceptance source digest | `84a2130b133c6572d6104eefc64de89a77dcab08378178db0d49732c4ffa23c6` |
| Evaluation source digest | `f44035e667db562ba8dc787ea8be925e4bdfde802cb49bac02e49fa4fca626c4` |
| HR FP16 engine | `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` |
| Backbone ONNX | `cc1fa6d35f45f5d23b6a0d350ff7f0101d4a4593debe6a5b590fc89a63151cf7` |
| Head bundle | `c6f3416f261d1eab26751c9a8ee63144f43620cdd2e77cec079be8e9cc1ffdea` |
| Deploy manifest | `d13434afb6246fa67467d9f9758cf897308c229bb0be69acb7275fe4a5410989` |
| Acceptance binary manifest SHA256 | `1b32787cef2dd2e3f69b7cc77469ec374574e1e79a0579e99f9d8c7abfe4d9d7` |
| Acceptance artifact manifest SHA256 | `5f64bb19f584a60be25864a29b21db2c4167bc6f4a399ae5199fc9b22ee8cb5e` |
| Evaluation artifact manifest SHA256 | `2d3a9474ba446fc626fb1fef9c23d0e35704fd8c27307584a9d31b232a7eae40` |
| Acceptance build input snapshot tar gzip | `864e3758538f3c0d5ff329a3259e9eb9ac8350f64e608f62d0195a76a2a41b87` |
| detect_sequence | `911568990e9c89d30bee507543a7b5bf822a9eed3a98ca56b272a370058b72eb` |
| track_sequence | `432ab2f0bb0512289c74617c33dc6a4bffa73db67d5dd929eb91f674f57847ed` |
| track_offline | `2790ad77ad7033b4ed6204b1c9ee0dd8a8836bd733d7042c83f178a0d06034d2` |
| HR MOT17 10 packets | `f5cb13847999e4bdbb2ce42457884ce9d1e75974a795d6b026e57474a6ee4216` |
| Stream checker | `dfd304b0e30c0fd15c2198268f31721e8f95cb879b6f585f7afb41a85e20e0f6` |

完整逐檔 binary hashes 見 [binaries.sha256](acceptance/binaries.sha256)，前後 manifest 相同；source file hashes、原始工作樹狀態與 patch 分別見 [acceptance source](acceptance/source.txt)、[source-files.sha256](acceptance/source-files.sha256)、[source-status.txt](acceptance/source-status.txt)、[source.diff](acceptance/source.diff) 與 [evaluation source](evaluation/source.txt)。acceptance source digest 不含 stream checker，evaluation source digest 包含已修正的 checker。[source.tar.gz](driver/source.tar.gz) 封存 acceptance build inputs，其明列檔案清單不包含 checker；修正後的評估腳本由 [evaluation source.diff](evaluation/source.diff)、[checker snapshot](source/check_stream_placement.py) 與上述 SHA-256 保存。

engine、deploy manifest、訓練 checkpoint、ONNX／head bundle、reference corpus、M1 records、七序列 image 身分鏈保存在 [acceptance artifacts](acceptance/artifacts.sha256)、[evaluation artifacts](evaluation/artifacts.sha256)、[identity.txt](evaluation/identity.txt) 與 [deploy manifest](evaluation/hr-deploy-manifest.json)。前後 artifact hashes 相同。完整數值與來源欄位保存在 [summary.json](summary.json)。

## 重跑與封存

以下從專案根目錄執行，輸出必須使用尚不存在的新目錄。`--allow-dirty` 明確保存開發來源身分。

```sh
export SACCADE_TRACKING_ENGINE="$PWD/build/t01-f03-evidence/stage/model.engine"
export SACCADE_DETECTION_REFERENCES="$PWD/build/preprocess-evidence/refs"
export SACCADE_TRACKING_SEQUENCES=/home/ray/developer/ai/saccade/datasets/MOT17/train
export SACCADE_TRACKING_RECORDS=/home/ray/developer/ai/saccade_v1-worktrees/m1-data
flock ~/.cache/saccade-v1/gpu.lock bash scripts/run_tracking_session_acceptance.sh build/NEW-acceptance --allow-dirty
flock ~/.cache/saccade-v1/gpu.lock bash scripts/run_mot17_evaluation.sh build/NEW-acceptance build/NEW-evaluation --allow-dirty
bash scripts/score_mot17_evaluation.sh build/NEW-evaluation --allow-dirty
```

本次 [driver/run.sh](driver/run.sh)、報告生成器、acceptance build input snapshot、checker replay、acceptance／evaluation capture、TrackEval summary／detailed 表及 logs 均在本目錄。大型 raw artifacts 留在本機忽略目錄：

- [build/mot-full-20261104-acceptance](../../../build/mot-full-20261104-acceptance/)：CPU／CUDA binaries、HR MOT／SCDR／SCMO、完整 HR10 packet streams。
- [build/mot-full-20261104-evaluation](../../../build/mot-full-20261104-evaluation/)：七序列 SCDR、native MOT、offline MOT、實際 nsys `.nsys-rep` 與 sqlite。
- [build/mot-full-20261104-driver](../../../build/mot-full-20261104-driver/)：執行狀態、source snapshot、checker fixtures 與原始 replay workdirs。

封存排除 raw binaries、重複的 TrackEval tracker inputs、真實／合成 sqlite 與大型 nsys captures。保留的每個檔案由 [SHA256SUMS](SHA256SUMS) 校驗：

```sh
cd evidence/mot/2026-11-04-full-development
sha256sum -c SHA256SUMS
```
