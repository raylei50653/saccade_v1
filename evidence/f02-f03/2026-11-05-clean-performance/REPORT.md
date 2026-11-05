# F02／F03 clean-source ABBA 效能量測

2026-11-05。**量測與等價守門 PASS，`source_dirty=false`。**
固定加速前 `f4865cdb27f0e624b8cc26219e78f4aa938690df` 與加速後
`23ade536906d07ccf09859f1b580b044dafe95dc`，使用原 `v1-motion/1`（cuFFT motion）。
這是 clean-source 效能補驗；原 [整合正確性證書](../2026-11-05-clean-23ade53/REPORT.md) 與開發量測保持原 bytes／來源身分。

## 結果

| 指標 | baseline | candidate | 時間降低 |
| --- | ---: | ---: | ---: |
| 隔離 F02 ordinary interval | 148.490000 µs | 87.931700 µs | 40.78% |
| 隔離 F02 Graph interval | 147.138000 µs | 88.483100 µs | 39.86% |
| 完整 F03 ordinary | 6.557451 ms | 6.195495 ms | 5.52% |
| 完整 F03 Graph | 5.241696 ms | 4.904589 ms | 6.43% |
| tracking single mean | 7.092724 ms | 6.626377 ms | 6.58% |
| tracking single p50 | 7.045901 ms | 6.568929 ms | 6.77% |
| tracking single p95 | 7.524795 ms | 7.059455 ms | 6.18% |
| tracking single p99 | 7.719933 ms | 7.312254 ms | 5.28% |
| queued wall interval | 5.536401 ms/frame | 5.123390 ms/frame | 7.46% |

Queued throughput 為 **180.623 → 195.183 FPS**，
吞吐增加 **8.06%**；wall interval 不是單幀交付延遲。
各 backend 的八次 queued interval 範圍為
5.441172–5.646719 與
5.043284–5.189516 ms/frame。

兩輪外層 ABBA 分開計算如下；所有長尾保留，沒有刪除 outliers。

| workload／mode／round | 平均時間降低 |
| --- | ---: |
| model／ordinary／第 1 輪 | 5.266% |
| model／ordinary／第 2 輪 | 5.769% |
| model／graph／第 1 輪 | 6.894% |
| model／graph／第 2 輪 | 5.984% |
| tracking／single／第 1 輪 | 7.622% |
| tracking／single／第 2 輪 | 5.523% |
| tracking／queued／第 1 輪 | 7.058% |
| tracking／queued／第 2 輪 | 7.863% |

完整 samples、每個程序與每輪統計、最大的 20 筆單幀 latency 見 [analysis.json](analysis.json)、
[model-samples.csv](model-samples.csv)、[tracking-runs.csv](tracking-runs.csv) 與 [tracking-latencies.csv](tracking-latencies.csv)。

## 固定條件與量測範圍

- GPU：RTX 5070 Ti Laptop，UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`；WSL2、driver 617.42、CUDA 13.4.92、TensorRT 10.16.1.11。未鎖 clocks；全程持有 GPU lock，保留 [telemetry](capture/gpu-telemetry.csv)。
- Engine 未重建，固定 SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；HR 1280×768 FP16，input／raw tensors 與 scan plugins 為 FP32。
- F02 是 deterministic xorshift RGB 的 1920×1080→1280×768 隔離 interval；四輪 ABBA，每 entry 暖機 40 次、batch 200 次，每 backend／mode 有八個 batch means。F02 Graph 只屬 primitive，TrackingSession 的 F02 仍在 Graph 外。
- 完整 F03 含 backbone／neck、head、scan plugins；固定 input 0 計時，排除 H2D／D2H、初始化、暖機、外部驗證及 profiler。兩輪外層 A→B→B→A；每程序內 ordinary→Graph→Graph→ordinary，每 entry 暖機 40、測 200，各 backend／mode 共 1,600 samples；cache=0，NVTX requested=engine／actual=detailed（兩版相同），沒有 layer profiler。
- Tracking 使用 MOT17-10 前 140 幀：20 幀暖機、120 幀量測，Graph 啟用、result_slots=4；兩輪 ABBA 共 32 fresh sessions。每 backend 有 960 個 single latency samples、八個 queued intervals。JPEG 預載，排除磁碟、constructor、暖機與輸出寫檔；首幀／初始化原值另存，不與 single latency 混算。這不是全七序列含 I/O 的 FPS。

## 正確性與來源保管

- F02 共 128 cases／65,864,364 元素，ordinary／repeat／Graph 全 bits 相同，guards／poison 通過；benchmark-only 的 equivalence=null 保持原樣，等價由獨立 [verify](primitive-verify/summary.json) 決定。
- 兩份 clean build 的 scan／resize suites 通過。四張完整模型的六 raw outputs＋九 statuses，在八個程序間直接逐 byte 比對相同；各程序內 ordinary／repeat／Graph 亦一致。
- 32 份 140 幀完整 packets（每份 1,731,520 B，包含暖機／unused bytes）直接逐 byte 比對相同，SHA-256 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`；非僅比較摘要 hash。
- 兩個 detached checkout 的 HEAD／clean status 在每步前後檢查，完整 tracked source manifests、engine／inputs 與接受 binaries 前後一致。來源、編譯與連結命令、退出碼均保留；大型 payload 與已核驗 binary tar 的位置／hash 見 [local-artifacts.json](local-artifacts.json)。
- Baseline 原 commit 沒有 model benchmark target：共用 driver object 由乾淨 `23ade53` 的 `tools/model_benchmark.cpp` 編譯，baseline model binary 只連乾淨 `f4865cd` 的 runtime libraries。這是明列 component provenance 的共用 harness 對照，不宣稱 baseline executable 全部出自單一 commit。Tracking binaries 各由其完整 clean commit 建置。
- 原 model harness 固定寫 `capture_kind=development-optimization`／`formal_acceptance=false`；raw outputs 與 labels 全保留。此字串不代表 dirty build，實際 compile／runtime identities 均核對 `dirty=false`；外層 contract 與本報告定義本輪 clean-source 效能補驗的範圍。

## 保留界線與重跑

本輪沒有讀 GT、重新計分、重建 engine 或重跑 sanitizer／semantic mutations。
效能限本機、固定 engine／input 與此 prefix，不延伸到任意 workload／跨 GPU／SDK。
現行 G1／`v1-motion/2` 完整 session memcheck 仍 OPEN，完整 HR session racecheck 與原 nvJPEG initcheck 診斷亦保留。
Tracking 結果同時涵蓋 F02＋F03 兩項改動，不能把合併收益分別歸給其中一項，也不能相加 primitive 百分比。

方法：[contract.json](contract.json)、[run.py](methods/run.py)、[analyze.py](methods/analyze.py)、[archive.py](methods/archive.py)。
重跑須把 methods 放入新的獨立 campaign 目錄、確認 source 路徑與乾淨 commit，並在
`flock ~/.cache/saccade-v1/gpu.lock` 下執行；所有輸出目錄必須全新，不能覆用本次 capture。
`run.py` 的 GPU campaign 與 `archive.py` 的目的地是本次固定路徑，重跑前須改成新的路徑；不可直接在 evidence 目錄執行。
Archive helper 不執行 GPU／build，僅在 campaign 與 hashes 通過後封存。所有封存檔由 `SHA256SUMS` 核驗。
