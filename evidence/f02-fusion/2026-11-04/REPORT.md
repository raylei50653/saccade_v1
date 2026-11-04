# HR F02 shared tile fusion

2026-11-04。本輪將 **1920×1080 → 1280×768** 的 F02 兩 pass 改成單一 shared-memory tile kernel，並在 registers 重用水平權重。FP32 normalization、水平捨入與兩軸累加順序保留。其他尺寸仍走原兩 pass。

這是基底 `f4865cdb27f0e624b8cc26219e78f4aa938690df` 加工作樹修改的**開發來源實作與量測**。原本已有的 README 修改與 `evidence/kernel-analysis/` 保留；沒有 commit／push、重建部署 engine、讀 GT 或重新評分。先前 clean-source 證書維持各自的受測來源，不移作本輪證書。

## 1. 實作與數值等價

[`cuda/resize.cu`](../../../cuda/resize.cu) 使用 `32×8` threads／輸出 tile，每 block 處理單一 channel。以 tile 首尾輸出列的原 `span()` 找出來源列聯集，先算水平 FP32 結果到 `shared float[16][32]`，經唯一、無條件的 `__syncthreads()` 後才讀取並產生垂直輸出。

- 固定比率 `sx=3/2`、`sy=45/32`；本尺寸的中心／span 座標可精確表示。原 `span()` expression 與算術旗標不變。
- 水平最多三 taps，邊界兩 taps；其正規化 FP32 權重在每個 thread 的 registers 中重用。先以 FP32 reciprocal-255 乘 uint8，再做原本的第一項 `__fmul_rn`／後續 `__fmaf_rn`。
- 水平 shared store 保留 FP32 pass 邊界；垂直仍按原 taps 順序做第一項 multiply／後續 FMA。沒有改成 tensor-product reduction、FP16 或近似插值。
- 共 96 個 y tiles，其中 50 個需 12 列、46 個需 13 列，均小於配置的 16 列。grid 與輸出尺寸整除，有 `static_assert`；每個使用到的 shared element 及 output 都有唯一 writer。

完整枚舉與 payload 模型見 [traffic-model.json](traffic-model.json)，可用 [traffic_model.py](traffic_model.py) 重算。水平結果由 4,147,200 增為 4,600,320，halo 重算約 **10.9%**；來源層 global logical payload 由 **73,967,280** 降至 **25,590,252 B/frame**。原 scratch 的 write＋tap-read 為 49,735,680 B，本輪移除這些來源層請求，但新增部分 RGB halo reads。**這不是 DRAM bytes 的量測或保證**。

編譯資源為 **34 registers/thread、2,048 B shared/block、0 stack／spill**，詳 [compiler-resources.log](compiler-resources.log)。原兩 pass 各 36 registers。所有 enqueue 前的容量、pointer／device、overlap、stream 驗證都保留。

`resize_workspace_elements()` 與 caller allocation 維持原 API，所以 16,588,800 B workspace 仍配置；融合分支不讀寫它。**本輪降低 global intermediate 流量，沒有降低配置容量。** 沒有改動 RGB／F05 的最後 reader、nvJPEG 前的 host wait、D/M events、candidate／MI slot 回收或 tracking commit。

## 2. 比較與測試

| Gate | 本次結果與證據 |
| --- | --- |
| Frozen baseline／candidate | [128 組 shape／pattern](primitive/verify/summary.json)，共 **65,864,364 FP32 元素**；ordinary、repeat、Graph replay 全 bits 相同；output／workspace NaN poison 與前後 guards 通過 |
| HR Python reference | [完整 CTest log](LastTest.log) 的 `preprocess_hr_reference`：原固定 4 張 HR JPEG，RGB 與 FP32 tensor 全 bytes 相同，26 checks PASS；沒有重建 reference 或調整門檻 |
| F02 unit tests | 72 組 binary64 reference cases，含 HR 四 patterns；repeat、Graph、Graph 換 input、HR／fallback held-stream 輸入更新順序均通過；binary64 bound 與舊／新 bitwise 對照分開判定 |
| 完整 build／CTest | [60/60 PASS](ctest.log)，504.51 s；含原 T01 七序列 CPU／ordinary serial／Graph pipelined 完整 packets 及 ring controls、lifecycle、HR preprocessing reference |
| HR pipeline A/B | [32 個 fresh sessions](pipeline/summary.json)，每個使用同一 140 幀 prefix（20 warmup＋120 measured）；[32 份完整 packet 檔逐 byte 比較相同](pipeline/packet-byte-comparison.json)，含 unused／reserved bytes |
| 新 kernel sanitizer | [memcheck／initcheck／racecheck／synccheck](sanitizers/summary.json) 全 exit 0；各只插樁 `tiled_hr` 的首三次 launches，即同一 random HR input 的 ordinary／repeat／Graph；0 errors／hazards |
| Static checks | [Bash syntax、ShellCheck、benchmark C++ warnings 與 whitespace](static-checks.json) PASS |

完整 CTest 的七序列仍使用原 T01 640 engine，比較 600／1050／837／525／654／900／750 幀，涵蓋的是 fallback 及既有 session 行為。**本輪新的 HR pipeline 比較是 MOT17-10 前 140 幀**；四張 HR Python references 也不擴張為七序列完整 HR detector 回歸。

Sanitizer 的 filter／launch-count、命令與完整 logs 都在 `sanitizers/`。三次 instrumented launches 不等於四種 patterns、全部 kernels 或完整 session 均插樁；未執行新 source 的全 session 四 sanitizer、正控制／語義 mutation 或 clean-source 正式 gate。正常 CTest 的 held-stream PASS 與這個有 filter 的 sanitizer 範圍分開。

所有 HR A/B packet 檔均為 1,731,520 B，SHA-256 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`。第一次 pipeline wrapper 在 baseline 完成後讀取不存在的 `modes` 欄位而 exit 1；該次 [log](pipeline-attempt.log)／[部分輸出](pipeline-attempt/) 保留。修正為實際 `groups` 後，於新目錄完整重跑兩輪 ABBA；失敗嘗試不混入最終統計。

## 3. 效能與 counter

### 3.1 隔離 F02

[最終 samples](primitive/benchmark/samples.csv)／[summary](primitive/benchmark/summary.json) 使用 deterministic xorshift RGB、原 deployed shape；四輪 ABBA、每段先 40 次 warmup，再計 200 次，每 mode／backend 共 8 個 batch means。event 與 host 都量一批工作平均 processing interval，不是單次 latency 分布，也不含 JPEG／模型或 concurrent M 工作。

| µs／operation | 舊兩 pass | 新 tile | event interval 降低 |
| --- | ---: | ---: | ---: |
| ordinary event | 149.608 | **88.480** | **40.86%** |
| ordinary wall | 149.837 | 88.653 | — |
| Graph event | 149.061 | **89.763** | **39.78%** |
| Graph wall | 149.306 | 89.984 | — |

### 3.2 完整 HR pipeline

[逐 session 統計](pipeline/runs.csv)、[逐 frame latency](pipeline/)、[彙整](pipeline/summary.json) 為兩輪外層 baseline／candidate／candidate／baseline；每個 binary invocation 再跑兩個 single 與兩個 queued sessions，故共 32 sessions。兩 backend 各有 **960 個 single latency samples、8 個 queued wall interval**。JPEG 預載，排除磁碟、constructor、first-frame setup 與 warmup；first-frame／constructor 原值仍留在 runs。

| 指標 | 舊兩 pass | 新 tile |
| --- | ---: | ---: |
| single mean，ms | 7.151735 | 6.954582 |
| single p50，ms | 7.010304 | **6.905291** |
| single p95，ms | 7.442117 | 7.386566 |
| single p99，ms | 7.653497 | 7.655475 |
| queued wall interval，ms/frame | 5.366908 | **5.312260** |
| queued 八次範圍，ms/frame | 5.337393–5.390104 | 5.286308–5.332031 |

Queued interval 降低 **1.02%**，是 throughput 結果。Single mean 降低 2.76%，但 baseline 包含 zero-based frame 110 的 **102.958 ms** 長尾樣本；[最大 latency 清單](pipeline/largest-latencies.json) 與全部原值保留，沒有剔除樣本。p50 降低約 1.50%，p95 約 0.75%，p99 幾乎相同；**不將 mean 的 2.76% 當成穩定可重現的全 pipeline 改善幅度**，也不把隔離 F02 的 39.8% 推廣到整條 pipeline。

### 3.3 Cache transactions

[NCU raw baseline](counters/baseline.csv)／[candidate](counters/candidate.csv)、[metric summary](counters/summary.json) 為同 probe、相同 xorshift RGB，20 次 warmup 後取第 21 次；application replay、cache-control none、clock-control none，無 TensorRT／M 同時執行。這是另外一組 profiler 觀測，不能替代上面的未插樁 ABBA。

| counter | 舊兩 pass 合計 | 新 tile |
| --- | ---: | ---: |
| L1 global-load sectors | 1,936,560 | 999,132 |
| L1 global-store sectors | 887,040 | 368,640 |
| L2 read sectors | 1,264,817 | 426,345 |
| L2 write sectors | 887,066 | 368,972 |

L1 global stores 減少的 518,400 sectors 與消失的 16,588,800 B scratch writes 相符；部分 global loads 改由 shared 供應。這些 counters 是 cache transactions，**不是 DRAM 流量**。本輪沒有以 L2 hit rate 或 logical payload 推論整條模型期間的 cache 常駐。

探索過 `32×8`、`32×16`、`64×4`、`64×8`、`128×4`，再比較水平權重重用。各次 source、resources、samples／summary 見 [exploration](exploration/summary.json)。較大 CTA 並沒有更好；保留較小的 `32×8`、2 KiB shared 配置。探索版 identity 的 tile 文字沿用初始模板；實際幾何以各目錄凍結 source 為準，不將探索版測時當作最終來源的新測時。

## 4. 重跑與來源保管

```sh
bash scripts/run_resize_optimization.sh build/NEW-F02-REPLAY
python evidence/f02-fusion/2026-11-04/traffic_model.py > build/NEW-F02-TRAFFIC.json
```

[`scripts/run_resize_optimization.sh`](../../../scripts/run_resize_optimization.sh) 預設從固定 `f4865cd` 取舊 kernel，凍結新 kernel／header／benchmark，採原 preprocessing Release 算術旗標，各自編譯並連結同一程序。先做完整 bitwise 對照，再在 GPU lock 下執行 ABBA。`--benchmark-only` 結果的 equivalence 為 null；正確性由同 capture 的 verify 明確決定，不能從測時成功推論。

完整 CMake build 在 `build/f02-opt/cuda/`，最終 primitive replay 在 `build/f02-fusion-replay-20261104/`。完整 pipeline 的 baseline binary 由同一組 candidate libraries 重新 link，**僅替換 resize object 為固定舊版**；[baseline link argv](baseline-tracking-link.json) 與 baseline source 保留。各 run 使用 fresh sessions、相同 JPEG prefix／warmup／tracker history，只有 F02 不同。pipeline／sanitizer／counter 的 capture orchestrators 同附本目錄，其 build 路徑固定於本次 capture；一般 F02 重跑使用上面的新目錄腳本。

硬體為 RTX 5070 Ti Laptop／WSL2、46 SM；CUDA 13.4.92、driver 617.42、TensorRT 10.16.1.11。部署 HR FP16 engine 仍是 SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`，model input／output 仍 FP32。未鎖 GPU clocks；效能限本機與此 workload。Binaries／大型 packet outputs 留在忽略的 `build/`，報告保存其 hashes、原始 log 與逐次測量；不是將 binaries 一併提交的可攜封裝。

## 5. 接續與範圍

下一個較窄的 tracking 候選是 `predict_kernel` 的穩定 compaction／state copy：保留 thread 0 的固定順序 metadata／rank scan，再由 block 平行搬運 state 與 priors。必須保留第一個 NUMERIC 前的部分輸出，以及 UID／順序／首錯語義，另以同起點完整 state／batch／rows／packet 對照與 single／queued 測時判定。F04b continuation acceptance 仍有先後依賴，不能改成無序接受。

本輪沒有修改 tracking、scan mapping、FP16 邊界或 L2 policy。primitive／有限幀通過不延伸為所有影像、所有 GPU／SDK 或全路徑 sanitizer 保證；nvJPEG 原 initcheck FAIL／原因 OPEN 不變。
