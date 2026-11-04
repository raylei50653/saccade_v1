# Saccade 各階段延遲與硬體資源報告

2026-11-04。本次 HR FP16 tracking 的暖機後單幀 enqueue→receive 平均為 **6.988 ms，p95 7.405 ms**；允許多幀在途時，平均每 **5.412 ms 完成一幀，約 184.8 FPS**。這兩個數字分別描述單幀延遲與穩態吞吐。預設四格結果 ring 的 caller 交付策略下，另外實測的 queued enqueue→receive 平均為 **21.217 ms**，不是 5.412 ms。

F03 是最長的 GPU 計算階段。現有 D/M 排程已讓 GMC／predict 與模型重疊，也讓前幀 tracking tail 與後幀模型重疊；共享 SM 與頻寬仍會增加同時執行階段的時間。資源 owner、交付與回收邊已逐項審查；保留 [ownership.csv](ownership.csv) 與 [安全審查](safety-review.md) 的前提及同步邊時，已列明共享資源的衝突存取有序。完整 session racecheck 本次仍未完成，不能稱為整條路徑的 sanitizer 全通過。

## 1. 來源、硬體與受測輸入

| 項目 | 本次身分 |
| --- | --- |
| 來源 | 基底 `865bb105b2b04e5e51a09c7106952184f5b2a9e1` 加本工作目錄修改；`source_dirty=true` |
| 來源保管 | [commit](source-commit.txt)、[Git 狀態](source-status.txt)、[patch](source.patch)、[source hashes](source.sha256)；包含既有 CUDA Graph、tracking 排序及本次 probe／文件修改 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM 12.0，46 SM；裝置回報 1 個 asynchronous copy engine |
| CPU／系統記憶體 | AMD Ryzen 9 8940HX，16 cores／32 threads；WSL2 可見約 47 GiB RAM |
| SDK | CUDA 13.4.92，nvJPEG 13.2.3，TensorRT 10.16.1.11 |
| 部署模型 | HR 1280×768，FP16 tactics、FP32 input/output；engine SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` |
| 輸入 | MOT17-10-SDP，同一批 1920×1080 JPEG；每 session 從 frame 0 開始，先 20 幀暖機，再量 zero-based frame 20–139 共 120 幀 |
| 配置 | motion downscale 4，MI 480×270；候選／motion／結果 slots 預設 2／2／4；固定原 tracker 設定 |
| Graph 範圍 | D 的 F03→producer→F04，M 的候選檢查→F04b→update→commit→packet；F01/F02、F05、GMC、predict、跨 stream events 與 D2H 保留在圖外 |

本次是**開發來源量測與驗證**，不是 clean-source 正式驗收；沒有重新選模型、調整 tracker 決策或重算品質分數。硬體未鎖 clock，結果限本機、此 SDK／engine／輸入及設定。程式先取得同一 GPU lock，量測與 sanitizer 分開執行。

目前執行順序是 D：**F01／F02 → F05 → F03 → producer → F04**；M：**F06 → F08a → 等候本幀候選 → F04b／F07／F08b → commit → packet → D2H**。F05 交付獨立 MI 後，M 可先處理 GMC，無須等 F03。邏輯流程到 F09，沒有額外 F10。

## 2. 延遲與吞吐的量測邊界

### 2.1 未插樁的端到端結果

[tracking_benchmark 摘要](raw/latency/summary.json) 使用兩輪 ABBA，ordinary／graph 各四個單幀 sessions，每模式 **480 個實際單幀樣本**；另各四個 queued sessions 量吞吐。每個 fresh session 使用相同 warmup 與 tracker 歷史。單幀 enqueue→blocking receive 起點沒有前幀在途，包含 JPEG host 提交、前處理、模型、tracking、完整 packet D2H 與 receive。JPEG 已預載，排除磁碟輸入、MOT 輸出、constructor 及 warmup。p50/p95/p99 使用逐幀 nearest-rank，沒有剔除 outliers。

| 指標 | ordinary | Graph，現行預設 |
| --- | ---: | ---: |
| 單幀平均 | 8.615 ms | **6.988 ms** |
| 單幀 p50 | 8.653 ms | 6.958 ms |
| 單幀 p95 | 9.364 ms | **7.405 ms** |
| 單幀 p99 | 9.780 ms | 7.682 ms |
| queued wall time/frame | 6.942 ms | **5.412 ms** |
| queued 四次範圍 | 6.887–7.045 ms/frame | 5.333–5.583 ms/frame |
| 首幀 enqueue→receive，含 lazy capture | 10.117 ms | 17.271 ms |

Graph 單幀平均降低 18.89%，queued 吞吐為 ordinary 的 1.283 倍。這裡 ordinary 表示不 replay CUDA Graph，與 `TrackingSessionMode::Serial` 是不同維度；兩模式都可用相同 D/M pipeline。constructor 跨 runs 包含程序初次載入與 cache 效應，原值留在 runs.csv，不把 constructor 平均差異當成 Graph 加速。

16 個 sessions 的 140 幀完整固定容量 packet bytes（含 warmup）全部相同，SHA-256 為 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`。

### 2.2 queued 每幀實際交付延遲

另一個未插樁 [resource probe](raw/probe/frames.csv) 記錄每個 frame 的 enqueue 起點到它實際被 receive 的時間。預設四格 ring 的 queued 平均 **21.217 ms，p95 21.768 ms**；同次 probe 的 queued wall time/frame 為 5.379 ms。前者包含在途排隊、共享資源等待及 caller 延後 receive 的時間。結果已完成但 caller 尚未取出的期間也算交付延遲。5.412 ms/frame 是主 ABBA 的吞吐換算，不能替代這個每幀量測。

額外 [ring 深度觀測](raw/queue-depth-summary.json) 各用一個 fresh session、120 個測量幀；完整 packet hash 相同。這是單次配置比較，沒有 ABBA，不據小差異宣稱穩定收益。

| result_slots | enqueue→receive 平均 | p95 | wall time/frame |
| ---: | ---: | ---: | ---: |
| 1 | 6.993 ms | 7.379 ms | 7.002 ms |
| 2 | 10.740 ms | 11.259 ms | 5.396 ms |
| 4 | 21.289 ms | 22.452 ms | 5.401 ms |

需要即時回傳時可選 depth 1；本次 depth 2 已保留近似 depth 4 的吞吐，且 caller 交付延遲較短。這只是結果 ring／消費策略選擇，尚未修改專案預設值，不能直接當成不同輸入或所有負載的最佳深度。

## 3. F01 與 F02 的獨立量測

[JPEG probe](raw/jpeg/summary.json) 不載入模型。它逐幀先確認本程序 GPU idle，再用與 NativePreprocessor 相同的 HARDWARE backend、batch 1、planar RGB、1920×1080 預配置、16 MiB pinned staging。預載 JPEG 後，量 staging memcpy、header／metadata／support 檢查、decode API，直到 detect stream synchronize 確認 RGB 交付；F02 在 F01 完成後獨立執行。

| 獨立階段，120 樣本 | 平均 µs | p95 µs |
| --- | ---: | ---: |
| F01 完整 host 起點→RGB ready | **1454.37** | 1715.97 |
| 其中 staging memcpy＋檢查／parse | 15.46 | 18.27 |
| decode API host call | 540.80 | 698.76 |
| decode API 起點→RGB ready | 1438.90 | 1702.52 |
| F02 CUDA-event envelope | 155.69 | 188.45 |
| F02 host enqueue | 36.10 | 61.38 |
| F02 host enqueue→完成 | 190.34 | 224.73 |

decode host call 是 decode→RGB ready 的子區間，不能相加。F02 event envelope 包含兩個 resize kernels 及中間空檔／提交間隙，不是純 kernel busy。F01 1454 µs 是這種隔離條件下的可觀測交付時間；CUDA trace 看不到 JPEG 硬體 decoder 的獨立 busy time。nvJPEG RGB conversion 還使用 GPU SM，且在私有 stream；這個 JPEG backend 不稱為 NVDEC。**F01／F02 的隔離數字不能再加到重疊 pipeline 延遲上。**

## 4. 各階段 GPU 時間與硬體

以下來自同次 Nsight capture 的 Graph `single` 與 `queued`，各排除 20 幀暖機，保留 120 幀。**busy 是同一階段、同一幀活動區間的聯集；envelope 是第一個活動開始至最後一個活動結束，包含中間空檔。** 這些是 profiler 下的 GPU 活動時間，與上面的未插樁 wall latency 分開。GPU SM/Tensor Core 工作共用 SM scheduler、cache 與記憶體頻寬；兩條 streams 並不代表兩套獨立算力。

| 階段 | single busy／envelope µs | queued busy／envelope µs | 硬體與位置 |
| --- | ---: | ---: | --- |
| F01 壓縮 JPEG H2D | 10.61／10.61 | 12.82／12.82 | copy engine |
| F01 RGB 轉換 | 32.73／32.73 | 29.46／29.46 | GPU SM，nvJPEG 私有 stream |
| F02 正規化與 HR resize | 142.96／145.71 | 149.52／153.11 | GPU SM，D |
| F05 灰階縮小與 Hanning | 6.61／8.70 | 6.50／8.64 | GPU SM，D |
| F03 完整 TensorRT 模型 | 4873.82／4989.81 | 5003.80／5132.68 | Tensor Cores＋SM，D graph |
| F03 producer finite/status | 27.50／27.89 | 28.18／28.57 | GPU SM，D graph |
| F04 decode／篩選／NMS | 95.76／96.47 | 99.65／100.39 | GPU SM，D graph |
| F06 GMC，含 cuFFT | 120.50／448.77 | 99.88／226.44 | GPU SM／cuFFT，M |
| F08a GMC control／predict | 67.70／114.02 | 61.52／107.42 | GPU SM，M |
| F04b continuation batch | 111.78／111.78 | 682.08／682.08 | GPU SM，M graph |
| F07／F08b 關聯與更新 | 96.29／96.46 | 219.32／251.67 | GPU SM，M graph |
| 成功／失敗 commit | 0.72／0.72 | 0.88／0.88 | GPU SM，M graph |
| ResultPacket 編碼 | 2.41／2.41 | 2.92／2.92 | GPU SM，M graph |
| F09 packet D2H | 2.00／2.00 | 2.12／2.12 | copy engine，M |

F03 的 FP16 convolution/GEMM tactics 使用 Tensor Cores，activation／transpose 等與 native FP32 selective scan 使用 SM。cuFFT 同樣是 GPU 計算，沒有專用的獨立 FFT 硬體。H2D 平均每幀約 137,007 bytes，D2H 固定每幀 **12,368 bytes**；本機只回報一個 copy engine，額外 H2D/D2H 流量須一起考慮。

M begin 的 busy 平均 single／queued 為 0.86／0.93 µs，其他 stage controls 為 4.63／5.83 µs，candidate checks 為 2.01／3.72 µs。這些控制 kernels 分散在整幀，`M_control` 的首尾 envelope 約 4.85／5.16 ms，跨過 GMC、候選等待及 tracking；**不能把這個 envelope 當成控制成本**。candidate checks 的 envelope 也包住 F04b，不能與 F04b 再相加。每幀兩次 predict／update launch 透過 slot guard 擇一執行；表中保留實際 launches 的活動時間，沒有將它們誤報為兩次邏輯 Kalman predict。

### 4.1 各階段 envelope p95

| 階段 | single envelope p95 µs | queued envelope p95 µs |
| --- | ---: | ---: |
| F01 壓縮 JPEG H2D | 11.65 | 16.96 |
| F01 RGB 轉換 | 37.51 | 30.98 |
| F02 正規化與 HR resize | 149.41 | 154.95 |
| F05 灰階縮小與 Hanning | 8.83 | 8.86 |
| F03 完整 TensorRT 模型 | 5043.09 | 5188.15 |
| F03 producer finite/status | 28.35 | 28.96 |
| F04 decode／篩選／NMS | 111.88 | 116.48 |
| F06 GMC，含 cuFFT | 472.68 | 298.98 |
| F08a GMC control／predict | 174.37 | 199.68 |
| F04b continuation batch | 138.24 | 743.66 |
| F07／F08b 關聯與更新 | 123.56 | 287.91 |
| 成功／失敗 commit | 0.77 | 0.96 |
| ResultPacket 編碼 | 2.53 | 3.04 |
| F09 packet D2H | 2.21 | 2.21 |

完整 mean／p50／p95／p99／min／max 及逐幀資料見 [stage_summary.csv](stages/stage_summary.csv)、[stage_samples.csv](stages/stage_samples.csv)、[frame_metrics.csv](stages/frame_metrics.csv)；kernel registers、shared／local memory 與 launches 見 [kernel_resources.csv](stages/kernel_resources.csv)。階段彼此可重疊，envelope 也可跨空檔，這些欄位不能直接加成整幀延遲。

## 5. 已觀測的重疊與可用排程

[事件排程表](SCHEDULE.md)、[schedule.csv](schedule.csv) 與 [queue-depth.csv](queue-depth.csv) 可直接查閱或匯入。[時間圖 SVG](schedule.svg)／[PNG](schedule.png) 直接使用實測區間；[timeline.json](stages/timeline.json) 保留可重建資料。逐資源的合法重疊及禁止條件見 [安全審查](safety-review.md)。

| 重疊 | 必要條件與實際行為 |
| --- | --- |
| 同幀 D 的 F03/F04 與 M 的 F06/F08a | F05 先寫獨立 MI，record ready；M wait ready 後開始。F04b 仍等本幀 candidates ready。 |
| D(k+1) 與 M(k) | CS／MI 必須等各自上一代 consumed；結果 ring 保留背壓；F08a(k+1) 仍不能越過 commit(k)。 |
| 下一幀 JPEG 硬體工作與前幀模型 | host 已等待前幀 F05 的 `rgb_read_done`，保護共用 RGB；F02(k+1) 留在 D，排於 F03(k)/F04(k) 之後。 |
| 下一代 F05 與前幀 cross-power／IFFT | MI consumed 已發布；後續 F06 只讀獨立 SPEC／scratch，不再讀該 MI slot。 |
| D2H 與後幀 D 計算 | D2H 與 packet 寫入維持 M 順序，host slot 尚未 receive 不得重用；copy engine 是否實際同時排程以 trace 為準。 |
| CPU writer 與 GPU | 可在別的 thread 處理 receive 回傳的 owned packet value；session API 仍由原 host thread 使用，不傳遞 pinned-slot 借用指標。 |

在 queued 的 120 個測量幀中，D/M kernel 活動的實際交集為 **125.555 ms，平均 1.046 ms/幀**，占 M kernel busy 的 **97.14%**。其中 M(k) 與 D(k+1) 119 個相鄰幀配對的平均交集為 **895.18 µs**。single 仍有同幀 GMC／predict 重疊，D/M 平均交集 184.76 µs/幀；它沒有跨幀交集。這些數字是活動區間的時間交集，**不是 SM utilization、occupancy 或省下的等量 wall time**。

F04b single／queued busy 為 **111.78／682.08 µs**，update 為 96.29／219.32 µs；F03 則為 4873.82／5003.80 µs。這些數字顯示同時排程的條件伴隨時間變長，與資源競爭相符；本次沒有以硬體 counter 或受控頻寬／occupancy 實驗，將差異全部歸因於某一種 contention。不能把 single kernel 時間當成 queued 服務時間，也不能靠增加 streams 推算線性加速。

每幀必要同幀次序的 14 類檢查，在三種模式各 120 幀共 360 次／類全部符合，沒有未分類的額外 sampled memcpy。檢查涵蓋 RGB→F02、MI→GMC、F03→producer→F04、候選→tracking、predict→tracking、update→commit→packet→D2H；它只確認可觀測必要邊，不代替生命週期／任意記憶體安全證明。Graph kernel 可能顯示合成 profiler streamId，本報告依 graph launch correlation 與內容歸入邏輯 D/M，沒有把合成 streamId 當成錯誤的 aux stream。

## 6. 記憶體與資源預算

### 6.1 可由來源與實測 ABI 算出的配置

[resources.csv](resources.csv) 是逐配置完整表；[resources.json](raw/jpeg/resources.json) 已實測 `sizeof`，確認 DeviceTrack=408、DeviceTrackState=208,928、DevicePredicted=225,384、DeviceBatch=10,824、DeviceTrackingRows=12,296、TrackingCudaScratch=234,000 bytes。候選每 slot 9,664 bytes，packet 每 slot 12,368 bytes。F04 workspace 已由本 build 查詢為 **698,432 bytes**，不能沿用舊 profile／版本的 workspace 常數。

| 顯式持久配置群組 | bytes | 所有權與用量 |
| --- | ---: | --- |
| RGB＋HR FP32 input＋horizontal resize workspace | 34,606,080 | D 專用單份；分別 6,220,800／11,796,480／16,588,800 |
| Raw predictions＋statuses＋producer＋F04 scratch | 7,474,520 | D 專用單份；raw 為 6,773,760 |
| Candidate slots | 19,328 | 2 × 9,664，ready/consumed 跨 D/M |
| MotionImage／headers、Hanning、FFT_TMP、SPEC[2]、cross、correlation、reduction／numeric | 3,656,700 | MI 2 格；其餘 M 專用或唯讀；未含 cuFFT workspace |
| TS[2]、predicted、batch、rows、tracking scratch、config/control/context、device packet | 913,024 | M 專用，working/committed 版本分離 |
| **顯式持久 device 小計** | **46,669,652** | **44.508 MiB** |
| Pinned JPEG staging | 16,777,216 | 單份 16 MiB |
| Pinned packet ring | 49,472 | 4 × 12,368 |
| **顯式 pinned host 小計** | **16,826,688** | **16.047 MiB** |

constructor 暖機 input 暫時另用 11,796,480 bytes，於接受 frame 0 前釋放，不加入持久小計。cuFFT workspace 為 forward／inverse 所需大小最大值，兩 plans 共用但只在 M 串行執行。TensorRT engine／context／plugin 私有配置、nvJPEG 預配置、cuFFT workspace、Graph executable、driver／模組／runtime cache 不在 44.508 MiB 的顯式表內。一般 host JPEG preload、packet 比對 bins 與 F09 向量也不是 pinned 小計。

### 6.2 不同口徑的實測峰值

| 觀測口徑 | 實測 | 如何解讀 |
| --- | ---: | --- |
| Nsight 攔截到的 device allocations 同時存活峰值 | **281,809,797 bytes，268.755 MiB** | 含攔截到的函式庫配置；不保證包含未追蹤的 driver 私有儲存 |
| Nsight pinned host allocations 峰值 | **16,826,688 bytes，16.047 MiB** | 與顯式 staging＋4 格 packet ring 相同 |
| Nsight static memory kind 峰值 | 15,352 bytes | 另一 memory kind，沒有混入 device allocation 小計 |
| nvidia-smi 採樣最高 memory.used | **480 MiB** | 100 ms 採樣的裝置 accounting；可能漏掉短暫峰值 |
| cudaMemGetInfo 的 global used，constructor 後／warm／測量後 | **1,487.4375 MiB** | total−free 的全裝置口徑，含程序以外／driver／WSL accounting；不能歸為本 session VRAM |
| resource probe 程序 peak RSS | **716,724 KiB，約 699.93 MiB** | 程序級歷史峰值，含 CPU allocations／preload／比對／函式庫；不是 pinned RAM |

device allocation 與 pinned allocation 的 tracked live bytes 在程式結束回到 0；static kind 的 15,352 bytes 仍存在於模組生命週期。上述值來自 [allocation-peaks.json](raw/allocation-peaks.json)、[probe runs.csv](raw/probe/runs.csv) 與 [GPU samples](raw/probe-gpu.csv)。**44.5、268.8、480、1487.4 MiB 是不同範圍，不能相加或互換。**

## 7. CPU 輸入與 F09 輸出

CPU／檔案量測獨立於 GPU timing，沒有將檔案時間加到已重疊的 pipeline 上。frame 20–139 的 cached file reads 平均 **38.48 µs，p95 77.51 µs，p99 93.95 µs**；這是檔案已在 cache 的本機條件，不是冷磁碟吞吐保證。資料見 [input-io.csv](raw/probe/input-io.csv)。

對 owned packets 在 GPU timing 外重播，CPU packet decode＋SequenceOutput::add 平均 **0.732 µs/幀，p95 1.092 µs，p99 3.977 µs**。140 幀 sequence tail 的 interpolation／sort／finish 共 **129.666 µs**，format＋write 共 **1845.48 µs**；後者沒有 fsync，不能稱為持久化完成時間。這些是序列尾一次性工作，不是每幀都要支付的 1.845 ms。[output-frames.csv](raw/probe/output-frames.csv)／[output-tail.json](raw/probe/output-tail.json) 保留原值。

CPU 目前負責序列／JPEG metadata、pinned staging memcpy、library/API 提交、事件等待與 packet 消費，GPU 負責前處理、推論、motion 與 tracking。將 F09 writer 拆到另一 thread 需交付 owned packet，不能讓第二個 thread 同時呼叫 session API 或存取仍被 GPU 使用的 pinned slot。

## 8. 無競態的條件與本次驗證

安全性依據是 [逐資源 owner 表](ownership.csv) 與 [happens-before 審查](safety-review.md)：同一 host thread 提交連續 frame、固定 device/profile/地址、每 slot 專屬 ready/consumed、RGB 最後讀者 host wait、D/M 各自專用 scratch、M 上單序列 commit、packet ring 背壓、close 先 drain 再釋放。這些條件保留時，所列共享記憶體每對衝突存取都有順序；不是由某一次 sanitizer 無錯誤推導普遍保證。

| 本次檢查 | 結果與範圍 |
| --- | --- |
| CPU 協定模型 | 216/216 baseline configurations PASS，15/15 mutation families 均被偵測；有限合成協定模型 |
| T01 session CPU/reference／ordinary serial／graph pipelined | 120 幀完整 packet bytes 相同，1772 checks PASS；含 ring 1/1/8、3/3/5 及兩個 ordinary perturb seeds 的 directed controls |
| ordinary／Graph lifecycle | 180 checks PASS；failure、committed-state 凍結、背壓、cancel、D-only tail、destructor、repeat checked close |
| HR single／queued／serial 與主 ABBA | 140 幀完整 packet hash 相同；含 warmup，沒有放寬 byte comparison |
| HR F04 正常測試 | 44,432 checks PASS；含 graph request update、arguments、streams／held controls |
| Tracking CUDA 與 Motion CUDA | 7,599,964／537 checks PASS |
| HR Graph 完整 session memcheck | 30 測量幀＋2 暖機幀，含函式庫 kernels，退出 0，ERROR SUMMARY 0 |
| HR F04 racecheck | `--no-held-stream`，44,127 checks，退出 0，0 hazards；held callback gates 在此 run 未執行 |
| Tracking component racecheck | 7,599,964 checks PASS，0 hazards；隔離 service 的 memory peak 約 418 MiB |
| Motion component racecheck | `--synthetic`，523 checks PASS，0 hazards；memory peak 約 464 MiB，不含 nvJPEG |
| HR F04 synccheck | `--no-held-stream`，44,127 checks，退出 0，0 errors |
| T01 session synccheck | limit 10，僅 tracking／motion／session-control kernels；222 checks、退出 0、0 errors；不含函式庫內部插樁驗收 |
| racecheck／synccheck 正控制 | 各退出 99，分別命中預期 race hazard／Barrier error，工具偵測有作用 |
| 完整 HR 自訂路徑 racecheck 初次嘗試 | **未完成**。包含 scan 等自訂 kernels 的插樁耗盡約 47 GiB WSL 記憶體，target 被 global OOM 殺掉，runner 退出 9；即使末尾列 0 hazards 也不是 PASS |
| 限資源 T01 session racecheck 重跑 | **未完成／timeout**。僅 tracking／motion／session-control，MemoryHigh 6 GiB、MemoryMax 8 GiB、SwapMax 0，150 秒退出 124，peak 約 6.1 GiB；無成功完成 marker |

[validation-results.json](raw/validation-results.json)、[bounded-validation-results.json](raw/bounded-validation-results.json) 與 [component-race-results.json](raw/component-race-results.json) 分開保管成功與失敗嘗試；初次失敗不被後續有界測試覆蓋。完整 TensorRT／nvJPEG／cuFFT 內部 racecheck 與完整 session initcheck 本次沒有完成驗收；既知 nvJPEG initcheck 診斷維持既有使用政策。shared-memory racecheck 也無法證明所有跨 stream global-memory lease。

`perturb_seed != 0` 刻意走 ordinary 以保留原始擾動點，不能稱為完整 Graph 擾動覆蓋；Graph 的 slot 代次另外以不等 ring periods、完整 packet identity/bytes、request metadata 與 lifecycle 測試支持。新增 stream、分裂 graphs、移動 consumed、改 shape／buffers 或讓多個 callers 並行時，必須更新 owner 表並重新檢查受影響的順序與 byte-equivalence。

## 9. 重跑與交付檔

完整量測、Nsight 分析及有記憶體上限的安全檢查方式見 [重跑說明](REPRODUCE.md)。[hardware.json](hardware.json)、[binary hashes](binaries.sha256) 與 [replay hashes](replay.sha256) 分別記錄硬體、執行檔及原始 trace 身分。

完整本地工作目錄為 `build/resource-schedule-20261104/`，包含 measured binaries／hashes、JPEG manifest、packet bins、Nsight `.nsys-rep`／SQLite、原始 logs、各命令與失敗嘗試。報告中的資料連結指向本目錄的可攜 raw 副本。來源與測量 binaries 必須一起保管，不能只以基底 commit 代表 dirty source。可攜摘要、samples、commands 與 logs 在 [raw](raw/)；大型原始 trace、packets 與 binaries 保留於上述 build 目錄。

端到端兩輪 ABBA：

```sh
flock ~/.cache/saccade-v1/gpu.lock build/resource-schedule-20261104/tracking_benchmark   ~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine   /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP   build/NEW-LATENCY --profile hr-1280x768 --frames 120 --warmup 20 --rounds 2
```

獨立 F01／F02：

```sh
flock ~/.cache/saccade-v1/gpu.lock build/resource-schedule-20261104/jpeg_stage_probe   /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP   build/NEW-JPEG-STAGES --frames 120 --warmup 20
```

新的輸出目錄必須不存在，防止覆寫原證據。Nsight stage 分析保留 busy/envelope、graph logical-stream 對應與必要邊檢查；[stage summary JSON](stages/summary.json) 的 PASS 指分類／數量／可觀測順序檢查通過，沒有延伸為 sanitizer 或任意輸入證明。資源完整表為 [resources.csv](resources.csv)，合法排程與條件保證分別由 [ownership.csv](ownership.csv)、[安全審查](safety-review.md) 和 [實測排程圖](schedule.svg) 交叉對照。
