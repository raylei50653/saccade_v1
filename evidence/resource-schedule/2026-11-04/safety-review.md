# TrackingSession 重疊排程與競態審查

2026-11-04，開發來源審查。對象為本次量測使用的工作目錄來源，身分以同目錄的 source hashes、patch、binary hashes 與 runtime capture 為準。機器可讀的逐資源清單在 [ownership.csv](ownership.csv)。

## 條件保證

現行 `TrackingSession` 對每一類跨執行者共享資源，都保留 producer 寫完到 consumer 開始讀的交付邊，以及 consumer 最後讀完到下次覆寫的回收邊。D 專用與 M 專用 scratch 則由各自 stream 順序保護。保留這些邊並符合下列前提時，排程本身不會使不同幀、不同 stream 或 host/device 對這些資源發生未排序的衝突存取：

- 整個 session 由同一 host thread 使用，frame 以 0、1、2…連續提交；`enqueue/receive/finish/cancel` 在建立 session 的 CUDA device 呼叫。`pending/state/identity/close` 未全面執行相同 owner 檢查，亦須遵守單一 thread 契約。
- 所有 graph、context、plans、workspace 與 buffers 由該 session 持有，不同 session 不共用隱藏工作區；每個 candidate/motion slot 保留自己的 ready/consumed events。
- 使用已驗證的固定形狀、容量及輸入契約，engine 的 auxiliary stream 數為零；D/M 的專用資源不被外部 kernel 存取。
- CUDA API、裝置與函式庫正常運作；fatal device error 不屬於正常完成及無死鎖保證。

這個保證指向**已列出的資料存取與同步協定**。NVIDIA 函式庫內部、任意新 kernel、違反輸入或 thread 契約的呼叫，以及不同 GPU/SDK 的行為，不由這份 source audit 證明。racecheck 的 shared-memory 診斷也不能證明跨 stream global memory 無競態。

## 每類衝突存取為何有序

### JPEG staging 與 RGB

`NativePreprocessor` 只有一份 pinned JPEG staging、nvJPEG state 與 RGB output。它在下一次複製壓縮資料前，先等前幀 F02 的完成 event，避免 staging/state 尚被使用便重用；來源是 [`src/preprocess.cpp`](../../../src/preprocess.cpp) 的 `wait_pending()` 與 `enqueue()`。

RGB 還有另一個回收限制：nvJPEG 硬體路徑的私有 stream 可能在 D 上先前工作前寫入 RGB，D 的 enqueue 順序不足以保護 F05。現行 session 在 F05 後 record `rgb_read_done`，下一次呼叫 decode 前由 **host synchronize 該 event**。因此 F05(k) 的最後 RGB 讀取先於 decode(k+1) 的首次 RGB 寫入。F02(k) 與 F05(k) 在同一 D 上依序完成，沒有額外 RGB reader。實作見 [`tracking_session.cpp`](../../../src/tracking_session.cpp) 的 frame enqueue。

本次已把這項必要等待同步寫入 [`preprocess.hpp`](../../../include/saccade/preprocess.hpp)、[PREPROCESS](../../../docs/PREPROCESS.md)、[ASYNC_CONTROL](../../../docs/ASYNC_CONTROL.md) 與 [DATAFLOW](../../../docs/DATAFLOW.md)。獨立使用 `NativePreprocessor` 的 caller 同樣必須等待其最後 RGB reader；模型 `images` 的讀取仍可由同一 D 上的 F03→下一次 F02 順序保護。

### D 專用與 M 專用資源

模型輸入、TensorRT execution context、raw predictions、plugin statuses、producer aggregate/scratch 與 F04 workspace 只在 D 上使用。F02(k)→F03(k)→producer/F04(k)→F02(k+1) 的 stream 順序，排序每一對讀寫或寫寫衝突。engine 在建立 session 時拒絕 auxiliary streams 非零的設定。

GMC 的 FFT_TMP、cross-power、correlation、reduction、numeric scratch、兩個 cuFFT plans 與共用 FFT workspace 只在 M 上使用。每次 forward/inverse 執行前把 plan 綁到 M；同一 workspace 的前後使用有 M stream 順序。predicted、batch、tracking scratch、rows、CTL 及單份 device packet 亦由 M 持有。D 與 M 同時執行時沒有讀寫彼此的專用 scratch。

### MI 與 CS 的交付與回收

對 MI slot `m = k mod N_MI`，D 先等上代 `mi_consumed[m]`，再寫 image/header，最後 record `mi_ready[m]`；M 在 forward FFT 與 header 檢查前 wait ready。forward FFT、guarded spectrum delivery 與最後 header 檢查後，M record consumed。所以下代 F05 的寫入不會和本代 MI payload/header 讀取重疊；後續 cross-power/IFFT 僅用 M 專用 scratch 與 SPEC，可與同 slot 的下代 F05 並行。來源為 [`cuda/motion.cu`](../../../cuda/motion.cu) 的 `enqueue_image/enqueue_estimate`。

對 CS slot `c = k mod N_CS`，D 在 F04 寫入前 wait 上代 consumed，F04 完成後 record ready；M 在任何候選 payload/header 存取前 wait ready。ordinary 路徑在 F04b 與第二次 candidate 檢查後 record consumed；graph 路徑保守地延至整個 M tail graph 後 record。F04b 產生獨立的 M-owned batch，所以 ordinary 提早釋放 CS 後，association/update 仍不讀 CS。

ready/consumed 都按 slot 配置。單一 host thread 依 D(k)→M(k)→D(k+1) 的順序 enqueue，上一代 consumed 的 record 已 enqueue 才會提交下一代 wait；每次 wait 因而綁定對應 slot 的正確代次。首次使用不等待未 record 的 consumed，初始化先全同步。這個依賴方向由本代 producer 指向本代 consumer，回收依賴只指向較早幀；配合同一 stream 順序，沒有互相等待未提交未來工作的環。

### 狀態提交與 host packet

M 的 begin 從 device CTL 決定 working slot。F08a 只讀 committed TS，association/update 與 guarded spectrum delivery 只寫 working TS/SPEC；成功 commit 才一起推進 TS/SPEC 的 frame/slot。F08a(k+1) 在 M 上排於 commit(k) 之後，不會讀到未提交版本。kernel 開始時的 guard 由前一個 M control kernel 固定，計算期間不由 host 或 D 改寫。

單份 device packet 的 packet(k)→D2H(k)→packet(k+1) 都在 M，D2H 尚未讀完時不會被下一幀覆寫。pinned host packet ring 在滿時先拒絕提交；`receive()` 等該 slot 的 done event，完整複製為回傳 value 後才增加 received。GPU 對下代同 slot 的寫入必須在舊結果已讀出後才能 enqueue，host 不會讀到正在 D2H 的 payload。

tracker kernels 以單一 block 執行，每個 phase 的 proposals、winner resolution、狀態更新及 bridge claim commit 間有 block barrier。各 track/detection 的並行寫入有獨立 index；競爭錯誤旗標使用 atomic OR；穩定壓實、birth UID 與 bridge commit 使用固定 traversal。這部分仍須以本次 kernel/session 比對及 sanitizer capture 支持，不能僅由外層 event 推導 kernel 內正確性。

## 已允許與禁止的重疊

| 排程 | 判定與必要依賴 |
| --- | --- |
| 同幀 D 的 F03/F04 與 M 的 GMC/F08a | 已實作。F05 交付 MI 後 M 即可前進，到 F04b 前才 wait CS。 |
| D(k+1) 與 M(k) | 已實作。MI/CS consumed、RGB 最後讀者 host wait 及 packet ring 背壓全部保留。 |
| 下代 F05 與舊幀 cross-power/IFFT | MI consumed 後可並行，因後段不再讀 MI。 |
| nvJPEG 硬體 JPEG 工作與較早幀的 F03 | 在舊幀 F05 完成後才可開始。此工作不應標為 NVDEC。 |
| 獨立 writer thread 處理 `receive()` 回傳 packet | 可使用該 thread 自己的 packet value/copy；不得保留 session pinned slot 指標。 |
| 多幀共用單份 staging/state/RGB 同時 decode | 禁止。需按 decode slot 複製資源並增加最後讀者 lease 才能改排程。 |
| 兩條 D streams 共用 TensorRT context/raw/status/F04 workspace | 禁止。現在的安全性依賴單一 D；複製 context 也須同時處理所有隱藏及顯式 workspace。 |
| 兩條 streams 共用現有 cuFFT workspace 執行 FFT | 禁止。forward/inverse 共用 workspace 的安全性依賴 M 串行。 |
| F08a(k+1) 越過 commit(k) | 禁止。同序列狀態有真實跨幀依賴，增加 buffers 不能消除此依賴。 |
| graph consumed 提到仍讀 CS 的 M graph 前 | 禁止。若拆圖或使用 event node，須重新驗證最後讀取與 event 代次。 |

允許重疊只表示資料依賴安全，不保證加速。D/M kernels 仍競爭 SM、cache 與記憶體頻寬；排程表中的同時段不能直接加總成吞吐收益。

## Failure 與收尾

stage control 與 commit 都在 M。第一個失敗由 commit 鎖定，後續自訂計算 guard 為 false，但 begin/control/commit/packet 與 events 照常執行。cuFFT 不受自訂 guard 取消，仍可寫 FFT_TMP/correlation 等隔離 scratch，不能直接覆寫已提交 SPEC。失敗不會提前歸還尚被函式庫讀取的 MI，亦不會省略 CS 回收 event。

`finish()` 與 `cancel()` 等 D/M 兩條 stream 的終端工作完成，保留 pending results 供依序 receive。`close()` 與 destructor 先各自 drain D/M，再銷毀 graph、context、buffers 及 events，包含只有 D 工作、沒有對應 packet 的失敗尾段。checked `close()` 回報清理錯誤，destructor 為 best effort；不能把 API 失敗視為正常 drain 成功。

## CUDA Graph

每個 candidate slot 各有 D graph 與 M tail graph。D graph 包 F03→producer→F04，M graph 包 candidate checks→F04b→association/update→commit→packet；ready/consumed、nvJPEG/F02、F05、GMC、predict、D2H 仍在 graph 外。

F04 request 的 epoch/frame/mapping/capacity 是 by-value 參數，每次 launch 前更新 initialize 與 finalize 兩個 nodes；M graph 從 CTL 取得本幀與 working slot。CUDA 13.4 安裝文件 `cuda_runtime_api.h` 對 `cudaGraphExecKernelNodeSetParams` 明定：修改只影響未來 launch，不影響已 enqueue 或正在執行的 launch。單一 host thread 的逐幀更新不改寫前代 kernel 的參數。allocation 地址與 profile 固定且 capture 後不變；close drain 後才銷毀借用這些地址的 graphs。

`perturb_seed != 0` 刻意走 ordinary 以保留原本的擾動邊界；producer flag 注入時當幀 D 亦走 ordinary。這些結果不能標成完整 graph 擾動覆蓋。graph 的代次與 slot 控制另由不等 ring periods、packet identity/bytes、failure/lifecycle 測例驗證。

## 本次檢查紀錄

CPU 協定模型已於本次獨立編譯並執行：

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror tools/protocol_model.cpp -o /tmp/saccade-protocol-model-audit
/tmp/saccade-protocol-model-audit --self-test
```

結果為 **216/216 baseline configurations PASS，15/15 mutation families 均被偵測**。每組 10 幀，檢查資源衝突、預期 producer、event 綁定及 DAG。這是合成協定的有限模型，不包含實際 nvJPEG 私有 stream、RGB、graph nodes 或完整 tracker，必須配合上面的實作審查。

本次驗證結果如下。成功檢查與因資源限制未完成的 racecheck 分列；**完整 session racecheck 未通過完成門檻**，不能由其他項目的 PASS 或失敗 log 中的零 hazards 摘要覆蓋。原始退出碼見 [validation-results.json](raw/validation-results.json)、[bounded-validation-results.json](raw/bounded-validation-results.json) 及 [component-race-results.json](raw/component-race-results.json)。

| 本次項目 | 狀態與範圍 |
| --- | --- |
| T01 CPU/reference、ordinary serial、graph pipelined 120 幀 packet bytes | **PASS，1,772 checks**。完整 bytes 相同；含 30 幀 ring 1/1/8、3/3/5 與兩個 ordinary perturb seeds。 |
| ordinary/graph lifecycle | **PASS，180 checks**。含首次失敗、committed state 凍結、背壓、cancel、D-only tail、destructor、checked repeat close。 |
| HR graph single/queued/serial packets | **PASS**。三個 fresh sessions 各 20 幀 warmup＋120 幀量測，完整 140 幀 packet bytes 相同。 |
| ordinary/graph ABBA packets | **PASS**。16 個 sessions 各 140 幀，包含 single/queued 與 warmup 的全部固定容量 packet bytes 相同。 |
| HR 全流程 memcheck | **PASS，0 errors**。30 幀量測＋2 幀 warmup，包含函式庫 kernels。此 sanitizer 的 packets 只與其自身已選模式比較，不把不同幀數的 hash 與 140 幀結果混比。 |
| tracking kernel racecheck | **PASS，7,599,964 checks，0 hazards**。`tracking_cuda_tests`，host memory peak 約 418 MiB；屬 component 測例，不能代替完整 session racecheck。 |
| motion kernel racecheck | **PASS，523 checks，0 hazards**。`motion_cuda_tests --synthetic`，host memory peak 約 464 MiB；不含真實 nvJPEG 影像解碼。 |
| HR F04 racecheck | **PASS，44,127 checks，0 hazards**。F04 bounded fixture suite；不涵蓋 TensorRT/nvJPEG/cuFFT 內部。 |
| T01 session synccheck | **PASS，222 checks，0 errors**。MOT17-10 前 10 幀；只過濾本專案 motion/tracker/session-control kernels。 |
| HR F04 synccheck | **PASS，44,127 checks，0 errors**。F04 bounded fixture suite。 |
| racecheck 正控制 | **PASS**。故意錯誤以 exit 99 及 `Race reported between` 確認工具有作用；摘要為 1 hazard displayed（內文列出 256 個相關存取 hazards）。 |
| synccheck 正控制 | **PASS**。故意錯誤以 exit 99 及 `Barrier error detected` 確認工具有作用；摘要為 2 errors。 |
| 第一次 HR session 自訂 kernels racecheck | **FAIL／未完成，exit 9**。host 使用約 47 GiB 後觸發全域 OOM。即使尾端曾印出 0 hazards 摘要，也沒有成功完成受測工作，不能標為 PASS 或據此判定無競態。 |
| 有界 T01 session racecheck | **FAIL／未完成，exit 124**。150 秒 timeout；設 `MemoryHigh=6 GiB`、`MemoryMax=8 GiB`、`MemorySwapMax=0`，實際峰值約 6.1 GiB。沒有取得完整檢查結論，不代表已發現競態，也不算 PASS。 |
| 本次 stream placement 與必要執行順序 | **PASS（觀察範圍）**。single/queued/serial 共 360 個暖機後樣本、123,480 個已分類 kernels，未知 kernel 為 0；14 類必要同幀順序各檢查 360 次，共 5,040 checks，無違反。每幀恰一個 JPEG compressed H2D 與一個 12,368-byte packet D2H，沒有其他量測範圍內的 copies。 |

上述 140 幀共同 packet SHA-256 為 `35ec34474f692e5e398790d5d81edbff3f94bf3bed431695d8b33714153775a9`，來源為 [probe runs](raw/probe/runs.csv) 與 [ABBA summary](raw/latency/summary.json)。同一 corpus 的完整 bytes 比較支持本次 ordinary/graph、serial/pipelined 與 queue strategy 的數值等價，不把有限輸入等價擴張成任意輸入證明。

Profiler 結果在 [stages/summary.json](stages/summary.json)。D/M 是依 graph 內容與 launch/frame correlation 判定的邏輯 stream；graph 的 profiler 虛擬 stream IDs 不等於新增可共用 context 的實體執行 stream，nvJPEG conversion 另列。主 trace 使用 `--cuda-event-trace=false`：5,040 項是實際 activity 的必要同幀先後順序檢查，**沒有直接驗證每個 CUDA event 的代次綁定、跨幀 slot 全部衝突或所有 lifetime**；這些仍由 source ownership 審查、CPU 協定模型、ring/lifecycle、定向 ordinary 擾動與 byte 對照共同支持。

完整 session racecheck 的兩次未完成嘗試及原始退出碼保留為本次驗證限制。component racecheck、session synccheck 與全流程 memcheck 是各自受測範圍的結果，不替未完成的 gate 改判；完整 TensorRT/nvJPEG/cuFFT 內部 racecheck/synccheck 與完整 session initcheck 亦未由本次關閉。

本次不是 clean-source 正式驗收，沒有重新選模型、修改 tracker 決策或重算品質指標。若輸入形狀、buffers、streams、graph 範圍或回收點改變，須更新 ownership 表並重跑受影響的協定與 byte-equivalence 檢查。
