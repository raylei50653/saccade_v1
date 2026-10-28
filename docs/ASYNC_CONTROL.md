# Saccade v1 非同步狀態流程控制

狀態：設計草案。日期：2026-10-28。上層文件：[DATAFLOW.md](DATAFLOW.md)。

本文件把 DATAFLOW §5–§7 的所有權、提交與結束語義落成可實作的執行協定，要求如下：

- 每一對跨執行者的衝突存取，都有明確的 happens-before 邊。
- 狀態提交、失敗與結束，不依賴 host 在幀中途讀回。
- 在任何 slot 數量下都不死鎖。

範圍是單一 session 內 host 與 stream 之間的順序。kernel 內部的 block 間順序（top-k、auction 的 phase boundary）屬 DATAFLOW §3.2，驗證方式見 §7。

## 1. 執行者與依賴的 CUDA 語義

| 執行者 | 內容 |
| --- | --- |
| H | 單一 host 提交執行緒：讀 JPEG、執行 nvJPEG host phase、依 k 遞增提交工作，並依序消費 ResultPacket。 |
| D | detect stream：H2D、nvJPEG decode、F05、F02、F03、F04。 |
| M | main stream：F06、F08a、F04b、F07、F08b、commit、packet、D2H。 |

D 與 M 都以 `cudaStreamNonBlocking` 建立。協定只依賴以下語義，依據見 [CUDA Runtime API：Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) 與 [Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)：

1. 同一 stream 依 enqueue 順序執行；前一個 kernel 的寫入，對同 stream 的後續 kernel 可見。
2. `cudaStreamWaitEvent` 綁定的是呼叫當下該 event 最近一次 enqueue 的 record。若 event 從未 record，這次等待是 no-op。
3. `cudaEventSynchronize`／`cudaEventQuery` 對從未 record 的 event，立即回報完成。

依第 2、3 點，若先 enqueue 等待、後 enqueue record，CUDA 不會報錯，只是等待落空。因此本協定把 enqueue 順序本身列為不變式（A04），不靠執行時剛好的時序。

## 2. 不變式

| 編號 | 不變式 |
| --- | --- |
| A01 單一歸屬 | 每塊記憶體只屬於下列一類：D 專用、M 專用、H 專用，或跨執行者 slot（H→D、D→M、M→H）。只有 slot 會被兩個執行者存取。函式庫的隱藏工作區（TensorRT execution context、nvJPEG state、cuFFT work area、CUB temp storage）也是記憶體，歸屬於執行它的 stream，不跨 stream 或跨 session 共用。 |
| A02 交付 | slot 的跨執行者寫後讀：producer 在最後一次寫入後 record ready；consumer 在第一次讀取前 wait。 |
| A03 回收 | slot 的跨執行者讀後寫：consumer 在最後一次讀取（含 header 核對與診斷讀取）後 record consumed；producer 在下一次覆寫前 wait。slot 重用不得靠 host 背壓或其他 slot 的等待間接成立。 |
| A04 enqueue 順序 | H 依「frame k 的 D 段 → frame k 的 M 段 → frame k+1」的順序 enqueue。每個 wait 都在對應的 record 之後、同一 slot 下一次 record 之前 enqueue。event 按 slot 配置，不在不同 slot 間共用。 |
| A05 host 與 device | H 改寫某個 pinned slot 前，須等 device 對它的最後讀取 event 完成；讀取前，須等 device 的寫入 event 完成。H 以 slot 帳本記錄每個 event 目前代表哪一幀，不對尚未 record 的 event 等待。 |
| A06 無隱式同步 | 幀路徑不使用 legacy default stream，不使用同步版 `cudaMemcpy`／`cudaMemset`、`cudaMalloc`／`cudaFree`，不對 pageable host 記憶體做 async copy，也不在 stream callback 中呼叫 CUDA API。所有資源在 session 開始時按 P04 的容量配置。隱式同步會遮蔽缺少的 edge：測試照樣通過，重構後才出錯。 |
| A07 device 提交 | 狀態版本只由 M 上的 commit kernel 推進。失敗以 sticky 旗標表示。host 不在幀中途讀回結果或決定提交（§5）。 |
| A08 進度保證 | 失敗後 kernel 可以不做事，但所有 record 照常 enqueue 並執行。每個 wait 只綁定已 enqueue 的 record，happens-before 圖無環，所以已提交的工作必定完成，不會死鎖。 |
| A09 生命週期 | session 初始化完成後，先作一次全 device 同步，才提交 frame 0。要釋放資源、交給下一個 session 重用或銷毀 event，須等 D、M 兩條 stream 的終端 event 都完成，且 H 已消費全部 packet（§6）。 |
| A10 時序無關 | 輸出只由輸入與設定決定。在同一裝置與 build 上，serial 與 pipelined 執行的每幀 ResultPacket 與最終 MOT 檔須逐位元相同。這依賴 DATAFLOW §3.2 的全序，以及其他元件本身的決定性。 |

## 3. 資源分類與數量

| 資源 | 類別 | 寫入者 | 讀取者 | 數量 | 重用保護 |
| --- | --- | --- | --- | --- | --- |
| STAGE[h]：JPEG 位元組、nvJPEG state 與 pinned buffer、host 端 FrameMeta | H→D | H | D（H2D、nvJPEG transfer） | N_H | H 等待 `staged[h]`（A05） |
| NVDEV[h]：nvJPEG state 綁定的 device buffer | D 專用 | D | D | 每個 state 一份 | D 順序 |
| DESC：device 端 FrameMeta | D 專用 | D（H2D） | F05、F04 | 1 | D 順序 |
| RAW：GpuFrame | D 專用 | decode | F05、F02 | 1 | D 順序 |
| MIN、RAWP：模型輸入與輸出 | D 專用 | F02、F03 | F03、F04 | 各 1 | D 順序 |
| MI[m]：MotionImage＋header | D→M | F05 | F06 | N_MI | `mi_ready[m]`／`mi_consumed[m]` |
| CS[c]：DetectionCandidates＋header | D→M | F04 | F04b | N_CS | `cs_ready[c]`／`cs_consumed[c]` |
| SPEC[2]：運動頻譜 | M 專用 | F06 | 下一幀 F06 | 2 | M 順序＋CTL 版本 |
| TS[2]：TrackState | M 專用 | F08a、F08b | 下一幀 F08a、packet | 2 | M 順序＋CTL 版本 |
| PRIOR、DB 與 M 端 scratch | M 專用 | F08a、F04b 等 | 同幀後段 | 各 1 | M 順序 |
| CTL：提交控制區 | M 專用 | M kernels | M kernels | 1 | M 順序 |
| RP：device 端 ResultPacket | M 專用 | packet | D2H | 1 | M 順序 |
| HP[r]：host 接收區 | M→H | D2H | H | N_RP | H 等待 `pkt[r]`，消費完才重新提交 |

- **原圖只需一份。** 原圖只在 D 內使用，D 串行執行已保證 decode(k+1) 在 F02(k) 讀完後才覆寫 RAW。真正需要兩份的是 STAGE：nvJPEG decoupled API 的 host phase 寫入 pinned buffer，transfer 以 DMA 讀取它；兩幀要重疊，就得各用一組 state 與 pinned buffer。這取代 DATAFLOW 早期寫的「兩組原圖」。
- **DB 的複製是必要的。** F04b 把 kept 與補回的候選壓實到 M 擁有的 DB，CS slot 因此在 F04b 後就能歸還，不必保留到 F08b。
- **slot 數只影響重疊程度。** 協定在任何 N ≥ 1 下都正確（§7 V01）。雙緩衝基線取 N_MI = N_CS = N_H = 2、N_RP ≥ 2，具體值屬 P04。
- **函式庫的 stream 行為須驗證。** TensorRT 若使用 auxiliary streams，enqueue 對外仍須表現為 D 上的有序工作。cuFFT plan 以 `cufftSetStream` 綁定 M，每個 session 各有一份 plan 與 work area。這兩點列入 P01／P05 驗證。

## 4. 每幀提交程序

以下虛擬碼是 H 的 enqueue 順序，不是 API 約定。其中 `h = k mod N_H`、`m = k mod N_MI`、`c = k mod N_CS`、`r = k mod N_RP`。

```text
submit(k):
  if k ≥ N_H:  host_wait(staged[h])                 // A05：前一次 transfer 已讀完 STAGE[h]
  if k ≥ N_RP: host_wait(pkt[r]); consume(k − N_RP) // A05：HP[r] 的前一次內容已消費
  讀 JPEG，nvJPEG host phase 寫入 STAGE[h]；填寫 host 端 FrameMeta

  ── D ──
  H2D FrameMeta → DESC；nvJPEG transfer(STAGE[h])
  record(staged[h])
  nvJPEG device decode → RAW
  if k ≥ N_MI: wait(mi_consumed[m])                 // A03
  F05 寫入 MI[m]，最後寫 header{epoch, frame, status}
  record(mi_ready[m])                                // A02
  F02 → MIN；F03 → RAWP
  if k ≥ N_CS: wait(cs_consumed[c])                 // A03
  F04 寫入 CS[c]，最後寫 header
  record(cs_ready[c])                                // A02

  ── M ──
  wait(mi_ready[m])
  F06：guard 與 header 核對；FFT 讀 MI[m]，寫 SPEC[k mod 2]；讀完再核對 header
  record(mi_consumed[m])                             // 此後 M 不再讀 MI[m]
  F08a：讀 TS[(k−1) mod 2]，寫 TS[k mod 2] 與 PRIOR
  wait(cs_ready[c])
  F04b：核對 header；讀 CS[c] 與 PRIOR，寫 DB；讀完再核對 header
  record(cs_consumed[c])                             // 此後 M 不再讀 CS[c]
  F07／F08b：讀 DB，更新 TS[k mod 2]
  commit；packet → RP
  D2H RP → HP[r]
  record(pkt[r])
```

- **重疊來自 device 依賴。** H 只在 STAGE 與 HP 重用時阻塞。D(k+1) 只等 M(k+1−N) 的 consumed event，所以在 GPU 上能與 M(k) 重疊，不靠 host 的時序。
- **運動與預測可先行。** F06／F08a 排在 `wait(cs_ready)` 之前，偵測尚未完成時就能執行，符合 DATAFLOW §6.2 第 4 點。
- **RP 只需一份。** 在 M 上，packet(k+1) 排在 D2H(k) 之後。
- **event 留在 stream 層。** 基線中跨 stream 的 record／wait 放在 graph launch 之間，graph 只包單一 stream 的段落。若改用 graph 內的 event 節點或跨 stream capture，P05 須證明每次 launch 的綁定仍滿足 A04。
- **兩種索引交叉核對。** slot 索引 m、c、r 由 H 在 enqueue 時決定（以 kernel 參數，或依 parity 分開的 graph）；TS／SPEC 的版本 parity 由 device 從 CTL 推導。兩者經 header 交叉核對（§5）。

## 5. Device 端提交控制

host 要到 packet(k) D2H 之後才看得到 status，此時 M(k+1) 及之後的工作可能已經 enqueue。所以「成功才提交」和「失敗後不再推進」都必須在 device 上決定。

CTL 欄位：

| 欄位 | 含義 |
| --- | --- |
| `epoch` | session 的 epoch。 |
| `committed_frame` | 最後一次成功提交的處理序號，初值 −1。 |
| `working_frame` | 本幀序號，等於 `committed_frame + 1`。 |
| `frame_status` | 本幀的錯誤位元，各階段以 OR 累積。 |
| `fail_frame`、`fail_code` | sticky 失敗記錄，設定後不再清除。 |

規則：

1. **起始。** M(k) 的第一個 kernel 若看到 `fail_code ≠ 0`，就不做事；否則設 `working_frame = committed_frame + 1`、`frame_status = 0`。
2. **guard。** 每個 M kernel 先檢查 `fail_code = 0` 且 `frame_status = 0`；不成立時，整個 grid 一致返回。guard 只看得到前序 kernel 的寫入，同一 kernel 內設下的錯誤從下一個 kernel 起生效。
3. **header 核對。** consumer 讀 slot 前，核對 header 的 `epoch`、`frame = working_frame` 與 producer status；讀完再核對一次，確認讀取期間 header 沒有改變。producer status 帶來的錯誤（D 端 overflow、NaN）與 header 不符，都 OR 進 `frame_status`，後者記為 PROTOCOL。header 的 frame 來自 H 經 D 傳來的 FrameMeta，`working_frame` 來自 device 計數；兩者各自推導。若 slot 綁錯或缺少 edge，而錯誤的交錯確實發生了，結果會是 PROTOCOL 失敗，不會讀到別幀資料還繼續執行。
4. **版本讀取。** F08a 讀 `TS[committed_frame mod 2]`，寫 `TS[working_frame mod 2]`。F06 只在 `committed_frame ≥ 0` 時讀 `SPEC[committed_frame mod 2]`，否則走 `uninitialized`。
5. **commit。** 由單一 thread 執行。`frame_status = 0` 時設 `committed_frame = working_frame`，TrackState 與 MotionState 頻譜因這一次寫入同時推進；否則寫入 `fail_frame` 與 `fail_code`。
6. **packet。** packet kernel 不受 guard 影響，每幀都執行。status 分 OK、FAILED（本幀失敗）與 SKIPPED_AFTER_FAILURE（附 `fail_frame`）三種；只有 status 為 OK 時，`count = 0` 才代表成功的空結果。

DATAFLOW §3.1 規定失敗即終止序列，所以不需要回滾。失敗之後的幀全部 no-op，`TS[(k−1) mod 2]` 保留最後一次提交的內容，可供診斷。device 上不設「失敗後繼續」的分支。

## 6. 失敗、取消、結束與 reset

| 事件 | 偵測者 | 傳播 | H 的處置 |
| --- | --- | --- | --- |
| D 端計算錯誤或容量 overflow | D kernel | slot header status → M 的 `frame_status` → sticky fail | 讀到 FAILED 後停止提交，執行收尾，序列失敗。 |
| M 端計算錯誤或容量 overflow | M kernel | `frame_status` → sticky fail | 同上。 |
| 協定違反 | M 的 header 核對 | PROTOCOL → sticky fail | 同上，並視為實作缺陷。 |
| host 端錯誤 | H：讀檔、nvJPEG host phase、metadata 或可回報的 CUDA API 錯誤 | 不再 enqueue 後續工作，可能留下只有 D 段的幀 | 停止提交，執行收尾，序列失敗。 |
| 取消 | H | 同 host 端錯誤 | 執行收尾，記錄為取消，不記為 EOF。 |
| sticky CUDA 錯誤 | CUDA API 回報 | context 已不可用 | 無法安全收尾，異常退出，不標記 complete（P07）。 |

正常 EOF、失敗與取消共用以下收尾程序：

1. 停止 enqueue。依 A04，已 enqueue 的 wait 都綁定已 enqueue 的 record，所以已提交的工作都會完成，包括只有 D 段的未完成幀。
2. 依序等待並消費所有已 enqueue 的 packet。
3. 在 D、M 各自的最後一個工作之後 record 終端 event，H 等待兩者完成。這一步不能省：取消時只有 D 段的幀，不在任何 packet 的覆蓋範圍內（見 §7 V01 的變異測試）。
4. 之後才釋放資源、交給下一個 session 重用或銷毀 event。
5. 判定結果：每幀 packet 都是 OK，且以正常 EOF 結束，才進入 DATAFLOW §7 的定稿與 completion。

**reset 與 epoch。** 基線中一個 session 處理一個序列，對應一個 epoch；序列中途表示 key 改變即失敗（DATAFLOW §5），沒有中途 reset。若日後擴充中途 reset，採靜止式做法：

1. 先完成上述收尾的第 1–3 步。
2. 在 M 上初始化 CTL、TS、SPEC 與 slot header，再作一次全同步。
3. 以新 epoch 從 slot 帳本起點重新提交。

這樣不會有在途的舊 epoch 工作，epoch 標記只作防禦性核對。

**多個 session。** 同一 run 的多個 session 依序執行，下一個 session 在前一個收尾後才初始化。模型與 nvJPEG handle 等共用資源只在 D 上使用，靠 D 順序與收尾即可安全共用。session 重疊屬擴充，須另列 edge。

## 7. 驗證

| 編號 | 方法 | 能抓到的問題 |
| --- | --- | --- |
| V01 協定模型 | 把 §4 的 enqueue 程序寫成 happens-before 模型（同 stream 順序、wait 綁定最近已 enqueue 的 record、host 等待），檢查三件事：每對衝突存取都有序、每次讀取讀到預期的 producer、圖無環。 | 協定本身的缺邊、錯綁與死鎖。 |
| V02 header 核對 | §5 規則 3，所有 build 常開。 | 實作缺邊或 slot 綁錯，且錯誤交錯確實發生的情況。 |
| V03 時序擾動 | debug build 在 D、M 各階段前後插入隨機長度的延遲 kernel，以多個 seed 執行；host 的 N_H、N_RP 設得大於 slot 數。 | 一般時序下不會出現的競態。 |
| V04 serial 對照 | serial 模式使用相同 kernel，但每個階段後同步；與 pipelined 及 V03 的輸出逐位元比較每幀 ResultPacket 與最終 MOT 檔。前提是兩次 serial 執行本身逐位元相同；若 TensorRT tactic 等元件不具決定性，須先固定，或改從錄製的 RawPredictions 開始比較。 | 任何改變結果的競態或非決定性。 |
| V05 compute-sanitizer | memcheck、initcheck、synccheck、racecheck。racecheck 只檢查 shared memory hazard，不涵蓋跨 stream 的 global memory，因此不能取代 V01–V04。 | kernel 內越界、未初始化讀取（例如首幀頻譜）、block 內同步錯誤。 |
| V06 審查清單 | 每個新 buffer 或 kernel 都標明：A01 類別、寫入者與讀取者、ready／consumed edge、最後讀取 event，以及失敗時的 guard 行為。幀路徑出現 A06 禁用的 API 即不通過審查。 | 新程式碼偏離協定。 |

設定治理與行為驗證的規則由 [DATAFLOW §4.4、§8.1](DATAFLOW.md) 定義：核對 resolved 設定與 build 身分，分別檢查狀態機合法性、同 build 執行一致性及觀察開關輸出不變。行為 hash 與閾值距離僅協助定位，不縮減 A10／V04 的逐位元比較範圍，也不取代 V01–V06 的同步驗證。這些新增工具尚待實作，不列入下述 V01 已執行的紀錄。

V01 已於 2026-10-28 對本協定執行，模型腳本尚未放入 repo。

- **範圍：** N_MI、N_CS ∈ {1, 2, 3}，N_H ∈ {1, 2, 3}，N_RP ∈ {1, 2, 3, 4}，每組 10 幀，分別在有、無取消情境下執行（取消情境中，最後一幀只 enqueue D 段），共 216 組。
- **結果：** 沒有競態，每次讀取都讀到預期的 producer，沒有死鎖。
- **變異測試：** 逐一拿掉 init 後的全同步、各個 ready／consumed 的 wait 或 record、host 對 STAGE 與 HP 的等待，以及終端收尾；每一項都至少在部分設定中被抓到，表示協定裡沒有多餘的 edge。另外，把 M(k) 排在 D(k) 之前 enqueue，不含取消情境的 108 組設定全部抓到未綁定的 wait。
- **遮蔽現象：** 拿掉 `mi_consumed`／`cs_consumed` 的 wait 時，只在 84／216 組被抓到。其餘設定中，host 對 HP 的等待，或另一組 slot 的 consumed 等待，剛好間接提供了相同的順序。因此 A03 要求 slot 重用不依賴這類間接順序，V03 也要求 host ring 大於 slot 數，以免測試被遮蔽。

## 8. 與 DATAFLOW 的對應

| DATAFLOW | 本文件 |
| --- | --- |
| E01、E02、E03 | D 專用資源，靠 D 順序。 |
| E04 | `cs_ready`／`cs_consumed`；DB 為 M 專用。 |
| E05 | `mi_ready`／`mi_consumed`。 |
| E06、E11 | M 專用，靠 M 順序。 |
| E07、E10 | CTL 的 device commit，TS 與 SPEC 同一次推進。 |
| E08 | `pkt[r]` 與 HP 帳本。 |
| E09 | 診斷讀取計入 slot 的最後讀取，排在 consumed record 之前。 |
| P05 | graph 內 event 的綁定、parity graph 與 kernel 參數的取捨、TensorRT aux stream 的驗證。 |
