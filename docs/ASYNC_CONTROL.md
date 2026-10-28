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
| A08 進度保證 | 失敗後自訂計算 kernel 停止計算；begin／階段控制／commit／packet 仍執行，函式庫工作仍可寫隔離暫存區，所有 record 照常 enqueue 並執行。每個 wait 只綁定已 enqueue 的 record，happens-before 圖無環；在 GPU／API 正常運作的前提下，已提交的工作可完成，不因協定失敗死鎖。GPU fatal 另依 §6 處理。 |
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
| FFT_TMP：本幀 FFT 輸出 | M 專用 | cuFFT | 受 guard 的頻譜交付 | 1 | M 順序；不得 alias SPEC、TS 或 packet |
| SPEC[2]：提交／工作頻譜 | M 專用 | 受 guard 的頻譜交付 | 同幀相關、下一幀 F06 | 2 | M 順序＋CTL 版本；不得寫已提交 slot |
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
  begin(epoch, k)                                    // 控制 kernel：每幀必執行
  wait(mi_ready[m])
  F06 前置控制：核對 MI header，固定本階段 run_stage
  cuFFT：讀 MI[m]，只寫 FFT_TMP                       // 即使已失敗仍可能執行
  F06 後置控制：再核對 header，固定交付階段 run_stage
  頻譜交付：run_stage 為真才複製 FFT_TMP → SPEC[working_slot]
  record(mi_consumed[m])                             // 此後 M 不再讀 MI[m]
  階段控制 → phase correlation／MotionContext        // IFFT、cross-power 也只寫 scratch
  階段控制 → F08a：讀 TS[committed_slot]，寫 TS[working_slot] 與 PRIOR
  wait(cs_ready[c])
  F04b 前置控制：核對 CS header，固定 run_stage
  F04b：依 run_stage 讀 CS[c] 與 PRIOR，寫 DB
  F04b 後置控制：再核對 header
  record(cs_consumed[c])                             // 此後 M 不再讀 CS[c]
  階段控制 → F07；階段控制 → F08b：讀 DB，更新 TS[working_slot]
  commit；packet → RP                                // 兩者都不套計算 guard
  D2H RP → HP[r]
  record(pkt[r])
```

- **重疊來自 device 依賴。** H 只在 STAGE 與 HP 重用時阻塞。D(k+1) 只等 M(k+1−N) 的 consumed event，所以在 GPU 上能與 M(k) 重疊，不靠 host 的時序。
- **運動與預測可先行。** F06／F08a 排在 `wait(cs_ready)` 之前，偵測尚未完成時就能執行，符合 DATAFLOW §6.2 第 4 點。
- **RP 只需一份。** 在 M 上，packet(k+1) 排在 D2H(k) 之後。
- **event 留在 stream 層。** 基線中跨 stream 的 record／wait 放在 graph launch 之間，graph 只包單一 stream 的段落。若改用 graph 內的 event 節點或跨 stream capture，P05 須證明每次 launch 的綁定仍滿足 A04。
- **兩種索引交叉核對。** slot 索引 m、c、r 由 H 在 enqueue 時決定（以 kernel 參數，或依 parity 分開的 graph）；TS／SPEC 的版本 parity 由 device 從 CTL 推導。兩者經 header 交叉核對（§5）。
- **函式庫工作隔離。** cuFFT 的執行 API 接收輸入／輸出指標與 plan，沒有本協定的 `frame_status`／sticky fail 參數；不能假設自訂旗標會取消已排入的 FFT。這是依 [cuFFT execution API](https://docs.nvidia.com/cuda/cufft/index.html#cufft-execution) 推導的隔離要求：forward FFT 只寫 FFT_TMP，cross-power／inverse FFT 只寫各自 scratch，所有庫工作區均不得 alias 已提交狀態。plan、位址與 shape 由已驗證的固定 session 設定提供，不從錯誤 header 推導；MI 的完整可讀範圍須初始化，即使 producer 失敗未寫完也不讀未初始化資料。頻譜交付另用受 guard 的自訂 kernel；較後階段失敗時，僅工作 slot 作廢。失敗後仍執行的 FFT 在 `mi_consumed` 之前讀完 MI，故不能提早歸還小影像。

## 5. Device 端提交控制

host 要到 packet(k) D2H 之後才看得到 status，此時 M(k+1) 及之後的工作可能已經 enqueue。所以「成功才提交」和「失敗後不再推進」都必須在 device 上決定。

CTL 欄位：

| 欄位 | 含義 |
| --- | --- |
| `epoch` | session 的 epoch。 |
| `committed_frame` | 最後一次成功提交的處理序號，初值 −1。 |
| `committed_slot` | 最後成功狀態所在 slot；初始化為 1，指向已初始化的空 TS。成功 commit 才改變。 |
| `working_frame` | 本次提交的處理序號 k；即使已失敗也更新，讓後續 packet 保留各自身份。尚未 sticky fail 時必須等於 `committed_frame + 1`。 |
| `working_slot` | `1 − committed_slot`；只容許工作階段寫這份 TS／SPEC。 |
| `frame_status` | 本幀的錯誤位元，各階段以 OR 累積。 |
| `run_stage` | 前一個階段控制 kernel 固定的執行旗標；計算 kernel 的整個 grid 只讀，不得改寫。 |
| `fail_frame`、`fail_code` | sticky 失敗記錄，設定後不再清除。 |

控制與計算的責任分開如下，不能對所有 M kernel 套同一個 guard：

| 類型 | 執行規則 |
| --- | --- |
| begin | 每幀必執行；初始化本幀身份、`frame_status = 0`、`run_stage = false` 與工作 slot。不得清除 sticky fail 或已提交版本。若未失敗，核對 epoch 與預期下一幀，錯誤 OR 入本幀 PROTOCOL。 |
| 階段控制 | 單一 thread 或具明確同步的控制 kernel，在計算前完成 header／狀態核對，再設 `run_stage = (fail_code == 0 && frame_status == 0)`。後置核對仍可累積錯誤，控制本身不受計算 guard。 |
| 自訂計算 kernel | 整個 grid 只讀固定的 `run_stage`；為假就返回。可用 atomic OR 累積 `frame_status`，但本 kernel 的任何 block 都不得再以該可變值決定執行／跳過。錯誤到下一次階段控制才阻止後段計算；本階段可能完成部分工作，但只可寫工作版本／scratch。 |
| cuFFT 等函式庫 | 不受自訂 guard 取消；只可寫隔離的暫存區。讀取的 slot 仍須遵守 ready／consumed；不能因已失敗就提早歸還。 |
| commit | 每幀必執行，單一 thread。先檢查 sticky fail：已有失敗就保留原失敗與提交版本；否則本幀失敗就鎖定 `fail_frame = working_frame`、`fail_code = frame_status`；只有本幀成功才同時推進 TS／SPEC 的 `committed_slot` 與 `committed_frame`。 |
| packet | 每幀必執行。無 sticky fail 為 OK；本幀等於 `fail_frame` 為 FAILED；之後為 SKIPPED_AFTER_FAILURE。FAILED／SKIPPED 均帶首次失敗資訊，count=0 且所有 rows 清零；OK 的 count=0 才代表成功空結果。 |

每次計算前的階段控制、計算、後續控制都在 M 上依序執行。若一個階段有多個會累積錯誤的 kernel，每個 kernel 前都須重新固定旗標；同一計算 kernel 內的資料依賴仍須符合 DATAFLOW §3.2 的 phase boundary，不用旗標取代 block 間同步。

**header 核對的界限。** consumer 前後核對 header 的 epoch、frame、producer status，身份不符累積 PROTOCOL；producer 的 overflow／NaN 等 status 也合入本幀錯誤。預期 frame 是 begin 核對過的 k，另與已提交版本的下一幀核對。兩次 header 相同只能作診斷：producer 可能先覆寫 payload、尚未更新 header，兩次都讀到相同舊 header。缺邊時還可能形成未定義的衝突存取，不能保證 PROTOCOL 一定被偵測。正確性由 A02／A03 的 ready／consumed 邊及 A04 綁定保證，header 不能補救缺少同步。

**初始化與版本讀取。** session 開始明定 `committed_frame = -1`、`committed_slot = 1`、`working_frame = -1`、`working_slot = 0`、`frame_status = fail_code = 0`、`fail_frame = -1`、`run_stage = false`。TS[0]／TS[1] 均初始化成同 epoch 的空 TrackState（count／UID 配置起點／生命週期與所有欄位有明定初值）；SPEC[0]／SPEC[1] 與 scratch 清零，但頻譜 valid=false、reference frame=-1。slot header 初始化 invalid（frame=-1），RP／HP／未使用 rows／reserved bytes 清零並填入固定 schema 欄位。初始化完成後依 A09 同步，實際型別與封包初值見 [RESULT_PACKET.md](RESULT_PACKET.md)。

F08a 一律讀 TS[committed_slot]，所以首幀讀的是空 TS[1]，寫 TS[0]，絕不對 −1 作 modulo。F06 僅在 `committed_frame ≥ 0` 時讀前譜；否則輸出 uninitialized MotionContext。SPEC 的 guarded 交付與 F08a／F08b 只能寫 working_slot，即使交付後或更新途中失敗也保留最後提交 slot。成功 commit 才切換共同版本。

DATAFLOW §3.1 規定失敗即終止序列。已排入的後續幀仍有 begin／控制／commit／packet、event 與可能的函式庫 scratch 寫入；「失敗後 no-op」僅指自訂計算，不指整條 stream。最後成功 TS[committed_slot]／SPEC[committed_slot] 與其版本保持不變，無需回滾；工作 slot 可包含失敗幀的部分資料，不得作為成功狀態讀出。

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
| V01 協定模型 | [C++ 模型](../tools/protocol_model.cpp) 把 §4 的 enqueue 程序寫成 happens-before 圖（同 stream 順序、wait 綁定最近已 enqueue 的 record、host 等待），檢查每對衝突存取有序、每次讀取的預期 producer、圖無環。執行與覆蓋見 [PROTOCOL_MODEL.md](PROTOCOL_MODEL.md)。 | 有限模型中的缺邊、錯綁與死鎖；不代表實際 CUDA 實作已符合模型。 |
| V02 header 核對 | §5 的 header 診斷，後續 GPU 實作所有 build 常開。 | 可定位身份／producer status 錯誤；兩次相同 header 無法證明 payload 未遭覆寫，不能取代 ready／consumed。 |
| V03 時序擾動 | debug build 在 D、M 各階段前後插入隨機長度的延遲 kernel，以多個 seed 執行；host 的 N_H、N_RP 設得大於 slot 數。 | 一般時序下不會出現的競態。 |
| V04 serial 對照 | serial 模式使用相同 kernel，但每個階段後同步；與 pipelined 及 V03 的輸出逐位元比較每幀 ResultPacket 與最終 MOT 檔。前提是兩次 serial 執行本身逐位元相同；若 TensorRT tactic 等元件不具決定性，須先固定，或改從錄製的 RawPredictions 開始比較。 | 任何改變結果的競態或非決定性。 |
| V05 compute-sanitizer | memcheck、initcheck、synccheck、racecheck。racecheck 只檢查 shared memory hazard，不涵蓋跨 stream 的 global memory，因此不能取代 V01–V04。 | kernel 內越界、未初始化讀取（例如首幀頻譜）、block 內同步錯誤。 |
| V06 審查清單 | 每個新 buffer 或 kernel 都標明：A01 類別、寫入者與讀取者、ready／consumed edge、最後讀取 event，以及失敗時的 guard 行為。幀路徑出現 A06 禁用的 API 即不通過審查。 | 新程式碼偏離協定。 |

設定治理與行為驗證的規則由 [DATAFLOW §4.4、§8.1](DATAFLOW.md) 定義。首批 [C++ 狀態／封包模型](RESULT_PACKET.md) 已覆蓋 begin／固定階段旗標／commit／packet、錯誤注入及穩定編碼；`packet_compare` 比較完整固定容量封包串流。resolved 設定 schema／hash、tracker lifecycle／bridge 事件模型、GPU serial／pipelined 輸出與 MOT 比較仍待實作。行為 hash 與閾值距離不縮減 A10／V04 的逐位元比較範圍，也不取代 V01–V06。

V01 的早期外部腳本紀錄沒有隨 `e08a1da` 入庫，不作可重跑證據。2026-10-28 已從本規格重新建立 repo 內的 C++ 模型，以下數字由 `build/protocol_model --self-test` 重跑得到：

- **範圍：** N_MI、N_CS ∈ {1, 2, 3}，N_H ∈ {1, 2, 3}，N_RP ∈ {1, 2, 3, 4}，每組 10 幀，分別在有、無取消情境下執行（取消情境中，最後一幀只 enqueue D 段），共 216 組。
- **結果：** 216／216 組中，模型的衝突存取皆有序、讀取來源正確、event 綁定正確且圖無環；不把 CPU 圖模型的結果升格成 GPU 無競態結論。
- **變異測試：** 15 類：移除 init 同步、MI／CS ready／consumed 的 wait 或 record、STAGE／HP host wait 及對應 record、終端等待，以及 M-before-D enqueue。每類至少有一組反例；移除終端等待在 108 組 D-only tail 情境被抓到，M-before-D 在全部 216 組被抓到（含尾端取消前已提交的完整幀）。這是有限設定的敏感度檢查，不證明任意擴充協定的邊皆最小。
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
