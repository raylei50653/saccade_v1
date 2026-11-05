# Tracking 半段規格（F05–F09）

規格日期：2026-11-03；文件狀態更新：2026-11-04。M0–M4 完成（M4 結果見 §17.5），P01 CLOSED（有範圍）；CUDA Graph／排序改版已完成最新 clean-source 驗收（§19）。上層文件：[DATAFLOW](DATAFLOW.md)、[ASYNC_CONTROL](ASYNC_CONTROL.md)、[ResultPacket](RESULT_PACKET.md)；總覽見 [STATUS](STATUS.md)。

本文件把 DATAFLOW 已定的 F05–F09 落成可實作的數值與決策規格，並收斂 P03（決策參數）、P04（容量）、P06（輸出）中 M1 需要的部分。profile 名稱：motion `v1-motion/1`、tracker `v1-tracker/1`、輸出 `v1-mot/1`。

## 1. 原則

- **精簡核心先跑通。** 舊專案的追蹤器累積了很多加權項與開關（OAO、front-occluder、velocity-direction、DDA、adaptive anchor、direction bonus 等）。v1 先做核心，端到端量到 HOTA 後，再以 MOT17-10（val）的量測逐項決定是否加入。沒有量測支持的項目不加。
- **調參只用 val。** T01 的 split 固定 MOT17-02 為獨立 eval；任何參數調整只看 MOT17-10。MOT17-04／05／09／11／13 是偵測器訓練序列，結果另列為 in-domain。
- **CPU reference 與 CUDA 逐位元對照。** tracker 的決策是離散的，浮點差一點就可能改變配對。因此 F05、F04b、F08 的 CPU reference 與 CUDA 共用 FP32 純量數學（§2），要求逐位元相同；只有 F06 的 FFT 用容差對照。
- **規格寫死邊界情況。** 下文明列的邊界處置就是全部；實作不自行新增驗證、fallback 或額外錯誤路徑。

## 2. 共用數值規則

- 全部 tracker 狀態與決策計算採 IEEE binary32。CPU TU 以 `-ffp-contract=off` 編譯；CUDA tracker TU 以 `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false` 編譯。加、減、乘、除、`sqrtf` 因此在兩邊都是正確捨入。
- box、IoU、Kalman、cost 與 `det_expf` 等純量函式放在 `include/saccade/tracking_math.hpp`，以 `SACCADE_HD inline`（CUDA 下為 `__host__ __device__`）讓 CPU reference 與 CUDA 共用同一份程式碼，運算順序自然一致。
- `det_expf(x)`：只用 float 加減乘、`floorf` 與整數位元操作的指數函式（Cody–Waite range reduction＋固定次數多項式＋位元組合 2^n）。輸入先 clamp 到 [-87, 88]。對 std::exp 的誤差 ≤ 2 ulp，以測試確認。不得呼叫 `expf`／`__expf`。
- 會進入決策或狀態的值若出現 NaN／Inf，記 NUMERIC 錯誤（§9.8）。

## 3. 參數（P03）與容量（P04）

所有參數都是 typed config 的欄位，預設值即本 profile。

| 欄位 | 值 | 階段 | 意義 |
| --- | --- | --- | --- |
| `motion.downscale` | 4 | F05 | 原圖到 MotionImage 的整數縮小倍率 |
| `motion.pcr_threshold` | 5.0 | F06 | PCR 達此值時信心為 1 |
| `motion.max_shift_fraction` | 0.25 | F06 | 位移超過小影像該軸的此比例即 rejected |
| `kalman.r_scale` | 3.5 | F08 | 量測噪聲倍率（舊 m preset 值） |
| `kalman.std_position` | 1/20 | F08 | 位置標準差相對高度 |
| `kalman.std_velocity` | 1/160 | F08 | 速度標準差相對高度 |
| `batch.track_thresh` | 0.05 | F04b | kept 候選進入 DetectionBatch 的最低 observed score |
| `continuation.enabled` | true | F04b | 是否組裝 continuation-only 候選 |
| `continuation.min_score` | 0.10 | F04b | suppressed 候選的最低 observed score |
| `continuation.prior_iou` | 0.30 | F04b | 與任一 prior 的 IoU 下限 |
| `continuation.dedup_iou` | 0.70 | F04b | 與 kept／已接受候選的 IoU 上限 |
| `continuation.max_candidates` | 50 | F04b | 每幀最多接受的 continuation 候選數 |
| `assoc.high_thresh` | 0.45 | F08b | 高分段下限 |
| `assoc.mid_thresh` | 0.10 | F08b | 中分段下限 |
| `assoc.match_thresh` | 0.50 | F08b | S1–S3 的 cost 上限 |
| `assoc.stage2_match_thresh` | 0.50 | F08b | S4 的 cost 上限 |
| `assoc.lambda` | 10 | F08b | `p = exp(-λc)` 的溫度 |
| `assoc.stability_cost_w` | 0.20 | F08b | size-stability reward 權重 |
| `assoc.top_k` | 3 | F08b | 每 track 每 stage 的候選數 |
| `assoc.auction_eps` | 0.01 | F08b | bid 的最小增量 |
| `assoc.stability_bid_w` | 0.10 | F08b | bid 的高度相似加成 |
| `life.new_track_thresh` | 0.28 | F08b | birth 的最低 observed score |
| `life.confirm_streak` | 3 | F08b | tentative → confirmed 的連續命中數 |
| `life.confirm_score` | 0.50 | F08b | 確認時的平均 observed score 下限 |
| `life.track_buffer` | 30 | F08a | lost 可保留的最大幀數 |
| `bridge.enabled` | true | F08b | bidirectional bridge relink |
| `bridge.trigger_streak` | 4 | F08b | 新 track 第幾次連續命中時嘗試 bridge |
| `bridge.min_lost` | 2 | F08b | lost 至少幾幀才可被接手 |
| `bridge.max_distance` | 0.4 | F08b | 正規化 bridge 距離上限 |
| `bridge.height_ratio_lo`／`_hi` | 0.6／1.7 | F08b | 平均高度比 gate |
| `bridge.margin` | 0.05 | F08b | 次佳與最佳距離的最小差 |
| `bridge.speed_norm` | 0.12 | F08b | 速度權重的正規化常數 |
| `output.interpolate_max_gap` | 35 | F09 | 可插值的最大缺幀數 |
| `output.interpolate_min_rows` | 5 | F09 | 至少有幾筆觀測才插值 |

| 容量 | 值 | 超出時 |
| --- | --- | --- |
| TrackState tracks | 512 | birth 超出：OVERFLOW，本幀失敗 |
| DetectionBatch rows | 300 | 不會超出（全部來自 F04 的 300 筆候選） |
| continuation 候選 | 50 | 選取上限，達到即停止，不是錯誤 |
| ResultPacket rows | 256（v1 wire profile） | OVERFLOW，本幀失敗 |
| bridge history | 每 track 4 筆 | ring，保留最近 4 筆 |
| MotionImage | 寬高各 ≥ 8 | 設定錯誤，session 建立時拒絕 |

舊專案已移除或不帶入：DDA stage、Mahalanobis gate（在 `match_thresh=0.5` 下實際不起作用，配對需要 IoU 約 0.49 以上）、OAO、front-occluder、velocity-direction、freshness bid、adaptive anchor、direction bonus、bridge TTL（由 `track_buffer` 取代）。

## 4. F05 MotionImage

輸入是 F01 的 planar RGB8（`PreprocessedFrame::rgb`，R、G、B 三個平面），原圖 W×H。

- `w = floor(W/d)`、`h = floor(H/d)`，d 為 `motion.downscale`；寬或高 < 8 時拒絕。
- 小影像像素 (x, y) 對應原圖 d×d 區塊 `[x·d, x·d+d) × [y·d, y·d+d)`。區塊內依列優先（dy 外層、dx 內層）累加：`g = (0.299f·R + 0.587f·G) + 0.114f·B`，其中 R、G、B 為 `float(uint8)`；`s += g`。
- `v = (s / float(d·d)) / 255.0f`，再乘 window：`(v · wx[x]) · wy[y]`。
- `wx[i] = float(0.5·(1 − cos(2π·i/(w−1))))`，以 binary64 計算後轉 float；wy 同理。window 表在 host 端建立一次，CUDA 使用同一張表。
- 輸出為 float、列優先、w×h，與原圖不 alias。CPU reference 與 CUDA 輸出逐位元相同。
- 表示 key 為 `(W, H, d, w, h)`；序列中途改變即序列失敗（DATAFLOW §5）。

## 5. F06 Phase correlation

- `S = FFT2(MotionImage)`。CPU reference 以 binary64 的可分離直接 DFT 計算；CUDA 用 cuFFT（R2C／C2R 可）。
- 每個 epoch 的第一幀：`status = uninitialized`、`t = 0`、peak 與 pcr 為 0；本幀頻譜照常成為下一幀的參考。
- 其後：`C = conj(S_prev) · S_cur`，`R = C / (|C| + 1e-6)`，`r = Re(IFFT2(R))`（尺度不影響結果）。
- peak `(px, py)` = r 的最大值位置；同值取列優先索引最小者。`px > w/2`（整數除法）時 `px −= w`，y 軸同理。
- `rms = sqrt(mean(r²))`。`rms ≤ 0` → `rejected`。`PCR = r_max / rms`，`confidence = min(1, PCR / pcr_threshold)`。
- `|px| > floor(max_shift_fraction · w)` 或 `|py| > floor(max_shift_fraction · h)` → `rejected`。
- 否則 `valid`：`tx = float(px·d) · confidence`、`ty = float(py·d) · confidence`。方向是「參考幀 → 當前幀」：正 tx 表示畫面內容往右移。
- `rejected` 的處置（P03 的繼續策略）：`t = 0`，序列繼續；本幀頻譜仍在成功 commit 時成為下一參考，MotionState 與 TrackState 同步推進。
- 任何非有限值 → NUMERIC。
- 對照：peak 必須相同；pcr 相對誤差 ≤ 1e-4；`confidence = 1` 時 t 逐位元相同。

## 6. F08a GMC control 與 predict

狀態 `x = (cx, cy, a, h, vx, vy, va, vh)`，`a = w/h`；共變異 P 為 8×8。每幀對每個 active track（依 UID 升序）恰好執行一次：

1. control：`valid` 時 `cx += tx`、`cy += ty`；其他狀態不動。
2. predict：`x ← F·x`，`P ← F·P·Fᵀ + Q`，F 為 constant velocity。Q 以 predict 後的 h 計算：`σp = std_position·h`、`σv = std_velocity·h`，`diag(Q) = (σp², σp², 1e-4, σp², σv², σv², 1e-10, σv²)`。
3. `age += 1`（距上次命中的幀數）。
4. 移除：`age > track_buffer`、`h < 1` 或 `a ≤ 0` 的 track 在此移除。x 或 P 出現非有限值 → NUMERIC。

PredictedPrior = 留下來的 confirmed tracks（含 lost）的預測 box。PredictedState 帶來源 committed frame（k−1，首幀為 −1）、目標 frame k 與使用的 MotionContext。

## 7. F04b DetectionBatch

先核對候選 header：epoch／frame 必須等於本幀，`done == kPostprocessDone`，`flags == 0`。`kPostprocessOverflow` 記 OVERFLOW，其他 flags 記 PRODUCER，不符記 PROTOCOL。

1. kept 候選：`score ≥ track_thresh` 者進入 batch；`observed = association = score`，`can_continue = can_birth = true`。
2. continuation（`enabled` 且有 prior）：suppressed 候選中 `score ≥ min_score` 者，依 `(score 降序, candidate_id 升序)` 逐一判斷，同時滿足下列條件才接受，直到達 `max_candidates`：
   - 與至少一個 prior 的 IoU ≥ `prior_iou`；
   - 與每個 kept 候選（不論是否通過 `track_thresh`）及每個已接受的 continuation 候選的 IoU ≤ `dedup_iou`。

   接受者 `observed = score`、`association = min(score, new_track_thresh)`、`can_continue = true`、`can_birth = false`。
3. batch 依 candidate_id 升序排列。

IoU：`inter = max(0, min(x2) − max(x1)) · max(0, min(y2) − max(y1))`，`union = area_a + area_b − inter`，`union > 0 ? inter/union : 0`。

## 8. F07 同幀核對

候選、MotionContext、PredictedState 的 epoch 與目標 frame 都必須是 k；PredictedState 的來源 committed frame 必須是 k−1（首幀 −1）。不符記 PROTOCOL。本幀已有任何錯誤時，不做關聯，也不 commit。

## 9. F08b 關聯、生命週期與提交

### 9.1 Cost

對 track i（預測 box，高 h_i）與 detection j（box，寬 w_j、高 h_j）：

- `IoU_ij ≤ 0` 的配對不成立。
- `R = (stability_cost_w / max(λ, 1)) / (1 + |h_i − h_j| / max(h_j, 1e-3))`
- `c = clamp(1 − IoU_ij · det_expf(R), 0, 1)`
- `p = det_expf(−λ·c) · G(w_j / h_j)`，其中 `G(r) = max(0.5, 1 − (r − 0.8))`（r > 0.8）、`max(0.5, 1 − 5·(0.15 − r))`（r < 0.15）、其餘為 1。
- DATAFLOW §3.2 的 `rank_cost = −p`。

### 9.2 Stages

依序執行；每個 stage 開始時固定未配對的 track 與 detection 集合。「confirmed」包含 lost 的 confirmed tracks。

| Stage | Tracks | detection 的 association score | cost 上限 |
| --- | --- | --- | --- |
| S1 | confirmed | `[high_thresh, ∞)` | `match_thresh` |
| S2 | confirmed | `[mid_thresh, high_thresh)` | `match_thresh` |
| S3 | tentative | `[mid_thresh, ∞)` | `match_thresh` |
| S4 | confirmed | `[track_thresh, mid_thresh)` | `stage2_match_thresh` |

配對在 stage s 合格的條件：兩邊都未配對、track 狀態與 detection 分數段符合、`IoU > 0`、`c ≤ 上限`。本 profile 只有 person 一類，不比 class。

### 9.3 單輪 auction

1. 每個 track 取合格 detections 的 top-K，依 `(p 降序, candidate_id 升序)` 排列。
2. 有候選的 track 對第一名 j* 出價：`Δ = (候選 ≥ 2) ? (p₁ − p₂) + ε : ε`，`bid = Δ + stability_bid_w / (1 + |h_i − h_j*| / max(h_j*, 1e-3))`。
3. 每個收到出價的 detection 取 bid 最高者；同 bid 依 `(p 降序, track UID 升序)`。
4. 得標者配對；落選的 track 不在本 stage 重試，留給後續 stage。

### 9.4 更新

配對的 track：

- Kalman update，`z = (cx, cy, w/h, h)`。`diag(R) = ((σh)², (σh)², 1e-2, (σh)²) · r_scale`，σh 為 `std_position ·` update 前的 h。`S = HPHᵀ + R`，`K = PHᵀS⁻¹`，`x += K(z − Hx)`，`P = (I − KH)P`。
- `age = 0`、`hit_streak += 1`、`score_sum += observed`，記錄本幀的 observed／association score、class 與 `last_observed_frame = k`。
- history 推入 detection 的 `(cx, cy, h)`；`h_ema = 0.95·h_ema + 0.05·h`。
- tentative 且 `hit_streak ≥ confirm_streak` 且 `score_sum / hit_streak ≥ confirm_score` → confirmed。

未配對：tentative 直接移除；confirmed 的 `hit_streak = 0`、`score_sum = 0`（age 已在 F08a 推進）。

### 9.5 Bridge relink

在更新之後、birth 之前執行。

- **接手者**：本幀命中、`hit_streak == trigger_streak`、`k − birth_frame == trigger_streak − 1`（出生後連續命中），且 `output_id == uid`（從未接手過身份）。
- **lost**：本幀未命中的 confirmed track，`min_lost ≤ age ≤ track_buffer`，且 `last_observed_frame < 接手者的 birth_frame`（兩者不曾同時出現）。
- anchor 使用 box 中心。ℓ = lost 最後一筆 history 的 (cx, cy)，c = 接手者最早一筆 history 的 (cx, cy)。v_ℓ、v_c 為各自 4 筆 history 的回歸速度：`v = (3p₃ + p₂ − p₁ − 3p₀) / 10`（p₀ 最舊；x、y 分開算）。lost 的 history 不足 4 筆時不成為候選。`g = 接手者 birth_frame − lost 的 last_observed_frame`。
- `h_ref = max((h_ema,ℓ + h_ema,c)/2, 1)`；`r_fwd = ‖ℓ + v_ℓ·g − c‖ / h_ref`；`r_bwd = ‖c − v_c·g − ℓ‖ / h_ref`；`d_h = ‖ℓ − c‖ / h_ref`；`w = sqrt(clamp(‖v_ℓ‖ / h_ref / speed_norm, 0, 1))`；`d = w·(r_fwd + r_bwd)/2 + (1 − w)·d_h`。
- gate：`d ≤ max_distance`，且 `h_ema,ℓ / h_ema,c ∈ [height_ratio_lo, height_ratio_hi]`。
- 每個接手者在通過 gate 的 lost 中，依 `(d 升序, lost UID 升序)` 選第一名；若有第二名且 `d₂ − d₁ < margin`，放棄本次 bridge。
- 多個接手者選中同一 lost：依 `(接手者本幀 observed score 降序, 接手者 UID 升序)` 取一個。輸家不改選。
- commit：得標者 `output_id = lost.output_id`，lost 移除。得標者保留自己的 UID、Kalman 狀態與 history。

### 9.6 Birth

所有 stage 結束後仍未配對的 detection，依 candidate_id 升序：`can_birth` 且 `observed ≥ new_track_thresh` 才建立 tentative track。

- `uid = next_uid++`（每 epoch 從 1 開始，不重用），`output_id = uid`。
- `x = (cx, cy, w/h, h, 0, 0, 0, 0)`，`diag(P) = ((2σp)², (2σp)², 1e-4, (2σp)², (10σv)², (10σv)², 1e-10, (10σv)²)`，σp、σv 以 detection 的 h 計算。
- `age = 0`、`hit_streak = 1`、`score_sum = observed`、`birth_frame = last_observed_frame = k`、history 一筆、`h_ema = h`；並套用 §9.4 的確認條件。
- active tracks 超過 512 → OVERFLOW。

### 9.7 TrackSnapshot 與 ResultPacket

rows 是本幀命中（`age == 0`）的 confirmed tracks，依 UID 升序。每筆 row：uid、output_id、class、flags = observed、更新後狀態換算的 xyxy、本幀 observed_score 與 association_score。v1 profile 不輸出 predicted rows。超過 256 rows → OVERFLOW。成功時 TrackState(k) 成為已提交版本，`committed_frame = k`。

確認之前的 tentative 幀不補輸出。

### 9.8 錯誤

錯誤以 ResultPacket 的位元回報：PROTOCOL（身份或版本不符）、PRODUCER（上游 flags）、OVERFLOW（容量）、NUMERIC（非有限值）。任何錯誤都使本幀 FAILED、序列 sticky 失敗、之後各幀 SKIPPED_AFTER_FAILURE（[RESULT_PACKET](RESULT_PACKET.md)）。

## 10. F09 MOT 輸出

- 只接受 OK 封包。host 依幀序處理；output_id 依首次出現順序（幀序，同幀依 row 順序）映射為 MOT ID，從 1 開始，每序列獨立。
- 每筆觀測記 `(frame, id, xyxy, conf = observed_score)`。MOT frame 號 = 處理序號 + 1（MOT17 `img1/000001.jpg` 對應 frame 1）。
- EOF 且所有幀 OK 後才插值：同一 ID 觀測數 ≥ `interpolate_min_rows` 時，相鄰兩筆觀測 `f_a < f_b` 若 `1 < f_b − f_a ≤ interpolate_max_gap + 1`，對中間每幀以 binary64 線性插值 xyxy，conf 寫 −1。
- 輸出依 `(frame, id)` 排序，每行 `frame,id,left,top,width,height,conf,-1,-1,-1`；left／top／width／height 由 xyxy 轉換，數值以 C locale `%.3f` 格式化，插值行的 conf 寫 `-1`。
- 先寫 `<path>.tmp`，完整寫完並 close 成功後 rename。任何幀非 OK、缺幀或寫入失敗時不產生輸出檔，並刪除 `.tmp`。

## 11. 驗證（依 2026-11-03 決定的驗收強度）

模組層級：

- CPU reference 單元測試，用手算案例涵蓋本文件明列的每條規則與邊界（首幀、空 batch、rejected GMC、同分全序、continuation 不可 birth、確認、lost、移除、bridge 各 gate、容量 overflow）。
- CUDA 與 CPU reference 對照：F05、F04b、F08 逐位元；F06 依 §5 容差。
- 新增 CUDA 程式碼跑 memcheck。

完整正式驗收（clean source、四種 sanitizer、evidence report）只在 M3 整合完成與 M4 結案時各做一次。

端到端：MOT17 七個 SDP 序列分 val（10）、eval（02）、in-domain（04／05／09／11／13）三組記錄 HOTA／IDF1／MOTA／FPS，不設門檻。

## 12. 里程碑與 M1 分工

| 里程碑 | 內容 |
| --- | --- |
| M0 | 本文件與 [`motion.hpp`](../include/saccade/motion.hpp)、[`tracking.hpp`](../include/saccade/tracking.hpp)、[`mot_output.hpp`](../include/saccade/mot_output.hpp) 契約。 |
| M1 | 三條平行實作（下表），出口是一個 MOT17 序列的端到端 MOT 檔與 HOTA。 |
| M2 | F04b／F07／F08 的 CUDA 版：單一 thread block 處理一幀（tracks ≤ 512、detections ≤ 300），block 內 barrier 分隔 phase，不使用跨 block atomics；與 CPU reference 逐位元對照。 |
| M3 | 完整 tracking session：先 serial 再 double-buffer，接 ASYNC_CONTROL §4 的 CTL／commit；serial 與 pipelined 的 ResultPacket 與 MOT 檔逐位元一致（A10／V04）。 |
| M4 | MOT17 全序列評估與 P01 收尾（§17）。 |

M1 分工（每條線只改自己擁有的檔案；契約 header 只能做本線範圍內的增補，跨線變更回報給整合者）：

| 線 | 內容 | 擁有的檔案 |
| --- | --- | --- |
| A | F05／F06：CPU reference、CUDA（F05 在 detect stream、F06 在 main stream 用 cuFFT、頻譜雙緩衝）、`motion_sequence` 工具輸出每幀 MotionContext | `motion.hpp`、`motion_cuda.hpp`、`src/motion*.cpp`、`cuda/motion.cu`、`tests/motion*`、`tools/motion_sequence.cpp`、`cmake/Motion.cmake` |
| B | F08a／F04b／F07／F08b 的 CPU reference 與 `TrackerReference` session、`tracking_math.hpp` | `tracking.hpp`、`tracking_math.hpp`、`src/tracking*.cpp`、`tests/tracking*`、`cmake/TrackingReference.cmake` |
| C | F09 輸出、偵測紀錄格式、`detect_sequence` 工具（用 DetectionSession 跑整個序列並存下每幀候選） | `mot_output.hpp`、`src/mot_output.cpp`、`tests/mot_output*`、`tools/detect_sequence.cpp`、`cmake/MotOutput.cmake` |

## 13. M1 結果（2026-11-03）

M1 以 `detect_sequence`（native F01–F04 detection session）與 `motion_sequence`（CUDA F05／F06）錄下 7 個 MOT17 SDP 序列，再用 `track_offline`（CPU `TrackerReference`＋F09）產生 MOT 檔，以 `scripts/eval_mot17.sh`（舊專案的 TrackEval，只作量測工具）評估。tracker 每序列 8–108 ms（CPU）。

| 組別 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| val（MOT17-10） | 42.2 | 42.1 | 42.7 | 50.8 | 45.5 | 94 |
| eval（MOT17-02） | 31.6 | 23.7 | 42.2 | 37.2 | 23.5 | 31 |
| in-domain（04／05／09／11／13） | 75.0 | 78.1 | 72.2 | 81.7 | 85.7 | 243 |

- GMC 的效果（MOT17-10，`track_offline` 的 motion 參數給 `-` 即停用）：HOTA 34.8 → 42.2，AssA 31.3 → 42.7，IDSW 202 → 94。
- 瓶頸在偵測器的泛化：直接把 MOT17-10 的 kept detections（score ≥ 0.3）當結果評估，DetA 40.4、DetRe 43.4、LocA 79.4；配對框對 GT 沒有系統性偏移（四邊平均偏移在 ±2% 以內），所以不是座標映射錯誤。這與 T01 記錄的 val AP50 0.676、eval AP50 0.544 一致。in-domain 序列的偵測好，tracking 指標也高。
- 依 §1，任何 tracker 參數或加權項的調整只看 MOT17-10。

## 14. M3：Tracking session 規格

`TrackingSession` 把 F01–F09 接成一條 native 執行路徑，依 [ASYNC_CONTROL](ASYNC_CONTROL.md) §3–§6 的協定實作。D 側沿用 DetectionSession 已驗證的做法（F01／F02 → F03 → producer → F04、CS ready／consumed、首次失敗鎖定、drain／close）；M 側換成 tracking。DetectionSession 本身不改。

### 14.1 每幀提交順序

D（detect stream）：F01／F02 → F05（等 `mi_consumed[m]` 後寫 MI[m]，record `mi_ready[m]`）→ F03 → producer → F04（等 `cs_consumed[c]` 後寫 CS[c]，record `cs_ready[c]`）。

F05 是 F01 RGB buffer 的最後一個讀者。F05 之後 record `rgb_read_done`，下一幀呼叫 F01 前 host 先等它（`cudaEventSynchronize`）：nvJPEG HARDWARE decode 寫 RGB 時不排在 D 上先前的工作之後，只靠 stream 順序會讓下一幀的解碼覆寫 F05 還沒讀的 RGB。獨立重現見 [evidence/preprocess/2026-11-03-nvjpeg-stream-order](../evidence/preprocess/2026-11-03-nvjpeg-stream-order/README.md)。

M（main stream）：

| 順序 | 工作 | 類型 |
| --- | --- | --- |
| 1 | begin | 控制，每幀必執行 |
| 2 | wait `mi_ready[m]`；cuFFT forward（MI[m] → FFT_TMP） | 函式庫，不受 guard |
| 3 | stage 控制；頻譜交付（FFT_TMP → SPEC[working]）；record `mi_consumed[m]` | 控制＋受 guard 的計算 |
| 4 | cross-power、IFFT、MotionContext | 計算受 guard；IFFT 只寫 scratch |
| 5 | stage 控制（併入 F06 的 NUMERIC）；F08a：讀 TS[committed]，寫 PREDICTED | 控制＋計算 |
| 6 | stage 控制（併入 PREDICTED.error）；wait `cs_ready[c]`；F04b：讀 CS[c]，寫 BATCH；record `cs_consumed[c]` | 控制＋計算 |
| 7 | stage 控制（併入 BATCH.error）；F07／F08b：寫 TS[working] 與 ROWS | 控制＋計算 |
| 8 | stage 控制（併入 ROWS.error）；commit | 控制 |
| 9 | packet：寫 device RP（v1 wire bytes）；D2H 到 HP[r]；record `pkt[r]` | 控制 |

### 14.2 CTL 與 guard

- CTL 欄位與初值依 ASYNC_CONTROL §5：`epoch`、`committed_frame=-1`、`committed_slot=1`、`working_frame`、`working_slot`、`frame_status`、`run_stage`、`fail_frame=-1`、`fail_code=0`。TS[0]、TS[1] 都初始化為同 epoch 的空 TrackState（`committed_frame=-1`、`next_uid=1`）；SPEC 清零。
- stage 控制是單 thread kernel：把前一階段輸出的 error 位元 OR 進 `frame_status`，再設 `run_stage = (fail_code == 0 && frame_status == 0)`。
- 自訂計算 kernel 開頭讀 `run_stage`，為假就整個 grid 直接返回。M2 的三個 tracker entry point 與 MotionCuda 的自訂 kernel 加一個可為 null 的 `run_stage` 指標（null 表示一律執行），保持 M2／M1 既有測試結果不變。
- commit：已有 sticky failure 就不動；否則本幀 `frame_status ≠ 0` 時鎖定 `fail_frame = k`、`fail_code = frame_status`；成功時同時翻轉 TS 與 SPEC 共用的 `committed_slot`，`committed_frame = k`。MotionCuda 的 committed 頻譜與 reference frame 改由 CTL 決定（M1 的 host 端 commit 保留給 `motion_sequence`）。
- packet：OK／FAILED／SKIPPED_AFTER_FAILURE 依 [RESULT_PACKET](RESULT_PACKET.md)；device 端逐欄寫出 12,368 bytes，必須與 host `encode_packet` 對同一邏輯封包的輸出逐位元相同（含 `-0 → +0`、reserved 與未用 rows 為 0）。

### 14.3 API 與模式

- `TrackingSessionConfig`：epoch、device、序列原圖寬高（建立 MotionImageKey）、`candidate_slots`、`motion_slots`、`result_slots`、`PreprocessConfig`、`MotionConfig`、`TrackerConfig`、`mode`（serial／pipelined）、測試用 `perturb_seed`（0 表示關閉）。
- `enqueue(jpeg, frame)`、`receive()`（回傳 raw `PacketBytes` 與 decode 後的 `ResultPacket`）、`pending()`、`finish()`、`cancel()`、`close()`、`state()`、`identity()`，語義與 DetectionSession 相同（結果 ring 滿時 `length_error`、可恢復）。
- serial：每次 enqueue 後 host 等 D、M 都完成，作為 V04 的依賴參考。pipelined：只靠 event 與 ring 背壓，detect(k+1) 可與 main(k) 重疊。
- `perturb_seed ≠ 0`：在 D、M 各 stage 邊界插入長度由 seed 決定的 spin kernel（V03），只供測試。另有兩段固定長度的 spin，讓對應的順序錯誤一定會顯現：建構時在 M 的 tracking storage 初始化前一段，以及每幀在 D 的 F05 之前一段（模擬 F05 晚於下一幀解碼）。
- 工具 `track_sequence ENGINE SEQ_DIR OUT.txt [--serial] [--packets OUT.bin] [--epoch N]`：TrackingSession → F09 SequenceOutput → MOT 檔；`--packets` 存下完整 raw packet stream。

### 14.4 驗收（M3 實作階段）

- device packet writer 對 `encode_packet`：隨機邏輯封包（含 FAILED／SKIPPED、負零、滿 256 rows）逐位元相同。
- 同一序列 serial 與 pipelined 的 packet stream 逐位元相同；兩個 `perturb_seed` 的 pipelined 也相同（A10／V04）。
- 與 CPU 對照：TrackingSession 的 packet stream 等於以 `m1-data` 的 SCDR／SCMO 跑 `TrackerReference` 再 `encode_packet` 的結果；`track_sequence` 的 MOT 檔與 `track_offline` 的逐位元相同。前提是 F03 與 cuFFT 在同一裝置與 build 下重跑決定性；若不相同，先定位是哪個輸入不同，不放寬比較。
- 失敗：producer flag 注入使第 k 幀 FAILED、之後 SKIPPED，committed 狀態凍結；EOF／cancel／D-only tail 都能 drain；close 可重複呼叫。
- 新 CUDA 程式碼 memcheck（短序列）。完整正式驗收（四種 sanitizer、clean source、evidence report）在 M3 合併後另做。

### 14.5 M3 正式驗收（在 capture 前固定）

由 [`scripts/run_tracking_session_acceptance.sh`](../scripts/run_tracking_session_acceptance.sh) 在 GPU lock 下一次跑完，證據存於 `evidence/tracking-session/`。部署設定為 §16 的 HR FP16 engine（`8292cc23…`）；CPU reference 的逐位元對照另用錄製 M1 records 的 T01 engine。capture 前的 sanitizer 探測找到兩個 session 的 cross-stream 順序錯誤與 `DeviceTrack` padding 的 initcheck 報告，已在 1c03e22 修正。要通過以下全部項目：

1. **Clean source**：從乾淨 checkout 的單一 commit 執行。每一步前後核對 HEAD 與 build inputs 的 SHA-256。capture 前後，binaries、兩個 engine、reference corpus、M1 records 與 7 個 MOT17 SDP 序列的影像 hash 都不變。
2. **Build 與測試**：全新 CPU build 的預設 CTest 全部通過。全新 CUDA build（所有功能開啟）的 CTest 全部通過，包含 7 序列 TrackingSession 對 CPU reference 的 packet 逐位元比較、serial／pipelined／兩個擾動 seed、lifecycle 與 `tracking_cuda_mot17`。
3. **部署路徑（HR FP16，MOT17-10）**：
   - `detect_sequence` 的 SCDR SHA-256 等於 §16 的 `98e2554f…`。
   - `motion_sequence` 的 SCMO 等於 M1 records 的 MOT17-10 SCMO。
   - `track_offline`（CPU reference）、`track_sequence` pipelined 與 `--serial` 的 MOT 檔逐位元相同，SHA-256 等於 §16 的 `1fb713f3…`。
   - serial 與 pipelined 的完整 packet stream 逐位元相同。
4. **Sanitizer**（compute-sanitizer，`--error-exitcode 99`）：
   - Kernel suites：`tracking_packet_tests`、`tracking_cuda_tests`、`motion_cuda_tests` 各跑 memcheck／initcheck／racecheck／synccheck，0 errors（racecheck 為 0 hazards），測試本身通過。
     - initcheck 不檢查 host API 複製（`--check-api-memory-access no`）：測試 harness 會把整個固定容量的 device 結構複製回 host，count 之後的部分依契約未定義，host 也不讀。kernel 的讀取照常全部檢查。
     - `motion_cuda_tests` 的 initcheck 加 `--synthetic`，跳過用 nvJPEG 解的兩張真實影像。nvJPEG 的既知報告（NVIDIA Bug 7021871）每次有數百萬筆，依 [PREPROCESS_USAGE_POLICY](PREPROCESS_USAGE_POLICY.md) §3 是診斷項而非門檻。motion kernels 由合成案例完整覆蓋；另外三種工具照常跑含真實影像的完整測試。
   - 正控制：`postprocess_sanitizer_control` 在四種工具下各自 exit 99，並出現該工具的預期診斷。
   - 完整 session memcheck（所有 kernel，含 nvJPEG、TensorRT、cuFFT）：T01 的 MOT17-10 前 30 幀、lifecycle 測試、HR FP16 `track_sequence` 的 MOT17-10 前 30 幀，皆 0 errors。
   - 完整 session racecheck／synccheck：T01 的 MOT17-10 前 10 幀，只檢查 tracking、motion 與 session control kernels（TensorRT、nvJPEG、cuFFT 不檢查），0 hazards／0 errors。
5. **Evidence report**：記錄 source、環境、每個命令的 exit 與時間、測試輸出、hash 與以上判定。

**不涵蓋**
- F03（TensorRT）kernels 的 racecheck／synccheck：沿用 S04 的狀態，仍未執行。
- 完整 session 的 initcheck：nvJPEG 的既知報告依上述政策處理，不在本驗收執行。
- 效能與追蹤品質：不設門檻。MOT17-02 的 GT 評估留給 M4。

**結果（2026-11-03）**：M3 正式驗收 **PASS**（e021bd1，乾淨 worktree，約 18 分鐘）。證據見 [evidence/tracking-session/2026-11-03-formal-e021bd1](../evidence/tracking-session/2026-11-03-formal-e021bd1/REPORT.md)。
- CPU CTest 16/16，CUDA CTest 59/59。7 序列 packet 與 CPU reference 逐位元相同。
- HR FP16 部署路徑的 SCDR 與 MOT 檔 hash 等於 §16；CPU reference、serial、pipelined 三者相同。
- Sanitizer 17 次執行皆 0 errors／0 hazards，4 個正控制都偵測到。

## 15. D1：HR 偵測器部署（2026-11-03 使用者核准）

部署 `saccade_v1_dev` 的 HR mixed Full25 e10（`hr-input/1`，1280×768 stretch；MOT17-10 val AP50 86.0093%）。runtime 新增固定的 `hr-1280x768` model profile，與既有 `t01-640x640` 並存；不做任意尺寸。產物與工具見 [training/hr_deploy](../training/hr_deploy/README.md)，檔案在 `~/.cache/saccade-v1/hr-deploy/`（含雜湊 manifest）。

**通過標準（在 native HR 結果產生前固定）**：以 native `detect_sequence --profile hr-1280x768` 錄下 MOT17-10-SDP，用 `training/hr_deploy/eval_scdr.py` 計算，AP50 必須落在 86.0093% ± 0.1 pp。參考尺度：T01 的 native 錄製與訓練時記錄只差 +0.0006 pp。MOT17-02 只在 M4 最終評估時開啟。

**結果（2026-11-03）**：
- 部署驗收 **PASS**：native HR 錄製的 MOT17-10 AP50 為 86.0092%，與訓練記錄 86.0093% 相差 −0.0001 pp。
- 權重身分：UA2 Teacher 的 primary EMA 權重，匯出 backbone 的特徵在 FP16 捨入後與訓練 cache 的 51,118,080 個元素逐位元相同。
- runtime：`hr-1280x768` profile 加入 S04 head、F03、producer、F04、DetectionSession 與 TrackingSession；640 profile 的既有測試（含 7 序列 session 對 CPU reference 的逐位元比較）全部照舊通過。
- Tracking（native `track_sequence --profile hr-1280x768`，pipelined，約 27 ms／1080p 幀）：

| 組別 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| val（MOT17-10），T01 → HR | 42.2 → **49.5** | 42.1 → 53.2 | 42.7 → 46.3 | 50.8 → 62.6 | 45.5 → 63.2 | 94 → 65 |
| in-domain（04／05／09／11／13），T01 → HR | 75.0 → 69.0 | 78.1 → 73.8 | 72.2 → 64.7 | 81.7 → 77.1 | 85.7 → 83.8 | 243 → 231 |

in-domain 下降是預期的：T01 只在這 5 個序列上訓練而過擬合；HR 另加 MOTSynth。它不是泛化指標。MOT17-02 的 HR 結果已產生但未評估，留給 M4。

## 16. D2：HR 部署改用 FP16（2026-11-03 使用者核准）

**為什麼改**
- v1 的時間幾乎全花在 FP32 卷積上（nsys：HR 每幀約 21 ms，佔 GPU 時間約 80%）。
- 完整模型的精度實驗見 [evidence/precision/2026-11-03](../evidence/precision/2026-11-03/README.md)。HR 的 FP16（optimization level 3）結果：
  - 模型 enqueue 時間：25.7 ms → 6.5 ms。
  - MOT17-10 AP50：與 FP32 差 −0.025 pp。
  - 原始輸出 relative L2：約 1%。
- BF16 在相同條件下較慢，誤差約為 FP16 的 7 倍，因此不採用。

**Build 模式**

`model_trt build` 新增 `--precision fp32|fp16`，預設 `fp32`。

| | fp32（預設，不變） | fp16 |
| --- | --- | --- |
| 用途 | 數值參考路徑；BACKBONE／HEAD／F03 的 FP32 比較政策照舊適用 | 部署用 |
| 網路類型 | strongly typed | weakly typed |
| Builder flag | — | `kFP16`；BF16、TF32、INT8 皆關閉 |
| Optimization level | 0 | 3 |
| Scan plugin | FP32 | 維持 FP32 |
| 模型輸入與六個原始輸出 | FP32 | FP32 |
| 其他 builder 設定 | workspace、aux streams、無 timing cache | 同左 |
| FP32 tactic 檢查（`head_tactic_violations`） | 適用 | 不適用 |

runtime（producer、F04、DetectionSession、TrackingSession）不分精度，照舊讀 FP32 I/O。

**通過標準（在正式 FP16 engine 產生前固定）**

正式 engine 由 `model_trt build --precision fp16 --profile hr-1280x768` 產生，不使用實驗 patch。要通過以下全部項目：

1. **Build report 讀回值**：weakly typed、fp16=on、bf16=off、tf32=off、int8=off、optimization level 3。Engine inspector 顯示：
   - 每個 scan／expand／merge plugin 的 payload 為 FP32。
   - 模型 I/O 為 FP32。
2. **AP50**：用 `detect_sequence --profile hr-1280x768` 錄下 MOT17-10-SDP，以 `training/hr_deploy/eval_scdr.py` 計算。AP50 必須落在 native FP32 的 86.0092% ± 0.1 pp 內。
3. **重跑一致**：同一 engine 跑兩次 `detect_sequence`，SCDR 的 SHA-256 必須相同。
4. **有限值**：在 `refs/model-images.f32` 的 4 張 HR 參考輸入上，六個輸出全為有限值，plugin statuses 為 PASS。

**通過後**
- 以 engine 的 SHA-256 固定部署。之後任何重新 build 都是新候選，要重做 1–4。
- 重跑 HR tracking（MOT17-10 與 in-domain 5 序列），記錄 HOTA／IDF1／MOTA 與 `track_sequence` 的每幀時間。這只是紀錄，不設品質門檻。
- MOT17-02 仍留到 M4。

**結果（2026-11-03）**

部署驗收 **PASS**。正式 engine 為 `~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine`：60,753,004 bytes，SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`。證據見 [evidence/precision/2026-11-03-formal](../evidence/precision/2026-11-03-formal/)。

| 項目 | 結果 |
| --- | --- |
| 1. 讀回值 | weakly typed、fp16=on、bf16／tf32／int8=off、optimization level 3。inspector：9 個 scan／expand／merge plugin 的 payload 為 Float、status 為 Int32；`images` 與六個輸出為 Float |
| 2. AP50 | **86.0156%**，比 FP32 的 86.0092% 高 0.0064 pp；recall@0.25 為 83.54%，FP@0.25 為 1,972 |
| 3. 重跑一致 | 兩次 `detect_sequence` 的 SCDR SHA-256 都是 `98e2554f…` |
| 4. 有限值 | 4 張參考輸入的輸出全為有限值，plugin statuses 為 PASS。同一 engine 重跑、CUDA graph 重播都逐位元相同 |

- **整幀覆蓋**：`detect_sequence` 的 producer 每幀都檢查所有輸出元素是否有限，並檢查 9 個 plugin status。654 幀全部通過。
- **FP32 路徑不受影響**：用新程式重新 build HR FP32，engine 位元組與原本不同（TensorRT plan 本來就不逐位元組重現，T01 的兩次 build 也不同）。但它在 4 張參考輸入上的六個輸出，與部署中的 `5120c747…` 逐位元相同，tactic 仍全為 FP32。

**速度**
- 模型本身：enqueue 6.42 ms，CUDA graph 5.09 ms（FP32 為 25.7／25.3 ms）。
- `track_sequence` pipelined 端到端：每幀 7.1–8.4 ms（FP32 約 27 ms），約 120–140 FPS。

**Tracking（`track_sequence --profile hr-1280x768`，FP32 → FP16）**

| 組別 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| val（MOT17-10） | 49.5 → **49.7** | 53.2 → 54.1 | 46.3 → 45.9 | 62.6 → 63.0 | 63.2 → 65.0 | 65 → 66 |
| in-domain（04／05／09／11／13） | 69.0 → 68.4 | 73.8 → 74.5 | 64.7 → 63.1 | 77.1 → 76.1 | 83.8 → 84.3 | 231 → 231 |

FP16 的偵測與 FP32 不完全相同，tracker 會放大這些差異，所以 MOT 檔不同；指標差異都在 1.6 以內。

部署 engine 自此改為 `model-hr-fp16.engine`。MOT17-02 仍留到 M4。

## 17. M4：MOT17 最終評估與 P01 收尾（在開啟 MOT17-02 前固定）

### 17.1 凍結的設定

- 部署路徑：§16 的 HR FP16 engine（`8292cc23…`）、`hr-1280x768`、`v1-motion/1` 與 `v1-tracker/1` 的 §3 預設值、`track_sequence` pipelined。本節提交後不再改任何參數。
- M1–M3 沒有調過 tracker 參數；§1 允許的 MOT17-10 調參不做，直接進 M4。
- MOT17-02 的 GT 只在 §17.3 的正式執行中讀一次。之前的開發與測試不得讀 `MOT17-02-SDP/gt/`。正式執行若在產生指標前因基礎設施中斷（當機、逾時），可用同一 commit 重跑；指標一旦產生就是最終結果。之後若再改偵測器、tracker 或參數，MOT17-02 的結果要標為「M4 之後」，不再算獨立 eval。

### 17.2 量測內容（不設品質門檻）

對 7 個 MOT17 SDP 序列，分 val（10）、eval（02）、in-domain（04／05／09／11／13）三組：

- **偵測**：`detect_sequence --profile hr-1280x768` 的 SCDR，以 `training/hr_deploy/eval_scdr.py` 算 AP50（每序列）。T01 的 native AP50 另以 `m1-data` 的 SCDR 算出，作為對照。T01 的 MOT17-02 已在 M1 評估過，不是新開啟。
- **追蹤**：`scripts/eval_mot17.sh` 的 HOTA／DetA／AssA／IDF1／MOTA／IDSW，三組各一列，另列每序列。
- **速度**：`track_sequence` 自己回報的 `ms_per_frame`（從第一張 JPEG 讀檔到最後一個封包收回，不含 engine 載入與 MOT 寫檔），每序列一個值，同時記下 GPU 型號與 driver。

### 17.3 正式執行

從同一個乾淨 commit，在 GPU lock 下依序執行：

1. **完整正式驗收**（§11 的「M4 結案時」那一次）：`scripts/run_tracking_session_acceptance.sh`，§14.5 的 1–5 項與預期 hash 全部不變，必須 PASS。
2. **評估**：`scripts/run_mot17_evaluation.sh` 執行與守門，`scripts/score_mot17_evaluation.sh` 只在守門 PASS 後讀 GT 計分（MOT17-02 最後）。只使用第 1 步 build 出的 CUDA binaries（先核對它們的 SHA-256 等於第 1 步的 `binaries.sha256`，且第 1 步的 revision 等於目前 HEAD）。以下守門項目都在讀任何 GT 之前完成，任一不符就停止，不讀 GT：
   - **身分鏈**：student checkpoint、teacher checkpoint、base 權重、backbone ONNX、head bundle 的 SHA-256 等於 `~/.cache/saccade-v1/hr-deploy/manifest.json` 的紀錄；build report（[evidence/precision/2026-11-03-formal/build-report.txt](../evidence/precision/2026-11-03-formal/build-report.txt)）的 `onnx_sha256`／`bundle_sha256`／`engine_sha256` 等於上述 ONNX、bundle 與部署 engine。
   - **CPU reference 對照**：每個序列的 `track_offline`（HR SCDR＋`m1-data` 的 SCMO）與 `track_sequence` 的 MOT 檔逐位元相同。
   - **與既有紀錄一致**：MOT17-10 的 SCDR 等於 `98e2554f…`、MOT 等於 `1fb713f3…`；5 個 in-domain 序列的 MOT 等於 [§16 的紀錄](../evidence/precision/2026-11-03-formal/tracking-mot.sha256)。
   - **Stream 位置**：以 `nsys` 追蹤 HR FP16 `track_sequence` 的 MOT17-10 前 30 幀，由 [`scripts/check_stream_placement.py`](../scripts/check_stream_placement.py) 判定。2026-11-03 M4 原驗收採單次 capture；**2026-11-04 CUDA Graph 改版後**，同一 binary、輸入與設定須分別執行 `--cuda-graph-trace=node` 與 `--cuda-graph-trace=graph` 兩次 capture，兩次輸出的 MOT 檔必須相同。從第一幀的 F02 kernel 起：
     - graph 外每個 kernel 歸入恰好一類。D：TensorRT engine（launch 在 `ExecutionContext::enqueueV3` NVTX range 內，含 scan plugin）、F02、F05、producer、F04；M：cuFFT、F06、tracking、stage control、commit、packet。
     - D 是 graph 外 F02 kernel 的 stream，M 是 graph 外 tracking kernel（predict）的 stream，兩者不同。兩次 capture 各自辨識 D／M，不假定兩次的 stream IDs 相同。
     - node capture 的 graph kernels 位於 profiler 合成 streams，且 replay 的 engine kernels 不在 `enqueueV3` NVTX range 內，故 graph 內改按內容分類：每張 graph 只能含 D 類或 M 類；D graph 內無法由名稱歸類的 kernel 才視為 engine，M graph 的所有 kernels 均須符合 M 名稱規則。
     - graph capture 記錄 `cudaGraphLaunch` 的實際 stream。兩次 capture 的 launches 依 host 呼叫順序逐一配對，數量相同、graph IDs 一對一；D graph 每次均須 launch 於該 capture 的 D，M graph 均須 launch 於 M。這依賴兩次執行 launch 順序相同，由相同 binary／輸入／設定與相同 MOT 輸出支持。
     - nvJPEG HARDWARE backend 的 `rgba2rgb` 跑在 nvJPEG 自己的 stream 上，不在 D。這就是 §14.1 那個順序問題的來源，session 以 host 等待 `rgb_read_done` 處理；這裡只記錄它的 stream。除了 nvJPEG，不得有 graph 外 kernel 在 D、M 之外；graph 內則核對其實際 launch stream。

     這對應 [ASYNC_CONTROL](ASYNC_CONTROL.md) 列入 P01／P05 的兩項。程式本身已保證這兩項：session 拒絕 aux streams ≠ 0 的 engine（`getNbAuxStreams() == 0`），cuFFT plan 每幀以 `cufftSetStream` 綁到 M。這一項是執行期的觀測確認。

     （修訂：本項原寫「nvJPEG 在 D」。開發 trace 顯示 `rgba2rgb` 在另一個 stream，因此在正式執行前改為只記錄 nvJPEG。這不影響 MOT17-02 的封存。）

     CUDA Graph 規則的受測結果與 8 個負控制見 [clean-source 報告](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) 及 [placement-checker](../evidence/cuda-graph/2026-11-04-gate-d21a144/placement-checker/results.tsv)。這是 stream 守門的後續修訂，不改寫 §17.5 的 M4 原始 capture 或重新計分。

   守門通過後才計算 §17.2 的指標。
3. **Evidence report**：記錄 source、環境、每個命令的 exit 與時間、hash、守門結果與全部指標。

### 17.4 P01 判定標準

P01（[DATAFLOW](DATAFLOW.md) §9.2：模型檔案的來源／版本、engine 精度／shape binding、可 native 載入的權重與 component I/O）以部署中的 HR FP16 模型判定。§17.3 全部通過時，P01 記為 **CLOSED（有範圍）**，各項證據：

| 項目 | 證據 |
| --- | --- |
| 來源／版本 | HR deploy manifest（`saccade-hr-deploy/1`）與 §17.3 身分鏈；訓練 split 固定 eval 為 MOT17-02 |
| 權重 native 交付與 component I/O | D1：head bundle 由 native converter 產生；backbone 匯出特徵與訓練 cache 逐位元相同；native AP50 與訓練記錄差 −0.0001 pp（§15） |
| Engine 精度與 shape binding | D2 build report 讀回值與 inspector（§16 第 1 項） |
| Engine 與 runtime 的身分一致 | §17.3 的身分鏈、M3 正式驗收的 SCDR／MOT hash |
| 函式庫的 stream 行為 | §17.3 的 stream 位置 |
| 模型品質 | §17.2 的 AP50 與追蹤指標（只記錄） |

**不涵蓋（明列）**
- TensorRT 內部 kernel 的 racecheck／synccheck：沿用 S04 起的決定，未執行（插樁成本極高，S04 試跑期間發生過原因未判定的 WSL 當機）。v1 自己的 scan plugin、producer、F04 與 tracking kernel 已有四種 sanitizer 的證據；TensorRT kernel 有完整 session 的 memcheck。
- 完整 session 的 initcheck：nvJPEG 的既知報告（NVIDIA Bug 7021871），依 PREPROCESS_USAGE_POLICY 處理。
- FP16 engine 是在 761889c 加上未提交的修改（之後提交為 04912e2）下 build 的（`build_dirty=true`）；部署身分以 engine SHA-256 固定，不重新 build。
- v1 的 native manifest 工具（schema 1–3）不描述 HR profile 或 FP16；HR 的身分紀錄是上述 deploy manifest，不另加 schema。
- 跨 GPU、driver 或 TensorRT 版本的支援（P07）。

### 17.5 結果（2026-11-03）

M4 正式執行 **PASS**（cad5298，乾淨 worktree）。證據見 [evidence/m4/2026-11-03-formal-cad5298](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md)。
- 完整正式驗收：CPU CTest 16/16，CUDA CTest 59/59；部署路徑 hash 等於 §16；17 次 sanitizer 執行皆 0 errors／0 hazards，4 個正控制都偵測到。
- 守門：身分鏈 39 個檔案一致；7 序列的 `track_offline` 與 `track_sequence` 逐位元相同；MOT17-10 與 in-domain 的 hash 等於 §16；stream 位置 PASS。TensorRT（含 scan plugin）、F02／F05／producer／F04 在 D，cuFFT／F06／tracking／session control／packet 在 M，nvJPEG 的 `rgba2rgb` 在它自己的 stream。
- MOT17-02 的 GT 只在計分時讀了一次。

| 組別 | HR AP50（T01） | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | ms／幀 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val（MOT17-10） | 86.02%（67.59%） | 49.7 | 54.1 | 45.9 | 63.0 | 65.0 | 66 | 8.46 |
| **eval（MOT17-02）** | **72.91%**（54.40%） | **40.6** | 40.9 | 40.7 | 49.3 | 46.8 | 103 | 8.51 |
| in-domain（04／05／09／11／13） | 92.7–97.4%（94.5–98.9%） | 68.4 | 74.5 | 63.1 | 76.1 | 84.3 | 231 | 7.2–8.3 |

- MOT17-02 對 T01（§13）：HOTA 31.6 → 40.6，DetA 23.7 → 40.9，AssA 42.2 → 40.7，IDF1 37.2 → 49.3，MOTA 23.5 → 46.8，IDSW 31 → 103。提升全部來自偵測，關聯沒有改善。
- val 與 eval 的 AP50 差距：HR 13.1 pp，T01 13.2 pp。
- 速度：`track_sequence` pipelined 每幀 7.2–8.5 ms，約 118–138 FPS（單一筆電 GPU、WSL2，沒有控制 clock）。

P01 依 §17.4 記為 **CLOSED（有範圍）**，不涵蓋項目照 §17.4 所列。之後若再改偵測器、tracker 或參數，MOT17-02 的結果要標為「M4 之後」。

## 18. 排序優化（2026-11-04）

F04b 在 device 驗證完整 pool 的 score/ID 順序；已有序時穩定壓實 suppressed，否則用完整 64-bit score/ID key 做 stable block radix。batch ID 排序對 2..32 rows 保留 rank，大於 32 用 stable radix；birth 對已排序 batch 直接遍歷，public 未排序輸入仍排序 index permutation。FP32 signed zero、完整 uint32 ID、原 index tie、finite checks 與 deferred error gates 均保持原語義。ABI、scratch allocation 及 tracker profile 不變。

固定 HR replay 的 F04b 組裝 kernel 平均 143.57→102.81 µs（−28.39%），但完整路徑單幀平均僅 7.037→6.978 ms，p95 7.383→7.369 ms，吞吐幾乎不變；不宣稱穩定端到端收益。詳細範圍、控制案例、CPU bit 比較與 sanitizer 結果見 [排序開發報告](../evidence/tracking-sort/2026-11-04/REPORT.md)。這些是開發來源證據，不改寫 §17 的正式驗收與品質結果。

## 19. CUDA Graph／排序改版正式驗收（2026-11-04）

[Clean-source 正式報告](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) 已補完 Graph、tracking 排序與相關資源排程修改的驗收：步驟 1–3 使用乾淨 `4329ed7`，步驟 4–5 使用乾淨 `d21a144`。CPU 16/16、F04 5/5、mutations 15/15、完整 CUDA CTest 59/59 與報告所列 sanitizer／正控制全部通過；七序列的 T01 CPU／ordinary serial／graph pipelined 完整 packets 相同，HR 路徑與 §17.3 更新後的 stream 守門亦通過。

HR FP16 七序列的 7 個 SCDR、7 個 MOT、7 個 offline MOT 與 M4 **21/21 逐位元相同**；未讀 GT、未重新計分。原 [CUDA Graph 測時](../evidence/cuda-graph/2026-11-04/REPORT.md)、§18 排序測時與 [資源排程報告](../evidence/resource-schedule/2026-11-04/REPORT.md) 保留開發來源身分；本次不改寫其失敗嘗試，也不將有限 sanitizer 結果延伸為完整 HR／TensorRT 內部驗收。仍未涵蓋項目見 §17.4 與 [STATUS](STATUS.md#未涵蓋與未完成邊界)。

## 20. v1-motion/2：OFA 全域光流 GMC 整合（2026-11-04 規格，實作前固定）

### 20.1 決定與依據

TrackingSession 的 F05／F06 由 `v1-motion/1`（§4–§5 的 PCR）改為 `v1-motion/2`，也就是光流實驗的 G1 臂：NVOFA 密集光流，取未遮罩 cells 的全域中位數。輸出仍是單一 `(tx, ty)`，所以 F07、F08a、F08b、CUDA tracker 與 ResultPacket wire format 都不變。

依據是兩輪預先登記的實驗，見 [PLAN](../evidence/gmc-flow/PLAN.md)、[REVIEW](../evidence/gmc-flow/REVIEW.md)、[PLAN-2](../evidence/gmc-flow/PLAN-2.md)、[REVIEW-2](../evidence/gmc-flow/REVIEW-2.md)。G1 相對 PCR 的 HOTA：MOT17-10 +1.19、in-domain +0.51、PersonPath22 移動 +0.62、固定 +0.07、MOT17-02 −0.06。局部光流臂不採用。

本節取代 §4–§5，以及 DATAFLOW／ASYNC_CONTROL 中 MI／SPEC／MotionState 的相關列；那些文件的改寫留到文件整理階段。PCR 程式（`MotionCuda`、CPU reference、`motion_sequence` 及其測試）保留為獨立工具，TrackingSession 不再使用。§17.3 的 M4 腳本屬於 `v1-motion/1` 時期的紀錄。

### 20.2 參數（`v1-motion/2`，全部固定）

| 欄位 | 值 | 意義 |
| --- | --- | --- |
| `flow.reference_width`／`_height` | 1920／1080 | 決定縮小倍率 s |
| `flow.grid` | 4 | NVOF output grid |
| `flow.perf` | SLOW | NVOF perf level |
| `flow.cell` | 8 | cell 邊長（OFA 解析度的像素） |
| `flow.mask_score` | 0.45 | 遮罩用候選的最低 score（kept、class 0） |
| `flow.min_cells` | 8 | 未遮罩 cells 少於此數即 rejected |

- 倍率：`s = max(1, ceil(W/1920), ceil(H/1080))`，OFA 影像為 `w = floor(W/s)`、`h = floor(H/s)`。w 或 h < 32 時，session 建立時拒絕。
- MOT17、PersonPath22 的 1080p 與 1280×718 序列都是 s = 1，與實驗完全相同；3840×2160 為 s = 2。
- 選 SLOW 的理由：OFA 是獨立引擎，F06 在 M 上與同一幀 D 的 F03（約 5 ms）重疊，SLOW 的 2.0 ms 預期能藏在其中。§20.8 第 8 項驗證這一點；不成立時停下回報，不自行改 perf。

### 20.3 F05（D stream）

輸入為 F01 的 planar RGB8，輸出灰階 MotionImage，寫入 MI slot（NVOF 可用的 buffer）。

- s = 1：與 `flow_sequence` 相同的算式、運算順序與編譯選項（`--fmad=false` 等），共用同一個 device 函式。輸出須與實驗的灰階逐位元相同。
- s > 1：對 s×s 區塊內每個像素的 `0.299R + 0.587G + 0.114B` 依列優先累加（dy 外層、dx 內層），除以 s²，再做 `uint8(min(255, floor(v + 0.5)))`。
- 發布 `mi_ready(k)` 的規則照舊；nvJPEG 前 host 等 `rgb_read_done` 的規則照舊。

### 20.4 MotionState 與遮罩擷取

- **MotionState**：每來源／epoch 一組，由 CTL 的 `committed_slot` 選擇，與 TS 由同一次 commit 發布。內容只有兩樣：
  - `gray[2]`：上一個 committed 幀的 OFA 解析度灰階；
  - `mask[2]`：該幀的 cell 遮罩（每 cell 一個 byte）。
- **遮罩擷取（M，第 k 幀）**：在 `M wait CS ready` 之後、F04b／tracking graph 之前，由 run_stage 守護的 kernel 讀第 k 幀候選，寫入 working slot 的 mask。
  - 取 `kDetectionKept`、class 0、`score ≥ 0.45` 的 rows。
  - cell (c, r) 的中心換回原圖座標為 `(s·(8c+4), s·(8r+4))`；落在任一框的 `[x1, x2) × [y1, y2)` 內就標記。
  - 這一步必須在 `candidate.consumed(k)` 之前完成。
- 第 k 幀 commit 後，`gray[committed]` 與 `mask[committed]` 就是第 k+1 幀 F06 的參考。

### 20.5 F06（M stream），第 k 幀

1. **首幀**（`committed_frame = −1`）：不呼叫 OFA。`status = uninitialized`，t = 0。
2. **輸入搬移**：守護的 kernel 把 `gray[committed]` 複製到固定的 OFA input buffer。OFA 只讀固定 buffer 與 MI(k)，不需要 host 知道 committed_slot。
3. **OFA**：`NvOFExecute(input = 第 k−1 幀灰階, reference = MI(k))`。設定依 §20.2，不用 hints、cost 或 ROI；IO streams 都設為 M。OFA 只寫自己固定的 output buffer，所以和 cuFFT 一樣不受 run_stage 守護也無害。
4. **Cells**：沿用實驗的 `reduce_cell`（2×2 個 grid-4 向量的逐分量中位數，單位 OFA 像素）。
5. **全域中位數**：只計 `mask[committed]` 未標記的 cells，逐分量取中位數（偶數取中間兩個的平均）。
   - 值都是 1/64 的整數倍，GPU 實作須與 CPU reference 逐位元相同。
   - 少於 8 個 cells：`status = rejected`，t = 0。
   - 否則 `status = valid`，`tx = s·median_x`、`ty = s·median_y`。
6. **MotionContext**：`peak_x = peak_y = 0`、`pcr = 0`；valid 時 `confidence = 1`，其他狀態為 0。
7. **輸出搬移與釋放**：守護的 kernel 把 MI(k) 複製到 `gray[working]`，然後記錄 `mi_consumed(k)`。

CPU reference 為實驗的 `gmc_flow::FrameControl`（G1），輸入是 OFA 原始 grid 輸出或 SCFL cells 加上候選。

### 20.6 OFA 的前置審查（整合前，未通過即停）

比照 Bug 7038860 的 [nvJPEG 重現方法](../evidence/preprocess/2026-11-03-nvjpeg-stream-order/README.md)，以獨立程式在 IO stream = 同一個非預設 stream 的條件下檢查以下各項。每項 20 次，前置 spin 0.2 ms 與 20 ms 各做一組，另加一組先同步的對照：

- **A1 讀取順序**：前方排 spin，再排「覆寫 input buffer」的 kernel，然後 Execute。若輸出的光流對應覆寫前的內容，就是違規。
- **A2 寫後讀**：Execute 後立刻排「覆寫 input buffer」的 kernel。若輸出受到覆寫內容影響，就是違規。
- **A3 輸出完成**：Execute 後立刻排 kernel 複製 output。若與同步後讀到的不同，就是違規。
- **A4 決定性**：同一對影像重跑，output 逐位元相同；`flow_sequence` 重跑 MOT17-10，SCFL 與實驗的檔案逐位元相同。
- **A5 host 阻塞**：量測 `NvOFExecute` 呼叫在 host 上的耗時（中位數／p95）。

A1–A4 任一違規即停止並回報，不自行加 host 同步或其他繞路，由審閱者另訂處置。A5 只記錄，影響由 §20.8 第 8 項判定。

### 20.7 4K 輸入與工具

- **4K 解碼容量**：`TrackingSession`、`detect_sequence`、`track_sequence` 的 `PreprocessConfig` 容量改為 `max(1920, W) × max(1080, H)`，W、H 為序列的宣告尺寸。1080p 以下的序列容量不變。
- **`flow_sequence`**：支援 §20.2 的 s。s = 1 的輸出不變。
- **`gmc_records SCDR SCFL OUT.scmo`**：新工具，以 CPU reference 產生 `v1-motion/2` 的 MotionContext 串流。沿用 SCMO 格式，key 欄位為 `(W, H, s, w, h)`。
- **新 records**：
  - `../saccade_v1-worktrees/m1-data-v2/`：T01 SCDR（與 `m1-data` 相同）＋ `v1-motion/2` SCMO。CTest 的 `SACCADE_TRACKING_RECORDS` 改指此處。
  - HR：M4 SCDR 搭配實驗的 SCFL 產生的 SCMO。

### 20.8 驗收（capture 前固定）

`scripts/run_tracking_session_acceptance.sh` 更新為本節版本，在 GPU lock 下一次跑完，證據放在 `evidence/tracking-session/<日期>-v1-motion-2/`。任一項不過即停止，不放寬比較。

1. **Clean source**：同 §14.5 第 1 項。另外核對 NVOF 載入路徑與 headers 的 SHA-256。
2. **Build 與測試**：全新 CPU build 與全新 CUDA build（所有功能開啟）的 CTest 全部通過。CUDA 部分至少包含：
   - 7 序列 T01 的 session packet 與 CPU reference（`m1-data-v2`）逐位元相同，serial、pipelined、兩個擾動 seed 都比；
   - F05 灰階（s = 1 對 `flow_sequence`；s = 2 對 CPU 算式）、cells、遮罩、中位數的 CUDA 對 CPU 逐位元比較；
   - §20.6 的 A1–A4；
   - lifecycle。
3. **MOT17 HR 部署路徑（7 序列）**：
   - SCDR 與 M4 逐位元相同；
   - session 的 MotionContext 串流與 `gmc_records`（M4 SCDR＋實驗 SCFL）相同；
   - `track_sequence` pipelined、`--serial` 與 `track_offline` 的 MOT 三者逐位元相同，而且等於實驗 G1 臂的 MOT（`../saccade_v1-worktrees/gmc-flow/build/gmc-flow/arms/G1/`）。
4. **PersonPath22 回歸**：
   - 實驗納入的 69 個序列，`track_sequence` 的 MOT 等於第二輪 G1 臂的 MOT（`../saccade_v1-worktrees/gmc-flow/build/gmc-flow2/pp22/arms/G1/`）；
   - 5 個 4K 序列須能完整跑完，計分只記錄。
5. **計分**：MOT17 val／in-domain／MOT17-02 與 PP-moving／PP-static／PP-all 重新計分。數值須等於實驗的 G1（由第 3、4 項的 bytes 推得，此處確認）。4K 另列。
6. **Sanitizer**：沿用 §14.5 第 4 項，motion suite 換成新的 F05／遮罩／cells／中位數／搬移 kernels。
   - 完整 session memcheck：HR MOT17-10 前 30 幀與 lifecycle。
   - racecheck／synccheck：T01 MOT17-10 前 10 幀，只檢查 v1 kernels（OFA、TensorRT、nvJPEG 內部不檢查）。
   - 正控制照舊。
   - 完整 session memcheck 的 exit 99（2026-11-05，capture 前補）：NVOF 內部的 `cuArrayGetPlane` 失敗無法由呼叫端避免（每個 session 在 `nvOFInit` 74 次、第一次 `nvOFExecute` 12 次；HR 86 次、lifecycle 1008 次），已回報 [Bug 7048273](https://developer.nvidia.com/bugs/7048273)，證據見 `evidence/nvof-getplane/2026-11-05/`。此項記為 OPEN，不算通過。其他 API 或 device 錯誤、次數不符、未全數列出，都算失敗。
7. **Stream 位置**：
   - `check_stream_placement.py` 更新：F05 kernels 在 D；遮罩、輸入／輸出搬移、OFA、cells、中位數在 M。
   - NVOF 函式庫發出的 kernels（`Convert_PL2BL`、`Convert_BL2PL`、`Gaussian_Subsample2x2_Luma_BL2BL_v2`）計為 OFA，必須在 M（2026-11-05，capture 前補）。
   - nsys 顯示 OFA 執行與同一幀 D 的 F03 時間重疊。
8. **吞吐**：無 profiler 的 HR pipelined `track_sequence`，7 序列各跑 3 次。以總幀數÷總時間計算 ms/frame，與同一次執行中 main `23ade53` 的同樣量測比較：增加不得超過 3%。serial 單幀延遲只記錄。
9. **Evidence report**：同 §14.5 第 5 項，另附 §20.6 的結果。

**不涵蓋**：OFA 內部的 sanitizer、MEDIUM／FAST 的品質、局部光流臂。

**結果（2026-11-05）**：**未完全通過**（fae5cb7，乾淨 worktree，約 41 分鐘）。第 6 步完整 session memcheck 的 exit 99 記為 OPEN（[Bug 7048273](https://developer.nvidia.com/bugs/7048273)），其餘八步通過；吞吐總計 +2.70%。證據見 [evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2](../evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md)。
