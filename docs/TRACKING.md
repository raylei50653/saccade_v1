# Tracking 半段規格（F05–F09）

日期：2026-11-03。狀態：M0 契約與 M1（CPU tracker 端到端）完成；M2 起依 §12 接續。上層文件：[DATAFLOW](DATAFLOW.md)、[ASYNC_CONTROL](ASYNC_CONTROL.md)、[ResultPacket](RESULT_PACKET.md)。

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
| M4 | MOT17 全序列評估與 P01 收尾。 |

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
- `perturb_seed ≠ 0`：在 D、M 各 stage 邊界插入長度由 seed 決定的 spin kernel（V03），只供測試。
- 工具 `track_sequence ENGINE SEQ_DIR OUT.txt [--serial] [--packets OUT.bin] [--epoch N]`：TrackingSession → F09 SequenceOutput → MOT 檔；`--packets` 存下完整 raw packet stream。

### 14.4 驗收（M3 實作階段）

- device packet writer 對 `encode_packet`：隨機邏輯封包（含 FAILED／SKIPPED、負零、滿 256 rows）逐位元相同。
- 同一序列 serial 與 pipelined 的 packet stream 逐位元相同；兩個 `perturb_seed` 的 pipelined 也相同（A10／V04）。
- 與 CPU 對照：TrackingSession 的 packet stream 等於以 `m1-data` 的 SCDR／SCMO 跑 `TrackerReference` 再 `encode_packet` 的結果；`track_sequence` 的 MOT 檔與 `track_offline` 的逐位元相同。前提是 F03 與 cuFFT 在同一裝置與 build 下重跑決定性；若不相同，先定位是哪個輸入不同，不放寬比較。
- 失敗：producer flag 注入使第 k 幀 FAILED、之後 SKIPPED，committed 狀態凍結；EOF／cancel／D-only tail 都能 drain；close 可重複呼叫。
- 新 CUDA 程式碼 memcheck（短序列）。完整正式驗收（四種 sanitizer、clean source、evidence report）在 M3 合併後另做。

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
