# 光流 GMC 第二輪計畫：地面先驗與依聚集程度加權

計畫日期：2026-11-04。基線：branch `exp/gmc-flow`（第一輪 [PLAN](PLAN.md)、[REPORT](2026-11-04/REPORT.md)、[REVIEW](REVIEW.md)）。本文件在任何第二輪臂於 PersonPath22 計分前提交，§6 的規則自此固定；結果寫在 `evidence/gmc-flow/<執行日期>-round2/REPORT.md`，本文件不回改。

## 1. 問題

第一輪：G1（全域光流中位數）與 L2（腳下帶中位數）都通過登記規則，但 L2 在 MOT17-02 掉約 5 HOTA。[REVIEW §2](REVIEW.md) 的診斷如下：

- 近鏡頭物體位於較遠 track 的腳下帶，常沒被偵測到，或框只蓋住一部分。
- 腳下帶分布有四型：A 單峰、B 視差梯度、C 地面＋遮擋物、D 遮擋物佔滿。中位數對 C 型會選到多數的遮擋物。
- 在原始光流空間依聚集程度選點：固定相機時有效；移動相機時會選到剛體遮擋物（§2.3）。

本輪問三件事：

- **Q1**：G1 對 G0 的增益，在偵測器與參數都沒看過的資料上是否成立？
- **Q2**：以逐列穩健擬合的地面模型在腳點取值（R），能否保留 L2 的視差收益，同時不受遮擋物影響？
- **Q3**：在扣掉地面先驗後的殘差空間依聚集程度加權（RC），能否再勝過 R？

第一輪後的設計曾在 MOT17-02 與 MOT17-10 上診斷，所以 MOT17 不再是獨立證據。本輪判定改以 PersonPath22 為主。

## 2. 評估資料

### 2.1 PersonPath22（主要，未用過）

- 來源：`/home/ray/developer/ai/saccade/datasets/PersonPath22/mot_test_kf/`，共 74 個序列、51,769 幀，原始幀率（24–60 fps）。解析度：68 個 1920×1080、5 個 3840×2160、1 個 1280×718。
- GT 只在約 5 fps 的關鍵幀上（`gt/gt.txt` 的 frame 編號依原始幀率）。
- HR 偵測器的訓練資料是 yolo26m（COCO）、MOT17 04/05/09/11/13、CrowdHuman、MOTSynth，不含 PersonPath22（`saccade_v1_dev/build-hr-mixed/HR01-MS-FULL25-20261103/source/training/hr/recipe.json`）。
- 評分：tracker 跑全幀率，MOT 輸出只保留 GT 檔中出現過的 frame，再用 TrackEval（與 `eval_mot17.sh` 相同的設定與指標）計分。舊專案的教訓：不可用 5 fps 抽幀版 `mot_test_full` 評追蹤。
- 新增 `scripts/eval_pp22.sh`，不修改 `eval_mot17.sh`。

### 2.2 相機運動分層（GT-free，先固定）

對每幀計算 `m_k`：未遮罩 cells 的 `|d|` 中位數，再乘 `1920/W` 換算到 1080p 寬度（第 0 幀除外）。序列的 `m` 為 `m_k` 的中位數。

- `m ≥ 0.5` px → **PP-moving**；否則 → **PP-static**。
- 分層只用光流與偵測遮罩，不讀 GT，在任何臂計分前算好並寫入 evidence。

### 2.3 MOT17（次要）

沿用第一輪的輸入與 SCFL。報告 MOT17-10 val、in-domain，以及 MOT17-02（已用於診斷，只報告）。

### 2.4 PersonPath22 的 GPU 輸入

每個序列在 `flock` 下各跑一次：

- `detect_sequence`：HR FP16 engine `~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine`（SHA256 `8292cc23…`），呼叫方式照 `scripts/run_mot17_evaluation.sh`，產出 SCDR。
- `motion_sequence`：`v1-motion/1`，產出 SCMO（G0 用）。
- `flow_sequence --perf slow`：產出 SCFL。

任一工具在某序列失敗時，該序列從所有臂排除並列出原因。失敗超過 5 個序列即停止並回報。

## 3. 臂

共用定義沿用第一輪 PLAN §4：光流 SCFL 第 k 幀（k−1 → k、k−1 座標），8 px cells；遮罩來自第 k−1 幀 kept、class 0、score ≥ 0.45 的候選；track box 為 committed state；`N_min = 8`。

| 臂 | 內容 | 角色 |
| --- | --- | --- |
| OFF | 無 GMC | 參照 |
| G0 | PCR（`v1-motion/1`） | 現行 |
| G1 | 第一輪 G1 | 現任候選 |
| L2 | 第一輪 L2 | 第一輪選擇 |
| LD | 腳下帶 cells 在原始光流空間的最高密度點 | 診斷：直接以聚集程度選點，不作採用候選 |
| R | 地面先驗在腳點的值 | Q2 |
| RC | 地面先驗＋殘差聚集加權 | Q3 |

### 3.1 地面先驗（逐列模型）

- 第 r 列的模型使用 r−1、r、r+1 三列（截在畫面內）中所有未遮罩的 cells。座標 `u = x_c − W/2`，其中 `x_c = 8c+4`。
- dx 與 dy 分別擬合 `d = a + b·u`，用 Tukey biweight IRLS：
  - 初值 `a = median(d)`、`b = 0`，固定 10 次迭代；
  - 每次以 `s = max(1.4826·MAD(r), 0.25)` 為尺度，`c = 4.685`；
  - 權重 `w = (1 − (r/(c·s))²)²`（|r| < c·s），否則為 0；
  - 以 double 解 2×2 加權最小平方。權重和 < 2 或行列式為 0 時停止迭代，保留上一組 (a, b)。
- 三列中未遮罩 cells 少於 16 個時，該列模型「不可用」。
- 在 (x, r) 求值：`a_r + b_r·(clamp(x, 0, W−1) − W/2)`。

### 3.2 R

- 腳點為 `(cx, y2)`，腳點列 `r_f = clamp(floor(y2/8), 0, rows−1)`。
- 依序找 r_f、r_f−1、r_f+1、r_f−2、r_f+2、…，取 ±4 列內第一個可用的列模型，在 cx 求值。
- 都不可用時用 G1。

### 3.3 LD（診斷）

- 腳下帶 cells 與 L2 相同；少於 `N_min` 時用 G1。
- 空間權重 `ω_c = exp(−((x_c − cx)/(0.75w))²/2)`。
- 每個 cell 的密度 `ρ_i = Σ_j ω_j·exp(−|d_i − d_j|²/(2h²))`，h = 1.0 px。
- 從 ρ 最大的 cell 出發（同值取 cell 序最前者），做加權 mean-shift（核寬 h、權重 ω，最多 50 次，步長 < 0.01 px 即停）。收斂點即為 control。

### 3.4 RC

1. 腳下帶 cells 與 L2 相同，排除所在列模型不可用的 cells。剩餘少於 `N_min` 時，control = R。
2. 殘差：`e_c = d_c −（c 所在列的模型在 x_c 的值）`。空間權重 ω 同 LD。
3. 從 μ = 0（地面先驗）出發，在殘差空間做加權 mean-shift，核寬 h = 1.0 px，最多 50 次，步長 < 0.01 px 或核權重和 < 1e−12 即停。
4. 聚集程度：`D = Σ ω_c·exp(−|e_c − μ|²/(2h²)) / Σ ω_c`（0–1，同時反映佔比與集中程度）。
5. 權重：`α = clamp((D − 0.2)/0.4, 0, 1)`。
6. control = `R + α·μ`。

意義：從先驗出發只會走到最近的一群，所以遠處的遮擋物群不會被選中。最近那一群越集中、佔比越高，越信任局部資料；分散或稀少時就退回地面先驗。

常數（三列、16 cells、±4 列、Tukey c 與尺度下限、h、0.75w、D 的 0.2／0.6、`N_min`）全部固定，本輪不掃參數。這些常數只在 MOT17-02／10 的 12 條 track 上檢查過行為（REVIEW §2.2–§2.3），沒有在任何資料集上依分數挑選。

## 4. 關卡

- **C0**：重建後，G0、G1、L2 在 MOT17 7 序列的 MOT 與第一輪 `build/gmc-flow` 的輸出逐位元相同。
- **C1**：PersonPath22 上，harness 的 G0（經 hook）與 stock `track_offline`（同一 SCDR＋SCMO）的 MOT 逐位元相同，全部序列。
- **C2**：單元測試全部通過，至少涵蓋：
  - Tukey 擬合在 < 50% 外點時還原真實直線；
  - 列模型的可用性與鄰列搜尋；
  - mean-shift 收斂到已知 mode；
  - α 的兩端與斜坡；
  - LD 的起點選擇。
- **C3**：§2.2 的分層在計分前寫入 evidence。

任一關卡不過即停止並回報。

## 5. 量測

- 每臂：PP-moving、PP-static、PP-all 的合併指標（HOTA、DetA、AssA、IDF1、MOTA、IDSW）；MOT17-10、in-domain、MOT17-02 同第一輪。
- PP-moving 上逐序列 HOTA，以及各臂相對 G1 的勝／負序列數。
- R／RC 的診斷：
  - R 退回 G1 的比例、鄰列搜尋的使用比例；
  - RC 的 α 分布（0、(0,1)、1 三段）；
  - RC 與 L2、R 的 control 差距分布。
- 固定相機（PP-static 與 MOT17 02／04／09）上每臂 control 的平均值與 p99 大小。

## 6. 預先登記的判定

**規則 A（G1 在新資料上是否成立；用於 G1 整合決策）**：PP-moving 上 `HOTA(G1) ≥ HOTA(G0)` 且 `AssA(G1) ≥ AssA(G0)`，並且 PP-static 上 `HOTA(G1) ≥ HOTA(G0) − 0.2`。

**規則 B（局部臂）**：候選 X ∈ {L2, R, RC}，相對 G1 同時滿足以下三項才 GO：

1. PP-moving：`HOTA(X) − HOTA(G1) ≥ +0.5`，且 `AssA(X) ≥ AssA(G1)`。
2. PP-static：`HOTA(X) ≥ HOTA(G1) − 0.2`。
3. MOT17-10：`HOTA(X) ≥ HOTA(G1)`。

多個通過時取 PP-moving HOTA 最高者；差距 < 0.2 時依 R、RC、L2 的順序優先。LD 只用來檢驗 REVIEW §2.3 的判讀，不作候選。PP-moving 在 +0.2 到 +0.5 之間的臂只報告，不追加變體。

## 7. 範圍外

整合進 TrackingSession、CUDA tracker、改動 tracker 或 Kalman 參數、以 PersonPath22 調任何常數、`docs/`。

## 8. 產出

`evidence/gmc-flow/<執行日期>-round2/` 存放：

- `REPORT.md`；
- 分層表；
- `inputs.sha256`（PP22 SCDR／SCMO／SCFL、MOT17 SCFL、engine、binaries）；
- C0–C2 的紀錄；
- 每臂、每集合的 TrackEval summary；
- 逐序列表；
- §5 的診斷 CSV。

SCDR／SCMO／SCFL／MOT 只留在 build 目錄。
