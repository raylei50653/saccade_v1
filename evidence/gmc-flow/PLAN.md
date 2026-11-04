# 光流 GMC 實驗計畫（exp/gmc-flow）

計畫日期：2026-11-04。基線：main `23ade53`。本文件在任何光流臂計分前提交，§6 的判定規則自此固定；結果寫在 `evidence/gmc-flow/<執行日期>/REPORT.md`，本文件不回改。

## 1. 問題

現行 `v1-motion/1`（[TRACKING §4–§5](../../docs/TRACKING.md)）在 d=4 縮小、Hanning window 的灰階小影像上做 phase correlation，只取整數峰值：`t = 4·peak·confidence`。因此全域平移以 4 px 為量化單位，且全畫面只有一個平移。HR 部署下 GMC 是追蹤裡最大的單項貢獻之一（T01 時 MOT17-10 HOTA +7.4）。

舊專案（只取設計意圖與教訓，不重做）已否證兩條路：

- LK 稀疏光流＋RANSAC 擬 affine（BoT-SORT 式）：IDF1 −0.8，MOT17-10 −4.6。
- NO-GO #40 tile-PCR similarity：−0.25，3 勝 4 負。根因是走動相機的背景橫跨多個深度，背景本身就有 ~1.15 px 的視差差異；**單一 2D warp 的額外自由度只會放大這個 misfit**。純平移安全，是因為殘差交給每條 track 的 Kalman 速度吸收。
- 另有 #20（PCR 前景遮罩無效）、#34／#49（GMC 共模已準，殘差主要是逐軌的、時變的）。

所以本實驗**不再測全域 affine／homography**，只問兩件事：

- **H1（精度）**：硬體密集光流給出次像素、穩健的全域平移，是否勝過 4 px 量化的 PCR？
- **H2（視差）**：每條 track 只取它附近背景的光流當 control，等於取「該 track 深度上的相機運動」。這直接對準 #40 的根因，而且不需要 3D。它是否勝過單一全域平移？

另附可行性問題 **F**：NVOFA 能否在本機 WSL2 實際 `Init`／`Execute`，延遲多少。硬體盤點只驗到 session 與 `GetCaps`（`~/reports/rtx5070ti-laptop-hardware-audit-2026-11-04.md` §6）。

## 2. 固定輸入

| 項目 | 來源 | 身分 |
| --- | --- | --- |
| 偵測 | `../saccade_v1-worktrees/m4-formal/build-m4-evaluation/scdr/<SEQ>.scdr`（M4 HR FP16，7 序列） | MOT17-10 SHA256 `98e2554f…`；其後 Graph、F04、F02 改版的 21 個輸出均與 M4 相同 |
| 基線 motion | `../saccade_v1-worktrees/m1-data/<SEQ>.scmo`（`v1-motion/1`） | M4 offline 使用的同一份 |
| 影像 | `/home/ray/developer/ai/saccade/datasets/MOT17/train/<SEQ>/img1` | 7 個 SDP 序列，共 5,316 幀 |
| tracker | `v1-tracker/1` 預設值，完全不改 | |
| 計分 | `scripts/eval_mot17.sh`（legacy TrackEval） | |

M4 基線分數（`track_sequence` 輸出；M4 已證明與 `track_offline` 逐位元相同）：

| 集合 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val MOT17-10 | 49.676 | 54.056 | 45.894 | 62.962 | 65.044 | 66 |
| in-domain 04/05/09/11/13 | 68.423 | 74.454 | 63.148 | 76.136 | 84.284 | 231 |
| MOT17-02（M4 之後，不計入判定） | 40.625 | 40.853 | 40.723 | 49.29 | 46.752 | 103 |

in-domain 逐序列 HOTA：04 74.802、05 57.999、09 57.724、11 65.821、13 54.361。其中 05、10、11、13 是移動相機；02、04、09 為固定相機，光流臂在這三個序列上應接近 0 位移，主要檢查有無噪聲注入。

## 3. 光流擷取（GPU，每序列一次）

工具 `flow_sequence SEQ_DIR OUT.scfl [--perf slow|medium|fast] [--timing OUT.csv]`：

1. 解碼：與 `motion_sequence` 相同，用 `NativePreprocessor`（nvJPEG HARDWARE）得到 planar RGB8。
2. 灰階：`g = uint8(min(255, floor(0.299R + 0.587G + 0.114B + 0.5)))`，float 計算，原解析度，不縮小。
3. NVOF API 2.0：標頭從 `~/reports/rtx5070ti-laptop-hardware-evidence-2026-11-04/nvof/` 複製到 `third_party/nvof/`（SHA256 `a83f6045…`、`07defc79…`，保留 BSD 授權註記與來源 commit `edb50da3…`）。`dlopen("libnvidia-opticalflow.so.1")`，並在工具輸出中印出實際載入的路徑（預期 `/usr/lib/wsl/lib/…`）。CUDA primary context，`NvOFSetIOCudaStreams` 設成工具自己的 stream。
4. 初始化：`NV_OF_MODE_OPTICALFLOW`、`outGridSize = 4`、`perfLevel` 依參數（品質資料一律 SLOW）、不用 hints、不開 cost、不開 ROI。
5. 第 k 幀（k ≥ 1）：**input = 第 k−1 幀，reference = 第 k 幀**，所以向量落在 k−1 幀的格點上，與 committed track box 同一座標系。儲存的位移 `d` 採 MotionContext 的方向：內容從 k−1 到 k 往右／下移時 `d > 0`。NVOF 原始向量與 `d` 的符號關係由 §3.1 的合成測試確定，工具存的是 `d`。
6. Cell 化：cell 邊長 8 px，`cols = floor(W/8)`、`rows = floor(H/8)`。每個 cell 含 2×2 個 grid-4 向量；cell 值為逐分量中位數（4 個值排序後取第 2、3 個的平均），單位 px（S10.5 ÷ 32）。
7. SCFL 格式（little-endian）：`"SCFL"`、u32 version=1、i32 W、H、cell、cols、rows、perf、u64 count；每幀 i64 frame、u32 status（0 = uninitialized，1 = valid）、`rows·cols·2` 個 f32（dx, dy 交錯，列優先）。第 0 幀為 uninitialized，cells 全 0。

### 3.1 正確性與可行性關卡

- **F1**：1920×1080 與 640×480 在 SLOW、grid 4 下 `Init`＋`Execute` 成功。失敗即停止，回報確切狀態碼；不改走 Windows、OpenCV 或軟體光流。
- **F2（合成測試，CTest）**：確定性紋理影像，以 bilinear 重取樣平移 `(+8, +4)`、`(−5, +3)`、`(+2.5, −1.5)` px，兩種解析度。全部 cell 的逐分量中位數須在真值 ±0.25 px 內，且符號符合第 5 點。失敗即停止並回報。
- CPU 單元測試：cell 中位數化簡、SCFL 讀寫來回。

### 3.2 延遲

MOT17-10 全序列，SLOW／MEDIUM／FAST 各一次：`Execute` 的 GPU 時間（IO stream 上的 CUDA events）中位數／p95／最大值，以及灰階＋cell kernel 時間。只記錄，不設門檻；與整條 pipeline 的重疊與爭用不在本實驗範圍。

## 4. GMC 臂（CPU，offline）

單一 harness `gmc_flow_offline`：讀 SCDR＋SCMO／SCFL，以 `TrackerReference` 加一個**實驗用 per-track control hook** 執行，輸出 MOT 檔與診斷。

Hook 規則：在 `predict_tracks` 中，若 hook 已設定且 `motion.status == valid`，對每條 track 以 hook 回傳的 `(dx, dy)` 取代全域 `(tx, ty)`；hook 未設定時行為與現在逐位元相同。hook 只存在 CPU reference，CUDA tracker 與 TrackingSession 不動。

共用定義（第 k 幀，k ≥ 1）：

- 光流：SCFL 第 k 幀（k−1 → k，k−1 座標）。
- track box：committed state（k−1 之後）以 `math::state_to_box` 取得 `(x1, y1, x2, y2)`，`w = x2 − x1`、`h = y2 − y1`、`(cx, cy)` 為中心。
- 遮罩 M：SCDR 第 k−1 幀中 kept、class 0、`score ≥ 0.45`（`assoc.high_thresh`）的候選。cell 中心 `(8c+4, 8r+4)` 落在任一遮罩框內（左閉右開）即排除。
- 中位數：逐分量；奇數取中間，偶數取中間兩個的平均。
- `N_min = 8`。

| 臂 | 第 i 條 track 的 control | 角色 |
| --- | --- | --- |
| OFF | 無（GMC disabled） | 參照 |
| G0 | SCMO 的 PCR `(tx, ty)`（現行 main） | 基線 |
| G1 | 全部未遮罩 cells 的中位數；少於 `N_min` 時本幀 rejected（t=0） | H1 |
| G1q | `4·round_half_away_from_zero(G1/4)`，逐分量 | 診斷：分離「次像素」與「估計器」兩個因素，不作採用候選 |
| L1 ring | cell 中心落在 `[cx−1.5w, cx+1.5w) × [cy−0.75h, cy+0.75h)`、不在遮罩、也不在本 track 自己的 box 內的 cells 之中位數；少於 `N_min` 時用 G1 | H2 |
| L2 foot band | cell 中心落在 `[cx−1.5w, cx+1.5w) × [y2, y2+0.3h)`、不在遮罩、不在自己 box 內的 cells 之中位數；少於 `N_min` 時用 G1 | H2（腳下地面與人同深度） |

G1 也套遮罩，所以 L 對 G1 的差別只剩「局部」一個因素。退回 G1 是 L 臂定義的一部分，不是錯誤處理。

上表常數（cell 8、遮罩分數 0.45、ring／band 幾何、`N_min = 8`、SLOW）全部固定。本輪不掃參數。

## 5. 量測

### 5.1 正確性關卡（計分前）

- **B0**：stock `track_offline`（M4 SCDR＋m1 SCMO）在 7 序列上的 MOT 與 `m4-formal/build-m4-evaluation/offline/*.txt` 逐位元相同；val 與 in-domain 分數重現 §2。
- **B1**：harness 的 G0 臂（經 hook）與 B0 的 MOT 逐位元相同，7 序列。
- **B2**：G1、G1q 另外寫成等價 SCMO（status valid、tx／ty 為該臂值、confidence 1、peak 與 pcr 為 0，key 抄自 m1 SCMO），stock `track_offline` 的 MOT 須與 harness 經 hook 的結果逐位元相同，7 序列。

任一關卡不過即停止並回報。

### 5.2 追蹤指標

6 個臂 × 7 序列。以 `eval_mot17.sh` 分別計分：val（MOT17-10）、in-domain 合併與逐序列、MOT17-02（標「M4 之後」，不計入判定）。指標 HOTA、DetA、AssA、IDF1、MOTA、IDSW。

### 5.3 GT 殘差（機制診斷，不計入判定）

以 harness 中與追蹤相同的 control 函式，在 GT box 上求值，不另寫第二份實作。GT 取 `consider = 1`、class 1、visibility ≥ 0.5，同一 id 在連續 k−2、k−1、k 都存在者；GT frame f 對應 SCDR frame f−1。遮罩用 SCDR 第 k−1 幀；L 臂的「自己的 box」為該 GT box。

- `e1_k = (c_k − c_{k−1}) − t_k(box_{k−1})`
- `e2_k = e1_k − e1_{k−1}`：扣掉行人等速運動，近似 constant-velocity Kalman 實際看到的誤差。主診斷。

每臂、每序列報 `‖e2‖/h_{k−1}` 與 `‖e1‖/h_{k−1}` 的中位數與 p90，以及 L 臂的退回次數比例。

### 5.4 Motion 統計

每序列：`|G1 − G0|` 的中位數／p95／超過 4 px 的幀數；G0 rejected 幀數；固定相機序列（02、04、09）上 G1 的平均 `|t|`。

## 6. 預先登記的判定

候選臂 X ∈ {G1, L1, L2}，相對 G0 同時滿足以下三項才算 **GO（進入整合規格）**：

1. MOT17-10：`HOTA(X) − HOTA(G0) ≥ +0.5`，且 `AssA(X) ≥ AssA(G0)`。
2. in-domain 合併：`HOTA(X) ≥ HOTA(G0) − 0.2`。
3. in-domain 每個序列：HOTA 下降不超過 0.5。

多個臂通過時取 MOT17-10 HOTA 最高者；差距 < 0.2 時優先 G1（較簡單）。只有 L 臂通過而 G1 沒過，表示 H2 成立而 H1 不成立。G1q 只用來解讀 G1。MOT17-02 只報告。

沒有臂通過即 **NO-GO**：記錄結果與 §5.3／§5.4 的機制解讀後結案。MOT17-10 在 +0.2 到 +0.5 之間的臂只報告，是否開第二輪由審閱者決定，本輪不追加變體。

## 7. 範圍外

整合進 TrackingSession（取代 F05／F06、stream 排序、CUDA Graph、比照 Bug 7038860 審查 OFA 的 stream 順序）、CUDA tracker 對等、全域 affine／homography、Kalman 或 tracker 參數、以光流做逐物件關聯、前景分割、`docs/` 修改。GO 之後的整合另訂規格。

## 8. 產出

`evidence/gmc-flow/<執行日期>/`：`REPORT.md`；`inputs.sha256`（SCDR、SCMO、SCFL、headers、binaries）；B0–B2 的比對紀錄；每臂、每集合的 TrackEval summary；GT 殘差與 motion 統計 CSV；延遲 CSV；環境（driver、CUDA、NVOF 載入路徑）。MOT 檔與 SCFL 只留在 build 目錄，evidence 只記 SHA256。
