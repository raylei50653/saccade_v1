# 原版 YOLO26m head 的 scan 契約

查核日期：2026-10-29。v1 模型基線依使用者決策改為 **YOLO26m backbone／neck＋原版 m T3→T1 對應的 Mamba-FPN head**，640×640、單幀 T=1。本文件固定該 head 的 scan 算子與 Cross-Scan 契約，作為 S01 scalar reference 的依據；重跑命令與原始 captures 見 [本輪證據](../evidence/s01/2026-10-29/REPORT.md)。

**狀態：** `saccade::SelectiveScan` 與四方向 Cross-Scan 的數學、tensor 邊界、狀態生命週期已由實作核實；S01 reference 與 fixtures 依 §5 驗收（結果見 [MODEL_FEASIBILITY §3](MODEL_FEASIBILITY.md#3-s01s04-的接續條件)）。S02／S03 已接上 CUDA 與 TensorRT plugin；**2026-10-30 S04 PASS**，以指定的 T1 候選 checkpoint 完成 head 權重 native 交付與完整 FP32 TensorRT head 的數值、執行及失敗傳遞驗證；正式 clean-source 結果與範圍見 [S04 證據](../evidence/s04/2026-10-30/REPORT.md)。完整 head 的 racecheck／synccheck 未執行，限制見 [HEAD_TRT §9](HEAD_TRT.md#9-執行紀錄與待完成事項)。P01 仍 OPEN：matching backbone／neck、前處理與 decode 未交付（§7、[readiness §8](MODEL_ARTIFACT_READINESS.md#8-2026-10-30-s04head-權重的-native-交付與完整-head)）。前一版的 s／D1 調查保留為歷史，見 §8。

狀態用語：**已核實**＝下列來源的實作或實際檔案如此記載；**v1 決策**＝來源未規定或互不一致，v1 依來源可支援的交集訂定；**缺件**＝本輪無法核實。

## 1. 目標與來源

舊庫 `raylei50653/saccade`（公開，Apache-2.0，`LICENSE` blob `d6456956`）。source 固定於 [TRAINING_WORKFLOW](TRAINING_WORKFLOW.md) 的整理快照 `e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2`；本輪只讀 Git objects 與本機 artifacts，未執行舊程式、Python 或載入 checkpoint 類別。

| ID | 來源與固定身分 | 用途 |
| --- | --- | --- |
| H | [mamba_head.py][head]，blob `ddd4524c`，SHA-256 `b0a11387…21167cc` | `MambaBlock`、`_cross_scan_mamba`、`_selective_scan_jit`、head 前向 |
| K | [mamba_scan.cu][kernel]／[.cuh][kernel-h]，blob `47673aba`／`8f51a9a9` | 訓練與 TensorRT 實際執行的 FP32／FP16 forward kernel |
| P | [mamba_scan_plugin.cpp][plugin]／[.hpp][plugin-h]，blob `cd2d4e06`／`21806834` | TensorRT `SelectiveScan` plugin 身分與 shape 推導 |
| X | [export_mamba_head_onnx.py][export]，blob `40c441cd` | ONNX 節點、opset、I/O 名稱 |
| T | 本機 T1 主線 `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt`，126,802,369 bytes，SHA-256 `a5a0e709…59c0676c` | 以 native pickle 容器讀取：epoch 15、`mamba_args`、state_dict 形狀與 frozen A_log |
| O | 本機 `models/yolo/mamba_head_26m.onnx`，40,768,852 bytes，SHA-256 `7330db0c…19405b167` | native protobuf dump：IR 8、opset `""` 17／`saccade` 1、494 nodes、65 initializers、0 functions |
| R | 訓練期 `09eb1747`（2026-06-14，T1 args 中 `legacy_n1_scan` 等鍵首次出現） | 與 `e48f3b39` 比對 scan 相關定義 |

**來源一致性（本輪新增）：** T 的 `mamba_args` 為 `in_channels=[256,512,512]`、`d_model=128`、`d_state=16`、`num_blocks=1`、`spatial_reduction=4`、`use_cross_scan=true`、`use_pixel_shuffle=true`、`per_channel_a=false`、`legacy_n1_scan=false`、`scan_stop_grad=true`、`use_temporal_mamba=true`；缺 `reg_max`／`head_depth`／`reduction_variant` 鍵，由 state_dict 形狀確認為 1／1／`conv`。三個 `A_log` 都是 `(1,16)`、值為 ln 1…ln 16（SSM 內部凍結於初值），`x_proj.weight` 為 `(33,256)`（C rank 1），`upsample.*` 存在。R→H 間 `_selective_scan*`、`MambaBlock`、`_cross_scan_mamba` 只多出 N 驗證；head 前向只多出預設關閉的 `_bypass_reduction`；K 的差異為 launch 形狀、N 驗證與 D skip 移入同一迴圈（FP32 結果相同）。O 的 65 個 initializer **全部**逐 byte 對上 `…_gpu_decode/best_recall.ckpt`（epoch 11，MatMul 權重以轉置比對），對 T1 主線只對上 17 個凍結參數；兩者 `mamba_args` 結構相同。因此 O 證明的是**同一架構**的部署圖，不是 T1 主線權重（§7）。

## 2. Head module map 與 tensor 邊界

每層 i（P3／P4／P5）各自一組權重；輸入 `p3`／`p4`／`p5` 為 FP32 NCHW `[batch,256,80,80]`／`[batch,512,40,40]`／`[batch,512,20,20]`（O graph inputs，batch 為動態維；v1 固定 batch 1）。

| 步驟 | 運算（O node／H locator） | 形狀（batch 1） |
| --- | --- | --- |
| 1 | `input_proj.i` Conv 1×1＋bias → 128（O 5；[H 1910][head-fwd]） | (1,128,H,W) |
| 2 | `downsample.i` Conv 4×4 stride 4＋bias（O 6；H 1917） | (1,128,H/4,W/4)＝20²／10²／5² |
| 3 | Cross-Scan 展開，directions 疊在 batch 前（O 26–66；[H 588–607][cross]） | (4,L,128)，L＝400／100／25 |
| 4a | `in_proj` Linear 128→512、無 bias；x＝[:256]，z＝[256:]（O 70–80；[H 478, 513–515][block]） | (4,L,512) |
| 4b | depthwise causal Conv1d k=4、兩側 pad 3、取前 L、＋bias，再 SiLU → **u**（O 81–91；H 480–486, 518–520） | (4,L,256) |
| 4c | `x_proj` Linear 256→33、無 bias → dt[:16]、B[16:32]、C[32:33]（O 92, 97, 104, 109；H 492, 523–526） | (4,L,16)／(4,L,16)／(4,L,1) |
| 4d | `dt_proj` Linear 16→256＋bias → **delta**（raw；delta bias 即 `dt_proj.bias`）（O 98–99；H 493, 524） | (4,L,256) |
| 4e | **A＝−exp(A_log)**，圖內計算，三層共用同一 initializer（O 110–111；H 528） | (1,16) |
| 4f | **`saccade::SelectiveScan`(u, delta, A, B, C, D)**（O 112／274／436；H 531） | (4,L,256) |
| 4g | gate：y·SiLU(z)；`out_proj` Linear 256→128、無 bias；殘差＋步驟 3 的 tokens（O 113–117；H 540–541, 609） | (4,L,128) |
| 5 | Cross-Scan 合併（O 118–158；H 611–620） | (1,128,H/4,W/4) |
| 6 | `upsample.i` Conv 3×3 128→2048＋bias，PixelShuffle 4（O DepthToSpace `CRD`, blocksize 4；[H 1525–1535, 1936][upsample]） | (1,128,H,W) |
| 7 | concat [步驟 1, 步驟 6] → cls：Conv 3×3 256→128＋SiLU＋Conv 1×1→80；reg：同結構→4（O 161–169；[H 1477–1498, 2052–2055][det]） | cls (1,80,H,W)，reg (1,4,H,W) |

T 另含 `temporal_blocks`／`flow_gate_conv`，T=1 時繞過，O 中沒有對應節點；v1 不實作。

## 3. SelectiveScan 算子契約

| 項目 | 契約 | 狀態與來源 |
| --- | --- | --- |
| 身分 | ONNX domain `saccade`、op type `SelectiveScan`、custom opset 1；節點有 STRING attribute `domain="saccade"`（exporter 以 `domain_s` 產生），無其他 attribute。TensorRT：IPluginV3 name `SelectiveScan`、version `1`、namespace `saccade`；creator 只解析 INT32 欄位 `has_D`、`a_per_channel`、`B`、`L`、`D`、`N`，實際值於 configure／shape change 由輸入形狀推導。 | 已核實：O 112、[X 110–140][export-sym]、[P hpp 8–35][plugin-h]、[P 42–60, 141–157, 260–305][plugin] |
| 輸入順序 | 0 u (B,L,Dc)；1 delta (B,L,Dc)；2 A (rows,N)；3 B (B,L,N)；4 C (B,L,cw)；5 D (Dc)。輸出 0 y (B,L,Dc)。m：B=4（四方向×batch 1）、L=400／100／25、Dc=256、N=16、rows=1、cw=1、D 存在。 | 已核實：O 112 inputs、value_info；P 接受 5–7 輸入、`has_D = nbInputs ≥ 6`。hpp 註解的第 7 個 params 輸入未由 O 使用。 |
| dtype／layout | dense row-major、TensorFormat LINEAR；plugin 接受 FLOAT 或 HALF，全部同 input 0，輸出同 input 0。O 為 FLOAT。索引：u/delta/y `(b·L+t)·Dc+d`，B `(b·L+t)·N+n`，A `r·N+n`。 | 已核實：[K 31–89][kernel-fwd]、P 89–108。部署 engine 是否以 HALF 執行為缺件（§7）。 |
| broadcast | A：rows=1 時所有 channel 共用；rows=Dc 時每 channel 一列。C：cw=1 時同一值用於全部 N（m 的 spatial block）；cw=N 時逐 state。D：每 channel 一值。 | 已核實：H 254, 258–261；P 154, 180。**v1 決策**：rows∉{1,Dc}、cw∉{1,N} 拒絕（H 與 P 對非法值的處理不同，§7）。 |
| 合法範圍與容量 | batch、Dc ≥ 1；L ≥ 0；N ∈ {1,2,4,8,16,32}；每個 tensor 的元素數與部分乘積 ≤ INT32_MAX（kernel 與 plugin 以 int32 索引）。 | 已核實：N 限制見 [H 29–33][validate]、K 22–24, 254；**v1 決策**：容量以 int32 為上限並先於資料檢查。 |
| 精確數學 | 對每個 (b,d)：Δ_t＝softplus(delta)，softplus(x)＝x（x>20）否則 ln(1+eˣ)；h₋₁[n]＝0；h_t[n]＝exp(Δ_t·A[r,n])·h_{t−1}[n]＋(Δ_t·B[b,t,n])·u[b,t,d]；y[b,t,d]＝Σ_n C[b,t,c(n)]·h_t[n]＋D[d]·u[b,t,d]（有 D 時）。 | 已核實：[H 279–304][jit]、[K 16–20, 64–87][kernel-fwd]；兩者相同。A 在圖外已是 −exp(A_log)，op 內沒有再取 exp(A_log)。 |
| 角色 | A：負的連續時間 decay（m 中凍結為 −1…−16）；B：輸入矩陣；C：輸出讀出；D：逐 channel skip（op 內）；delta bias：只在上游 `dt_proj` 中；softplus：op 內、先於與 A／B 相乘；gate SiLU(z) 與殘差：op 外。 | 已核實：§2 步驟 4d–4g。外層殘差與 U-Net concat 不是 D skip。 |
| 狀態生命週期 | 每次呼叫、每個 (b,d) 從零開始；沒有 initial-state 輸入、不輸出 final state；四方向、三層、各幀之間不保留狀態。L=0 時輸出空 tensor。 | 已核實：H 295、K 56；T=1 部署（[TRAINING_WORKFLOW §3](TRAINING_WORKFLOW.md#3-共用模型與資料規格)）。**不支援並明列拒絕**：非零 initial state、final state 輸出、跨呼叫狀態（ABI 中沒有對應欄位）。 |
| 非有限值 | **v1 決策**：任何輸入 NaN／±Inf 拒絕；decay、drive、state、C·state、D·u、output 任一量級超過 FP32 最大有限值即拒絕。 | 來源不檢查（K 直接傳播）。v1 不讓非有限值進入 F04；拒絕不輸出部分結果。 |

## 4. Cross-Scan 次序與合併

輸入 x 為 (batch,C,H,W)。方向 k 的 token t 讀取 k 翻轉後地圖的 row-major 位置：0 原圖、1 翻轉 H 與 W、2 翻轉 W、3 翻轉 H（H 598–601；O 26／31／36 的 step −1 Slice，Concat 45 依此順序）。四個方向疊成 (4,batch,C,H,W)，再 reshape 為 (4·batch,C,L)、transpose 為 (4·batch,L,C)，所以序列列號＝方向×batch＋影像（H 606–607；O 57–66）。

每個方向以**同一組** MambaBlock 權重處理，殘差後（H 609）依原翻轉還原，四方向 `mean`：O 為 Gather 0／1／2／3 → 對 1 翻 (H,W)、2 翻 W、3 翻 H → Concat → ReduceMean axes=[0]（O 130–158；H 611–620）。翻轉皆為 involution，所以還原時位置 p 在方向 k 的 token 即 source(k,p)。reference 以 ((y0+y1)+y2)+y3)/4 計算；四項加總順序屬實作細節，不影響契約。

## 5. 數值比較政策

在首次執行前固定，並與受測 source 一同提交；SDK affine 小網路的 exact 結果與此無關。

| 項目 | 固定內容 |
| --- | --- |
| reference 計算 | 輸入 FP32 精確轉為 binary64，全部中間量與輸出為 binary64；`std::exp`／`std::log1p`（glibc libm）。次序固定：Δ；n 由小到大：decay、drive＝(Δ·B)·u、h＝decay·h＋drive、acc＋＝C·h；最後 acc＋＝D·u。 |
| build 選項 | GCC／Clang `-ffp-contract=off`，不使用 `-ffast-math`；CMake Release 與 ASan／UBSan build 均須通過。 |
| expected | 只來自 [fixtures](../fixtures/selective-scan/README.md)：bc 任意精度 closed form／明確和（截至 40 位小數，以 long double 解析）、手算，或契約等式；reference 不自產唯一 expected。 |
| metric | 逐元素 \|a−e\| ≤ atol＋rtol·\|e\|；報告 max abs、max rel、最差索引。 |
| `unit` 容差 | atol＝rtol＝1e−10，用於項量級 S ≤ 約 2.5×10² 的案例。依據：binary64 前向誤差界 ≈ c·(L+N)·u·S，u＝2⁻⁵³，c≈22（exp／log1p 各 ≤1 ulp 及 \|Δ·A\| ≤ 16 ln2 的引數放大）；各案例 (L+N, S) 代入後 ≤ 1.5e−11，留 6 倍以上餘裕。 |
| `relative` 容差 | atol＝0、rtol＝1e−10，用於量級極端但無相消的案例（相對界 ≈ 1.5e−14）。 |
| 等式 | broadcast、共用 A、batch 隔離、重設、缺 D、零長度為值相等。 |
| 門檻點 | softplus(20) 另以 1e−13 直接檢查 log1p 分支；20.5 為 passthrough。 |
| 適用範圍 | 本政策只驗 reference 是否實現契約。S02 的 FP32 CUDA 比較政策另訂於 [SCAN_CUDA §5](SCAN_CUDA.md#5-數值比較政策首次執行前固定)（逐元素 running error bound，不沿用本節 1e−10）；FP16 I/O 誤差屬 S03。 |

## 6. Head 的 ordered cls／reg 與 decode

| 邊界 | 已核實內容 | 狀態 |
| --- | --- | --- |
| 輸出順序 | `cls_p3`, `cls_p4`, `cls_p5`, `reg_p3`, `reg_p4`, `reg_p5`；FP32 NCHW `[batch,80,H,W]`／`[batch,4,H,W]`，raw logits 與 raw distances（圖內無 sigmoid）。 | 已核實：O outputs、[X 229–258][export-onnx] |
| 類別 | 80 類（COCO 定義由 teacher／base 繼承）；訓練時 MOT17 pedestrian 映射 class 0。 | 已核實：T `num_classes=80`、TRAINING_WORKFLOW §3；類別名單本身未由 head 交付 |
| box 編碼 | reg_max=1、無 DFL；ltrb 以 cell 為單位，anchor＝(x+0.5, y+0.5)，xywh 後乘 stride 8／16／32；score＝sigmoid 後取 max class。三層依 P3→P5、row-major 展平串接，共 8400 raw anchors。 | 已核實：[mamba_gated_detector.py 117–195][decode]（同 revision）。屬舊庫 Python 後處理，不是 head 圖的一部分 |
| decode 責任與容量 | head 圖不含 decode／NMS。v1 由 F04 負責；候選上限、padding、invalid mask、class tie 規則未定。 | 缺件：P02／F04；S04 的驗收邊界止於六個 raw outputs，不含 decode |

## 7. 來源差異、衝突與缺件

下表的舊 engine 與 checkpoint 調查保留 S01 查核時的來源事實。S04 已交付指定 T1 候選的 native head bundle 與 v1 自建 head engine；matching backbone／neck 仍未驗收，更新狀態見 [readiness §8](MODEL_ARTIFACT_READINESS.md#8-2026-10-30-s04head-權重的-native-交付與完整-head)。

| 項目 | 具體內容與處理 | 影響階段 |
| --- | --- | --- |
| softplus 精度 | K：x>20 原值、x<−20 為 `expf(x)`、其餘 `logf(1+expf(x))`（FP32）；H JIT：`F.softplus` 門檻 20、`log1p(exp)`。數學相同，K 在 (−20,−8) 有 FP32 相消誤差。reference 採 log1p 形式。 | S02 已處理：v1 kernel 用 `log1pf(expf(x))`，x < −80 另以縮放的 e^x 避免下溢（[SCAN_CUDA §3](SCAN_CUDA.md#3-fp32-算法s02-基準)） |
| reduction 次序 | K 為 warp xor tree，JIT 為 `sum(-1)`，reference 依 n 遞增。 | S02 已處理：v1 kernel 為固定 butterfly，誤差界見 SCAN_CUDA §5 |
| A rows 判定 | H：`rows == Dc` 才 per-channel，否則讀第 0 列；P：`rows > 1` 即 per-channel，rows<Dc 會越界。合法值 {1,Dc} 下一致，非法值 v1 拒絕。 | 已處理（v1 決策） |
| C width | H：`expand` 只允許 1；P：只有 1 才 broadcast，其他值會讀越界。合法 {1,N}，其餘拒絕。 | 已處理（v1 決策） |
| D skip 時點 | 訓練期 K（R）在第二個迴圈對已存的 y 加 D·u；H 版同一迴圈。FP32 相同、FP16 會多一次 half 捨入。 | S02 為同一迴圈（FP32）；FP16 屬 S03 |
| 部署精度 | 現行 `build_mamba_head_trt.py` 預設開 FP16 與 TF32，且 plugin 接受 HALF；7 月 engine 的實際 build revision 與 flags 未記錄。 | 缺件：P01／S03 |
| checkpoint ↔ 部署 | preset 指定 T1 主線 T，但 head engine 的 sibling ONNX O 來自 `…_gpu_decode/best_recall.ckpt`（X 的預設值，65/65 initializer 相符）。engine `c0c0e3e1…` bytes 本身仍未歸屬。 | 缺件：P01 |
| 訓練 source 身分 | checkpoint 未記錄 source revision；T 的 args 鍵指向 ≥ R 或含相同變更的 dirty tree。scan 定義在 R 與 H 間語義相同，但這不是逐位元 provenance。 | 記錄於 evidence；不阻塞 S01 |
| native 權重 | S01 查核時 T1 主線只有 pickle checkpoint，O 屬另一個 checkpoint。S04 已 native 讀取指定 T1 候選並交付 C++ bundle；matching backbone／neck 僅查舊 ONNX ordered I/O，其權重歸屬、engine 與推論未驗。新 run 尚未啟動。 | head 工程候選已交付；P01 仍 OPEN（[readiness §8](MODEL_ARTIFACT_READINESS.md#8-2026-10-30-s04head-權重的-native-交付與完整-head)） |

## 8. 歷史：s／D1 查核（2026-10-29 先前一輪）

前一輪以當時的 s 基線查核 D1（舊 math_model §4.1 的 down4／two-block／up4 設計意圖，128／256／512 channels）、同名 MambaFPN 論文（U1）與 MambaPSA（U2），結論是三者都不是目標 head 的實作，S01 因缺 scan 公式而 OPEN，並提出 D01–D04 待裁定。該版全文保留於 [`2aef5f8` 的 SCAN_CONTRACT](https://github.com/raylei50653/saccade_v1/blob/2aef5f8feaec6126f814cd1ed1cabf85f8994c9b/docs/SCAN_CONTRACT.md)，證據見 [scan-contract/2026-10-29](../evidence/scan-contract/2026-10-29/REPORT.md)。

本輪的 m 決策取代該輪的目標：D1 的 two-block lane（`num_blocks=2`）與原版 m 的 `num_blocks=1` 不同，D01–D04 不再是 v1 待裁定項；U1／U2 未被採用。D1 的 cls80／reg4／8400 anchors／decode 意圖與 §6 一致，但 §6 以 m 實作為準。

[head]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py
[validate]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L29-L33
[jit]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L279-L304
[block]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L453-L543
[cross]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L588-L620
[det]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L1477-L1498
[upsample]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L1524-L1535
[head-fwd]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_head.py#L1905-L2055
[kernel]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/tracking/mamba_scan.cu
[kernel-fwd]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/tracking/mamba_scan.cu#L16-L89
[kernel-h]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/include/tracking/mamba_scan.cuh
[plugin]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/tracking/mamba_scan_plugin.cpp
[plugin-h]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/include/tracking/mamba_scan_plugin.hpp#L8-L35
[export]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/scripts/model/export_mamba_head_onnx.py
[export-sym]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/scripts/model/export_mamba_head_onnx.py#L110-L140
[export-onnx]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/scripts/model/export_mamba_head_onnx.py#L229-L258
[decode]: https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/src/saccade/perception/temporal_yolo/mamba_gated_detector.py#L117-L195
