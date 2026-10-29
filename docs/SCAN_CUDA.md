# S02：SelectiveScan／Cross-Scan 的 CUDA FP32 forward

日期：2026-10-29。基線：`60d722cfa45bb6d9763740cf6c8103c9defa9c0d`（S01 PASS 與其證據）。本文件固定 S02 的呼叫契約、FP32 算法、支援範圍與數值比較政策。**§3–§5 在第一次 GPU 執行前提交**；之後若驗收失敗，修正實作，或在本文件明列政策修訂及理由，不在失敗後默默放寬門檻。

數學契約不變：仍是 [SCAN_CONTRACT](SCAN_CONTRACT.md) §3–§4 的 `saccade::SelectiveScan` 與四方向 Cross-Scan，以 S01 的 binary64 [`selective_scan_reference`](../include/saccade/selective_scan.hpp) 為主要對照。S02 不含 FP16 I/O、TensorRT plugin（S03）、MambaBlock 的投影／conv1d／gate 與完整 head（S04）。

## 1. 交付與範圍

| 項目 | 內容 |
| --- | --- |
| 算子 | `selective_scan_cuda`：六輸入 u／delta／A／B／C／D 的順序、layout、broadcast 與 S01 相同；N ∈ {1,2,4,8,16,32}；A rows ∈ {1, Dc}；C width ∈ {1, N}；D 有／無；每次呼叫零狀態；L=0 輸出空 tensor。 |
| Cross-Scan | `cross_scan_expand_cuda`、`cross_scan_merge_cuda`：方向 0 原圖、1 翻 H 與 W、2 翻 W、3 翻 H；序列列號＝方向×batch＋影像；合併依 ((y0+y1)+y2)+y3 再取平均。 |
| m 的必要形狀 | P3：(4, 400, 256, 16)；P4：(4, 100, 256, 16)；P5：(4, 25, 256, 16)；shared A、rank-1 C、D 存在。 |
| 責任邊界 | A 是算子的實際輸入（上游 `−exp(A_log)`），即使 m 凍結 A_log 也照讀；C rank 1 只在 state 維度 broadcast，仍做 N 維 state 遞推。 |

## 2. 呼叫、stream 與 status 契約

| 項目 | 契約 |
| --- | --- |
| host 拒絕 | 在提交任何 GPU 工作前檢查：shape 規則與 int32 容量（診斷與 S01 相同）、非空 tensor 的指標非 null 且 4-byte 對齊、y 與 status 不與任何輸入或彼此重疊、stream 為顯式非預設 stream（拒絕 `0`、`cudaStreamLegacy`、`cudaStreamPerThread`）。違反時拋 `std::invalid_argument`，不 enqueue 任何工作。 |
| 提交順序 | 全部在 caller 的 stream：①`cudaMemsetAsync` 清零 status；②有限值檢查 kernel；③scan kernel（L=0 時省略）；④finalize kernel。不配置記憶體、不同步、不讀回 host、不使用 default stream，也不 record event；完成通知由 caller 在其後 record。 |
| workspace | **0 bytes。** 遞推狀態全部在 registers；唯一的每次呼叫 device 記憶體是 caller 擁有的 8-byte `ScanStatus`，每次呼叫先清零。不跨呼叫保留任何狀態。 |
| status | `flags`：失敗位元（§2.1）；`done`：finalize 最後寫入 `kScanDone`（0x4E414353）。**輸出有效 ⇔ `done == kScanDone` 且 `flags == 0`**；kernel launch 成功不代表數值成功。 |
| 失敗輸出 | `flags ≠ 0` 時，finalize 將整個輸出覆寫為 quiet NaN（0x7fc00000），不留下部分結果；後續 stage 須先查 status。 |
| CUDA API 失敗 | 拋 `std::runtime_error`；已提交的部分不會寫入 `done`，因此不會被當成有效結果。 |

### 2.1 失敗位元

| 位元 | 名稱 | 意義 |
| --- | --- | --- |
| 0–5 | `kScanNonFiniteU`／`Delta`／`A`／`B`／`C`／`D` | 該輸入 tensor 含 NaN／±Inf。**精確**：該 tensor 有非有限值 ⇔ 位元設定；L=0 時仍檢查 A、D（與 S01 相同）。 |
| 8–13 | `kScanDecayRange`／`DriveRange`／`StateRange`／`TermRange`／`SkipRange`／`OutputRange` | 每個 state thread 依 reference 次序（decay、drive、state、C·state）回報第一個超出 FP32 有限範圍的量；同一 (b,d) 的 N 個 state 取聯集後該 lane 停止；再依序檢查 D·u 與輸出。不同 lanes 取聯集。有任何輸入位元時，範圍位元不具意義。 |
| 16 | `kCrossScanNonFiniteInput` | Cross-Scan 展開或合併的輸入含非有限值。 |

reference 只回報 (b,d,t,n) 迴圈中的第一個失敗；GPU 回報聯集。單一 lane 的案例兩者應一致。

## 3. FP32 算法（S02 基準）

建置：`nvcc --ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo`，不使用 `--use_fast_math`。scan 與 merge 的每個算術步驟都以 `__fmul_rn`／`__fadd_rn`／`__fmaf_rn` 明寫，contraction 旗標不能改變結果。函式庫誤差依 [CUDA Programming Guide 數學函式附錄](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)：`expf` 2 ulp、`log1pf` 1 ulp、`frexpf`／`ldexpf` 0 ulp；NVIDIA 註明這些界來自廣泛但非窮舉的測試。

| 步驟 | 算法 | 理由 |
| --- | --- | --- |
| Δ＝softplus(x) | x > 20：x；−80 ≤ x ≤ 20：`log1pf(expf(x))`；−1024 ≤ x < −80：e^x，以 Cody–Waite（k＝rint(x·log₂e)，r＝(x−k·ln2_hi)−k·ln2_lo，兩步皆 FMA）得 e^r·2^k；x < −1024：0。Δ 以 (m, e)、m ∈ [0.5,1) 保存。 | 門檻與數學語義同 S01。舊 kernel 的 `logf(1+expf(x))` 在 x=−18 時 1+1.5e−8 捨入成 1，Δ 變 0；`log1pf` 沒有這個相消。x < −80 時 log1p(e^x) 與 e^x 相對差 < 1e−35，(m, e) 避免 Δ 落入 FP32 次正常數（例如 S01 的 `extreme_fp32_max_input`：delta=−100 時 FP32 的 e^−100 只剩約 5 位有效位）。x < −1024 時 Δ < 2^−1477，對任何 FP32 drive 皆無影響。 |
| decay | `expf(ldexpf(m_Δ·A, e_Δ))` | 與 FP32 的 Δ·A 相同捨入；m_Δ·A 不會溢位。 |
| drive | `ldexpf((m_Δ·m_B)·m_u, e_Δ+e_B+e_u)`，B、u 以 `frexpf` 拆開 | 指數精確相加，常態範圍內與一般 FP32 同為兩次捨入；中間量不會溢位或下溢。Δ=2¹⁰⁰、B=2¹⁰⁰、u=2⁻¹⁰⁰ 時一般 FP32 的 Δ·B 已溢位，此式得精確的 2¹⁰⁰。 |
| state | `h = __fmaf_rn(decay, h, drive)` | 單次捨入。 |
| 輸出 | term＝C·h；同一 (b,d) 的 N 個 thread 以 `__shfl_xor_sync` butterfly 求和（深度 log₂N，次序固定，各 thread 結果逐位元相同）；有 D 時再加 D·u。若所有 term 有限但部分和超出 FP32 範圍，改以 2⁻⁸ 精確縮放重算，再乘回 2⁸。 | 與 reference 的 n 遞增次序不同，誤差界見 §5。縮放重算讓 reference 有限的結果不因 FP32 部分和溢位而失敗。 |
| Cross-Scan | 展開為純搬移；合併為 ((y0+y1)+y2)+y3 再乘 0.25，和溢位時改為先各乘 0.25 再相加。 | 展開與 reference 逐位元相同。 |

thread 配置：每個 (b,d) lane 用 N 個連續 threads（N 整除 32），256 threads／block，所有索引 64-bit。int32 容量是契約上限，不是 kernel 的索引上限。

## 4. 支援範圍與限制

以 §3 的算法，S01 接受的合法輸入中，下列兩類以外都由 FP32 路徑處理，並以 §5 驗收：

| 限制 | 內容 | 處理 |
| --- | --- | --- |
| 邊界分類灰區 | decay、drive、state、C·state、D·u、output 的量級在 FP32 最大有限值附近幾個 ulp 內時，FP32 捨入（含 `expf` 2 ulp）可能使 GPU 與 binary64 reference 對「是否超出範圍」判定不同，雙向皆可能。 | 明列為 S02 支援限制；驗收案例的量級與 FLT_MAX 至少相差 2 倍，邊界判定不列入 PASS。 |
| 次正常數精度 | 結果或中間量低於 2⁻¹²⁶ 時，每次捨入的絕對誤差可達 2⁻¹⁵⁰（`expf` 為 2⁻¹⁴⁸），相對誤差可很大。 | §5 的 η 項；另以 `extreme_subnormal_input` 確認次正常數未被 flush（`--ftz=false` 的實際行為）。 |

不支援：FP16 I/O、非連續 stride、in-place（y 與輸入重疊）、非零 initial state 與 final state（S01 契約沒有這些欄位）。效能不在 S02 驗收範圍。

## 5. 數值比較政策（首次執行前固定）

**同一份輸入。** GPU 與 reference 使用同一批 FP32 bytes：GPU 直接讀取，reference 精確升為 binary64 計算。差異只來自運算過程。

**逐元素誤差界。** 對每個輸出元素，測試以 binary64 依 §3 的實際算法計算一階 running error bound E。記 u＝2⁻²⁴，k＝log₂N，γ_j＝ju/(1−ju)，η＝2⁻¹⁵⁰，η_q＝2⁻¹⁴⁸，ε_Δ＝6u。對每個 (b,d,n)，Δ_t、a_t＝Δ_t·A_n、q_t＝e^{a_t}、r_t＝Δ_t·B·u、h_t＝q_t·h_{t−1}＋r_t 為 binary64 的精確值：

- ρ_t＝expm1(|a_t|·(ε_Δ＋u))·(1＋4u)＋4u
- E_{n,t}＝[(q_t(1＋ρ_t)＋η_q)·E_{n,t−1}＋(ρ_t q_t＋η_q)·|h_{t−1}|＋(ε_Δ＋2u)·|r_t|＋u·|h_t|＋2η]／(1−u)，E_{n,−1}＝0
- S_t＝Σ_n |C·h_{n,t}|＋|D·u_t|；E_y＝(1＋γ_{k+2})·Σ_n |C|·E_{n,t}＋γ_{k+2}·S_t＋(N＋1)·η

依據：Δ 的三個分支相對誤差 ≤ 6u（`expf` ≤ 4u，經 log1p 傳遞不放大，`log1pf` ≤ 2u；Cody–Waite 分支 ≤ 4.4u；x>20 精確）；drive 兩次 mantissa 捨入加 `ldexpf` 次正常捨入；decay 的引數相對誤差 ε_Δ＋u、`expf` 4u 與次正常 2 ulp；state 一次 FMA；term 一次捨入、butterfly 深度 k、D·u 與最後加法各一次。縮放重算只在部分和 ≥ 2¹²⁸ 時發生，其額外誤差 ≤ 2⁻¹⁴²·N，遠小於 γ·S。

**判定。** 每個元素 |ŷ − y_ref| ≤ 2·E_y（係數 2 吸收一階截斷與 reference 本身 ≤ 1e−14·S 的誤差）。E 隨項量級 S 而非 |y| 縮放，所以同一規則涵蓋一般、近零／相消與次正常數案例；不另設全域 atol／rtol。S01 的 1e−10 容差只屬 binary64 reference，不沿用。

**Cross-Scan。** 展開必須與 reference 逐位元相同。合併：|ŷ − y_ref| ≤ 2·(γ₄·Σ_k|y_k|／4＋η)。展開→scan→合併的串接：|ŷ − y_ref| ≤ 2·[(1＋γ₄)·Σ_k E_k／4＋γ₄·Σ_k|y_k|／4＋η]，E_k 為該 token 的 scan 誤差界。

**失敗行為。** reference 回傳結果時，GPU 須 `flags == 0`、`done == kScanDone`。reference 因非有限輸入拒絕時，GPU 的輸入位元須恰為該 tensor；因範圍拒絕時，單一 lane 案例的 `flags` 須恰為對應範圍位元。兩者都須 `done == kScanDone` 且輸出全為 quiet NaN。

**報告。** 每個案例列出元素數、max abs、max rel（對 |y_ref|，參考用）、max(誤差／門檻)、超出門檻的元素數，以及比值最大元素的索引、GPU 值（含 bits）、reference 值與門檻。平均誤差不作為依據。

**辨識力。** 門檻須能分辨已知錯誤：review 的測例（L=N=1，delta=−18、A=−1、B=1e8、u=C=1、D=0，reference ≈ 1.523）以 host FP32 模擬舊式 `logf(1+expf(x))` 得 0，測試要求該偏差超出門檻；softplus 掃描列出舊式在多少點超出門檻。實際 CUDA mutation 另於 evidence 執行（§6）。S01 的 softplus `>`／`>=` mutation 在 FP32 下兩分支都捨入成 20，不可觀察，不列入。

## 6. 測試矩陣與 mutation

| 測試 | 要確認的事情 |
| --- | --- |
| S01 全部數值 fixtures（含三個 m ABI 形狀）＋三個 m 形狀的實際尺度隨機資料 | 遞推、broadcast、D skip、部署尺寸 |
| N=1…32 × shared／per-channel A × rank-1／full C × D 有無；L=0、L=1、Dc·N 非 256 倍數、長序列慢衰減 | 邊界、尾端、誤差累積 |
| softplus 掃描（−120…1000，含 −80、20 與相鄰 float）、review 的 −18 測例、相消案例 | softplus 穩定性與容差辨識力 |
| 中間溢位但 reference 有限：drive（2¹⁰⁰·2¹⁰⁰·2⁻¹⁰⁰）、butterfly 部分和、部分和由 D·u 拉回；Δ 下溢：delta=−100／−150 配大 B·u | §3 的縮放處理 |
| 六個 tensor 的 NaN／±Inf（含 L=0 的 A、D）；六類範圍失敗；多 lane 中單一 lane 失敗 | status 位元、整個輸出 NaN、`done` |
| 同一組 buffers 連續「A → B → A」（B 分別為失敗與另一組有效輸入） | 每次呼叫重設 status 與輸出，無殘留 |
| host callback 擋住非預設 stream：放行前 scan 的完成 event 未完成；放行後結果對應放行後才寫入的輸入。stream capture（global mode）成 graph、檢查節點只有 1 個 memset＋3 個 kernel、以不同輸入重播。兩個 stream 同時執行不同案例。拒絕 default stream。 | 工作確實在指定 stream、無隱式同步／配置／D2H、呼叫之間無共享狀態 |
| host 拒絕：S01 的 shape／容量條件、null／未對齊指標、輸出重疊；拒絕後 status sentinel 不變 | 提交前拒絕 |
| Cross-Scan：S01 手列順序、非對稱矩形（3×5、1×4、4×1）、batch 2、m 的 20²／10²／5²×128；合併溢位；非有限值；展開→scan→合併串接 | 翻轉、展平、還原順序 |
| 同一輸入重跑 | 逐位元決定性 |

compute-sanitizer memcheck／initcheck／racecheck／synccheck 執行完整 GPU 測試。kernel 不用 shared memory；synccheck 檢查 warp shuffle／reduce 的 mask。

Mutation 以私有副本修改一處 CUDA 實作，未修改的測試必須失敗：舊式 softplus、移除 Cody–Waite 分支、一般 FP32 drive、移除部分和縮放重算、去掉 D skip、per-channel A 讀第 0 列、full C 讀第 0 個 state、butterfly 少一層、status 不清零、失敗不覆寫 NaN、不寫 `done`、有限值檢查略過 D、scan 改送 default stream、合併不取平均、方向 2 改翻 H。
