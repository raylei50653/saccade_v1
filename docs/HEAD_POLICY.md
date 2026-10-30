# S04 完整 head 的 FP32 數值比較政策（`s04-head-fp32/1`）

固定日期：2026-10-30。**本政策在第一次完整 head 的 native GPU 比較之前提交**，與其實作 [`head_reference.hpp`](../include/saccade/head_reference.hpp)／[`head_reference.cpp`](../src/head_reference.cpp) 同一 commit。之後若 GPU 驗收失敗，只能修正實作，或以新版本號明列修訂與理由；不依觀測到的 GPU 差異放寬門檻。整體架構與交付見 [HEAD_TRT](HEAD_TRT.md)。

適用對象：T=1 YOLO26m Mamba-FPN head 的六個 raw outputs（`cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5`），由同一份 FP32 P3／P4／P5 feature bytes 與同一份權重 bundle bytes 產生。S02 的 scan 誤差界（[SCAN_CUDA §5](SCAN_CUDA.md#5-數值比較政策首次執行前固定)）只沿用到契約相符的 scan 步驟本身；S01 的 1e−10 容差只屬 binary64 scan reference。

## 1. Reference：獨立性、精度與捨入位置

| 項目 | 內容 |
| --- | --- |
| 資料 | FP32 feature 與權重精確升為 binary64；之後**所有**中間量與輸出都是 binary64，reference 中沒有任何 FP32 捨入。 |
| Conv／Linear／causal Conv1d | binary64 乘加，依 (c, i, j) 或 i 遞增次序累加，最後加 bias；zero padding；Conv1d 取 PyTorch `padding=K−1` 輸出的前 L 個（y[t] 只讀 x[t−3..t]）。 |
| SiLU | x·σ(x)，σ 以 `std::exp` 穩定形式計算（x≥0：1/(1+e^{−x})；x<0：e^x/(1+e^x)）。 |
| A | −exp(A_log)，binary64 `std::exp`。 |
| SelectiveScan | SCAN_CONTRACT §3：softplus 門檻 20、`log1p(exp)`；n 遞增；最後加 D·u。對 FP32 可表示輸入與 S01 `selective_scan_reference` **逐位元相同**（測試要求）。 |
| Cross-Scan | 方向 0 原圖、1 翻 H 與 W、2 翻 W、3 翻 H；合併 (((y0+y1)+y2)+y3)/4；索引與 S01 `cross_scan_source` 一致（測試要求）。 |
| PixelShuffle | CRD：out[c][h·4+i][w·4+j] = in[c·16+i·4+j][h][w]。 |
| 建置 | GCC `-ffp-contract=off`，無 `-ffast-math`；glibc libm。 |
| 獨立性 | 不呼叫 TensorRT、CUDA 或 S02 kernels；索引與運算由 SCAN_CONTRACT §2 重新撰寫。只與 GPU 路徑共用唯讀權重 bundle 解碼與 feature 產生器。GPU 的中間結果不回灌 reference。 |
| reference 自身誤差 | 每步 binary64 誤差 ≤ 約 K·2⁻⁵³ 乘以項量級，比下列 FP32 尺度小約 2⁻²⁹，忽略。 |

## 2. GPU 實作的運算模型

TensorRT 10.16.1.11 strongly-typed FP32 network，TF32 關閉；scan／Cross-Scan 為 S03 plugins（S02 kernels）。u＝2⁻²⁴。

| 運算 | 模型 |
| --- | --- |
| Conv／Linear／Conv1d（K 個乘積，有 bias 時 terms＝K+1） | 任意加總次序的直接求和：2·terms 次捨入，每次 \|δ\| ≤ u，作用在絕對值 ≤ M＝Σ\|w·x\|＋\|b\| 的部分值上。轉換型演算法（Winograd、FFT）不在本模型內；實際 tactics 記入 build evidence。 |
| exp、sigmoid、SiLU | TensorRT 未公開其 FP32 pointwise kernel 精度。**假設**相對誤差 ≤ ε_t＝2⁻¹⁷（128 ulp）：涵蓋 CUDA `__expf` 的 2＋⌊1.173\|x\|⌋ ulp 至 \|x\| ≤ 108；sigmoid 下溢區另加 2⁻¹²⁵·\|x\| 絕對項。 |
| SelectiveScan 本身 | S02 kernel 的 SCAN_CUDA §5 running error bound（S03 已證明 plugin 與 S02 API 逐位元相同）。 |
| 展開、slice、reshape、PixelShuffle、concat | 純搬移，無誤差。 |
| Cross-Scan 合併 | SCAN_CUDA §5 的合併式。 |

## 3. 每個元素的尺度 σ

σ 是「FP32 實作 − 精確數學」的機率尺度：各次捨入視為獨立、零均值、有界的隨機變數（Higham–Mary 2019 的機率誤差分析）。reference 與 σ 同時逐層計算：

| 步驟 | σ 規則 |
| --- | --- |
| 輸入 feature | 0 |
| Conv／Linear／Conv1d | σ_y² ＝ Σ w²·σ_x² ＋ 2·terms·u²·M²（輸入誤差在不同元素間視為獨立；局部捨入以 M 上界部分和） |
| SiLU | σ_y ＝ hypot(SiLU′(x)·σ_x, ε_t·\|y\|) ＋ 2⁻¹²⁵·\|x\| |
| A | ε_t·\|A\| |
| 乘法（gate）、加法（殘差） | 兩運算元可能共享來源，**線性相加**（Minkowski）：\|b\|σ_a＋\|a\|σ_b＋u\|ab\|；σ_a＋σ_b＋u\|a+b\| |
| Cross-Scan 合併 | Σ_k σ_k／4 ＋ √3·u·Σ\|y_k\|／4（四方向載有同一 token，線性相加） |
| SelectiveScan | S02 bound（scan 本身）＋ 輸入的一階影響：同一步 t 的 Δ、B、u 項線性相加；不同步以平方和遞推 G_t² ＝ q²G_{t−1}² ＋ step²；共用的 A_n 跨步同號累加；各 state 線性相加進 y（共享 Δ、u、C）；C 與 D·u 項線性相加。 |

另計 first-order 最壞界 E（所有捨入同號、輸入誤差以 Σ\|w\|E 傳遞）。E 永遠成立但經多層後膨脹（候選權重下為輸出最大量級的 10³–10⁵ 倍），**只作報告，不作驗收**。測試要求每個元素 σ ≤ E。

## 4. 判定規則

**對六個輸出的全部元素**：\|ŷ − y_ref\| ≤ λ·σ ＋ 2⁻¹¹⁰，**λ＝16**。

- λ 的依據：對獨立、零均值、\|δ\| ≤ u 的捨入，Hoeffding 不等式給 P(\|Σc_jδ_j\| ≥ λ_H·u·‖c‖) ≤ 2·exp(−λ_H²/2)。λ_H＝8 時每元素 ≤ 2.6×10⁻¹⁴，三組輸入 × 705,600 元素 × 重跑仍 < 10⁻⁶。再乘 2，涵蓋層界「不同元素誤差獨立」的近似（例如 Conv1d 4-token 窗口最多 √4 的相關）。
- 2⁻¹¹⁰ 只處理 σ 恰為 0 的元素（全零輸入的精確路徑），對實際量級無影響。
- 接近零與相消：σ 由 Σ\|w·x\| 與傳遞的尺度組成，從不由 \|y\| 推出，所以接近零或相消的輸出保有完整容差，不會因 \|y\| 小而要求過嚴，也不因相對誤差而放寬。
- 平均誤差、cosine、argmax 一致都不能替代逐元素判定。

**報告**：每個輸出的元素數、max abs、max rel（只計 \|y_ref\| ≥ λσ 的元素，並報告被排除的數量）、max(誤差／門檻)、超界元素數、比值最大元素的位置 (c, h, w)、GPU 值與 bits、reference 值、門檻；另報 max(誤差／2E)。

## 5. GPU 執行前已完成的 CPU 檢查

以下只在 CPU 執行，沒有任何 GPU 數字（`head_reference_tests`）：

| 檢查 | 結果（候選 T1 bundle） |
| --- | --- |
| 每個 primitive 的手算／bc 數值 | 全部相符 |
| scan 對 S01 reference | 逐位元相同（L=25、100） |
| 輸入擾動 ≤ 其 bound 時 scan 的變化 | ≤ 0.96 倍 bound |
| 單一 primitive 的 FP32 計算 | ≤ 0.11 倍 bound |
| **整個 head 的 FP32 CPU 實作**（float 累加、反向通道次序、expf） | 兩組輸入、六個輸出全部 ≤ 0.006 倍 λσ |
| 13 個語義錯誤變體（PixelShuffle DCR、反因果 conv、方向、合併方向、concat 次序、kernel 翻轉、x／z、dt／B、rank-1 C、delta bias 位置、D skip、殘差、P4／P5 權重互換） | 全部有元素落在 λσ 之外；最弱者 `delta_bias_after_softplus` 有 56% 元素、最大 24 倍門檻 |

FP32 CPU 實作不是 GPU 路徑，只用來檢查模型的尺度合理；GPU 驗收另依 §4。

## 6. 固定的驗收輸入與權重

- 權重：S04 候選 T1 bundle，SHA-256 `e1a8660fb9ad510f20ccc3d56a4feac61c6a2e5f771aab531caa741e7677f310`（來源 checkpoint `a5a0e709…59c0676c`）。
- Features（`head_test_features`，定義見 header）：`asymmetric` seed 1（座標／通道不對稱）、`random` seed 1 與 seed 2（SiLU(1.5·N(0,1)) 型非退化資料）。各組 bytes 的 SHA-256 記入 evidence。
- 合成權重（`synthetic_head_weights`）只用於結構與失敗路徑測試，不作數值驗收。

## 7. 支援範圍與限制

- S02 的支援限制照常適用：FP32 最大有限值附近的範圍判定灰區、次正常數精度。
- 運算模型假設直接求和；若 build evidence 顯示 Winograd／FFT 類 tactic，結果須另行說明，不能直接以本政策 PASS。
- TF32 必須關閉：TF32 的 2⁻¹¹ 相對誤差遠超 λσ，若誤開會被本政策判為失敗。
- ε_t 是對 TensorRT pointwise kernel 的假設；若 GPU 逐元素結果超界且定位在 SiLU／exp，屬政策假設不成立，依 §8 處理，不能就地放寬。
- 非有限輸入或 status 失敗的 frame 走失敗路徑（HEAD_TRT §5），不進入數值比較。

## 8. 修訂規則

GPU 驗收失敗時：先以中間 tensor 定位首次偏差；屬實作錯誤則修正實作重跑；屬模型假設不成立（例如 pointwise 精度、轉換型 tactic）則以 `s04-head-fp32/2` 新版本提交修訂、理由與受影響的結果，原版本與失敗結果保留在 evidence。
