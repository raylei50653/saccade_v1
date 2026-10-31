# F04 偵測後處理政策（`f04-postprocess-fp32/1`）

固定日期：2026-10-31。本檔、CPU scalar reference 與固定測例須在首次 F04 GPU 比較前保存 SHA-256；不得依 GPU 誤差改變容差或離散判斷。若政策或算術假設改變，保留原失敗結果並另立版本。本政策固定 [BACKLOG](BACKLOG.md#1-f04先固定觀測政策再實作) 的 F04；不改寫已結案的 [T01 evaluator](../training/t01/t01_eval.py) 或其歷史結果。

## 1. 輸入、身份與容量

輸入為單幀、batch 1、640×640 的六個 dense FP32 tensor。每個 level 為一對 cls `[1,80,H,W]` logits 與 reg `[1,4,H,W]` raw ltrb；依 P3、P4、P5 排列，H=W 分別為 80、40、20，stride 分別為 8、16、32。reg 通道順序為 left、top、right、bottom；沒有 DFL、softmax、額外 exp、rectify 或角點量化。

候選 ID 在任何篩選前固定：P3 row-major 為 0–6399，P4 為 6400–7999，P5 為 8000–8399；同一 level 的序號為 `y*W+x`。class ID 固定為 0–79，其中 0 為 person。身份是 caller 提供的 epoch 與 processing frame，寫入輸出 header，不把模型候選 ID 當 frame 身份。

`ImageMapping` 沿用 [native F01／F02](PREPROCESS.md) 的連續 image-edge 座標契約，僅支援無 crop／pad 的 640 stretch。原圖 width／height 的支援範圍為 1–65535，model width／height 必須為 640；x／y scales 必須分別等於以 binary64 計算的 `640.0/original_width` 與 `640.0/original_height`，以 exact 比較驗證。不得將等比例 letterbox、反向 scale 或已映射框冒充本輸入。

輸出 slot 固定容納最多 **300 rows**，kept 與 suppressed 共用這份容量，另帶 device header。caller 可請求 0–300 的有效 capacity；超出此區間屬 host argument error。若本幀應輸出的完整 person pool 超過請求 capacity，整幀回報 overflow，沒有截斷或部分成功。請求 capacity=0 且有效 person pool 為空是合法空結果。top-300 是下節明定的觀測政策，不是容量 overflow 的靜默修補。

每個 row 保留原始觀測 probability、class、穩定 candidate ID、原圖 xyxy，以及互斥的 kept／suppressed 標記。suppressed 僅指被本次 NMS 抑制的合法 person 候選；F04b 可據此組裝 continuation-only 候選，該階段仍須自行禁止 birth。低分、top-k 淘汰、非 person、退化框都不得標成 suppressed。

## 2. FP32 算術與觀測次序

所有決策算術均為 IEEE-754 binary32、round-to-nearest，保留 subnormal，不用 fast math、融合乘加或近似 reciprocal。CPU reference 用 float 運算並關閉 contraction；CUDA 用明確 FP32 加／減／乘／除，不能藉由重新結合算術改變邊界。

1. **驗證完整 raw tensors。** 掃描全部 672,000 logits 與 33,600 reg 值。任意 NaN／Inf 均使整幀 numeric failure，包括本來會被低分、top-k 或 class 篩選淘汰的元素。
2. **Decode 全部 8,400 個模型框。** anchor 為 `(x+0.5,y+0.5)`，逐軸先做 `anchor-left`、`anchor-top`、`anchor+right`、`anchor+bottom`，再乘該 level 的 stride。任何 decode 算術溢位或非有限座標均使整幀 numeric failure，不以低分掩蓋。有限負 ltrb 本身不是輸入錯誤，也不額外 clamp。
3. **計算 probability 與 class。** 對每個 class 分別計算 stable sigmoid：`e=exp(-abs(logit))`，`p=(logit>=0 ? 1 : e)/(1+e)`。CPU reference 使用 `std::exp(float)`，CUDA 使用 `expf`；每個中間及最終值均為 float。有限極端 logits 可正常飽和至 0／1。class 為這 80 個 FP32 probabilities 的 argmax；probability 相同時取最小 class ID。必須比較 sigmoid 後的值，不能用 logit argmax 取代，因 FP32 飽和可使不同 logits 變成同分。
4. **Confidence 篩選與全類別 top-k。** score 為上一步的最大 probability。只保留 `score >= float(0.001)`；依 `(score 降序, candidate ID 升序)` 建立全序，最多取前 300 個。這一步在 person 篩選之前執行，非 person 或退化框仍占 top-300 名額。此處的 `float(0.001)` 指 binary32 最近值，不是高精度十進制比較。
5. **Person 與退化框篩選。** 從 top-300 依原排序先檢查 class，再保留 class 0 且 `x2>x1 && y2>y1` 的框；非 person 計入 class 淘汰，不重複計為退化框。退化判斷在未 clip 的模型座標上執行，淘汰後不從 top-300 之外補名額。有效 person pool 的全部成員都須交付 kept 或 suppressed row。此時先檢查 pool 是否超過 caller capacity；超過即整幀 overflow，不再進行 NMS。
6. **Greedy NMS。** 依上述全序巡訪有效 person pool。若框與任何較前的 **kept** 框的 IoU 嚴格大於 `float(0.5)`，標記 suppressed；否則 kept。被 suppressed 的框不能再抑制後面的框。所有標記與 rows 保持原 score／ID 排序，不重新 compact 成 kept-first，也不執行第二次 NMS。
7. **原圖座標映射。** NMS 完成後，模型 x 座標除以 FP32 x scale，模型 y 座標除以 FP32 y scale；scale 由 `ImageMapping` 的 double 值各自轉為 float。輸出每個座標必須有限，否則整幀 numeric failure。kept／suppressed 都使用同一映射。

**Clipping 固定為 none。** 這與 T01 evaluator 沒有 clip 的行為一致。有限且非退化的越界框可保留負座標或超出原圖邊界的座標；不在 NMS 前後 clamp，也不在映射後因越界而丟框。避免先以框抑制其他候選、再因 clipping 刪除該框的額外語義。若將來需要 clipping，須另立政策並重驗。

### 2.1 IoU 的運算順序

IoU 在未 clip 的 640 模型座標計算，採連續 xyxy，沒有像素 `+1`。每個面積先計算 FP32 width 與 height，再相乘。交集 width／height 分別為 `max(0,min(x2a,x2b)-max(x1a,x1b))` 與 y 軸對應式，之後相乘；union 是 `(area_a+area_b)-intersection`，最後以 intersection 除以 union。

每個 selected valid person 的面積都須有限且嚴格大於 0，即使該幀只有一個框或它已可被其他框抑制，也不能跳過本框的面積檢查。對 greedy NMS 實際比較的 kept pairs，座標差、交集、area sum、union 與 IoU 的中間值均須有限，union 必須嚴格大於 0；失敗均為整幀 numeric failure。若正寬／高的相乘 underflow 成 0，亦依 nonpositive area 拒絕，不能把溢位後的 Inf／NaN 或 `0/0` 當作不重疊。未實際比較的 pair 不額外計算 IoU。IoU 恰為 0.5 時保留，不用 epsilon 移動閾值，也不用分數容差容許不同的 NMS 決策。

## 3. Device status、完整輸出與所有權

F04 必須讀取 caller-owned、device-resident producer status；它包含 flags 與 done marker。producer pointer 為 null 是 host argument error，在 enqueue 前以 `invalid_argument` 拒絕；有效 pointer 所指的 done marker 不符，則在 device 回報 protocol failure。非零 flags 回報 upstream failure，與 done 檢查彼此獨立；flags 非零且 done 不符時兩個 failure bits 都設定。上游狀態只讀，不消費或退休；caller 負責每幀重新產生狀態並建立 stream／event 依賴，不能重用前幀的成功 marker 冒充本幀成功。

完整 F03 整合時，producer status 必須已彙整該 engine 的全部 plugin statuses 與輸入／輸出 finite checks。本 F04 replay 驗證提供已錄製 raw predictions 與明確 producer status，不代表此彙整器或完整 native session 已完成。

每次 enqueue 在 caller 指定、非 default 的 stream 上重設自己的 header、rows 與 scratch，再執行 device 工作；最後才寫完成 marker。成功 header 保留 epoch／frame、capacity、有效 row count、kept／suppressed counts 與篩選統計。失敗 header 保留身份／capacity／明確 error 與完成 marker，所有輸出 counts／篩選統計為 0，整份 rows 清零。成功空候選必須 flags=0 且 count=0，與失敗 count=0 分開判斷。未使用 rows 及 reserved／padding bytes 固定全 0；比較包含這些 bytes。

正常 enqueue 不配置記憶體、不同步、不讀回 count／status、不做 postprocess D2H，也不由 host count 決定後段 launch。kernel loading 與所有 slot／scratch 配置在 frame 提交前完成。caller 保留 raw inputs、producer status、輸出與 scratch 直到最後一次讀寫完成；跨 stream 交付後，在 F04b 的 consumed 邊之前不能覆寫候選 slot。[ASYNC_CONTROL](ASYNC_CONTROL.md) 的完整 CS ready／consumed 整合仍由 session 任務實作。

host 的 shape、capacity、mapping、null／device／allocation bounds、重疊區間與 stream 拒絕必須在 enqueue 任何工作之前完成。kernel launch 或 CUDA API 失敗以明確 runtime error 傳播，不把未完成 marker 解讀為合法空觀測。

## 4. Reference 與比較判定

CPU scalar reference 使用與 §2 相同的 FP32 中間步驟，但獨立完成全候選排序、person 篩選及 greedy NMS。固定測例亦有手算答案／不變量，不能只以兩份相同實作互相背書。

- score：每個有效 row 必須有限，且 `abs(cuda_score-reference_score) <= 4*FLT_EPSILON`（即 `2^-21` 絕對界）。此預算只涵蓋 CPU float exp 與 CUDA expf 的固定 sigmoid 路徑差異，不能容許不同 class／top-k／NMS 決策。
- 其他全部欄位：candidate ID、class、kept／suppressed、順序、counts、統計、status、metadata、capacity、原圖 xyxy 均要求 exact；座標按 FP32 bits 比較，不加容差。
- 同一 GPU／binary／輸入的重跑，包含 slot reuse 與 graph replay，整份輸出 bytes 必須相同，包括 scores、未使用 rows、reserved／padding。不同 GPU／toolchain 不繼承此判定。
- 失敗／空結果：核對完整 header 與全部 rows，不只看 exit code、count 或 kept boxes。

任何超界、非有限輸出、離散決策不符、未重設 bytes 或未完成 marker 均為 FAIL。score 在容差內但排序／class／threshold 決策不同仍是 FAIL；不能使用 tolerance 把不同偵測集合合併成 PASS。

## 5. 首次 GPU 比較前固定的測例

以下測例由合成 raw logits／reg 與手算期望構成；不能依 F04 GPU 結果選取或刪除 cases。

| 組別 | 必須區分的行為 |
| --- | --- |
| Level／layout | P3／P4／P5 首末 anchor、非對稱 ltrb、跨 level candidate ID、CHW class／reg 通道、原圖非方形 x／y scale。 |
| Empty／capacity | 全部低分；top-300 中無 person；合法零 capacity 空幀；非空零 capacity overflow；正容量恰滿與少一 row；超 300 capacity 的 host 拒絕。 |
| Score／class | 閾值上下相鄰 logits、恰好等於可表示閾值的 probability 若可構造、class tie 最小 ID、異 class logits sigmoid 飽和同分、全類別先 top-300、同分第 300／301 名以 candidate ID 決勝。 |
| Top-k／invalid | 高分非 person 排擠 person；top-300 退化框占名額且不補；負 ltrb 可有效或退化；低分／非 person／top-k 淘汰與 suppressed 明確不同。 |
| NMS | 相同框、互不相交、IoU 恰 0.5 與其上下邊界、同分不同 ID、A 抑制 B 但 B 不能抑制 C 的 greedy chain、保留完整 suppressed pool。 |
| 座標／數值 | 不 clip 的負／越界框、identity 與分軸映射、有限極端 logits、raw NaN／Inf（含低分／未入選 rows）、decode overflow、IoU area／union overflow、固定 anchor 附近相鄰 FP32 邊界形成的最小可表示有效框；若可到達 nonpositive area／union，核對其明確 failure，不能宣稱已構造不可達的 underflow case。 |
| Status／reset | null producer 的 host 拒絕；錯 done、上游失敗、兩者同時錯誤；成功→失敗→空→成功重用，明確核對前幀數據不可殘留。 |
| Ownership／stream | 非 default stream、延遲 producer、跨 stream event、延遲 consumer 後再重用、兩個獨立 slot／session、graph capture／replay；所有 allocations 的 null、範圍與 alias 拒絕。 |

錄製的真實模型輸出沿用 F03 已固定的六個 inputs：`MOT17-02-SDP/1`、`MOT17-10-SDP/1`、`MOT17-13-SDP/400`、zeros、ones、random/1。從 [F03 證據](../evidence/t01/2026-10-30/f03/REPORT.md) 對應的大型 raw output bytes 重播時，先核對 source engine／raw bytes hashes、level order 與形狀；如未實際執行或無法取得，明列 UNEXECUTED，不以合成案例替代真實 replay 的聲明。

測例應能辨識錯誤變體：person-only top-k、只交付 kept、忽略 candidate ID tie、把 class argmax 移到 sigmoid 前、改用 `IoU>=0.5`、讓 suppressed 繼續抑制、錯 stride／anchor／ltrb 次序、x／y scale 互換、增加 clipping、不清零失敗／未用 rows、忽略 upstream failure。變體檢查不是額外放寬正確輸出的容差。

## 6. 證據邊界

本項驗證 raw predictions 到完整 `DetectionCandidates`，以及此 API 的 stream／重用／失敗契約。Sanitizer 的 memcheck、initcheck、racecheck、synccheck 分別記錄實際執行範圍與結果；未執行的不列 PASS。正式報告保存政策／reference／cases hashes、source／binary／compiler／CUDA／GPU 身分、完整命令與退出碼、逐 case 比較及未驗清單。

F04 PASS 不代替 native F01／F02→F03 的完整接續、F03 producer 聚合器、F04b、GMC、tracker、ResultPacket、MOT writer 或完整 async session 的驗收，不關閉 P01，也不改變 T01 detection AP／模型品質結論。
