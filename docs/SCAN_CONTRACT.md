# 新 head 的 scan 契約查核

查核日期：2026-10-29；v1 source／交付基線 `9f2c11a838a229c05ba7e9e17fa182bf0799746b`。
**精確 scan 契約未核實，S01 OPEN。** 本文件固定已知的設計約束、未知欄位及解鎖輸入；尚不是可供實作的算子規格。完整查找範圍與命令見 [本輪證據](../evidence/scan-contract/2026-10-29/REPORT.md)，模型取得需求續記於 [readiness §6](MODEL_ARTIFACT_READINESS.md#6-2026-10-29-追加來源查核與解除阻塞)。

## 1. 來源與適用層級

| ID | 固定來源與 locator | 已核實的內容與界線 |
| --- | --- | --- |
| V1 | [DATAFLOW B03／F03／§9.2](DATAFLOW.md)；上述 v1 revision | YOLO26s backbone／neck、P3／P4／P5、Mamba-FPN head、640、幀內推論為既定家族；可查材料尚未固定目標 scan variant。 |
| D1 | [既有架構規格 §4.1](https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/docs/reference/math_model.md#L397-L464)，revision `e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2` | 核實 DATAFLOW 引用的歷史設計意圖，詳下表；未取得對應的新 head source/config/artifact。文件 hash／尺寸／原文行號見 [design 查核](../evidence/scan-contract/2026-10-29/design/FINDINGS.md)。不復用舊程式、權重或數值驗收。 |
| U1 | [MambaFPN 出版來源](https://doi.org/10.1016/j.neunet.2026.108544)，搜尋索引中的 introduction 預覽 | 同名論文討論 locality-aware、bidirectional 的 **neck**；未建立其與 V1／D1 head 的映射。直接取得全文失敗，預覽不是完整公式；DOI 不等於程式 revision。 |
| U2 | [MambaPSA 作者設定](https://github.com/henrychan0719/MambaPSA/blob/2940f1634cfce1f63201e6411ffe9008951461f3/yolo26n-mambapsa.yaml)，revision `2940f1634cfce1f63201e6411ffe9008951461f3` | C2PSA／neck 替換搭配標準 Detect；有 checkpoint inventory，但未提供本目標的 native bundle。不是 D1 的獨立三層 head；其 scan 或數值不能轉作 V1 驗收。 |

U1／U2 是候選身分不符，尚無兩份「同一目標」實作可判定相互矛盾。沒有定位到目標 source，也不足以斷言目標實作不存在或 head 一定尚未設計；此狀態須由 v1 模型負責人確認。

## 2. Scan 必須固定的契約

「已核實」在此指指定來源確實這樣記載；只有對應目標 source/config/算子後，才可升為 S01 實作依據。

| 契約項目 | 來源／locator 與本輪結果 | 仍須交付的精確欄位；阻塞階段 |
| --- | --- | --- |
| 目標與算子身分 | V1 B03、D1 lines 420–432 只標兩個 MambaBlock。**缺件**：目標算子身分未核實。 | 新 head 中的 module/node locator、source revision/config；scan domain、op type、opset、全部 attributes、plugin 是否需要及其邊界。S01 需先對應數學算子；plugin binary/build 身分屬 S03。 |
| Tensor 邊界 | D1 lines 397–432：reduced tokens 400/100/25、`d_model=128`、`d_state=16` 為**已核實設計意圖**。 | 真正 scan 的 ordered input/output 名稱與角色、shape/dtype/layout/element stride、B/C grouping/broadcast、state size/channel 合法範圍、容量乘積上限與有限值範圍。`d_model` 不必等於內部 scan channels；不得由 128 推算內部 expansion。S01。 |
| Scan 次序 | D1 lines 413–432 有 flatten／up4 標籤，U1 公開說明提及 bidirectional；**缺件**：沒有目標 index formula。 | `(h,w,c)` 到 token 的精確映射、方向數及每方向順序、逆向還原、merge 時點與 sum/mean/concat 等運算。不可把 U1 的雙向或一般 VMamba 四向套給 D1。S01。 |
| 精確數學語義 | D1 兩個 MambaBlock 沒有 recurrence。**缺件**。 | 依目標核實離散化／狀態更新、A/B/C/D（若有）的 shape/角色、delta bias 與 softplus 順序、gate/skip 與輸出投影位置。外層 U-Net skip 不能當成 D skip；不得以通用 Mamba 式填空。S01。 |
| 狀態生命週期 | V1 B03：**已核實**不跨幀保留 ModelState。 | 每次 scan 的 initial state、是否支援非零 state、final state 是否輸出、各 lane/block/frame reset、零長度接受／拒絕與回傳值；不支援項要明列拒絕。幀內不等於所有 invocation 自動有相同 state ABI。S01。 |
| 數值比較 | V1 C++／CUDA、模型輸入 FP32；D1 未定 scan arithmetic。**缺件**。 | compute/accumulation dtype、FMA／fast-math／TF32等選項、獨立 expected 推導方法與適用範圍、metric、事先固定且有依據的 atol/rtol。公式核實後可獨立手算微型案例，不要求先有真實權重才可推導 fixtures。S01。 |

尚未指定任何 scan atol/rtol 或產生 scan fixture；affine 小網路 exact 結果與本表無數值推導關係。零 token、非零 initial state、gate、skip 等案例的接受／拒絕條件，也必須等契約固定後才能寫成測試。

## 3. Head 的 ordered cls／reg 與 decode

| 邊界 | D1 已核實的設計意圖 | 新 head 未核實欄位與影響 |
| --- | --- | --- |
| Backbone／neck → head | lines 397–404：P3/P4/P5，stride 8/16/32，channels 128/256/512，80²/40²/20²。 | 真正公開匯出的 ordered 名稱、shape/dtype/layout/stride與匹配新權重；一般 YOLO26s 的內部值仍不是已交付的 component I/O。P01／S04。 |
| 每層 lane | lines 406–432：input projection 128、down4、兩個 block、up4，再與 input projection concat 成 256 channels。 | 新 head 是否採此設計、各投影／卷積／上採樣的精確參數及 scan-to-head 對應。後者阻塞 S01，完整算子及權重阻塞 S04／P01。 |
| Raw cls／reg | lines 434–443：cls 80、reg 4、`reg_max=1`，無 DFL；lines 450–453 按 levels flatten/concat 共 8400 anchors。 | ordered 輸入輸出名稱與實際 tensor order；新 class 名單及 ID、box distance 編碼／範圍、候選容量／padding／invalid mask。8400 是歷史 raw anchor 設計數，不能當正式候選容量。S04／P01；不單獨阻塞獨立 scan。 |
| Decode | lines 445–464：grid center +0.5、dist2bbox×stride、sigmoid/max-class。V1 F03 raw cls/reg → F04 decode/NMS（允許圖內融合，suppression 只執行一次）。 | 新 head 的 score/box 編碼、decode 是否圖內實現、class tie/合法範圍、穩定 anchor ID、padding 與完整 cls/reg 對照。P02／S04／P01。不可改用一般 YOLO 的 `[1,300,6]` 取代。 |

## 4. 最小待裁定設計與 S01 解鎖

若模型負責人確認尚無新實作，以下是待裁定項，**不是已採納契約**：

| 決策 | 候選依據與建議 | 裁定者／所需材料 |
| --- | --- | --- |
| D01 Head 身分 | 保留 D1 的 per-level down4／two-block lane，或交付另一份符合 B03 的新 head 規格。建議先以已引用的 D1 作起點，避免將 public neck 誤當本 head；不能據此宣稱已完成新設計。 | v1 模型負責人（具名持有人未定位）；確認架構並提供固定 config/module map。 |
| D02 Scan variant／公式 | 單序列／雙向／多方向、real/complex state、B/C sharing、delta transform、gate/D skip 都影響輸出。D1 沒有足夠依據推薦某個 recurrence；建議先固定權威數學規格及 head 中的位置，再做 native 實作。 | 模型／scan 設計者；不可變 source/spec＋逐式 locator＋所有 attributes。若採新設計，明標新決策，不能稱核實既有模型。 |
| D03 Scalar ABI／能力 | 建議以 dense FP32、每幀 reset 作最小候選，與既有固定輸入和幀內基線一致；仍須裁定 stride/broadcast/容量/初末狀態/零長度及拒絕策略。 | 模型／native 實作者；有版本的 scan tensor 與 failure profile。 |
| D04 比較依據 | 建議用固定小尺寸、非退化輸入，依核定公式獨立手算，再補可執行的 native 對照；固定 metric/容差/編譯選項後執行。真實模型對照另驗。 | 模型／數值驗證負責人；獨立推導方法與容差理由，不由 reference 自產唯一 expected。 |

最小取得包是：**新 head 的固定 source/config/module map ＋上述 scan 精確規格／ABI／數值比較政策**。核對後即可從零實作 standalone C++ scalar reference，加入 CPU CTest 正反案例與 ASan／UBSan。新模型權重、完整 ONNX/external data、cls/reg/decode 與 producer lineage 另解鎖 P01／S04；不把這些與 standalone scan 的最低條件混成一個 gate。CUDA forward 與 plugin 分別留在 S02／S03。
