# P01-A 新模型 artifact 查核

查核日期：2026-10-28。來源、檔案與內容分別記錄。此次下載新上游檔案，沒有使用舊專案的程式、權重或數值結論，也沒有執行 Python。

2026-10-29 追加來源查核見 [§6](#6-2026-10-29-追加來源查核與解除阻塞)；前五節保留 P01-A 原始觀察，未重新下載既有 YOLO26s 或把工具 fixtures 升格為模型證據。同日使用者決策將 v1 目標改為 YOLO26m＋原版 m head：§1–§6 是 s 時期紀錄，m 的 artifact 對應見 [§7](#7-2026-10-29-原版-m-head-的-artifact-對應)。§7 保留 2026-10-29 的查核狀態；H04／H06／H07 與 backbone I/O 的最新狀態以 §8 為準。

2026-10-30 S04 更新見 [§8](#8-2026-10-30-s04head-權重的-native-交付與完整-head)：候選 T1 checkpoint 的 head 權重已 native 交付、head engine 由 v1 自建、manifest schema 2 完成，指定候選的 S04 有範圍 PASS 見 [正式證據](../evidence/s04/2026-10-30/REPORT.md)；backbone／neck 只完成舊 ONNX 的 ordered I/O 結構查核，P01 仍 OPEN。

2026-10-30 T01 更新見 [§9](#9-2026-10-30-t01matching-backboneneck-與完整模型)：T01 的 matching backbone／neck ONNX 與 head bundle 已 native 交付並建成完整模型 engine，backbone 與 F03 依 [BACKBONE_POLICY](BACKBONE_POLICY.md) PASS（有範圍）；P01 仍 OPEN。

2026-10-31 F04 更新見 [§11](#11-f04-正式驗收後的狀態2026-10-31)：乾淨 `995bca5` 的獨立 F04 驗收 PASS（受測範圍）；F03 producer 的接續實作與開發驗證見 [§12](#12-f03-producer-聚合器2026-10-31)；完整 async session 與 P01 仍 OPEN。§1–§11 保留各階段當時的狀態。

**完整 YOLO26s backbone／neck＋Mamba-FPN head readiness：OPEN。** 已取得並查核官方一般 YOLO26s ONNX；其內含一般 detection head，不能代表目標 Mamba-FPN head 已交付。

## 1. 來源與完整性

| 項目 | 本次結果 |
| --- | --- |
| 官方來源 | [Ultralytics assets v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 的 [yolo26s.onnx](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.onnx) |
| 上游識別 | release tag `v8.4.0`；tag commit `dcececebff9fe00420c144baa5cd25a641d10aa6`；GitHub asset ID `406692590`；asset created `2026-04-27T15:37:15Z`、updated `2026-04-27T15:37:25Z`。release tag 不等於 exporter code revision。 |
| 大小 | `38,291,130` bytes，下載後以 `stat` 查核，與 GitHub API asset metadata 一致。 |
| SHA-256 | `d26b65c432111eb95798cd2320603d4d75627605dbec6c6b7f98c499a80e7321`；下載後 `sha256sum -c` 通過，與官方 API `digest` 一致。 |
| 本機儲存 | `/home/ray/.cache/saccade-v1/p01-a/artifact/yolo26s.onnx`；大型上游權重不放入 Git。重跑腳本會重新下載與查 hash。 |
| 授權 | ONNX metadata 與 [Ultralytics 授權來源](https://www.ultralytics.com/license) 均宣告 AGPL-3.0；上游 LICENSE 快照保存在 evidence。此處記錄上游宣告，沒有取得 Enterprise 授權。 |
| 產生來源 | 檔案 metadata 宣告 author `Ultralytics`、Ultralytics version `8.4.41`、producer `pytorch` version `2.11.0`、date `2026-04-27T18:09:02.941072`，訓練資料路徑指向上游 COCO dataset config。這是檔案內宣告，沒有獨立重建訓練過程。 |
| Export metadata | batch 1、half false、dynamic false、simplify true、nms false；實際 opset 為 20。未提供完整 exporter commit、完整 build environment 或可重建訓練 lineage；不以目前 upstream main commit 補推。 |
| native 入口 | 已提供 ONNX，可直接由 native parser 讀取 protobuf；此次不需要載入 checkpoint 或執行 Python exporter。尚未對此 YOLO 檔案建立 TensorRT engine。 |

官方同 release 亦列出 `yolo26s.pt`（20,422,725 bytes、API digest `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`）。此次未下載或載入 checkpoint，其檔案完整性未驗證；不把 checkpoint 當成已核定的 C++ 權重入口。

原始 API 回應、查核命令、exit codes 與輸出見 [artifact evidence](../evidence/p01-a/2026-10-28/artifact/)。

## 2. 實際檔案的 native 結構查核

[onnx_contract_inspect.cpp](../tools/onnx_contract_inspect.cpp) 為此次獨立撰寫的 C++ 結構 inspector。它使用本專案 checked I/O 讀檔，以官方 [ONNX v1.20.1 schema commit](https://github.com/onnx/onnx/tree/d3f6b795aedb48eaecc881bf5e8f5dd6efbe25b3) 的 `onnx.proto` 及 native `protoc` 產生 C++ parser；沒有加入預設建置相依。實測 `libprotoc 36.1`／protobuf `36.1.0`；compiler、schema／binary／source hashes 存於 capture。

Inspector 解析成功並列出 graph I/O、所有 `value_info`、所有 node 的 domain／type／連接／attribute 名稱及型別，以及遞迴子圖的算子統計。這是結構讀取，沒有執行 ONNX semantic checker、shape inference、YOLO 推論或數值比較。

| 實際查核欄位 | 結果與界線 |
| --- | --- |
| ONNX 結構 | IR version 9，opset `""` domain version 20；root graph 384 nodes、228 initializers、local functions 0。 |
| 實際輸入，order 0 | `images`，FLOAT，`[1,3,640,640]`。NCHW／RGB／0–1 是目標輸入宣告；shape 與 dtype 可由檔案確認，RGB／範圍仍需 producer preprocessing 規格佐證。 |
| 實際輸出，order 0 | `output0`，FLOAT，`[1,300,6]`。官方 [YOLO26 文件](https://docs.ultralytics.com/models/yolo26/) 說明 one-to-one 端到端輸出；不能將其當成未交付 Mamba head 的 cls／reg 輸出。 |
| 組件邊界 | 同一 ONNX 內含 backbone／neck／一般 YOLO detection head；目前沒有分開匯出的 backbone／neck 介面，未修改或切割圖。 |
| P3 候選內部值 | `/model.16/cv2/act/Mul_output_0`，FLOAT `[1,128,80,80]`。 |
| P4 候選內部值 | `/model.19/cv2/act/Mul_output_0`，FLOAT `[1,256,40,40]`。 |
| P5 候選內部值 | `/model.22/cv2/act/Mul_output_0`，FLOAT `[1,512,20,20]`。以上是檔案 `value_info` 與 node 連接中觀察到的值；不是可直接傳給新 head 的已交付公開 I/O。 |
| scan operator | node inventory 僅有標準空 domain 算子，沒有 `Scan`、`Loop` 或 custom-domain scan。結合上游一般 YOLO 架構，未找到目標 Mamba-FPN head／scan 契約；不從「沒有名為 Scan」推定任意圖的數學語義。 |
| 精度 | I/O 為 FLOAT；metadata half false。未查核每一數學運算的實際計算／累加精度、build tactic 或數值誤差。 |

[完整 inspector 輸出](../evidence/p01-a/2026-10-28/artifact/onnx-inspection.tsv) 保留實際名稱與順序。檔案含重複名稱的 `value_info` 紀錄；本次 inspector 逐項保留，未替上游修改，也沒有據此宣告 ONNX semantic validation 通過。

## 3. 其他上游調查與未採用原因

| 已查核來源 | 結果 |
| --- | --- |
| [Ultralytics model architecture](https://github.com/ultralytics/ultralytics/blob/bb3e1550dbb819bd9846d4c99f368aebe8792a0e/ultralytics/cfg/models/26/yolo26.yaml) | 於查核時固定 upstream main `bb3e1550dbb819bd9846d4c99f368aebe8792a0e`，架構使用一般 Conv／C3k2／C2PSA／Detect。此快照用於來源比對，不當作已下載 ONNX 的 exporter code revision。 |
| [MambaFPN 原論文](https://doi.org/10.1016/j.neunet.2026.108544) | 可確認同名 FPN 研究存在；出版摘要未交付本專案指定 YOLO26s＋Mamba-FPN 的 ONNX／權重／精確 scan 契約。此次未取得全文規格或可用完整 artifact，不能據名稱採用其公式。GitHub repository search `MambaFPN` 當次回應 0，屬有限查找結果，並非不存在其他發布的證明。 |
| [MambaPSA 作者 repository](https://github.com/henrychan0719/MambaPSA/tree/2940f1634cfce1f63201e6411ffe9008951461f3) 與 [作者論文](https://arxiv.org/abs/2607.12681v1) | 查核 README／論文摘要：主題為 YOLO26 的 C2PSA 替換及 neck BiViM 放置，示例為 nano 尺度、PyTorch／mamba-ssm 路徑。沒有據此取得與本專案要求一致的 small backbone／neck＋指定 head native bundle；未下載／移植其實作或權重，不拿它的 scan 代替未知 head。 |

## 4. 三層查核狀態與接續輸入

| 層級 | 此次狀態 |
| --- | --- |
| Manifest 宣告已驗證 | **PASS**：[實際候選 JSON](../evidence/p01-a/2026-10-28/artifact/yolo26s.candidate.json) 與 canonical SHA-256 `999f7a5ba64fa05d34d2e8e978a610e65c78142f9a1754a060dc8d7f9f3da63c` 已用 C++ CLI 驗證。role=`upstream_detector`，boundary incomplete，target head 缺失，scan unverified。CLI 的成功只表示語法及內部一致性。 |
| Artifact 檔案已驗證 | **PASS，限此官方一般 YOLO26s ONNX**：下載新檔、size 與 SHA-256 相符。 |
| 模型內容已核對 | **PARTIAL**：上表實際 I/O、graph inventory 與 metadata 已 native 讀取。未完成 preprocessing、完整 producer lineage、切分後介面、Mamba head 或必要 scan 語義核對。 |
| 目標真實 artifact readiness | **OPEN**：一般 YOLO26s ONNX 不滿足新 Mamba-FPN head 交付條件。Fixtures 只驗證工具，不能提高此狀態。 |

接續所需交付：

1. 新 YOLO26s backbone／neck ONNX（固定 `images` FP32 `[1,3,640,640]`，RGB CHW 0–1），以及明確匯出的 ordered P3／P4／P5 名稱、shape、dtype、layout；或直接交付完整合併 ONNX。不能只把此處的內部候選名稱當成已實現介面。
2. 與該 backbone／neck 匹配的新 Mamba-FPN head：ONNX＋必要 external tensor data，或文件化 C++ 可讀權重格式與架構；附來源 URL、不可變 revision、授權、產生／訓練 lineage、所有檔案大小與 SHA-256。單獨 pickle checkpoint 尚不足以滿足 native 入口。
3. Ordered cls／reg 輸入輸出與 decode 責任：分類類別定義、回歸編碼、尺度／stride、box／score／class 的確切語義，含候選數／容量與 padding 約定。
4. 與實際 head 逐節對上的 scan 身分及完整 attributes、輸入 shape／stride／broadcast、方向、空間展平順序、A／B／C／D 角色、delta bias／softplus 次序、離散化精確公式、gate、initial state、final state 輸出與每幀 reset；規格需有不可變來源與精確 locator。
5. 計算及累加精度、TF32／FP16／INT8 build 選項、獨立對照來源、誤差 metric 與預先固定的 atol／rtol。正式 engine／plugin／tactic 身分於產生後記錄 build evidence。

**S01 保持 OPEN。** 目前沒有已核實、屬於新 head 的 scan 精確契約，因此尚不能實作有模型依據的獨立 scalar reference 或數值 fixtures。收到第 4／5 項與對應 tensor 邊界後，再依 [MODEL_FEASIBILITY §3](MODEL_FEASIBILITY.md#3-s01s04-的接續條件) 開始；不自行選通用 Mamba recurrence。

## 5. 重跑

從 repository root 執行（需要 native C++20 compiler、curl、protobuf C++ development files、protoc、pkg-config）：

```sh
sh evidence/p01-a/2026-10-28/artifact/replay.sh
```

預設將重跑輸出放入 ignored `build-artifact-replay/evidence`，不覆蓋已提交 capture。每步 `.command`、`.exit`、`.log` 保存下載、大小／hash、schema 產生、C++ 編譯、完整結構查核及與原始輸出的逐 byte 比較。成功結論限於上述檔案的結構讀取；沒有執行真實 YOLO 推論或新 scan。

候選 JSON 的相對 artifact 路徑以 JSON 父目錄為基準。要驗證實際下載檔，先將候選宣告放入 cache 內相同目錄，再執行 CLI（或建立等價的獨立 bundle）：

```sh
cp evidence/p01-a/2026-10-28/artifact/yolo26s.candidate.json /home/ray/.cache/saccade-v1/p01-a/artifact/
build-p01-cpu/model_manifest verify /home/ray/.cache/saccade-v1/p01-a/artifact/yolo26s.candidate.json build-artifact-replay/yolo26s.canonical.json
sha256sum build-artifact-replay/yolo26s.canonical.json
```

[實際 CLI 輸出](../evidence/p01-a/2026-10-28/artifact/manifest-verify.log)／[exit code](../evidence/p01-a/2026-10-28/artifact/manifest-verify.exit) 保存 `manifest_declaration=verified`、`artifact_files=verified`、`model_content=not_checked`、`real_model_ready=false`。上面的人工外部結構查核不被 CLI 冒充為內容驗證。

## 6. 2026-10-29 追加來源查核與解除阻塞

本輪 fetch 後 `HEAD=origin/main=9f2c11a838a229c05ba7e9e17fa182bf0799746b`，起始工作樹乾淨，無後續 main 差異。這輪完成下列先前未做的查核；沒有重跑一般 YOLO26s 下載／inspection、舊模型或舊數值實驗。所有新增來源、原始命令／exit／輸出、日期／環境與 hashes 見 [本輪報告](../evidence/scan-contract/2026-10-29/REPORT.md)。

### 6.1 新增證據及來源身分

| 查找範圍 | 本輪新增核實 | 結果與界線 |
| --- | --- | --- |
| v1 repository 交付渠道 | [GitHub API captures](../evidence/scan-contract/2026-10-29/repository/README.md)：branches、releases、all-state PRs／issues | 僅 `main` 在上述 SHA；其餘三項皆空。未定位其他已發布 head 包；不代表沒有私人／未發布實作。 |
| DATAFLOW 明列的歷史架構規格 | 固定 `e48f3b39` 的 math_model §4.1 lines 397–464、SVG／授權文件；[design evidence](../evidence/scan-contract/2026-10-29/design/FINDINGS.md) | 新定位 down4 → two-block → up4、128 model size／16 state size、400/100/25 tokens、80 cls／4 reg／8400 raw anchors 及 decode 意圖。未提供 scan recurrence／方向／state ABI；只作架構依據，沒有讀取／復用舊程式或權重。 |
| MambaFPN DOI metadata／全文入口／作者／repository | [upstream evidence](../evidence/scan-contract/2026-10-29/upstream/FINDINGS.md)：Crossref、Elsevier API、PubMed／大學作者頁、arXiv title query、GitHub README 作者限定查詢 | Elsevier HTTP200 僅 coredata，全文未取得；出版搜尋預覽描述 neck，不能當目標 head。arXiv title query=0 屬有限負結果。新查詢找到 `raphaelkach/mamba-neck@6dec69675c362ef9837800ad084657850756e56b`：ResNet50＋ATSS、四入五出 neck／MMDetection，身分不同。README 6,830 bytes，SHA-256 `111addbc7dcd46ccb951d41255caf4fcc8b7255c069916a884e3d9b6e5a105aa`；tree 無 native bundle／LICENSE，API license=null。不以其四向 scan 取代本目標。 |
| MambaPSA 作者完整 tree／modules／YAML／dependencies／training args／論文 v1 | 固定 `2940f1634cfce1f63201e6411ffe9008951461f3`，current main 相同；[MambaPSA evidence](../evidence/scan-contract/2026-10-29/mambapsa/FINDINGS.md) | 主 YAML 改造 backbone、P4 變體改造 neck，都使用一般 Detect。small scale 確實存在，但 paper／交付 args 是 nano/VOC。13 個 `.pt` 已存在（未下載／載入，僅 tree 宣告），沒有所需 ONNX/native/export 交付。CPU 分支直接 bypass scan，不可作 oracle；沒有推定論文實測用了此分支。README AGPL 宣告與缺少 LICENSE 文本分開記錄。 |

取得檔案的 URL、固定 revision（有提供者）、size／SHA-256 與 lineage／授權欄位見各 evidence 的 FINDINGS 與 hash/size 清單。沒有取得的權重、exporter revision、訓練 lineage 或授權不補推。公開論文 DOI、Git blob SHA、package version 各有用途，不能互當新 head 的不可變 source revision。

### 6.2 精確缺件、持有人與解鎖步驟

本輪仍無目標新 head 的固定 source/config/artifact。**是否已實作但尚未交付、或尚未設計定案，目前未知**；已見候選不是同一目標，不能把身分差異稱為目標內部公式衝突。逐欄語義與待裁定決策見 [SCAN_CONTRACT](SCAN_CONTRACT.md)。

| 缺件／分類 | 可直接交付的材料與責任來源 | 阻塞範圍 → 解鎖後下一步 |
| --- | --- | --- |
| H01 目標身分未取得；是否存在未知 | v1 模型負責人提供新 head repository／不可變 revision／config／module map，確認是否採用已引用 D1 lane 設計；具名實作者／artifact custodian 未定位。 | S01、P01、S04 → 逐 module 對上 scan 與 component I/O。若尚無設計，裁定 SCAN_CONTRACT 的 D01–D04，明標新設計。 |
| H02 目標 scan 精確公式／ABI 未取得；設計決策未能由文件核實 | 模型／scan 設計者提供精確 recurrence/discretization、A/B/C/D（若有）、delta/bias/softplus/gate/skip、方向與 index/merge、ordered tensor/stride/broadcast/容量、initial/final/empty/reset、operator attributes 的固定規格與 locator。 | S01 → 按已核實範圍從零實作 scalar reference；不等待無關 GPU/plugin 工程。 |
| H03 目標數值比較政策缺失 | 模型／數值驗證者固定 compute/accumulation precision、build options、metric／有依據 atol/rtol、獨立 expected 推導方法。可在公式核實後手算；不強制先取得作者數值向量。 | S01 → 先固定政策，製作小尺寸非退化 fixtures／正反案例，CPU CTest＋ASan／UBSan。 |
| H04 新模型權重及 native 交付缺失 | 模型交付者提供新 ONNX＋external data，或明確 C++ 可讀 tensor bundle／架構。包括 matching backbone/neck ordered P3/P4/P5 或合併圖、URL/revision/授權/訓練與匯出 lineage/size/hash。不能交舊權重或只重發 pickle。 | P01／S04；S03 與實際 head 整合另需此材料，獨立 plugin 驗證不以完整權重為前提。不單獨阻塞 standalone scan → native inspection／manifest 檔案查核，再依相應階段驗收。 |
| H05 新 cls/reg/decode 契約缺失 | 模型／F04 負責人給出 ordered cls/reg 名稱、class list/ID、stride、box/score 編碼、decode/NMS owner、容量/padding/invalid mask，核對 D1 與新圖差異。 | P01、S04／P02 → 完整輸出對照；不單獨阻塞 S01。 |
| U01 論文全文／作者實作未取得，與 target 映射未知 | 若 v1 owner 確認指的是 Liang/Wang/Zhang 論文，可向通訊作者 Lefei Zhang（PubMed：`zhanglefei@whu.edu.cn`）索取固定作者實作/config／公式／授權／native 交付；出版預覽將 software contribution 列給 Cheng Wang。只定位取得渠道，未認定作者持有 v1 head，也未對外聯絡。 | 只有被 owner 確認選用後才影響 H01–H04；論文全文本身仍不等於新 artifact。 |
| U02 候選已有 PyTorch 實作／checkpoint，但 native 途徑與授權文本未交付 | 僅當 owner 明確選用 MambaPSA 時，向作者 repo 持有人／README 聯絡 Jen-Shiun Chiang 取得 matching small lineage、固定 dependency code、LICENSE 與 native bundle；先記錄對 B03 的架構影響。 | 不是目前 target readiness；不建議因有 checkpoint 就替代既定 head。 |

最直接的後續動作：**由 v1 模型負責人交付 H01 的固定 source/config/module map；若沒有現成設計，裁定 D01–D04。** 這一步能決定要向哪個實作者索取 H02/H03，而不是再重跑一般 YOLO 或舊權重。未知公式未填入 manifest 的 `specified` 欄位；S01、P01 與真實 artifact readiness 維持 OPEN。

## 7. 2026-10-29 原版 m head 的 artifact 對應

依使用者決策，v1 目標改為 YOLO26m backbone／neck＋原版 m T3→T1 Mamba-FPN head；§1–§6 為 s 時期的原始紀錄，結論維持其原適用範圍。本輪查核 [TRAINING_WORKFLOW](TRAINING_WORKFLOW.md) 定位的本機 artifacts（舊庫 `/home/ray/developer/ai/saccade`，source 快照 `e48f3b39`），只以本輪撰寫的 native C++ 讀取 pickle 容器與 ONNX protobuf；沒有執行 Python、舊程式或匯入 checkpoint 類別，大型檔案不放入 Git。命令、輸出與 hashes 見 [S01 evidence](../evidence/s01/2026-10-29/REPORT.md)。

### 7.1 檔案身分與 lineage

下列檔案都位於舊庫 gitignored 的 `runs/`、`models/yolo/`，沒有遠端 URL 或不可變 revision；舊庫 tracked 檔案為 Apache-2.0，這些產物本身沒有授權宣告，base `yolo26m.pt` 的上游授權本輪未查核。

| Artifact | bytes／SHA-256 | 本輪 native 讀取的內容 |
| --- | --- | --- |
| T1 主線 `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt` | 126,802,369／`a5a0e7091cd854ad68210ae4bbd43af73c7d3d098ea119f02ea65fab59c0676c` | archive `best`；epoch 15；parent `…_t3/best.ckpt`；`selection.status=diagnostic_train_loss`、holdout 空；108 個 student tensors；`mamba_args` 與形狀見 [SCAN_CONTRACT §1](SCAN_CONTRACT.md#1-目標與來源)。訓練 source revision 未記錄。 |
| GPU-cache 追加 `…_t3_t1_gpu_decode/best.ckpt` | 126,802,625／`477fe1ab41bd7948028fd6a84f7ef63f6f5af4a70849ff352199d7d82fe6f3f9` | epoch 15；parent 為 T1 主線；`mamba_args` 結構與 T1 相同，另有 `reg_max=1`、`head_depth=1`、`reduction_variant=conv`。 |
| 同 run `best_recall.ckpt` | 126,816,653／`dc186f546a9e33afe3566a07346f2cf5e42735fcac7274cb8f362355ac912626` | archive 名稱 `epoch_0011`、epoch 11；`selection` 欄位仍是 train-loss，以 recall 選取的程序未記錄於檔案。 |
| `models/yolo/mamba_head_26m.onnx` | 40,768,852／`7330db0c40a026064d65d1d60a219eb7f019952041bd3c5ade849ba19405b167` | producer pytorch 2.11.0；IR 8；opset `""` 17、`saccade` 1；494 nodes、65 個內嵌 initializer（無 external data）、3 個 `saccade::SelectiveScan`。**65/65 initializer 逐 byte 對上 `best_recall.ckpt`**，對 T1 主線只對上 17 個凍結 SSM 參數；與匯出腳本預設值一致。 |
| `models/yolo/mamba_head_26m.engine` | 22,549,364／`c0c0e3e1a99e1707996d876334af2217c9c725d03bb1fe6838073a47723806cf` | TensorRT plan，本輪未解析。build 腳本現行版本修改於 2026-09-27，晚於 engine 的 2026-07-07；實際 flags、TensorRT 版本與 plugin binary 未記錄。 |

### 7.2 對 m 目標的判定

| 層級 | 狀態 |
| --- | --- |
| 架構與 scan 契約 | **已核實**：T 的 `mamba_args`／state_dict 形狀與 O 的圖結構一致，S01 PASS（[MODEL_FEASIBILITY §3](MODEL_FEASIBILITY.md#3-s01s04-的接續條件)）。 |
| 權重的 native 交付 | **OPEN**：T1 主線只有 pickle checkpoint，不是 native 入口；O 雖可 native 讀取，但屬 `best_recall.ckpt`，不是 preset 主線；新訓練尚未啟動，既有 checkpoint 也未被採為 v1 權重。 |
| backbone／neck component I/O | **未查核**：舊庫有 `models/yolo/yolo26m_backbone_640.onnx`（75,900,061 bytes；舊 inventory 將其 initializers 歸屬 m teacher），本輪未 native 檢查其 ordered P3／P4／P5。 |
| engine／plugin | **OPEN**：bytes 未歸屬，不作 v1 驗收；v1 plugin 依 S03 從零實作並記錄 build evidence。 |
| 目標真實 artifact readiness | **OPEN**。manifest schema 1 的 `model_family` 只接受 s 字串（[MODEL_MANIFEST](MODEL_MANIFEST.md)），m 候選需 schema 升版。 |

### 7.3 缺件更新

§6.2 的 H01–H03 已由原版 m 實作解決；U01／U02 因 m 決策不再適用。

| 缺件 | 本輪狀態與所需材料 | 阻塞 → 解鎖後下一步 |
| --- | --- | --- |
| H01 目標身分 | **已解決**：原版 m head，module map 見 SCAN_CONTRACT §2。 | — |
| H02 scan 公式／ABI | **已解決**：SCAN_CONTRACT §3–§4。 | — → S02 |
| H03 數值比較 | **S01 已固定** binary64 reference 政策；FP32／FP16 kernel 對照容差未定。 | S02 → 依 kernel 實際運算訂定容差後再比較。 |
| H04 權重 native 交付 | **OPEN**：v1 選定權重的 head ONNX＋external data 或文件化 C++ 權重，以及匹配的 m backbone／neck（ordered P3 `[1,256,80,80]`、P4 `[1,512,40,40]`、P5 `[1,512,20,20]`），附 lineage、size、hash、授權。持有人為 v1 模型負責人（本 repository 擁有者）。 | P01／S04 → 依 TRAINING_WORKFLOW §5 固定新 run 後匯出；或明確指定沿用既有 checkpoint，再做 native inspection 與 manifest 檔案查核。 |
| H05 cls／reg／decode | **部分**：輸出順序、box／score 編碼已核實（SCAN_CONTRACT §6）；候選容量、padding、invalid mask、class tie 未定。 | P02／S04 → F04 規格。 |
| H06 engine 身分（新增） | **OPEN**：舊 engine 的 build revision、flags、TensorRT 與 plugin 身分未記錄。 | P01／S03 → 以 v1 自建 plugin 重建，不沿用舊 engine。 |
| H07 manifest schema（新增） | **OPEN**：schema 1 僅接受 s family。 | P01 → schema 2 加入 m family 與 scan 欄位後再宣告 m 候選。 |

## 8. 2026-10-30 S04：head 權重的 native 交付與完整 head

S04 依使用者指定，以 T1 主線 `best.ckpt`（§7.1）作為**工程驗證候選**，交付 native 權重與完整 FP32 head；**S04 PASS**，限指定候選、固定介面與受測 FP32 head 範圍，正式 clean-source 結果見 [S04 證據](../evidence/s04/2026-10-30/REPORT.md)，契約與開發紀錄見 [HEAD_TRT](HEAD_TRT.md)。head racecheck／synccheck 未執行；該限制與已執行的 memcheck／initcheck、S03 四種 sanitizer 分別記錄。這不是模型品質驗收：該 checkpoint 以 training loss 選取、無 holdout，未重訓。

| 缺件 | S04 後狀態 |
| --- | --- |
| H04 權重 native 交付 | **head 部分已交付（候選）**：native C++ 只讀 zip／pickle 資料轉出 `SCWB 1` bundle（40,711,552 bytes，SHA-256 `e1a8660fb9ad510f20ccc3d56a4feac61c6a2e5f771aab531caa741e7677f310`），69 個 active 參數逐 key 映射、39 個 temporal／flow-gate 參數排除並列理由；108 個 tensor 的 hash 與 §7.1 的 S01 稽核相同。matching backbone／neck 權重仍未交付。 |
| H06 engine 身分 | **head engine 由 v1 source 自建**：native network API、S03 plugins、FP32、builder optimization level 0、Winograd／FFT／低精度 tactic 在建置時拒絕；plan、SDK、tactics 與 hash 記於 build evidence。舊 `mamba_head_26m.engine` 仍不使用、仍未歸屬。 |
| H07 manifest schema | **已解決**：schema 2（[MODEL_MANIFEST](MODEL_MANIFEST.md#schema-2s04-head-only)）表達 head-only 邊界、ordered P3／P4／P5、六個 raw outputs、plugin statuses 與比較政策文件；S04 候選 [manifest](../manifests/s04/yolo26m-head-t1.json) 的宣告與檔案完整性已查核。 |
| backbone／neck component I/O | **ordered I/O 結構已查**：舊 `yolo26m_backbone_640.onnx`（75,900,061 bytes，SHA-256 `2154f937dec832dcefc4229a66dfc438c8fb99ee0c1547d80c00cc0143e56a6a`）輸入 `images [batch,3,640,640]`，輸出依序 `p3_feat [batch,256,80,80]`、`p4_feat [batch,512,40,40]`、`p5_feat [batch,512,20,20]`，形狀與 head 輸入相符。其權重歸屬（是否為 T1 訓練時的 teacher e12 features 來源）、native engine 與推論**未驗**。 |
| H05 decode | 未變：head 只輸出 raw cls／reg；候選容量、padding、invalid mask、class tie 仍屬 F04。 |

**S04 當時的目標真實 artifact readiness（P01）：仍 OPEN。** 該 head 驗收未涵蓋 matching backbone／neck 權重歸屬與 native engine、影像前處理、decode／NMS 與容量、新訓練或正式選模。後續進展分別見 §9 的 T01 與 §10 的 native F01／F02；訓練側輸入契約見 [T01_TRAINING §2.1](T01_TRAINING.md#21-輸入前處理契約-t01-input1)。

## 9. 2026-10-30 T01：matching backbone／neck 與完整模型

T01 重新訓練（[T01_TRAINING](T01_TRAINING.md)）取代 S04 的工程驗證候選：teacher e12、T1 epoch 1。依使用者核准，pinned SDK 追加同版本 ONNX parser（[NATIVE_TENSORRT](NATIVE_TENSORRT.md#onnx-parsert01-backbone)）。

| 缺件 | T01 後狀態 |
| --- | --- |
| H04 權重 native 交付 | **已交付（T01 候選）**：head bundle `4fccfda6…`（HEAD_TRT §10）；matching backbone／neck 為 T01 匯出的 ONNX `007c3260…`，440 個 initializer 逐 byte 對應 teacher e12 checkpoint，並與 PyTorch、ONNX Runtime 及訓練 cache 對照過 |
| H06 engine 身分 | backbone 與 F03 engine 由 v1 source 自建（`nvonnxparser`＋S04 network API），builder 設定、tactics 與 hash 記於 [F03 證據](../evidence/t01/2026-10-30/f03/REPORT.md)；舊 engine 仍不使用 |
| H07 manifest schema | schema 3（[MODEL_MANIFEST](MODEL_MANIFEST.md#schema-3t01-完整模型)）表達完整模型；[T01 manifest](../manifests/t01/yolo26m-t1.json) verified |
| backbone／neck component I/O 與推論 | **已驗**：ordered `p3_feat`／`p4_feat`／`p5_feat`；backbone engine 與 F03 engine 依 [BACKBONE_POLICY](BACKBONE_POLICY.md) 逐元素 PASS（[MODEL_TRT](MODEL_TRT.md)） |
| H05 decode | 未變：仍屬 F04 |

**目標真實 artifact readiness（P01）：仍 OPEN。** 此 engine 驗收不涵蓋 native 影像前處理（F01／F02，現況見 §10）、decode／NMS 與候選容量（F04），以及 P01 原列的其他缺件；head 與 F03 的 racecheck／synccheck 未執行。

## 10. Native F01／F02 後的狀態（2026-10-31）

Native 前處理已實作，詳見 [PREPROCESS](PREPROCESS.md) 與 [證據](../evidence/preprocess/2026-10-31/REPORT.md)。乾淨 source `2b39dda` 的 19/19 CTest 通過；7 張真實 JPEG 加 10 組合成輸入的 RGB／FP32 輸出與訓練參考逐位元相同，前三張亦與既有 F03 輸入 hashes 相同。F02 的四種 sanitizer 在受測範圍通過。

2026-10-31 使用者將 F01／F02 的使用結案目標定為正常輸出相同、重跑與重用正確，規則見 [使用驗收增補](PREPROCESS_USAGE_POLICY.md)；原數值門檻與固定 references 不變。**既知 nvJPEG initcheck FAIL 保留為診斷項，不再作為使用結案門檻，也不要求先取得 NVIDIA 分類。** 原始報告不改寫成 sanitizer PASS 或已確認誤報，API 錯誤、輸出不符與重用錯誤仍須拒絕。

**F01／F02 使用驗收 PASS（受測範圍）**：[正常使用證據](../evidence/preprocess/2026-10-31-usage/REPORT.md)沿用原 public runtime，以 7 張固定真實 JPEG、4 個 sessions 共 208 幀驗證串行、尺寸切換、延後 consumer、ring 重用與獨立雙 instance；全部 RGB／FP32 輸出逐位元符合固定 reference，重跑、mapping 與 checked close 通過。結果限受測硬體／SDK、固定輸入與呼叫合約。

F01／F02 尚未接入 F05／F03 的完整 async session，F04 decode／NMS 與容量仍待實作；P01 仍 OPEN。

## 11. F04 正式驗收後的狀態（2026-10-31）

**F04 clean-source 正式驗收 PASS（受測範圍），獨立結案**。乾淨 `995bca5` 的 [正式證據](../evidence/f04/2026-10-31-formal-995bca5/REPORT.md) 固定 source、實際執行 binary、sanitizer 與六份 F03 replay bytes；完整 `DetectionCandidates`、容量／失敗輸出、穩定順序、重跑／重用及獨立 API stream 契約依 [POSTPROCESS_POLICY](POSTPROCESS_POLICY.md) 驗證。四種 sanitizer 的 callback gates 使用 `--no-held-stream`，為 UNEXECUTED；正常 gates 已驗。

**P01 仍 OPEN**：完整 F03 producer 聚合器尚須彙整全部 plugin statuses 與 input／output finite checks；native F01／F02→F03→F04 的完整 session、CS ready／consumed、first-error latch、drain／checked close 仍待整合證據。Head／F03 racecheck、synccheck 的 UNEXECUTED 邊界不因 F04 通過而改變；模型品質與完整 tracking 亦不在本項範圍。後續依 [BACKLOG](BACKLOG.md) 接續。

## 12. F03 producer 聚合器（2026-10-31）

[`f03-producer/1`](MODEL_PRODUCER.md) 已實作全部 9 個 plugin statuses＋input／output finite checks 的 device 聚合與退休，發布 F04 已有的 aggregate ABI。[開發證據](../evidence/f03-producer/2026-10-31/REPORT.md) 列出獨立 API、六個固定輸入的 live engine→F04、重用／失敗／graph 及 sanitizer 的受測界線；dirty-source 開發驗證不稱 clean-source 正式結案。

**P01 仍 OPEN**：正常 native F01／F02→F03→F04 完整 session、CS ready／consumed、first-error latch、drain／checked close 與其他既列缺件仍待整合證據。Head／F03 racecheck／synccheck、模型品質及 tracking 指標不由本項推導。
