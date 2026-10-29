# P01-A 新模型 artifact 查核

查核日期：2026-10-28。來源、檔案與內容分別記錄。此次下載新上游檔案，沒有使用舊專案的程式、權重或數值結論，也沒有執行 Python。

2026-10-29 追加來源查核見 [§6](#6-2026-10-29-追加來源查核與解除阻塞)；前五節保留 P01-A 原始觀察，未重新下載既有 YOLO26s 或把工具 fixtures 升格為模型證據。

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
