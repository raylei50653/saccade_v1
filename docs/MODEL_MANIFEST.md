# 模型 manifest：schema 1（P01-A）、schema 2（S04 head-only）與 schema 3（T01 完整模型）

本工具驗證模型**輸入宣告**及 artifact 檔案完整性；模型圖結構與 scan 公式的實際核對另存 evidence。P08 的完整設定系統仍依原規劃接續。格式選定為 **ASCII JSON profile**；schema 1 範例見 [parser fixture](../fixtures/model-manifest/contract.json)，schema 2 見 [head fixture](../fixtures/model-manifest/head-contract.json) 與 [S04 候選](../manifests/s04/yolo26m-head-t1.json)，schema 3 見 [model fixture](../fixtures/model-manifest/model-contract.json) 與 [T01 候選](../manifests/t01/yolo26m-t1.json)。不使用 Python、外部 JSON 套件或舊模型程式。

`schema` 欄位決定整份文件依哪個版本解析：`2` 走 §Schema 2，`3` 走 §Schema 3，其餘依 schema 1 規則（因此 `4` 等值以 `$.schema: expected 1` 拒絕；2026-10-30 加入 schema 3 之前，`3` 也是如此）。**schema 1 與 schema 2 的欄位、規則、錯誤訊息、canonical bytes 與既有 fixture hash 不變**；一份文件改標成另一個 schema 會因欄位不符被拒絕，不會被重新解讀。CLI 輸出第一行為實際的 `schema=1`、`2` 或 `3`。

## Schema 2（S04 head-only）

用於 [HEAD_TRT](HEAD_TRT.md) 的交付：T=1 YOLO26m Mamba-FPN head，以 P3／P4／P5 features 為輸入、六個 raw outputs 為輸出。沒有影像輸入 profile，也不宣告 backbone artifact。共用 schema 1 的 JSON 子集、canonical 編碼、artifact 物件（`id`…`sha256`）、相對路徑與 `verify` 的檔案完整性規則。

| object | 欄位與限制 |
| --- | --- |
| root | `schema` 2；`purpose` `test_fixture`／`model_candidate`；`model_family` 只接受 `YOLO26m-Mamba-FPN-head-T1`；`artifacts`、`boundary`、`tensors`、`statuses`、`frames`、`scan`、`precision`、`content_review`，全部必填、無其他欄位。 |
| artifact.role／format | role：`mamba_fpn_head`、`checkpoint`、`specification`、`fixture`；format：`native_weights`、`checkpoint`、`text`、`fixture_bytes`。candidate 不得含 fixture。 |
| boundary | `scope` 只能是 `head_only`；`head` 必須指向 role `mamba_fpn_head`＋format `native_weights` 的 artifact；`source_checkpoint` 指向 `checkpoint`＋`checkpoint`（candidate 必填，fixture 可為 null）；**`backbone_neck` 必須為 null**（不虛填不存在的 backbone）；classification／regression owner＝`head`，decode owner＝`external`。test_fixture 可用 fixture artifact 代替。 |
| tensors | 恰為九個、固定次序：輸入 `p3 [1,256,80,80]`、`p4 [1,512,40,40]`、`p5 [1,512,20,20]`（order 0..2），輸出 `cls_p3`、`cls_p4`、`cls_p5`、`reg_p3`、`reg_p4`、`reg_p5`（order 0..5，`[1,80,H,W]`／`[1,4,H,W]`）；component `head`、dtype `float32`、layout `NCHW`；`semantic` 非空。 |
| statuses | 非空；每個 `{name, producer, dtype:int32, shape:[2]}`；名稱不可與 tensors 或彼此重複。數量由實際圖決定，schema 不預設。 |
| frames | `batch` 1、`temporal_length` 1、`state_lifetime` `per_frame_reset`。 |
| scan | `status` `specified`；`contract`、`formula_source`／`revision`／`locator`、`semantics` 非空；`plugins` 恰為 `CrossScanExpand` v1（無 fields）、`SelectiveScan` **v2**（`has_d`）、`CrossScanMerge` v1（`height`、`width`），namespace `saccade`，fields 型別 `int32`，`status_output` 為 `int32[2] {flags, done}`。SelectiveScan v1 以「v1 has no status output」拒絕。 |
| precision | `compute_dtype`、`accumulation_dtype` 皆 `float32`；`tf32`、`fp16`、`int8` 必須為 false；`builder_options` 同 schema 1（不放 engine／tactic 身分）；`comparison` 為 `{status: specified, policy, policy_artifact, reference_source, metric}`，`policy_artifact` 必須指向 role `specification`＋format `text` 的 artifact，`verify` 因此一併核對政策文件 bytes。 |
| content_review | 同 schema 1。 |

S04 候選 [`manifests/s04/yolo26m-head-t1.json`](../manifests/s04/yolo26m-head-t1.json) 宣告 checkpoint、native bundle 與 [HEAD_POLICY](HEAD_POLICY.md)（`s04-head-fp32/1`）三個 artifact 與 9 個 plugin statuses。artifact 路徑相對於 manifest 所在目錄；大型檔案不入 Git，所以 `verify` 在 cache staging 目錄執行（manifest 與政策文件複製進去，checkpoint 以既有 symlink 指向舊庫檔案）。`validate`／`verify` 仍回報 `model_content=not_checked`、`real_model_ready=false`：宣告、檔案完整性、內容核對（evidence 的轉換報告）與實際推論結果（HEAD_TRT 驗收）分開記錄；engine／plugin binary hash、SDK、tactics 與實際 builder 設定屬 build evidence。

## Schema 3（T01 完整模型）

用於 [MODEL_TRT](MODEL_TRT.md) 的交付：YOLO26m backbone／neck ONNX 接 schema 2 的同一個 head，影像輸入、六個 raw outputs。實作與 schema 2 共用同一個 parser（`parse_head_family`），只在下列欄位不同；其餘（statuses、frames、scan、content_review、artifact 物件與 candidate 不得含 fixture）與 schema 2 相同。

| object | 與 schema 2 的差異 |
| --- | --- |
| root | `schema` 3；`model_family` 只接受 `YOLO26m-Mamba-FPN-T1`；多一個必填的 `input_profile`。 |
| artifact.role／format | role 另加 `backbone_neck`；format 另加 `onnx`。 |
| boundary | `scope` 只能是 `complete_model`；`backbone_neck` 必須指向 `backbone_neck`＋`onnx`，`backbone_source_checkpoint` 指向其來源 checkpoint（candidate 必填）；`head`、`source_checkpoint` 與 owner 同 schema 2。 |
| input_profile | 同 schema 1 的 `tensor`（必須是 `images`）、`batch` 1、`color` `RGB`、`layout` `CHW`、`dtype` `float32`、`range_min` 0、`range_max` 1、`width`／`height` 640，另加 `resize` `stretch` 與非空的 `preprocessing`（前處理契約的文字定位）。 |
| tensors | 恰為十個、固定次序：輸入 `images [1,3,640,640]`（component `model`，order 0）、intermediate `p3_feat`、`p4_feat`、`p5_feat`（component `backbone`，order 0..2，形狀同 schema 2 的 p3..p5）、六個輸出（component `head`，同 schema 2）。 |
| precision | tf32／fp16／int8 為 true 時的訊息為「the F03 model is an FP32 path」；`comparison` 另需 `head_policy_artifact`（`specification`＋`text`）：`policy_artifact` 是 backbone／F03 政策，`head_policy_artifact` 是 head 的比較政策，`verify` 兩份文件都核對 bytes。 |

T01 候選 [`manifests/t01/yolo26m-t1.json`](../manifests/t01/yolo26m-t1.json) 宣告 teacher e12 checkpoint、backbone ONNX、T1 checkpoint、T1 head bundle、[BACKBONE_POLICY](BACKBONE_POLICY.md) 與 [HEAD_POLICY](HEAD_POLICY.md) 六個 artifact；`verify` 同樣在 cache staging 目錄執行，仍回報 `model_content=not_checked`、`real_model_ready=false`。

---

以下為 schema 1（P01-A）原文。

## 執行與三種查核狀態

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
build/model_manifest validate fixtures/model-manifest/contract.json build/manifest.canonical.json
build/model_manifest verify fixtures/model-manifest/contract.json
```

`validate` 完成語法、型別、必要欄位及內部一致性查核，回報 `manifest_declaration=verified`、`artifact_files=not_checked`。`verify` 在相同查核後讀取**全部** artifact，核對 size 與 SHA-256，成功才回報 `artifact_files=verified`。兩者都回報 `model_content=not_checked`、`real_model_ready=false`；外部內容核對文件須分別指明哪個欄位與原始檔案相符，工具不把宣告中的 `content_review` 視為獨立查核結果。

選用的第三個參數是 canonical 輸出檔案；只有完整要求的查核成功後才寫入。工具使用既有 checked I/O，檢查讀取、寫入及明確 close，失敗 exit 2；stdout 寫入失敗也 exit 2。拒絕輸出覆寫 manifest 或 artifact 的同路徑及既有 hardlink／symlink 別名；不承諾跨程序競態保護或 fsync 耐久性。輸出父目錄需事先存在。

Artifact `path` 必須是相對路徑，禁止 `..` 與反斜線；以 **manifest 所在目錄**為基準。CLI 先將 manifest 路徑轉為 absolute，再查核所有檔案，不以 artifact 的執行時 cwd 為基準。manifest relocation 改變 bundle 基準，但不改變 canonical 宣告 hash。允許既有 symlink 指向檔案；查核的是讀取到的 bytes，沒有 sandbox 或 confinement 承諾。檔案須是 regular file；讀取前 size、讀取到的 size 與 hash 都需相符。

## 格式與穩定編碼

所有 object 的欄位均為下列封閉集合；**每一欄必填、無 defaults**，未知／重複欄位一律拒絕，包括巢狀 object 與經 escape 解碼後重複的 key。array 順序是契約的一部分；所有 object key 以 ASCII 字典序輸出，無縮排或空白，尾端一個 LF。SHA-256 包含該 LF。空白、CRLF 與 object 欄位排列不同、等價 ASCII string escapes 不影響 canonical bytes/hash。Array 重新排序會改變契約。

支援 JSON object、array、string、boolean、null 與 uint64 integer。數字不支援負數、小數、exponent 或 leading zero；precision 的有限小數使用下述明確字串型別。字串限 ASCII；接受標準 JSON escapes 及 `\u0000`–`\u007e`，canonical 會統一 escape；實際 schema 字串內容必須是 printable ASCII。網址可使用 percent encoding。非 ASCII 來源標籤需在 schema 升版時定義 UTF-8 normalization。最大巢狀深度 64；不接受 comments、trailing comma 或 trailing data。

所有 dtype／layout／enum 值區分大小寫。未驗證資料明寫 `unverified`、空列表或指定的 null，不能省略欄位。這些值有效代表「知道尚未查核」，不代表模型可建置。

## 必填欄位

| object | 欄位與型別／限制 |
| --- | --- |
| root | `schema`: integer 1；`purpose`: `test_fixture` / `model_candidate`；`model_family`: `YOLO26s-backbone-neck+Mamba-FPN-head`（schema 1 制定時的 s 目標家族，不證明檔案已包含 head；v1 已改為 m，m 候選需 schema 升版，見 [readiness §7](MODEL_ARTIFACT_READINESS.md#7-2026-10-29-原版-m-head-的-artifact-對應)）；`artifacts`: 非空 array；`boundary`, `input_profile`, `scan`, `precision`, `content_review`: object；`tensors`: 非空 array。 |
| artifact | `id`: 唯一非空 string；`role`: `backbone_neck` / `mamba_fpn_head` / `combined_model` / `upstream_detector` / `checkpoint` / `specification` / `fixture`；`path`: 相對 string；`format`: `onnx` / `checkpoint` / `native_weights` / `text` / `fixture_bytes`；`source`, `revision`, `license`, `generated_by`: 非空 string；`size_bytes`: uint64；`sha256`: 64 lowercase hex。`upstream_detector` 明示一般上游 detector，不能映射目標 Mamba head。 |
| boundary | `packaging`: `merged` / `separate` / `incomplete`；`components`: 恰含 backbone／neck／head 三個唯一 role；每個 component 只有 `role` 與 `artifact`（artifact id 或 null）。`classification_owner`, `regression_owner`, `decode_owner`: `head` / `external` / `unverified`。 |
| tensor | `name`: 全域唯一非空 string；`order`: 同 component/direction 從 0 起連續，需與 array 順序一致；`direction`: `input` / `output` / `intermediate`；`component`: `model` / `backbone` / `neck` / `head` / `scan`；`shape`: 1–8 個正 integer；`dtype`: `float32` / `float16` / `int32` / `int64` / `bool`；`layout`: `NCHW` / `NC` / `NCL` / `NLC` / `linear`；`semantic`: 非空 string，記錄 cls／reg／decode 等語義。 |
| input_profile | `tensor`: 唯一 model/input tensor 名；`batch`: 1；`color`: `RGB`；`layout`: `CHW`；`dtype`: `float32`；`range_min`: 0；`range_max`: 1；`width`, `height`: 640。以上數字為 integer。 |
| scan | `status`: `unverified` / `specified`；`operator`: object；`attributes_status`: `unverified` / `complete`；`attributes`, `inputs`, `outputs`: array；`direction`, `flatten_order`, `initial_state`, `formula_source`, `formula_revision`, `formula_locator`, `semantics`: 非空 string；`state_lifetime`: `per_frame_reset`。`semantics` 記錄 A/B/C/D 角色、delta bias/softplus 先後、離散化、gate、final state 等精確語義，`formula_*` 必須定位原始規格，不採通用 Mamba 猜測。 |
| scan.operator | `domain`, `plugin_namespace`: string（允許空字串）；`op_type`, `plugin_name`, `plugin_version`: 非空 string；`opset`: uint64。未核實時明寫 `unverified`、opset 0。plugin 尚未選定可記錄 `not_applicable` 並在 evidence 說明；這不是 plugin 二進位驗證。 |
| scan.attribute | `name`: 唯一非空 string；`type`: `unsigned_integer` / `boolean` / `string` / `decimal`；`value`: 對應 integer / boolean / string / canonical 非負 finite decimal string。未知 attribute 型別不能降成字串冒充完整契約，需先擴充並版本化 schema。 |
| scan.input | `tensor`: 已宣告 tensor 名；`role`, `stride`, `broadcast`: 非空 string，明列精確規格。shape/dtype/layout 由 tensor 宣告引用。`outputs` 為已宣告 tensor 名稱 array，需明列 final state（若存在）。 |
| precision | `compute_dtype`: `float32` / `float16`；`accumulation_dtype`: `float32` / `float64` / `unverified`；`tf32`, `fp16`, `int8`: boolean；`builder_options`: array；`comparison`: object。float16 compute 要求 fp16=true；schema 1 尚未定義 quantization，int8 必須 false。 |
| precision.builder_option | 只有 `name`, `value`: 非空 string；name 不可重複。用於明列擬採用的輸入 build 選項，不能放實際 engine／plugin／tactic 身分。 |
| precision.comparison | `status`: `unverified` / `specified`；`reference_source`, `metric`: 非空 string；`atol`, `rtol`: canonical 非負 finite decimal string，如 `0`、`0.00001`。不接受符號、exponent、leading integer zero 或 trailing fractional zero（故 `0.0` 拒絕）。比較結果另存 evidence；status=unverified 的數字不表示已有驗收門檻。 |
| content_review | `status`: `not_reviewed` / `external_review`；`method`, `evidence_source`: 非空 string；`unverified_fields`: 非空 string array；external_review 時可為空 array。這只是可追蹤的宣告，不是 CLI 檢查過圖的結果。fixture 禁止 external_review。 |

`incomplete` 必須且僅在至少一個 component 的 artifact 為 null 時使用；`merged` 三者需映射同一 artifact；`separate` 需多檔。model component 不能映射 specification/text；backbone_neck 不能映射 head，mamba_fpn_head 不能映射 backbone／neck。fixture 只可在 purpose=test_fixture 下使用，不能把 fixture role／format 改成 model_candidate。上游完整 detector 即使包含一般 head，也不因此成為目標 Mamba head。

Named tensor layout 需符合 rank：NCHW=4、NC=2、NCL/NLC=3、linear=1；每維 ≤ INT64_MAX，完整維度乘積與 dtype byte size 逐步檢查 uint64 overflow，結果亦需可由 size_t 表示。固定 model/input 唯一且必須 float32 NCHW `[1,3,640,640]`。其他 tensors 的尺寸只是宣告，須由實際 artifact／規格核對，不預設 P3/P4/P5 常數。

scan.status=specified 時 inputs/outputs 不可空、attributes_status=complete、opset>0，算子身分與公式欄位不可仍是 `unverified`。這只證明宣告結構完整；沒有公式比對或 scan 執行。只有實際內容查核後，S01 才能以該公式實作 scalar reference。

## 輸入契約與 build evidence

首次驗證不要求 engine、plugin binary、tactic、timing cache 或 GPU 存在。這些欄位若放入 manifest 均因未知欄位拒絕。實際 engine hash、SDK/header/loaded library、CUDA/driver/compiler/GPU、build commit、runtime 選項及執行比較結果，應在產生後寫入 build evidence，並引用 canonical manifest hash。這避免 build 前 manifest 依賴尚未產生的輸出。

共用 SHA-256 從原 synthetic config 實作原樣抽出；synthetic 設定編碼及兩種 hash 的既有行為保持不變。Manifest 採自己的 JSON canonicalization，不改 synthetic key=value profile。

## 測試界線

`model_manifest_tests` 檢查合法 fixture、idempotence、無語義 JSON 排列與 escape 正規化、獨立 `sha256sum` fixture hash、schema 各層拒絕、shape/product/byte overflow、錯誤路徑與實際檔案 size/hash/read failures。Linux 以 `/proc/self/mem` 的 offset-zero EIO 驗證 checked read，避免 root 可越過權限的測例失真。

`model_manifest_cli` 以 CMake 獨立 SHA256 核對 canonical 檔案，從不同 cwd 執行相對 artifact 查核，檢查成功時完整 stdout、缺檔／目錄／壞 size/hash、輸入別名覆寫、寫入開啟失敗及 Linux `/dev/full` 的 canonical/stdout 失敗。Fixture 是文字 `abc`，沒有 ONNX 圖或真實模型權重；完整性通過完全不代表真實模型 readiness。
