# F04：native 偵測後處理

F04 已實作 C++ FP32 scalar reference 與 CUDA decode／篩選／NMS，將 F03 的六個 raw cls／ltrb outputs 與 `ImageMapping` 轉成完整 `DetectionCandidates`。**本機開發驗證通過；clean-source 正式驗收未執行**。驗證執行時 source 尚未提交、`source_dirty=true`，實際身分與結果見 [F04 證據](../evidence/f04/2026-10-31/REPORT.md)，不稱 clean-source formal F04 PASS。政策 [`f04-postprocess-fp32/1`](POSTPROCESS_POLICY.md) 及 reference／固定測例在首次 GPU 比較前固定。

## 1. 支援範圍與資料

固定 batch 1、640×640、FP32 NCHW，依 P3／P4／P5 讀取 cls `[1,80,80,80]`、`[1,80,40,40]`、`[1,80,20,20]` 及對應 reg `[1,4,H,W]`。共 8,400 個模型位置、80 類；每個位置先計算 80 類 sigmoid 的 max／argmax，取 conf≥0.001 的**全類別 top-300**，再保留 class 0 的非退化框。score 同分以原始 candidate ID 升序決勝，class 同分取最小 class ID。

Greedy NMS 使用未 clip 模型座標、IoU>0.5；被 suppressed 的框不再抑制後面的框。結果保留 kept 與 suppressed，維持 score／candidate ID 全序，再各除以 `float(mapping.scale_x/y)` 映射原圖。**不 clipping**，有限越界框保留；不從 top-300 之外補回被 class／退化篩選移除的名額。完整算術、invalid／overflow 處置與比較門檻見 [政策](POSTPROCESS_POLICY.md#2-fp32-算術與觀測次序)。

[`postprocess.hpp`](../include/saccade/postprocess.hpp) 是不依賴 CUDA 的公開契約，`ImageMapping` 共用型別在 [`image_mapping.hpp`](../include/saccade/image_mapping.hpp)。

| 型別／函式 | 契約 |
| --- | --- |
| `PostprocessRawView` | 每個 level 的 cls／reg pointer 與精確 element counts；reference 讀 host memory，CUDA 讀 device memory。 |
| `PostprocessRequest` | epoch、processing frame、640 stretch `ImageMapping`、capacity 0–300；原圖尺寸 1–65535，double scales 須 exact 等於 `640.0/original_dimension`。 |
| `PostprocessProducerStatus` | 8 bytes，`flags` 與 `done`；成功必須 flags=0、done=`kPostprocessProducerDone`。 |
| `DetectionCandidate` | 32 bytes；原圖 x1／y1／x2／y2、原始觀測 score、class ID、candidate ID、kept／suppressed flag。 |
| `DetectionCandidates` | **9,664 bytes**：64-byte header＋300×32-byte rows；header 含 epoch／frame、schema、count／capacity、status／done、kept／suppressed 與篩選統計。內部 device ABI，不能當作 ResultPacket wire bytes。 |
| `validate_postprocess_arguments` | 檢查固定 shape／counts、mapping、capacity、raw pointers／alignment，不解參考 tensor。 |
| `postprocess_reference` | CPU scalar FP32 完整結果；另以手算不變量核對排序、閾值、NMS 與映射。預設 CPU build 即可執行。 |

capacity 限制 kept＋suppressed 的完整 person pool；不足時整幀 overflow，不只交付 kept 或截斷 rows。數值／上游／protocol／overflow failure 的 count、統計與全部 rows 為 0，身份、schema、capacity 與明確 error 保留，最後寫完成 marker；合法空結果則 flags=0、count=0。未使用 rows 及 reserved bytes 永遠全 0。

## 2. CUDA API、初始化與所有權

低階介面在 [`postprocess_cuda.hpp`](../include/saccade/postprocess_cuda.hpp)：

```cpp
void postprocess_cuda(const PostprocessRawView& raw, const PostprocessRequest& request,
                      const PostprocessProducerStatus* producer,
                      DetectionCandidates* output, std::size_t output_bytes,
                      void* workspace, std::size_t workspace_bytes,
                      cudaStream_t stream);
```

每個 host submitter thread 須在 frame 工作開始前呼叫 `cudaSetDevice(device)`，在該 thread 建立／啟用正確的 CUDA primary context；不能只依另一個 thread 已初始化 device。呼叫者建立 explicit non-default stream、配置 raw tensors、producer status、output slot 與 workspace，並對每個 context 在 frame 工作開始前呼叫一次 `postprocess_cuda_load_kernels()` 完成 eager loading。F04 enqueue 不代 caller 初始化 thread context。`postprocess_workspace_bytes()` 在此版本為 **202,812 bytes**，至少 float alignment；`cudaMalloc` 滿足需求。`postprocess_cuda_identity()` 回報編譯 source／dirty、nvcc／flags／架構、CUDA runtime／driver、GPU 與 workspace 身分。

所有存取範圍必須位於 current device、互不重疊且有足夠實際 allocation 範圍。可使用 `cudaMalloc` allocation 或同一 arena 的互不重疊 subranges；以 `cuMemGetAddressRange` 查不到 extent 的 allocation 類型在 enqueue 前拒絕，不假定支援 memory pool／virtual allocation。output 與 workspace 的傳入 bytes 亦須足夠，所有 host argument 拒絕都發生在第一個 kernel 前。

初始化後 enqueue 只在指定 stream 排入 initialize、decode、stable rank、finalize 四個 kernels；不配置、不傳輸、不同步、不建立 event，也沒有中途 count／status D2H。rank 採 bounded exact-rank scan，沒有吞吐或低延遲承諾。固定 pointers／request 可被 CUDA graph capture／replay。

producer 是 caller-owned **唯讀**狀態。caller 必須在同 stream 排入本幀 raw／status producer，或以 producer-ready event 建立跨 stream 依賴；不能沿用前幀成功 marker。F04 不退休 producer、不鎖定整個 session 的首次失敗，也不代管下游 slot lease。raw、producer、output 與 workspace 必須活到其工作完成；交付 main stream 後，直到 F04b 最後讀取並發出 consumed，detect 才可覆寫該候選 slot。

**完整 F03 producer 聚合器尚未整合。** 它將來須彙整該 engine 的全部 9 個 plugin statuses 與 input／output finite checks，再寫 aggregate flags／done。F04 replay 測試由 caller 提供明確 status，不能視為已完成此聚合器或 native F01→F03→F04 接續。完整 ready／consumed、session failure latch、drain／checked close 仍依 [ASYNC_CONTROL](ASYNC_CONTROL.md) 接續實作；CUDA API／launch error 直接拋 runtime error，不保證輸出完成 marker。

## 3. 建置與重跑

預設 CPU build 不需要 CUDA、TensorRT、nvJPEG 或 Python，包含 reference 與 CPU tests：

```sh
cmake -S . -B build-f04-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-f04-cpu --target postprocess_tests --parallel 2
ctest --test-dir build-f04-cpu -R '^postprocess_reference$' --output-on-failure
```

CUDA F04 由 `SACCADE_ENABLE_CUDA=ON` 與 `SACCADE_ENABLE_POSTPROCESS=ON` 開啟，不需要 TensorRT 或 nvJPEG。以下 `120` 是目前本機 sm_120 的明確設定，其他 GPU 須指定自己的架構：

```sh
cmake -S . -B build-f04 -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-f04 --target postprocess_tests postprocess_cuda_tests --parallel 2
ctest --test-dir build-f04 -R '^postprocess_' --output-on-failure
```

測試工具另有以下入口：

```sh
build-f04/postprocess_cuda_tests --list-cases
build-f04/postprocess_cuda_tests --check-fixtures
build-f04/postprocess_cuda_tests --streams-only
CUDA_MODULE_LOADING=LAZY build-f04/postprocess_cuda_tests --streams-only
```

`--list-cases` 列出固定合成案例；`--check-fixtures` 以 CPU reference 檢查其手算不變量，兩者在初始化 CUDA device 前返回，但使用的是已建置的 CUDA-linked 測試工具。`--streams-only` 只跑 stream／ownership 測試，不代表完整數值案例通過。`--no-held-stream` 略過 host callback gate 的延遲測試，輸出會明列 held=UNEXECUTED；不能據此宣稱已驗證 pending producer／consumer 或 enqueue 在 held stream 下無同步。

F03 raw dumps 是可選的大型外部測試資料。指向含 `input-0.f32` 至 `input-5.f32` 的目錄時，工具先核對每份固定 2,822,400 bytes／SHA-256，再依正式 F03 file schema 讀取：**`cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5`**，先三份 cls、再三份 reg，不能交錯成各 level 的 cls／reg pairs。其 float element offsets 依序為 `0, 512000, 640000, 672000, 697600, 704000`，對應原始 [F03 payload 紀錄](../evidence/t01/2026-10-30/f03/capture/model-run-report.txt)。

```sh
build-f04/postprocess_cuda_tests --f03-dumps /absolute/path/to/model-dump
```

這是 [F03 固定六輸入](../evidence/t01/2026-10-30/f03/REPORT.md) 對應的錄製 raw outputs，不是重新執行 TensorRT 或 native 前處理。除了 CPU／CUDA 完整結果比較，工具亦獨立核對 [原 training-side diagnostic](../evidence/t01/2026-10-30/f03/capture/f03-diagnostics.json) 的六個 kept counts：`32, 28, 39, 2, 3, 1`，避免共同的 parser 錯誤讓兩份實作一起讀錯仍互相通過。若要把此項加入 CTest，configure 另給 `-DSACCADE_POSTPROCESS_F03_DUMPS=/absolute/path/to/model-dump`。沒有給路徑時不註冊 replay test，不把缺少 replay 說成已驗真實模型輸出。

## 4. 驗證重點與目前邊界

`postprocess_tests` 檢查 CPU 手算／排序／class tie、sigmoid 飽和、全類別 top-300、無回填、NMS 邊界／greedy chain、容量與數值／參數失敗。`postprocess_cuda_tests` 對固定 cases 比較完整 header／rows、score 界與座標 bits、重跑／graph、未用 bytes、allocation bounds／alias 拒絕；stream cases 包含 producer-ready／consumer-consumed、held enqueue、獨立 instances。

2026-10-31 的 [本機開發驗證](../evidence/f04/2026-10-31/REPORT.md) 結果如下；source、binary、library 與固定 inputs 以報告保存的 hashes 為準：

| 項目 | 實際結果與範圍 |
| --- | --- |
| 預設 Release CPU | 12/12 CTest；F04 reference 442 checks。 |
| F04 CUDA CTest | 4/4，含固定案例、lazy streams 與已註冊的 F03 replay。 |
| 完整 CUDA／replay 執行 | 31 個固定合成 cases、完整數值／順序／bytes、repeat／graph、argument 拒絕、正常執行下的 pending producer／consumer gates、held enqueue 與獨立 instances 均通過；六份 pinned F03 raw dumps 依正確 file schema 與獨立 historical kept counts 核對。共 13,983 checks，最大 score 絕對差約 1.19209×10⁻⁷，小於固定界 2⁻²¹。 |
| F04 sanitizers | memcheck／initcheck／racecheck／synccheck 各執行 13,678 checks 與六份正確解析的 dumps，0 errors／hazards。各工具使用 `--no-held-stream`，host callback 延遲 gates 在 sanitizer 下為 **UNEXECUTED**；不將正常執行的 gates 結果混列為 sanitizer 覆蓋。 |
| 正控制與語義變體 | 四個 sanitizer positive controls 各以預定診斷及 exit 99 偵測；10/10 mutations 以預定完整診斷及 exit 1 偵測。 |
| 正式驗收 | **UNEXECUTED**：該次 capture 使用尚未提交、`source_dirty=true` 的開發 source，沒有 clean-source formal F04 capture。 |

正確解析後的六份 F03 replay 結果如下；kept 亦與上述獨立歷史資料一致。早期誤用交錯 file layout 的 capture 已作錯誤 attempt 保留，不用其 output hashes／counts 支持此結果。

| 固定 input | 完整 rows | Kept | Suppressed |
| --- | ---: | ---: | ---: |
| MOT17-02-SDP/1 | 237 | 32 | 205 |
| MOT17-10-SDP/1 | 246 | 28 | 218 |
| MOT17-13-SDP/400 | 285 | 39 | 246 |
| zeros | 7 | 2 | 5 |
| ones | 5 | 3 | 2 |
| random/1 | 3 | 1 | 2 |

上述驗證限本機受測 GPU／CUDA 與固定 inputs，不繼承為其他環境的結果。F04 CPU ASan／UBSan 單項 1/1 通過；額外全 CPU Debug suite 中途停止，未列全套 ASan PASS。前處理 header 相容性 rebuild 與指定測試 2/2 通過，不重分類既知 nvJPEG initcheck 診斷。實際執行與開發失敗紀錄以原始報告為準。

此實作的範圍是 raw predictions→`DetectionCandidates`；不包含 F03 producer 聚合器、native 前處理與模型接續、F04b 續追、GMC、tracker、ResultPacket 或 MOT writer。**完整 async session 與 P01 仍 OPEN**；T01 實驗不因此重開，也不由 F04 數值或執行驗證推導新的模型品質成績。
