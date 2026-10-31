# Saccade v1

Saccade v1 以 C++／CUDA 重新設計 GPU 追蹤資料流。目前包含架構、可重跑的 C++ 協定驗證、合成資料的 CUDA serial／pipelined 執行骨架，以及原版 YOLO26m head scan 的 CPU scalar reference（S01）、CUDA FP32 forward（S02）、TensorRT plugins／frame-end harness（S03）、完整 T=1 head（S04）、T01 backbone／neck 與完整模型推論（F03）。native 影像前處理（F01／F02：nvJPEG hardware decode＋CUDA antialiased resize）已實作；依 [使用驗收增補](docs/PREPROCESS_USAGE_POLICY.md)，**F01／F02 使用驗收 PASS（受測範圍）**，正常輸出、重跑與資源重用結果見 [使用證據](evidence/preprocess/2026-10-31-usage/REPORT.md)。既知 nvJPEG initcheck FAIL 保留為診斷項，不再作為使用結案門檻；decode／NMS（F04）、完整 GMC、tracker 與 MOT writer 仍待各自任務實作。

以舊架構的 YOLO26＋Mamba-FPN、GPU GMC、geometry-first／ReID-free tracker 與 MOT 輸出作為設計基線；重新抽象資料與責任邊界，全部重新實作。

已確定的範圍：從零實作、不復用舊程式、不使用 Python、不遷移舊實驗數據，也不要求保證舊實驗成績。唯一例外是模型訓練（[T01](docs/T01_TRAINING.md)）：訓練以 PyTorch 執行舊庫固定 revision 的訓練程式加上記錄的 patch，產物交給 C++／CUDA／TensorRT 推論，runtime 不依賴 Python。

- [資料流與架構規劃](docs/DATAFLOW.md)：邏輯資料流、資料契約、跨幀狀態、GPU 所有權、同步依賴、結束流程及待定設計；§4.4 規定設定治理，§8.1 規定行為驗證。
- [非同步狀態流程控制](docs/ASYNC_CONTROL.md)：跨 stream 與 host↔device 的 event 協定、device 端提交、失敗收尾與競態驗證。
- [ResultPacket 與失敗狀態模型](docs/RESULT_PACKET.md)：固定容量布局、穩定編碼、初始化與完整封包比較工具。
- [可重跑的協定模型](docs/PROTOCOL_MODEL.md)：216 組 happens-before 檢查與同步邊變異測試。
- [合成資料 CUDA 執行骨架](docs/CUDA_SYNTHETIC.md)：可選建置、typed resolved 設定、真實 streams／events／cuFFT、原始 ResultPacket 輸出、重跑命令與實機證據。
- [模型 artifact／native scan 可行性](docs/MODEL_FEASIBILITY.md)：P01-A 的三項獨立驗收與 S01–S04 的接續條件與判定。
- [模型 manifest schema 與 C++ CLI](docs/MODEL_MANIFEST.md)：嚴格 JSON、穩定序列化／hash、artifact 檔案完整性及 fixtures；schema 2 表達 S04 的 head-only 邊界、ordered P3／P4／P5、六個 raw outputs、plugin statuses 與比較政策；schema 3 表達 T01 的完整模型（backbone ONNX＋head bundle、影像輸入 profile、兩份政策）。
- [Native TensorRT SDK 最小驗證](docs/NATIVE_TENSORRT.md)：獨立可選 SDK 目標、版本 pin、固定小網路與重跑命令。
- [模型來源與缺件](docs/MODEL_ARTIFACT_READINESS.md)：s 時期的官方一般 YOLO26s native 查核；§7 為原版 m head 的 checkpoint／ONNX／engine 對應與剩餘缺件；§8 為 S04 後的狀態；§9 為 T01（matching backbone 與完整模型）後的狀態；§10 為 native F01／F02 的使用驗收與診斷界線。
- [原版 YOLO26m head 的 scan 契約](docs/SCAN_CONTRACT.md)：v1 模型基線（YOLO26m＋原版 m T3→T1 head、T=1）的 module map、`saccade::SelectiveScan`／Cross-Scan 契約、比較政策與剩餘缺件；S01 scalar reference 依此實作。
- [S02 CUDA FP32 scan](docs/SCAN_CUDA.md)：`selective_scan_cuda` 與 Cross-Scan 展開／合併的 stream／status 契約、FP32 算法（穩定 softplus、精確指數的 drive、部分和縮放重算）、支援限制，以及首次 GPU 執行前固定的逐元素 running error bound 比較政策。
- [S03 TensorRT FP32 scan](docs/SCAN_TRT.md)：SelectiveScan v2／Cross-Scan plugins、INT32[2] status output、CUDA context 初始化、最小 engine 與 frame-end gate／publish／commit／ResultPacket 交接；驗收範圍與證據見該文件。
- [S04 完整 head（TensorRT FP32）](docs/HEAD_TRT.md)：指定 T1 checkpoint 的 native 轉換與 `SCWB 1` bundle、native network API 建圖、六個 raw cls／reg 輸出與 9 個 plugin status＋輸入／輸出有限值檢查共同決定 frame 發布、生命週期與重跑命令；數值政策在首次 GPU 比較前固定於 [HEAD_POLICY](docs/HEAD_POLICY.md)。**S04 PASS**，限指定候選、固定介面與受測 FP32 head 範圍；正式結果見 [S04 證據](evidence/s04/2026-10-30/REPORT.md)。head racecheck／synccheck 未執行；該 head 驗收不涵蓋 backbone／neck、前處理、decode 或模型品質，各自依所屬階段判定；P01 仍 OPEN。
- [舊版 m 模型訓練規格與流程](docs/TRAINING_WORKFLOW.md)：以 preset 原版 YOLO26m T3→T1 為主，整理 teacher／cache／distill／GT 交接、微調設定與缺件；僅整理文件，未搬移訓練實作或模型產物。
- [T01 backbone engine 與 F03](docs/MODEL_TRT.md)：T01 backbone ONNX 經 pinned SDK 的 ONNX parser 建成 FP32 engine，與 S04 head 組成完整模型 engine；政策在首次 GPU 比較前固定於 [BACKBONE_POLICY](docs/BACKBONE_POLICY.md)。**backbone PASS、F03 PASS**，限六個固定輸入與受測環境；該 engine 驗收不涵蓋 native 前處理或 decode／NMS；P01 仍 OPEN。
- [Native 影像前處理 F01／F02](docs/PREPROCESS.md)：獨立於 TensorRT 的 nvJPEG hardware batch-1 planar RGB decode、FP32 正規化與 antialiased stretch、ImageMapping、同 stream 重用與 checked drain；使用 `SACCADE_ENABLE_PREPROCESS` 可選建置。數值仍依 [PREPROCESS_POLICY](docs/PREPROCESS_POLICY.md)，既有 17 組對照逐位元相同、19/19 CTest；依 [PREPROCESS_USAGE_POLICY](docs/PREPROCESS_USAGE_POLICY.md) 的正常使用驗收亦 **PASS（受測範圍）**：7 張固定真實 JPEG、4 個 sessions 共 208 幀，全部 RGB／FP32 輸出逐位元對齊，涵蓋尺寸切換、延後 consumer、ring 重用與雙 instance（[使用證據](evidence/preprocess/2026-10-31-usage/REPORT.md)）。既知 nvJPEG initcheck FAIL 保留為診斷項，原始 [證據](evidence/preprocess/2026-10-31/REPORT.md)不改寫；單 staging 尚未整合雙 STAGE 的完整 async session。
- [T01 重新訓練](docs/T01_TRAINING.md)：YOLO26m＋Mamba-FPN 全鏈重訓的事前規格（來源與 patch、sequence-level split、nvJPEG GPU decode＋float antialiased stretch 的輸入契約、配方、val AP50 選模政策、預算）與執行紀錄；重跑入口在 [training/t01](training/t01/)。

使用 CMake ≥ 3.20 與支援 C++20 的 compiler，不需要 CUDA、TensorRT 或 Python：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`build/protocol_model --self-test` 檢查同步協定；`build/protocol_tests` 驗證首幀初始化、錯誤注入、失敗鎖定、頻譜隔離與封包編碼。`build/packet_compare serial.bin pipelined.bin` 比較整份固定容量封包串流，包含未使用 rows 與 reserved bytes。

`synthetic_config_tests` 另外驗證最小設定 schema、非法輸入、穩定序列化與 hash；`file_io_tests` 驗證讀取及明確 close 的錯誤傳播，`cleanup_tests` 以 CPU fake 驗證資源釋放錯誤與所有權政策。`model_manifest_tests` 與 CLI 測試加入模型輸入契約及失敗條件；`selective_scan_tests` 以 bc 推導、手算與契約等式驗證 S01 的 binary64 scan reference 及其拒絕條件（[fixtures](fixtures/selective-scan/README.md)）。預設建置不需 GPU。

```sh
build/model_manifest validate fixtures/model-manifest/contract.json
build/model_manifest verify fixtures/model-manifest/contract.json build/canonical-manifest.json
```

這份 fixture 只驗證工具；CLI 將宣告、檔案完整性、模型內容分開回報，不能用 fixture 或檔案 hash 通過推導真實模型已就緒。

CUDA 骨架採可選建置，預設關閉；需 CUDA Toolkit、cuFFT 及可執行的 NVIDIA GPU：

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON
cmake --build build-cuda --parallel 2
ctest --test-dir build-cuda --output-on-failure
build-cuda/cuda_synthetic config.txt output-directory
```

CPU 模型與合成 GPU 測試各驗證明列的有限範圍。真實模型推論、phase correlation／GMC、tracker、MOT 輸出及模型 artifact P01 不在本骨架驗收範圍；實際 GPU 與工具身分、已執行及未驗證項目見 [CUDA_SYNTHETIC](docs/CUDA_SYNTHETIC.md)。

CUDA 建置另含 `selective_scan_cuda_tests`：以相同 FP32 輸入對照 S01 reference，並驗證失敗 status、每次呼叫重設、指定 stream、graph capture、host 拒絕與 Cross-Scan；`--no-held-stream` 只供 initcheck 略過 host callback 擋住 stream 的那一項（[SCAN_CUDA §6](docs/SCAN_CUDA.md#6-測試矩陣與-mutation)）。結果與 sanitizer、mutation 證據見 [S02 證據](evidence/s02/2026-10-29/REPORT.md)；PASS 限 SCAN_CUDA 所列範圍。plugin 與最小 engine 的 S03 契約見 [SCAN_TRT](docs/SCAN_TRT.md)。完整 head 與候選權重見 [HEAD_TRT](docs/HEAD_TRT.md)：預設 CPU 建置含 `head_weights_tests`（checkpoint／bundle 拒絕條件）與 `head_reference_tests`（binary64 reference、primitive、語義變體）；給 `-DSACCADE_HEAD_CHECKPOINT` 時另轉換真實候選。CUDA＋TensorRT 建置另含 `head_trt` 生命週期工具與 `head_trt_tests`。FP16 與效能不在範圍內。

CUDA suite 保留原有 72 個 sessions，另加 8 個 terminal observation 開關對照；CLI／證據檔案寫入會明確 close 並檢查結果，正常 session close 亦檢查所有資源釋放結果。這些檢查不承諾 `fsync` 耐久性；CPU cleanup fake 不代表實際 GPU API 錯誤注入已驗證。[原始 GPU 證據](evidence/cuda-synthetic/2026-10-28/REPORT.md) 保留不變，後續收尾與 I/O 驗證見 [closeout 證據](evidence/cuda-synthetic/2026-10-28-closeout/REPORT.md)；sanitizer 結論以該報告為準。
