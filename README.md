# Saccade v1

Saccade v1 以 C++／CUDA 重新設計 GPU 追蹤資料流。目前包含架構、可重跑的 C++ 協定驗證，使用合成資料的 CUDA serial／pipelined 執行骨架，以及原版 YOLO26m head scan 的 CPU scalar reference（S01）與 CUDA FP32 forward（S02）。YOLO／Mamba 推論、完整 GMC、tracker 與 MOT writer 仍待各自任務實作。

以舊架構的 YOLO26＋Mamba-FPN、GPU GMC、geometry-first／ReID-free tracker 與 MOT 輸出作為設計基線；重新抽象資料與責任邊界，全部重新實作。

已確定的範圍：從零實作、不復用舊程式、不使用 Python、不遷移舊實驗數據，也不要求保證舊實驗成績。

- [資料流與架構規劃](docs/DATAFLOW.md)：邏輯資料流、資料契約、跨幀狀態、GPU 所有權、同步依賴、結束流程及待定設計；§4.4 規定設定治理，§8.1 規定行為驗證。
- [非同步狀態流程控制](docs/ASYNC_CONTROL.md)：跨 stream 與 host↔device 的 event 協定、device 端提交、失敗收尾與競態驗證。
- [ResultPacket 與失敗狀態模型](docs/RESULT_PACKET.md)：固定容量布局、穩定編碼、初始化與完整封包比較工具。
- [可重跑的協定模型](docs/PROTOCOL_MODEL.md)：216 組 happens-before 檢查與同步邊變異測試。
- [合成資料 CUDA 執行骨架](docs/CUDA_SYNTHETIC.md)：可選建置、typed resolved 設定、真實 streams／events／cuFFT、原始 ResultPacket 輸出、重跑命令與實機證據。
- [模型 artifact／native scan 可行性](docs/MODEL_FEASIBILITY.md)：P01-A 的三項獨立驗收與 S01 接續條件。
- [模型 manifest schema 與 C++ CLI](docs/MODEL_MANIFEST.md)：嚴格 JSON、穩定序列化／hash、artifact 檔案完整性及 fixtures。
- [Native TensorRT SDK 最小驗證](docs/NATIVE_TENSORRT.md)：獨立可選 SDK 目標、版本 pin、固定小網路與重跑命令。
- [模型來源與缺件](docs/MODEL_ARTIFACT_READINESS.md)：s 時期的官方一般 YOLO26s native 查核；§7 為原版 m head 的 checkpoint／ONNX／engine 對應與剩餘缺件。
- [原版 YOLO26m head 的 scan 契約](docs/SCAN_CONTRACT.md)：v1 模型基線（YOLO26m＋原版 m T3→T1 head、T=1）的 module map、`saccade::SelectiveScan`／Cross-Scan 契約、比較政策與剩餘缺件；S01 scalar reference 依此實作。
- [S02 CUDA FP32 scan](docs/SCAN_CUDA.md)：`selective_scan_cuda` 與 Cross-Scan 展開／合併的 stream／status 契約、FP32 算法（穩定 softplus、精確指數的 drive、部分和縮放重算）、支援限制，以及首次 GPU 執行前固定的逐元素 running error bound 比較政策。
- [舊版 m 模型訓練規格與流程](docs/TRAINING_WORKFLOW.md)：以 preset 原版 YOLO26m T3→T1 為主，整理 teacher／cache／distill／GT 交接、微調設定與缺件；僅整理文件，未搬移訓練實作或模型產物，新訓練尚未啟動。

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

CUDA 建置另含 `selective_scan_cuda_tests`：以相同 FP32 輸入對照 S01 reference，並驗證失敗 status、每次呼叫重設、指定 stream、graph capture、host 拒絕與 Cross-Scan；`--no-held-stream` 只供 initcheck 略過 host callback 擋住 stream 的那一項（[SCAN_CUDA §6](docs/SCAN_CUDA.md#6-測試矩陣與-mutation)）。結果與 sanitizer、mutation 證據見 [S02 證據](evidence/s02/2026-10-29/REPORT.md)；PASS 限 SCAN_CUDA 所列範圍，FP16、plugin 與完整 head 仍待 S03／S04。

CUDA suite 保留原有 72 個 sessions，另加 8 個 terminal observation 開關對照；CLI／證據檔案寫入會明確 close 並檢查結果，正常 session close 亦檢查所有資源釋放結果。這些檢查不承諾 `fsync` 耐久性；CPU cleanup fake 不代表實際 GPU API 錯誤注入已驗證。[原始 GPU 證據](evidence/cuda-synthetic/2026-10-28/REPORT.md) 保留不變，後續收尾與 I/O 驗證見 [closeout 證據](evidence/cuda-synthetic/2026-10-28-closeout/REPORT.md)；sanitizer 結論以該報告為準。
