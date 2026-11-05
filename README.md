# Saccade v1

Saccade v1 以 C++／CUDA／TensorRT 實作 JPEG sequence → 偵測 → GMC → geometry-first／ReID-free tracking → MOT 輸出。runtime 不依賴 Python；模型訓練與匯出使用 PyTorch／Python。專案從零重寫，以舊架構作設計基線，不復用舊 runtime 程式或遷移舊實驗數據。

文件狀態更新：**2026-11-05**。目前部署為 **HR 1280×768 FP16**，CUDA Graph 預設啟用，GMC 已採 **G1 NVOFA／v1-motion/2**。M0–M4 完成，**P01 CLOSED（有範圍）**；後續 [G1 驗收](evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md)的完整 session memcheck 仍為 OPEN，其餘八步通過。

原 [CUDA Graph 正式驗收](evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md)與 [F02／F03 加速後七序列 MOT 開發測試](evidence/mot/2026-11-04-full-development/REPORT.md)保留各自受測來源與 motion profile；後者 5,316 幀、21/21 份輸出與 M4 相同，品質不變，吞吐 195.85 FPS，屬 M4 之後的開發來源證據。

F02 tile fusion／F03 N16 與部署特化已完成 [clean-source 整合補驗](evidence/f02-f03/2026-11-05-clean-23ade53/REPORT.md)：乾淨 `23ade53`、`v1-motion/1`，原 session gate、F02／scan 四種 sanitizer 及 HR 七序列回歸 PASS，M4 輸出 21/21 相同。兩個 kernels 與目前版本的 bytes 相同；現行 G1 的完整 session memcheck 仍 OPEN。

另完成 [clean-source ABBA 效能補驗](evidence/f02-f03/2026-11-05-clean-performance/REPORT.md)：固定 engine、`f4865cd`→`23ade53`、原 `v1-motion/1`，完整 F03 Graph 5.242→4.905 ms（−6.43%），tracking 單幀平均 7.093→6.626 ms，queued 180.6→195.2 FPS。四張模型 outputs／statuses 與 32 份完整 packets 相同；tracking 限 MOT17-10 的 140 幀 prefix、JPEG 預載，原開發測時仍保留各自來源身分。

[OAO duration-ramp／front-occluder 四臂消融](evidence/occlusion/2026-11-05/REPORT.md)只用 MOT17-10 val 的固定輸入 CPU replay，舊 preset 參數未改善關聯，未加入部署預設。完整判定與未涵蓋項目見 [目前狀態](docs/STATUS.md)，待辦見 [BACKLOG](docs/BACKLOG.md)。

## 文件分組

### 狀態與交付

| 文件 | 用途與狀態 |
| --- | --- |
| [目前狀態](docs/STATUS.md) | 現行部署、各階段驗收、文件狀態與未完成邊界的總入口 |
| [後續工作](docs/BACKLOG.md) | 已結案工作的接續關係與仍待處理項目 |
| [T01 實驗結案與恢復](docs/history/2026-10-31-t01-closeout.md) | 歷史封存；固定 tag `t01-experiment-final-2026-10-31`（`a47fb16`），含 PR #1 合併與 archive custody |

### 架構與同步協定

| 文件 | 用途與範圍 |
| --- | --- |
| [資料流與架構](docs/DATAFLOW.md) | 邏輯階段、資料契約、GPU 所有權、設定治理與行為驗證 |
| [非同步流程控制](docs/ASYNC_CONTROL.md) | streams／events、ready／consumed、首次失敗與 drain／close |
| [ResultPacket](docs/RESULT_PACKET.md) | 固定容量封包、穩定編碼、失敗狀態與完整 bytes 比較 |
| [C++ 協定模型](docs/PROTOCOL_MODEL.md) | 216 組有限 happens-before 檢查與同步邊變異測試 |
| [合成 CUDA 骨架](docs/CUDA_SYNTHETIC.md) | streams／events／cuFFT 的獨立有限驗證；保留原始與 closeout 證據 |

### 前處理、偵測與追蹤

| 文件 | 用途與狀態 |
| --- | --- |
| [F01／F02 native 前處理](docs/PREPROCESS.md) | nvJPEG hardware decode、FP32 antialiased resize；使用驗收 PASS，initcheck 診斷保留 |
| [F03 producer 聚合器](docs/MODEL_PRODUCER.md) | input／raw finite checks、九個 plugin statuses 與 F04 aggregate；獨立正式驗收 PASS |
| [F04 decode／篩選／NMS](docs/POSTPROCESS.md) | 全類別 Top-300、kept／suppressed pool、device 分級排序與平行 finalize；正式驗收 PASS |
| [F01–F04 detection session](docs/DETECTION_SESSION.md) | 獨立 bounded detection API；原開發來源驗證 PASS，不等同 TrackingSession |
| [F05–F09 tracking](docs/TRACKING.md) | GMC、prior、F04b、關聯更新、ResultPacket／MOT、M0–M4；正式驗收及 P01 有範圍結案 |
| [CUDA Graph 與延遲量測](docs/CUDA_GRAPH.md) | D 的 F03→F04 與 M tracking tail；預設啟用，另有最新 clean-source 驗收 |
| [即時影像串流](docs/STREAMING.md) | FFmpeg 收流／NVDEC GPU 解碼、CUDA 色彩轉換與即時結果 CLI；CPU decode 為顯式選項，驗證邊界獨立記錄 |

### 模型來源與 native 推論

| 文件 | 用途與範圍 |
| --- | --- |
| [模型／scan 可行性](docs/MODEL_FEASIBILITY.md) | P01-A 與 S01–S04 的分階段契約及歷史判定 |
| [Artifact readiness](docs/MODEL_ARTIFACT_READINESS.md) | 候選來源、缺件演進與現行 HR／P01 證據入口 |
| [Model manifest](docs/MODEL_MANIFEST.md) | schemas 1–3、C++ CLI、完整性與 fixtures；HR FP16 使用獨立 deploy manifest |
| [Native TensorRT SDK](docs/NATIVE_TENSORRT.md) | pinned SDK、可選建置及固定小網路驗證 |
| [原版 YOLO26m scan 契約](docs/SCAN_CONTRACT.md) | T=1 SelectiveScan／Cross-Scan、module map、tensor 與狀態邊界 |
| [S02 CUDA FP32 scan](docs/SCAN_CUDA.md) | FP32 forward、stream／status、數值與 sanitizer 證據 |
| [S03 TensorRT scan](docs/SCAN_TRT.md) | plugins、最小 engine、frame-end gate／publish／commit |
| [S04 完整 FP32 head](docs/HEAD_TRT.md) | native 權重 bundle、建圖、六個 raw outputs、生命週期；歷史階段 PASS |
| [T01 backbone／完整 F03](docs/MODEL_TRT.md) | backbone ONNX＋head、原 FP32 驗收與後續 FP16 build 模式 |

### 數值與使用政策

以下政策保留各自固定的版本、比較方法與驗收範圍；新部署與效能改版的結果另由對應報告判定。

| 文件 | 固定內容 |
| --- | --- |
| [HEAD_POLICY](docs/HEAD_POLICY.md) | S04 FP32 head reference 與逐元素比較 |
| [BACKBONE_POLICY](docs/BACKBONE_POLICY.md) | T01 backbone／完整 F03 FP32 比較 |
| [PREPROCESS_POLICY](docs/PREPROCESS_POLICY.md) | F01 RGB／F02 FP32 輸出與固定 cases |
| [PREPROCESS_USAGE_POLICY](docs/PREPROCESS_USAGE_POLICY.md) | 正常使用、重跑與資源重用驗收；既知 initcheck 診斷的處置 |
| [POSTPROCESS_POLICY](docs/POSTPROCESS_POLICY.md) | 全類別 Top-300、ties、FP32 算術、閾值、capacity 與完整候選輸出 |

### 訓練與部署操作

| 文件 | 用途與狀態 |
| --- | --- |
| [T01 重新訓練](docs/T01_TRAINING.md) | 事前規格、split／選模政策、執行與最終結果；2026-10-31 已封存結案 |
| [舊版訓練流程](docs/TRAINING_WORKFLOW.md) | 舊庫 teacher／cache／distill／GT／T3→T1 的來源參考 |
| [T01 重跑程式與配方](training/t01/) | pinned source、patch、lockfiles 與原實驗入口 |
| [HR 部署操作](training/hr_deploy/README.md) | HR checkpoint、匯出、reference corpus 與 manifest；目前部署身分見 TRACKING §15–§17 |

### 驗收與效能證據

報告的判定只適用於各自記錄的來源、環境與受測範圍。較新的正式驗收不改寫早期開發、FAIL 或未完成紀錄。

| 證據 | 狀態與閱讀重點 |
| --- | --- |
| [串流／NVDEC 開發驗證與交付檢查](evidence/streaming/2026-11-05-development/REPORT.md) | 原生 GPU decode／CUDA 色彩轉換、完整 packet 比較與資料搬移 trace；保留開發來源、CLI 修正與外部裝置未連線邊界 |
| [G1 NVOFA／v1-motion/2 驗收](evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md) | 完整 session memcheck OPEN（Bug 7048273），其餘八步通過；runtime 已整合，原證據與支援邊界保留 |
| [F02／F03 clean-source 整合補驗](evidence/f02-f03/2026-11-05-clean-23ade53/REPORT.md) | 乾淨 `23ade53`、`v1-motion/1`；CPU 16/16、CUDA 59/59、原 session gate、F02／scan 四 sanitizer、M4 21/21 輸出 PASS；G1 memcheck OPEN 分列 |
| [F02／F03 clean-source ABBA 效能](evidence/f02-f03/2026-11-05-clean-performance/REPORT.md) | 乾淨 `f4865cd`→`23ade53`、固定 engine、`v1-motion/1`；完整 F03 Graph −6.43%，queued 180.6→195.2 FPS；四張 raw／status、32 份 packets 相同，限固定輸入／140 幀 prefix |
| [最新 CUDA Graph 分支正式驗收](evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) | 乾淨 `4329ed7`／`d21a144`；F04、tracking、stream 守門與 M4 21/21 輸出比對 PASS |
| [加速後完整 MOT 開發測試](evidence/mot/2026-11-04-full-development/REPORT.md) | F02 fusion／F03 特化；CPU 16/16、CUDA 59/59、有範圍 sanitizer 及七序列守門 PASS；5,316 幀、M4 輸出 21/21 相同、品質不變，吞吐 195.85 FPS；非 clean-source 正式證書 |
| [M4 正式評估與 P01](evidence/m4/2026-11-03-formal-cad5298/REPORT.md) | `cad5298`；P01 CLOSED（有範圍），MOT17-02 HOTA 40.6／AP50 72.91%；品質只作記錄 |
| [M3 TrackingSession 正式驗收](evidence/tracking-session/2026-11-03-formal-e021bd1/REPORT.md) | `e021bd1`；完整 tracking 接續、packet／MOT 與 CPU reference 比對 |
| [F04 平行 finalize 正式驗收](evidence/f04/2026-11-04-parallel-finalize-e4d45de/REPORT.md) | `e4d45de`；HR finalize 約 1,539→50 µs，輸出 bytes 不變；各輪結果見 POSTPROCESS §5 |
| [F03 producer 正式驗收](evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md)／[F04 原 adaptive 正式驗收](evidence/f04/2026-10-31-formal-995bca5/REPORT.md) | 各自的 640 FP32 primitive 範圍與歷史來源 |
| [F01／F02 使用驗收](evidence/preprocess/2026-10-31-usage/REPORT.md)／[detection session 開發驗證](evidence/detection-session/2026-10-31/REPORT.md) | 各自固定輸入、輸出等價與資源重用範圍 |
| [CUDA Graph 開發量測](evidence/cuda-graph/2026-11-04/REPORT.md) | 單幀平均 8.59→7.06 ms；保留開發來源與首幀成本，不混作正式驗收測時 |
| [Tracking 排序開發量測](evidence/tracking-sort/2026-11-04/REPORT.md) | F04b replay 143.57→102.81 µs；完整吞吐幾乎不變，不宣稱穩定端到端收益 |
| [OAO／front-occluder 消融](evidence/occlusion/2026-11-05/REPORT.md) | HR＋G1 的 MOT17-10 四臂 CPU replay；舊固定參數未改善關聯，未加入部署預設；交付前 CPU 20/20、baseline packets／四臂 MOT bytes 相同 |
| [硬體資源與階段延遲](evidence/resource-schedule/2026-11-04/REPORT.md)／[排程表](evidence/resource-schedule/2026-11-04/SCHEDULE.md) | 單幀平均 6.988 ms、queued 吞吐約 184.8 FPS；2 格 ring 的單次交付延遲 10.740 ms；完整 HR racecheck 曾 OOM／timeout，限制保留 |
| [Kernel 資料格式與 cache 分析](evidence/kernel-analysis/2026-11-04/REPORT.md) | 現行 source 的 buffer 大小／strides、47 個函式編譯資源、隔離 cache counters，以及後續 fusion／shared／L2 復用候選；尚未修改 runtime |
| [HR F02 tile fusion 開發量測](evidence/f02-fusion/2026-11-04/REPORT.md) | 固定 1920×1080→1280×768 的 F02 interval 149.06→89.76 µs；queued 5.367→5.312 ms/frame。128 組 bitwise、60/60 CTest 與有範圍 sanitizer 通過；workspace 容量仍保留，該輪為開發量測，clean-source 補驗另列 |
| [F03 N16 特化開發量測](evidence/f03-scan/2026-11-04/REPORT.md) | scan API Graph interval 1083.52→770.103 µs，完整 F03 Graph 5.051→4.762 ms；新舊 bits、四張模型輸出與固定 140 幀 packets 相同；保留較慢的 broadcast 試驗 |
| [llama.cpp 啟發的 F03 加速與讀取試驗](evidence/f03-llama/2026-11-04/REPORT.md) | 在既有N16之上，部署shape特化＋128-thread block使完整F03兩輪改善1.15%／1.64%；輸出bits相同，scan四種sanitizer通過。共用load／shared copy另有負控制與NCU counters；開發證據 |
| [F03 backbone／head 瓶頸分析](evidence/f03-backbone-head/2026-11-04/REPORT.md) | 正常Graph的6040個kernel calls完整對回layers，backbone／neck占kernel duration加總66.12%；cache、builder及causal改寫均未帶來可採用的新收益，部署engine保持原版 |

## 建置與驗證

所有建置與驗收輸出都放在 `build/<名稱>/`，整個 `build/` 不入 Git：日常建置用 `build/cpu`、`build/cuda`，驗收腳本只接受新的 `build/<名稱>` 輸出目錄。根目錄不放其他建置資料夾；根目錄的 `build-*` 不再被忽略，誤建會出現在 `git status`，要求乾淨工作樹的驗收腳本也會拒絕執行。

### CPU 協定與 reference

使用 CMake ≥ 3.20 與支援 C++20 的 compiler，不需要 CUDA、TensorRT 或 Python：

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu --parallel 2
ctest --test-dir build/cpu --output-on-failure
```

`build/cpu/protocol_model --self-test` 檢查同步協定；`build/cpu/protocol_tests` 驗證首幀初始化、錯誤注入、失敗鎖定、頻譜隔離與封包編碼。`build/cpu/packet_compare serial.bin pipelined.bin` 比較整份固定容量封包串流，包含未使用 rows 與 reserved bytes。

`synthetic_config_tests` 另外驗證最小設定 schema、非法輸入、穩定序列化與 hash；`file_io_tests` 驗證讀取及明確 close 的錯誤傳播，`cleanup_tests` 以 CPU fake 驗證資源釋放錯誤與所有權政策。`model_manifest_tests` 與 CLI 測試加入模型輸入契約及失敗條件；`selective_scan_tests` 以 bc 推導、手算與契約等式驗證 S01 的 binary64 scan reference 及其拒絕條件（[fixtures](fixtures/selective-scan/README.md)）。預設建置不需 GPU。

```sh
build/cpu/model_manifest validate fixtures/model-manifest/contract.json
build/cpu/model_manifest verify fixtures/model-manifest/contract.json build/cpu/canonical-manifest.json
```

這份 fixture 只驗證工具；CLI 將宣告、檔案完整性、模型內容分開回報，不能用 fixture 或檔案 hash 通過推導真實模型已就緒。

### 合成 CUDA 骨架

CUDA 骨架採可選建置，預設關閉；需 CUDA Toolkit、cuFFT 及可執行的 NVIDIA GPU：

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON
cmake --build build/cuda --parallel 2
ctest --test-dir build/cuda --output-on-failure
build/cuda/cuda_synthetic config.txt output-directory
```

CPU 模型與合成 GPU 測試各驗證明列的有限範圍。真實模型推論、phase correlation／GMC、tracker、MOT 輸出及模型 artifact P01 不在本骨架驗收範圍；實際 GPU 與工具身分、已執行及未驗證項目見 [CUDA_SYNTHETIC](docs/CUDA_SYNTHETIC.md)。

CUDA 建置另含 `selective_scan_cuda_tests`：以相同 FP32 輸入對照 S01 reference，並驗證失敗 status、每次呼叫重設、指定 stream、graph capture、host 拒絕與 Cross-Scan；`--no-held-stream` 只供 initcheck 略過 host callback 擋住 stream 的那一項（[SCAN_CUDA §6](docs/SCAN_CUDA.md#6-測試矩陣與-mutation)）。結果與 sanitizer、mutation 證據見 [S02 證據](evidence/s02/2026-10-29/REPORT.md)；PASS 限 SCAN_CUDA 所列範圍。plugin 與最小 engine 的 S03 契約見 [SCAN_TRT](docs/SCAN_TRT.md)。完整 head 與候選權重見 [HEAD_TRT](docs/HEAD_TRT.md)：預設 CPU 建置含 `head_weights_tests`（checkpoint／bundle 拒絕條件）與 `head_reference_tests`（binary64 reference、primitive、語義變體）；給 `-DSACCADE_HEAD_CHECKPOINT` 時另轉換真實候選。CUDA＋TensorRT 建置另含 `head_trt` 生命週期工具與 `head_trt_tests`。此處原 S04 FP32 head 驗收不涵蓋 FP16 與效能；現行 HR FP16 部署見 [TRACKING §16](docs/TRACKING.md#16-d2hr-部署改用-fp162026-11-03-使用者核准)。

CUDA suite 保留原有 72 個 sessions，另加 8 個 terminal observation 開關對照；CLI／證據檔案寫入會明確 close 並檢查結果，正常 session close 亦檢查所有資源釋放結果。這些檢查不承諾 `fsync` 耐久性；CPU cleanup fake 不代表實際 GPU API 錯誤注入已驗證。[原始 GPU 證據](evidence/cuda-synthetic/2026-10-28/REPORT.md) 保留不變，後續收尾與 I/O 驗證見 [closeout 證據](evidence/cuda-synthetic/2026-10-28-closeout/REPORT.md)；sanitizer 結論以該報告為準。
