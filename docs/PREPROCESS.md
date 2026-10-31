# F01／F02：native JPEG decode 與 antialiased resize

[`NativePreprocessor`](../include/saccade/preprocess.hpp) 實作 `t01-input/1`：C++ 讀壓縮 JPEG，nvJPEG hardware backend 解碼成原尺寸 planar RGB uint8，再以 CUDA FP32 `/255` 與 antialiased bilinear、`align_corners=False` stretch 產生 `[1,3,640,640]`。不取整、不 clamp，沒有 mean／std、letterbox 或額外色彩轉換。數值比較政策為 [`t01-preprocess-fp32/1`](PREPROCESS_POLICY.md)，訓練側來源見 [T01 §2.1](T01_TRAINING.md#21-輸入前處理契約-t01-input1)。結果見 [2026-10-31 證據](../evidence/preprocess/2026-10-31/REPORT.md)：17 組輸出與訓練參考逐位元相同；F02 在受測範圍 PASS，F01 數值與 API 通過，但 nvJPEG 內部 initcheck FAIL／原因 OPEN。

## 1. 支援範圍與 API

F01 使用 `nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE)`、batch size 1 的 `nvjpegDecodeBatched` 與 `NVJPEG_OUTPUT_RGB`；只接受 baseline DCT、8-bit、三 component YCbCr 4:2:0 JPEG，並逐張核對 `nvjpegDecodeBatchedSupported`。沒有 hardware backend、progressive／gray／CMYK／4:4:4、錯誤或截斷 JPEG、超容量與 metadata 尺寸不符均明確失敗，不轉用其他 decoder。

`PreprocessConfig` 在建構時固定：

| 欄位 | 預設／限制 |
| --- | --- |
| `max_width`／`max_height` | 1920／1080，各為 1–8192 |
| `output_width`／`output_height` | 640／640，各為 1–4096；非 640 值供 primitive 測試與明確指定的呼叫者 |
| `max_jpeg_bytes` | 16 MiB，1 byte–256 MiB |

一個 instance 對應一個序列、一個 host submitter、一條呼叫者持有的 explicit non-default detect stream；stream 必須比 instance 活得久。建構時配置 pinned JPEG staging、nvJPEG state 與預配置 workspace、RAW、模型輸入、resize horizontal intermediate 與 completion event。`enqueue(jpeg, frame, expected_width, expected_height)` 複製壓縮 bytes，因此呼叫者的 JPEG 容器可在返回後釋放。frame 從 0 連續遞增；expected dimensions 必須同為 0，或同為正數並符合 JPEG。

返回的 `PreprocessedFrame` 包含 frame ID、`ImageMapping`、原尺寸 RGB 指標與模型輸入指標。指標借用至下一次 `enqueue`；F05／F03 等 GPU consumer 必須在下一次 `enqueue` **之前**排入同一條 detect stream。`wait()` 只供需要 host 觀察 preprocessing 完成的呼叫者，並不代替下游工作的完成事件。

`ImageMapping` 使用連續影像邊界座標：`model_x = original_x * scale_x`、`model_y = original_y * scale_y`，兩個 scale 分別為 `output_width/original_width` 與 `output_height/original_height`，以 double 保存；回原圖各除以自己的 scale。這份框座標映射與 resize 的 half-pixel 取樣中心是不同用途，不能把取樣的半像素偏移加入框座標。

F02 的獨立低階介面 [`resize_rgb_chw_cuda`](../include/saccade/resize_cuda.hpp) 接受 caller-owned、互不重疊的 input／output／workspace 與 stream；容量分別以 bytes／float elements 計。horizontal intermediate 是 `[3,input_height,output_width]`。enqueue 路徑不配置記憶體、不傳輸、不同步；呼叫者先以 `resize_cuda_load_kernels()` 完成 lazy loading。此 primitive 可被 CUDA graph capture／replay；含 nvJPEG 的 `NativePreprocessor` 拒絕 graph capture。

## 2. Stream、重用與失敗收尾

目前採一份 host staging 與一份 decode state。下一次 `enqueue` 在改寫 staging 前，host 等待前一幀 **F02 完成**的 event；之後在同一 detect stream 排入下一幀 decode 與 resize。RAW 與模型輸入的下游讀取，即使排在該 event 之後，也由 stream 順序保護，不會被下一幀提前覆寫。

這是 F01／F02 的獨立串流實作；[ASYNC_CONTROL §3–4](ASYNC_CONTROL.md#3-資源分類與數量) 的雙 STAGE、decoupled host／transfer／device API 與跨幀 host phase 重疊仍屬完整 session 的接續整合。此版本不把單 staging 的 host 等待冒稱為雙 slot pipeline，也未接入 F05、F03、ResultPacket 或 tracker。

任一 `enqueue`／`wait` 失敗即 latch 序列失敗，後續呼叫拒絕繼續；下一序列建立新 instance。呼叫者須明確 `close()`，其 checked drain 等待 borrowed stream 已排入的工作，再釋放所有 nvJPEG／CUDA 資源並回報釋放錯誤；destructor 僅作 best effort。檔案讀取錯誤由 checked file I/O 回報；此工具沒有 sequence iterator，因此也不以 decode 錯誤表示正常 EOF。

## 3. 建置與工具

CUDA Toolkit 須提供 nvJPEG，GPU 須支援 hardware backend；不需要 TensorRT。CPU 預設建置維持不啟用此模組。以下 `120` 是本機 sm_120 GPU 的架構設定，其他 GPU 請指定自己的架構。

```sh
cmake -S . -B build-preprocess -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-preprocess --parallel 2
ctest --test-dir build-preprocess -L preprocess --output-on-failure
```

單張 JPEG：

```sh
build-preprocess/preprocess jpeg /path/to/frame.jpg build-preprocess-single
```

輸出 `frame.rgb`（原尺寸 contiguous planar RGB uint8）、`frame.f32`（640×640 RGB CHW FP32）與 `mapping.txt`。後者記錄原圖／模型尺寸、獨立 x／y scales、JPEG hash、nvJPEG／CUDA／GPU 與 build source 身分。此 CLI 使用預設 1920×1080 與 16 MiB 容量；其他容量由 C++ API 設定。

## 4. 離線 references 與重跑

完整 capture 入口為 `bash scripts/run_preprocess_acceptance.sh build-NEW-OUTPUT`；預設要求 clean committed source，在新的忽略目錄保存 source／library hashes、references、完整 CTest、獨立比較、CLI 與 sanitizer 的命令、退出碼、log 和 RSS。`--allow-dirty` 只產生明確標記的開發證據。Sanitizer 執行前須有至少 4 GiB MemAvailable，並設逐項 timeout。

Reference 工具使用 T01 pinned export venv、torch 2.11.0／torchvision 0.26.0 與 `djpeg`；Python 僅用於產生與比較證據，不是 native runtime 依賴。輸出目錄必須為空；預設 dataset root 來自 T01 recipe，可用 `--dataset` 明確指定。

```sh
~/.cache/saccade-v1/t01/venv-export/bin/python \
  training/t01/t01_preprocess_ref.py generate --out build-preprocess-ref
build-preprocess/preprocess run build-preprocess-ref build-preprocess-result
~/.cache/saccade-v1/t01/venv-export/bin/python \
  training/t01/t01_preprocess_ref.py compare \
  --refs build-preprocess-ref --dump build-preprocess-result
```

References 包含 7 張 MOT17 幀、10 組 synthetic，`cases.tsv` 給出順序、輸入／輸出尺寸、JPEG（synthetic 為 `-`）、planar `.rgb` 與 CHW little-endian `.f32` 路徑。`manifest.json` 固定 hashes、policy snapshot、library／GPU 身分與語義對照。MOT17-02／10／13 的三個 float32 inputs 與已固定的 F03 輸入逐 byte 相同；其餘 synthetic 用於邊界、identity、upsample、非整除 downsample 與 singleton。

Native `run` 產生 `case-000.rgb`／`.f32` 等檔案，真實 JPEG 各有 `case-000-identity.txt`，另以 `report.tsv` 記錄 RGB bitwise、同 session 重跑 bitwise、逐元素 FP32 門檻、誤差與範圍。F01 要求 RGB 全 bytes 相同；F02 必須全部有限且每個元素滿足 [固定絕對界](PREPROCESS_POLICY.md#2-f02-算法與判定)。native 比較失敗保留已完成 prefix 的 report 並回傳非零；Python `compare` 額外核對 reference hashes 並產生 `comparison.json`。

只執行指定 case 可用 `preprocess run REFS OUT --case N`，N 為 0-based TSV 索引；供受限範圍的 sanitizer 或定位使用，不等於完整 cases 通過。要把完整 reference 比較註冊為 CTest，可於 configure 另加 `-DSACCADE_PREPROCESS_REFERENCES=/absolute/path/to/build-preprocess-ref`。

## 5. 測試與目前證據邊界

`resize_cuda_tests` 檢查低階 resize 的 CPU reference／手算案例、stream、repeat／graph 與 host 參數拒絕；`preprocess_tests` 使用 repo fixtures 檢查 odd-size decode、尺寸切換／重用、caller JPEG bytes 複製、independent sessions、ImageMapping、格式／容量／metadata／frame order／default-stream 拒絕、failure latch、capture 拒絕與 checked close。具體執行結果與 sanitizer 覆蓋須由實際紀錄佐證，不能以測試存在推論已通過。

乾淨 source `2b39dda` 的 19/19 CTest 通過，17 組共 13,522,182 個 FP32 輸出元素與訓練參考逐位元相同。F02 memcheck／initcheck／racecheck／synccheck 均為 0 errors／hazards；session memcheck／racecheck／synccheck 亦通過。session initcheck 在 nvJPEG 私有 `rgba2rgb` 報 8,151 errors，capture 因此 exit 99；不使用本專案程式的獨立 HW decode control 同樣報 1,890 errors。原因尚未由 NVIDIA 確認，不能當作已確認誤報，也不列全路徑 sanitizer PASS。完整結果與補跑範圍見上述報告。此實作不關閉 F04 decode／NMS、完整 async session 整合、P01 或模型品質；七張真實影像也不代表本次已重驗全部 5,316 張 MOT17 frame。
