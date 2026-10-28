# Saccade v1

Saccade v1 以 C++／CUDA 重新設計 GPU 追蹤資料流。目前包含架構、可重跑的 C++ 協定驗證，以及使用合成資料的 CUDA serial／pipelined 執行骨架。YOLO／Mamba 推論、完整 GMC、tracker 與 MOT writer 仍待各自任務實作。

以舊架構的 YOLO26＋Mamba-FPN、GPU GMC、geometry-first／ReID-free tracker 與 MOT 輸出作為設計基線；重新抽象資料與責任邊界，全部重新實作。

已確定的範圍：從零實作、不復用舊程式、不使用 Python、不遷移舊實驗數據，也不要求保證舊實驗成績。

- [資料流與架構規劃](docs/DATAFLOW.md)：邏輯資料流、資料契約、跨幀狀態、GPU 所有權、同步依賴、結束流程及待定設計；§4.4 規定設定治理，§8.1 規定行為驗證。
- [非同步狀態流程控制](docs/ASYNC_CONTROL.md)：跨 stream 與 host↔device 的 event 協定、device 端提交、失敗收尾與競態驗證。
- [ResultPacket 與失敗狀態模型](docs/RESULT_PACKET.md)：固定容量布局、穩定編碼、初始化與完整封包比較工具。
- [可重跑的協定模型](docs/PROTOCOL_MODEL.md)：216 組 happens-before 檢查與同步邊變異測試。
- [合成資料 CUDA 執行骨架](docs/CUDA_SYNTHETIC.md)：可選建置、typed resolved 設定、真實 streams／events／cuFFT、原始 ResultPacket 輸出、重跑命令與實機證據。
- [模型 artifact／native scan 可行性](docs/MODEL_FEASIBILITY.md)：P01 的輸入契約、工具觀察與待驗證範圍。

使用 CMake ≥ 3.20 與支援 C++20 的 compiler，不需要 CUDA、TensorRT 或 Python：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`build/protocol_model --self-test` 檢查同步協定；`build/protocol_tests` 驗證首幀初始化、錯誤注入、失敗鎖定、頻譜隔離與封包編碼。`build/packet_compare serial.bin pipelined.bin` 比較整份固定容量封包串流，包含未使用 rows 與 reserved bytes。

`synthetic_config_tests` 另外驗證最小設定 schema、非法輸入、穩定序列化與 hash；CUDA 關閉時上述 CPU 建置與測試均可獨立執行。

CUDA 骨架採可選建置，預設關閉；需 CUDA Toolkit、cuFFT 及可執行的 NVIDIA GPU：

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON
cmake --build build-cuda --parallel 2
ctest --test-dir build-cuda --output-on-failure
build-cuda/cuda_synthetic config.txt output-directory
```

CPU 模型與合成 GPU 測試各驗證明列的有限範圍。真實模型推論、phase correlation／GMC、tracker、MOT 輸出及模型 artifact P01 不在本骨架驗收範圍；實際 GPU 與工具身分、已執行及未驗證項目見 [CUDA_SYNTHETIC](docs/CUDA_SYNTHETIC.md)。
