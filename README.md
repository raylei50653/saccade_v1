# Saccade v1

Saccade v1 以 C++／CUDA 重新設計 GPU 追蹤資料流。目前包含架構草案與第一批可重跑的 C++ 協定驗證，尚未實作 GPU 追蹤 pipeline。

以舊架構的 YOLO26＋Mamba-FPN、GPU GMC、geometry-first／ReID-free tracker 與 MOT 輸出作為設計基線；重新抽象資料與責任邊界，全部重新實作。

已確定的範圍：從零實作、不復用舊程式、不使用 Python、不遷移舊實驗數據，也不要求保證舊實驗成績。

- [資料流與架構規劃](docs/DATAFLOW.md)：邏輯資料流、資料契約、跨幀狀態、GPU 所有權、同步依賴、結束流程及待定設計；§4.4 規定設定治理，§8.1 規定行為驗證。
- [非同步狀態流程控制](docs/ASYNC_CONTROL.md)：跨 stream 與 host↔device 的 event 協定、device 端提交、失敗收尾與競態驗證。
- [ResultPacket 與失敗狀態模型](docs/RESULT_PACKET.md)：固定容量布局、穩定編碼、初始化與完整封包比較工具。
- [可重跑的協定模型](docs/PROTOCOL_MODEL.md)：216 組 happens-before 檢查與同步邊變異測試。
- [模型 artifact／native scan 可行性](docs/MODEL_FEASIBILITY.md)：P01 的輸入契約、工具觀察與待驗證範圍。

使用 CMake ≥ 3.20 與支援 C++20 的 compiler，不需要 CUDA、TensorRT 或 Python：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`build/protocol_model --self-test` 檢查同步協定；`build/protocol_tests` 驗證首幀初始化、錯誤注入、失敗鎖定、頻譜隔離與封包編碼。`build/packet_compare serial.bin pipelined.bin` 比較整份固定容量封包串流，包含未使用 rows 與 reserved bytes。

這些 CPU 模型驗證明定的有限協定，不代表 CUDA 排程、cuFFT、推論、tracker 或 MOT 輸出已通過 V03–V05。文件區分沿用的架構基線、重抽象的設計差異與待補齊的工程細節；GPU kernel 與其 runtime 驗證仍待後續實作。
