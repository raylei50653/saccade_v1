# Saccade v1

Saccade v1 以 C++／CUDA 重新設計 GPU 追蹤資料流。現階段交付是供後續實作引用的架構草案。

以舊架構的 YOLO26＋Mamba-FPN、GPU GMC、geometry-first／ReID-free tracker 與 MOT 輸出作為設計基線；重新抽象資料與責任邊界，全部重新實作。

已確定的範圍：從零實作、不復用舊程式、不使用 Python、不遷移舊實驗數據，也不要求保證舊實驗成績。

- [資料流與架構規劃](docs/DATAFLOW.md)：邏輯資料流、資料契約、跨幀狀態、GPU 所有權、同步依賴、結束流程及待定設計；§4.4 規定設定治理，§8.1 規定行為驗證。
- [非同步狀態流程控制](docs/ASYNC_CONTROL.md)：跨 stream 與 host↔device 的 event 協定、device 端提交、失敗收尾與競態驗證。

文件區分沿用的架構基線、重抽象的設計差異與待補齊的工程細節。階段和資料名稱描述責任邊界；C++ API、類別、目錄與 kernel 由後續設計決定。
