# MOT17-02 展示來源擷取與資產核對

2026-11-11。**完整 600 幀來源擷取與展示資料核對 PASS。** 本輪為展示台新增
MOT17-02-SDP，使用固定 `3cbaa518f0dc37d691404e95ed85c0109c81130d` 的既有乾淨
runtime 與原 HR engine 執行一次完整追蹤，保留 packets、motion、MOT 及設定。
本輪沒有讀 GT、重新計分、訓練或執行新的 sanitizer gate；不是新的獨立 held-out 評估。

## 版本與條件

- 乾淨 detached source：`build/report1-live-20261111/source`，固定上述完整 commit。
- Binary：同一 clean build 的 `track_sequence`，SHA-256
  `d4d13caad5eba59143089f1fb9420d3ebb7f810cd08d06eab68b02a302e70a72`；本輪未重建。
- Engine：HR 1280×768 FP16，SHA-256
  `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；本輪未重建。
- G1／`v1-motion/2`、Graph ON、pipelined、result ring=4、NSA OFF、measurement box OFF。
- 原圖 1920×1080，30 FPS，600 幀。來源直接使用共享 MOT17/train/MOT17-02-SDP。
- RTX 5070 Ti Laptop、WSL2、driver 621.14、CUDA 13.4.92、TensorRT 10.16.1.11。
- 擷取 campaign 持有既有 `~/.cache/saccade-v1/gpu.lock`；未修改 runtime 或預設。

## 擷取結果

一次完整執行 exit 0。600 份 canonical ResultPackets 全部 OK，對應 JPEG 000001–000600。
packet frame 0 對應來源 JPEG 1；前兩幀尚無 confirmed 輸出。

| 項目 | 結果 |
| --- | --- |
| 逐幀 packet rows | **11,454**，全部 observed |
| 本幀輸出數 | 0–29；初始展示 JPEG 221 為 19 筆 |
| Motion records | 600；首幀 uninitialized，後 599 幀 valid |
| Packet stream | 7,420,800 B；SHA-256 `50a7b97d5dad96199d2457da35a8bf1f274e9f977d27b2cb0377e26b5cd43caf` |
| Motion stream | 26,444 B；SHA-256 `f9209da865ea02ef3b8a681be87803e38902c4e4b7bcb957413095be9f22e59c` |
| 完成 MOT | 13,494 列，其中 2,040 列為 EOF 缺幀插值 |

展示使用原始 packets 的 output ID、UID、FP32 狀態框與觀測分數，不混入 MOT 定稿的插值
與重新編號。超出原圖的 336 筆框只在繪製時裁切。6 個 UID 在 Bridge 接回時改變 output ID，
因此顏色與尾線不是身份正確的證明。

這次程序的觀測 wall time 為 6.783119 秒，CLI 內部 loop 自報 3,208.52 ms。只有單次來源
擷取，含預置 hash 可能暖化檔案 cache，沒有獨立逐幀延遲樣本、暖機剔除或重複量測；
**不作效能 benchmark，MOT17-02 展示 metrics 明確為 null**，不套用 MOT17-10 的量測。

## 保管與展示資產

source、engine、binary／libraries、seqinfo 與全部 600 張 JPEG 前後 bytes／hash 完全相同。
`run1.command.json`、`run1.result.json` 與 `run1.log` 保留實際參數、退出碼及來源身分。
完整原始輸出存於 `raw/run1/`，packets 與 motion 以 gzip 保留；未壓縮檔案在
`build/report1-mot02-demo-20261111/run1/`。

`docs/demos/report1/methods/build_data_02.py` 另從上述不可變輸出產生資料與來源影片：

- `data-02.json`／`data-02.js`：600 幀、11,454 筆 rows 與 600 個 motion contexts。
- `source-02.mp4`：1920×1080，30 FPS，600 幀，20 秒，23,293,112 B。
- 來源影片為 CPU libx264 編碼，yuv420p、BT.709＋sRGB metadata、faststart；它不是模型輸入。
- `poster-02.jpg`：來源 JPEG 000221。
- `provenance-02.json`：原始來源、資產 hashes、編碼命令、工具與 ffprobe 結果。

原有 MOT17-10 展示資產未改寫。新的 annotated 備用影片由另一支 render script 產生，
其完整來源與輸出收據為 `docs/demos/report1/assets/video-render-02.json`。

## 核對與範圍

```sh
python3 -B evidence/report1/2026-11-11-mot17-02-demo/verify.py
```

只讀驗證 evidence checksums、完整 packet／motion、全部輸出與前後保管、JSON／JS
逐幀數值及來源影片身分；不啟動模型、讀 GT 或重新計分。此處的 PASS 只適用完整來源擷取
與展示資產一致性，不關閉完整 G1 memcheck、原生 Graph racecheck 或其他既有 OPEN。
