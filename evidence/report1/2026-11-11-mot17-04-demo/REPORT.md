# MOT17-04 展示來源擷取與資產核對

2026-11-11。**完整 1,050 幀來源擷取與展示資料核對 PASS。** 本輪只為現有本機展示台新增
MOT17-04-SDP，使用固定 `3cbaa518f0dc37d691404e95ed85c0109c81130d` 的既有乾淨 runtime
與原 HR engine 執行一次完整追蹤，保留 packets、motion、MOT 與 resolved config。
本輪不讀 GT、不重新計分、訓練或執行新的 sanitizer gate；MOT17-04 是既有 in-domain 資料，
不稱新的獨立 held-out 評估。

## 版本與條件

- 乾淨 detached source：`build/report1-live-20261111/source`，固定上述完整 commit。
- Binary：既有 clean build 的 `track_sequence`，SHA-256
  `d4d13caad5eba59143089f1fb9420d3ebb7f810cd08d06eab68b02a302e70a72`；本輪未重建。
- Engine：HR 1280×768 FP16，SHA-256
  `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`；本輪未重建。
- G1／`v1-motion/2`、Graph ON、pipelined、result ring=4、NSA OFF、measurement box OFF。
- 原圖 1920×1080，30 FPS，1,050 幀；資料直接連共享 MOT17/train/MOT17-04-SDP。
- RTX 5070 Ti Laptop、WSL2、driver 621.14、CUDA 13.4.92、TensorRT 10.16.1.11。
- 擷取持有既有 `~/.cache/saccade-v1/gpu.lock`，沒有修改 runtime 或預設。

## 擷取與資料

一次完整執行 exit 0。1,050 份 canonical ResultPackets 全部 OK，對應 JPEG 000001–001050。
packet frame 0 對應來源 JPEG 1；前兩幀尚無 confirmed 輸出。

| 項目 | 結果 |
| --- | --- |
| 逐幀 packet rows | **43,003**，全部 observed |
| 本幀輸出數 | 0–47；初始展示 JPEG 221 為 42 筆 |
| Motion records | 1,050；首幀 uninitialized，後 1,049 幀 valid |
| Packet stream | 12,986,400 B；SHA-256 `b6a3a3922ddce40aac30d96d85ca5b2e33170cb75fbe36e8245812d3f7d8b483` |
| Motion stream | 46,244 B；SHA-256 `86c391a3dc905f6a9bb60ad999cb54d47a5b06bf6147204e3725530f72c54f75` |
| 完成 MOT | 45,056 列，其中 2,053 列為 EOF 缺幀插值 |

展示只使用原始 packets 的 output ID、UID、FP32 狀態框與觀測分數，不混入 MOT 定稿的插值
與重新編號。10,045 筆框部分超出原圖，僅在繪製時裁切。12 個 UID 在 Bridge 接回時改變
output ID，因此顏色與尾線不構成身份正確的證明。

本次程序的觀測 wall time 為 6.933592 秒，CLI 內部 loop 自報 5,514.95 ms；這只是單次來源
擷取，沒有獨立逐幀延遲樣本、暖機剔除或重複量測，且前置 hash 可能暖化檔案 cache。
**不作效能 benchmark，MOT17-04 展示 metrics 明確為 null**，不套用 MOT17-10 的量測。

## 保管與展示資產

source、engine、binary／libraries、seqinfo 與全部 1,050 張 JPEG 前後 bytes／hash 完全相同。
`run1.command.json`、`run1.result.json` 與 `run1.log` 保留實際參數、退出碼與來源身分。
完整輸出位於 `raw/run1/`，packets 與 motion 以 gzip 保留；未壓縮檔案在
`build/report1-mot04-demo-20261111/run1/`。原有 MOT17-10／02 資產與腳本未改寫。

新的 `docs/demos/report1/methods/build_data_04.py` 從該輸出產生：

- `data-04.json`／`data-04.js`：1,050 幀、43,003 筆原觀測與 1,050 個 motion contexts。
- Data JS 只以 Object.assign 延伸 MOT17-04 的 catalog 項目，不覆寫原有 10／02。
- `source-04.mp4`：1920×1080，30 FPS，1,050 幀，35 秒，25,336,552 B。
- 影片為 CPU libx264、yuv420p、BT.709＋sRGB metadata、faststart；不是模型輸入。
- `poster-04.jpg`：原始 JPEG 000221。
- `provenance-04.json`：來源／輸出 hashes、編碼命令、工具、ffprobe 與限制。

全部 JPEG 在編碼前後再次核對 hashes。來源影片的有損壓縮不改變 JSON 中的原始數值。
新的 annotated 備用影片另由 `render_video_04.py` 產生，其收據是
`docs/demos/report1/assets/video-render-04.json`。

## 只讀核對

```sh
python3 -B evidence/report1/2026-11-11-mot17-04-demo/verify.py
```

核對 evidence checksums、完整 packets／motion／MOT、前後 source/input/binary 保管、
JSON／JS 的逐幀數值與影片身分；不啟動模型、讀 GT 或重新計分。
此 PASS 只適用完整來源擷取與展示資料一致性，沒有關閉完整 G1 memcheck、原生 Graph
racecheck 或其他既有 OPEN，也不是外部攝影機／RTSP 或新模型品質驗收。
