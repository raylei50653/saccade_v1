# 串流／NVDEC 開發驗證與交付檢查

日期：2026-11-05。狀態：**開發驗證 PASS（下列範圍）**，不是 clean-source 正式驗收。部署模型、tracker 與原 JPEG／P01 證書未由本報告重新判定。

## 來源、環境與功能

最初 GPU 驗證使用基底 `6d4c3d8a32b0ae9637e33aa3224ad69041814ce1` 加未提交變更，`source_dirty=true`、`formal_acceptance=false`。原始來源／binary／engine／CUDA library SHA-256 保存在 [development-source-manifest.json](capture/development-source-manifest.json)，不因後續提交改寫。

環境為 RTX 5070 Ti Laptop GPU（SM 12.0）、WSL driver 617.42、CUDA 13.4.92、TensorRT 10.16.1.11、FFmpeg libavformat 63.1.102。實際載入 `/usr/lib/wsl/lib/libnvcuvid.so.1` 與 libcuda；identity 與 SHA-256 依 manifest。

CLI 預設 FFmpeg 收流／demux → NVDEC device NV12／P010 → CUDA 色彩轉換／F02 → TrackingSession。GPU 解碼失敗不自動退回 CPU；CPU 路徑須明確選 `--decode cpu`。surface owner 保留 AVFrame、primary context、producer stream 與 ready event，直到 D 完成讀取。即時結果用 native output ID，不做離線 MOT 尾段插值。API 與操作見 [STREAMING](../../../docs/STREAMING.md)。

## 初輪 GPU 驗證

61 項配置 CTest 分批均有通過紀錄：

| 範圍 | 結果與紀錄 |
| --- | --- |
| device 前處理與 device session | 2/2 PASS；[初輪新路徑測試](capture/initial-new-paths-and-fixture-failure.log) 前兩項 |
| 其餘 CPU／CUDA 回歸 | 58/58 PASS；[摘要](capture/initial-regression.log)、[完整輸出](capture/initial-regression-tests.log.gz) |
| 影片 CLI | 1/1 PASS、32 個案例；[CTest](capture/initial-cli-final.log)、[CLI report](capture/initial-cli-report.json) |
| NVDEC source | 326 checks；H.264、HEVC 8／10-bit 各 9 幀含 B-frame；同 plane layout 的 CPU RGB reference max／mean error 均為 0 |
| device color／F02 sanitizer | memcheck、racecheck、synccheck、initcheck 各 3,345 checks，0 errors／hazards；[memcheck](capture/device-video-memcheck.log)、[racecheck](capture/device-video-racecheck.log)、[synccheck](capture/device-video-synccheck.log)、[initcheck](capture/device-video-initcheck.log) |

完整 CTest stdout 以 gzip 保留原始 bytes（含 TensorRT 輸出的行尾空白），未重排或修剪。

device 前處理涵蓋 NV12／P010、三種 matrix × full／limited range、奇數尺寸與 pitches、ready event、owner lease、重用及失敗。scalar binary64 RGB reference 容許最多 1 byte 的 FP32 量化邊界差異。sanitizer 使用 `--no-held-stream`，producer host callback／owner gate 在正常測試執行；sanitizer 範圍不包含 NVDEC、TensorRT、NVOFA 或完整 session 的函式庫內部 kernels。

session 比較使用相同 CUDA 轉換後 RGB 的 host baseline，完整 packets 與 device ordinary serial／Graph pipelined／perturbed 路徑相同；另驗 source close 後 surface 存活、cancel、尺寸失敗。測試 oracle 才做影像 D2H，生產 GPU 路徑不做。

CLI 另讀 MOT17-10 前 9 張 JPEG，編為 H.264／HEVC 8／10-bit BT.709 limited-range SDR B-frame 影片。各影片 Graph／ordinary serial 完整 packets 與即時 MOT bytes 相同，並有實際 tracking rows；prefix 限制、既有檔案保護、錯誤、SIGINT 130／SIGTERM 143 均通過。只讀影像，沒有讀 GT 或重新計分；[輸入檔案身分](capture/input-jpegs-observed-sha256.json)是交付檢查時的唯讀觀測。

初輪 CLI 失敗是測試影片缺少實際 `color_transfer` metadata，保留在 [失敗紀錄](capture/initial-new-paths-and-fixture-failure.log)。修正 fixture 的色彩轉換與 `setparams` 後，以新輸出目錄重跑通過，未放寬 assertions。來源測試也曾遇到 swscale 不同 pixel layout 的 chroma 取樣差異；最終 oracle 先無損對齊 plane layout，再比較 decoder reconstruction，不以不同取樣政策當作 decoder 誤差。

## 影像搬移檢查

H.264 前 3 幀的 Nsight `cuda,nvvideo` node capture 觀察到：

- `cuvidCreateDecoder` 1 次、`cuvidDecodePicture` 5 次、`cuvidMapVideoFrame64` 3 次、destroy 1 次。decode 前緩衝數與交付數分列。
- D 上 3 次 CUDA 色彩轉換。從首次轉換 kernel 的 GPU start 起，H2D 為 0；D2H 只有 3×12,368-byte ResultPacket 與 3×56-byte MotionContext，共 37,272 bytes。
- decoder stream 仍有 GPU 內 surface 複製。初始化 engine 等 H2D 為 58,265,984 bytes，在該邊界之前；不納入每幀影像搬移結論。
- trace 的 3 幀 packets 與未 profile 的同影片 Graph 輸出前 3 幀相同。

詳細結果及原 SQLite SHA-256 見 [trace analysis](capture/nvdec-trace.analysis-final.json)；唯讀分析程式為 [analyze_trace.py](analyze_trace.py)（Python ≥ 3.11）。原 `.sqlite`／`.nsys-rep` 留在本機 `build/streaming-nvdec/`；可依報告中的資料庫 SHA-256 驗證。

NVVIDEO intervals 是 host API 時間，不代表硬體 decoder engine busy time。CPU／GPU 時間域分列，不據此宣稱引擎 overlap、吞吐提升或完整函式庫 race safety。

## 交付前審查與修正

獨立唯讀審查確認 GPU 提交前驗證、ready／consumed、source frame lease、D-only 錯誤尾段、primary context 互通及 JPEG 預設數值路徑。另修正兩個 CLI 邊界：輸出在實際 open 時原子拒絕已存在路徑；來源建構期間的取消在 timeout 邊界回傳 130／143，不掩蓋後續 CUDA／輸出錯誤。

來源測試也加強為跨 thread 釋放**最後一個** frame owner，並以 weak pointer 驗證析構完成。交付檢查另使用 `build/streaming-review/` 新建置，保留 `build/streaming-nvdec/` 的原 binary。GPU decoder、色彩轉換、preprocessor 與 session 的生產程式未因這兩個 CLI 修正改動；提交前的完整來源與新 binary SHA-256 見 [review-source-manifest.json](capture/review-source-manifest.json)，與初輪來源的變動檔案也明列其中。

| 交付檢查 | 結果 |
| --- | --- |
| 預設 CPU configure／build／CTest | 20/20 PASS；[紀錄](capture/review-cpu-ctest.log)；同 CI 的預設無 CUDA／FFmpeg 建置路徑 |
| 新建置 NVDEC source 測試 | 1/1 PASS、329 checks，含最後 owner 跨 thread 析構；[紀錄](capture/review-source-cuda-ctest.log) |
| 新建置影片 CLI | 1/1 PASS、35 個案例；[CTest](capture/review-cli-ctest.log)、[完整 case report](capture/review-cli-report.json) |

新增案例驗證來源開啟期間取消、雙程序競爭建立同一輸出時只有一方成功且封包前綴完整。測試 hash helper 不再要求 Python 3.11；CMake 明確傳入找到的 FFmpeg／ffprobe 路徑。原初輪開發 binary 的 61 項結果保留來源身分，未把本輪新 CLI 測試冒稱為新 binary 重跑全部 61 項。

## 未涵蓋

外部 RTSP／攝影機實際連線未執行，未提供 endpoint 或裝置。沒有自動重連、主動丟幀、可變 Kalman dt、HDR tone mapping、BT.2020 constant-luminance 或其他 surface format 支援。FFmpeg demux／startup probing 仍為 host 操作。既有 nvJPEG／NVOFA OPEN 診斷保留；本次 commit／push 不把上述開發測試改稱 clean-source 正式證書。
