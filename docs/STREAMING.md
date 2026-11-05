# 即時影像串流組件

2026-11-05：新增可選的 native C++ 輸入擴充，支援 FFmpeg 能開啟的影片、RTSP 與攝影機裝置。既有 JPEG sequence 入口仍依原本政策執行；本擴充的驗證狀態分列，不沿用 P01／M4 的正式證書。

## 資料路徑與 API

預設路徑為 `FFmpeg 收流／解封裝 → NVDEC → device NV12／P010 → D 上 CUDA 色彩轉換／F02 → F05、F03／F04 → M tracking → ResultPacket`。解碼影像不下載 CPU，不經 host RGB staging。

`--decode cpu` 才選用初版 `CPU decode／色彩轉換 → host planar RGB8 → pinned staging → H2D／F02`；GPU 不支援或失敗時不自動退回 CPU。

- `include/saccade/video_source.hpp`：`VideoSource::read(VideoFrame&)` 依呈現順序讀取一幀。CUDA backend 交付 `device_frame`，`rgb` 保持空；CPU backend 才交付連續 R、G、B 三個 uint8 planes。兩者帶零起始 `index` 與可選 `pts_us`。EOF 會先交付 decoder 延遲的尾幀，再回傳 false。
- `VideoSource` 由一個 caller 使用；只有 `request_stop()` 可跨 thread 呼叫，用於中斷阻塞的輸入操作。close 前須停止並 join 呼叫 stop 的 thread。
- `PreprocessConfig.input = PreprocessInput::HostRgb` 選用預配置的 RGB staging，不建立 nvJPEG decoder；`NativePreprocessor::enqueue_rgb()` 返回前已擁有輸入的副本。F02 沿用既有 GPU resize。
- `TrackingSession::enqueue_rgb()` 接受設定中的固定原圖尺寸，frame 由 0 連續遞增。thread、device、result ring、failure latch、finish／cancel／close 與既有 session 相同。
- `PreprocessInput::DeviceVideo`／`TrackingSession::enqueue_device()` 接受具 owner lease 與 ready event 的 GPU surface。CUDA 色彩轉換先等待 producer event，preprocessor 持有 AVFrame／CUDA context 的 lease 至 D 完成讀取；來源關閉不會提前釋放尚有 owner 的 frame。

FFmpeg 與 CUDA runtime 使用同裝置的 primary context；來源自行 retain primary context，FFmpeg 使用 current-context 模式，避免修改已啟動 context 的排程 flags。decoder 的 producer stream 與 session D 以 event 排序。現有 JPEG sequence 仍走原生 nvJPEG。

`track_stream` 使用目前部署的 `hr-1280x768` 作為預設 profile，CUDA Graph 預設啟用。CLI 每幀提交後立即接收並 flush 輸出，再等待下一幀，避免等待後續輸入才交付結果；因此 CLI 不宣稱跨幀吞吐 overlap。可重用的 TrackingSession RGB API 仍支援原有多格 ring 排程。

## 建置與執行

影片來源可獨立在 CPU 建置；需要 pkg-config、FFmpeg CLI（產生測試影片）與 libavformat、libavcodec、libavutil、libswscale、libavdevice 開發函式庫。NVDEC 的來源／CLI 測試另需 ffprobe，以及 FFmpeg 的 libx264、libx265、FFV1 encoder 與 lavfi test source；runtime 不呼叫這些 encoder：

```sh
cmake -S . -B build/video -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_VIDEO=ON
cmake --build build/video --parallel 2
ctest --test-dir build/video --output-on-failure
build/video/video_probe input.mp4
```

完整追蹤沿用 native TensorRT／CUDA 建置需求，另開啟 VIDEO、NVDEC 與 STREAMING：

```sh
cmake -S . -B build/streaming -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON \
  -DSACCADE_ENABLE_POSTPROCESS=ON -DSACCADE_ENABLE_MODEL_PRODUCER=ON \
  -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
  -DSACCADE_ENABLE_DETECTION_SESSION=ON \
  -DSACCADE_ENABLE_VIDEO=ON -DSACCADE_ENABLE_NVDEC=ON -DSACCADE_ENABLE_STREAMING=ON \
  -DSACCADE_TENSORRT_ROOT=/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/streaming --parallel 2

build/streaming/track_stream MODEL.engine input.mp4 tracks.txt --packets packets.bin
build/streaming/track_stream MODEL.engine 'rtsp://camera/path' tracks-rtsp.txt \
  --rtsp-transport tcp --timeout-ms 5000
build/streaming/track_stream MODEL.engine /dev/video0 tracks-camera.txt --format v4l2
```

engine 必須符合 profile；其他模型使用 `--profile t01-640x640`。`--decode cuda` 是 CLI 預設，`--decode cpu` 明確選用 CPU；`--device N` 同時綁定 decoder 與 tracking device。`--frames N` 限制交付幀數，`--serial`／`--no-cuda-graph` 提供比較入口，`--epoch N` 指定序列身分。輸出須為新路徑；實際建立時使用原子 exclusive create，同時啟動的程序也不覆寫既有檔案。攝影機必須先在執行環境可見，例如 Linux `/dev/video0`；未壓縮的攝影機輸出須顯式選 CPU 入口，本版 GPU 入口接受 NVDEC 可解碼的壓縮格式。

## 時間、背壓與生命週期

來源與結果不按整部影片累積；一幀解碼後才送出，CLI 一格 result ring 隨即交付，輸出每幀寫入檔案。FFmpeg 的 codec／demux 自身仍有有限緩衝。CLI 不主動丟幀、不自動重連；負載超過來源速度時，底層攝影機／網路緩衝可能增加延遲。

每個 source／session 固定尺寸；尺寸改變、來源錯誤或逾時會失敗，已提交結果盡量 drain，並 checked close。`--timeout-ms` 限制來源開啟與每次 read 的 FFmpeg I/O 等待，依賴該 protocol／device 的 interrupt callback 支援，不是 CUDA 計算或輸出寫檔的硬期限。

EOF 先 drain decoder 和 session；Ctrl-C／SIGTERM 停止接收並 drain 已提交結果，取消退出碼分別為 130／143。開啟來源期間的取消受開啟 timeout 約束。錯誤退出碼為 1。輸出是可即時讀取的部分前綴，只有正常 EOF／frame-limit 與成功退出表示所請求範圍完成；取消或錯誤不會把部分檔案偽裝為完整序列。

`pts_us` 是來源時間戳，可能不存在或不規則。tracker 仍每個解碼幀執行一次固定步長 predict；時間戳未改變 Kalman dt，不提供可變幀率時間補償。

## 即時輸出契約

CLI 明列 `output_profile=stream-observations/1`：每筆使用 packet 的穩定 `output_id`，依 frame／ID 排序，格式為 MOT 十欄文字，無序列尾插值與全序列首次出現 ID 重編號。它與 `track_sequence` 的離線 `v1-mot/1` 是不同輸出政策；需要精確比對時使用 `--packets` 的完整 12,368-byte ResultPackets。

失敗／skipped packets 可保留在 packet 檔，失敗後不新增有效 MOT rows。close／flush 檢查不等於 fsync 耐久性。CLI 隱藏 FFmpeg 原生日誌以避免 RTSP credential 出現在 URL 診斷中；可重用來源庫不改寫應用程式的全域 FFmpeg logger。

## 支援與驗證邊界

GPU 入口支援 NV12（8-bit）與 P010（10-bit）surface，CUDA 轉換政策為 `device-video-color/1`：BT.601／BT.709／BT.2020 NCL、full／limited range、以 `(x/2,y/2)` 取 chroma，clamp 後 round-to-nearest-even 成 RGB8，再套用原 F02。HDR transfer 與 BT.2020 constant luminance 明確拒絕；本版沒有 tone mapping、旋轉 metadata、去交錯或可變時間差追蹤。

host RGB 入口契約仍為 `host-rgb-input/1`；不同 decoder／chroma 取樣不宣稱與 nvJPEG 或 CPU FFmpeg 色彩轉換逐位元相同，不改寫 JPEG 模型訓練與原驗收來源。FFmpeg CUDA frame/context 的所有權與 stream 使用依[官方 CUDA backend](https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/hwcontext_cuda.c)及 [NVDEC 實作](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/nvdec.c)接入；實際支援以本機 build 與測試為準。

新增測試涵蓋來源解碼／尾幀／停止／失敗、RGB ownership 與 F02 數值，以及同一 nvJPEG RGB 的 JPEG／RGB session 完整 packet 比較。可用下列設定啟用 HR 真實影像的 GPU 比對（若有 Python interpreter，亦註冊 CLI 整合測試；runtime 不使用 Python）：

```sh
cmake -S . -B build/streaming \
  -DSACCADE_STREAMING_TEST_ENGINE=/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  -DSACCADE_STREAMING_TEST_SEQUENCE=/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP
ctest --test-dir build/streaming -R 'video_source|video_device|device_video|host_rgb_preprocess|tracking_rgb_session|track_stream_cli' --output-on-failure
```

初版 CPU decode 的歷史開發驗證：CPU／CUDA CTest **57/57 PASS**；最後來源改版重跑 **1/1 PASS**，CLI 測試 **1/1 PASS**（16 個案例）。來源端 137 checks 包含真實 loopback HTTP timeout／阻塞讀取取消；同一 nvJPEG RGB 的 9 幀 JPEG／RGB ordinary serial／Graph pipelined／perturbed 路徑完整 packets 相同，涵蓋 ring 背壓、取消與失敗。影片 CLI 的 9 幀 Graph／ordinary serial packets 與 MOT 相同，另驗幀數限制、檔案保護、SIGINT 130／SIGTERM 143；這些結果不直接當作 NVDEC 驗證。

新增 RGB 前處理的 memcheck／racecheck／synccheck／initcheck 各 **3,478 checks、0 errors／hazards**，使用 `--no-held-stream`，sanitizer 未執行 host callback 擋住 stream 的測試；正常 CTest 已涵蓋該 gate。此範圍只有 RGB upload／F02，並非 TensorRT／NVOFA／完整 session sanitizer 結案。

NVDEC 接入後的新建置為 `build/streaming-nvdec/track_stream`。**61 項已配置 CTest 分批全通過**：新 device 前處理與 session 2/2、其餘回歸 58/58、修正測試影片 metadata 後 CLI 1/1（32 個案例）。NVDEC source 326 checks 通過，H.264、HEVC 8-bit／10-bit 各 9 幀含 B-frame；下載至測試 oracle 並對齊 plane layout 後，CPU decode RGB reference 的 max／mean error 均為 0。生產路徑不做此測試下載。

同一 CUDA 轉換後 RGB 的 host／device serial、device Graph pipelined 與 perturb 完整 packets 相同，來源 close 後仍有效的 surface lease、cancel、尺寸失敗均通過。真實 MOT17-10 前 9 張 JPEG 另外編為三種壓縮影片，Graph／ordinary serial 的 packets 與即時 MOT bytes 相同，並確認有實際追蹤 rows；沒有讀 GT 或重新計分。

新增 device 色彩轉換／F02 的 memcheck／racecheck／synccheck／initcheck 各 **3,345 checks、0 errors／hazards**；sanitizer 使用 `--no-held-stream`，ready-event 擋住 producer 與 owner lease gate 在正常測試執行。獨立 scalar binary64 RGB reference 的差異門檻為最多 1 byte，以涵蓋 FP32 計算在量化邊界的差異；這不是 CPU／GPU decoder 的共同色彩取樣政策。此 sanitizer 範圍不含 NVDEC、TensorRT 或 NVOFA 內部 kernels。

H.264 前 3 幀的 Nsight CUDA／nvvideo capture 實際觀察到 `cuvidDecodePicture` 5 次、`cuvidMapVideoFrame64` 3 次與 D 上 3 次 CUDA 色彩轉換；額外 decode 為呈現前緩衝。從第一個色彩轉換 kernel 起，**H2D 為 0，D2H 只有 3×12,368-byte ResultPacket 加 3×56-byte MotionContext**。decoder stream 仍有 GPU 內 surface 複製；初始化的 engine uploads 在該邊界之前。此 trace 驗證本次執行的影像資料搬移，未量測 NVDEC engine overlap，也不代表函式庫內部 race safety。

交付前檢查另外修正 CLI 同時建立輸出的競態與來源開啟期間取消的退出碼，並強化 source frame 最後 owner 在不同 thread 釋放的測試。最後 CLI 新建置在 `build/streaming-review/track_stream`，35 個 CLI 案例通過；新 source 測試 329 checks、預設 CPU CTest 20/20 通過。原 `build/streaming-nvdec/` binary 與開發紀錄保留。完整來源、測試摘要、修正與未涵蓋項目見[開發驗證與交付檢查](../evidence/streaming/2026-11-05-development/REPORT.md)。

初版本機紀錄在 `build/streaming/VALIDATION.md`，NVDEC 接入的本機 capture 在 `build/streaming-nvdec/`（build 目錄不入 Git）。攝影機及外部 RTSP 裝置的實際連線須依可用來源另測。上述 GPU 量測使用 `source_dirty=true` 的提交前開發來源，不是 clean-source 正式驗收，不關閉既有 nvJPEG／NVOFA sanitizer 的 OPEN 診斷。
