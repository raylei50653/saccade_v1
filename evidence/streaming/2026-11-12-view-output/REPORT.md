# `track_stream --view`：單機開發驗證

日期：2026-11-12。結果：**全部指定 gates PASS（development verification，非正式 acceptance）**。本輪在 `feat/stream-view`、前次 evidence commit `015d1740a0756454afe1a6cad96f7ae6e8d9f113` 上完成；base 為 `3cbaa518f0dc37d691404e95ed85c0109c81130d`。只修改此 worktree，未 push／merge／rebase。

## 目的與設計

增加可直接觀看的 tracking 畫面：`NVDEC NV12 surface → encoder pool device copy → CUDA 方框／ID → h264_nvenc → RTSP/TCP 或本機容器`。保留現有 HR／G1／CUDA Graph、TrackingSession、VideoSource、tracking／preprocess kernels、預設值與 resolved-config schema。view 是純輸出選項，不改變 MOT 或 packet。

`VideoView` retain 裝置 primary context，FFmpeg 以 `AV_CUDA_USE_CURRENT_CONTEXT` 建立裝置並核對 context；使用自己的 non-blocking stream，先等來源 ready event，再複製 Y／UV，不改寫 decoder surface。每個 output sample 的 CUDA thread 依 ID 排序走過所有 outlines，再走 labels；chroma 依覆蓋任一 2×2 luma pixel 的最後元素決定，glyph 只改 Y。CPU reference 使用逐元素 painter，核對整個 allocation（含 pitch padding）。

encoder 為 `h264_nvenc`、p4／ll／zerolatency、bf=0、g=30、vbr、8M／12M；encoder time_base 為 1/1000000，framerate 30/1 只作 rate-control hint，PTS 一律為來源 `pts_us - first_pts_us`，缺失／不遞增即失敗。duration 使用已觀察的前一 PTS 間隔，尚無間隔時為一個 time-base tick，避免 MP4 將尾幀標為 discard；沒有合成 PTS。[FFmpeg NVENC 的 duration 傳遞](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/nvenc.c)及本機 `avcodec.h` 的 `AV_CODEC_FLAG_FRAME_DURATION` 契約為實作依據。

RTSP socket timeout 以既有 `--timeout-ms × 1000` 設定共享 RTSP `timeout` option，並用 interrupt callback 限制 open／write／trailer I/O；鍵的 microsecond 單位與 TCP 傳遞可見 [FFmpeg RTSP 實作](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/rtsp.c)。CLI 關閉 FFmpeg 全域 logger，所有自身診斷不含 view URL。檔案經同等輸出檢查後，以 O_EXCL descriptor＋custom AVIO 持有，不以 pathname 再次開啟。

VideoView 在 session 之後、resolved-config／MOT／packet 輸出之前開啟；P010 第一幀更早拒絕，CPU view 在 source open 前拒絕。每幀 receive 後核對 packet frame 與仍持有的 surface index；正常 ok packet 才寫 view。catch drain 不寫 view，view checked close 在 source close 之前。

## 結果與 gates

| Gate | 結果 | 數字／證據 |
| --- | --- | --- |
| 完整 preflight | PASS | 17 項；engine／source／修正後 MediaMTX config SHA-256 全符合；SDK／img1／工具存在，GPU 無競爭程序，8554 可用；`preflight.json` |
| 指定 configure／完整 build | PASS | `configure-command.sh`、`configure.log.gz`、`build-duration-flag.log.gz`；C++ 使用 -Wall -Wextra -Wpedantic -Werror，CUDA 使用 nvcc warnings-as-errors 與 host -Wall/-Wextra/-Werror |
| overlay CPU／GPU | PASS | 8 組逐 byte 相同，1920×1080／322×194，奇數 Y／UV pitches，0／9／5／256 boxes，負座標／外框／零面積／反向／重疊／頂端 labels／指定 IDs；`overlay.log.gz` |
| overlay memcheck | PASS | `compute-sanitizer --tool memcheck`，0 errors；未 kill；`overlay-memcheck.log.gz` |
| CLI 擴充 | PASS | 55 cases；NVDEC fixture（9 幀與單幀）及 MOT17 H.264 的 MOT／packet byte parity、H.264／尺寸／submitted 幀數、CPU／P010／碰撞／RTSP secret 隱藏；`cli-report.json`、`cli-logs.tar.gz` |
| 完整 serial CTest | PASS | 67/67，含 `track_stream_cli`（55 cases）、`tracking_p08_native_cli`（18 cases）；`ctest-final.log.gz` |
| 真實資料 file parity | PASS | 654 submitted／654 view decoded；MOT 與 packets bytes 完全相同；`file-parity.json` |
| Live loopback | PASS | exit 0，end=frame-limit，submitted=received=450；recording 301 decoded frames，10 秒 prefix decode exit 0、error log 空；`loopback.json` |
| Demo SIGINT | PASS | 約 20 s 後 SIGINT；script／tracker 皆 130，submitted=received=434，end=cancelled，MOT 356280 bytes；無 mediamtx／ffmpeg／track_stream child；`demo.json` |

GPU 工作依序執行並由 timeout 保護；未在 CTest、sanitizer、parity 或 live run 間重疊執行 GPU 工作。原 CLI 的 concurrent exclusive-create 測試保留其原本的競態案例。

## Timing

| 路徑 | 數字 | 範圍 |
| --- | --- | --- |
| 654 frames，無 view | wall_ms=6139.20 | source-open through output-close |
| 654 frames，有 view | wall_ms=6002.68 | 同上，含完整 view 輸出 |
| 450-frame RTSP loopback | wall_ms=19977.10；450/(wall_ms/1000)=22.5258 fps | 含 source open／等待 keyframe／初始化／drain；不是純 tracking throughput |
| view 每幀 | mean=1281.337 μs；p95=1538.360 μs；n=450 | host steady_clock：ready-event wait／pool acquisition 前，到 avcodec_send_frame 成功後；含 copy、box upload、overlay、stream synchronize；不含 packet drain／mux／tracking |

兩個 file run 各執行一次，包含初始化與暖機差異，不能從單次 wall_ms 差異推論效能改善。

Timing 使用同 build 目錄下獨立的 `timing/track_stream`，沒有 env 或 CLI hook，production binary 與來源不含 instrumentation。量測修改保留於 `timing-instrumentation.patch.gz` 供審查，temporary source／object／builder 已移除；binary 雜湊列於 `summary.json`。p95 使用 nearest rank（ceil(0.95×450)）。這是 host 觀察到的 view submit 路徑時間，不是 NVENC 硬體完成延遲或 viewer glass-to-glass latency。

## 媒體與重現

大型產物只留在 build，不提交影片或錄影；完整路徑、bytes 與 SHA-256 見 `media-manifest.json`。主要 view 為 `/home/ray/developer/ai/saccade_v1-worktrees/stream-view/build/stream-view/runs/r1/view.mp4`（21945458 bytes，SHA-256 `f888aeb6f33ff1d07a8a8b82e8522d820a041d6819de2706dc9d585289015ce1`）；live recording 為 `/home/ray/developer/ai/saccade_v1-worktrees/stream-view/build/stream-view/runs/live/rec.mp4`（10193730 bytes，SHA-256 `b85829e53aa58aa1d83c9b17daa4fb57235b84cdba4911d11fb35d7d2d81d86e`）。

`frame-180.jpg`、`frame-195.jpg`、`frame-210.jpg` 是零起始索引、ffmpeg JPEG q=3（約 q90），每張 ≤600 KB；供目視審查。實際 MediaMTX config 為 `mediamtx.yml`，與 committed demo config 相同；live 錄影為 `-c copy -t 10` 的有限前綴；publisher 以 `-re`、不 loop、`-c copy` 發布來源，使用既有 `--timeout-ms 15000` 等待長 GOP。demo 則按規格 `-stream_loop -1`。viewer 使用方式見唯一更新的文件 `docs/STREAMING.md`；未宣稱實際開啟 VLC 或 ffplay UI。

CLI／file／live 命令與結果分別在 `cli-report.json`、`file-parity.json`、`loopback.json`、`demo.json`；原始日誌以 gzip 保存，含失敗後修正的執行，沒有把失敗紀錄改稱 PASS。`summary.json` 記錄測試 binary 身分與 gate；`SHA256SUMS` 涵蓋本 evidence bundle（自身除外）。

## 過程修正

- 第一輪依原規格 `/etc/mediamtx.yml` 停止；這是規格路徑錯誤，正確參考為 `/etc/mediamtx/mediamtx.yml`，不是實作缺陷。
- 初次編譯修正 nvcc-generated line directive 與 host -Wpedantic 的衝突、測試 cudaMalloc 的 void** 轉型，以及 protocol uint64 output_id 轉為指定 ViewBox uint32 的範圍檢查；完整重建成功，保留 build*.log.gz。
- 第一輪 CTest 66/67；新增 CLI gate 發現 9 幀 MP4 的最後 sample duration 為 0，edit list 將其標 discard，僅 decode 8 幀。先補 duration 後仍失敗，進一步依本機標頭啟用 AV_CODEC_FLAG_FRAME_DURATION；9 幀及新增單幀回歸通過，CLI 全通過後重跑完整 CTest 67/67。原失敗見 ctest.log.gz、cli-r2.log.gz，修正後見 cli-r3.log.gz、ctest-final.log.gz。
- 第二次完整 CTest 因本輪第一輪留下的 ofa-audit-raw 目錄，觸發既有 ofa_stream_audit 的 new-directory 保護；將舊產物移至 runs 保存後再跑全套，未修改原測試；見 ctest-rerun.log.gz。
- 封裝檢查發現 MediaMTX v1.19.1 的 moq 預設啟用；補上參考設定中的 moq: false，重新執行 live 與 demo gates，確認只有 RTSP listener。前次錄影／日誌／自動產生的測試憑證留在 build 的 runs/before-moq-fix 與 live-before-moq-fix，沒有提交測試私鑰。
- evidence 封裝器原先依舊版 CTest 摘要文字判斷成功，本機版本摘要格式不同；改核對 67 筆 Passed 與沒有 Failed，不涉及測試 gate 變更。
- 錄影初次在 tracker 斷流時才結束，ffmpeg exit 0 且 counted 450 frames，但解碼日誌出現一筆 partial P-frame 錯誤。原先測試立即訂閱以排查起點，現象仍在；逐幀日誌定位於尾端，最後採 10 秒 bounded recording prefix（ffmpeg -c copy -t 10），保留 tracker 450-frame gate，並要求 prefix decode exit 0、error log 為空。沒有修改 encoder／relay／來源或隱藏前次日誌；見 recording-tail-decode.log.gz／recording-tail-details.log.gz。
- 實作檢查補上 view.write 失敗時同步其 stream，確保 caller unwind 前完成對 decoder surface 的讀取；demo 以 RTSP metadata 快速檢查可讀性，並在啟動 publisher 前等待 server listen，以避免 harness 自身啟動競態。

## 範圍外發現

- 固定 demo 來源的 keyframe PTS 為 0、5.8、12.3、19.666667 s，長 GOP 會使中途 RTSP 訂閱等待下一 keyframe；script 只使用既有 timeout 選項，不改來源、不轉碼、不增加 reconnect 或 retry。
- MediaMTX relay 在 tracker teardown 邊界的錄影曾出現尾幀部分解碼、首 packet timestamp unset／DTS 修正日誌；未定位為本庫、FFmpeg 或 relay 的已知根因，也未擴充 teardown／reconnect。最後 gate 使用明列的 10 秒錄影前綴；不主張錄影涵蓋整個 tracker 關閉邊界。
- protocol output_id 是 uint64，指定的 ViewBox 是 uint32；超過 UINT32_MAX 會明確失敗，本輪未更動 protocol 或 tracking ID 分配。
- 沒有實作 HUD、額外文字、音訊、多串流、P010／HEVC-10 view、reconnect／retry、CPU overlay 或 software encoder fallback。

## 限制與交付邊界

這是單一 RTX 5070 Ti Laptop／本機 driver、FFmpeg n9.0.2、MediaMTX v1.19.1 的 dirty development source 驗證，**不是 clean-source 正式 acceptance、外部攝影機驗收或 CI 結果**。build identity 的 parent 為前次 evidence commit；本次未 push／merge／rebase。既有 nvJPEG／NVOFA sanitizer **OPEN 不受影響**；本輪 memcheck 只涵蓋新的 overlay 測試。

NV12 only；來源 PTS 必須存在且嚴格遞增；本機檔案不 pacing；looped source 場景重設時 tracks 持續；view 錯誤 fatal；無 reconnect、retry 或 frame dropping。測試不量測完整 viewer latency，也不主張 RTSP 對外網路條件下的穩定性。
