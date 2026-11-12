# `track_stream --view`：環境前置核對停止報告

日期：2026-11-12。狀態：**STOPPED_ENVIRONMENT，尚未實作**。

## 目的與預定設計

依指定設計增加畫面輸出：NVDEC NV12 surface → 裝置複製 → CUDA 方框與 ID → NVENC H.264 → RTSP 或檔案。畫面輸出獨立於 tracking resolved config，須保持 MOT／packet bytes 相同；本輪未進入實作階段，沒有設計變更。

## 停止原因

指定必須從 `/etc/mediamtx.yml` 核對已安裝 MediaMTX v1.19.1 的合法設定鍵；該檔案不存在。`stat /etc/mediamtx.yml` 回傳 exit 1、`No such file or directory`，詳見 `preflight.log`。

依任務「anything else environmental: stop and report」規則停止，沒有以其他位置的設定、網路文件或猜測值代替，也沒有建立 demo 設定或啟動 MediaMTX。恢復該指定輸入後才能繼續本次工作。

## 結果與 gates

| 項目 | 結果 | 證據／數字 |
| --- | --- | --- |
| worktree／branch／初始 HEAD | PASS | `feat/stream-view`；`3cbaa518f0dc37d691404e95ed85c0109c81130d`；初始工作目錄乾淨 |
| HR FP16 engine SHA-256 | PASS | `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` |
| demo source SHA-256／metadata | PASS | `122283393b7d622559d9b85465c140d960e06b1fecdd0259e17188591acfb0ce`；H.264 High、1920×1080、30/1、654 frames |
| MOT17-10 `img1/`／TensorRT SDK | PASS | 兩個指定目錄皆存在 |
| 工具 | PASS | FFmpeg／ffprobe n9.0.2；列有 `h264_nvenc`；MediaMTX v1.19.1；nvcc V13.4.92；Compute Sanitizer 2026.3.0.0 |
| GPU／衝突程序／8554 | PASS | GPU 0 MiB、0% utilization、無 GPU 程序；無 `track_*`／`ctest`／`compute-sanitizer`／`mediamtx`；8554 無 listener |
| 指定 MediaMTX 設定來源 | **FAIL** | `/etc/mediamtx.yml` 不存在（ENOENT） |
| configure／build | NOT RUN | `configure-command.sh` 保存指定命令，未執行 |
| overlay CPU／GPU byte parity、memcheck | NOT RUN | 無實作，無 sanitizer 執行 |
| CLI parity／負面測試／完整 CTest | NOT RUN | 環境停止 |
| 654-frame 真實資料 parity／JPEG | NOT RUN | 無 MOT、packet、view 或 JPEG 產物 |
| 450-frame live loopback／錄影 decode | NOT RUN | 無 RTSP 連線或錄影 |
| demo SIGINT／130／程序清理 | NOT RUN | 未啟動 demo 或子服務 |

## Timing 與產物

無 tracking run，因此兩組 `wall_ms`、live FPS、view mean／p95 均為 N/A，不能以 0 代替。沒有影片或錄影，因此沒有可記錄的媒體路徑、大小或雜湊；沒有實際使用的 MediaMTX 設定與三張 JPEG。

本目錄保留 `summary.json`、`preflight.json`、`preflight.log`、`verification.log`、未執行的 `configure-command.sh` 與 `SHA256SUMS`。只提交此 evidence bundle，未變更程式、測試、demo script 或 `docs/STREAMING.md`，未建立 build 目錄，未 push／merge／rebase。

## 過程修正

封裝 evidence 時，第一次雜湊命令已在 evidence 目錄內，卻又使用相對於 worktree 的 evidence 路徑，造成 `FileNotFoundError`；已修正為目前目錄，重新驗證 JSON 與所有 evidence 雜湊。這是封裝命令的路徑錯誤，未影響前置核對結果。本次停止原因仍為環境缺少指定設定來源，未嘗試環境修復或放寬 gate。

## 範圍外發現

未進行額外研究、效能診斷或功能擴充。沒有 NVENC session creation 的執行證據。既有 nvJPEG／NVOFA sanitizer OPEN 項目不受影響。

## 限制

這是單機開發驗證的前置核對紀錄，**不是正式 acceptance**，也不是 `--view` 功能完成或可用的證明。
