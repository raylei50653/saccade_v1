# 本機重跑量測

以下在 `/home/ray/developer/ai/saccade_v1` 執行。保留既有部署 engine，不重新建 engine、不讀 GT、不改 tracker。每個輸出目錄必須是新路徑。所有 GPU 工作使用同一個 `~/.cache/saccade-v1/gpu.lock`。

## Build

本次 Release build 的完整設定在 [raw/CMakeCache.txt](raw/CMakeCache.txt)，使用 CUDA architecture 120、CUDA/TensorRT/model/preprocess/postprocess/detection session 全部開啟。新增工具由 `cmake/TrackingSession.cmake` 建置：

```sh
cmake --build build/resource-schedule-20261104 -j 8 --target tracking_benchmark resource_schedule_probe jpeg_stage_probe
```

## 主要延遲與吞吐

```sh
flock ~/.cache/saccade-v1/gpu.lock build/resource-schedule-20261104/tracking_benchmark \
  ~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP \
  build/NEW-latency --frames 120 --warmup 20 --rounds 2
```

這個工具逐session做ordinary/graph ABBA、分single/queued，保留首幀/capture/warmup，並比較全部packet bytes。不要把queued wall/frame當成單幀延遲。

## 階段與排程追蹤

```sh
flock ~/.cache/saccade-v1/gpu.lock nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --cuda-memory-usage=true --cuda-event-trace=false \
  -o build/NEW-trace build/resource-schedule-20261104/resource_schedule_probe \
  ~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP \
  build/NEW-trace-probe --frames 120 --warmup 20
nsys export --type sqlite -o build/NEW-trace.sqlite build/NEW-trace.nsys-rep
python3 -I scripts/analyze_stage_schedule.py build/NEW-trace.sqlite \
  build/NEW-stages --warmup 20 --frames 120 --plot-prefix build/NEW-schedule
```

不要開啟 `--cuda-event-trace=true`：本次初次診斷顯示Nsight警告可能引入假依賴，正式採用的trace已關閉。node tracing本身仍有overhead，因此階段表與未插樁wall benchmark分開解讀。圖檔轉PNG需 `rsvg-convert`。

## JPEG 獨立階段

```sh
flock ~/.cache/saccade-v1/gpu.lock build/resource-schedule-20261104/jpeg_stage_probe \
  /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP \
  build/NEW-jpeg --frames 120 --warmup 20
```

此工具重現NativePreprocessor的nvJPEG參數、容量、檢查與resize，在idle起點量F01至RGB交付，再獨立量F02。它用於隔離階段延遲，不是完整session，也不宣稱測到JPEG硬體engine獨占busy time。

## 在途幀數

`resource_schedule_probe` 的 `--only queued --result-slots 1|2|4` 可選收送深度，MI/CS仍各2slots。每組使用新輸出目錄，保存完整packet bins並跨組比較。範例：

```sh
flock ~/.cache/saccade-v1/gpu.lock build/resource-schedule-20261104/resource_schedule_probe \
  ~/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine \
  /home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP \
  build/NEW-depth2 --frames 120 --warmup 20 --only queued --result-slots 2
```

## 同步與安全檢查

本次每個命令的完整argv保存在 `raw/*.command.json`，正常與未完成結果分別在 `raw/validation-results.json`、`raw/bounded-validation-results.json`、`raw/component-race-results.json`。完整HRmemcheck包含所有函式庫；session synccheck只過濾自訂motion/tracking/control；F04和tracking/motion另跑獨立racecheck，positive controls須exit99且出現預期診斷。

Sanitizer須使用有上限的隔離service：MemoryHigh=6G、MemoryMax=8G、MemorySwapMax=0、timeout及RuntimeMaxSec，並用 `/opt/cuda/bin/compute-sanitizer` 絕對路徑（user service的PATH不同）。不要直接重跑無記憶體上限的完整session racecheck。本次無上限attempt曾耗盡WSL記憶體；有上限的session racecheck則150秒逾時，均不是PASS。

## 來源與證據

`source.sha256`記錄交付原始碼，`source.patch`包含測量所用dirty工作目錄相對HEAD的修改，`binaries.sha256`記錄本機build binaries。原始Graph/latency/session probe為v1；其後僅替probe新增 `--result-slots` 選項用於depth比較，v1/v2 binary與v1工具來源另存於 `build/resource-schedule-20261104/measured-binaries/`。runtime數值路徑沒有因報告量測而改動。

Nsight原始 `.nsys-rep`／SQLite和packet bins保留在 `build/resource-schedule-20261104/`；便攜的JSON/CSV/log/hash都已複製到本目錄。此為開發來源量測和同步審查，不取代clean-source正式驗收。
