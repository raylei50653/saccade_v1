# P01-A native TensorRT SDK 最小驗證

本工具只驗證指定環境的 C++ builder → checked save → checked reload → deserialize → 新 execution context → explicit stream enqueue 路徑。網路是新建的合成 `y = 2x + 1`，不是 YOLO、Mamba、scan 或 F03；結果不接入每幀資料流，也不提升真實模型 readiness。

## SDK 選擇與來源

固定 **TensorRT 10.16.1.11，CUDA 13.2 native package**。2026-10-28 從 NVIDIA 公開 Ubuntu 24.04 x86-64 repository 新下載以下兩個檔案，使用 `ar`／`tar` 解壓到獨立 SDK root，沒有安裝或執行 Python，也沒有讀取舊 Python 環境。這是 headers＋shared native runtime 的最小 SDK；不需要 plugin、ONNX parser、samples 或靜態庫。

來源共同前綴：`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/`。大小與 SHA-256 交叉核對該來源的 [Packages.gz](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/Packages.gz)。

| 檔案 | Bytes | SHA-256 |
| --- | ---: | --- |
| `libnvinfer-headers-dev_10.16.1.11-1+cuda13.2_amd64.deb` | 121306 | `8447e3fc65f46bf28907c7d03c39f3fa86ef2bf6ec46afb848ea6533fb7fcd65` |
| `libnvinfer10_10.16.1.11-1+cuda13.2_amd64.deb` | 1854823754 | `3d2caaf0796fab517fffea327727ef829f46510fb1cd6bcd0efc84cb3b1ef002` |

[NVIDIA 10.16.1 support matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html) 的內嵌 `supportData['10.16.1']` 列出 Linux x86-64 CUDA 13.x、最低 driver R580、SM 12.0。其 compiler/platform 表列出 Ubuntu／Rocky／Debian／SLES 及 GCC 11.2.1；本機 Arch Linux／GCC 16.2.1 不在該測試平台清單。因此本次是此實機的 native 實測，不宣稱整個本機平台獲 NVIDIA 官方認證。CUDA 13.4 compiler／runtime 與 driver API 的實際身分另存 execution evidence。

API 依據：[NVIDIA 10.x C++ API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/c-api-docs.html)。下載 package 的 `usr/share/doc/libnvinfer10/copyright` 標示 Apache-2.0，原文保留於 exploratory evidence；header 亦帶 Apache-2.0 標示。產品授權另見 [NVIDIA TensorRT License](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/license.html)。套件不加入 repository。

## 重跑

首次準備 SDK 可在獨立目錄執行下列命令；先核對上述官方 hashes，再解壓，無需系統安裝：

```sh
mkdir -p build-sdk-download/sdk
cd build-sdk-download
curl -fLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnvinfer-headers-dev_10.16.1.11-1+cuda13.2_amd64.deb
curl -fLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnvinfer10_10.16.1.11-1+cuda13.2_amd64.deb
printf '%s  %s\n' 8447e3fc65f46bf28907c7d03c39f3fa86ef2bf6ec46afb848ea6533fb7fcd65 libnvinfer-headers-dev_10.16.1.11-1+cuda13.2_amd64.deb 3d2caaf0796fab517fffea327727ef829f46510fb1cd6bcd0efc84cb3b1ef002 libnvinfer10_10.16.1.11-1+cuda13.2_amd64.deb | sha256sum -c -
for package in *.deb; do ar p "$package" data.tar.xz | tar -xJ -C sdk; done
cd ..
```

SDK root 可用 tar SDK 的 `include/`＋`lib/`，或本次 extracted Debian 的 `usr/include/x86_64-linux-gnu/`＋`usr/lib/x86_64-linux-gnu/`。CMake 預設 `SACCADE_ENABLE_TENSORRT=OFF`，CPU build 不搜尋 CUDA／TensorRT。開啟後只搜尋指定 root，不從 Python 或其他系統 SDK 補依賴；缺件、header／library 版本不符、CUDA 不在 [13.2,14)、driver API 低於 runtime、GPU 不可用或 SM < 7.5 均以明確錯誤停止。該 target 以 C++ compiler 編譯，不需 `enable_language(CUDA)`。

```sh
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_TENSORRT=ON \
  -DSACCADE_TENSORRT_ROOT=/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk \
  -DCUDAToolkit_ROOT=/opt/cuda
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
build-native/tensorrt_smoke build-native/test.engine build-native/test.txt
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-native/tensorrt_smoke build-native/memcheck.engine build-native/memcheck.txt
```

改變 source commit 後須重新 configure，讓 binary 內嵌的 commit／dirty 身分對應編譯來源。原始命令、exit code、完整輸出、SDK／GPU 身分與 hashes 見 P01-A evidence。執行報告包含 header 路徑與 hash、`dladdr` 核對的實際 nvinfer library（必須等於 configure 的 library）、`/proc/self/maps` 的實際 CUDA／TensorRT libraries、CUDA runtime／driver API、GPU 名稱／PCI bus／SM、compiler、engine hash 及全部 256 個輸出。OS、kernel driver、compiler 詳細身分另由執行命令保存。

## 固定測試契約與失敗邊界

在建立網路前已固定：shape `[1,1,1,256]`、strongly typed FP32、TF32 關閉、builder optimization level 0、builder workspace 上限 16 MiB、auxiliary streams 0、不匯入或匯出 timing cache。TensorRT 自行選定的 engine tactics 固定於所存 engine；不聲稱跨環境 engine hash 可重現。

`x[i]=(i-128)/8`，所有輸入及 `2*x+1` 在這個範圍都是精確可表示的 FP32。獨立 C++ scalar 計算 256 個預期值；容許誤差預先固定 `atol=0, rtol=0`，逐項拒絕非有限值或任何不符。host I/O 是 pinned allocation，device I/O／execution workspace 在 enqueue 前明確配置，輸出先填 poison，使用 nonblocking non-default CUDA stream 完成 H2D／enqueueV3／D2H／同步後才比較。詳細 engine inspector JSON 與 hash 記錄實際 fused layer／tactic 身分，builder 使用的 local timing cache 不匯入或匯出；loaded-library maps 在 context teardown 前擷取。此工具以 ELF `DT_RPATH` 供 TensorRT 尋找同一 SDK 的 per-SM builder resource，避免 `DT_RUNPATH` 的非遞移查找限制。

builder、network 在重載前已銷毀，保存的 plan 經 size／hash 讀回核對後反序列化；execution context 採 `kUSER_MANAGED`，明確提供 `getDeviceMemorySizeV2()` 要求的 workspace（零時不額外配置）。正常流程在同步後先銷毀 context／engine／runtime，再以共用 `CleanupReport` 檢查所有 CUDA free／destroy；失敗流程仍排空 stream 並嘗試所有獨立釋放，回傳非零。

engine、report 使用既有 checked I/O，stdout 亦 flush 並檢查；檔案或輸出失敗回傳非零。`native_tensorrt_cli_io` 驗證 usage、缺少 output parent、`/dev/full` engine／report／stdout 失敗。原子寫入、`fsync` 耐久性、惡意 engine 載入、scan、custom plugin、效能及完整模型數值比較都不在此最小工具的成功結論內。

## ONNX parser（T01 backbone）

2026-10-30 依使用者核准，為 T01 backbone／neck 的 ONNX 匯入（[BACKBONE_POLICY](BACKBONE_POLICY.md)）在同一 SDK root 追加**同版本**的 ONNX parser。來源同上（NVIDIA Ubuntu 24.04 x86-64 repository），大小與 SHA-256 交叉核對同日下載的 Packages.gz（SHA-256 `a2cd4fee1271f9be2e40b1db66540cccd19d6246e166ced3d613f0977066c8b5`）；兩個 package 都依賴 `libnvinfer10 (= 10.16.1.11-1+cuda13.2)`，copyright 為 Apache-2.0。

| 檔案 | Bytes | SHA-256 |
| --- | ---: | --- |
| `libnvonnxparsers10_10.16.1.11-1+cuda13.2_amd64.deb` | 1534774 | `cd379a1d03d92f104d5659e0cb62302a781a17806ba103c01a699da5c8648dd6` |
| `libnvonnxparsers-dev_10.16.1.11-1+cuda13.2_amd64.deb` | 2230514 | `9904cbbc32d4b0f8eb665ffd2758ee10b74a3a3afb19c8e164b2c4a83a98bcd1` |

只解出 shared library（`libnvonnxparser.so.10.16.1`，SHA-256 `2b517c9d21d90d8eef5c9c5b1b9acbe46c39ccd4bb36b927ef29776d5fde48e3`，與其 symlinks）、`NvOnnxParser.h`、`NvOnnxConfig.h` 與 doc；靜態庫不解。解壓使用 `tar --keep-old-files`；前後的 SDK 檔案清單與每個檔案的 SHA-256 已比對：原有 24 個項目逐 byte 不變，只新增 9 個（清單見 T01 F03 evidence）。parser 本身不連結 `libnvinfer`（`ldd` 只列 libc／libstdc++ 等），透過呼叫端傳入的 network 建圖。

```sh
cd /home/ray/.cache/saccade-native-trt/10.16.1.11/downloads
for p in libnvonnxparsers10_10.16.1.11-1+cuda13.2_amd64.deb libnvonnxparsers-dev_10.16.1.11-1+cuda13.2_amd64.deb; do
  curl -fLO "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$p"; done
printf '%s  %s\n' cd379a1d03d92f104d5659e0cb62302a781a17806ba103c01a699da5c8648dd6 libnvonnxparsers10_10.16.1.11-1+cuda13.2_amd64.deb \
  9904cbbc32d4b0f8eb665ffd2758ee10b74a3a3afb19c8e164b2c4a83a98bcd1 libnvonnxparsers-dev_10.16.1.11-1+cuda13.2_amd64.deb | sha256sum -c -
for p in libnvonnxparsers*.deb; do ar p "$p" data.tar.xz | tar -xJ -C ../sdk --keep-old-files \
  --exclude=./usr/lib/x86_64-linux-gnu/libnvonnxparser_static.a --exclude=./usr/lib/x86_64-linux-gnu/libonnx_proto.a; done
```

CMake 只在 `SACCADE_ENABLE_MODEL_TRT=ON` 時於 SDK root 內搜尋 parser（`NO_DEFAULT_PATH`），並核對 header 與 library 的版本；未開啟時 S01–S04 的建置不受影響。
