# nvJPEG HARDWARE decode 不遵守 CUDA stream 順序（2026-11-03）

**結論**：`nvjpegDecodeBatched` 使用 `NVJPEG_BACKEND_HARDWARE` 時，會在同一條 stream 上排在前面的工作完成之前，就把解碼結果寫進輸出 buffer。

- 這和 nvJPEG 文件不符。文件寫的是「All GPU tasks for this function will be submitted to the provided stream」（§2.3.3.3），沒有提到任何例外。
- `NVJPEG_BACKEND_HYBRID` 遵守 stream 順序。
- 這個行為是在 M3 驗收時發現的，見 [TRACKING §14.1](../../../docs/TRACKING.md) 與 [M3 證據](../../tracking-session/2026-11-03-formal-e021bd1/REPORT.md)。本目錄以獨立程式直接證實它。
- 原生 Windows 執行檔結果相同，不限於 WSL2。
- 已回報 NVIDIA [Bug 7038860](nvidia-report/README.md)，尚無回覆。和 [Bug 7021871](../2026-10-31-f01-investigation/nvidia-report/README.md)（initcheck）是不同的問題。

## 重現程式

[`repro_order.cu`](repro_order.cu) 只用 CUDA 與 nvJPEG，不含 Saccade 程式碼。每個 trial 在一條 non-blocking stream S 上依序做：

1. 解碼 A 到 DST，host 等它完成。
2. 在 S 上排入 spin kernel（SPIN_MS），再排入 copy kernel，把 DST 複製到 SNAP。
3. 不做任何同步，直接呼叫 `nvjpegDecodeBatched(B → DST, S)`。
4. 同步 S，把 SNAP 下載回 host 比對。

依 stream 順序，步驟 2 的 copy 應該在步驟 3 的所有 GPU 寫入之前完成，所以 SNAP 必須等於 A。`control` 模式會在步驟 3 前呼叫 `cudaStreamSynchronize(S)`，用來確認比對方法本身沒有問題。

輸入是 [`generate.py`](generate.py) 產生的兩張 1920×1080 合成 baseline 4:2:0 JPEG（Pillow 12.2.0，quality 95）：

| 檔案 | SHA-256 |
| --- | --- |
| a.jpg | `690d3813a0db124098b0ff4df5fec1109c4b60208aa17fea3388559798f9d81b` |
| b.jpg | `91cdc26ac1e6036572415abe22652c0c487cd325bffe0f4bb70e512c0221ac49` |

建置與執行：

```sh
nvcc -O2 -std=c++17 -arch=sm_120 repro_order.cu -lnvjpeg -o repro_order
./repro_order hw default order 20 20 a.jpg b.jpg
```

`hw|hybrid` 選 backend，`default|no-pipeline` 選是否加 `NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE`，`order|control` 選模式；最後依序是 SPIN_MS、TRIALS 與兩張圖。exit code：0 表示全部 trial 都遵守順序，3 表示觀察到違反，1 表示 API 錯誤。

## 結果

環境：RTX 5070 Ti Laptop（sm_120）、driver 617.42、WSL2、CUDA 13.4（runtime 與 driver API 皆 13040）、nvJPEG 13.2.3。每組 20 trials，A 與 B 有 6,188,210 bytes 不同。

| 設定 | 違反順序的 trial | 呼叫 decode B 花的時間 | 紀錄 |
| --- | ---: | --- | --- |
| HARDWARE、預設、spin 20 ms | **19／20** | 約 0.4–0.7 ms | [log](capture/hw-default-order-20ms.log) |
| HARDWARE、`NO_PIPELINE`、spin 20 ms | **19／20** | 約 0.4–0.7 ms | [log](capture/hw-nopipeline-order-20ms.log) |
| HARDWARE、預設、spin 20 ms、`control` | 0／20 | — | [log](capture/hw-default-control-20ms.log) |
| HYBRID、預設、spin 20 ms | 0／20 | 約 20.6 ms | [log](capture/hybrid-default-order-20ms.log) |
| HARDWARE、預設、spin 1 ms | 0／20 | — | [log](capture/hw-default-order-1ms.log) |
| HARDWARE、預設、spin 0.2 ms | 0／20 | — | [log](capture/hw-default-order-0.2ms.log) |

- **SNAP 整張變成 B**：每個違反的 trial，SNAP 的 6,188,210 個相異 bytes 全部等於 B。也就是說，排在前面的 copy kernel 執行時，B 已經完整寫進 DST。
- **HYBRID 的差別在 host 端**：呼叫本身會阻塞約 20 ms，等於等到 stream 上先前的工作完成，所以順序正確。HARDWARE 的呼叫約 0.5 ms 就返回，寫入卻沒有排在先前工作之後。
- **`NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE` 沒有幫助。**
- **前面工作的長度決定會不會發生**：前面只有 1 ms 以下的工作時沒有觀察到違反，因為它在硬體解碼器寫入之前就做完了。這也解釋了平常為什麼看不到：Saccade 的 F05 只要約 0.1 ms。
- **每次 run 的第一個 trial 都遵守順序**：原因不明，未進一步追查。
- **測試範圍**：WSL2 與原生 Windows（見下節），原生 Linux 未測。2026-11-03 送出回報前在 WSL2 重新建置執行，HARDWARE 19／20、`control` 0／20、HYBRID 0／20，與上表相同；環境紀錄見 [wsl-identity.log](capture/wsl-identity.log)。

## 原生 Windows

同一份 `repro_order.cu`（SHA-256 相同）以 Windows 原生工具鏈建置：Windows 11 Pro 10.0.26300、WDDM、driver 617.42、nvcc 13.4.92、nvJPEG 13.2.3.58 redist DLL、VS 2022 Build Tools（MSVC 14.44）。工具鏈與 DLL 沿用 [F01 Windows 驗證](../2026-10-31-f01-investigation/windows-native/REPORT.md) 留在 Windows Temp 的副本；建置與執行腳本為 [build.cmd](windows-native/build.cmd)、[run.cmd](windows-native/run.cmd)，exe 與 DLL 的 hash 見 [hashes.log](windows-native/capture/hashes.log)。

| 設定 | 違反順序的 trial | 呼叫 decode B 花的時間 | 紀錄 |
| --- | ---: | --- | --- |
| HARDWARE、預設、spin 20 ms | **19／20** | 0.04–0.06 ms | [log](windows-native/capture/win-hw-default-order-20ms.log) |
| HARDWARE、`NO_PIPELINE`、spin 20 ms | **19／20** | 0.04–0.08 ms | [log](windows-native/capture/win-hw-no-pipeline-order-20ms.log) |
| HARDWARE、預設、spin 20 ms、`control` | 0／20 | — | [log](windows-native/capture/win-hw-default-control-20ms.log) |
| HYBRID、預設、spin 20 ms | 0／20 | 20.4–20.6 ms | [log](windows-native/capture/win-hybrid-default-order-20ms.log) |
| HARDWARE、預設、spin 1 ms | 0／20 | — | [log](windows-native/capture/win-hw-default-order-1ms.log) |

- 結果與 WSL2 一致，違反時 SNAP 同樣整張等於 B，第一個 trial 同樣遵守順序。HARDWARE 呼叫在 Windows 上返回得更快。
- `build.log` 的 MSVC C4819 警告與 F01 相同（code page 950），由 CP950 轉成 UTF-8 保存。`windows-env.log` 中 OS 名稱的本地化字串無法解碼，已改寫為英文並註明。

## 對 Saccade 的影響與處理

- **TrackingSession**：F05 是唯一在 F02 之後還讀 RGB 的工作，原本只靠 stream 順序保護。1c03e22 改為在 F05 之後記 `rgb_read_done`，下一幀解碼前由 host 等它完成。擾動測試在 F05 前固定插入一段 spin，用來重現這個問題。
- **NativePreprocessor**：自己的 F02 在下一次解碼前已有 host 等待，不受影響。
- **DetectionSession**：F02 之後不再讀 RGB，不受影響。
- **`motion_sequence`**：每幀都等 F05 完成，所以 M1 records 不受影響。
- **文件**：[PREPROCESS.md](../../../docs/PREPROCESS.md) 的 RGB 借用規則已改為 consumer 必須在下一次 `enqueue` 前執行完成。

## 回報 NVIDIA

2026-11-03 已回報為 [Bug 7038860](https://developer.nvidia.com/bugs/7038860)，附件 zip 寄至 NVSDKIssues@nvidia.com。送出的表單文字、email 與附件見 [nvidia-report/](nvidia-report/README.md)。回報附上 WSL2 與原生 Windows 的結果，並請 NVIDIA 確認 HARDWARE backend 是否應遵守 stream 順序；若不應遵守，請文件寫明。
