# NVOF 內部的 cuArrayGetPlane 失敗：最小獨立重現

**結果：** 106 行、只用 CUDA Driver API 與 NVOF 函式庫的程式，在 WSL2 和原生 Windows 上都重現相同次數。所有 CUDA 與 NVOF 呼叫都成功，只有 memcheck 回報錯誤。是否無害要等 NVIDIA 說明，狀態 OPEN，不算 PASS。

相同 GPU（RTX 5070 Ti Laptop）與 driver（617.42）。原生 Linux driver、其他 GPU 和其他 driver 版本都未測試。

| 設定 | WSL2 | 原生 Windows |
| --- | --- | --- |
| 無 sanitizer，2 次 Execute | exit 0 | exit 0 |
| `explicit`，只到 `nvOFInit` | 74 個，exit 99 | 74 個，exit 99 |
| `explicit`，1 次 Execute | 86 個，exit 99 | 86 個，exit 99 |
| `explicit`，2 次 Execute | 86 個，exit 99 | 86 個，exit 99 |
| `explicit`，30 次 Execute | 86 個，exit 99 | 86 個，exit 99 |
| `extended`，2 次 Execute | 172 個，exit 99 | 172 個，exit 99 |
| `--report-api-errors no`，2 次 Execute | 0 個，exit 0 | 0 個，exit 0 |

- 86 個錯誤分布為 `nvOFInit` 74 個、第一次 `nvOFExecute` 12 個，之後不再增加。
- `extended` 多出的 86 個是驅動訊息：`CUDA API Error: Check failed: Cannot use this API for non-planar arrays`。
- 每個堆疊都是 nvcuvid 在 NVOF 函式庫之下：WSL2 為 `libnvcuvid.so.1`、`libnvidia-opticalflow.so.1`，Windows 為 `nvcuvid.dll`、`nvofapi64.dll`。

## 為什麼判斷像刻意探測

`libnvcuvid.so.1`（SHA-256 `eef76fcf…`）的呼叫點反組譯顯示：回傳值有被檢查。plane 0 失敗時改用 array 本身，plane 1 失敗時結束迴圈並正常離開函式，每個 array 正好失敗兩次。這是閉源機器碼的判讀，NVIDIA 沒有證實，也沒有放進回報。全部 86 + 1008 個錯誤都來自同一個呼叫點（`0x15e3d`）。

## 檔案

- [repro_getplane.cpp](repro_getplane.cpp)：重現程式
- [run_wsl2.sh](run_wsl2.sh)、[capture/](capture/)：WSL2 腳本與紀錄
- [windows-native/run.cmd](windows-native/run.cmd)、[windows-native/capture/](windows-native/capture/)：原生 Windows 腳本與紀錄
- [nvidia-report/](nvidia-report/)：給 NVIDIA 的表單內容、email 與 zip，已回報為 [Bug 7048273](https://developer.nvidia.com/bugs/7048273)，尚無回覆
- [SHA256SUMS](SHA256SUMS)

較大的探測程式、CUPTI 紀錄和完整排查報告在 [investigation/](investigation/REPORT.md)。報告連到的其他 log 和輸出留在 `build/nvof-memcheck/investigation-20261105/`（Git 忽略），雜湊見 [investigation/not-copied.sha256](investigation/not-copied.sha256)。

## 重跑

```sh
flock ~/.cache/saccade-v1/gpu.lock bash run_wsl2.sh ../../../build/nvof-getplane-repro
```

`run.cmd` 要連同 `repro_getplane.cpp` 和 `third_party/nvof/` 的兩個 header 複製到 Windows 目錄後執行，Windows 上也要在 GPU 鎖內跑。
