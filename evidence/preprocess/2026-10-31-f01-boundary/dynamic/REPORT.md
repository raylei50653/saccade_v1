# F01 動態邊界：實際參數與 converter → caller 依賴

**三種配置、共六次正常 decode，均取得實際 driver launch 參數及成功的 converter → event → caller wait 提交鏈；RGB 全 bytes 相同。硬體 producer → converter 的完成／可見性語義仍未證明。** 本次使用 CUPTI，不同時執行 Compute Sanitizer，不改 F01 的 initcheck FAIL 或 P01 OPEN。

## 1. 身分與方法

成功 capture 是 ignored `build-f01-boundary-cupti-2/`，逐檔來源與壓縮映射見 [IMPORT.json](IMPORT.json)，完整小型輸出封存於 [capture-success/](capture-success/)。[identity.json](capture-success/identity.json) 核對：

- 原 standalone executable `58e2c0ea589fd64ef9e334793c2af8230f4e580d6a303888de769313b5da27c3`，與既有 SDK control 相同，沒有重編該 executable。
- 實際載入 nvJPEG `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338`，與靜態 SASS／歷史 initcheck library 相同。
- CUPTI `4c57c8ba32ef52c7426d93adb0318a4a234942c2e20955774077b5918825189d`；實際載入的 WSL driver `libcuda.so.1.1` 為 `70d8cd3774f4b0cf9359af3ca894f8e2586687cdc4c1fbb63af786fb69d90354`。沒有以 `/usr/lib/wsl/lib/libcuda.so.1` forwarding library 的 hash 取代實際 driver。
- 輸入、source、tracer binary、預先選定 SDK libraries 的前後 hashes 相同；實際 loaded-module maps 與 hashes 另存。新證據目錄當時是 untracked，這是診斷，不是新的 clean-source application acceptance。

[cupti_trace.cpp](cupti_trace.cpp) 是獨立 preload 診斷：訂閱 driver／runtime callbacks，在 callback 有效期內複製三個 by-value kernel args，保存 API enter／exit、correlation、stream／event、host stack module offsets。**Callback 裡沒有 CUDA API calls。**

只在 public `nvjpegDecodeBatched` 返回之後，由外層 wrapper 呼叫 `cuMemGetAddressRange`／`cuPointerGetAttribute`，查詢剛才實際 launch 所用 source 的範圍與 buffer ID。這個觀察點早於原 harness 的 `decode_sync`，不能稱為「GPU decode 已完成後」。沒有加入新的 GPU 同步、私有記憶體寫入、poison、NVTX exclusion 或 suppression；callback／host logging 仍會改變 timing。

三個 process 的全部 callback 記錄各自只有一個 TID。此實作及核對限這個單 host submitter、單 converter 的 harness，不推廣到 SDK worker／多 session 併發。普通與 extended kernel-argument paths 不可混用；本次實際使用 `cuLaunchKernel`＋非空 `kernelParams`，沒有 `extra` packed-buffer ABI。

## 2. 實際 converter source 與 allocation

| 正常配置 | 實際 source | pitch | allocation bytes | 兩次 API 返回後的 ID | RGB |
| --- | --- | ---: | ---: | --- | --- |
| DEFAULT＋PreAllocate | `0xf09e00000` | 512 | 16,384 | 987／987 | exact |
| NO_PIPELINE＋PreAllocate | `0xf09e00000` | 512 | 16,384 | 985／985 | exact |
| DEFAULT、不 PreAllocate | `0xf0a004000` | 512 | 16,384 | 984／984 | exact |

每次 driver callback 都直接複製出 width＝35、height＝27、source.channel[0] 與 pitch；destination 是另一塊配置，三個 plane 的 offsets 為 0／945／1,890、pitches 均為 35。Source 與 range query 的 base 相同，最後一個 32-bit load 的 exclusive end 為 base＋13,452，在 16 KiB 範圍內。

這補上「靜態地址式是否真的是 runtime 所用參數」的缺口。[完整核對](analysis.json)同時保存六個 snapshots／stack offsets。ID 的相同只涵蓋兩次 public API 返回後的觀察；**不證明 PreAllocate 到第一次 decode 之間沒有重配**。

16 KiB 是 allocation 大小，沒有識別為一個實體 page。35×27 共 945 次 32-bit pixel loads，涉及 3,780 個不同 bytes；其中 2,835 RGB bytes 進入輸出。其餘 pitch gaps／padding 的初始化狀態沒有由這次 trace 判定，不能把有效像素報告改寫成「只有頁尾未初始化」或「整頁都未初始化」。

## 3. 每次 decode 的成功依賴提交

每個 decode window 都包含以下 driver API，並核對所有 return status＝0：

```text
cuLaunchKernel(rgba2rgb, private stream S)
cuEventRecord(event E, S)
cuStreamWaitEvent(caller stream C, E)
nvjpegDecodeBatched returns success
```

六次皆是 S ≠ C；同一次 decode 的 record／wait 使用同一 event handle。每筆的 seq、correlation、source／destination、stream／event、API exits 都在 [analysis.json](analysis.json)。不是將 runtime／driver 的巢狀 callbacks 當成兩次操作，也不以跨 process 的位址相同拼接證據。

結合 CUDA stream／event 的公開契約，這支援 caller 下游在 C 上依序使用結果的設計；nvJPEG 可以在內部使用 S，再以 event 接回 C。它不是對 hardware producer → S 的獨立同步證明。CUPTI API callback timestamps 是 host 觀察／提交順序，沒有 GPU activity completion timestamp，不拿它當硬體完成時刻。

實際 converter launch stack 包含 nvJPEG offsets `0xd374f → 0xd4ffb → 0xd0913 → 0x73909`，與 [CPU GDB 實際 NVJPGV2 dispatch](../host-static/REPORT.md)相容。Private provider 的呼叫並不保證出現在一般 CUDA callback domain；沒有 producer callback 不等於沒有 producer／wait／allocation。

## 4. 結果、失敗與重驗

三組 GPU process 都 exit 0，兩次 decode 的 API／cleanup 成功，RGB 各 2,835 bytes、SHA-256 均為 `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162`，與固定獨立 djpeg reference 相同。這些是 standalone normal tracing 結果，不繼承 application suite 的 2,977 checks 或任何 sanitizer PASS。

準備失敗另外保留：`attempts/duplicate-header-compile.json.gz` 是 `cupti.h` 已含 generated metadata、重複 include 導致 compile exit 1；`attempts/preload-wrapper/` 是最初將 LD_PRELOAD 套到 time／timeout wrappers，第二個 constructor 拒絕同一 log 路徑，exit 91，target main 未執行。兩者不列 GPU／數值證據。成功 runner 將環境變數只套到 target process。

離線核對（不用 GPU）：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/dynamic/analyze.py
```

正常 GPU 重跑（須新的目錄及原 hashed standalone executable）：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/dynamic/capture.py \
  --out build-f01-boundary-cupti-NEW
```

每步至少 4 GiB MemAvailable、60 秒 timeout，保存 argv／cwd／環境覆寫、stdout／stderr／exit／RSS。新 tracer compile hash 必須隨 run 固定；不沿用舊 binary 的實測結論。
