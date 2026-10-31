# F01 host／private provider 的可觀察邊界（2026-10-31）

**已把本機 HARDWARE batch-1 的實際呼叫路徑追到 `NVJPGV2BatchedDecoder` → `libnvcuvid` 私有介面 → 真正 WSL driver `libcuda.so.1.1+0x4c7a50`。另從既有 trace 找到 converter stream → event → caller stream 的配對依賴。閉源 driver 操作的硬體完成／可見性保證，以及 initcheck shadow-state 轉換仍未證明。**

這是獨立診斷及歷史 trace 的離線分析，沒有改 Saccade runtime、nvJPEG、政策或 sanitizer suppression；F01 原始 initcheck FAIL／原因 OPEN、P01 OPEN 均保留。CPU debugger 會改變 host timing，本次 normal decode 不能代替未插斷點的 runtime／sanitizer 驗收。

## 1. 身分與捕捉

- nvJPEG：`/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58`，SHA-256 `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338`，與原始 F01 證據相同。
- 實際 provider：`/usr/lib/wsl/lib/libnvcuvid.so.1`，SHA-256 `eef76fcf52bb7a594e6ef4dcc70f0e2579c183240037a67099646911acc273cf`。
- 受測 standalone：原 `build-f01-investigation/repro`，SHA-256 `58e2c0ea589fd64ef9e334793c2af8230f4e580d6a303888de769313b5da27c3`；無重編。fixture 與 2,835-byte 獨立 planar RGB reference 沿用原 35×27 baseline。
- [capture/analysis.json](capture/analysis.json) 綁定靜態 library／工具／trace 身分；[gdb-capture-4/status.json](gdb-capture-4/status.json) 保存完整 debugger argv、60 秒 timeout、至少 4 GiB MemAvailable 預檢、source／binary／已列 library 的前後 hash、實際退出碼及 observations。
- 成功 capture 的 debugger exit **0**，明確 inferior exit **0**，全部 API／cleanup 成功；同圖兩次 RGB exact、repeat exact，完整輸出 SHA-256 `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162`。
- 在 decode 斷點擷取實際 `/proc/PID/maps`，對 mapped file 當時重新計算 SHA-256。資料包含 WSL `libcuda` stub **及** `/usr/lib/wsl/drivers/.../libcuda.so.1.1`，還有 `libdxcore`、`libnvcuvid`、nvJPEG；不將 stub 的 hash 冒稱為所有 driver 元件。這些是 mapped file hashes，不是全部 live executable pages 的逐 byte dump。

## 2. 真正執行到的 dispatch

所有下列數值均為對應 hashed ELF 的相對位移；live ASLR 位址保存於 log，不能跨 process 當作固定地址。

`nvjpegDecodeBatched` 的正常 dispatch 位於 nvJPEG **`0x73905`**：

```text
handle+0x88 -> backend object
backend object -> vtable address point
call [vtable+0x38]
```

兩次實際 decode 的 RTTI 名稱均為：

```text
N6nvjpeg18decoder_hw_batched21NVJPGV2BatchedDecoderE
```

實際 vtable address point 為 nvJPEG **`0x809c78`**，slot `0x38` → **`0xd0900`**。這和 RTTI／`R_X86_64_RELATIVE` 的靜態重建相同；`0xd0900` 是 wrapper，繼續呼叫 vtable slot `0x40`，此表對應 **`0xd4bf0`**。這補上了原先僅有兩個 hardware backend 候選、尚未辨認實際 dispatch 的缺口。[靜態 dispatch](capture/commands/decode-batched-dispatch.stdout)、[wrapper](capture/commands/nvjpeg-v2-decode-wrapper.stdout)、[動態 log](gdb-capture-4/stdout.log)、[離線核對](gdb-analysis.json)。

## 3. 私有 provider 與下一個明確邊界

nvJPEG `DT_NEEDED` 沒有 `libcudart`／`libcuda`；它的 loader 會在 **`0x6fab2`** 呼叫 `dlopen("libcuda.so.1",2)`，在 **`0x6fae6`** 呼叫 `dlopen("libnvcuvid.so.1",2)`，再從後者解析 **`__std_2U4S4U_X08`**，於 **`0x6fbbb`** 呼叫 getter 填入函式表。另有 `cuGetProcAddress[_v2]` 的解析與函式指標快取路徑。[loader](capture/commands/nvjpeg-private-loader.stdout)、[CUDA resolver](capture/commands/nvjpeg-cuda-proc-loader.stdout)。因此，只有攔一般 `cuda*` PLT imports 不足以保證覆蓋此路徑。

本機 `libnvcuvid+0x4d220` 的 getter 會填入 21 個可由反組譯重建的 slots；其中 `0x70` 指向 **`libnvcuvid+0x4ce10`**。成功 GDB capture 直接觀察 getter 填表，並命中這個 slot，stack 為：

```text
libnvcuvid +0x4ce10
  <- nvJPEG +0xd2fd6  （call site 是 +0xd2fd2 的 call [r15+0x70]）
  <- nvJPEG +0xd4ffb
  <- nvJPEG +0xd0913
  <- nvjpegDecodeBatched +0x99
  <- standalone main
```

getter 缺 debug return type，GDB `return_value` 為 `None`；本報告只主張已看到正確填表及後續使用，不補造 typed getter return code。

`libnvcuvid+0x4ce10` 先檢查 object，tail-jump 至 **`+0x4e000`**；後者讀取 `[object+0x20]` 的 inner object，於 **`+0x4e028`** 呼叫 inner vtable **slot `0xb8`**，其中 `r8d=5`，隨後清零原第二參數所指結構的 `+8` 欄位。[provider wrapper](capture/commands/nvcuvid-private-wrappers.stdout)、[inner call](capture/commands/nvcuvid-private-slot70-inner.stdout)、[nvJPEG callsite](capture/commands/nvjpeg-observed-slot70-callsite.stdout)。

這把第一輪可觀察邊界縮到一個具體的 private virtual call；後續 §3.1 已辨認這次實際 inner callee，尚未證實其同步保證。`libnvcuvid` 的 `DT_NEEDED libdxcore.so` 是 WSL bridge 的靜態依賴；單靠依賴或這次 stack 不能推導 Windows KMD／硬體排程的語義。

### 3.1 一次窄範圍 inner call 對照

另一次 [gdb-inner-capture](gdb-inner-capture/status.json) 在同一原始 standalone／fixture、相同 60 秒與 4 GiB 限制下，以 one-shot CPU breakpoint 捕捉 `libnvcuvid+0x4e028` 的 inner virtual call。debugger／明確 inferior exit 均 **0**，兩次 RGB exact、完整 cleanup，沒有新增 sanitizer 或 runtime synchronization。

實際 slot `+0xb8` 的 callee 是同一 hashed provider **`libnvcuvid.so.1+0x17ee0`**；呼叫後 `+0x4e02e` 的 **EAX=0**。provider object 的 RTTI 不可用，typeinfo 解參考失敗已記為 `rtti_error`，不補造類別名稱。[原始 log](gdb-inner-capture/stdout.log)、[離線 10 項核對](inner-analysis.json)。

本次 callsite 的 **RCX=0**，所以 [實際 callee 反組譯](inner-static/actual-inner-callee.stdout) 可進一步限定所選分支：

```text
libnvcuvid +0x17ee0
  CL==0 -> +0x17f30
  read DWORD [第三參數+0x30]
  table = [innerObject+0x20]
  +0x17f53: call [table+0x40]
  on EAX==0: clear WORD [第三參數+0x1e]
```

`+0x17f53` 所呼叫的 private table slot `+0x40`，已由 §3.2 的單一 edge capture 辨認至 driver；沒有藉此把 R8=5 或無名稱結構欄位解釋為已知同步協定。library 雖有 D3DKMT wait／signal／submit imports，鄰近其他分支也使用它們，但不可將鄰近分支當成這次 CL=0 的已執行路徑。`objdump` 顯示的最近 exported-symbol labels 也不等於本函式的真實名稱。

### 3.2 已確認跨入真正 WSL driver 的單一 edge

[gdb-driver-edge-capture/status.json](gdb-driver-edge-capture/status.json) 另以同一 standalone／fixture，one-shot 捕捉 `libnvcuvid+0x17f53` 的 table `+0x40` 呼叫。debugger／明確 inferior exit 均 **0**，兩次 RGB exact、完整 cleanup、沒有 timeout。這次只多觀察一個 edge，未執行更深的 GPU 探查。

actual target 為：

```text
/usr/lib/wsl/drivers/nvamsi.inf_amd64_e6537d0cc841801c/libcuda.so.1.1
  +0x4c7a50
SHA-256 70d8cd3774f4b0cf9359af3ca894f8e2586687cdc4c1fbb63af786fb69d90354
```

這是實際 mapped target file 的路徑／hash／offset，與 decode entry 的 maps 身分相同，**不是** `/usr/lib/wsl/lib/libcuda.so.1` stub。GDB 沒找到 exported symbol；不能因 `objdump` 的最近符號寫成 `cuVDPAUCtxCreate+...`，就把它命名為該 API。呼叫後 `libnvcuvid+0x17f56` 的 **EAX=0**；raw 參數保存於 log，其中第二參數為 `0x40005300`，其物件／同步語義仍未定名。[原始 log](gdb-driver-edge-capture/stdout.log)、[離線 12 項核對](driver-edge-analysis.json)。

[實際 driver callee 反組譯](driver-edge-static/actual-driver-callee.stdout) 顯示：`+0x4c7a50` 先呼叫內部 helper `+0x1303fa0` 取得結構指標並檢查狀態，成功路徑把該 32-bit 參數作為單元素資料交給 **`+0x4c76e0`**。該 [直接 helper](driver-edge-static/direct-driver-helper.stdout) 有額外狀態分支及內部操作，不含足以辨識完整同步契約的 public API 名稱。未再對其分支加入 GPU breakpoint；不宣稱靜態存在的每條分支都在這次執行。

**最終可觀察邊界已跨入閉源 driver。** 已定位到具體檔案、指令與成功回傳；該操作何時代表硬體 producer 已完成、如何保證 surface 對 SM 的可見性、及 sanitizer 如何更新初始化狀態，仍沒有可由這些 artifacts 證明的契約。`EAX=0` 與輸出 exact 不能代替這份完成／可見性證明。

## 4. 已有 trace 的 converter → caller 依賴

[capture/trace-analysis.json](capture/trace-analysis.json) 用唯讀 SQLite 重新分析已封存 `surface-poison/raw/nsys-rgb.sqlite`，保存查詢、全部相關 rows 與精確 correlation。這次未重跑 profiler，亦不補造原 scratch source／command 身分。

| repeat | converter kernel | record | caller wait | inspector |
| --- | --- | --- | --- | --- |
| 0 | corr 770，stream 16 | corr 772，stream 16，event 2／sync 1 | corr 773，stream 13，同 event 2／sync 1 | corr 778，stream 13 |
| 1 | corr 785，stream 16 | corr 787，stream 16，event 2／sync 2 | corr 788，stream 13，同 event 2／sync 2 | corr 791，stream 13 |

配對同時核對 process、context、event ID、event sync generation；不是單看同名 API 或相近時間。這支持「記錄了一條 converter stream 到 caller stream 的 event 依賴」，把剩餘疑問集中到 **hardware producer → converter**；不表示已證明 hardware completion。

該 trace 的限制不能忽略：

- `META_DATA_CAPTURE` 明列 **`CUDA_SKIP_SOME_API_CALLS=true`**，diagnostic 說明使用 CUDA hardware tracing。
- 沒有 `CUPTI_ACTIVITY_KIND_DRIVER`／`CUDA_CALLCHAINS` table；所有 runtime `callchainId` 為 NULL。因此，沒有某筆 driver／private API row 不能證明沒有執行該操作。
- CUDA event completion `timestamp` 均為 **0**。第二次 converter GPU start 為 `810906066 ns`，caller stream-sync API 的 host end 卻為 `810900402 ns`；本報告不把這些 host/GPU timestamps 當作精確共同時鐘或完成時間證明，也不據此判定真的發生違序。
- GPU-visible H2D／converter／inspector／D2H rows 沒有揭露 private surface 的硬體寫入或 driver fence；缺少可見 producer 不等於證明 NVJPG 是唯一可能寫入者。

## 5. 保留的準備失敗

| 目錄 | 實際結果 | 使用範圍 |
| --- | --- | --- |
| `gdb-capture/` | debugger exit 1；`repro.tested` 沒 executable bit，startup code 126 | 未啟動 target／GPU，純準備失敗；未 chmod |
| `gdb-capture-2/` | debugger exit 0，但錯將 symbolic breakpoint 的 post-prologue PC 當 entry，dispatch 觀察 Python memory error；無 target completion／RGB file | getter／一次 private stack 的部分診斷，**不列成功** |
| `gdb-capture-3/` | debugger exit 1；dispatch 正確、兩次 RGB exact 且已寫出 file，但 private raw breakpoint 遇 provider dlclose，debugger 不能重插，target exit 未證實 | 部分診斷，**不列成功** |
| `gdb-capture-4/` | 改用 symbol entry 計算 dispatch，private slot breakpoint 一次觀察後停用；debugger／inferior exit 均 0，全 cleanup 完成 | 本報告 normal 診斷結果 |

各次保存當次 `observe.gdb` bytes／命令／退出與原始 stdout／stderr；不把後來修正的 script hash 套入舊 attempt。後續 §3.1–3.2 的獨立 GPU captures 另存，沒有混成同一次執行。Debugger 斷點會改變 host timing，沒有宣稱這是原本未受干預排程的形式化證明。

## 6. 重驗

從 repository root，CPU-only 靜態／歷史 trace 重驗使用**新目錄**：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/host-static/capture.py \
  --out build-f01-host-static-NEW
```

固定 offset／RTTI 是對這次 hashed binary 的控制，版本變動時會拒絕沿用。實際 normal GPU diagnostic 的入口為：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/host-static/run_gdb.py \
  --out build-f01-host-gdb-NEW
```

後者要求舊 standalone executable 存在且 SHA 相同，不重編、不加 synchronization、不跑 sanitizer；保存 debugger 與 target 各自結果。只有 debugger exit 0 不足以判成功，還須明確 target exit、RGB 全 bytes、完整 result 與無 Python 例外。

已封存 capture 的離線交叉核對：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/host-static/analyze_gdb.py
```

此核對驗證 10 個身分／offset／填表／完整退出／RGB 條件，不執行 GPU。硬體 producer 的 submit／完成／可見性語義與 initcheck shadow-state 轉換仍為 OPEN；不因 converter→caller event 配對或本次正常輸出就改寫原 initcheck FAIL。

§3.1 的 GPU runner 是 `run_inner_gdb.py --out NEW`（使用獨立 `observe_inner.gdb`），已封存結果的離線核對為 `python3 evidence/preprocess/2026-10-31-f01-boundary/host-static/analyze_inner.py`。§3.2 的單一 edge runner 是 `run_driver_edge.py --out NEW`（使用 `observe_driver_edge.gdb`），離線核對為 `python3 evidence/preprocess/2026-10-31-f01-boundary/host-static/analyze_driver_edge.py`。兩者都要求新目錄，沿用相同 timeout／memory／typed target exit／完整 RGB／前後 hash 控制。
