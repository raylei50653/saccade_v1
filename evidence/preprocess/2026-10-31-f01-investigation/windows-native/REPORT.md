# 原生 Windows F01 對照（2026-10-31）

**結果：原生 Windows 同樣重現 HARDWARE RGB 的 initcheck 1,890 errors／exit 99，位置仍為 `nvjpeg::rgba2rgb<3> +0x170` 的 3-byte read；RGB 與獨立 reference 逐位元相同，memcheck 為 0 errors。WSL 不是重現此問題的必要條件。**

本次由 WSL 啟動 Windows PowerShell，但編譯器、受測 `repro.exe`、nvJPEG DLL 與 Compute Sanitizer 都是 **Windows 原生 PE 程式**，執行於 Windows 驅動／WDDM。不是在 WSL Linux process 或 WSL Docker 裡跑 CUDA。PE dependencies 與實際 Windows host stack 見 [pe-dependencies.log](capture/pe-dependencies.log)、[windows-rgb-initcheck.stdout.log](capture/windows-rgb-initcheck.stdout.log)。

## 身分與配置

| 項目 | 觀察 |
| --- | --- |
| Windows | Windows 11 專業版，10.0.26300、64-bit；見 [windows-os.json](capture/windows-os.json) |
| GPU | RTX 5070 Ti Laptop、SM 12.0、PCI `00000000:01:00.0`、UUID `GPU-825ae551-ba16-539a-e444-22726b59ee9a`；Windows 與 WSL query 相同 |
| Driver | Windows NVIDIA-SMI／KMD 617.42，WDDM；原生與 WSL 都回報 driver 617.42 |
| CUDA compiler | Windows nvcc 13.4.92，build `38855100_0` |
| Sanitizer | Windows Compute Sanitizer 2026.3.0.0，build `38855100`，與 WSL 版本相同 |
| nvJPEG | Windows redist 13.2.3.58；每個 run 的 `nvjpegGetProperty` 回報 13.2.3，與 WSL 所用 library 版本對齊 |
| Host compiler | 現有 VS 2022 Build Tools 17.14.39，MSVC 14.44；原始 compile log 保存於 `capture/compile.*` |
| Source | 與先前 Linux standalone harness **同一份 bytes**，SHA-256 `c3b0bbf242218a22d86568b527df363978588f9ee1d56f7c65891e4e0cdbc977`；沒有加入 poison、NVTX、private pointer probe 或算法修改 |
| Windows executable | `repro.exe`，336,384 bytes，SHA-256 `857c79671c6556dd5fa5da0eafa0f8d88dc93e67ff89f47f57a52f50fc1e86eb`；受測檔另保存在 ignored `build-f01-windows/repro-tested.exe` |
| JPEG | 同一 repo 35×27 baseline 8-bit 4:2:0，1,124 bytes，SHA-256 `3dedf93555bdc2aff5aab2a608a99f69cd8d6cc5c0fd64d8d3404ebb2f78da56` |
| RGB reference | 同一獨立 djpeg nosmooth planar reference，2,835 bytes，SHA-256 `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162` |
| Windows nvJPEG DLL | app-local `nvjpeg64_13.dll`，SHA-256 `a1b26f49c3aa0ef6107d70f3588089aad05720209c9cbe16f5f5f7526df83b9c` |

Windows 原先沒有 CUDA Toolkit，使用 NVIDIA 官方 [CUDA 13.4.2 redistrib manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.2.json)取得可攜套件，解壓到獨立測試目錄；未做全域安裝、未更換系統驅動。七份 archive 的 NVIDIA SHA-256 與實際下載值全部相同；URL、版本與 hashes 在 [sdk/SDK.json](sdk/SDK.json)、原始 manifest、`sdk/records/`。元件包括 nvJPEG、cudart、sanitizer、nvcc、CRT、CCCL 與 NVVM。

nvJPEG／cudart DLL 放在受測 exe 旁，工具與 DLL 路徑另加入該測試 process 的 PATH。nvcc 預設使用靜態 cudart；PE direct imports 僅列 `nvjpeg64_13.dll` 與 `KERNEL32.dll`，所以 adjacent cudart DLL 不代表受測 exe 有直接使用它。`artifact-identity.json` 記錄的是磁碟 artifact hashes，沒有逐一擷取 target process 的 loaded-module 完整路徑與 hash。native stack 中另有 `nvcuda64.dll`／`nvcudart_hybrid64.dll`，不將 `System32/nvcuda.dll` 的 hash 冒稱為所有 driver components 的 hash。

## 執行與控制

11 個 GPU commands 均各自啟動新 process、依序執行；每步至少 4 GiB Windows free physical memory、90 秒 timeout，全部正常結束而非 timeout。主控制採 HARDWARE、DEFAULT flags、PreAllocate、未初始化 caller output、解碼兩次。

| Native Windows 控制 | 實際 exit | sanitizer errors | 其他結果 |
| --- | ---: | ---: | --- |
| HARDWARE RGB，normal | 0 | 不適用 | RGB exact、repeat exact |
| HARDWARE RGB，memcheck | 0 | 0 | RGB exact、repeat exact |
| HARDWARE RGB，initcheck | **99** | **1,890** | RGB exact；SDK `rgba2rgb<3> +0x170`，3-byte reads |
| 跳過 decode、未初始化 output，memcheck | 0 | 0 | 僅工具控制，沒有 RGB 等價 |
| 跳過 decode、未初始化 output，initcheck | 99 | 2,835 | 自訂 `inspect_caller_output` 正確抓到未初始化讀取 |
| 跳過 decode、清零 output，initcheck | 0 | 0 | 負控制，沒有 RGB 等價 |
| RGB，不 PreAllocate，initcheck | 99 | 1,890 | RGB exact |
| RGB，NO_PIPELINE，initcheck | 99 | 1,890 | RGB exact |
| HARDWARE RGBI，initcheck | 99 | 1,890 | canonical planar RGB exact |
| HARDWARE Y，initcheck | 99 | 108 | repeat exact，未做獨立 Y 數值驗收；SDK cudaMemcpy source 報告 |
| HYBRID RGB，initcheck，僅診斷 | 0 | 0 | 每次 93 bytes 不同、最大 2 LSB，不符合指定 RGB 契約 |

所有 API 與 cleanup 回傳值為成功；HW RGB／RGBI 完整輸出與 reference hashes 相同。正控制說明 Windows initcheck 確實有檢查自訂 kernel 的未初始化讀取；不把 SDK 失敗當成工具沒啟動，也不以 HYBRID 取代指定 HARDWARE runtime。

## 與 WSL 的對照

| 項目 | 既有 WSL 結果 | 本次 Windows 原生結果 |
| --- | --- | --- |
| Source／fixture／RGB reference | 固定 hashes | 相同 bytes |
| nvJPEG／nvcc／sanitizer 版本 | 13.2.3／13.4.92／2026.3.0.0 | 相同版本；各 OS 使用自己的 build |
| HARDWARE RGB memcheck | 0 errors、exit 0 | 0 errors、exit 0 |
| HARDWARE RGB initcheck | 1,890 errors、exit 99 | 1,890 errors、exit 99 |
| 已印出 RGB 診斷 | `rgba2rgb<3> +0x170`，3-byte reads | 相同 kernel／offset／read size |
| RGB bytes | CPU reference exact | CPU reference exact |
| Y initcheck | 108 errors | 108 errors |
| Inspector 正／負控制 | 2,835／0 errors | 2,835／0 errors |

這個跨環境重現排除「只有 WSL 才會出現」的解釋，並支持問題位於兩者共用的 nvJPEG hardware 路徑／sanitizer／Windows driver 相關邊界。**尚不能區分哪個元件負責，也沒有測原生 Linux driver。** 兩邊共用同一張 GPU 與 Windows host KMD，不能據此排除 Windows 驅動因素，或證明所有 WSL 與 native 行為皆相同。

## Capture、準備失敗與界線

- [`run.ps1`](run.ps1)在 Windows 用 `.NET Process` 啟動原生 exe，保存原始 stdout／stderr 的文字內容、實際 exit、timeout flag、argv、working directory、記憶體與時間。合併 `.log` 明確標記兩個 streams，**不是跨 stream 的時間排序**；sanitizer summary 不重複計入。
- compiler log 有 MSVC C4819（系統 code page 950 與 CUDA header 字元）警告；編譯 exit 0，受測 source 為原始相同 bytes。文字 capture 對 compiler 非 ASCII 訊息有解碼替換，沒有宣稱 compiler log 是 byte-exact 的原始 code-page 串流；GPU diagnostics 的相關欄位是 ASCII。
- `preparation-attempt1/`保留第一次準備失敗：SDK 複製尚未完整時啟動 compile，並遇到舊 `Start-Process` capture 的空 exit／stream handle 問題。它沒有成功建置或執行 GPU 測試，不補造其 compiler exit。完成 SDK 複製、改用 `.NET Process` 後的正式 compile exit 0，全部上述 GPU commands 都使用後者 binary／runner。
- `artifact-identity.json`／`post-artifact-identity.json` 核對 source、binary、inputs、DLL 的前後 hashes。`source-state.json`保留 Git HEAD 與既有 docs／evidence 工作樹修改；這是固定 standalone source 的診斷，沒有冒稱新一輪 clean-source application acceptance。
- 本次沒有在 Windows 執行 private-surface poison、allocation-ID、NVTX annotation、racecheck／synccheck、全量影像或 Saccade F01／F02 完整 API suite；不將此結果提升為完整 Windows runtime 驗收，也不重寫原 capture 的失敗。
- **F01 initcheck FAIL／既有驗收分類與 P01 OPEN 均保留。** 新增確認的是 Windows 原生也重現，而非修復或 suppression PASS。

## 離線核對與重跑

[`analyze.py`](analyze.py)核對 11 組實際退出碼／summary、API 成功、source 與 reference hashes、artifact 前後身分、HW RGB／RGBI 全 bytes、工具正反控制與 WSL baseline，生成 [summary.json](summary.json)。它不執行 GPU：

```sh
python3 evidence/preprocess/2026-10-31-f01-investigation/windows-native/analyze.py
```

原生工作目錄記錄於 `source-state.json`／`host-launch/native-root.txt`。要重跑，先按 `sdk/SDK.json` 及 acquisition script 準備新的 Windows portable `cuda/`，複製 source、fixture／reference、runner，並將 nvJPEG DLL 放在新 exe 旁；從 Windows PowerShell 依序執行：

```powershell
.\run.ps1 -Phase build
.\run.ps1 -Phase identity
.\run.ps1 -Phase baseline
.\run.ps1 -Phase controls
.\post.ps1
```

每次使用新的目錄；runner 拒絕覆寫已有 capture。driver script 的 exit 0 不代表所有 GPU 子命令 PASS，必須檢查 `.exit`／`.status.json`／summary。SDK 與受測 exe 留在本機 ignored／Windows temp 目錄；本 evidence 保存重跑 source、套件來源與 hashes、完整 capture 和所有小型輸出，由上層 `SHA256SUMS` 綁定。
