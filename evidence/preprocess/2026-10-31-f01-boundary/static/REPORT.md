# F01 靜態邊界：`rgba2rgb<3>` 的實際讀寫

**已確認到受測 nvJPEG 的 GPU 機器碼與參數 ABI；尚未證明硬體寫入與 converter 的一般同步順序，也未確認 initcheck 的內部 shadow 行為。** 本次只執行 CPU 靜態工具與 host ABI 小程式，沒有 GPU 執行、修改 SDK 或 runtime，F01 驗收狀態不變。

本機 `libnvjpeg.so.13.2.3.58` SHA-256 是 `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338`，與[既有追查](../../2026-10-31-f01-investigation/REPORT.md)的 on-disk library 身分相同。從其 `sm_120` cubin 取出的目標是 `nvjpeg::rgba2rgb<(nvjpegOutputFormat_t)3, nvjpeg::RGBAToRGBConversion::LaunchParams<32,8>>(nvjpegImage_t, NppiSize, nvjpegImage_t)`。`cuobjdump` 與 `nvdisasm` 的保存輸出可逐指令比較；source library、抽出 cubin、工具、headers、runner 與既有完整 initcheck log 的 hash 均在 [identity.json](capture/identity.json)。不是重新執行歷史 process，也沒有宣稱 instrument 後的實際指令與 on-disk code byte-identical。

[離線核對](capture/verification.json) PASS：兩工具的 48 條指令一致（將 `nvdisasm` tail-loop label 依保存的 label 定義正規化成數值位址），38 條 address／byte-extraction／store／exit 指令的 operands 符合重建；所有 captured subprocess 均 exit 0，原始 stdout／stderr hashes 核對相同。此 PASS 僅指靜態產物一致性與重建核對。

## 1. 已確認的 ABI 與位址

`nvjpeg.h` 的 `nvjpegImage_t` 是四個 channel pointers 加四個 `size_t` pitches。Host-only compile／run 確認 x86_64 的大小為 64 bytes，channel offset 0、pitch offset 32；`NppiSize` 為兩個 32-bit int、共 8 bytes，width／height offsets 0／4。kernel metadata 獨立給出三個 by-value args：

| argument | param-relative offset | bytes | 型別／用途 |
| --- | ---: | ---: | --- |
| 0 | `0x00` | `0x40` | source `nvjpegImage_t` |
| 1 | `0x40` | `0x08` | `NppiSize` |
| 2 | `0x48` | `0x40` | destination `nvjpegImage_t` |

總參數大小是 `0x88`。`.nv.info` 的 `EIATTR_PARAM_CBANK` raw values `0x12b 0x880380` 與指令中的 loads 支持 cbank 0 的參數基址 `0x380`；不是對未公開 metadata 格式的獨立完整解碼。

| 欄位 | `c[0]` offset | 指令用途 |
| --- | --- | --- |
| source.channel[0] | `0x380` | input base |
| source.pitch[0] | `0x3a0` | input row stride |
| width／height | `0x3c0`／`0x3c4` | early exit |
| destination.channel[0..2] | `0x3c8`／`0x3d0`／`0x3d8` | RGB plane bases |
| destination.pitch[0..2] | `0x3e8`／`0x3f0`／`0x3f8` | RGB row strides |

完整[指令](capture/rgba2rgb.sass.txt)給出以下資料流。在 launch 座標非負、正尺寸可用 signed int 表示、座標及 address arithmetic 不 overflow、輸入／輸出配置與 pitches 有效的條件下，可還原為：

```text
x = 32 * blockIdx.x + threadIdx.x
y =  8 * blockIdx.y + threadIdx.y
if x >= width or y >= height: return
word32 = load_u32(source.channel[0] + y * source.pitch[0] + 4*x)
destination.channel[c][y * destination.pitch[c] + x] = byte_c(word32)
    for c = 0, 1, 2
```

這是對特定 binary 的人工 register／expression reconstruction，配合 operand 與 byte-permutation model 核對；不是 GPU ISA 的形式化定理。Indexing 的 32／8 直接來自 `LEA` 的 shift，完整 launch block shape 須由動態觀察另驗。`MAX_THREADS = (0x100,1,1)` 是 metadata 上限，不能將它當作實際 block shape。

## 2. 釐清「3-byte read」的意義

`+0x170` 實際是 `LDG.E R13, desc[UR4][R2.64]`，載入一個 32-bit word，**物理載入包含第四個 byte**。後續只有三條 `STG.E.U8`：直接取低 byte、`PRMT 0x7771` 取 byte 1、`PRMT 0x7772` 取 byte 2，寫入三個 RGB planes。Byte 3 沒有到輸出 store 的資料流。`PRMT` 的 byte-selection 解讀與 NVIDIA [PTX prmt 說明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)相符；SASS opcode 類型與工具能力見 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。

同一 kernel `.nv.info` 還有：

```text
Attribute: EIATTR_UNUSED_LOAD_BYTE_OFFSET
Value:     0x170 0x7
```

原始值已保存在 [kernel-info.txt](capture/kernel-info.txt)及完整 lossless gzip ELF dump。`0x7` 與低三個 bytes 的使用一致，**這是解讀，尚無此欄位編碼的公開規格或證據證明 initcheck 就是根據它選擇檢查範圍**。32-bit physical load、三個有效輸出 bytes 與歷史 initcheck 的「3-byte read」必須分開陳述，不能把報告簡化成「只是 alpha／padding 報錯」。

[歷史完整 log](../../2026-10-31-f01-investigation/capture/new-rgb-once-full-initcheck.log)仍是 945 筆、每筆 3-byte read、每個有效像素一次。其 source address `base + 512*y + 4*x` 與上述機器碼一致；35×27 baseline 的最後一個物理 load 起點是 offset 13,448、結束 exclusive 13,452，位於已觀察的 16 KiB 配置內。這只確認這份 baseline 的 converter 範圍，不證明一般 caller 的容量或任何 byte 的初始化狀態。

## 3. 還沒跨過的邊界

- 此 converter 在輸入讀取前有尺寸 early-exit，但機器碼沒有建立硬體 decode completion 的外部依賴；該依賴可能在 host／driver submission 層，不由 kernel 本體判斷。
- 相同 on-disk library hash＋相同 kernel／offset＋相同 address formula 是歷史定位的強對照；仍未擷取 initcheck instrument 後的執行機器碼或其 shadow state。
- 已知 RGB 三個 bytes 會直接傳到輸出，不代表能由此判定寫入者是誰、是否每種排程都先寫後讀，或為什麼初始化追蹤缺失。
- 本次沒有執行任何 image decode、sanitizer 或新 numeric comparison，不能把靜態核對提升為 F01 PASS 或 NVIDIA 誤報分類。

重跑 capture（須新的輸出目錄；只執行靜態工具及 host ABI executable）：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/static/capture.py \
  --out build-f01-static-NEW
```

離線核對已保存的雙工具 disassembly／全部 address dataflow operands／ABI／stdout hashes（不需 SDK binary，不執行 GPU）：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/static/verify.py \
  evidence/preprocess/2026-10-31-f01-boundary/static/capture
```

`capture/work/` 的 extracted proprietary cubin 與 host ABI executable 只留本機並由 `.gitignore` 排除；其 hashes 已保存，其餘文字 capture、metadata、腳本與報告由本目錄 `SHA256SUMS` 綁定。
