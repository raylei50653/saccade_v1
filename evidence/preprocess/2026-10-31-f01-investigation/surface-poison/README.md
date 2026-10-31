# 私有 surface poison／allocation identity／trace 補充證據

此目錄封存使用者提供的 scratch 診斷，來源與逐檔 hash 見 [IMPORT.json](IMPORT.json)。[原始分析](USER_ANALYSIS.md)保留來源敘述；本檔與 [analysis.json](analysis.json)記錄封存時的離線核對及修正。未重跑 GPU、未改 native runtime，也未改 F01／P01 驗收狀態。

**本機受測路徑已取得直接干預證據，支持 initcheck 未辨識硬體 surface 寫入的機制解釋。** 這比原先只靠輸出相同、錯誤位置與排除控制更強。官方分類、內部同步的一般正確性及其他環境適用性仍未確認。

## Trace 核對

[nsys SQLite](raw/nsys-rgb.sqlite)記錄兩次 decode。每次 nvJPEG decode 的 CUDA 可見工作是一次 **1,124-byte H2D** 與一次 `rgba2rgb`；1,124 恰為 repo baseline JPEG 的檔案大小。每次 decode 後另有 harness 的 `inspect_caller_output` 與 2,835-byte D2H。SQLite 沒有 MEMSET activity table；這次 trace 沒有顯示為私有 surface 產生像素的其他 CUDA kernel 或 memcpy／memset。

CSV 的 MB 欄位有捨入，1,124-byte 的精確數量由 SQLite 取得。Trace 直接支持「CUDA 可見操作未呈現該 surface 的像素寫入」；將寫入者歸因於 HARDWARE backend 的 NVJPG 引擎，是結合 backend、source 與實際像素結果的機制判斷，不把 profiler 未顯示活動當成對所有可能寫入機制的形式化排除。

## Poison 與 allocation identity

本變體以先前 initcheck 報告取得的私有地址，呼叫 `cuMemGetAddressRange` 後用 tracked `cudaMemset` 填滿配置，再 decode。這次寫的是 SDK 私有 surface；與先前只清零 caller RGB output 或未被呼叫的 allocator callbacks 是不同實驗。

| 原始 log | repeat 0 | repeat 1／2 | allocation／數值觀察 |
| --- | --- | --- | --- |
| [attr-a5-initcheck](raw/capture/attr-a5-initcheck.log) | 945 errors | 0／0 | 同址 16 KiB、buffer ID 1065→1079；之後保持 1079；三次 RGB exact |
| [Y preallocate](raw/capture/poison-y-preallocate-initcheck.log) | 54 errors | 0／0 | 同址 16 KiB、buffer ID 1065→1087；之後保持 1087；repeat exact，未做獨立 Y 數值驗收 |
| [RGB no-preallocate，正確 surface](raw/capture/poison-rgb-no-preallocate-surface-initcheck.log) | 945 errors；尚無 surface 可 poison | 0／0 | 第一次 decode 建立 `0xf0e404000`、16 KiB、buffer ID 1073；後兩次 poison 與 RGB exact |

上述三份 log 均完整列出錯誤且 summary 與 repeat 0 的錯誤數相同；因此後續 repeat 沒有新增報告。三份均沒有獨立保存的 `.exit`，表中的數值是 log diagnostics，**不合成退出碼或 acceptance PASS**。

同一位址／大小的 allocation ID 改變，直接顯示原配置身分被替換；結合第一次 poison 仍報錯、後兩次穩定配置 poison 後乾淨，支持「重新配置重設 initcheck 初始化狀態」的機制。shadow 狀態本身沒有被直接讀出，不能把這一部分寫成觀察到工具內部 shadow memory。

## Surface dump：確認及修正

6 份 16 KiB [surface dumps](raw/dump/)以 pitch 512、像素 stride 4 重建：**每份的 945／945 有效像素均與獨立 RGB reference 相同，alpha 全為 0**。有效像素在 A5／5A 兩種 poison 後仍被覆寫成參考值，沒有 poison 混入這次輸出。

但原分析「其餘區域仍是 0xA5」須修正：有效 RGBA 以外有 12,604 bytes，A5 dump 只有 8,198 bytes 等於 poison，另 4,406 bytes 不同；5A dump 則為 8,199／4,405。明確全部保留 poison 的區域是 **32 列各自後 256 bytes，共 8,192 bytes**。其餘額外區域也被改寫，與 padded decode footprint 相符；這不是 SDK 私有布局的支援契約。

dump-enabled initcheck 的總數為 **1,457 errors**，不可寫成整次 0 errors：第一個 repeat 的 `rgba2rgb` 945 筆之外，整塊私有 allocation 的 D2H dump 另讀到未被追蹤初始化的區域，增加 API 報告。後續 poison 保留的配置才不再新增報告。完整數字與分段計數見 [analysis.json](analysis.json)。

## 保留的失敗與來源限制

- 較早的 [no-preallocate 控制](raw/capture/poison-rgb-no-preallocate-initcheck.log)使用 `0xf0e200000`，當時它是 caller output（2,835 bytes），並非私有 surface；三次仍有 2,835 errors。只將後來 `0xf0e404000`／16 KiB 的控制用作私有 surface poison 證據。
- [普通 poison run](raw/capture/poison-a5-normal.log)與 [memcheck poison run](raw/capture/poison-a5-memcheck.log)沿用 initcheck 的固定地址，query 返回 CUDA driver status 500，實際未 poison。前者 exit 0；後者 exit 99／2 errors，不能列為成功 poison 或乾淨 memcheck。
- 匯入包含 14 份 `.log`、8 份 `.exit`、0 份 `.command`。nsys SQLite 保留了該次 target argv；其餘不能由檔名或 log summary 補造完整命令或退出碼。這不是原 acceptance runner 所產生的完整 capture。
- source 曾在實驗間演進：早期 log 沒有最終 source 一定輸出的 `surface_attr`。匯入時 final `repro_alloc.cu` SHA-256 為 `84bde8a912a78c91e9ac84a9fe18c943001a1a6bee77ff1c9a653538208efec8`，binary 為 `d40828a290f0197945abe003d0611d2aa5728cbbdcd003e497330f702451e91d`；這是**匯入時身分**，不能追溯當作每次執行已固定的身分。原 source、binary、全部失敗與 dumps 均按 bytes 保留。
- source 有 `cudaDeviceSynchronize`、poison 與 dump 等診斷干預。這些 run 只能支持本機實際觀察的順序，不能證明原 runtime 在所有排程下的 nvJPEG 內部先後順序；racecheck／synccheck 的覆蓋也不因此擴張。
- 這次新增材料的 HW log 回報 nvJPEG 13.2.3；環境依使用者陳述及 nsys 裝置資料，沒有每次保存完整 SDK／driver／library hash 身分。不將舊 bundle 的所有身分自動套入這些 scratch runs，也不外推至不同裝置或版本。

## 離線重驗與驗收邊界

```sh
python3 evidence/preprocess/2026-10-31-f01-investigation/surface-poison/analyze_import.py
```

此工具核對 47 份匯入檔案的 hash、逐 repeat diagnostics、6 份 surface dump、raw outputs 與 SQLite trace，輸出 `analysis.json`。不執行匯入 binary，也不寫入任何 SDK 記憶體。整個 bundle 的 `SHA256SUMS` 綁定補充檔案。

Poison 是依特定位址改寫 SDK 私有記憶體，**只作診斷，不能進 runtime**。本機報告的機制已有直接實驗支持；是否將原始 unsuppressed initcheck FAIL 分類為「已解釋工具誤報」，另由驗收政策決定。本次封存不改既有狀態，也不把經 poison 改變的執行當作原 runtime 的 sanitizer PASS。
