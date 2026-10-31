# F01：逆向與初始化診斷的證明邊界

**已把證據推進到實際 GPU 指令、當次 launch 參數／私有配置、實際 hardware backend／private provider，以及 converter → caller 的 event 依賴。硬體 producer → converter 的初始化／完成語義仍有未跨過的邊界；原始 F01 initcheck FAIL／原因 OPEN、P01 OPEN 均保留。**

本次是接續 [2026-10-31 原始驗收](../2026-10-31/REPORT.md)與 [F01 investigation](../2026-10-31-f01-investigation/REPORT.md)的獨立診斷。沒有修改 Saccade runtime、比較政策、decoder backend 或 suppressions；沒有向 NVIDIA 追加傳送訊息，也沒有 commit／push。準備失敗、部分 debugger 結果與成功捕捉分開保存，不改寫舊證據。

## 1. 本次完成的證據

| 問題 | 現在的證據 | 層級／限制 |
| --- | --- | --- |
| `+0x170` 到底讀了什麼？ | 同一 nvJPEG hash 的 `sm_120`，兩個 disassembler 核對 48 條指令；`LDG.E` 讀 32-bit word，低三 bytes 分別寫 RGB planes | 特定 binary 的靜態 register／dataflow 重建，非 ISA 形式化證明；[static](static/REPORT.md) |
| source 是否只是從歷史報錯猜的？ | 六次 driver launch 都複製到實際 source pointer、pitch＝512、35×27、獨立 destination；API 返回後查到 16 KiB allocation 與當次 buffer ID | 三種正常配置、每個 process 兩次 decode；不跨 process 拼接位址／ID；[dynamic](dynamic/REPORT.md) |
| 哪個 hardware backend 在跑？ | CPU debugger 看到 `NVJPGV2BatchedDecoder` 的實際 vtable／dispatch；經私有 getter 使用 `libnvcuvid` 函式表 | 正常 decode，debugger 會改變 timing；[host-static](host-static/REPORT.md) |
| 私有呼叫追到哪個模組？ | 實際追到 `libnvcuvid+0x17f53` 的 table slot `0x40`，目標是實際 WSL driver `libcuda.so.1.1+0x4c7a50`，CPU 呼叫返回 0 | 已確認 module／offset／呼叫邊；沒有公開符號能把它直接命名為硬體完成 fence |
| converter 結果如何交給 caller？ | 六次都捕捉到成功的 `cuLaunchKernel(S) → cuEventRecord(E,S) → cuStreamWaitEvent(C,E)`，S≠C；RGB exact | 已提交的 stream/event 依賴，依公開 API 契約支援下游順序；不當成 GPU 完成 timestamp 或硬體 producer 證明 |
| caller 使用 stream 的依據？ | nvJPEG 公開契約要求透過傳入 stream 使用／同步輸出；內部可以用 private stream 接回 caller | API 契約與 reverse-observed implementation 分層；[contracts](contracts/REPORT.md) |

上述三種正常 CUPTI 配置是 DEFAULT＋PreAllocate、NO_PIPELINE＋PreAllocate、DEFAULT＋不 PreAllocate。每個 process 完整正常退出，兩次 2,835-byte RGB 與固定 djpeg reference 全 bytes 相同。這不是 application 全套測試或 sanitizer 重驗。

## 2. 「整頁地址沒完全初始化」的精確邊界

**現有證據不能將問題定義成整頁實際未初始化，也不能縮成只有 padding 未填。**

- 16 KiB 是查到的 **allocation**，未確認其物理 page 劃分。
- 35×27 baseline 的 945 個有效像素，每個都有歷史 initcheck 的 3-byte read 報告；這些是會進入 RGB 輸出的 bytes。
- 實際 converter 有 945 次 32-bit pixel loads，涉及 3,780 個不同 bytes；其中 2,835 RGB bytes 傳入輸出，第四 byte 不影響 RGB stores。
- 最後一個 load 的 exclusive end 是 allocation base＋13,452，pitch gaps／padding 並非全被讀到。這次不能判定整塊 16 KiB 的初始化狀態。
- 既有六份 private-surface dumps 的有效 RGB 像素均 exact、alpha＝0；poison 後有效 RGB 被覆寫。這是實際內容證據，與 initcheck 的初始化追蹤狀態分開；參見[既有干預](../2026-10-31-f01-investigation/surface-poison/README.md)。

這些結果支持「正確的硬體輸出沒有被工具視為已初始化」的機制解釋。尚未直接讀取 initcheck shadow，也沒有將 compiler metadata `EIATTR_UNUSED_LOAD_BYTE_OFFSET=(0x170,0x7)` 與 sanitizer 內部演算法建立已公開定義的連結。

## 3. 不改像素的初始化標記對照

另以**同一個新診斷 binary** 在五個獨立 Initcheck process 執行固定 A→B→A：A 是 repo baseline，B 是 GPU 執行前固定的不同 35×27 baseline 4:2:0 JPEG，兩者 reference 都由獨立 `djpeg -dct int -nosmooth -rgb` 產生。不是以 GPU 結果調整輸入或 reference。詳見 [annotation 證據](annotation/REPORT.md)。

每個 process 的首輪 A 不做 annotation；完整取得當次 945 個有效像素報告，從每筆 thread／block 座標及地址推導 base／4-byte stride／512-byte pitch，驗證全部座標與 live allocation／buffer ID。沒有沿用其他 process 的固定地址，也沒有使用 CUPTI。

後兩輪依各組模式使用 NVIDIA 公開的 [NVTX mark-initialized API](https://docs.nvidia.com/compute-sanitizer/SanitizerNvtxGuide/index.html#mark-initialized-api)，**不使用 poison、不由診斷程式改寫像素**。每組都在 annotation 點額外同步 caller stream，確保標記先完成；這是明確的診斷干預，不是原 runtime 排程。

| 同一 binary 的控制 | 首輪原始 errors | 後兩輪新增 errors | 三次 RGB |
| --- | ---: | ---: | --- |
| 不標記，NVTX yes | 945 | 1,890 | A／B／A 各自 exact |
| 只標記私有 surface 的有效 RGB bytes，NVTX yes | 945 | **0** | A／B／A 各自 exact |
| 只標記私有 surface 的 alpha bytes，NVTX yes | 945 | 1,890 | A／B／A 各自 exact |
| 只標記 caller output，NVTX yes | 945 | 1,890 | A／B／A 各自 exact |
| 標記有效 RGB bytes，但 NVTX no | 945 | 1,890 | A／B／A 各自 exact |

全部 process 都保留首輪 FAIL，sanitizer 均 exit 99；完整 log／summary 沒有漏印或 overflow。每次介入前後 allocation ID 相同，actual mapped SDK／WSL driver 身分核對相符。

**本機可主張的增量**：報告依賴有效 RGB bytes 的工具初始化追蹤／排除狀態，並非只靠 alpha 或 caller destination 的標記就能消除；消除後兩輪報告不需要 poison 改寫資料。A→B→A 全 bytes 對照也排除了這些受測輸出只沿用 A 的解釋。

**不能由此推出的事**：NVTX 的設計就是排除指定範圍的 Initcheck 檢查，因此 annotation 生效本身不證明硬體真的已初始化、producer 的唯一身分或一般同步正確性。對另一次 decode，原 allocation ID 穩定仍不能獨立證明 converter 沒改用另一塊 live surface；完整定位依賴本次觀察、靜態資料流與前述分開捕捉的動態 launch 證據，不能將不同 process 的位址／ID 混用。原始未標記執行不因此改成 PASS。

## 4. 尚未完成的證明義務

1. **硬體完成／可見性**：converter 在 private stream 上讀 source 之前，private provider 如何保證 hardware 已完成寫入。目前實際追到 WSL driver `libcuda.so.1.1+0x4c7a50` 的未匯出操作；其靜態 success branch 又呼叫 helper `+0x4c76e0`，但沒有把這個操作與已定義的 completion／visibility 語義建立對照。CPU 返回 0 不等於 GPU 寫入已完成；converter → caller event 只解決下游依賴，不證明上游 producer。
2. **診斷分類**：初始化追蹤缺失、SDK 初始化／同步缺陷與 driver／tool 交互，仍須進一步辨認。缺少公開 CUDA producer activity 不是唯一寫入者的證明。
3. **原始 application 報告範圍**：standalone 的完整 945／1,890 診斷不等於 session 的 8,151 筆都已分類；其 8,051 筆未印出紀錄不補造。
4. **適用環境**：本次限既有 GPU／WSL driver／SDK 的 hashed binary。先前 native Windows 重現是獨立證據，不能自動繼承本次 userspace 路徑；原生 Linux／其他 GPU 與版本仍另驗。

正常重跑、debugger 與 tracer 對 host timing 的影響，以及後續診斷額外同步，都必須隨結果保留。有限 runs 不能證明所有排程；caller 正確性可依 SDK 契約論證，但 SDK 的硬體完成機制與誤報分類不能靠契約字句自動補完。

## 5. 離線核對

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/verify.py
```

入口核對各子目錄的 SHA-256 manifest，並重建靜態指令／ABI、三層 CPU GDB observations、CUPTI 參數與 event 配對、SDK 契約 helper，以及五組 NVTX 控制的原始 diagnostics／座標／輸出／身分。它不執行 GPU；[verification.json](verification.json)保存本次結果。這是證據一致性與有限重建的核對，不會把原始 initcheck FAIL 改為成功。
