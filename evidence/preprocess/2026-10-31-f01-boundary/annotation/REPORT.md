# F01：不改像素 bytes 的 Initcheck 初始化標記對照

**新增結果：同一受測 binary、固定兩張 JPEG 的 A→B→A 序列中，只把私有
RGBA surface 的有效 RGB bytes 標記為 initialized，即使不清零、不 poison、
不改寫任何像素，後兩次 decode 的 Initcheck 報告也從 1,890 筆變為 0。
Alpha-only、caller-output 與關閉 NVTX 的控制都未消除報告；所有輸出仍與各自的
獨立 reference 逐 byte 相同。**

每個 process 的第一張 A 均不標記，保留 945 筆原始報告；**五個 process
都實際 exit 99**。這是對本機工具追蹤狀態的診斷干預，不是原 runtime 的
unsuppressed sanitizer PASS，不改 F01 FAIL／P01 OPEN。

## 固定資料與身分

- A：repo 的 35×27 baseline 8-bit 4:2:0 JPEG，SHA-256
  `3dedf93555bdc2aff5aab2a608a99f69cd8d6cc5c0fd64d8d3404ebb2f78da56`。
- B：在首次 GPU 執行前，以固定 RGB 算式產生 PPM，再由 libjpeg-turbo `cjpeg`
  產生 baseline 4:2:0 JPEG。SHA-256
  `cc242dbb8a1ba821927724cfdda66e9a93c6fdddc3fc29cc0c0a3b04ab39c36c`。
  輸入沒有依 GPU 結果改選。
- 兩張 reference 都在 GPU 前以獨立 `djpeg -dct int -nosmooth -rgb` 固定。
  A 的 planar RGB 與舊 frozen reference 相同，SHA-256
  `761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162`；
  B 為 `695db4a9efe440c839129ee3ff721de9b5663753689cf7b73be2c85ee7a623a9`。
- v4 受測 source：`ca8370be84c93c38ce5c7d75498dbb79b752b8e007f602dabc20579961cc9fea`；
  binary：`95f070ace8853c2125d7d1ebafa08ce7e5bfe99124f1d22ee6abe9f69ee48bc1`。
  五組控制使用同一 binary，前後 hashes 未變。
- 實際 process maps 同時包含 WSL 的 `libcuda.so.1` shim 和 driver 目錄下的
  `libcuda.so.1.1` implementation；兩者分別預先 pin，不把 shim hash 當作所有
  driver components 的 hash。實際 nvJPEG 為原先 13.2.3.58，SHA-256
  `c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338`。

固定 inputs、獨立 decoder commands、版本、source／helper／runner 與執行前後
身分都保存在 [capture-v4](capture-v4/)；執行檔留在 ignored
`build-f01-boundary-annotation-v4/annotation`，不把重編檔自動視為此受測 binary。

## 同一 process 的私有 surface 發現

每次啟動都先對 A 做未標記 decode 並同步，接著讀該 process 專用的
`--log-file`。只有完整取得 945 筆 `rgba2rgb<3> +0x170`、size-3 報告才繼續：

1. 由 thread／block 算出所有 x=0…34、y=0…26，要求無重複、無缺漏。
2. 由相鄰 x 的地址差推得 stride=4，由相鄰 y 推得 pitch=512，然後核對每筆
   address 都等於 `source + 512*y + 4*x`。沒有借用其他 process 的固定地址。
3. 在普通 host 程式點以 `cuMemGetAddressRange`／`BUFFER_ID` 核對 live 配置，
   有效像素 footprint 均在 16 KiB 配置內。後續每次標記前與 decode 同步後，
   base／size／BUFFER_ID 都必須保持不變。

五個 process 恰好都回報 ID 1077，但 ID 只在自己的 process 內比較，沒有把
跨 process 的相同數字當作同一 allocation。這裡沒有 CUPTI、固定私有指標、
poison 或額外 GPU inspect kernel；host 只讀取正常 decoder 的輸出。

## 五組同 binary 對照

每次序列都是 A→B→A。第一張固定不標記；後兩張分別使用下列模式。
所有 commands 都保留完整 backtrace、`--print-limit 0`、至少 4 GiB
MemAvailable 預檢、60 秒 timeout、實際退出碼及 `/usr/bin/time -v`。
沒有 timeout、遺失報告或 overflow 的 v4 capture。

| 控制 | 第一張 A 報告 | 後續 B／A 報告合計 | 全程 summary | 三張 RGB |
| --- | ---: | ---: | ---: | --- |
| 不標記，NVTX yes | 945 | 1,890 | 2,835／exit 99 | 全部 exact |
| 有效 RGB 三 bytes／pixel，NVTX yes | 945 | **0** | 945／exit 99 | 全部 exact |
| 只標記 alpha byte／pixel，NVTX yes | 945 | 1,890 | 2,835／exit 99 | 全部 exact |
| 只標記 caller output，NVTX yes | 945 | 1,890 | 2,835／exit 99 | 全部 exact |
| 標記有效 RGB，NVTX no | 945 | 1,890 | 2,835／exit 99 | 全部 exact |

RGB 模式每次只標記 945 個不連續的 3-byte ranges，共 2,835 bytes；alpha
模式為 945 bytes；caller 模式為另一個 caller-owned 2,835-byte output。
使用官方 `nvtxMemCudaMarkInitialized`，其作用是排除指定範圍的 Initcheck
檢查，不是寫入實際像素；呼叫本身沒有 acknowledgement，作用由正反控制觀察。
API 依據見 [contracts 報告](../contracts/REPORT.md)。

標記之後，在呼叫下一次 decode 前，**全部控制都額外同步 caller stream**。
這是刻意加入的診斷同步：既有 trace 表明 converter 可在 SDK 私有 stream 上
執行，因此不假設 caller stream 上的標記自然與私有 stream 有序。除此之外，
每次 decode 後的同步和讀回仍有保留。這些 host 開銷與同步可能改變排程，
不能將本實驗當作原 runtime 排程下的一般證明。

## 可以與不能推出的結論

本次能直接確認：

- 報告對初始化標記的依賴可以與像素值分離；不用以 memset 改寫私有 surface
  就能得到有效 RGB ranges 特有的差異。
- 報告牽涉被 converter 消費的 RGB bytes；標記 alpha 或 caller output 不能
  得到相同結果。不是只消除 padding／alpha 讀取，也不是沒有開啟工具。
- 交替 A、B、A 的 15 份輸出全部符合各自獨立 reference，沒有觀察到舊影像
  或其他值混入；所有已列出的報告在此資料中都可由完整座標與地址核對。

這些結果強化「正確硬體寫入未被工具當作初始化」的本機機制解釋，但
**標記本來就會隱藏未初始化檢查**，不能單靠報告消失反過來證明資料已初始化。
資料正確的依據仍是原始未標記控制、完整輸出比對與既有 surface 干預證據。
本次沒有直接觀察 initcheck shadow memory，沒有取得 NVJPG 內部完成訊號，
也沒有證明所有排程或跨 driver／GPU 均正確。

前後相同 allocation ID 亦不能單獨證明每個未報錯 converter 必然使用該配置，
或排除 SDK 在其他時間點讀取其他 live surface；本次是結合當次 warmup 地址、
range-specific 正反控制、穩定配置及輸出結果的有界證據。

## 保留的失敗與修正

- [v2-overflow](attempts/v2-overflow/)：第一輪使用 `--show-backtrace no`。
  Warmup 只印出 256 筆，工具明確報 overflow、689 errors skipped。程式因此拒絕
  發現／標記，沒有繼續後面的實驗，也沒有任何數值 PASS 主張。
- [v3-classifier](attempts/v3-classifier/)：恢復歷史 capture 的完整 backtrace 後，
  當次 945 筆發現完整，A→B→A 數值 exact、2,835 errors／exit 99，但初版
  離線分類器誤將所有 `libcuda*` 都要求等於 shim hash，拒絕同時載入實際 driver
  implementation 的正常 WSL stack。原分析與 runner failure 原樣保留。
- v4 在下一輪 GPU 前同時 pin shim／implementation，並使用單一新 binary
  重跑所有五組。v3／v4 source、helper 與 A／B inputs 相同，但 binary hashes
  不同；本表只使用 v4，不把未知的重編差異當作已證等價。

`ARCHIVE.json` 記錄原始檔案 hash 和 lossless gzip 轉存的對應；大型 tool logs
只是壓縮，退出碼、stdout／stderr、每個 RGB output 和原始失敗均保留。

## 離線重驗與新一輪 capture

以下 verifier 不執行 GPU，會重讀解壓後的原始診斷、座標、配置記錄、15 份
RGB bytes、輸入 references、命令／退出碼及 source／loaded-module 身分，
不只相信既存的 `analysis.json` PASS booleans：

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/annotation/verify.py
```

新 capture 先以離線 prepare 固定資料與 binary，再串行執行五組；不得覆寫既有
目錄，並須重新檢查受測身分。`run` driver exit 0 代表診斷矩陣完整，**不代表
子命令 sanitizer PASS**；五個預期 child exits 都是 99。

```sh
python3 evidence/preprocess/2026-10-31-f01-boundary/annotation/capture.py \
  --out build-NEW-annotation --phase prepare
python3 evidence/preprocess/2026-10-31-f01-boundary/annotation/capture.py \
  --out build-NEW-annotation --phase run
```
