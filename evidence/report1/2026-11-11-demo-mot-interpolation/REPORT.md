# 展示補框與正式 F09 的獨立對照

本次把展示台的缺幀補間規則對齊正式 `v1-mot/1`，並使用既有完整擷取輸出進行唯讀核對。沒有執行推論、讀 GT、重算 TrackEval 或變更 runtime。

規則以 `output_id` 分組，至少有 5 筆原始輸出，才補相鄰兩筆之間最多 35 個缺幀；使用 binary64 線性 xyxy、confidence −1，不向首尾外推。UID 變化不阻止補間，這與 `src/mot_output.cpp` 的正式輸出一致。正式 MOT ID 依每序列首次出現順序重新編號；展示中的輸出 ID 與此編號分開核對。

| 序列 | 原始 rows | 補間 rows | 定稿 MOT rows | 跨 UID 補間 rows |
| --- | ---: | ---: | ---: | ---: |
| MOT17-10-SDP | 10,385 | 934 | 11,319 | 98 |
| MOT17-02-SDP | 11,454 | 2,040 | 13,494 | 149 |
| MOT17-04-SDP | 43,003 | 2,053 | 45,056 | 123 |
| 合計 | 64,842 | 5,027 | 69,869 | 370 |

`docs/demos/report1/methods/check_mot_gap_parity.py` 以 Python 獨立重現 F09，再透過 Node 載入展示台實際的 `gap-fill.js`。每一補間 row 的來源幀、輸出 ID、正式 MOT ID、兩端 UID、端點位置、缺幀數、fraction、confidence 與 xyxy 均核對；5,027 筆補間 row 的 xyxy 最大絕對誤差為 **0.0**，來源 frames 在 JavaScript 建表後保持不變。

另將全部 69,869 筆原始／補間 row 與各自已擷取的 `raw/run1/mot.txt` 比較。row 數、frame／MOT ID key、補間標示皆一致；因正式 MOT 只輸出三位小數 xywh，採每欄 `0.0005000001` 的捨入誤差界。實際最大絕對誤差為 `0.0005000000001018634`。這是完整逐列數值對照，**不宣稱 JSON／JavaScript 與文字 MOT 逐位元組相同**。

跨 UID witness（皆為零起始來源幀；UI／JPEG 編號需加 1）：

- 10：輸出 ID 71，UID 71→106，端點 86→101，frame 90 補間，正式 MOT ID 24。
- 02：輸出 ID 12，UID 12→205，端點 211→238，frame 220 補間，正式 MOT ID 5。
- 04：輸出 ID 20，UID 20→69，端點 229→256，frame 240 補間，正式 MOT ID 20。

完整座標、hash、原設定與比較結果保存於 `parity.json`。27 個原展示來源檔案在核對前後 hashes 一致；輸入 JSON、既有 MOT／resolved.config、實際 JavaScript 模組與 checker 的 hashes 均有記錄。模組與原始封包維持分開，補間框是顯示衍生資料，不應冒充當幀觀測或已確認的遮擋原因。

重跑：

```sh
python3 -B evidence/report1/2026-11-11-demo-mot-interpolation/verify.py
```

此收據驗證補間方法與既有正式輸出的對齊，不構成新追蹤品質、身份正確率、效能或 sanitizer 驗收；本次未 commit、push 或對外發布。

## 互動接入驗證

補框預設開啟，以虛線、補字標籤與獨立座標／筆數顯示；原始觀測分數不填造。Chromium file／HTTP 的 43 項互動檢查與 17 項純規則測試通過，另以 04 第 241 來源幀／輸出 ID 20 檢查補框 ID 按鈕與 320 px 選取版面。UI 原稿快照為 `app-after.js`，測試收據為 `browser-validation.json`、`unit-validation.json` 與 `gap-preview-validation.json`。

保留原生影片繪製與來源幀快取；本機五組 headless 回放量測合計 527 個影片影格、0 dropped frames，繪圖 p95 約 1.1–1.8 ms，詳見 `playback.json` 與 `playback-summary.json`。這不是推論效能或普遍螢幕流暢度保證；背景仍為 30 FPS。
