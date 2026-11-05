# MOT17-10 val：IDSW／斷軌事件回放

2026-11-05，本地 CPU 固定輸入診斷，`source_dirty=true`、`formal_acceptance=false`。
只讀 MOT17-10-SDP validation；未調參、未重跑 GPU session、未更改部署 tracker。

## 使用入口

- [互動回放（本機 build 產物）](../../../build/tracking-events/publication-viewer/index.html)：瀏覽器直接開啟，不需 server。
- [預覽](../../../build/tracking-events/publication-viewer.png)。
- [事件 CSV](events.csv)、[事件完整證據](events.json)、[核對與輸入 hash](summary.json)。

左側篩選 IDSW、CLEAR fragment、間隔後接回或末段未接回；選取事件從前 15 幀開始，
可拖動完整 654 幀時間軸、逐幀前後或播放。預設聚焦事件 GT，可關閉查看全圖。
框層包含經 TrackEval 保留的 pedestrian GT、最終 MOT、F04b batch、F04 kept／suppressed
候選池與預測 track。細節欄顯示 motion、前後 track 狀態及相關 UID 的實際 cost calls。
交付版的 `frame`、`birth_frame`、`last_observed_frame` 統一從 1 起算；
`age`／`streak` 是計數。末段未接回單獨顯示尾段起訖與最後配對，不顯示接回箭頭。

`uid`、`output_id`、MOT ID 分開標示；依 packet 首次 output_id 出現順序重建 MOT ID，
逐幀核對所有非插值 MOT rows 的 ID 與座標（MOT 三位小數寫出容差 0.0011 px）。
虛線 MOT 框是 F09 插值，不表示當幀有觀測或成功關聯。

## 結果

| 項目 | 數量 |
| --- | ---: |
| 完整回放 | 654 幀 |
| IDSW | 56 |
| CLEAR Frag | 150 |
| 間隔後接回 | 150 |
| IDSW 與接回合併去重 | 159 個事件 |
| 已配對 GT 的標註末段未接回 | 10 個 |
| 回放清單合計 | 169 個事件 |

IDSW 與 Frag 會重疊，不可直接相加。56 次 IDSW 中，47 次發生在間隔後接回，
9 次發生在連續配對。10 個未接回尾段只表示該 GT 最後一次成功配對後仍有 GT 標註，
沒有後續成功配對；不能直接判成遮擋恢復失敗或排除出畫邊界。
從未配對成功的 GT 不列為「斷軌」事件，仍計入 CLEAR FN。

事件抽取使用目前安裝的 TrackEval `MotChallenge2DBox` 前處理，包括 distractor
移除、zero-marked／class 篩選及 ID relabel；CLEAR 使用 IoU 0.5、前次 timestep
配對優先與 Hungarian 配對。保留 upstream 在空 GT／空 tracker 幀 early-continue
的行為，因此另列實際 gap recovery，不能一般化為永遠等於 CLEAR Frag。

| 核對 | 抽取結果 | 原版 TrackEval |
| --- | ---: | ---: |
| IDSW | 56 | 56 |
| Frag | 150 | 150 |
| CLR_TP | 9704 | 9704 |
| CLR_FN | 3135 | 3135 |
| CLR_FP | 1060 | 1060 |

## 初步篩查與下一步

從前次成功配對之後到本次事件，逐幀查詢該 GT 與 F04 候選池／F04b batch 的最大 IoU。
分類採互斥優先序：GT 缺席 → 任一幀候選 IoU < 0.5 → 任一幀 batch IoU < 0.5 → 其他。
這是幾何覆蓋的排查入口，不是因果歸因；0.5 是此診斷的覆蓋門檻，不是 tracker 的
關聯閾值。候選池已經歷 F04 的 score／top-k／class 等處理，不等於模型所有 raw outputs。

| 56 次 IDSW 的區間線索 | 次數 | 可先看的事件 |
| --- | ---: | --- |
| 曾有 F04 候選覆蓋不足 | 25 | #16：f33，GT22，MOT17→20，gap8 |
| 候選覆蓋足夠，但曾有 F04b batch 覆蓋不足 | 9 | #34：f112，GT34，MOT31→26，gap2 |
| 候選及 batch 每幀都有 IoU≥0.5 的框 | 22 | #15：f30，GT26，MOT5→19，gap1 |

**建議先檢查最後一組 22 個事件**，確認 GT 對應的舊 track 是否仍存活、預測偏離程度、
候選分數所在階段、競爭 track 與輸出插值。例 #15 的 f29／f30，最佳候選 IoU
分別為 0.865／0.831，最佳 batch IoU 為 0.776／0.641，不能直接說成當時沒有偵測框。

`cost_calls` 由 CPU association 的既有 hook 記錄，回傳相同的 stock cost，
只包含實際進入成本計算的 pair，可能跨階段重複；沒有記錄 auction winner、stage、
未進入成本計算的拒絕原因或完整 bridge claim。不能把最小 cost 當成實際配對結果。
GMC tx／ty 只供查看，這輪沒有介入控制或建立 GMC 因果判定。

## 驗證與來源

- Instrumented replay 的完整 encoded ResultPacket **654/654** 與 stock `TrackerReference` 相同。
- 最終 MOT 與已凍結 OAO 實驗的 G1 baseline **bytes 相同**。
- 新建 CPU build，既有 CTest **20/20 PASS**：[交付前完整 log](publication-validation/ctest.log)（原附件缺少測試內容，見下節）。
- 抽取器測試 **3/3 PASS**，涵蓋換 ID、gap、空幀 upstream 行為與 30 組隨機配對／閾值案例；
  每組對照原版 TrackEval：[log](event-tests.log)。
- 拒絕既有 trace 輸出目錄；拒絕被改動的 baseline，且不建立輸出：[負控制](negative-controls.json)。
- Chromium 實際載入本機頁面，檢查 169 個事件、56 個 IDSW、10 個 tails、切換／前進／資料欄與圖片，
  JavaScript exception 為零：[browser check](browser-check.json)。
- [執行命令與 binary／輸入 hash](run-manifest.json)、[原始 trace log](trace.log)、
  [抽取 log](extraction.log)、[source hashes](source-files.sha256.json)、[source delta](source.diff)、
  [新增 source 快照](source-untracked/)、[大型本地產物 hash](payload.sha256.json)。
  執行前後 binary、SCDR、SCMO 與工具 source hashes 相同；抽取前後來源、GT、MOT、TrackEval source 相同。

大型 trace、含縮圖的 viewer 留在 `build/tracking-events/`，不提交資料集影像。
這是觀測工具的 CPU 正確性／事件抽取核對，不是新的 CUDA、session、sanitizer、模型品質或效能驗收。

## 交付前檢查與補驗（2026-11-05）

檢查發現原 [cpu-ctest.log](cpu-ctest.log) 只有測試開始／結束，沒有測試內容；
先前 `ctest -N` 查詢覆寫了 `LastTest.log`。原檔保留，不能用作 20/20 的證明。
已使用新的 `publication-cpu` build 重新執行 CTest，完整 stdout 保存在
[publication-validation/ctest.log](publication-validation/ctest.log)。

另修正四項診斷呈現／工具問題：track 的出生／最近觀測幀改為一基底且明確命名；
未接回尾段顯示完整區間；候選層標成「F04 候選池」；尾段用語改為「仍有 GT 標註」。
原分類未使用 visibility，所以不能把 GT 標註存在稱為可見。這些更正不改動事件分類或部署。
Exporter 也改為原子取得新的輸出目錄，避免同一路徑並行執行覆寫；
Python 工具／測試支援 `SACCADE_LEGACY_ROOT`。

[交付檢查 runner](publication-validation/run.py) 記錄命令、退出碼、binary／來源／輸入雜湊與
[來源差異](publication-validation/source-delta.json)。新 binary 的 654/654 packet 比較仍通過；
新 trace 與 viewer 的 `events.csv`／`events.json` 與原檔 bytes 完全相同，MOT 與 G1 baseline 相同。
逐幀檢查一基底 track 幀欄位；兩個 exporter 同時要求同一新目錄時，只有一個成功且保留完整輸出。
拒絕既有 trace／viewer 與被改動 baseline 的負控制通過。原始 evidence 與 source 快照 bytes 保留。

補驗記錄：[manifest](publication-validation/manifest.json)、[event tests](publication-validation/event-tests.log)、
[trace](publication-validation/trace.log)、[抽取](publication-validation/extract.log)、
[瀏覽器檢查](publication-validation/browser-check.json)。本機大型產物依補驗 manifest 的 hashes 判定；
前段 `run-manifest.json`、`summary.json`、source 快照及原 browser check 仍代表修正前的開發執行。
原始 CSV 的 170 行 CRLF 在完整 staged whitespace check 中被判為 trailing whitespace；
保留既有 bytes，source／文件檢查無診斷，見 [分類紀錄](publication-validation/whitespace-check.json)。
這次補驗仍為 dirty-source CPU 診斷，不升格為 clean-source 正式驗收。

## 重跑

在 repo 根目錄，使用新的輸出路徑：

```sh
cmake -S . -B build/tracking-events/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/tracking-events/cpu -j 6
build/tracking-events/cpu/tracking_event_trace \
  /home/ray/developer/ai/saccade_v1-worktrees/hr-records-v2/MOT17-10-SDP.scdr \
  /home/ray/developer/ai/saccade_v1-worktrees/hr-records-v2/MOT17-10-SDP.scmo \
  build/tracking-events/NEW-TRACE
/home/ray/developer/ai/saccade/.venv/bin/python scripts/tracking_events.py \
  --trace build/tracking-events/NEW-TRACE \
  --baseline evidence/occlusion/2026-11-05/run/baseline/MOT17-10-SDP.txt \
  --output build/tracking-events/NEW-VIEWER
/home/ray/developer/ai/saccade/.venv/bin/python tests/tracking_events_tests.py
```

Python 使用既有 legacy venv 的 NumPy、SciPy、Pillow 與本機 TrackEval；不是 runtime 依賴。
抽取器刻意限定目前 MOT17-10 G1 baseline，要求 IDSW=56、Frag=150；若將來更換 baseline，
應另立診斷版本與期待值，不能直接套用本輪判定。
