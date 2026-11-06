# P08 設定與事件工具

2026-11-06。本輪提供正式 CPU／native CLI 的 resolved tracking 設定、來源收據、
CPU 實際決策事件與獨立 lifecycle／身份／提交檢查。部署參數與原版 Bridge 保持原值。
這是本機開發來源工具驗證；完整 GPU 中間事件與原有 sanitizer OPEN 仍各自保留。

## 設定入口與範圍

`track_offline`、`track_sequence`、`track_stream` 接受：

```text
--config FILE
--set KEY=VALUE
--resolved-config RECEIPT
```

同一個 C++ resolver 依 **入口預設 < 設定檔 < 已宣告的環境變數 < CLI** 展開。
`--serial`、`--profile`、`--epoch`、`--no-cuda-graph` 及串流 `--decode`／`--device`
均轉成同一 CLI layer；與 `--set` 重複指定相同 key 會被拒絕。同 layer 的重複項
不採最後一個覆蓋；不同 layer 的覆寫保留最後有效值與來源。僅 CLI 讀取
`SACCADE_CFG_*`，未知名稱會失敗，階段與 kernel 不讀取環境變數。

檔案使用 ASCII `key=value`，允許行首尾空白、空行與整行 `#` 註解。
非空設定檔必須宣告 `schema=saccade-tracking-config/1`。例如：

```text
schema=saccade-tracking-config/1
tracker.bridge.enabled=true
tracker.life.confirm_streak=3
```

`tracking_config schema` 列出每項的類別、型別、owner、B／F 行為、用途、預設、
限制與環境變數名稱。schema 包含所有 TrackerConfig 欄位、來源身份／尺寸、
前處理容量、實作／model／motion profiles、排程／slots、觀察開關與 MOT 插值政策。
固定的 candidate／track／packet 容量、排序／數值／前處理政策也進入 canonical 記錄，
不能透過設定檔改成另一個未實作的 ABI 或演算法。

未知、重複、缺少 schema、錯誤型別、整數溢位、非有限 FP32、超界、非法組合及
build 未提供的實作都會在提交 frame 前拒絕。CPU reference 只接受 records／serial；
native session 使用固定 G1。原有 C++ typed API 保留並繼續驗證型別化設定。
benchmarks、合成 profile、訓練與研究 harness 並未改用這份 runtime schema。

各入口保留自己的預設：

| 入口 | 實作與有效預設 |
| --- | --- |
| `track_offline` | CPU records、serial、原 tracker／MOT；epoch 取 SCDR，motion `-` 是顯式 disabled；有 SCMO 時尺寸取其 key。disabled 時沒有原圖尺寸 metadata，尺寸為宣告值，reference 不讀影像。 |
| `track_sequence` | CUDA JPEG、原 T01 profile、pipelined、Graph ON、result slots 4；尺寸鎖定 seqinfo。 |
| `track_stream` | CUDA NVDEC、HR profile、pipelined、Graph ON、result slots 1、無插值、每幀立即 receive；尺寸鎖定首張解碼 frame。 |

native 前處理 max dimensions 記錄 session 實際使用的 `max(1920,width)`／
`max(1080,height)`，以 `derived` 來源分別記錄輸入 metadata 名稱與 session 推導規則。
來源覆寫與實際輸入不符會失敗。
stream 的格式、timeout、transport、frame limit 仍是 VideoSource 的獨立 typed CLI
選項，另記收據，未假裝已納入 tracking config hash。來源 URL 不寫入收據。

## 穩定設定與收據

canonical 設定固定 key 順序、LF、十進位 unsigned、`true/false`、版本化 enum，
FP32 使用 `f32:0xXXXXXXXX` 的 bits。因此原檔空白／key 排列或覆寫來源不同，
只要有效值相同就有相同 SHA-256；`+0`／`-0` 的實際位元差異則保留。

`--resolved-config` 以不覆寫的方式保存 canonical 值、hash、每項來源，以及另列的
build identity（source commit／dirty、實際 executable hash，native 另有 engine hash）。
engine 的內部精度與建置來源仍依 engine provenance，不能從 model shape 或設定 hash 推定。
收據 parser 核對完整 canonical bytes、所有 origins、metadata schema 與 config hash。
CLI 標準輸出也記錄收據；寫檔檢查 write／close，不承諾 fsync 耐久性。

```sh
build/p08/cpu/tracking_config schema
build/p08/cpu/tracking_config resolve --config settings.cfg --output canonical.cfg --record resolved.txt
build/p08/cpu/tracking_config compare first.resolved second.resolved --allow observer.event_dump
```

允許不同的欄位必須逐項宣告，而且只能是明列的排程／觀察欄位。行為或實作差異
不能加入 allowlist；其他欄位仍逐項比較。設定相同不代替 build／engine／輸入相同，
也不表示 packets、MOT 或事件相同。CPU-only `tracking_config` 只載入本 build 提供
實作的設定；native 收據請用對應 native build 的工具核對。

## CPU 決策事件與輸出比對

```sh
build/p08/cpu/track_offline input.scdr motion.scmo result.txt \
  --events result.events --packets result.packets --resolved-config result.resolved
build/p08/cpu/tracking_events validate result.events --config result.resolved --packets result.packets
build/p08/cpu/tracking_events compare first.events second.events --config result.resolved
build/p08/cpu/packet_compare first.packets second.packets
```

`--events` 在實際 CPU predict、關聯、birth、confirm、lost、remove、Bridge relink、
emit 與 frame terminal 決策點取得不可回寫的快照；不從 packet 或前後 state 猜配對。
每筆含 source／epoch／frame、F 階段、語義順序、穩定 UID／candidate ID、
Bridge 的獨立 output ID、相關資格／前後值／原因。batch candidate 與 prediction
也明列，讓模型能拒絕漏事件、重複消費或 continuation-only birth。

`tracking-events/1` 使用 little-endian、有 payload 長度的版本化編碼，只寫事件
有效欄位及有效 history；不寫位址、計時、padding 或未使用 history。decode 拒絕
未知版本／enum、非 canonical bool、非有限值、錯誤長度、截斷與 trailing bytes。
比較保留原語義順序，回報第一個 event／欄位／值差異，不重新排序消除分歧。

合法性模型從空的 source／epoch、frame 0 開始，獨立維護 lifecycle、age、streak、
score history、候選資格與唯一消費、UID／output owner、確認條件、Bridge donor
移除、必要 emit／commit、錯誤回滾與後續 skipped。它串接 Kalman 的 x／P 快照，
**不重新計算 GMC／Kalman 數值，也不重新求解 auction 或 Bridge 幾何／排名**。
這些數值與勝者仍由既有 reference／GPU parity 驗證。

有 `--config` 時另核 source／epoch；offline 收據的輸入 frame 數會偵測缺少整段尾幀。
沒有預期長度時，合法性只驗證完整 frame 前綴，不能推定整個輸入都已處理。
`--packets` 根據 emit／terminal 重新編碼並比較每份完整 ResultPacket bytes，
包含未使用 rows／padding；事件有效內容 hash 不縮小 A10／V04 的封包比較範圍。
失敗前綴可保留事件與 packet，不能生成完成的 MOT。

三種檢查仍分開：狀態轉移合法、同 build／固定輸入的執行一致、觀察開關一致。
event／config hash 只是定位工具，不能據此判定無競態。

## 驗證與仍未涵蓋

本輪收據與命令見 [P08 開發驗證](../evidence/p08/2026-11-06-development/REPORT.md)。
CPU tests 包含每條模型規則的非法反例、codec／設定拒絕例與實際 observer ON／OFF
的完整 packets／committed state bits。CLI 測試另外覆蓋來源、覆寫、收據、輸出檔與
第一分歧。固定 MOT17 records 的比對不讀 GT、不重新調參或計分。

GPU 正常路徑沒有新增 per-frame D2H；CUDA `observer.event_dump=true` 明確拒絕。
GPU 的完整中間 lifecycle／Bridge 決策 dump 及其觀察開關一致性仍 **OPEN**，
目前 packets 無法驗證未保留的中間轉移。現行 CPU 模型與 native CLI 測試不重發
完整 CUDA 正式證書，也不關閉 G1 memcheck／原生完整 HR racecheck。
