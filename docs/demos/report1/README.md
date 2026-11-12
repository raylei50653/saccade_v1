# Saccade v1 現場技術展示

先以瀏覽器開啟 `index.html`。展示台、影像與逐幀資料均在此目錄，可離線操作；不需要模型、GPU、Python 或網路。

如果瀏覽器對本機影片有限制，可在此目錄執行 `python3 serve.py`，再開啟終端顯示的本機網址。按 Ctrl-C 停止。

右上「展示序列」可切換 **MOT17-10-SDP**、**MOT17-02-SDP** 與 **MOT17-04-SDP**，預設為 10。切換會暫停、清除選取的 ID，並載入新序列的來源影格 221；按「回到起點」可從第一幀觀看。速度、畫面模式及框／尾線／顯示插值／MOT 缺幀補框偏好會保留，不同序列的身份與歷史不混用。「中段／尾段」定位隨序列長度更新。

## 建議的現場順序（約兩分鐘）

1. **先看原圖與結果**：選擇展示序列，切換並排模式，以原速播放，說明每個框顯示的是追蹤輸出 ID。
2. **停住一幀操作**：點選一個框或 ID，查看該軌跡的內部 UID、觀測分數與原圖座標。
3. **逐幀與慢動作**：用前後幀按鈕或方向鍵檢查相鄰影格；切換 0.25×／0.5×，比較「顯示插值（播放時）」與「MOT 缺幀補框」。實線為原始觀測，虛線與「補」標籤為依正式 MOT 規則衍生的缺幀插值。可在 04 跳至來源影格 31，查看 ID 23 的補框。
4. **分開說明速度**：展示台播放速度是來源 30 FPS 的回放；10 的效能面板引用報告獨立實測，02／04 顯示「—／尚未量測」，不沿用其他序列的數字。
5. **說明範圍**：本示範為固定版本的預先計算結果回放，播放時不執行推論；相機／RTSP 全路徑延遲尚未測量。

`assets/tracking-demo.mp4`（10）、`tracking-demo-02.mp4`（02）與 `tracking-demo-04.mp4`（04）是原始逐幀畫框、ID 與軌跡的影片備用版，可直接播放或放入簡報；它們保持舊版原始觀測畫面，**沒有加入 F09 缺幀補框或播放時計插值**。展示台的開關不會改寫這些影片。`source.mp4`／`source-02.mp4`／`source-04.mp4` 是相應序列的原圖編碼，用於互動展示台。頁尾下載連結隨序列更新。

## 顯示插值

「顯示插值（播放時）」預設勾選，可隨時關閉。播放時，瀏覽器依螢幕更新節奏對相鄰框作線性補間，讓框與尾線終點平滑移動；0.25×／0.5× 時較容易比較差異。

- 此選項只處理相鄰兩幀都有 observed row，且 `output_id`、內部 `UID` 均相同的 xyxy，不新增輸出 rows。缺幀另由下節「MOT 缺幀補框」處理，兩個開關可獨立使用。
- 來源幀由影片影格回呼決定，顯示時計只在當前幀與下一幀的座標之間取值，不提前切換來源幀。背景影片仍保留來源 30 FPS 的節奏；勾選插值只平滑框與尾線，不會將影片變成 60 FPS。
- 暫停、跳幀與逐幀操作時，原始觀測框立即使用原始座標；啟用 MOT 缺幀補框時，另保留該來源幀的虛線補框。原始分數、ID 與 observed rows 不變。切換速度、等待影片或跳轉時會清除舊插值時計。
- 狀態列以「顯示插值」／「逐幀補框」／「原始逐幀」標示當前模式。沒有 `requestVideoFrameCallback` 的瀏覽器會停用播放時計的相鄰框平滑，仍可顯示來源幀的 MOT 補框。

## MOT 缺幀補框

「MOT 缺幀補框」預設勾選，可隨時關閉。它使用與正式 F09 `v1-mot/1` 相同的方法：依 **`output_id` 分組**，同一 ID 至少有 **5 筆原始輸出**，才對相鄰兩筆之間最多 **35 個缺幀**作 binary64 線性 xyxy 補間。依正式規則，**允許兩端 UID 不同**；不向第一筆之前或最後一筆之後外推。

補出的框使用虛線及「ID … 補」標籤。暫停、跳幀與逐幀仍保留對應來源幀的補框，可點選檢查兩端影格、UID 與衍生座標；UID 變化會顯示為箭頭。正式補間 confidence 為 **−1**，當幀原始觀測分數、關聯分數與原圖 xyxy 則顯示「—」，Row 狀態為 `interpolated`。側欄分開列出原始觀測筆數與補框筆數，不把補框當作新觀測；原圖模式不畫疊圖。

10／02／04 分別產生 **934／2,040／2,053 筆**補框，合計 **5,027 筆**。獨立 Python F09 重現與實際 JavaScript 模組逐筆核對相符；另與既有 **69,869 筆**定稿 MOT rows 完整對照，文字 MOT 的三位小數 xywh／conf 在捨入誤差界內，不宣稱與全精度 JSON 逐位元組相同。模組保留首次出現順序的正式 MOT ID 映射，畫面仍顯示原始 `output_id`，不套用定稿時的重新編號。

補框利用完整離線序列中前後兩端的輸出，不代表當幀有新觀測，也不能單憑缺幀判定原因一定是遮擋。原始 JSON／ResultPacket 與備用影片均未改寫；此功能不觸發推論、品質重評或變更既有量測結果。背景仍為來源 **30 FPS**，兩個插值選項都不會產生影像中間幀。

## 播放繪製與選用量測

疊圖與原圖模式由瀏覽器直接顯示影片，透明 canvas 只畫框、標籤與尾線，省去反覆複製整張影片像素的成本。並排模式需要同時顯示兩張影像，因此只在來源幀變更或版面需要重畫時複製並快取；插值更新時重用該影像。尾線的歷史點按幀快取，ID 按鈕與側欄也重用既有內容，減少重複掃描與 DOM 更新。影片影格回呼立即對齊 packet；暫停與逐幀保留原始觀測座標，MOT 補框另以虛線標示。

若需檢查本機回放成本，可啟動上述 HTTP server，再於本目錄執行選用 profiler（需要 Node.js、Playwright 與 Chromium）：

```sh
node methods/profile_playback.cjs http://127.0.0.1:8765/ app.js playback-profile.json
```

三個參數依序為展示網址、要量測的 `app.js`、輸出 JSON。可用 `SACCADE_PLAYWRIGHT_MODULE` 與 `SACCADE_CHROMIUM_BIN` 指定已安裝的工具路徑。量測範圍是 headless Chromium 的展示台繪製成本與影片掉幀計數，並非 GPU 推論效能，也不保證實際螢幕固定達到 60 FPS；瀏覽器、硬體與同時執行的工作都會影響結果。

## 資料與解讀

- 固定 runtime：`3cbaa518f0dc37d691404e95ed85c0109c81130d`；HR 1280×768 FP16、G1 NVOFA、CUDA Graph ON、NSA／量測框 OFF。
- MOT17-10-SDP：1920×1080、654 幀、來源 30 FPS，原速 21.8 秒；10,385 筆 observed rows，重用原完整離線量測輸出。
- MOT17-02-SDP：1920×1080、600 幀、來源 30 FPS，原速 20 秒；11,454 筆 observed rows。使用相同乾淨基準、engine 與設定完整擷取一次作為展示來源；不是獨立效能 benchmark，也沒有開 GT 或重新計分。
- MOT17-04-SDP：1920×1080、1,050 幀、來源 30 FPS，原速 35 秒；43,003 筆 observed rows。沿用相同乾淨基準／engine 擷取一次，不是新品質評估或獨立效能 benchmark。
- 三序列分別使用 `evidence/report1/2026-11-11-g1-offline-e2e/raw/run1/`、`2026-11-11-mot17-02-demo/raw/run1/` 與 `2026-11-11-mot17-04-demo/raw/run1/` 的 ResultPacket／motion（皆在 `evidence/report1/` 下），對齊各自 JPEG；`packet.frame=0` 對應 `000001.jpg`。
- 原始框為 Kalman state box，原始分數是觀測分數；補框是由兩端框衍生的插值，每幀 row 數不代表場景真實人數。
- 顯示 `output_id`，UID 是內部 birth identity。Bridge 可能變更 output ID；顏色或尾線不構成身份正確的證明。
- 原始 JSON 保留 packet rows，MOT 缺幀補框由獨立模組衍生，UI 不套用正式 MOT 的重新編號。GMC 的 tx／ty 是前幀至本幀的畫面內容位移（原圖 pixels），不是相機姿態或速度。
- 原圖與影片重新編碼會有壓縮誤差；JSON 保留 packet 的數值，座標只在繪製時裁切。新增 04 前後，原有 10／02 的資料與影片 hashes 保持不變；來源擷取沒有讀 GT、重評品質或修改 runtime。

## 檔案

- `index.html`、`app.js`、`style.css`：互動展示台。
- `gap-fill.js`：與 F09 同規則的缺幀索引／取樣及正式 MOT ID 映射，不改寫來源 frames。
- `assets/data.js`、`assets/data.json`：相同的逐幀資料，前者讓直接開啟檔案時無須 fetch。
- `assets/data-02.js`、`assets/data-02.json`：02 的逐幀資料；JS 註冊 02 並保留 10，02 的 `metrics` 為 null。
- `assets/data-04.js`、`assets/data-04.json`：04 的逐幀資料；JS 只擴充 catalog，保留既有 10／02；04 的 `metrics` 為 null。
- `assets/provenance.json`：來源、hashes、編碼命令與幀數核對。
- `assets/provenance-02.json`：02 來源擷取、資料與 source video 的保管資訊。
- `assets/provenance-04.json`：04 來源擷取、資料與 source video 的保管資訊。
- `assets/video-render.json`、`assets/video-render-02.json`、`assets/video-render-04.json`：三支備用影片的重建與驗證紀錄。
- `methods/build_data.py`／`render_video.py` 及對應 `_02.py`／`_04.py`：由原始影像／封包重建展示資產；原始資料需另行備妥。
- `methods/check_gap_fill.cjs`：補框規則、35／36 缺幀邊界、最低筆數及身份條件的模組檢查。
- `methods/check_mot_gap_parity.py`：獨立重現 F09，核對模組與專案內既有完整 MOT 輸出；需要 Node.js、Python 與原 evidence，離線展示包本身不需要執行。
- `validation.json`：實際瀏覽器操作與媒體檢查結果。

## 已完成的檢查

Chromium 156 實測 file:// 與本機 HTTP：43 項互動檢查通過，包含逐幀、ID、顯示插值、MOT 缺幀補框及跨 UID 接回、全螢幕與 320／390 px 排版，以及三序列切換、偏好保留、600／654／1,050 幀邊界、場景定位、播放中切換、快速連續切換的舊回呼取消、02／04 不混用效能數字；無 JavaScript 錯誤或外部網路請求。三序列合計 2,304 個 packet／motion contexts 與 64,842 筆 observed rows，均與各自原始二進位輸出精確對照。原有 10／02 資料與影片 hashes 未變。

02／04 的來源收據可在專案內分別執行 `python3 -B evidence/report1/2026-11-11-mot17-02-demo/verify.py` 與 `python3 -B evidence/report1/2026-11-11-mot17-04-demo/verify.py` 核對；兩份各 20 個 hashes、完整輸出與前後保管均通過。六支影片均為 1080p／30 FPS，10 為 654 幀／21.8 秒，02 為 600 幀／20 秒，04 為 1,050 幀／35 秒。

補框規則與正式輸出的獨立對照已通過，收據在專案內的 `evidence/report1/2026-11-11-demo-mot-interpolation/`；可從專案根目錄執行：

```sh
node docs/demos/report1/methods/check_gap_fill.cjs
python3 -B evidence/report1/2026-11-11-demo-mot-interpolation/verify.py
```

選用的本機 server 支援 HTTP byte ranges，拖曳或跳幀會讀取所需的影片片段。`SHA256SUMS` 用於核對展示包檔案；解壓 `saccade-v1-technical-demo.zip` 後，開啟其中的 `Saccade-v1-demo/index.html` 即可。展示不依賴專案 checkout 或原始資料集。

本次展示為本機交付，未 commit、push 或發布到外部網站；正式報告與原驗收的 OPEN／FAIL 保持原判定。
