# Saccade v1 第一輪圖解報告

v1.5（2026-11-12）先講目的，再看實際追蹤畫面、各資料集結果和速度取捨。主文七頁、全稿 11 頁，補回 n／n+1 排程與既有 Nsight 時間軸；v1.5 在第 4 頁標明 MOT17 七段中 04／05／09／11／13 為本地訓練、10 為選模，另列 held-out 02 與未訓練 MOT20 的比較，並把第 6 頁開頭改回平均延遲／吞吐代價，MOT17-10 single 尾端差距不再當成結論；原本 20 頁的 v1.1 技術報告另行保留，完整表格、來源與驗證邊界沒有刪除。

- `report.pdf`／`report.tex`：目前主報告，目的、結果與跨幀排程、比較條件、完整附表與來源。
- `technical-reference-v1.1.pdf`／`.tex`：原 20 頁技術版；`technical-reference-v1.1.json` 保存原交付收據與檔名映射。
- `assets/story-*.pdf`：六張圖，包含工作流程、真實三影格範例、n／n+1 排程、五資料集三方比較、七群組增減、延遲與吞吐。
- `latency-percentiles.tex`／`latency-percentiles-audit.json`：第 6 頁 single／queued P95／P99 表；由原 48 份逐幀 CSV 獨立核對 12 組、每組 480 筆，保存 50 個來源 hashes。
- `numeric-tables.tex`：主報告附表；`model-comparison.tex` 保留技術版的完整比較段落。
- `live-results.tex`／`offline-e2e.tex`：技術版使用的既有逐幀延遲與完整離線程序結果。
- `render_comparison.py`／`comparison-sources.json`：從既有 JSON 產生表格與 Markdown §9.2，核對 14 個來源的原始 hashes。
- `render_story_figures.py`／`story-figures.json`：由既有分數產生四張向量圖與來源收據。
- `render_schedule.py`／`schedule-sources.json`／`schedule-results.tex`：從保存的 G1 Nsight 分析產生排程圖與白話結果；示意和實測分開。
- `render_tracking_example.py`／`example-sources.json`：從既有展示影片擷取三個影格，不重新推論或讀 GT。

Markdown 版的開頭採相同圖文說明；§4 保留原排程圖並連至新 Nsight 分析，其餘完整技術內容保留：`docs/reports/report1-3cbaa518f0dc.md`。

## 編譯 PDF

需要 XeLaTeX、latexmk、ctex、fontspec、tcolorbox 與常見 TeX Live 套件，以及 Noto Sans／Serif、Noto CJK TC、DejaVu Sans Mono 字型。

```sh
make -C docs/reports/latex/report1
```

已產生的 `.tex` 與圖稿可獨立編譯，不需要 GPU、影片、模型或資料集。`make clean` 只移除主報告輔助檔，不刪 PDF。原技術版可在同目錄以 `latexmk -xelatex technical-reference-v1.1.tex` 重建。

## 更新既有數字與圖表

於專案根目錄執行：

```sh
python3 -B docs/reports/latex/report1/render_comparison.py
python3 -B docs/reports/latex/report1/render_story_figures.py
python3 -B evidence/report1/2026-11-12-schedule-trace/analyze.py
python3 -B docs/reports/latex/report1/render_schedule.py
make -C docs/reports/latex/report1
```

這些指令只整理已保存的分數，不啟動模型、重新計分、訓練或調參。圖表產生另需 `rsvg-convert`、`pdffonts`。已提供 PDF 圖稿時，單純編譯不需要這些轉圖工具。

真實畫面範例可另以 `python3 -B docs/reports/latex/report1/render_tracking_example.py` 重建，需要本機既有展示影片、`data.json`、`video-render.json` 及 FFmpeg。影格索引 180／195／210 對應第 181／196／211 幀，採相同局部裁切，真實輸出 ID 71／UID 106 在三幀均有 observed rows；這是輸出示範，沒有把視覺延續當作正確性證明。30 FPS 是展示影片播放速率。交付 ZIP 內已含圖稿，未打包整支影片。

## 數字來自哪裡

排程時間軸重新分析 2026-11-05、乾淨 `fae5cb7` 的 G1／HR 30 幀 trace，未新增 GPU capture。原 node／graph SQLite 均對照當時 `nsys.sha256`，壓縮副本與可重算程式保存於 `evidence/report1/2026-11-12-schedule-trace/`；影格同時保留零起 index 與一起算 JPEG 編號。Node capture 的 Graph kernels 依 graph 角色及 launch 對應回實體 D／M，不把 synthetic streams 當成真實 stream。圖中階段起訖區間不代表全程硬體忙碌，OFA 本體也沒有被直接量到；profiler 時間不取代無 profiler 的逐幀延遲。完整 F06 28/28 位於同幀 F03 區間，27/27 相鄰穩態對有跨幀 CUDA 活動交集，平均約 0.94 ms。獨立 raw SQL 重算差值為零，圖中 30 個階段的起訖與幾何一致，見證據包的 `independent-review.json`。

品質沿用 `evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle/`，速度沿用 `2026-11-11-yolo26m-trt-performance/`。五資料集是 MOT17、MOT20、MOT15、DanceTrack 與 PersonPath22，每模型 156 段／135,503 幀。七主要群組相對微調版 HOTA 為 3 勝／4 負，逐段 55 勝／101 負；圖表同時呈現增益與退步。

官方／微調版採 raw6 one2many 接本專案共用 F04／G1；官方原始 one2one 只另有模型層控制，沒有混入品質圖。全部資料先前已曝光，FP16 數值 screen FAIL 與既有 sanitizer OPEN／FAIL 保留。PersonPath22 的 48,452 是完整序列幀數，計分只保留 8,753 個有 GT 的原始影格。SportsMOT 排除主要評估。

速度圖只引用 MOT17-10／02 各前 140 幀的受控測試：20 幀暖機、每模式四次 sessions／480 samples、JPEG 預載。第 6 頁左圖以長條顯示 single 平均值、○標 P95、◆標 P99，右圖保留 queued 吞吐，下方另列 single／queued P95／P99 的 ms 表。分位數從每組四次 session 合併的 480 筆樣本取 nearest-rank，不平均各次分位數；P95／P99 分別是排序第 456／476 筆，超過 P99 的樣本最多四筆。MOT17-10 的 HR 相對微調版雖然 P95 略低，P99 卻較高；MOT17-02 兩者都較高。原稽核核對 mean／P95，本次另從 CSV 補核 P99，未重測。單幀延遲與 queued 吞吐分開顯示，不當成五資料集速度或相機到消費端的完整延遲，也沒有從品質 capture wall time 推算 FPS。

技術版另保留 G1 的逐幀補測與完整 654 幀離線程序結果，兩組原始收據在 `evidence/report1/2026-11-11-g1-live/` 與 `2026-11-11-g1-offline-e2e/`。可分別執行其 `verify.py` 核對，`render_live_results.py` 只重算技術版的量測段落。它們與三方速度研究是不同 campaign，不合併樣本或計算跨時段加速。

## 交付與核對

`delivery.json` 記錄目前主報告、保留技術版、圖稿與來源 hashes；`sha256sum -c SHA256SUMS` 可於本目錄核對。`docs/reports/report1-20261111-latex.zip` 沿用第一輪檔名，內容更新為 v1.5，包含兩版 PDF／原稿、圖表、表格重建所需摘要、排程 trace 副本與重算程式、48 份尾延遲來源 CSV 及核對收據、兩組原延遲量測證據；不含模型、資料集、展示影片或 primary156 全量原始輸出。

本次只是圖文修訂，程式基準仍為 `3cbaa518f0dc37d691404e95ed85c0109c81130d`，runtime、模型、預設與原驗收結論維持原樣；沒有新實驗、commit、push 或遠端 CI。
