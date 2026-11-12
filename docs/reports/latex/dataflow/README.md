# Saccade v1 Dataflow：從影格輸入到追蹤結果

獨立的 9 頁 LaTeX／PDF 圖解文章，固定程式基準 `3cbaa518f0dc37d691404e95ed85c0109c81130d`，說明 HR 1280×768、G1 NVOFA、CUDA Graph ON 的資料流。內容沿著影像、候選、運動、追蹤狀態與結果交付展開。

- `dataflow.pdf`：可直接閱讀的文章，含四張向量圖。
- `dataflow.tex`／`output-notes.tex`：可編譯原稿。
- `assets/flow.*`：一幀資料的兩條分支與回饋。
- `assets/leases.*`：MI／candidate／result rings、ready／consumed、RGB 保護與背壓。
- `assets/state.*`：committed／working、獨立預測暫存、成功發布與失敗保留。
- `assets/schedule.*`：n／n+1 依賴示意與既有 G1 Nsight 時間軸。
- `sources.json`／`contract-audit.json`：31 個固定來源物件及資料物件、型別、閾值、owner／lifetime 的實作核對。
- `figures.json`／`copied-assets.json`：圖稿來源與 hashes；三張新圖依實作繪製，排程圖沿用已核對圖稿的相同 bytes。
- `delivery.json`／`SHA256SUMS`：文章、圖稿與編譯檢查的收據。

## 編譯

需要 XeLaTeX、latexmk、ctex、fontspec、tcolorbox、常見 TeX Live 套件，以及 Noto Sans／Serif、Noto CJK TC、DejaVu Sans Mono 字型。

```sh
make -C docs/reports/latex/dataflow
```

也可直接進入本目錄執行 `make`。原稿與現成 PDF 圖稿可以獨立編譯，不需要 GPU、模型、資料集或原報告目錄。`make clean` 移除輔助檔，保留 PDF。

## 圖稿與來源

三張新圖可於專案根目錄重建：

```sh
python3 -B docs/reports/latex/dataflow/render_figures.py
```

產生器使用 Python 標準函式庫、`rsvg-convert`、`pdffonts`、`pdftotext`，先核對九份實作來源的固定 hashes，再輸出 SVG 與嵌入字型的向量 PDF。它是手工布局的資料流說明，不由 source hashes 自動證明圖文語義。

排程圖來自 `evidence/report1/2026-11-12-schedule-trace/` 已保存的 G1 trace 分析。其 SQLite 壓縮副本與 `analyze.py` 可用 Python 標準函式庫重算，不需要原 worktree 或 GPU；原 capture 是 2026-11-05、乾淨 `fae5cb7`、driver 617.42。這個歷史時間軸和本篇 `3cbaa518f0dc` 實作概念分開標示，OFA 本體未被直接量到，profiler 區間不替代產品延遲。

## 本文保留的精確區分

Predict 先寫獨立 `DevicePredicted` 暫存，update 才寫 working TrackState。TrackState／gray／mask 共用提交索引；Control 本身只有一份。MI／CS 的兩格 ring、host result 的四格 ring，以及跨幀 state 的兩份 payload 是不同資源。

每幀 D2H 是 native MotionContext 56 B 加 canonical packet 12,368 B，合計 12,424 B；不是下載整張影像。先依 packet status 解讀結果，FAILED／SKIPPED 所附 motion 未必是新當幀估計。

JPEG sequence 的 EOF 插值與 ID 定稿、stream 逐幀保留原 output ID、GPU done／receive／最終 MOT 完成三種終點都分開說明。現行 tracker 輸出 confirmed、age=0 的 observed rows，預設框來自 Kalman state；writer 接收全部 packet rows，不另篩 flags。

## 交付包

`docs/reports/dataflow-20261112-latex.zip` 包含本篇 PDF／原稿、四張圖、產生器、來源快照與排程重算資料，保留專案相對目錄。解壓後直接到 `docs/reports/latex/dataflow/` 執行 `make`；`sha256sum -c SHA256SUMS` 可核對本篇檔案。

這次只做 source review、圖文與文件編譯，沒有更動 runtime／模型／預設、重跑 GPU、品質或 sanitizer。完整 G1 memcheck、原生完整 Graph racecheck、nvJPEG initcheck 等 OPEN／FAIL 保留；文件審查不升格為 runtime 正式驗收。
