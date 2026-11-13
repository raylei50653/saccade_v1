# Saccade v1 模型訓練圖解報告

v1.0（2026-11-13）說明部署中的 HR Mamba 偵測模型是怎麼訓練出來的：資料怎麼分工、Teacher → Cache → Distill → GT1 四個階段各學什麼、怎麼選 epoch、從 10／30 的 T01 到 11／03 定案之間改了哪些地方，以及權重怎麼匯出成 TensorRT FP16 engine 並驗收。主文七節、附錄三節，共 11 頁。

- `training.pdf`／`training.tex`：報告本體。
- `assets/train-*.pdf`／`.svg`：五張圖，分別是訓練鏈、資料切分與前處理、選模曲線、逐步改進、匯出與部署驗收。
- `milestone-table.tex`：附錄 B 的逐步改進表，與第 5 節的圖取自同一份 CSV。
- `data/`：圖表用到的資料快照。四份 CSV 原樣取自研究庫 `saccade_v1_dev` 的 `research` 分支 `92542386`（`evidence/hr/2026-11-02/figures/`、`evidence/hr/2026-11-03/full25/epochs.csv`）；`milestones.csv` 整理各步驟的選定 AP50，每列註明出處。
- `render_figures.py`／`figures.json`：由 `data/` 產生圖與附錄表，並記錄輸入與輸出的 SHA-256。

## 重建

於專案根目錄執行：

```sh
python3 -B docs/reports/latex/training/render_figures.py
make -C docs/reports/latex/training
```

產生器只用 Python 標準函式庫、`rsvg-convert` 與 `pdffonts`；編譯需要 XeLaTeX、latexmk、ctex 與 Noto Sans／Noto Sans CJK TC、DejaVu Sans Mono 字型，與第一輪報告相同。不需要 GPU、模型、資料集或研究庫。`make clean` 只移除輔助檔。

## 數字從哪裡來

- 訓練配方、各 run 的選定點與曲線：研究庫 `92542386` 的 `docs/HR_TRAINING.md` §1–§16、`docs/HR_MOTSYNTH_STUDENT.md`、`docs/TEACHER_CW_EXPERIMENT.md`、`docs/CW_PIPELINE_TRAINING.md`、`docs/GT1_NATIVE_RESULTS.md`，以及同 commit 的 `evidence/hr/`。
- T01（640）與資料切分：本庫 `docs/T01_TRAINING.md`、`manifests/t01/split.json`。
- 匯出與部署驗收：本庫 `training/hr_deploy/README.md`、`docs/TRACKING.md` §15–§17。
- 跨資料集與速度比較：第一輪報告 `docs/reports/latex/report1/` 與其引用的 `evidence/model-comparison/2026-11-11-*`。

所有分數都是既有紀錄；製作這份報告時沒有重新訓練、推論、計分或調參，部署模型與預設設定不變。MOT17-10 是所有選模決定所用的驗證序列，分數偏樂觀；學生階段多為單一 seed。研究庫的 `research` 分支目前沒有對應的遠端分支，HR 訓練程式與原始紀錄只在本機。
