# TrackEval MOT 評分子集

本目錄提供本專案 MOT17／PersonPath22 評分所需的 TrackEval 程式，評分時不需舊 Saccade checkout。上游為 [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval)，依 [MIT LICENSE](LICENSE) 保存。

## 固定來源

來源是舊 Saccade 已使用的 vendor snapshot，沒有更新到新的上游版本：

- 舊 repository snapshot commit：`e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2`。
- 原目錄：`third_party/TrackEval`；git tree：`b0ba3a0c5b82550156c6afd1ca949fe5ec155161`。
- 原目錄最後變更 commit：`0f164c989b98a2750f1dc6e75185c7f26d3d2220`。
- 此 snapshot 沒有記錄確切 upstream commit，不能把舊 Saccade 的 commit 當作 upstream revision。

[SOURCE.json](SOURCE.json) 記錄來源身分及每個複製檔的 SHA-256。十個評分／dataset／metric Python 檔與 LICENSE 均保存原始 bytes。只有三個 `__init__.py` 改為匯入此處實際提供的 API，避免載入無關 dataset／metric 的額外依賴。來源路徑只作 provenance 記錄，執行時不會讀取該舊目錄。

## 支援範圍

提供 `Evaluator`、`datasets.MotChallenge2DBox`、`metrics.HOTA`、`metrics.CLEAR`、`metrics.Identity` 及 evaluator 自動加入的 `metrics.Count`。本專案的 PersonPath22 評分也使用既有的 MOT box adapter，並由入口腳本先執行 GT keyframe 過濾。

這個子集支援 MOT box 評分與文字／CSV 輸出；呼叫端必須設定 `PLOT_CURVES=False`。未收錄其他 dataset、metric、plotting 模組或上游 CLI。原算法內部保留的 mask／plot helper 並非此子集的支援介面。

必要 Python 套件為 NumPy、SciPy；`tqdm` 為可選套件。此 snapshot 使用舊 NumPy 的 `np.float`／`np.int`，本專案評分入口提供相容 alias。來源演算法保留原樣以維持既有評分語義；套件環境與評分設定由本專案入口管理。
