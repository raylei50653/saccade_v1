# MambaPSA 固定來源與 native 交付查核 — 2026-10-29

這次新增 repository tree、作者模組、YAML、dependency pins、訓練參數及論文全文的唯讀查核。沒有執行 Python、載入 checkpoint、複用作者程式或產生任何 scan 數值結果。下載的 `.py.txt` 只作來源快照。

固定 source：[`henrychan0719/MambaPSA@2940f1634cfce1f63201e6411ffe9008951461f3`](https://github.com/henrychan0719/MambaPSA/tree/2940f1634cfce1f63201e6411ffe9008951461f3)。本次 `current-main.json` 的 SHA 相同，recursive `tree.json` 的 `truncated=false`；查核限此完整 tree 與 releases API 當次結果，沒有據此聲稱不存在私有或其他未連結交付。所有取得檔案的 byte size、SHA-256、URL、命令及 exit/output 分別見 `SIZES.txt`、`SHA256SUMS`、`*.command`／`*.exit`／`*.log`。Git blob SHA 另見 `inventory.log`，不冒充檔案 SHA-256。

| 新增核實 | 固定 locator 與結論 |
| --- | --- |
| 主模型位置 | `yolo26n-mambapsa.yaml:12–24` 在 backbone 尾端以 MambaPSA 替換 C2PSA；`:26–39` 仍以一般 `Detect` 消費 P3/P4/P5。這是 backbone 改造，沒有建立本專案「保留 YOLO26s backbone／neck，另交付 Mamba-FPN head」的身分對應。 |
| 模型尺度 | YAML `:5–10` 確實列有 small scale，不能說作者程式不支援 small；但已交付 `mambapsa-args.yaml.txt:3–9` 指向 nano config、VOC、100 epochs、640。論文 §4.1／§4.2 亦是 VOC／YOLO26-Nano。沒有已核實的新 small head lineage。 |
| Neck 變體 | `yolo26n-mamba-p4.yaml:23,34–39` 保留 C2PSA，在 P4 加 BiVisionMamba，再接一般 Detect；不是獨立的新 cls/reg head。 |
| 可見空間與方向語義 | `mamba_psa.py:24–28` 將 NCHW 的 H/W flatten 成 token 後轉成 NLC，再還原；mono-direction core。`bivision_mamba.py:28–34` 同樣 flatten，兩個不同 Mamba instance 分別前向與反向，逆向結果 flip 還原後相加，再 linear projection 與 residual。這些是作者候選模組的語義，尚未獲選為目標 head。 |
| Scan 內部來源 | `mamba_psa.py:4,13–17` 與 `bivision_mamba.py:4,10–18` 委託 `mamba_ssm.Mamba`；requirements `:78` pin `mamba-ssm==2.3.2.post1`，`:190` pin torch `2.8.0+cu126`，`:204` pin ultralytics `8.4.67`。這是作者 repo 的 package version 宣告，不是本專案 target source revision，也尚未固定 dependency 的實際 code revision。作者模組本身沒有提供完整 A/B/C/D、delta、broadcast/stride、initial/final state、精度或容差契約；若將來採用，還須固定 dependency 的來源與路徑。 |
| CPU 對照限制 | `mamba_psa.py:21–23`、`bivision_mamba.py:22–26` 在 CPU 直接返回輸入，呼叫此路徑會跳過 scan。本次沒有證據判定論文 CPU 測量實際使用此路徑；論文 §4.2 也沒有提供 scan 的 native expected vectors，CPU throughput 不能解除數值 oracle 缺件。本次沒有重跑或判定作者數據正誤。 |
| 權重已存在，native 交付未見 | 完整 tree 有 13 個 `.pt` 檔案（`checkpoint-inventory.log`），其中 VOC MambaPSA `best.pt` 為 tree 宣告 5,285,786 bytes、Git blob `edbc43688e2e1ebed4df770c1c707bf8c45490cf`。這次沒有下載或載入權重，沒有宣稱檔案 SHA-256／內容已驗證。tree 未見 ONNX、external tensor data、safetensors、C++/CUDA 實作、engine 或 export script；`native-candidates.log` 空，releases API 為 `[]`。存在 PyTorch checkpoint 不能解除本專案 native 入口缺件。 |
| 授權有文件缺口 | README 的 License 節宣告 AGPL-3.0 並連結 `LICENSE`，但固定 tree 沒有該檔案，repository API `license=null`。記錄宣告與缺少法律文本兩項事實，未自行補上 grant。arXiv HTML 宣告 perpetual non-exclusive license，不能當作程式授權。 |
| Lineage | 固定作者 repo、論文 v1、YAML、訓練 args 與 `.pt` tree inventory 可定位作者候選；沒有 exporter commit、native bundle、與本專案新 head 的 training/architecture binding。舊模型或同名研究的結果沒有轉入驗收。 |

本次取得格式為 UTF-8 原始碼／設定的文字快照、GitHub JSON 與 [arXiv v1 HTML](https://arxiv.org/html/2607.12681v1)，不是可執行模型。

缺件分類與可解除材料：

| 分類 | 精確缺件與下一步 |
| --- | --- |
| 目標身分／設計待定 | 本專案模型設計 owner 尚未提供將此候選選為新 head 的決定。若採用將改動既有 backbone／neck，須先記錄對 B03 的影響；不推薦靠同名 Mamba 或 small scale 欄位直接替代。S01 仍缺目標對應。 |
| 實作已存在但不等於目標 | 作者模組與 PyTorch checkpoints 在固定 tree 已存在；如專案設計 owner 明確採用，作者 repository 持有人或 corresponding author 可提供對應 small 版 architecture/weights/training lineage。README 聯絡列 Jen-Shiun Chiang，`chiang@mail.tku.edu.tw`；本次沒有聯絡。作者不對本專案獨立 head 的存在負責。 |
| Native 途徑缺失 | 如採用，需由該模型交付者提供 ONNX＋external data 或 C++ 可讀權重 bundle，附 immutable revision、grant、size/SHA-256 與 plugin/scan 映射。只要求重發 `.pt` 不能解除 P01。 |
| Scan 契約與數值依據缺失 | 模型設計 owner 需先確認 operator 與 dependency revision、精確公式／tensor 邊界／reset／empty 規則、獨立 expected 及事先選定的 metric/atol/rtol，才能開始 S01 scalar reference。作者 CPU identity 路徑不能充當此 oracle。 |
| 文件不一致／授權缺文 | README 指向不存在的 LICENSE；請 repository 持有人交付固定 revision 的法律文本。論文 CPU 數據與 CPU bypass 的數值語義應另由作者說明，不能由本專案猜測。 |

重跑：`sh evidence/scan-contract/2026-10-29/mambapsa/replay.sh`，預設寫入 ignored `build-scan-source-replay/mambapsa`。不改任何既有 evidence，不啟動安裝或模型程式。
