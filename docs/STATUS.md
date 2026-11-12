# Saccade v1 目前狀態與文件狀態

更新日期：2026-11-12。依目前 checkout 的文件、正式報告與驗收腳本整理；F02／F03 加速已補上乾淨 `23ade53`、`v1-motion/1` 的正式整合驗收，以及固定 engine 的 clean-source ABBA 效能量測；原開發測時與現行 G1 memcheck OPEN 分列。完整文件分組見 [README](../README.md#文件分組)，接續工作見 [BACKLOG](BACKLOG.md)。

[第一輪詳細報告](reports/report1-3cbaa518f0dc.md)固定以 `3cbaa518f0dc` 整理系統／模型／同步架構、端到端計時邊界與既有延遲證據；G1 全序列吞吐、較早 cuFFT 的逐幀分位數與 dirty-source 階段歸因分別標示。該詳細稿的原次整理只重算既有資料，未新增 GPU 量測或改變部署預設。

2026-11-12 報告 v1.1 補入[官方／微調 YOLO26m 與 HR Mamba 三方比較](reports/report1-3cbaa518f0dc.md#92-官方原版對照與各資料集表現)：五資料集、156 段、每模型 135,503 幀；HOTA 七個主要群組 HR 對微調為 3 勝／4 負，逐段 55 勝／101 負。MOT17／MOT20 合併較高，DanceTrack／PP22 較低，另列受控 TRT 速度與 engine 成本。品質採 raw6 one2many＋共用 F04／G1；官方原始 one2one 僅有模型層控制。既有資料已曝露、FP16 數值 screen FAIL 與原 sanitizer OPEN 保留；本次只整理已完成研究，沒有新推論、計分、調參或部署變更。

同日 v1.2 改為[圖解報告](reports/latex/report1/report.pdf)：主文先講目的，再用真實三影格、五步流程、五資料集比較、七群組增減與延遲／吞吐圖解讀結果。主 PDF 共 10 頁，原 20 頁 [v1.1 技術版](reports/latex/report1/technical-reference-v1.1.pdf)及詳細稿的技術段落保留；只改圖文、重建文件，未新增推論、評分或實驗。

同日 v1.3 將 n／n+1 排程加回圖解主文第 3 頁（全稿 11 頁），以[原始 G1 Nsight trace 重算](../evidence/report1/2026-11-12-schedule-trace/summary.json)畫出示意與實測時間軸。歷史乾淨 `fae5cb7`／driver 617.42、30 幀中，排除初始化與首次 Execute 後完整 F06 28/28 落在同幀 F03 內；27/27 組相鄰穩態幀有跨幀 CUDA 活動交集，平均 0.937 ms，主要為 F03(n+1) 與上一幀 mask／tracking tail。壓縮原始 SQLite、frame／graph 角色映射、逐幀表及獨立稽核一併保存；本次沒有新 GPU capture，profiler 觀察不作新延遲、加速或 sanitizer 驗收。

同日 v1.4 在圖解報告第 6 頁加入 single 平均值、P95／P99 標記，以及 single／queued 尾延遲表。原 48 份逐幀 CSV 合併後重算 12 組、每組 480 筆樣本，P95／P99 採 nearest-rank 第 456／476 筆，與保存摘要相同；沒有平均各 run 分位數或重新測速。MOT17-10 的 HR single P95 雖略低於微調版，P99 卻較高，02 則兩者都較高。全稿仍為 11 頁，核對收據與來源檔隨交付包保存。

同日 v1.5 只改文字：第 4 頁與詳細稿 §1.4 標明 MOT17 七段中 04／05／09／11／13 為兩個本地模型的訓練序列、10 為選模，held-out 02 的 HOTA 為官方／微調／HR 36.761／37.710／40.565，選模 10 為 HR 低 1.332；官方到微調的 MOT17 合併差距 15.02 在 02 只剩 0.949。未參與本地訓練的 MOT20 為較能代表跨資料集的比較（微調較官方 +12.756，HR 再 +7.742）。第 6 頁開頭改回 HR 平均延遲 +14.79–17.42%、吞吐 −15.13–15.52%；MOT17-10 single 尾端差距小於 0.2 ms、時脈未鎖定，不再當成結論。數字均取自既有逐段 CSV 與摘要，沒有新推論、計分或測速。

同日另製作[獨立 Dataflow LaTeX／PDF](reports/latex/dataflow/dataflow.pdf)，9 頁、四張圖，固定 `3cbaa518f0dc`／HR／G1 說明資料物件、D／M 分支、ready／consumed、預測暫存、共同 commit 版本及 CPU 輸出邊界。31 個來源物件與基準 bytes 核對，保留已核對的歷史 Nsight 時間軸；只做圖文與編譯，沒有新推論、測速或驗收，原第一輪報告保持原樣。

2026-11-11 另製作[主管／專案交付正式報告（PDF）](reports/latex/report1/report.pdf)與 [LaTeX 原稿](reports/latex/report1/README.md)，並補跑乾淨 `3cbaa518f0dc` 的 [G1 逐幀延遲](../evidence/report1/2026-11-11-g1-live/REPORT.md)：JPEG 預載、MOT17-10 前 140 幀、每模式 4 fresh sessions／480 筆暖機後樣本。Single mean／p95 為 6.712865／6.981673 ms；四格 queued 交付 mean／p95 為 20.254368／21.181116 ms，interval 5.138200 ms/frame（194.620666 FPS）。八份完整 packets 相同；driver 621.14，與歷史 617.42 分列。完整 654 幀離線程序（含啟動／讀寫／定稿）三次平均 4.836436 秒，MOT／packets／motion／設定三次相同；不當作逐幀 latency，檔案 cache 未清，另存[獨立收據](../evidence/report1/2026-11-11-g1-offline-e2e/REPORT.md)。兩組皆不構成品質重評或 sanitizer 驗收，部署預設與既有 OPEN／FAIL 保留；報告及新增收據尚未 commit／push。

同日新增[現場技術展示台](demos/report1/index.html)與[離線操作說明](demos/report1/README.md)：支援 MOT17-10（654 幀／10,385 筆 observed rows）、02（600 幀／11,454 筆）與 04（1,050 幀／43,003 筆）切換，提供並排、逐幀、慢放、ID／UID、G1 位移、播放框插值與預設開啟的 MOT 缺幀補框。補框與正式 F09 相同：依 output ID、至少 5 筆 rows、最多 35 個缺幀，允許 UID 接回；虛線與「補」標籤區分原始觀測，分數、筆數與原座標分開顯示。三序列補框 934／2,040／2,053 筆，合計 5,027 筆與獨立 F09 計算完全一致，另核對既有 69,869 筆 MOT 的三位小數捨入界，見[對照收據](../evidence/report1/2026-11-11-demo-mot-interpolation/REPORT.md)。原圖／資料／三支備用 MP4 保持原始逐幀；不改寫來源封包。一般畫面使用原生影片搭配透明框，並排模式快取來源幀。Chromium file／HTTP 的 43 項互動檢查與 17 項補框規則測試通過，見 [validation.json](demos/report1/validation.json)。

02／04 的展示來源仍為相同乾淨 `3cbaa518f0dc`／原 engine 各完整擷取一次，見 [02 收據](../evidence/report1/2026-11-11-mot17-02-demo/REPORT.md)及 [04 收據](../evidence/report1/2026-11-11-mot17-04-demo/REPORT.md)；補框接入沒有新推論、讀 GT 或重評品質。02／04 的效能欄位仍顯示尚未量測，切回 10 恢復原量測；27 個展示來源資產 hashes 未變，runtime、預設、正式驗收與外部發布狀態未改。

2026-11-11 [第一輪報告版本收尾](history/2026-11-11-first-report-baseline.md)：固定核心完成、既有正式驗收、效能與 OPEN 項目的引用入口；收妥評分／資料集直連及 base weights 路徑更新。MOT17／PP22 評分已使用本庫 TrackEval 與獨立 CPU 環境，共享資料與權重的身分核對見[離線評分](EVALUATION.md)、[外部路徑](EXTERNAL_PATHS.md)及收尾紀錄。本輪僅執行本機 CPU 回歸與路徑核對，未重跑 GPU、訓練、品質或效能驗收；部署預設與既有正式證書的受測來源保持原判定。

2026-11-09 [NSA v2 與量測框（TRACKING §22）](TRACKING.md#22-nsa-v2-與量測框opt-in2026-11-08-規格實作前固定)以**預設關閉**的可選設定加入 tracker：`tracker.kalman.nsa_enabled`、`nsa_s0`（0.9305）與 `tracker.rows.measurement_box`。乾淨 `527ffe5` 的[正式驗收](../evidence/nsa-v2/2026-11-09-formal-527ffe5/REPORT.md) **PASS**：預設關閉時 80 序列 offline 與 76 序列 HR native 輸出不變；開啟時 offline 80/80 等於研究 G2，CUDA 對 CPU 80 序列逐位元相同，HR pipelined／serial 等於 offline，tracker 四種 sanitizer 0 errors。品質沿用研究 G2 結果（探索性候選），未重新計分；部署預設未改，改為預設須另做部署驗收。預設 tracker config hash 因新增欄位改為 `d846e2a5…`，事件檔 schema 升為 `tracking-events/2`。

新增可選的 [即時影像串流組件](STREAMING.md)：FFmpeg 收流 → NVDEC GPU 解碼 → CUDA 色彩轉換／前處理／TrackingSession，提供 `video_probe` 與 `track_stream`。CLI 預設 GPU、CPU decode 須明確選擇；即時輸出政策與開發測試獨立記錄，實際裝置連線邊界見該文件，不改寫原 JPEG／P01 正式判定。

2026-11-06 [正式發布與獨立研究分離](RESEARCH_BOUNDARY.md)：整理前 runtime R0 固定 `dacb91a`；正式 CUDA 依賴改連 tracking core／flow records，CPU hooks 與九臂研究實作由獨立研究建置管理。三庫來源及15個資料單位已保存並實際還原核對；本輪針對性驗證以研究區的新收據記錄，既有正式證書與 sanitizer OPEN 邊界保持原判定。

2026-11-06 [P08 設定與事件工具](P08_TOOLS.md)：`track_offline`、`track_sequence`、`track_stream` 共用 resolved tracking schema，提供明定的覆寫優先序、穩定設定 hash、有效值與來源，以及獨立 build／binary／engine 身分收據。CPU 實際決策事件、lifecycle／身份／提交模型與完整 packet binding 已落地，結果見[本機開發來源報告](../evidence/p08/2026-11-06-development/REPORT.md)。video I/O 的格式、timeout、transport、frame limit 另列收據，未納入 tracking config hash；GPU 完整中間事件 dump 與觀察開關一致性仍 OPEN，P08 尚未整體結案。

2026-11-05 接續：目前 TrackingSession GMC 已採 [G1 NVOFA／v1-motion/2（TRACKING §20）](TRACKING.md)；該輪驗收除完整 session memcheck OPEN 外，其餘八步通過，依 [原報告](../evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md) 判定。以下較早正式證書仍保留各自來源與 motion profile。

2026-11-05 已追加 [HR 完整路徑 sanitizer 補驗](../evidence/hr-sanitizer/2026-11-05-a00ab3a/REPORT.md)：乾淨 `a00ab3a` runtime、現行 G1，CUDA CTest 68/68、HR 30 幀 602 checks、HR lifecycle 180 checks；四種排程及 HR lifecycle 的**未過濾完整 synccheck PASS**。四種排程 memcheck 均完成且 outputs 相同，但各有 86 次 NVOF 診斷，lifecycle 為 1,008 次，仍 OPEN。未過濾 racecheck／lazy retry 及自訂＋CUB racecheck 仍 24 GiB OOM，initcheck 出現 nvJPEG 診斷後 900 秒 timeout；**不是 sanitizer 全通過**。

[Racecheck RAM 來源追蹤](../evidence/hr-sanitizer-memory/2026-11-05/REPORT.md)已定位到 target 內 `libsanitizer-collection.so` worker 的大型暫存配置；在已追蹤 ≥64 KiB heap requests 中占約 97–98%。Pointwise 原生排隊碰到 6 GiB OOM，強制逐次同步對照則完整暖機、峰值約 1.08 GiB；最小 CUDA probe 亦量到 grid 相關配置成本。這是來源診斷，私有 payload／洩漏與工具 bug 分類仍未確認，原生完整 HR racecheck 仍 OPEN。

2026-11-06 [Racecheck 有界排隊補驗](../evidence/hr-sanitizer/2026-11-06-bounded-queue/REPORT.md)：相同乾淨 `a00ab3a` runtime，**未過濾 ordinary serial 30 幀**以 8 workers、sync limit 1 完整跑完，exit 0、零 errors／warnings，約 50 分 20 秒、cgroup peak **1.031 GiB**，MOT／完整 packets／SCMO 與普通及 Graph baselines 逐 byte 相同，前後 custody 一致；標為 **PASS_FORCED_SYNC**。此入口使用 `--serial --no-cuda-graph`，部署 runtime／engine／Graph 預設維持原版。

全部自訂 kernels＋CUB 另以 8 workers、sync limit 8 跑完四種排程各 30 幀及 HR lifecycle 180 checks，零 errors／warnings、outputs 相同、最高 peak 6.270 GiB，標為 **PASS_FILTERED_FORCED_SYNC**。未過濾 Graph 的 sync8／8 GiB 與 sync1／24 GiB 重試仍 OOM；原生排隊／未過濾 Graph gate 仍 OPEN，不由上述診斷入口關閉。

三輪 sanitizer 工作的[發布前核對](../evidence/hr-sanitizer/2026-11-06-publication/REPORT.md)另行保留：新建 CPU CTest 20/20、Python checks 全通過，250 個 frozen artifact hashes／sizes 與受測 source 身分一致；不重寫原 GPU 收據或擴大正式驗收範圍。

本輪 [OAO duration-ramp／front-occluder 消融](../evidence/occlusion/2026-11-05/REPORT.md)只讀 MOT17-10 val；四臂 CPU replay 未見收益，both 的 HOTA 50.869→50.634、IDF1 64.975→62.998、IDSW 56→67，**不加入部署預設**。交付前 CPU 20/20、baseline 654 幀 packet 相同、補上 CLI 檢查後四臂 MOT bytes 相同；非 CUDA／session／效能驗收。

本輪 [MOT17-10 IDSW／斷軌回放](../evidence/tracking-events/2026-11-05/REPORT.md)核對 56 次 IDSW、150 次 CLEAR Frag，合併後 159 個事件另加 10 個末段未接回。654/654 CPU packets 與 stock reference 相同，MOT 與 G1 baseline bytes 相同；提供本機互動回放與候選／batch／track／成本線索，尚非因果歸因，未更改部署。

2026-11-06 [外部路徑與符號連結整理](EXTERNAL_PATHS.md)：修復七個搬移後的測試 alias，T01 cache 的 head bundle 改連 cache 內同 hash 副本，解除對本專案 `build/` 的反向依賴；新增唯讀盤點工具。HR manifest 的 11 個歷史來源路徑缺口保留，原 manifest／驗收證據未改寫。`g1-ofa-base` 是驗收第 8 步新建的輸出，其不存在不列為缺件。

## 目前部署與最新判定

目前路徑為 **JPEG sequence → F01／F02 → F03／F04 → F05–F09 tracking → MOT**；部署模型為 **HR 1280×768 FP16**，scan plugins 與模型 input／raw outputs 維持 FP32。部署 engine 以 SHA-256 `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876` 固定；來源、精度、shape binding 與支援邊界見 [TRACKING §15–§17](TRACKING.md#15-d1hr-偵測器部署2026-11-03-使用者核准)。

**M0–M4 完成，P01 CLOSED（有範圍）**。P01 的正式判定由 [M4 報告](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md) 與 [TRACKING §17.4](TRACKING.md#174-p01-判定標準) 定義，不表示任意模型、GPU 或 SDK 的一般支援。

CUDA Graph 預設啟用，D graph 包含 F03→producer→F04，M graph 包含候選檢查→F04b→關聯更新→commit→packet；前處理、motion／GMC、跨 stream events 與結果 D2H 仍由 session 排程。[最新 clean-source 正式驗收](../evidence/cuda-graph/2026-11-04-gate-d21a144/REPORT.md) 全部 PASS：

| 項目 | 受測來源與結果 |
| --- | --- |
| CPU、F04、語義變體 | 乾淨 `4329ed7`；CPU 16/16、F04 CTest 5/5、mutations 15/15；F04 四種 sanitizer 與正控制通過 |
| TrackingSession | 乾淨 `d21a144`；CPU 16/16、CUDA 59/59，七序列 CPU／ordinary serial／graph pipelined packet bytes 相同，含 HR 路徑及報告所列 sanitizer 範圍 |
| HR 七序列回歸 | 乾淨 `d21a144`；身分鏈、offline／sequence 比對及 stream 守門通過；7 個 SCDR、7 個 MOT、7 個 offline MOT 與 M4 **21/21 相同**；未讀 GT、未重新計分 |
| Stream 位置 | node／graph 兩次 capture 配對 CUDA Graph launch；8 個負控制均被偵測，規則見 [TRACKING §17.3](TRACKING.md#173-正式執行) |

各步的受測 source 分開記錄；後續開發來源重測不改寫這些正式證書。

後續 [HR F02 tile fusion](../evidence/f02-fusion/2026-11-04/REPORT.md) 為開發來源改版，僅分派 1920×1080→1280×768；128 組舊／新 bits、4 張 HR Python references、60/60 CTest 與新 kernel 的有範圍 sanitizer 通過。該輪尚未執行新的 clean-source 正式 gate，不改寫上表的證書來源。

再後續的 [F03 N16 特化](../evidence/f03-scan/2026-11-04/REPORT.md) 保留既有 Graph；scan 精確比較、四張完整模型輸出、16 sessions 的 140 幀完整 packets、此 build 的 49/49 CTest 與 scan 四種 sanitizer 通過。該輪為開發來源，未執行新的 clean-source 正式 gate 或品質重評。

上述 F02 fusion、F03 N16 與部署 shape／128-thread 特化已一併重新建置，完成[完整 MOT 開發測試](../evidence/mot/2026-11-04-full-development/REPORT.md)：CPU 16/16、CUDA 59/59、報告所列 17 次 sanitizer 與四個正控制、七序列 offline／native MOT 及 node／graph stream 守門均通過。HR 七個序列共 **5,316 幀**，7 個 SCDR、7 個 MOT、7 個 offline MOT 與 M4 **21/21 逐位元相同**；重新讀 GT 計分後品質指標不變。吞吐為 **195.85 FPS**（加權 5.10608 ms／幀，包含 JPEG 讀檔及 pipeline，排除 engine 初始化與輸出寫檔）。本次保留 `source_dirty=true`、`formal_acceptance=false`，MOT17-02 記為 **M4 之後**的重跑，不作新的獨立 eval 或 clean-source 正式證書。

2026-11-05 已完成 [F02／F03 clean-source 整合補驗](../evidence/f02-f03/2026-11-05-clean-23ade53/REPORT.md)：乾淨 `23ade53`、原 `v1-motion/1`，CPU 16/16、CUDA 59/59、原固定 session gate、F02／scan 各四種完整 suite sanitizer 與四個正控制通過。四張 HR RGB／FP32 references bitwise 相同，七序列共 5,316 幀的 SCDR／MOT／offline MOT 與 M4 **21/21 相同**。受測兩個 kernels 與目前 checkout bytes 相同；接受 binaries 已先封存，前後 source／binary／input hashes 一致。本輪未重測 ABBA、未讀 GT 或重跑 semantic mutations，不重發原 S02 primitive 證書；現行 G1／`v1-motion/2` 的完整 session memcheck 仍 **OPEN／NOT A FULL PASS**。

同日另完成 [clean-source ABBA 效能補驗](../evidence/f02-f03/2026-11-05-clean-performance/REPORT.md)：乾淨 `f4865cd`→`23ade53`、同一部署 engine、原 `v1-motion/1`，兩輪 ABBA 的完整 F03 Graph 平均 5.241696→4.904589 ms（−6.43%）；tracking 單幀平均 7.092724→6.626377 ms（−6.58%）、p95 7.524795→7.059455 ms，queued interval 5.536401→5.123390 ms/frame（−7.46%，180.6→195.2 FPS）。四張完整模型 outputs／statuses 與 32 份 140 幀 packets 逐 byte 相同；tracking 限 MOT17-10 prefix、JPEG 預載，排除磁碟／初始化／暖機／輸出，不關閉 G1 或完整 HR racecheck。原 harness 固定 development 標籤保留，clean source／binary custody 由外層 contract 與報告判定。

## 各階段狀態

| 階段 | 狀態與依據 | 範圍與接續 |
| --- | --- | --- |
| 協定／合成 CUDA | [協定模型](PROTOCOL_MODEL.md)、[CUDA 骨架](CUDA_SYNTHETIC.md) 已有有限驗證 | 合成 profile 與真實 tracking 的證據分開 |
| S01–S04 scan／head | [分階段判定](MODEL_FEASIBILITY.md#3-s01s04-的接續條件)、[S04 正式證據](../evidence/s04/2026-10-30/REPORT.md) PASS（各自受測範圍） | 原 FP32／640 候選；不可直接作 HR FP16 的品質判定 |
| T01 實驗 | [2026-10-31 結案與封存](history/2026-10-31-t01-closeout.md)，固定 tag `t01-experiment-final-2026-10-31`／`a47fb16` | 原 checkpoint、選模與封存身分保留；目前部署已換 HR 模型 |
| T01 backbone／F03 | [正式證據](../evidence/t01/2026-10-30/f03/REPORT.md) PASS，乾淨 `4b52b2f` | 六個固定輸入的 FP32 驗收；後續 FP16 模式見 [MODEL_TRT](MODEL_TRT.md) |
| F01／F02 | [使用驗收](../evidence/preprocess/2026-10-31-usage/REPORT.md) PASS（固定輸入與使用範圍） | 原 7 張 JPEG、4 sessions／208 幀；nvJPEG initcheck 診斷獨立保留 |
| F03 producer | [正式驗收](../evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md) PASS，乾淨 `a9c3d9b` | 原 primitive 驗收與後續 HR／session 的證據分列 |
| F04 | [原 adaptive 正式驗收](../evidence/f04/2026-10-31-formal-995bca5/REPORT.md)、[平行 finalize 驗收](../evidence/f04/2026-11-04-parallel-finalize-e4d45de/REPORT.md) 及最新 Graph gate PASS | 保留版本化政策、kept／suppressed、穩定 ID 與完整輸出 ABI；各輪結果見 [POSTPROCESS §5](POSTPROCESS.md#5-平行-finalize2026-11-04-實作前固定) |
| DetectionSession | [2026-10-31 開發來源驗證](../evidence/detection-session/2026-10-31/REPORT.md) PASS（受測範圍） | 獨立 F01–F04 API；不將後續 TrackingSession 正式驗收移作它的 clean-source 證書 |
| TrackingSession／M3 | [乾淨 `e021bd1` 正式驗收](../evidence/tracking-session/2026-11-03-formal-e021bd1/REPORT.md) PASS | 完整 F05–F09、CPU／GPU packets、MOT、生命週期與有範圍 sanitizer；Graph／排序改版另由最新 gate 驗證 |
| HR／M4／P01 | [乾淨 `cad5298` 正式評估](../evidence/m4/2026-11-03-formal-cad5298/REPORT.md) 完成，P01 CLOSED（有範圍） | MOT17-02 HOTA 40.6、AP50 72.91%；只記錄品質，不設品質門檻；最新 gate 只比對輸出 |

## 效能報告的狀態

| 報告 | 已觀察結果 | 判讀限制 |
| --- | --- | --- |
| [CUDA Graph 開發量測](../evidence/cuda-graph/2026-11-04/REPORT.md) | 單幀平均 8.59→7.06 ms，p95 9.08→7.44 ms | 開發來源的 ABBA 量測；首幀 setup 與 steady-state 分列 |
| [Tracking 排序量測](../evidence/tracking-sort/2026-11-04/REPORT.md) | 固定 F04b replay 143.57→102.81 µs；完整 tracking 7.037→6.978 ms | 吞吐幾乎不變，不宣稱穩定端到端加速 |
| [硬體資源／排程](../evidence/resource-schedule/2026-11-04/REPORT.md) | Graph 單幀平均 6.988 ms；queued wall time/frame 5.412 ms，約 184.8 FPS | 後者是吞吐換算；預設 4 格 ring 的另次交付延遲平均 21.217 ms。2 格 ring 的單次觀測為 10.740 ms，不能混作同一量測 |
| [HR F02 tile fusion](../evidence/f02-fusion/2026-11-04/REPORT.md) | 隔離 F02 interval 149.06→89.76 µs（−39.8%）；queued 5.367→5.312 ms/frame | 同起點兩輪 ABBA、32 sessions／140 幀完整 packets 相同；single p50 7.010→6.905 ms，mean 受一筆 102.958 ms baseline 長尾影響。workspace 配置仍保留 |
| [F03 N16 kernel 特化](../evidence/f03-scan/2026-11-04/REPORT.md) | scan API Graph 1083.52→770.103 µs（−28.9%）；完整 F03 Graph 5.051→4.762 ms；tracking 單幀平均 7.013→6.647 ms，queued 5.361→5.068 ms/frame | 同起點 ABBA，原部署 engine；四張模型輸出與 16 sessions／140 幀完整 packets 相同。先測的 broadcast 較慢未採用，正式 gate／品質重評未執行 |
| [llama.cpp 啟發的 F03 部署特化](../evidence/f03-llama/2026-11-04/REPORT.md) | 在既有N16之上，shape＋block128使完整F03 Graph兩輪改善1.15%／1.64%；四張輸出與statuses相同 | 開發來源；scan suite與四種sanitizer通過，未跑新正式gate／tracking重評。raw load＋shuffle與B／C shared copy使完整模型變慢，保留負控制及NCU counters |
| [F03 backbone／head 瓶頸分析](../evidence/f03-backbone-head/2026-11-04/REPORT.md) | 20次Graph／6040 kernel calls完整對映：backbone／neck占kernel duration加總66.12%，conv／GEMM占63.73%，scan core占15.63% | 本輪沒有可採用的新加速。Cache收益確認輪反轉；causal直接對現行engine只差0.014%，重建FP16 engines均非bitwise替代品；開發來源，未跑新正式gate／品質重評 |
| [加速後完整 MOT 開發測試](../evidence/mot/2026-11-04-full-development/REPORT.md) | 七序列共 5,316 幀，吞吐 195.85 FPS；M4 輸出 21/21 相同，品質指標不變 | 每序列單次完整執行；吞吐不當作單幀交付延遲，與歷史測時未做 ABBA。未提交來源，MOT17-02 為 M4 之後重跑；sanitizer 範圍依報告 |
| [F02／F03 clean-source ABBA](../evidence/f02-f03/2026-11-05-clean-performance/REPORT.md) | 隔離 F02 Graph 147.138→88.483 µs；完整 F03 Graph 5.242→4.905 ms（−6.43%）；tracking single mean 7.093→6.626 ms；queued 180.6→195.2 FPS | 乾淨 `f4865cd`→`23ade53`、原 `v1-motion/1`、固定 engine；兩輪 ABBA，四張 outputs／statuses、32 份 packets 相同。限本機／固定模型輸入與 MOT17-10 140 幀 prefix，保留全部長尾 |

上述較早開發報告保留各自的來源身分、樣本與失敗嘗試。先前 Graph／排序的 clean-source gate 補上對應來源的正確性與七序列回歸，不改寫原測時身分或取代其量測條件；後續完整 MOT 開發重測補上 F02／F03 加速的全序列回歸與計分。2026-11-05 的 clean-source 整合補驗與另一輪 ABBA 效能補驗分別提供指定來源／motion profile 的正確性和效能證據；新的量測不改寫原開發 A/B 的原值。

## 未涵蓋與未完成邊界

- **完整函式庫插樁**：[HR 補驗](../evidence/hr-sanitizer/2026-11-05-a00ab3a/REPORT.md)已補上四種排程各 30 幀及 HR lifecycle 的未過濾完整 synccheck，包含受測 TensorRT／nvJPEG／NVOF CUDA kernels；OFA 硬體本體不在 CUDA 插樁範圍。完整 memcheck 仍因 NVOF 診斷 OPEN；完整 initcheck 已嘗試，出現 nvJPEG 未初始化讀取後 timeout，未完成。原 T01 filtered racecheck 的 PASS 不擴成完整 HR racecheck。
- **完整 HR racecheck**：原生排隊／未過濾 Graph gate 仍 OPEN。2026-11-06 [有界排隊補驗](../evidence/hr-sanitizer/2026-11-06-bounded-queue/REPORT.md)完成未過濾 ordinary serial 30 幀（工具逐次同步，`PASS_FORCED_SYNC`），以及自訂＋CUB 四種 30 幀排程／lifecycle（`PASS_FILTERED_FORCED_SYNC`）；Graph 全 kernel 的同步化重試仍碰到 24 GiB OOM。原 [資源報告](../evidence/resource-schedule/2026-11-04/REPORT.md#8-無競態的條件與本次驗證) 與 [HR 補驗](../evidence/hr-sanitizer/2026-11-05-a00ab3a/REPORT.md) 的 OOM／timeout 保留，不以普通 serial、同步化、過濾或 component PASS 取代原生 Graph 驗收。
- **nvJPEG**：F01 initcheck FAIL／原因 OPEN 保留；[使用政策](PREPROCESS_USAGE_POLICY.md) 與 [診斷紀錄](PREPROCESS.md#6-逆向與初始化診斷的證明邊界) 分開判定，不將介入後零報告當作原 runtime 修復。
- **部署與支援**：HR FP16 engine 的 build provenance、HR deploy manifest 與 native schemas 1–3 的邊界見 [TRACKING §17.4](TRACKING.md#174-p01-判定標準)。跨 GPU／SDK 支援、GPU fatal 後恢復與額外 durability 仍依 [BACKLOG](BACKLOG.md) 及 [DATAFLOW §9.2](DATAFLOW.md#92-真正待補齊的工程規格) 各自判定。
- **P08 GPU 事件與觀察驗證**：[設定與 CPU 事件工具](P08_TOOLS.md)已有本機開發來源驗證；CUDA 完整中間事件 dump 與對應觀察開關一致性仍 OPEN，工具明確拒絕 CUDA 完整事件請求。CPU 事件模型與完整 packet binding 不證明 GPU 中間決策、無競態或完整 session sanitizer 通過；video I/O 設定另列收據，不以 tracking hash 宣稱全部串流設定相同。
- **排程擴充**：雙 host staging／完整 host decode overlap 尚未實作；多 caller、新 streams、shape／buffer 或 slot 回收邊改動，需更新 owner／last-use 及對應驗證。當前安全使用條件見 [ownership](../evidence/resource-schedule/2026-11-04/ownership.csv) 與 [安全審查](../evidence/resource-schedule/2026-11-04/safety-review.md)。

## 文件狀態與維護方式

| 類別 | 文件入口 | 維護規則 |
| --- | --- | --- |
| 現行導覽與狀態 | [README](../README.md)、本頁、[BACKLOG](BACKLOG.md) | 隨新增證據更新目前結論、日期與來源；驗收範圍保留 |
| 架構／API／操作 | README 的架構、runtime、模型分組 | 保留契約與重跑入口；早期缺件標明原日期，再連目前部署與對應正式報告 |
| 固定政策 | [HEAD](HEAD_POLICY.md)、[BACKBONE](BACKBONE_POLICY.md)、[PREPROCESS](PREPROCESS_POLICY.md)、[USAGE](PREPROCESS_USAGE_POLICY.md)、[POSTPROCESS](POSTPROCESS_POLICY.md) | 原版本、方法與門檻不因狀態整理變更 |
| 歷史規格與實驗 | [舊訓練流程](TRAINING_WORKFLOW.md)、[T01](T01_TRAINING.md)、[結案報告](history/2026-10-31-t01-closeout.md) | 原選模、checkpoint、實驗與封存結果保留；不當作目前 HR 狀態 |
| Evidence／archive | [README 證據分組](../README.md#驗收與效能證據)、`evidence/`、`docs/history/` | 原路徑、報告、FAIL／UNEXECUTED、hash 與 custody 保留；後續結果另建並由入口連接 |

狀態衝突時，先核對對應階段的最新正式證據、受測 source 與 profile；早期文件的「當時 OPEN」是歷史紀錄，不是目前待辦。文件檢查通過只表示索引與連結一致，不代表重跑 GPU 或品質驗收。
