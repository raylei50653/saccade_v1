# F02 tile fusion／F03 N16 clean-source 整合補驗

2026-11-05。**PASS，限乾淨 `23ade536906d07ccf09859f1b580b044dafe95dc` 與 `v1-motion/1`。**
本輪以新的隔離 checkout 重新建置並執行原 TRACKING §14.5／§17.3 gate，
另補 F02／scan 的完整獨立 suite 四種 sanitizer；`source_dirty=false`、`formal_acceptance=true`。
最終判定由[獨立 postflight](control/methods/postflight.py)與[closeout.json](control/capture/closeout.json)支持。

受測 profile 為舊 `v1-motion/1`（cuFFT motion），不是現行 G1／`v1-motion/2`。
現行 G1 的[正式報告](../../tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md)
仍是完整 session memcheck **OPEN／NOT A FULL PASS**；本輪不關閉它。
元件 hashes 與核對時的 main `23e2fbad218e6bb58184cd073adae97149e6c6f8` 相同，範圍僅為
[equivalence 清單](control/capture/component-equivalence.json)，不將此結論擴大為兩個完整 runtime 相同。

| 項目 | 結果與證據 |
| --- | --- |
| Clean source／build | CPU CTest 16/16、CUDA CTest 59/59；[原 gate capture](acceptance/capture/acceptance.txt) |
| Session 與生命週期 | 原固定 gate 的 packet／serial／pipelined、HR MOT17-10、session sanitizer 範圍通過；命令、exit、summary 均保留於 [acceptance/capture](acceptance/capture/) |
| HR Python references | 四張 RGB／FP32 全元素 bitwise 相同；[log](control/capture/hr-python-reference.log)、[manifest](hr-references/manifest.json) |
| F02 standalone | 72 cases；memcheck／initcheck／racecheck／synccheck 均零 errors／hazards，沒有 kernel filter 或 launch-count 限制 |
| Scan standalone | 四工具通過；正常及其他三工具 368 checks，initcheck 360 checks；完整 groups、output／status／stream／graph 等既有測試 |
| Sanitizer 正控制 | 四工具各 exit 99 且診斷相符，postflight 逐個獨立核對；不只採信原 acceptance.txt |
| HR 七序列 | 5,316 幀；7 SCDR、7 MOT、7 offline MOT 與 M4 **21/21 相同**；[expected hashes](references/m4-outputs.sha256)、[新 outputs](evaluation/capture/outputs.sha256) |
| 保管 | 原接受 CUDA binaries 在補充工作前先封存；後續不重建；source、binaries、inputs 與 references hashes 經核對；[summary.json](summary.json) |

initcheck 依原 harness 契約使用 `--no-held-stream`，只略過 host callback 擋住 stream 的測項；
一般執行與其他三工具仍執行該項。Scan 的 shuffle mask 正確性仍依既有數值與 group-uniform 契約，
不由 synccheck 零報告單獨推導。原 gate 的正控制條件可能未阻擋非 99 狀態；
本輪另外以 postflight 明確驗四個 exit=99 及預期診斷後才定最終 PASS。

本輪**未重跑 ABBA、未讀 GT 或重新計分、未重跑 S02／F04 semantic mutations**。
這是原整合 gate 的補驗，不重新核發 S02 的完整原始 primitive 證書。
全序列時間僅為單次觀測，不宣稱新的加速比或品質結果。
原 11/04 dirty-source 加速量測與品質重測保留其 `source_dirty=true`、`formal_acceptance=false` 身分，
原始報告、失敗嘗試與 bytes 均不改寫。部署 engine 未重建，其原 build provenance 亦保留。

完整 HR session racecheck、TensorRT／nvJPEG／cuFFT 內部 racecheck／synccheck、
完整 session initcheck及跨 GPU／driver／SDK 支援不由本輪 standalone／scoped 結果關閉。

## 重跑與封存

外層方法為 [run.sh](control/methods/run.sh)，GPU lock 與獨立 clean checkout 的執行方式
保留於 [control capture](control/capture/)；原受測 scripts 及契約放在
[source/contracts-and-runners](source/contracts-and-runners/)。
Runner 的輸出目錄必須全新；不可直接覆用本次已存在的目錄。
Postflight 在 campaign 結束後驗 exits、精確 summary、source／binary custody 與 21 個 M4 outputs。
Archive helper 只在 closeout PASS 後封存，不執行 build、測試或 GPU 工作。

本機 checkout：`/home/ray/developer/ai/saccade_v1-worktrees/f02-f03-clean-20261105`。大型 binaries／packets／tensors／outputs／nsys／sqlite
不複製入 Git；位置及 hashes 見 [local-artifacts.json](local-artifacts.json)。
CTest logs、CMake cache、compile commands 保留於 [build-metadata](build-metadata/)。
每個封存檔案以 `SHA256SUMS` 核對；[archive-provenance.json](archive-provenance.json)
記錄來源位置、複製檔案與 helper hash。

```sh
cd evidence/f02-f03/2026-11-05-clean-23ade53
sha256sum -c SHA256SUMS
```
