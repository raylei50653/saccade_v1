# 新 head 契約核實與 S01 接續調查 — 2026-10-29

**新增來源調查與契約矩陣已完成；目標 scan 尚未核實，未進入 S01 reference／數值 fixtures。P01、S01、真實 artifact readiness 仍 OPEN。**

本輪為文件／來源查核交付，沒有修改 native runtime、hash、manifest schema 或預設 CPU build。沒有執行 Python、舊程式、舊權重或舊模型實驗；同名論文／其他候選的數值未轉作新模型證據。原有 evidence 未更動。

## Source 與執行身分

| 欄位 | 本輪記錄 |
| --- | --- |
| 開始／fetch | [initial-session.txt](initial-session.txt)：`HEAD=origin/main=9f2c11a838a229c05ba7e9e17fa182bf0799746b`，初始工作樹乾淨，main 無後續差異。 |
| 既有實作 source | `622a33b8bd2a1a593fcfbc8a12f6b6e30e770d8a`，P01-A 的既有實測留在原 evidence。 |
| 本輪受測 source | `9f2c11a838a229c05ba7e9e17fa182bf0799746b` 的 `git archive`，重新解出後 configure/build；沒有從帶文件變更的工作樹推定 clean build。[source](local/source.log)、[完整 tree](local/tracked-tree.log)、[source hashes](local/source-file-hashes.log)。 |
| Source archive SHA-256 | `91d50f4cb3dcadafb0b19b8653ad5755b3bf6cd702d768dc7f94595797579bdc`；[archive-hash.log](local/archive-hash.log)。 |
| 環境／日期 | 2026-10-29 Asia/Taipei；WSL2 Linux 6.18.33.1、GCC 16.2.1、CMake 4.4.4、Git 2.56.0、Node 24.14.1、jq 1.8.2；[environment](local/environment.log)。 |
| 執行時工作樹 | [working-tree.log](local/working-tree.log) 保留調查中新增文件的 dirty 狀態；這不改變 archive 的受測內容。 |
| 交付身分 | 本報告與契約／來源 captures 為一個文件 bundle；提交 SHA 與實際提交／推送狀態由交付時的 Git 狀態及最終回報記錄，避免報告自含其 commit hash。 |

## 本輪新增的核實結果

1. **引用架構有更具體設計，但無 scan 公式。** 固定舊設計 revision `e48f3b39` 的 math_model §4.1 明列 down4／兩個 MambaBlock／up4、token lengths 400/100/25、`d_model=128`／`d_state=16`、cls80／reg4／decode。這是 DATAFLOW 允許引用的設計意圖，不是新 artifact；未接觸舊程式或權重。[design/FINDINGS](design/FINDINGS.md)
2. **v1 尚未定位其他交付入口。** 讀取全部可見 branches／releases／all-state PRs／issues，只有 main，後三項為空。查找範圍限定此 repository 與此次觀察。[repository capture](repository/README.md)
3. **同名論文仍未對上本 head；已有具體取得渠道。** Elsevier API HTTP200 只取得 coredata，無全文；搜尋索引的 publisher introduction 預覽將 MambaFPN 稱為 neck。PubMed／作者頁定位通訊作者 Lefei Zhang；若 v1 owner 確認此論文為依據，可向其要求作者固定實作／公式／授權／native bundle。沒有對外聯絡。[upstream/FINDINGS](upstream/FINDINGS.md)
4. **新增 repository 候選已排除身分。** `raphaelkach/mamba-neck@6dec69675c362ef9837800ad084657850756e56b` 是獨立 bachelor project，ResNet50＋ATSS、四入五出 neck。這輪取得固定 README/tree 身分、size/hash，沒有把其四向 scan 當目標。arXiv title query=0 僅為有限查詢結果。[upstream capture](upstream/FINDINGS.md)
5. **MambaPSA 並非「沒有權重」，而是目標與 native 交付不符。** 完整 tree 有 13 個 `.pt`；本輪未載入／下載。YAML 改造 backbone／neck並接一般 Detect，與 D1 三層獨立 head 不同。已定位 wrapper 的 flatten／方向／merge，但內部委託 dependency，未建立本 head 對應。已見 CPU 分支 bypass 不能當 scan oracle；沒有判定作者的論文測量採用了此分支。[mambapsa/FINDINGS](mambapsa/FINDINGS.md)

具體交付為 [SCAN_CONTRACT](../../../docs/SCAN_CONTRACT.md) 與 [同一份 readiness 缺件表 §6](../../../docs/MODEL_ARTIFACT_READINESS.md#6-2026-10-29-追加來源查核與解除阻塞)。逐項區分目標未取得／是否存在未知、候選已存在但不等於目標、native 途徑缺失、授權缺文，以及需要 owner 裁定的設計欄位。沒有把「未取得」斷言為「尚未設計」。

出版 introduction／CRediT 的觀察來自搜尋索引預覽，另以同一查詢複核並保留 [選段／request／locator](upstream/publisher-index.json)。沒有取得全文；HTTP403／API coredata 檔案不是該預覽的重現證據，native curl replay 不能重現搜尋索引內容。此限制不影響「未建立目標對應」的結論。

## 驗證與限制

| 驗證 | 本輪結果與證據 | 適用範圍 |
| --- | --- | --- |
| 新 source archive CPU configure／build／CTest | 全部 exit 0；**8/8 PASS**，[cpu-ctest.log](local/cpu-ctest.log)，各步命令與 exit 同目錄。 | 既有 protocol／I/O／config／manifest regression；不是 scan／head 推論驗收。預設 build 未啟用 CUDA／TensorRT，不需要 protobuf 或 Python。 |
| Runtime／既有 evidence 無變更 | [no-runtime-change.log](local/no-runtime-change.log)、[historical-evidence-unchanged.log](local/historical-evidence-unchanged.log) 均空；原 P01-A [SHA256SUMS](../../p01-a/2026-10-28/SHA256SUMS) 全通過，見 [baseline-evidence-hashes](local/baseline-evidence-hashes.log)。 | 舊 evidence 的完整性，不是重做舊模型／GPU 驗收。 |
| 來源 captures | 每步 command／exit／輸出；HTTP status 與內容另判。MambaPSA 來源 hash 檢查通過；所有本輪保留檔案受根目錄 SHA256SUMS 約束。 | Curl exit0 不等於取得全文。publisher 403、PubMed 後續 203 challenge／locator exit1 有明記，初次成功 PubMed contact snapshot 保留；不把失敗當 pass。 |
| 文件連結／hash 完整性 | [verification](verification/) 保存本輪檢查命令、輸出與 exit；根目錄 `SHA256SUMS` 排除自身。 | 只檢查本輪 bundle 的本機檔案連結與 hashes，不驗證外部服務永遠可用。 |
| Scan reference／fixtures／ASan／UBSan | **未執行／未產生**：缺少目標精確契約，依停止條件不實作猜測語義。 | 沒有新 C++ 程式，不需要以一般工具 sanitizer 代替 scan 驗收。 |
| GPU／TensorRT／80 sessions | **本輪未重跑**：純文件與來源調查，未觸及共用 hash／設定／runtime。 | 原 P01-A 的 GPU／SDK 結論保持其原有限範圍。 |
| 遠端 CPU CI | **本輪無新遠端驗收結果**；本機 CPU PASS 與遠端 CI 分開。 | 不能以記憶中的舊 CI 結果替本輪交付背書。 |

Whitespace 檢查限編寫的 Markdown／重跑腳本／link checker；原始 HTTP headers、上游文字與命令／輸出 captures 保留其 bytes（含 CRLF／行末空白），未為格式檢查改寫來源。全量 `git diff --cached --check` 會因此回報原始 captures 的 whitespace；不將該項聲稱為通過。

## 重跑與下一步

```sh
bash evidence/scan-contract/2026-10-29/capture-local.sh build-scan-local-replay
bash evidence/scan-contract/2026-10-29/design/capture.sh build-scan-design-replay
sh evidence/scan-contract/2026-10-29/upstream/replay.sh build-scan-upstream-replay
sh evidence/scan-contract/2026-10-29/mambapsa/replay.sh build-scan-mambapsa-replay
sha256sum -c evidence/scan-contract/2026-10-29/SHA256SUMS
```

需 native C++20 compiler、CMake、Git、curl、jq、Node、Bash／coreutils；design capture 另需引用的歷史 Git objects，不修改該 repository。各重跑使用新 output，不能覆蓋已封存 captures；浮動 metadata／遠端分支可能改變，須當新觀察。巨大 publisher denial body 未提交，HTTP status/size/hash 保留於 upstream capture；它不是取得的論文。

**最直接的下一步：v1 模型負責人提供固定的新 head source/config/module map；若沒有，先裁定 SCAN_CONTRACT 的 D01–D04。** 然後核對 recurrence、tensor／state ABI 與數值政策，才進行 S01 standalone scalar／獨立 expected／CPU 正反測試。新模型權重與 native artifact、ordered cls/reg/decode 另按 P01／S04 驗收，不阻塞已固定契約的單獨 scan 實作。
