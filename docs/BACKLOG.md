# Saccade v1 後續工作

日期：2026-10-31。T01 實驗與有界 F01／F02 使用結案見 [closeout](history/2026-10-31-t01-closeout.md)。[F04 已獨立結案（受測範圍）](../evidence/f04/2026-10-31-formal-995bca5/REPORT.md)；[F03 producer 聚合器](MODEL_PRODUCER.md) 已完成乾淨 `a9c3d9b` 的 [正式驗收（受測範圍）](../evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md)；[F01–F04 detection session](DETECTION_SESSION.md) 已實作，[開發來源驗證 PASS（受測範圍）](../evidence/detection-session/2026-10-31/REPORT.md)。完整 tracking session 與 P01 保留 OPEN。

## 1. F04：先固定觀測政策，再實作

狀態：**clean-source 正式驗收 PASS（受測範圍），獨立結案**；受測 source 為乾淨 `995bca5`。

輸入是 F03 的六個 raw cls／ltrb outputs 與 `ImageMapping`；輸出為保留穩定 candidate ID、原始 score／class、原圖 xyxy、kept／suppressed、device count／capacity／status 的 `DetectionCandidates`。依 [DATAFLOW](DATAFLOW.md#4-跨階段資料契約)，suppressed 候選須留給 F04b，suppression 只執行一次。

已依版本化 [`f04-postprocess-fp32/1`](POSTPROCESS_POLICY.md) 實作 [CPU scalar reference 與 CUDA API](POSTPROCESS.md)，沿用 T01 已固定的語義：P3→P5 row-major、stride 8／16／32、anchor `(x+0.5,y+0.5)`、raw ltrb、80 類 sigmoid 後 max／argmax、conf≥0.001、**class 篩選前 top-300**、只留 class 0、greedy NMS IoU>0.5，再映射原圖。保留完整 kept／suppressed pool，不能將此契約縮成 person-only top-k 或只交付 kept boxes。

政策已固定 class／top-k／NMS ties、FP32 計算與閾值邊界、不 clipping、退化／非有限框、capacity／padding／overflow，以及被 NMS suppressed 與低分／非 person／top-k 淘汰的區別。T01 evaluator 的 invalid-topk 只有歷史統計；新增 native 處置不回寫其實驗結果。候選 ID 由篩選前模型序號建立並在 compact 後保留。

完成條件：依 GPU 前固定的 reference／cases／政策，驗證完整輸出與穩定順序、重跑與資源重用、同分與閾值邊界、空候選及明確失敗；保留 F04b 可用的 suppressed pool。正常 runtime 的中途 count／status 留 device，不增加 per-frame postprocess D2H。

目前實作為 **decode 融合壓實＋device 分級 Top-300**：M 是 class／退化篩選前的合格位置數，M≤100 用 compact shared-memory rank，101–512 用單 block radix，M>512 用 tiles 與全域合併；唯一 key 保留原始 ID 決勝。[正式驗收](../evidence/f04/2026-10-31-formal-995bca5/REPORT.md) 的 49 cases、10-phase 重用、完整輸出 bytes、repeat／graph、六份 replay 與 F04 CTest 4/4 通過，正常 suite 共 34,915 checks。四種 sanitizer 各 34,610 checks、0 errors／hazards，四個正控制及 13/13 語義變體均以預期診斷與退出碼偵測；sanitizer callback gates 為 UNEXECUTED，正常 gates 已驗。

先前 [adaptive dirty development 報告](../evidence/f04/2026-10-31-adaptive/REPORT.md) 保留原身分與 attempt 紀錄；三份真實輸入的完整 F04 API 測時為 1.77–1.97× 加速，排除 H2D／D2H 與 CPU reference。正式驗收未重測效能，也未驗完整模型或 tracking 端到端速度。

[原始 rank 版本的 F04 證據](../evidence/f04/2026-10-31/REPORT.md) 保留 31 個固定 cases、正確解析與獨立 historical kept-count 核對的六份 F03 dumps、正常 pending gates、四種 sanitizer、正控制與 10 個 mutations 的歷史結果；當時 `source_dirty=true`、sanitizer callback gates 為 UNEXECUTED，clean-source 正式驗收亦未執行，不能將該 PASS 套用到本次 adaptive 改版。F03 plugin／finite statuses 的 [producer 聚合器](MODEL_PRODUCER.md) 已完成有界 engine→F04 乾淨 source 正式驗收；native 前處理／模型接續已另行實作於 [detection session](DETECTION_SESSION.md)，依其獨立證據判定。

## 2. F03 producer 與 bounded detection session

[`f03-producer/1`](MODEL_PRODUCER.md) 以三個同 stream kernels 完成全部模型 input／raw finite checks、九個 plugin status 的角色檢查、來源退休及 F04 aggregate 發布；**乾淨 `a9c3d9b` 正式驗收 PASS（受測範圍）**，見 [正式證據](../evidence/f03-producer/2026-10-31-formal-a9c3d9b/REPORT.md)。原 [開發證據](../evidence/f03-producer/2026-10-31/REPORT.md) 保留。Fresh slot 初始化、ready／consumed、identity 與 session 首次失敗屬 caller 責任，不由 primitive 的 PASS 推導。

[F01–F04 bounded async detection session](DETECTION_SESSION.md) 已實作：D 串接原 nvJPEG／F02、完整 F03、producer 與 F04；M 管理候選 consumer、device first-error latch 及固定容量 detection result。CS ready／consumed ring 防止尚未消費即覆寫，pinned HP ring 在 host receive 後重用；支援 frame／epoch identity、overflow／failure／後續 skip、EOF／cancel、D-only tail 的雙 stream drain、checked close 及新 instance 隔離。[開發來源驗證 PASS（受測範圍）](../evidence/detection-session/2026-10-31/REPORT.md)；不能記為 clean-source session 正式結案，也不納入 `a9c3d9b` 的既有 producer 驗收。

本版沿用單 JPEG staging，尚未實作雙 host staging overlap。正常 detection 資料面每幀交付一次完整固定容量 `DetectionSessionResult` D2H，沒有中間 count／status 讀回；這不是 tracking `ResultPacket`，M consumer 也不是 F04b。

完整 tracking session 仍須依 [ASYNC_CONTROL](ASYNC_CONTROL.md)、[DATAFLOW](DATAFLOW.md#6-gpu-所有權與同步依賴) 與 [ResultPacket](RESULT_PACKET.md)，實作並接通 F05、main 的 F06→F08a→F04b→F07／F08b 及最後 F09。新增 MotionImage／prior／tracker state 與 ResultPacket 時須擴充 owner／ready／last-use／consumed；候選須活到 F04b 最後消費，F04b 使用本幀 GMC＋predict prior，continuation-only 候選維持 `can_birth=false`。

完整 tracking 完成條件仍含真實影像 serial／queued、跨幀／跨 session 隔離、pending consumer 下的重用、frame／epoch／版本一致性、錯誤／overflow／缺幀／drain／checked close，以及 tracking state／MOT bytes 的獨立對照。既有前處理 usage 或新 bounded detection 驗證皆不能替代這些新增階段的證據。

## 3. P01：依原 readiness 缺件整體判定

以 [MODEL_ARTIFACT_READINESS](MODEL_ARTIFACT_READINESS.md)、[MODEL_MANIFEST](MODEL_MANIFEST.md) 與各階段已驗界線為準，F04／容量已有獨立正式證據；F03 producer 聚合器已有乾淨 `a9c3d9b` 正式驗收；正常 native 輸入至完整模型與 F04 已另行實作 bounded detection session，[開發來源驗證 PASS（受測範圍）](../evidence/detection-session/2026-10-31/REPORT.md)；其餘原列缺件仍依各自證據判定。指定 checkpoint／ONNX／bundle／engine 與實際 runtime 的身分須一致，未執行項目維持明列。

P01 保留 **OPEN**。Head／F03 racecheck、synccheck 目前 UNEXECUTED；模型品質、tracker／GMC／MOT writer 與完整 tracking 指標依各自驗收，不能由 detection AP 或有限數值對照推導。T01 不因本項未完成而重開已完成的訓練實驗。

## nvJPEG／NVJPG 診斷：目前不主動續查

原 initcheck FAIL、Bug 7021871、逆向／trace／annotation／poison 與所有失敗紀錄保留；依 [使用政策](PREPROCESS_USAGE_POLICY.md)，正常使用輸出等價才是目前驗收目標，不要求 NVIDIA 分類或消除既知報告。本診斷不作上述使用結案的前置工作。

只有新增輸出不符、重跑／重用漂移、API／同步失敗、新環境無法通過輸出驗證，或 NVIDIA 提供可行動的新資訊時，才按具體訊號建立新的診斷項。既有 annotation 不進正常 runtime，也不把被排除的檢查改稱原路徑 PASS。
