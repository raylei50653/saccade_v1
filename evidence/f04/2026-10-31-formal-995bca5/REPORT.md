# F04 clean-source 正式驗收與獨立結案

日期：2026-10-31。**F04 正式驗收 PASS（受測範圍），獨立結案。** 本次從乾淨 detached checkout `995bca50d41510611b3990c9f8a020a9d8b76074` 重新建置並實跑，不沿用 [adaptive 開發驗證](../2026-10-31-adaptive/REPORT.md) 的 dirty-source PASS。驗收範圍為六個 raw predictions＋明確 producer status＋`ImageMapping` → 完整 `DetectionCandidates`，依 [`f04-postprocess-fp32/1`](../../../docs/POSTPROCESS_POLICY.md)；完整 async session、F03 producer 聚合器與 P01 仍 OPEN。

## 固定身分與證據保管

| 項目 | 本次固定值 |
| --- | --- |
| Source | `995bca50d41510611b3990c9f8a020a9d8b76074`；`source_dirty=false`、`allow_dirty=false` |
| Build-input manifest digest | `0c787d7d94adeb15525c0c68a3621fb7681814cf53eca1af772464d172697dc8` |
| GPU suite SHA-256 | `259550c8a3f9c1f0a1310e2648feb505a1bd9ed0e912b452979e2c642ad9c4db` |
| CUDA library SHA-256 | `676238c880191f77e65423c2bfe62b9af927385a75b87fa3d1567988ea83955d` |
| Sanitizer executable SHA-256 | `fc3b9d2abad07999a56631a6be69c320461ebdb80a5ab0044775c44d301c9488` |
| Compiler／CUDA／CUB | GCC 16.2.1、nvcc 13.4.92、runtime／driver API 13.4、CUB／CCCL 3.5.0 |
| Compute Sanitizer | 2026.3.0.0（build 38855100） |
| GPU／driver | RTX 5070 Ti Laptop、sm_120、driver 617.42、WSL2 |
| CUDA flags | `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false -lineinfo` |
| ABI／workspace | 9,664-byte output；295,712-byte scratch、8-byte alignment |

GPU 執行前已保存 [政策／reference／cases hashes](capture/frozen-policy-reference-cases.sha256)、[逐檔 build inputs](capture/source-files.sha256)、[binary hashes](capture/binaries.log) 與 fixture 不變量；本次不改政策、reference、cases、容差或 production 程式。完整 tracked source、HEAD／tree／乾淨狀態在 [before](custody/source-before.json)／[after](custody/source-after.json) 相同，各步驟前後亦檢查 source。原有 evidence 檔逐份核對未改；數量與總結見 [summary](custody/summary.json)。

測試 executable 與 sanitizer control 的最終 bytes 與 GPU 前 hash 相同；七個 executable／static libraries 在 mutations 前即封存實際 bytes，controls 完成後仍相同，見 [驗收後](custody/binary-library-hashes-after-acceptance.json)／[最終](custody/binary-library-hashes-final.json) 與 [即時 binary 封存](custody/accepted-binary-bundle.json)。不宣稱 CUDA 重建產物逐位元可重現。[工具 executable hashes](custody/tool-executable-hashes.json) 在 acceptance 前固定並於各外層階段核對；額外 sanitizer installation hashes 在 acceptance 期間保存，結束後再次核對。建置／工具／環境細節見 [CMakeCache](custody/CMakeCache.txt)、[compile commands](capture/compile_commands.json) 與 [environment](capture/environment.log)。

[受測 build inputs archive](capture/source-build-inputs.tar.gz)、原始 logs／命令／退出碼及 [外部 custody wrapper](custody/run_formal.py) 隨本報告保存。完整 Git source archive 與 actual binaries／libraries／六份 raw dumps bundle 留在本機 `build-f04-formal-995bca5-final/`；路徑、大小、SHA-256 見 summary 的 `local_artifacts`。乾淨受測 checkout 留在其 `source/`。整次執行置於 MemoryMax=12 GiB、MemorySwapMax=0 的獨立 scope，資源樣本見 [resource samples](custody/resource-samples.jsonl)。

## 正式結果

| 檢查 | 本次結果 |
| --- | --- |
| CPU reference | 442 checks PASS |
| GPU 前 fixture 不變量 | 49 cases、2,805 checks PASS |
| F04 CTest | **4/4 PASS**：reference、CUDA、lazy streams、F03 replay；[CTest](capture/ctest.log)／[完整輸出](capture/LastTest.log) |
| 正常 suite＋六份 replay | **34,915 checks PASS**；[log](capture/full-replay.log) |
| Adaptive／重用 | M=0／1／99／100／101／299／300／301／511／512／513／8,400、cross-tile／level、dense 高 ID winners、10-phase dense／sparse／empty／failure 重用 PASS |
| 所有權／重跑 | pending producer／consumer、held enqueue、獨立 instances、repeat／graph 及完整 bytes PASS；normal 與 lazy 的 callback gates 均實跑 |
| Memcheck | **34,610 checks、0 errors**；[log](capture/sanitizer-memcheck.log) |
| Initcheck | **34,610 checks、0 errors**；[log](capture/sanitizer-initcheck.log) |
| Racecheck | **34,610 checks、0 hazards（0 errors、0 warnings）**；[log](capture/sanitizer-racecheck.log) |
| Synccheck | **34,610 checks、0 errors**；[log](capture/sanitizer-synccheck.log) |
| Sanitizer 正控制 | **4/4 DETECTED**，各有預期 planted diagnostic 與 exit 99 |
| 語義變體 | **13/13 DETECTED**，各有完整預期 diagnostic 與 exit 1；[results](mutations/results.tsv) |
| 還原 baseline | production CUDA bytes 還原、重建後 **34,610 checks PASS**；[log](mutations/restore-baseline.stdout) |
| Source／binary／工具／replay custody | **PASS**；[正式狀態](custody/formal-status.json) |

四種 sanitizer 均含全部 49 cases、重用與六份 replay，使用 `--no-held-stream`；**sanitizer 下的 host callback 延遲 gates 為 UNEXECUTED**，不將正常／lazy 的 gates PASS 擴張為該覆蓋。完整 header／rows、metadata、candidate ID、class、kept／suppressed、順序、counts、座標 bits、未用 rows／padding 均依 exact 規則；score 沿用預先固定的 `4*FLT_EPSILON` 界，最大差 `1.19209e-7`。同 binary／input 的重跑與 graph 檢查完整 output bytes。

13 個 mutations 只改 disposable copied CUDA source，正式 checkout 與 binary 不變。它們覆蓋 IoU 邊界、greedy chain、sigmoid 後 argmax、ID tie、stride／ltrb／mapping、suppressed pool、upstream flags、row reset、person-only top-k、dense 末 tile 及 first-512 截斷。編譯失敗、crash、timeout 或其他 diagnostic 不算 DETECTED。複製樹可能繼承外層 Git metadata；**mutation 的真實身分由各 variant 的 source hash／完整 diff／binary hash 判定**，不能以其輸出的 `source_dirty=false` 稱 mutant 為正式 baseline。

## 六份 replay 與 F03 來源

本次只重播既有 F03 raw outputs，未重跑 TensorRT。[來源紀錄](custody/replay-inputs.json)連結乾淨 F03 source `4b52b2fbbf54344a1c7dcb7a68812fb2a9e63bd9`、engine `c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5`、[歷史 model run](custody/f03-model-run.log) 與 [binary hashes](custody/f03-binary-hashes.log)。原 engine 的現存 bytes 另作 [hash 核對](custody/live-original-engine.json)。

六份資料各 2,822,400 bytes，在第一次 GPU 工作前複製並核對歷史 SHA-256；順序為 **cls_p3、cls_p4、cls_p5、reg_p3、reg_p4、reg_p5**。CPU／CUDA 對照之外，獨立歷史 kept counts 仍要求 32／28／39／2／3／1。正常、四種 sanitizer 及 restored baseline 的六份完整 output hashes 全部相同。

| 固定 input | 完整 rows | Kept | Suppressed |
| --- | ---: | ---: | ---: |
| MOT17-02-SDP/1 | 237 | 32 | 205 |
| MOT17-10-SDP/1 | 246 | 28 | 218 |
| MOT17-13-SDP/400 | 285 | 39 | 246 |
| zeros | 7 | 2 | 5 |
| ones | 5 | 3 | 2 |
| random/1 | 3 | 1 | 2 |

## 保留的第一次 custody attempt

第一輪同樣使用乾淨 `995bca5`，主驗收與 13 個 mutations 均通過，但外部 wrapper 額外 touch／重建後，CUDA library、GPU suite 與 sanitizer control 的 hash 改變，wrapper 因此 **exit 1**。該輪在重建前未封存 actual binaries，原受測 bytes 被覆寫；不將重建後的檔案冒充原受測 binary。差異根因未判定，也不宣稱 source 被更改或 runtime 比較失敗。

[該輪完整紀錄](custody-attempt-rebuild/README.md)獨立保留，沒有改寫其失敗狀態。本次另建 fresh clean checkout，全套 acceptance／sanitizers／controls／mutations 重跑，於 acceptance 完成後立即封存 actual binaries，以 GPU 前 hashes、封存內容與 controls 後 hashes 一致完成 custody；沒有改 source、編譯 flags、測例、容差或 sanitizer 判定。

## 結案邊界與重跑

F04 的 raw predictions→完整 `DetectionCandidates`、獨立 API 的正常 stream／重用／失敗契約在本機受測 GPU／SDK 與固定 inputs 範圍正式結案。完整 async session、全部 9 個 F03 plugin statuses＋input／output finite checks 的 producer 聚合器、native F01／F02→F03→F04 接續、CS ready／consumed ring、session first-error latch／drain／checked close、F04b／GMC／tracker／ResultPacket／MOT writer 仍待各自整合驗收；**P01 OPEN**，模型品質不由本項推導。

本次沒有重新驗收效能；先前 1.77–1.97× 的三份真實輸入 F04 API 測時仍屬 [adaptive dirty-development 報告](../2026-10-31-adaptive/REPORT.md)，不升格為本次正式效能或端到端結果。既有 rank／adaptive、T01、nvJPEG 與 head／F03 sanitizer 歷史判定保留原狀。

重跑時從同 revision 建立新的乾淨 checkout，先取得六份 pinned dumps，指定其絕對路徑；out／mutation stage 均須是不存在的 root `build-*`：

```sh
SACCADE_F04_F03_DUMPS=/absolute/path/to/pinned-dumps \
  bash scripts/run_postprocess_acceptance.sh build-f04-new
bash scripts/check_postprocess_mutations.sh \
  build-f04-new/cuda build-f04-new-mutations
```

逐步命令與退出碼以本目錄的 capture／mutations／custody 為準；整份保存證據的相對路徑 hashes 見 [SHA256SUMS](SHA256SUMS)。
