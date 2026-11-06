# HR sanitizer 發布前核對（2026-11-06）

本輪整理並提交 HR 測試入口、sanitizer 驗證 runner、三輪報告與原始證據。部署 runtime／kernel／engine 維持原版；受測 CUDA runtime 仍由各報告固定在乾淨 `a00ab3a`，不以本次文件與工具提交重發原正式驗收。

| 檢查 | 本次結果 |
| --- | --- |
| 新建 Release CPU build／CTest | 20/20，52.79 秒；configure／build／ctest exit 0 |
| 隔離 staged 快照的新建 Release build／CTest | 20/20，49.50 秒；排除並行外部路徑變更 |
| Python runner／custom／lifecycle／archive／observer | 175／131／175／48／33 通過，失敗皆 0；只在 build scratch 副本執行 |
| 三輪原始證據 | 62＋130＋58＝250 個檔案的 SHA-256 與 sizes 全部符合各自 frozen manifest |
| 待提交 test harness | 與已執行的封存 source 逐 byte 相同 |
| 待提交 sanitizer runner | 與最新受測 captured source 逐 byte 相同 |

Archive scratch 的 48 項比原始紀錄 46 多兩項，因 scratch 同時含本地 runner 副本及 repo runner，增加編譯／來源不變斷言；原始 46 項收據與全部 frozen evidence bytes 保留。

本機本輪執行的是 CPU build／CTest 與 Python 分類、資源及封存規則檢查。HR／CUDA 執行結果沿用以下獨立報告與完整 hashes，不將 CPU CTest 或遠端 CPU CI 算成新的 CUDA／正式驗收：

- [HR 完整路徑補驗](../2026-11-05-a00ab3a/REPORT.md)：完整 synccheck、NVOF memcheck OPEN、nvJPEG initcheck 診斷及原始 OOM／timeout。
- [RAM 來源追蹤](../../hr-sanitizer-memory/2026-11-05/REPORT.md)：collection worker 配置點、block 成本與同步對照；沒有認定任意長跑 leak-free。
- [Racecheck 可執行診斷入口](../2026-11-06-bounded-queue/REPORT.md)：未過濾 ordinary serial 30 幀為 `PASS_FORCED_SYNC`；自訂＋CUB 四種 30 幀排程及 lifecycle 180 checks 為 `PASS_FILTERED_FORCED_SYNC`。**原生／未過濾 Graph gate 仍 OPEN**，Graph 同步化重試仍 24 GiB OOM。

[CHECKS.json](CHECKS.json)記錄本輪來源 hashes 與驗證範圍；[CPU CTest 原始紀錄](ctest.log)、[命令與 exits](build-results.json)、[Python 檢查總表](cpu-checks/PUBLICATION_CPU_CHECKS.json)與各 stdout／stderr 均保留。根目錄三輪 frozen manifests 未重寫；本發布前收據獨立存放。提交 SHA、push 與對應遠端 CI 狀態須由該 commit 的 Git／GitHub 記錄另行核對。

[隔離快照核對](ISOLATED_VALIDATION.json)及其 [CTest log](isolated-ctest.log)另外記錄實際待提交來源的檢查。平行外部路徑整理檔案保留在原工作樹，不併入此套件。

[WHITESPACE.json](WHITESPACE.json)明列全檔檢查對五個 frozen SASS／CSV／CMake 原始紀錄的 1271 則 whitespace 診斷；程式與目前文件檢查通過。保留這些原始 bytes，不藉由格式化改寫既有 artifact hashes。
