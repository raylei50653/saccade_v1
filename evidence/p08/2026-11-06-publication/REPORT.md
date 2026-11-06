# P08 發布前核對

2026-11-06。本提交交付 P08 設定／CPU 事件工具、CLI、測試、導覽文件與對應的
[開發證據](../2026-11-06-development/REPORT.md)。本機 `main` 的前置提交為
`dacb91a`（外部路徑整理）及 `7dfd513`（正式／研究依賴分離）；正常 fast-forward
push 會包含這兩個既有祖先，不另外改寫其歷史。

[checks.json](checks.json) 記錄發布前實際核對：14 份 frozen artifacts、227 個
來源檔、9 個 binaries 的大小與 SHA-256 全部相符。開發證據約 147 KB，沒有新增
大型 payload；完整 runtime outputs 仍依原報告留在本機 build 目錄。

原始 development REPORT、logs、收據與 manifest 維持原 bytes。其「沒有 commit／push」
描述受測當時的狀態，不表示本次交付禁止發布；原 binaries 的 dirty build 身分不因
本次提交變成 clean-source acceptance。

本輪只核對發布 custody，沒有重跑或擴大原測試：CPU 21/21、native CLI 18/18、
既有串流 CLI 35/35 與固定 records 的 5,316 幀輸出等價均依原開發收據判讀。
**P08 未整體結案**，GPU 完整中間事件及觀察開關一致性、既有完整 session memcheck／
原生 HR racecheck 仍 OPEN。

遠端 commit SHA 與同 SHA 的 CI 判定在 push 後另外核對；此發布前收據不預先宣稱
push 成功或 CI PASS。
