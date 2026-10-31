# 第一次 clean-source custody attempt：未結案

此輪使用乾淨 `995bca5`，主驗收與 13 個語義變體通過；外部 custody wrapper 額外 touch／重建後，CUDA library、GPU suite 與 sanitizer control 的 hash 改變，故 [formal-status](custody/formal-status.json) 明列 **exit 1**。

原受測 bytes 在封存前被重建覆寫；此輪不能提供與測試時 hash 對應的 actual binaries，不作為最終 custody PASS。差異根因未判定；未依此更動 source、編譯 flags、cases、容差或 sanitizer 判定。原始 acceptance、mutations、重建 logs 與前後 hashes 保留原樣。

後續在另一 fresh clean checkout 重跑完整驗收，立即封存實際受測檔案，並以驗收前／後、封存內容及 controls 後 hashes 一致完成 custody。正式結果與邊界見 [結案報告](../REPORT.md)。不宣稱跨重建 binary 逐位元可重現。
