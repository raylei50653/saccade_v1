# 保留的錯誤 replay attempt

這批紀錄的 F03 replay 讀取器將檔案誤解為 cls／reg 交錯的 level pairs。正式檔案實際為 `cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5`，見 [原始 F03 payload](../../../t01/2026-10-30/f03/capture/model-run-report.txt)。

CPU reference 與 CUDA 收到同一份誤解讀的資料，因而互相比較通過。**這些 F03 comparisons 與其 sanitizer 執行不構成預定真實 F03 replay 的驗證。** 原始 logs、hashes、snapshot、mutation results 與當時尚未發布的報告文字均保留；報告文字是錯誤草稿，不是有效結論。

最終有效結果另見 [F04 report](../REPORT.md)。修正只涉及測試讀取器，runtime kernels、CPU reference 與數值政策沒有改動；最終測試另以既有訓練側 diagnostic 的六個 kept counts 作獨立檢查，避免雙方同錯再次通過。
