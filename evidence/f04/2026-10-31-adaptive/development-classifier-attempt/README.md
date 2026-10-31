# 保留的 mutation classifier attempt

這批 runtime／49-case／sanitizer capture 成功；其後 mutation runner 的前 12 項匹配，最後一項 dense-first512-only 也正常以 exit 1 被拒絕，但先觸發較早的 `deterministic-dense-random: score exceeds fixed bound`，不是 runner 原先預期的較晚高-ID case。

因此 mutation capture 為 **exit 1、complete=false**，不是完整 13/13 PASS。原始 logs、source／binary hashes、diffs 與未 restore 的錯誤 staged variant 紀錄保留於 [mutations](mutations/)，沒有拿這批資料取代最終控制。

修正僅涉及分類器的預期完整診斷；runtime、數值政策、reference、cases 與容差未改。最終結果另見 [adaptive report](../REPORT.md) 與新的完整 capture／13 controls。此資料夾的 source snapshot 保留當時的舊 classifier bytes，不宣稱與後續 helper 相同。
