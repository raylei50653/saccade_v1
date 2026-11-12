# primary156 closeout（準備完成後才由 root 執行）

本工具不做推論、訓練、GPU 工作或重新計分。僅新建 closeout-tools 的方法與最後 evidence 封裝；不更動來源原始輸出或 frozen scripts。

必須先完成 capture 全468 cells、scoring 全7組、完整 inputs-after、final capture audit、independent scoring audit。live/bounded PASS 不接受。

先做唯讀 final gate 檢查：

```sh
python -I -B build/yolo26m-trt-primary156-20261111/closeout-tools/closeout.py
```

上述完整 PASS 後，root 才執行：

```sh
python -I -B build/yolo26m-trt-primary156-20261111/closeout-tools/closeout.py --execute > build/yolo26m-trt-primary156-20261111/closeout-driver.log 2>&1
```

預設 final authorities：preflight-tools/inputs-after.json、preflight-tools/final-capture-audit.json、validation-tools/scoring-audit.json。若 final output 命名不同，用 --inputs-after／--capture-audit／--scoring-audit 指定；final capture audit 必須綁定相同 inputs-after。

執行會先在 closeout-tools/staging-* 完整封裝，逐一驗證 1404 個 gzip 的解壓 SHA 與 bytes、來源 receipt、完整 SHA256SUMS 及檔案集合；四份 CSV 也須逐值符合 final JSON。完成後以一次同 filesystem 的 directory rename 發布到 evidence study 的新 bundle/ 子目錄，REPORT 與 manifest 一起出現，既有 capture authority 原位保留。任何錯誤會保留 staging，拒絕宣稱完整交付；失敗 attempt 也不會被新的 staging 遞迴打包。拒絕覆寫已發布 bundle。來源 mot/packets/motion 保留。不要在失敗後直接刪除 FAIL 或覆寫既有 evidence；應先定位錯誤並保留失敗證據。

完整結果：evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle/REPORT.md；打包自檢 receipt：closeout-tools/closeout-receipt.json（置於 bundle 外以避免 manifest 自我參照，不稱獨立封裝驗證）。本包不含 images、weights、ONNX、engines 或可執行檔。

發布後可獨立唯讀核對：

```sh
python -I -B build/yolo26m-trt-primary156-20261111/closeout-tools/closeout.py --verify-bundle evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle
```

數字全部讀取最終 summary/audit；報告保留 FP16 1% screen FAIL、非 blind、21 imported +447 fresh、own G1、非正式部署驗收、本輪不測速，以及七 primary groups 優先／五 dataset 與 pooled 僅描述性補充。
