# 可重跑的協定模型

`tools/protocol_model.cpp` 是從零建立的 C++20 CPU 工具，不需要 CUDA、GPU 或 Python。它驗證 [ASYNC_CONTROL](ASYNC_CONTROL.md) 的 H／D／M 提交順序；不是 GPU kernel 的模擬器。

```sh
cmake -S . -B build
cmake --build build
./build/protocol_model --self-test
ctest --test-dir build --output-on-failure
```

也可獨立編譯：

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror tools/protocol_model.cpp -o /tmp/protocol_model
/tmp/protocol_model --self-test
```

`protocol_model --mutate remove-mi-ready-wait` 可單獨產生缺邊反例，預期印出 `FAIL` 並退出 1；其他變異名稱由 `--self-test` 的輸出列明。未知參數退出 2。

模型建立 host enqueue 與 device 執行節點；同執行者順序、enqueue-before-execute、event record→wait、host 同步共同形成 happens-before 圖。初始化使用獨立執行者，避免把 legacy default stream 的隱式同步當成保證。wait 在 enqueue 當下綁定最新已 enqueue 的 record；未 record 的 event 不產生等待邊，並回報綁定錯誤。每個 event 另核對所代表的 frame，防止其他邊遮蔽錯綁。

MI／CS 的 payload 寫入、header 寫入，以及 consumer 的讀前／讀後 header 核對各自成為節點。cuFFT 寫 FFT_TMP，後續受 guard 的頻譜交付才寫工作頻譜。其餘同一 operation 的 read-modify-write 視為先讀後寫；kernel 內部執行正確性另由實作測試負責。

本模型在 MI 的最後一次讀取後、頻譜交付前 record consumed；實際排程也可在頻譜交付後 record。交付只讀 M 專用 scratch，因此兩種位置都保護 MI 的最後讀取。本工具驗證存取依賴，沒有逐 kernel 重現所有控制段落；階段旗標與失敗提交由獨立語義模型驗證。

每組模型檢查：

- 圖無環。
- 每對相同資源且至少一方寫入的跨 operation 存取都有序，包括初始化與釋放。
- 每次讀取的預期 producer 已完成，且其他 writer 不會夾在 producer 與 reader 之間。
- event 綁定正確的 slot 與 frame generation。

基準涵蓋 N_MI、N_CS、N_H ∈ {1, 2, 3}，N_RP ∈ {1, 2, 3, 4}；各 10 幀，正常結束及最後一幀只提交 D 段的取消情境，共 216 組。所有基準組合均須通過；15 類變異各須至少有一個反例，否則工具退出失敗。工具列出每類變異的拒絕組數與第一個反例。

2026-10-28 本版本執行結果：216／216 基準通過，15／15 變異類被抓到。移除 mi_consumed 或 cs_consumed wait 各被 84／216 組抓到，移除 host staging wait 被 108／216 組抓到；其他組合存在間接同步遮蔽。移除終端等待只在 108 組 D-only tail 中失敗，因正常 drain 的 packet 已間接涵蓋 D／M 工作。這些是本工具的重跑結果，不是舊模型腳本的移植或舊執行紀錄的重現。

這是有限組合的協定檢查，不能據此宣稱所有 slot 數或任意 CUDA 實作均已證明正確。模型以單一 operation 表示計算階段，不驗證 kernel 內部 block 的 guard、原子操作、浮點數值、TensorRT／cuFFT 的實際 stream 行為、裝置錯誤、tracker 或 MOT。失敗控制與 packet 編碼由獨立 CPU 測例驗證；合成資料 CUDA 執行器以真實 streams／events／cuFFT 補充 V03–V05，實機命令、結果及有限覆蓋範圍見 [CUDA_SYNTHETIC](CUDA_SYNTHETIC.md)。模型與 GPU 測試互相補充，均不涵蓋完整模型、GMC、tracker 或 MOT。
