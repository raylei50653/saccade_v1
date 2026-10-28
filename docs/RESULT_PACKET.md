# ResultPacket v1 與失敗狀態模型

日期：2026-10-28。這是第一個 C++ 協定實作 profile，配合 [ASYNC_CONTROL](ASYNC_CONTROL.md) 的控制規則；不代表 CUDA pipeline、實際 tracker 或 MOT writer 已實作。程式入口是 [protocol.hpp](../include/saccade/protocol.hpp)、[protocol.cpp](../src/protocol.cpp) 與 [protocol_tests.cpp](../tests/protocol_tests.cpp)。

## 型別與初始化

`FrameId = int64_t` 是處理序號，從 0 開始，`-1` 只作未提交／無失敗 sentinel；不代替來源 frame ID。`Epoch = uint64_t` 在一個 FailureModel session 內固定，重設需建立新模型。`begin(frame)` 的 frame 必須比先前提交工作的 frame 大；尚未鎖定失敗時，必須恰為 `committed_frame + 1`，否則鎖定 PROTOCOL。重複提交相同 frame 是 API 錯誤，不能重發 OK 或 FAILED。

| 欄位／資源 | 初始化與更新規則 |
| --- | --- |
| `Control.epoch` | constructor 的明確參數；session 內不變。 |
| `committed_frame`、`current_frame`、`fail_frame` | 初值 -1；current_frame 是當次 begin 的已提交工作身份（對應規格 working_frame），包含失敗後的幀。 |
| `committed_slot`、`working_slot` | 初值 1、0；begin 設 working_slot = 1 − committed_slot，只有成功 commit 同時發布 committed_frame 與 committed_slot。 |
| `frame_status`、`fail_code` | 初值 0；begin 只清 frame_status，絕不清 fail_code／fail_frame。frame_status 以 OR 累積，第一個失敗 commit 把完整位元集合鎖入 fail_code。 |
| `run_stage` | 初值 false；begin 與 commit 設 false；前序 prepare_stage 設成 `fail_code == 0 && frame_status == 0`。同一計算階段只能讀它，錯誤 OR 不改它。 |
| TS[2]、SPEC[2]、FFT scratch | 所有 payload words 初始化為 0、frame 為 -1；TS[1] 是首幀的空 TrackState。沒有前譜時回傳 null，不對 -1 作 modulo。 |
| readiness | 每幀清 spectrum／predict／update 完成記號；缺少必要產物不能成功 commit；重複 predict／update 記 PROTOCOL。 |
| 邏輯 ResultPacket | epoch=0，frame／committed_frame／fail_frame=-1，status=OK，錯誤=0，tracks 為空；預設物件需填有效身份才可編碼。 |
| TrackRow | IDs、class、flags、座標、scores 全為 0；有效 row 要求非零 UID／output ID 與合法 flags。 |

模型以 4 個 uint64 payload words 表示追蹤／頻譜內容，驗證的是提交與保護語義。FFT 每幀都執行並只改 FFT scratch；prepare_stage 之後的 publish 才能把 scratch 交付到 working SPEC。predict／update 只寫 working TS，成功 commit 才切換共同 slot。失敗後即使後續 FFT 工作繼續完成，也不碰最後成功的 TS／SPEC。

`commit()` 與 `packet()` 不受 run_stage 限制。已有 sticky failure 時，commit 不更動首次錯誤及版本。packet 使用 current_frame 回報身份：首次失敗幀 FAILED；後續幀 SKIPPED_AFTER_FAILURE，附原 fail_frame／fail_code。SKIPPED 的 frame_error 固定 0，後續診斷錯誤不冒充新的首次失敗。模型以 API 順序表示 stream 先後；實際 GPU 仍需以 atomic OR 累積錯誤並遵守控制 kernel phase boundary。

## 固定 wire profile

本 profile 容量是 **256 tracks**，header **80 bytes**，每 row **48 bytes**，每幀 **12,368 bytes**。這是第一批協定工具的明確容量，不表示偵測、tracker、top-k、影像／頻譜等 P04 上限全部已定案。需要不同容量應新增明確版本／profile，不能靜默改 v1。

所有多位元欄位採 little endian，signed frame 使用 64-bit 二補數。float 採 IEEE-754 binary32，只接受有限值，`-0` 編碼成 `+0`。不透過 `sizeof`、C++ struct padding 或 raw struct memcpy 定義 ABI；GPU packet writer 也必須逐欄遵守相同布局。

| Header offset | bytes | 欄位 |
| --- | --- | --- |
| 0 | 4 | ASCII `SCRP` |
| 4 | 2 | version = 1 |
| 6 | 2 | header bytes = 80 |
| 8 | 4 | total bytes = 12368 |
| 12 | 4 | capacity = 256 |
| 16 | 8 | epoch |
| 24 | 8 | 本次工作 frame |
| 32 | 8 | committed_frame |
| 40 | 8 | fail_frame，無失敗為 -1 |
| 48 | 4 | ResultStatus：OK=0、FAILED=1、SKIPPED_AFTER_FAILURE=2 |
| 52 | 4 | count，0..256 |
| 56 | 4 | frame_error 位元 |
| 60 | 4 | sticky fail_code 位元 |
| 64 | 16 | reserved，固定全 0 |

Error 位元固定為 PROTOCOL=0x01、PRODUCER=0x02、OVERFLOW=0x04、NUMERIC=0x08、COMPUTE=0x10；0 表示無錯，未知位元拒絕。OVERFLOW 是錯誤集合的一部分，不另外用 count 截斷掩蓋。

第 i 個 row 位於 `80 + 48*i`。前 count 筆保留 producer 的穩定順序；本層不排序、不量化浮點數。

| Row offset | bytes | 欄位 |
| --- | --- | --- |
| 0 | 8 | 不重用的 track UID |
| 8 | 8 | tracker 交付的內部 output identity；不是 F09 首次出現映射後的 MOT ID |
| 16 | 4 | class ID |
| 20 | 4 | flags：observed=1 或 predicted=2，恰為其中之一 |
| 24、28、32、36 | 各 4 | 原圖像素 xyxy |
| 40 | 4 | observed_score：原始觀測機率，範圍 [0,1] |
| 44 | 4 | association_score：續追／關聯使用的有限分數；範圍與產生政策屬 P03 |

predicted row 的兩個 observation scores 固定 0。其他 tracker 分數若需輸出，須先增加明確定義，不能覆用這兩個欄位。具體 threshold、clipping、退化框與 MOT 輸出政策仍由對應模組決定；這裡只驗證有限座標與穩定編碼。

每次編碼先把 **整個封包** 清零；reserved 與未用 rows 全為 0。OK 必須 committed_frame=frame、錯誤=0、fail_frame=-1。FAILED 必須 fail_frame=frame、committed_frame<frame、frame_error=fail_code≠0。SKIPPED 必須 committed_frame<fail_frame<frame、fail_code≠0、frame_error=0。FAILED／SKIPPED 的 count=0，所有 rows=0；只有 OK 的 count=0 才表示合法空結果。

decode_packet 嚴格核對 profile、身份／狀態、數量、數值，以及重新編碼後的全部 bytes，拒絕非零 reserved／未用 rows、負零與其他非 canonical 編碼。

## 重跑與完整比較

從 repo root 執行：

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
build/packet_compare serial.bin pipelined.bin
```

packet_compare 接受一個或多個完整 v1 封包串接而成的檔案，逐位元比較 **所有 bytes**，包括 reserved 與未用 rows。它刻意不先 decode，讓 reserved／padding 污染也以差異回報；canonical 合法性由 decode_packet 另外檢查。檔案為空或不是 12,368 的整數倍會拒絕；完整封包數不同則是比較失敗。

exit code：0 全部相同；1 內容或完整封包數不同；2 用法、讀檔、空檔或截斷錯誤。差異回報含檔案 byte offset、packet 序號及 packet 內 offset。

protocol_tests 涵蓋首幀／兩種 parity、失敗各階段、本幀不提交、首次錯誤不可覆寫、後續多幀 FFT scratch 寫入及 SKIPPED、同 grid 凍結旗標、缺少產物、重複 predict、錯誤序號、golden endian／float bytes、完整 roundtrip、初始化、容量／非有限值及全 byte 比較。`protocol_tests --fixtures DIR` 可建立同封包、reserved／未用 byte 差異、空檔與截斷 fixture，供 CLI 測試。這些測試不取代 GPU 時序擾動、sanitizer、實際 serial／pipeline 或 MOT 的逐位元比較。
