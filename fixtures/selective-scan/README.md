# SelectiveScan／Cross-Scan 數值 fixtures

供 `tests/selective_scan_tests.cpp`（CTest `selective_scan_reference`）使用。契約、來源與容差依據見 [SCAN_CONTRACT](../../docs/SCAN_CONTRACT.md)。

**Expected 的來源分三類，均不由 reference 自身產生：**

| 類別 | 來源與產生方式 | 適用範圍 |
| --- | --- | --- |
| H：獨立推導 | `derive.sh` 以 GNU bc 任意精度（`scale=80`，極端案例 200）計算 closed form 或明確雙重求和，不使用 recurrence；結果截至 40 位小數寫入 `expected.tsv`（`value × 2^exp2`）。部分 closed form 先手算，記在下表。 | 合法輸入的契約數值；一般工具 fixture，**不是**真實模型輸出 |
| E：契約等式 | 兩次獨立 reference 呼叫，契約規定結果必須相等（值相等，`-0 == +0`）。 | broadcast、共用 A、batch 隔離、每次呼叫重設、缺 D、零長度 |
| R：拒絕 | 預期的 exception 與診斷子字串。 | 非法 shape、容量、非有限輸入、超出 FP32 範圍 |

重產：`bash fixtures/selective-scan/derive.sh > expected.tsv`（約 30 秒）；evidence 會把重產結果與提交版逐 byte 比較。

## 共用記號

形狀依 `(batch, length, channels, state, a_rows, c_width, has_d)`。`delta=0` 時 Δ = softplus(0) = ln 2；A = −k 時每步 decay 為 2^−k。所有輸入都是二進位有限小數或 2 的冪，在 FP32 中精確。生成式 `g(x, m, o, s) = (x mod m − o) / s`。

## H 類案例

| 案例 | 形狀與輸入 | Expected 推導 |
| --- | --- | --- |
| `single_token` | (1,1,2,2,1,2,D)；u=[2,−4]，delta=[0,21]，A=[−1,−2]，B=[0.5,0.75]，C=[1,2]，D=[0.5,−1] | 手算：ch0 Δ=ln2，h=ln2·[1,1.5]，y=1+4 ln2；ch1 Δ=21（>20 passthrough），h=[−42,−63]，y=−164 |
| `three_token_decay` | (1,3,1,1,1,1,—)；u=[1,2,3]，delta=0，A=−1，B=C=1 | 手算 decay ½：y=ln2·[1, 2.5, 4.25] |
| `softplus_branches` | (1,1,6,1,1,1,—)；delta=[−30,−20,−1,0,20,20.5]，u=B=C=1 | y=softplus(delta)；20 走 log1p 分支，20.5 原值；相對容差 |
| `geometric_p3_length` | (1,400,4,16,1,1,D)；A_n=−(n+1)，delta=0，B=C=1，u_d=(d+1)/4，D=0.5 | y=u_d(ln2·G_t+½)，G_t=Σ_n (1−2^−(n+1)(t+1))/(1−2^−(n+1)) |
| `per_channel_a` | (1,25,3,2,3,2,D)；A=[[−1,−2],[−0.5,−4],[0,−1]]，B=[1,−0.5]，C=[2,1]，u=[1,−2,0.5]，D=[0,0.25,−1]，delta=0 | y=D_d u_d + Σ_n C_n ln2 B_n u_d S(2^A_dn,t)，S(a,t)=(1−a^(t+1))/(1−a)，a=1 時 t+1 |
| `state_32_p4_length` | (1,100,2,32,1,1,—)；A_n=−(n+1)/8，delta=0，B=1，C=0.5，u=[1,−1] | y=0.5 u_d ln2 Σ_n S(2^A_n,t) |
| `varying_direct_sum` | (2,7,3,4,1,4,D)；delta=g(7b+3t+5d,13,6,2)，u=g(5b+7t+3d,11,5,4)，B=g(3b+5t+7n,9,4,4)，C=g(7b+3t+5n,7,3,2)，A=[−0.5,−1,−2,−3]，D=(d−1)/2 | 不經 recurrence 的明確和：y=Σ_n C_tn Σ_{s≤t} exp(A_n Σ_{s<r≤t} Δ_r) Δ_s B_sn u_s + D u_t |
| `m_head_abi_L25/100/400` | m 的三個 scan 節點：(4,L,256,16,1,1,D)，L=25/100/400；A_n=−(n+1)，delta=0，B_n=(n mod 4+1)/4，C_b=(b+1)/2，u=g(37b+11d,17,8,8)，D=g(7d,5,2,4) | bc 產生 `m_head_k` 表 K_t=ln2 Σ_n B_n(1−2^−(n+1)(t+1))/(1−2^−(n+1))；測試端以 long double 計算 C_b u K_t + D_d u（輸入皆為精確二進位小數） |
| `extreme_large_delta_memoryless` | (1,3,1,1,1,1,—)；delta=2^100，A=−1，B=2^−100，u=[3,−5,7]，C=1 | decay=exp(−2^100)=0：y=u |
| `extreme_negative_delta` | (1,2,1,1,1,1,D)；delta=−2^100，u=[2^100,−2^100]，D=2^−100 | Δ·u < 10^−(10^29)，只剩 D·u=[1,−1] |
| `extreme_fp32_max_input` | (1,1,1,1,…,—)；u=2^127，delta=−100，B=C=1 | y=2^127·ln(1+e^−100)，bc `scale=200` |
| `extreme_near_fp32_limit_output` | 同上；delta=21，B=2^−5 | y=21·2^122≈1.117e38，仍在 FP32 範圍 |
| `extreme_subnormal_input` | (1,2,1,1,…,—)；u=2^−149（FP32 最小 subnormal），delta=0，A=−1 | y=ln2·[1,1.5]·2^−149 |
| `extreme_most_negative_a` | (1,3,1,1,…,—)；A=−FLT_MAX，delta=0，u=[1,2,3] | decay=0：y=ln2·u |
| `cross_scan_prefix_merge` | x=[[1,2,3],[4,5,6]]（C=1）；四方向展開後以 A=0、delta=0、B=C=1、無 D 的 scan 作 prefix sum，再合併 | 手算：方向 0 前綴 1,3,6,10,15,21；方向 1（翻 H+W）位置 (0,0)…(1,2) 為 21,20,18,15,11,6；方向 2（翻 W）6,5,3,21,17,12；方向 3（翻 H）16,18,21,4,9,15。各位置和 44,46,48,50,52,54，除 4 後乘 ln2：y=ln2·[11,11.5,12,12.5,13,13.5] |

`unit` 容差用於 |y| 與項量級 ≤ 約 10² 的案例；`relative` 用於量級極端、無相消的案例。數值在 [SCAN_CONTRACT §5](../../docs/SCAN_CONTRACT.md#5-數值比較政策) 固定。

## E、R 與 Cross-Scan 索引

E 類使用 `varied()` 的固定生成式（不是亂數）。R 類覆蓋 N=0／3／64、A rows≠1 或 channels、C width≠1 或 N、batch 或 channels 為 0、六個 tensor 的數量錯誤、缺 D 卻提供 D、三種容量溢位（含 size_t 等級）、六個 tensor 的 NaN／±Inf，以及 decay、drive、state、C·state、D·u、output 超出 FP32 有限範圍。

Cross-Scan 索引案例以 x[c,h,w]=100c+10h+w（2×2×3）手列四方向 token 順序，另驗 H=1／W=1 的退化順序、batch=2 的 direction-major rows（列 = 方向×batch+影像）、merge 對 expand 的精確還原，以及零維度、數量、非有限值、方向／token 超界與容量拒絕。
