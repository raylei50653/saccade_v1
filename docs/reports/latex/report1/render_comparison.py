#!/usr/bin/env python3
"""Render report tables from saved metrics; never infer, rescore, or use a GPU."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QUALITY = ROOT / "evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle"
PERF = ROOT / "evidence/model-comparison/2026-11-11-yolo26m-trt-performance"
MARKDOWN = ROOT / "docs/reports/report1-3cbaa518f0dc.md"
START = "<!-- BEGIN GENERATED MODEL COMPARISON -->"
END = "<!-- END GENERATED MODEL COMPARISON -->"
DATASETS = ["MOT17", "MOT20", "MOT15", "DanceTrack", "PP22"]
GROUPS = ["MOT17/all", "MOT20/all", "MOT15/moving", "MOT15/other",
          "DanceTrack/all", "PP22/moving", "PP22/static"]
ARMS = ["official", "finetuned", "main"]
NAMES = {"official": "官方 YOLO26m", "finetuned": "微調 YOLO26m", "main": "現行 HR Mamba"}
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]
SOURCES = [QUALITY / f for f in ["REPORT.md", "metrics/summary.json", "audits/scoring-audit.json",
           "training-provenance.json", "evaluation-scope.json", "methods/scoring-tools/selection-frozen.json",
           "metrics/groups/PP22__static/keyframe-filter.json", "metrics/groups/PP22__moving/keyframe-filter.json"]] + [PERF / f for f in [
           "REPORT.md", "performance/model/summary.json", "performance/pipeline/summary.json",
           "model-memory.json", "comparison.json", "validation.json"]]


def read(path):
    return json.loads(path.read_text())


def load():
    # Check each cited input against its existing evidence manifest.
    for directory in [QUALITY, PERF]:
        expected = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            expected[name.removeprefix("*").removeprefix("./")] = digest
        for path in SOURCES:
            if path.is_relative_to(directory):
                rel = path.relative_to(directory).as_posix()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[rel], rel
    quality = read(QUALITY / "metrics/summary.json")
    audit = read(QUALITY / "audits/scoring-audit.json")
    validation = read(PERF / "validation.json")
    assert quality["result"] == audit["result"] == "PASS"
    assert quality["sequence_count"] == 156 and quality["frames_per_arm"] == 135503
    assert len(quality["primary_groups"]) == 7 and len(quality["datasets"]) == 5
    assert audit["capture_validation_receipts_checked"] == 468
    assert validation["full_matrix_complete"] and not validation["formal_acceptance"]
    assert validation["performance_statistics_independent_audit"] == "PASS"
    assert validation["FP16_1percent_raw_relative_L2_gate"] == "FAIL_RETAINED"
    pipeline = read(PERF / "performance/pipeline/summary.json")["groups"]
    model = read(PERF / "performance/model/summary.json")["groups"]
    memory = read(PERF / "model-memory.json")
    timing = []
    for arm in ARMS:
        single, = [r for r in pipeline if r["arm"] == arm and r["path"] == "tracking"
                   and r["mode"] == "single" and r["sequence"] == "MOT17-10-SDP"]
        queued, = [r for r in pipeline if r["arm"] == arm and r["path"] == "tracking"
                   and r["mode"] == "queued" and r["sequence"] == "MOT17-10-SDP"]
        graph, = [r for r in model if r["arm"] == arm and r["comparison"] == "raw6"
                  and r["mode"] == "graph"]
        assert single["delivery_ms"]["samples"] == queued["delivery_ms"]["samples"] == 480
        timing.append([NAMES[arm], f'{graph["gpu_event_single_ms"]["mean"]:.3f}',
                       f'{single["delivery_ms"]["mean"]:.3f}', f'{single["delivery_ms"]["p95"]:.3f}',
                       f'{queued["throughput_fps"]:.3f}', f'{memory[arm]["engine_plan_bytes"] / 2**20:.3f}'])
    original = []
    for arm in ["official-default", "main"]:
        graph, = [r for r in model if r["arm"] == arm and r["comparison"] == "original-default" and r["mode"] == "graph"]
        original.append(["官方原始 one2one" if arm == "official-default" else NAMES[arm],
                         f'{graph["gpu_event_single_ms"]["mean"]:.3f}',
                         f'{graph["gpu_event_single_ms"]["p95"]:.3f}'])
    keyframes = sum(r["GT_present_original_frames"] for group in ["static", "moving"]
                   for r in read(QUALITY / f"metrics/groups/PP22__{group}/keyframe-filter.json")
                   if r["arm"] == "main")
    assert keyframes == 8753
    selection = read(QUALITY / "methods/scoring-tools/selection-frozen.json")
    assert sum(r["frames"] for r in selection["sequences"]) == 135503
    dance = [r for r in selection["sequences"] if r["benchmark"] == "DanceTrack"]
    assert sum(r["split"] == "train" for r in dance) == 40
    assert sum(r["split"] == "val" for r in dance) == 25
    return quality, audit, timing, original


def tables(quality, audit):
    datasets, groups, details = [], [], []
    for name in DATASETS:
        d = quality["datasets"][name]
        m = d["metrics"]
        datasets.append([name, str(d["sequence_count"]), f'{d["frames"]:,}'] +
                        [f'{m[a]["HOTA"]:.3f}' for a in ARMS] +
                        [f'{m["main"]["HOTA"] - m[a]["HOTA"]:+.3f}' for a in ARMS[:2]])
        for arm in ARMS:
            details.append([name, NAMES[arm]] + [str(m[arm][k]) if k == "IDSW"
                           else f'{m[arm][k]:.3f}' for k in METRICS])
    comparisons = {c["group"]: c for c in audit["context_comparisons"] if c["kind"] == "primary_group"}
    for name in GROUPS:
        g = quality["primary_groups"][name]
        m = g["metrics"]
        w = comparisons[name]["per_sequence_win_tie_loss"]["HOTA"]
        groups.append([name, str(g["sequence_count"])] + [f'{m[a]["HOTA"]:.3f}' for a in ARMS] +
                      [f'{m["main"]["HOTA"] - m["finetuned"]["HOTA"]:+.3f}',
                       f'{w["wins"]}/{w["ties"]}/{w["losses"]}'])
    return datasets, groups, details


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def tex_table(caption, spec, headers, rows):
    return "\n".join([r"\begin{table}[H]", r"\caption{" + caption + "}", r"\small",
                     r"\noindent\begin{tabularx}{\textwidth}{@{}" + spec + "@{}}", r"\toprule",
                     " & ".join(headers) + r" \\", r"\midrule"] +
                    [" & ".join(row) + r" \\" for row in rows] +
                    [r"\bottomrule", r"\end{tabularx}", r"\end{table}"])


def main():
    quality, audit, timing, original = load()
    datasets, groups, details = tables(quality, audit)
    pipeline = read(PERF / "performance/pipeline/summary.json")["groups"]
    percentile_rows = []
    short_names = {"official": "官方 YOLO", "finetuned": "微調 YOLO", "main": "HR Mamba"}
    for sequence in ["MOT17-10-SDP", "MOT17-02-SDP"]:
        for arm in ARMS:
            single, = [r for r in pipeline if r["sequence"] == sequence and r["arm"] == arm
                       and r["path"] == "tracking" and r["mode"] == "single"]
            queued, = [r for r in pipeline if r["sequence"] == sequence and r["arm"] == arm
                       and r["path"] == "tracking" and r["mode"] == "queued"]
            assert single["delivery_ms"]["samples"] == queued["delivery_ms"]["samples"] == 480
            percentile_rows.append([sequence.removesuffix("-SDP"), short_names[arm]] +
                                   [f'{row["delivery_ms"][key]:.3f}' for row in [single, queued]
                                    for key in ["p95", "p99"]])
    percentile_headers = ["序列", "模型", "Single P95", "P99", "Queued P95", "P99"]
    percentile_md = md_table(percentile_headers, percentile_rows)
    percentile_tex = tex_table("逐幀交付延遲的 P95／P99（ms）；每組 480 筆合併樣本", "P{21mm}Yrrrr",
                               percentile_headers, percentile_rows)
    (HERE / "latency-percentiles.tex").write_text("% Generated from saved pipeline summary; modes remain separate.\n" + percentile_tex + "\n")
    data_headers = ["資料集", "段數", "幀數／模型", "官方 HOTA", "微調 HOTA", "HR HOTA", "Δ HR−官方", "Δ HR−微調"]
    group_headers = ["主要群組", "段數", "官方 HOTA", "微調 HOTA", "HR HOTA", "Δ HR−微調", "逐段 HR／微調 勝／平／負"]
    detail_headers = ["資料集", "模型"] + METRICS
    timing_headers = ["模型", "F03 Graph GPU mean ms", "Tracking single mean ms", "single p95 ms", "queued FPS", "Engine MiB"]
    method = """本節補入 2026-11-11 已完成的本機 TensorRT 三方研究，固定 runtime 基準仍為 `3cbaa518f0dc`；這些新增引用的比較證據未包含在原報告基準 commit 中。本次只讀取既有分數與效能摘要，沒有重跑推論、計分、訓練或調參。模型分為官方預訓練 YOLO26m、UA2 teacher e4 primary EMA 微調 YOLO26m，以及目前 Full25 mixed student e10 primary EMA HR Mamba。

三方採相同 native F01／F02、1280×768、batch 1、FP16 allowed、FP32 I/O、共用 F04 與 G1 tracker；Graph ON，candidate／motion／result slots=2／2／4，NSA 與量測框 OFF。各模型使用自己的偵測及 detector-conditioned GMC，並非固定同一份 motion 的單因素消融。

**官方對照的範圍：**此處官方與微調 YOLO26m 都使用 raw6 **one2many** head，接本專案共用 F04／G1 tracker；不是 Ultralytics 預設 **one2one/end2end** 完整流程，也不是官方追蹤器的排名。既有研究另有官方原始 one2one 的模型層控制，但未對其完整追蹤品質重新評分，因此不混入本節品質表。

所有品質指標為 0–100 尺度，差值為百分點（pp）；IDSW 是次數，越低越好。DetA／AssA 為 HOTA 的偵測／關聯指標，不是 AP50。彙總使用 frozen TrackEval 的完整 raw metric objects 與 `combine_sequences`，沒有平均各段 HOTA。表格顯示三位小數，完整精度及 FP／FN／Frag 見原始 JSON／逐段 CSV。
"""
    coverage = """每模型完整覆蓋 **156 段／135,503 幀**，三模型合計 468 cells／406,509 幀；21 個 MOT17 cells 逐檔核對後沿用前輪 native TRT 結果，其餘 447 cells 為該輪新增推論。PP22 即 PersonPath22，48,452 是完整序列幀數，其中 **8,753 個有 GT 的原始影格**參與計分；其餘幀仍推論，計分前只保留 GT 存在幀的預測，保留原幀號。這些是本機指定 split／資料池，並非各資料集官方隱藏測試集排名。

資料來源為 MOT17 `train` 七個不同 SDP 序列、MOT20 `train` 四段、MOT15 `train` 十一段、DanceTrack `train` 40＋`val` 25 段，以及 PP22 `mot_test_kf` 69 段。這裡的 train 是原始資料集目錄名，實際本地模型訓練角色另見下述訓練來源界線。
"""
    interpretation = """HR 相對微調 YOLO26m 的七個主要群組為 **3 勝／0 平／4 負**，156 段為 **55 勝／0 平／101 負**。MOT17／MOT20 合併 HOTA 分別增加 2.053／7.742 pp；DanceTrack 與 PP22 則分別下降 0.543／1.974 pp。MOT15 合併僅增加 0.153 pp，moving 子群仍下降 0.849 pp，且全 11 段只有 2 段增加。即使相對官方預訓練版，HR 在 DanceTrack 與 PP22 的 HOTA 也較低。因此現有結果支持資料集相關的取捨，不能宣稱普遍優於官方或微調原版。

MOT17 雖然 HOTA／IDF1 合併增加，IDSW 仍由微調版 364 增至 HR 385；MOT20 由 2,401 增至 3,754，PP22 由 4,484 增至 5,138。需同時檢視偵測、關聯與身份切換，不能從單一總分推論身份穩定全面改善。
"""
    speed = """此表引用另一輪受控效能研究：RTX 5070 Ti Laptop、相同 TensorRT SDK／輸入與交錯順序；tracking 使用 MOT17-10 前 140 幀，20 幀暖機、每模式四個 fresh sessions／480 個量測樣本，JPEG 預載。Single 為 enqueue→receive；queued FPS 為整段完成吞吐。均排除讀檔、初始化、暖機與寫檔，不代表完整資料集或攝影機端到端效能。F03 為模型層 GPU events，與完整 tracking 的量測終點不同。
"""
    limits = """- 全部 156 段皆已曝露；MOT17 的 04／05／09／11／13 為兩條本地模型 lineage 的訓練序列、10 用於選模，02 是 M4 之後重用的原 held-out。官方預訓練 corpus 未全面稽核，不能套用本地訓練分工。DanceTrack 為 train＋val 的既定資料池，其餘來源／split 見 frozen selection。
- 微調版與 HR 的訓練配方、來源及 score calibration 不同；差值不能單獨歸因於 Mamba。七組與逐段結果優先，跨資料集 pooled 值不作主要排名或採用門檻。
- 官方／微調 FP16 engine 的既有 1% raw relative-L2 screen 仍為 **FAIL**，最大 2.406%／1.537%；任務分數、finite／repeat／Graph 通過不取代數值驗收，也不升格成正式 session／sanitizer／部署驗收。
- SportsMOT 依既定政策排除主要推論、評分、gate 與參數選擇；本節的五資料集與七群組均不含它。
- primary156 不測速，capture wall time 含 lock wait，不用來推算各資料集 FPS。各資料集沒有完整同口徑 AP50 與效能矩陣者不補造數字；現有可比項目以本節為準。部署模型與預設維持不變。
"""
    qlink = "../../evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle"
    plink = "../../evidence/model-comparison/2026-11-11-yolo26m-trt-performance"
    speed_conclusion = "HR 相對微調版在 MOT17-10 的 single 平均延遲增加 14.79%、queued 吞吐下降 15.52%；另測 MOT17-02 為 +17.42%／−15.13%。F03 Graph mean 增加 25.76%，engine 檔案增加 41.77%；execution-context scratch bound 同為約 135.17 MB，不代表完整行程顯存相同。MOT17-10 的 HR single p95 略低，02 則較高，尾延遲亦非全場景一致。"
    md = "\n".join([START, "### 9.2 官方原版對照與各資料集表現", "", method,
         "#### 五資料集覆蓋與 HOTA 對照", "", coverage, md_table(data_headers, datasets), "",
         "#### 七個主要群組：保留場景差異", "", md_table(group_headers, groups), "", interpretation,
         "#### 偵測、關聯與身份完整指標", "", md_table(detail_headers, details), "",
         "#### 同條件 TensorRT 的速度與成本", "", speed, md_table(timing_headers, timing), "", speed_conclusion, "",
         "逐幀交付延遲的 P95／P99（ms）：各組四次 session 的 480 筆暖機後樣本合併後取 nearest-rank，single 與 queued 分開。", "",
         percentile_md, "",
         "", "另列官方原始 one2one/end2end 的模型層 Graph 控制（四張固定輸入、每模型四個 fresh processes／1,280 樣本）：", "",
         md_table(["模型", "GPU mean ms", "GPU p95 ms"], original), "",
         "官方輸出為 `[1,300,6]`，engine 內含 decode／TopK；HR 仍輸出 raw6，兩者輸出終點不同。這一列控制不提供完整 tracking 品質或吞吐，也不與前表不同 campaign 的均值相減推導階段成本。", "",
         "#### 解讀與證據界線", "", limits,
         f"來源：[primary156 報告]({qlink}/REPORT.md)、[完整分數]({qlink}/metrics/summary.json)、[逐段 CSV]({qlink}/metrics/per-sequence.csv)、[獨立 scoring audit]({qlink}/audits/scoring-audit.json)、[訓練來源]({qlink}/training-provenance.json)、[固定 selection]({qlink}/methods/scoring-tools/selection-frozen.json)；[TRT 效能比較]({plink}/REPORT.md)、[pipeline 原始摘要]({plink}/performance/pipeline/summary.json)、[驗證界線]({plink}/validation.json)。本節表格由 [render_comparison.py](latex/report1/render_comparison.py) 讀取以上既有 JSON 產生，來源 hashes 見 [comparison-sources.json](latex/report1/comparison-sources.json)。",
         "", END])
    text = MARKDOWN.read_text()
    assert text.count(START) == text.count(END) == 1
    before, rest = text.split(START)
    _, after = rest.split(END)
    text = before + md + after
    p_start, p_end = "<!-- BEGIN GENERATED LATENCY PERCENTILES -->", "<!-- END GENERATED LATENCY PERCENTILES -->"
    assert text.count(p_start) == text.count(p_end) == 1
    p_before, p_rest = text.split(p_start)
    _, p_after = p_rest.split(p_end)
    text = p_before + p_start + "\n\n" + percentile_md + "\n\n" + p_end + p_after
    MARKDOWN.write_text(text)
    tex = r"""% Generated by render_comparison.py from saved evidence; do not hand-edit tables.
\subsection{比較對象與共同條件}
本節整理 2026-11-11 的本機三方 TensorRT 研究。程式基準為 \code{3cbaa518f0dc}，新增比較證據未包含在原基準 commit 中。本次只讀取既有分數與測時，不重跑推論、計分、訓練或調參。

三模型為官方預訓練 YOLO26m、UA2 teacher e4 primary EMA 微調版，以及現行 Full25 mixed student e10 primary EMA HR Mamba。

三方同為 native F01／F02、1280×768、batch 1、FP16 allowed、FP32 I/O、共用 F04 與 G1 tracker；Graph ON，slots=2／2／4，NSA 與量測框 OFF。各模型產生自己的偵測及 detector-conditioned GMC，並非固定同一份 motion 的單因素消融。

\begin{finding}
\textbf{官方對照使用 raw6 one2many head，接本專案共用 F04／G1 tracker。}\par
它與 Ultralytics 預設 one2one/end2end 的輸出契約不同，也不是官方追蹤器排名。既有研究另有原始 one2one 的模型層控制，但未重評其完整追蹤品質，故不混入品質表。
\end{finding}

品質指標為 0–100 尺度，差值為百分點（pp）；IDSW 是次數，越低越好。DetA／AssA 是 HOTA 的偵測／關聯指標，不是 AP50。合併採 frozen TrackEval 的完整 raw objects 與 \code{combine_sequences}，不平均各段 HOTA。表格取三位小數，完整精度與 FP／FN／Frag 留在原始 JSON／CSV。\cite{primary156,trtcompare}

\subsection{五資料集覆蓋與整體結果}
每模型覆蓋 156 段、135,503 幀，合計 468 cells／406,509 幀；21 個 MOT17 cells 逐檔驗證後沿用前輪 native TRT，其餘 447 cells 於該輪推論。PP22 為 PersonPath22，48,452 是完整序列幀數，其中僅 8,753 個有 GT 的原始影格參與計分；其餘幀仍推論，計分前只保留 GT 存在幀的預測與原幀號。這是本機指定 split／資料池，並非官方隱藏測試集排名。

來源為 MOT17 train 七個不同 SDP 序列、MOT20 train 四段、MOT15 train 十一段、DanceTrack train 40＋val 25 段、PP22 mot\_test\_kf 69 段。Train 指原始資料集目錄名，實際本地模型訓練角色另述。
"""
    tex += tex_table("各資料集 HOTA 與樣本量；Δ 為 HR 減比較對象（pp）", "Yrrrrrrr",
                     ["資料集", "段數", "幀數", "官方", "微調", "HR", "Δ 官方", "Δ 微調"], datasets)
    tex += "\n\\source{primary156 的 \\path{metrics/summary.json} 及獨立 scoring audit\\cite{primary156}。}\n"
    tex += r"\subsection{七個主要群組與逐段差異}" + "\n"
    tex += tex_table("主要群組 HOTA；勝／平／負為逐段 HR 相對微調版", "Yrrrrrr",
                     ["群組", "段數", "官方", "微調", "HR", "Δ 微調", "勝／平／負"], groups)
    tex += "\n" + interpretation.replace("**", "") + "\n"
    tex += r"\subsection{偵測、關聯與身份指標}" + "\n"
    tex += tex_table("三模型完整品質結果；各資料集採正式合併方法", "P{22mm}Yrrrrrr", detail_headers, details)
    tex += "\n" + r"\subsection{同條件速度與成本取捨}" + "\n" + speed.replace("→", "\\ensuremath{\\to}") + "\n"
    tex += tex_table("MOT17-10 prefix 的受控效能；F03 與 tracking 分開量測", "Yrrrrr",
                     ["模型", "F03 ms", "Single mean ms", "Single p95 ms", "Queued FPS", "Engine MiB"], timing)
    tex += "\n" + speed_conclusion.replace("%", r"\%") + "\n"
    tex += "\n另列官方原始 one2one/end2end 的模型層 Graph 控制，四張固定輸入、每模型四個 fresh processes／1,280 樣本。\n"
    tex += tex_table("官方原始輸出契約的模型層控制；輸出終點不同", "Yrr",
                     ["模型", "GPU mean ms", "GPU p95 ms"], original)
    tex += "\n官方輸出為 \\code{[1,300,6]}，engine 內含 decode／TopK；HR 仍為 raw6。此控制未重評完整 tracking 品質或吞吐，亦不與前表不同 campaign 的均值相減推導階段成本。\n"
    tex += r"\source{TensorRT 性能比較、model／pipeline summaries 與 model-memory.json\cite{trtcompare}。}" + "\n"
    tex += r"\subsection{資料曝光、數值與驗收邊界}" + "\n\\begin{itemize}\n"
    for line in limits.strip().splitlines():
        tex += "\\item " + line.removeprefix("- ").replace("**", "").replace("%", r"\%") + "\n"
    tex += "\\end{itemize}\n"
    tex += r"\source{原始 selection、training-provenance、evaluation-scope、獨立 scoring audit 與 TRT validation\cite{primary156,trtcompare}。}" + "\n"
    (HERE / "model-comparison.tex").write_text(tex)
    numeric = "% Generated from the same saved metrics as model-comparison.tex.\n"
    numeric += tex_table("資料覆蓋與 HOTA；Δ 為 HR 減比較對象（pp）", "Yrrrrrrr",
                         ["資料集", "段數", "幀數", "官方", "微調", "HR", "Δ 官方", "Δ 微調"], datasets) + "\n"
    numeric += tex_table("完整品質數字；IDSW 是次數，其餘為百分點尺度", "P{22mm}Yrrrrrr", detail_headers, details) + "\n"
    numeric += tex_table("七個既定群組；逐段勝／平／負為 HR 相對微調版", "Yrrrrrr",
                         ["群組", "段數", "官方", "微調", "HR", "Δ 微調", "勝／平／負"], groups) + "\n"
    numeric += tex_table("MOT17-10 前 140 幀；F03 和完整 tracking 分開測量", "Yrrrrr",
                         ["模型", "F03 ms", "Single mean ms", "Single p95 ms", "Queued FPS", "Engine MiB"], timing) + "\n"
    (HERE / "numeric-tables.tex").write_text(numeric)
    receipt = {"schema": "report-comparison-sources/1", "revision_date": "2026-11-12",
               "runtime_base": "3cbaa518f0dc37d691404e95ed85c0109c81130d",
               "operation": "render existing saved metrics only; no inference, rescoring, training or GPU benchmark",
               "evidence_manifest_checks": "all cited inputs match their existing SHA256SUMS",
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES],
               "outputs": {"markdown_section": "9.2", "latex": "model-comparison.tex"},
               "primary_cells": 468, "sequences": 156, "frames_per_model": 135503,
               "formal_acceptance": False, "deployment_changed": False}
    (HERE / "comparison-sources.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(f"Rendered comparison and single/queued P95/P99 tables; {len(SOURCES)} cited input hashes verified.")


if __name__ == "__main__":
    main()
