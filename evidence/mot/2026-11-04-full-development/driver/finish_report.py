"""Archive the completed MOT run and write its local Markdown report.

This generator never runs inference or scores ground truth. It refuses before
the driver, acceptance, evaluation gates and scoring have all completed.
"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "build/mot-full-20261104-driver"
ACCEPT_ROOT = ROOT / "build/mot-full-20261104-acceptance"
ACCEPT = ACCEPT_ROOT / "capture"
EVAL_ROOT = ROOT / "build/mot-full-20261104-evaluation"
EVAL = EVAL_ROOT / "capture"
DEST = ROOT / "evidence/mot/2026-11-04-full-development"
M4 = ROOT / "evidence/m4/2026-11-03-formal-cad5298/evaluation"


def require(condition, message):
    if not condition:
        raise SystemExit("REFUSED: " + message)


def read(path):
    return path.read_text() if path.is_file() else ""


def values(path):
    return dict(line.split("=", 1) for line in read(path).splitlines() if "=" in line)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def hashes(path):
    return {name.lstrip("*"): value for line in read(path).splitlines()
            for value, name in [line.split(maxsplit=1)]}


def relative(path):
    return str(path.relative_to(ROOT))


def link(path, label=None):
    target = os.path.relpath(path, DEST)
    return f"[{label or path.name}]({target})"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def fmt(value, decimals=3):
    return "—" if value is None else f"{value:.{decimals}f}" if isinstance(value, (int, float)) else str(value)


def ctest_result(path):
    text = read(path)
    match = re.search(r"(?P<percent>\d+)% tests passed(?:, (?P<failed>\d+) tests failed)? out of (?P<total>\d+)", text)
    require(match is not None, f"CTest summary missing: {path}")
    require(match["percent"] == "100" and int(match["failed"] or "0") == 0, f"CTest failed: {path}")
    passed_lines = re.findall(r"^\s*\d+/\d+\s+Test\s+#\d+:.*?\bPassed\b", text, re.MULTILINE)
    require(len(passed_lines) == int(match["total"]), f"CTest Passed line count differs from summary: {path}")
    seconds = re.search(r"Total Test time \(real\) =\s*([\d.]+) sec", text)
    return {"passed": int(match["total"]), "failed": 0, "passed_result_lines": len(passed_lines), "seconds": float(seconds[1]) if seconds else None}


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    status = values(DRIVER / "status.txt")
    require(status.get("exit") == "0", "driver status.txt must contain exit=0")
    for path in (ACCEPT / "acceptance.txt", EVAL / "gates.txt", EVAL / "scoring.txt"):
        require(read(path).strip() == "PASS", f"{path} must contain PASS")
    require(status.get("finished"), "driver finished timestamp missing")
    started = values(DRIVER / "started.txt").get("started")
    require(started, "driver started timestamp missing")
    start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
    end_time = datetime.fromisoformat(status["finished"].replace("Z", "+00:00"))
    duration = (end_time - start_time).total_seconds()
    require(duration >= 0, "driver timestamps are reversed")

    subprocess.run([sys.executable, "-I", str(DRIVER / "prepare_report_inputs.py")], cwd=ROOT, check=True)
    inputs = json.loads((DRIVER / "report-inputs.json").read_text())
    require(inputs["current"]["metrics"] is not None, "current metrics missing")
    timing = inputs["current"]["timing"]
    frames = {row["sequence"]: row for row in inputs["frames"]["sequences"]}
    require(timing is not None and len(timing) == 7, "seven timing rows required")
    require({row["sequence"] for row in timing} == set(frames), "timing sequence set differs")
    for row in timing:
        require(row["frames"] == frames[row["sequence"]]["frames"], f"incomplete timing frame count for {row['sequence']}")
    total_frames = sum(row["frames"] for row in timing)
    wall_ms = sum(row["wall_ms"] for row in timing)
    require(total_frames == 5316 and wall_ms > 0, "full 5316-frame timing required")
    average_ms = wall_ms / total_frames
    fps = total_frames * 1000 / wall_ms

    output_hashes = hashes(EVAL / "outputs.sha256")
    expected_paths = {f"{folder}/{sequence}.{extension}" for folder, extension in (("scdr", "scdr"), ("mot", "txt"), ("offline", "txt")) for sequence in frames}
    require(set(output_hashes) == expected_paths, "output manifest must contain exactly seven SCDR, seven MOT and seven offline MOT files")
    for path, expected_sha in output_hashes.items():
        require(sha(EVAL_ROOT / path) == expected_sha, f"output file changed: {path}")
    m4_compare = inputs["current"]["output_manifest_comparison_to_m4"]
    require(m4_compare["expected"] == 21, "M4 manifest must contain 21 outputs")

    cpu, cuda = ctest_result(ACCEPT / "cpu-ctest.log"), ctest_result(ACCEPT / "ctest.log")
    packet_lines = re.findall(r"^MOT17-\d\d-SDP frames=\d+ CPU/ordinary serial/graph pipelined packet bytes identical; graph ring controls=\d+ frames$", read(ACCEPT / "LastTest.log"), re.MULTILINE)
    require(len(packet_lines) == 7, "all seven T01 packet-comparison lines required")
    hr_hashes = hashes(ACCEPT / "hr-outputs.sha256")
    require(hr_hashes["serial.bin"] == hr_hashes["pipelined.bin"], "HR MOT17-10 packet hashes differ")
    require(len({hr_hashes[name] for name in ("offline.txt", "serial.txt", "pipelined.txt")}) == 1, "HR MOT17-10 MOT hashes differ")
    require(read(ACCEPT / "hr-path.txt").strip() == "PASS", "HR path gate missing")

    sanitizer_names = [f"kernels-{tool}-{test}" for tool in ("memcheck", "initcheck", "racecheck", "synccheck")
                       for test in ("tracking_packet", "tracking_cuda", "motion_cuda")]
    sanitizer_names += ["session-memcheck", "lifecycle-memcheck", "hr-memcheck", "session-racecheck", "session-synccheck"]
    sanitizer_results = []
    for name in sanitizer_names:
        require(read(ACCEPT / (name + ".exit")).strip() == "0", f"sanitizer did not exit zero: {name}")
        log = read(ACCEPT / (name + ".log"))
        verdict = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if "racecheck" in name else "========= ERROR SUMMARY: 0 errors"
        require(log.splitlines().count(verdict) == 1, f"zero-error sanitizer summary missing: {name}")
        checks = re.findall(r"\b(\d+) checks passed", log)
        sanitizer_results.append({"name": name, "exit": 0, "summary": verdict, "checks": [int(item) for item in checks]})
    controls = []
    expected_diagnostics = inputs["acceptance_protocol"]["positive_controls"]["expected_diagnostics"]
    for tool, diagnostic in expected_diagnostics.items():
        name = "control-" + tool
        require(read(ACCEPT / (name + ".exit")).strip() == "99", f"positive control expected exit 99: {name}")
        require(diagnostic in read(ACCEPT / (name + ".log")), f"positive-control diagnostic missing: {name}")
        controls.append({"name": name, "exit": 99, "diagnostic": diagnostic})
    unexpected = []
    for capture in (ACCEPT, EVAL):
        for path in sorted(capture.glob("*.exit")):
            code = read(path).strip()
            expected = "99" if capture == ACCEPT and path.stem.startswith("control-") else "0"
            if code != expected:
                unexpected.append({"path": relative(path), "expected_exit": expected, "actual_exit": code})

    require(read(ACCEPT / "binaries.sha256") == read(ACCEPT / "binaries-after.sha256"), "acceptance binaries changed")
    require(read(ACCEPT / "artifacts.sha256") == read(ACCEPT / "artifacts-after.sha256"), "acceptance artifacts changed")
    require(read(EVAL / "acceptance-binaries.sha256") == read(EVAL / "binaries-after.sha256"), "evaluation binaries changed")
    require(read(EVAL / "artifacts.sha256") == read(EVAL / "artifacts-after.sha256"), "evaluation artifacts changed")
    require(inputs["identity"]["acceptance_source"]["allow_dirty"] == "true", "expected development source capture")
    require(inputs["identity"]["evaluation_source"]["allow_dirty"] == "true", "expected development evaluation capture")
    checker = ROOT / "scripts/check_stream_placement.py"
    source_files = hashes(EVAL / "source-files.sha256")
    require(sha(checker) == source_files["scripts/check_stream_placement.py"], "checker changed after evaluation capture")
    checker_tests = json.loads((DRIVER / "stream-checker-tests/results.json").read_text())
    require(checker_tests["status"] == "PASS" and checker_tests["total_cases"] == 14, "checker controls must pass 14 cases")

    metrics = inputs["current"]["metrics"]
    by_group = {row["row"]: row for row in metrics}
    sequence_metrics = {sequence: by_group["eval" if sequence == "MOT17-02-SDP" else "val" if sequence == "MOT17-10-SDP" else sequence] for sequence in frames}
    quality_keys = ("HR_AP50", "T01_AP50", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")
    quality_changes = {row["row"]: {key: {"baseline": old[key], "current": row[key]} for key in quality_keys if old[key] != row[key]}
                       for row in metrics for old in inputs["m4_baseline"]["rows"] if old["row"] == row["row"]}
    quality_changes = {name: changes for name, changes in quality_changes.items() if changes}
    d21 = {row["sequence"]: row for row in inputs["d21_timing_baseline"]["rows"]}
    environment_lines = read(ACCEPT / "environment.log").splitlines()
    gpu_line = next((line for line in environment_lines if line.startswith("NVIDIA GeForce ")), "GPU identity: see environment.log")
    platform_line = next((line for line in environment_lines if line.startswith("Linux ")), "see environment.log")
    binaries = hashes(ACCEPT / "binaries.sha256")
    tool_hashes = {name: next(value for path, value in binaries.items() if Path(path).name == name)
                   for name in ("detect_sequence", "track_sequence", "track_offline")}
    source = inputs["identity"]["acceptance_source"]
    evaluation_source = inputs["identity"]["evaluation_source"]
    summary = {
        "status": "PASS" if not unexpected else "FAIL",
        "formal_acceptance": False, "source_dirty": True, "post_m4": True,
        "driver": {"start": started, "finish": status["finished"], "wall_seconds": duration, "exit": 0},
        "full_dataset_frames": total_frames, "sequence_count": len(frames),
        "timing": {"total_processing_wall_ms": wall_ms, "weighted_ms_per_frame": average_ms,
                   "weighted_fps_throughput": fps, "rows": timing, "boundaries": inputs["timing_boundaries"]},
        "quality_groups": [by_group[name] for name in ("val", "eval", "in-domain")], "quality_sequences": sequence_metrics,
        "quality_detailed": inputs["current"]["detailed"],
        "quality_changes_from_m4": quality_changes,
        "outputs_vs_m4": m4_compare, "output_files_verified_against_manifest": len(output_hashes),
        "cpu_ctest": cpu, "cuda_ctest": cuda, "t01_full_sequence_packet_comparisons": packet_lines,
        "hr_mot17_10_packet_sha256": hr_hashes["pipelined.bin"], "hr_path_hashes": hr_hashes,
        "sanitizers": sanitizer_results, "positive_controls": controls,
        "sanitizer_scopes": inputs["acceptance_protocol"], "checker_controls": checker_tests,
        "source_identity": inputs["identity"], "tool_sha256": tool_hashes,
        "unexpected_nonzero_exits": unexpected,
        "report_infrastructure_first_attempt": json.loads((DRIVER / "report-first-attempt.json").read_text()) if (DRIVER / "report-first-attempt.json").is_file() else None,
        "raw_build_paths": {"driver": relative(DRIVER), "acceptance": relative(ACCEPT_ROOT), "evaluation": relative(EVAL_ROOT)},
    }

    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ACCEPT, DEST / "acceptance", dirs_exist_ok=True)
    shutil.copytree(EVAL, DEST / "evaluation", dirs_exist_ok=True)
    for path in DRIVER.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".sqlite", ".db"} or path.name.endswith((".sqlite-shm", ".sqlite-wal")) or "__pycache__" in path.parts:
            continue
        copy_file(path, DEST / "driver" / path.relative_to(DRIVER))
    copy_file(checker, DEST / "source/check_stream_placement.py")
    checker_diff = subprocess.run(["git", "diff", "HEAD", "--", "scripts/check_stream_placement.py"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    (DEST / "source/checker.diff").write_text(checker_diff)
    for trackeval in sorted((EVAL_ROOT / "score").glob("*/trackeval")):
        group = trackeval.parent.name
        for path in trackeval.rglob("*"):
            if path.is_file() and (path.name in {"pedestrian_summary.txt", "pedestrian_detailed.csv", "trackeval.log", "run.py"}):
                copy_file(path, DEST / "evaluation/trackeval" / group / path.relative_to(trackeval))
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    group_labels = {"val": "val MOT17 10", "eval": "eval MOT17 02 M4 之後", "in-domain": "in domain 04 05 09 11 13"}
    group_rows = [[group_labels[name], fmt(by_group[name]["HR_AP50"], 4), *[fmt(by_group[name][key], 0 if key == "IDSW" else 3) for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")]] for name in ("val", "eval", "in-domain")]
    quality_rows = [[sequence, fmt(sequence_metrics[sequence]["HR_AP50"], 4), *[fmt(sequence_metrics[sequence][key], 0 if key == "IDSW" else 3) for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")]] for sequence in frames]
    speed_rows = [[row["sequence"], row["frames"], fmt(row["ms_per_frame"], 5), fmt(1000 / row["ms_per_frame"], 2), fmt(d21[row["sequence"]]["ms_per_frame"], 5)] for row in timing]
    result_rows = [
        ["CPU CTest", f"PASS {cpu['passed']}/{cpu['passed']}", "acceptance/cpu-ctest.log"],
        ["CUDA CTest", f"PASS {cuda['passed']}/{cuda['passed']}", "acceptance/ctest.log"],
        ["HR MOT17 10", "CPU offline、serial、pipelined MOT 相同；serial／pipelined packet bytes 相同", "acceptance/hr-outputs.sha256"],
        ["HR 七序列守門", "身分鏈、offline／sequence、既有輸出 hash、兩次 nsys stream placement PASS", "evaluation/gates.log"],
        ["M4 輸出比對", f"{m4_compare['status']} {m4_compare['matched']}/21", "summary.json"],
        ["Sanitizer", "17 次 0 errors／0 hazards；4 個正控制皆 exit 99 且診斷符合", "acceptance/"],
        ["品質計分", "三組與七序列完成；MOT17 02 最後計分", "evaluation/metrics.tsv"],
        ["Stream checker controls", "14/14 PASS", "driver/stream-checker-tests/results.tsv"],
    ]
    result_table = table(["項目", "結果", "證據"], [[name, result, f"[{path}]({path})"] for name, result, path in result_rows])
    identity_rows = [
        ["HEAD", source["revision"]], ["Acceptance source digest", source["source_digest"]],
        ["Evaluation source digest", evaluation_source["source_digest"]],
        ["HR FP16 engine", inputs["identity"]["engine_sha256"]],
        ["Backbone ONNX", inputs["identity"]["engine_build_report"]["onnx_sha256"]],
        ["Head bundle", inputs["identity"]["engine_build_report"]["bundle_sha256"]],
        ["Deploy manifest", inputs["identity"]["deploy_manifest_sha256"]],
        ["Acceptance binary manifest SHA256", sha(ACCEPT / "binaries.sha256")],
        ["Acceptance artifact manifest SHA256", sha(ACCEPT / "artifacts.sha256")],
        ["Evaluation artifact manifest SHA256", sha(EVAL / "artifacts.sha256")],
        ["Acceptance build input snapshot tar gzip", sha(DRIVER / "source.tar.gz")],
        *[[name, value] for name, value in tool_hashes.items()],
        ["HR MOT17 10 packets", hr_hashes["pipelined.bin"]],
        ["Stream checker", sha(checker)],
    ]
    quality_statement = "本次八列品質彙整與 M4 相同，AP50／HOTA／DetA／AssA／IDF1／MOTA／IDSW 的差值均為 0。" if not quality_changes else "本次品質與 M4 的差異逐欄保存於 [summary.json](summary.json) 的 quality_changes_from_m4；不以本次結果重新選模型或調參。"
    equality_statement = f"HR 七序列的 SCDR、native MOT 與 CPU offline MOT 共 {m4_compare['matched']}/21 份與 M4 逐位元相同。" if m4_compare["status"] == "PASS" else f"HR 七序列共 {m4_compare['matched']}/21 份輸出與 M4 相同；差異保存在 [summary.json](summary.json) 的 outputs_vs_m4。"
    failure_statement = "capture 中除四個預期 exit 99 的 sanitizer 正控制外，沒有非預期非零退出。" if not unexpected else "非預期退出碼與其預期值保存在 [summary.json](summary.json) 的 unexpected_nonzero_exits。"
    report_issue = "第一次報告生成因 parser 只接受舊 CTest summary 格式而 REFUSED／exit 1；修正為支援新舊格式，並核對 CPU 16 與 CUDA 59 個 Passed 結果行後重新生成。這是報告工具問題，沒有重跑 GPU 測試或計分，紀錄見 [report-first-attempt.json](driver/report-first-attempt.json)。" if summary["report_infrastructure_first_attempt"] else ""
    report = f"""# MOT17 完整開發來源測試

日期：2026-11-04。現行工作樹的 HR 1280×768 FP16 模型完成七個 MOT17 SDP 序列共 **{total_frames:,} 幀**，tracking acceptance、輸出守門與品質計分均 PASS。{equality_statement}七序列合計的處理吞吐為 **{fps:.2f} FPS**（加權 {average_ms:.5f} ms／幀）。

受測 HEAD 為 `{source['revision']}`，`source_dirty=true`、`allow_dirty=true`、`formal_acceptance=false`。來源包含現行 F02 tile fusion 與 F03 scan 修改；這是保留未提交內容的開發來源測試。MOT17-02 已在 M4 開啟，本次記為 **M4 之後的重跑**。部署 engine 與 tracker 設定沿用現行版本。

## 執行結果

{result_table}

driver 實際開始於 `{started}`，結束於 `{status['finished']}`，牆鐘歷時 **{duration:.0f} 秒**（{duration / 60:.2f} 分鐘）。這段時間涵蓋 build、CTest、sanitizer、MOT 產生、profiler 守門與計分，與下方 pipeline 處理時間分開。原始狀態見 [driver/status.txt](driver/status.txt) 與 [driver/run.log](driver/run.log)。

## 品質

依 [TRACKING 第 17 節](../../../docs/TRACKING.md) 分組，不設品質門檻。AP50 為 HR FP16 kept detections 的百分比；其他品質分數維持 TrackEval 的 0–100 尺度，IDSW 為次數。MOT17-02 在其他序列之後計分。

{table(['組別', 'HR AP50 %', 'HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW'], group_rows)}

{table(['序列', 'HR AP50 %', 'HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW'], quality_rows)}

{quality_statement}精確 AP50、T01 AP50 對照與逐列數值見 [metrics.tsv](evaluation/metrics.tsv)；FP／FN、identity counts 與完整表見 [TrackEval](evaluation/trackeval/) 及 [summary.json](summary.json)。M4 原始對照為 [metrics.tsv](../../m4/2026-11-03-formal-cad5298/evaluation/metrics.tsv)。

## 處理速度

每個序列各跑一次，沒有開 profiler。`track_sequence` 在 session 建構後、第一張 JPEG 讀取前開始計時，在 `finish()` 與全部 ResultPacket 收回後停止；包含讀圖、host submission 與 pipeline，排除 engine 初始化、`session.close()`、MOT／packet 檔寫入。實作見 [track_sequence.cpp](../../../tools/track_sequence.cpp)。

{table(['序列', '幀數', '本次 ms／幀', '本次 FPS', 'd21a144 ms／幀'], speed_rows)}

本次七序列回報的處理時間合計 `{wall_ms:.2f} ms`；以總幀數除以總時間得到 **{fps:.2f} FPS**。這是 pipelined 吞吐，並非單幀交付 latency。原值見 [timing.tsv](evaluation/timing.tsv)。d21a144 是先前獨立 capture 的單次觀察，見 [歷史 timing.tsv](../../cuda-graph/2026-11-04-gate-d21a144/mot17-run/timing.tsv)；兩次沒有控制 clock／溫度或做 ABBA，並列數值不作為可歸因的加速比。歷史報告寫入 MOT 寫檔的措辭，以現行 executable 的上述計時邊界為準。

## 輸出與驗證範圍

T01 640×640 reference engine 的七序列測試涵蓋全部 {total_frames:,} 幀，CPU reference、ordinary serial 與 graph pipelined 的固定容量 packet bytes 相同，另檢查 ring controls；完整行保存在 [LastTest.log](acceptance/LastTest.log)。這項 packet 測試使用 T01 engine。

HR FP16 的 MOT17-10 全部 {frames['MOT17-10-SDP']['frames']} 幀另比較 serial／pipelined packet bytes，SHA-256 為 `{hr_hashes['pipelined.bin']}`；CPU offline、serial 與 pipelined MOT 相同。HR 的七個完整序列均另外產生 detection SCDR、native MOT 與 CPU offline MOT，七序列 offline／native MOT 逐位元一致；全七序列的 HR serial／pipelined packet 對照不在這次範圍。

{equality_statement}全部 21 份本機檔案在生成本報告前再次核對本次 [outputs.sha256](evaluation/outputs.sha256)；M4 比較依據 [M4 outputs.sha256](../../m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256)。

Sanitizer 的 17 次成功執行分為 12 個 kernel suites、3 個 bounded session memcheck、2 個 filtered session racecheck／synccheck。kernel suites 為 tracking packet、tracking CUDA、motion CUDA，各跑四種工具；initcheck 關閉 API memory copy tail 檢查，motion 使用 synthetic 案例避開既有 nvJPEG 診斷。三個 session memcheck 分別是 T01 前 30 幀、T01 lifecycle、HR FP16 前 30 幀，包含該 bounded run 的 TensorRT／nvJPEG／cuFFT kernels。session racecheck／synccheck 僅對 T01 前 10 幀的 tracking／motion／session control kernels 插樁；不含 TensorRT 內部、nvJPEG、cuFFT、F02 resize 或 scan kernels。四個植入缺陷的正控制均 exit 99，且各有預期診斷，原始命令與 log 在 [acceptance](acceptance/)。

本次沒有執行完整 session initcheck、TensorRT 內部 racecheck／synccheck、全幀 HR sanitizer 或新的 semantic mutation suite。既有 nvJPEG initcheck 診斷維持原狀，見 [PREPROCESS](../../../docs/PREPROCESS.md)。

## Stream 判定與執行問題

在評估前發現 stream checker 只辨識舊 F02 `horizontal`／`vertical`，而現行 1920×1080→1280×768 路徑改用 `tiled_hr`。本次只在 F02 的名稱規則加入 `tiled_hr`，仍要求 F02 位於 D stream；[原始 diff](source/checker.diff) 與 [受測 checker](source/check_stream_placement.py) 保留。其餘 D／M、graph content、launch 配對與 unknown-kernel 規則不變。

checker replay 的 14 案全部符合預期：原始 real-capture 的八個負控制與 unchanged 正例，加上受控 fused fixture 正例、node／graph 的錯 stream、缺少 F02、unknown kernel 四個負控制。fused fixture 是刪除舊 horizontal、將 vertical 改為精確 `tiled_hr` 名稱的 sqlite 副本；這些 fixture 只驗證 checker。實際新 kernel 的 stream 判定由本次 node／graph 兩個真實 nsys capture 通過，見 [stream-placement.txt](evaluation/stream-placement.txt)。重播腳本與全部 verdict 在 [stream-checker-tests](driver/stream-checker-tests/)，synthetic sqlite 不封存。

{failure_statement}checker 相容性在完整 MOT 評估前修正，沒有將未執行的舊 checker 失敗記為一次 capture 失敗。

{report_issue}

## 來源與產物身分

環境：`{platform_line}`；GPU capture：`{gpu_line}`。工具版本與完整環境見 [environment.log](acceptance/environment.log)。engine 為 {inputs['identity']['engine_bytes']:,} bytes，profile `hr-1280x768`；model input／raw outputs 與 scan 邊界維持 FP32，backbone／head tactics 為 FP16，aux streams 為 0。engine 最初建置仍保留既有 `build_dirty=true` 身分，並以固定 engine SHA-256 識別。

{table(['對象', 'SHA256 或 revision'], [[name, '`' + value + '`'] for name, value in identity_rows])}

完整逐檔 binary hashes 見 [binaries.sha256](acceptance/binaries.sha256)，前後 manifest 相同；source file hashes、原始工作樹狀態與 patch 分別見 [acceptance source](acceptance/source.txt)、[source-files.sha256](acceptance/source-files.sha256)、[source-status.txt](acceptance/source-status.txt)、[source.diff](acceptance/source.diff) 與 [evaluation source](evaluation/source.txt)。acceptance source digest 不含 stream checker，evaluation source digest 包含已修正的 checker。[source.tar.gz](driver/source.tar.gz) 封存 acceptance build inputs，其明列檔案清單不包含 checker；修正後的評估腳本由 [evaluation source.diff](evaluation/source.diff)、[checker snapshot](source/check_stream_placement.py) 與上述 SHA-256 保存。

engine、deploy manifest、訓練 checkpoint、ONNX／head bundle、reference corpus、M1 records、七序列 image 身分鏈保存在 [acceptance artifacts](acceptance/artifacts.sha256)、[evaluation artifacts](evaluation/artifacts.sha256)、[identity.txt](evaluation/identity.txt) 與 [deploy manifest](evaluation/hr-deploy-manifest.json)。前後 artifact hashes 相同。完整數值與來源欄位保存在 [summary.json](summary.json)。

## 重跑與封存

以下從專案根目錄執行，輸出必須使用尚不存在的新目錄。`--allow-dirty` 明確保存開發來源身分。

```sh
export SACCADE_TRACKING_ENGINE="$PWD/build/t01-f03-evidence/stage/model.engine"
export SACCADE_DETECTION_REFERENCES="$PWD/build/preprocess-evidence/refs"
export SACCADE_TRACKING_SEQUENCES=/home/ray/developer/ai/saccade/datasets/MOT17/train
export SACCADE_TRACKING_RECORDS=/home/ray/developer/ai/saccade_v1-worktrees/m1-data
flock ~/.cache/saccade-v1/gpu.lock bash scripts/run_tracking_session_acceptance.sh build/NEW-acceptance --allow-dirty
flock ~/.cache/saccade-v1/gpu.lock bash scripts/run_mot17_evaluation.sh build/NEW-acceptance build/NEW-evaluation --allow-dirty
bash scripts/score_mot17_evaluation.sh build/NEW-evaluation --allow-dirty
```

本次 [driver/run.sh](driver/run.sh)、報告生成器、acceptance build input snapshot、checker replay、acceptance／evaluation capture、TrackEval summary／detailed 表及 logs 均在本目錄。大型 raw artifacts 留在本機忽略目錄：

- [build/mot-full-20261104-acceptance](../../../build/mot-full-20261104-acceptance/)：CPU／CUDA binaries、HR MOT／SCDR／SCMO、完整 HR10 packet streams。
- [build/mot-full-20261104-evaluation](../../../build/mot-full-20261104-evaluation/)：七序列 SCDR、native MOT、offline MOT、實際 nsys `.nsys-rep` 與 sqlite。
- [build/mot-full-20261104-driver](../../../build/mot-full-20261104-driver/)：執行狀態、source snapshot、checker fixtures 與原始 replay workdirs。

封存排除 raw binaries、重複的 TrackEval tracker inputs、真實／合成 sqlite 與大型 nsys captures。保留的每個檔案由 [SHA256SUMS](SHA256SUMS) 校驗：

```sh
cd evidence/mot/2026-11-04-full-development
sha256sum -c SHA256SUMS
```
"""
    (DEST / "REPORT.md").write_text(report)
    files = sorted(path for path in DEST.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (DEST / "SHA256SUMS").write_text("".join(f"{sha(path)}  {path.relative_to(DEST)}\n" for path in files))
    for path in files:
        require(sha(path) == hashes(DEST / "SHA256SUMS")[str(path.relative_to(DEST))], f"archive hash verification failed: {path}")
    print(json.dumps({"report": relative(DEST / "REPORT.md"), "summary": relative(DEST / "summary.json"),
                      "archived_files": len(files), "frames": total_frames, "fps_throughput": fps,
                      "m4_outputs_equal": m4_compare["matched"], "m4_outputs_total": m4_compare["expected"],
                      "wall_seconds": duration}))


if __name__ == "__main__":
    main()
