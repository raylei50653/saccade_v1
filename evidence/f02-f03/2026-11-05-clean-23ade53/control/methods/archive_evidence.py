#!/usr/bin/env python3
"""Archive a completed, independently audited campaign; never runs GPU work."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "build/closeout-control"
MAIN = Path("/home/ray/developer/ai/saccade_v1")
DEST = MAIN / "evidence/f02-f03/2026-11-05-clean-23ade53"
REVISION = "23ade536906d07ccf09859f1b580b044dafe95dc"
REFS = Path("/home/ray/.cache/saccade-v1/hr-deploy/refs")
ACCEPTANCE = ROOT / "build/clean-acceptance"
EVALUATION = ROOT / "build/clean-evaluation"
TEXT_SUFFIXES = {".txt", ".log", ".time", ".command", ".started", ".exit",
                 ".json", ".sha256", ".diff", ".tsv", ".csv", ".md", ".sh", ".py"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit("REFUSED: " + message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())


def main() -> None:
    audit = read_json(CONTROL / "capture/closeout.json")
    require(audit.get("status") == "PASS", "independent closeout is not PASS")
    require(audit.get("formal_acceptance") is True and audit.get("source_dirty") is False,
            "closeout is not clean-source formal acceptance")
    require(audit.get("revision") == REVISION and audit.get("motion_profile") == "v1-motion/1",
            "unexpected accepted source or motion profile")
    require(git("rev-parse", "HEAD") == REVISION, "checkout revision changed")
    require(not git("status", "--porcelain"), "accepted checkout is dirty")
    require(not DEST.exists() and not DEST.is_symlink(), f"destination already exists: {DEST}")
    for path, expected in ((ACCEPTANCE / "capture/acceptance.txt", "PASS"),
                           (EVALUATION / "capture/gates.txt", "PASS")):
        require(path.is_file() and path.read_text().strip() == expected, f"gate missing: {path}")
    require(audit.get("m4_identical_outputs") == 21, "M4 21-output audit missing")
    require((CONTROL / "capture/status.txt").read_text().startswith("exit=0\n"),
            "wrapper did not complete successfully")
    equivalence = read_json(CONTROL / "capture/component-equivalence.json")
    require(equivalence.get("accepted_revision") == REVISION and equivalence.get("files"),
            "component equivalence source mismatch")
    for row in equivalence["files"]:
        path = Path(row["path"])
        require(not path.is_absolute() and ".." not in path.parts, "unsafe component path")
        require(row.get("identical") is True and digest(ROOT / path) == row["sha256"],
                f"component drift: {path}")

    # Finish all checks above before creating anything in the publication checkout.
    DEST.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + DEST.name + "-partial-", dir=DEST.parent))
    copied = []

    def copy_file(source: Path, relative: str | Path) -> None:
        require(source.is_file() and not source.is_symlink(), f"missing/linked input: {source}")
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(digest(source) == digest(target), f"copy mismatch: {source}")
        copied.append({"source": str(source), "archive": str(target.relative_to(staging)),
                       "sha256": digest(target), "bytes": target.stat().st_size})

    def copy_capture(source: Path, relative: str) -> None:
        require(source.is_dir(), f"missing capture: {source}")
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                continue
            require(not path.is_symlink(), f"linked capture input: {path}")
            require(path.suffix in TEXT_SUFFIXES, f"unexpected capture payload: {path}")
            copy_file(path, Path(relative) / path.relative_to(source))

    copy_capture(ACCEPTANCE / "capture", "acceptance/capture")
    copy_capture(EVALUATION / "capture", "evaluation/capture")
    copy_capture(CONTROL / "capture", "control/capture")
    for name in ("run.sh", "postflight.py", "archive_evidence.py"):
        copy_file(CONTROL / name, "control/methods/" + name)
    for path in sorted(CONTROL.iterdir()):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.name not in {
                "run.sh", "postflight.py", "archive_evidence.py"}:
            copy_file(path, "control/" + path.name)
    for kind in ("cpu", "cuda"):
        for relative in ("CMakeCache.txt", "Testing/Temporary/LastTest.log"):
            copy_file(ACCEPTANCE / kind / relative, Path("build-metadata") / kind / relative)
    copy_file(ACCEPTANCE / "cuda/compile_commands.json", "build-metadata/cuda/compile_commands.json")
    for name in ("manifest.json", "cases.tsv"):
        copy_file(REFS / name, "hr-references/" + name)
    for row in equivalence["files"]:
        copy_file(ROOT / row["path"], Path("source/components") / row["path"])
    for name in ("scripts/run_tracking_session_acceptance.sh", "scripts/run_mot17_evaluation.sh",
                 "scripts/check_stream_placement.py", "docs/TRACKING.md", "docs/SCAN_CUDA.md",
                 "docs/PREPROCESS_POLICY.md", "docs/PREPROCESS_USAGE_POLICY.md"):
        copy_file(ROOT / name, Path("source/contracts-and-runners") / name)
    copy_file(ROOT / "evidence/m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256",
              "references/m4-outputs.sha256")

    # Retain paths and existing hash manifests, never the binaries/output payloads.
    local_payloads = {
        "accepted_cuda_binaries_and_libraries": {
            "directory": str(ACCEPTANCE / "cuda"),
            "manifest": "acceptance/capture/binaries.sha256"},
        "accepted_cuda_binary_archive": {
            "path": str(CONTROL / "accepted-binaries.tar.gz"),
            "manifest": "control/capture/accepted-binaries.sha256"},
        "hr_session_packets_and_outputs": {
            "directory": str(ACCEPTANCE / "hr"),
            "manifest": "acceptance/capture/hr-outputs.sha256"},
        "seven_sequence_outputs": {
            "directory": str(EVALUATION),
            "manifest": "evaluation/capture/outputs.sha256"},
        "hr_python_reference_payloads": {
            "directory": str(REFS),
            "manifest": "control/capture/hr-reference-before.sha256"},
        "inputs_and_records": {
            "manifests": ["acceptance/capture/artifacts.sha256",
                          "evaluation/capture/artifacts.sha256"]},
    }
    profiler = EVALUATION / "nsys"
    local_payloads["profiler"] = {"directory": str(profiler), "files": [
        {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(profiler.iterdir()) if path.is_file()
    ]}
    (staging / "local-artifacts.json").write_text(json.dumps(local_payloads, indent=2) + "\n")
    (staging / "archive-provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkout": str(ROOT), "destination": str(DEST),
        "archive_helper_sha256": digest(Path(__file__)), "copied_files": copied,
    }, indent=2) + "\n")

    timings = list(csv.DictReader((EVALUATION / "capture/timing.tsv").open(), delimiter="\t"))
    require(len(timings) == 7 and sum(int(row["frames"]) for row in timings) == 5316,
            "unexpected seven-sequence population")
    summary = dict(audit)
    summary.update({
        "archive_schema": "f02-f03-clean-integration/1",
        "certificate_scope": "23ade53 v1-motion/1 integration and standalone F02/scan sanitizer coverage",
        "accepted_checkout": str(ROOT),
        "current_component_equivalence_is_snapshot": True,
        "ground_truth_scoring": "UNEXECUTED; no new GT scoring or independent evaluation",
        "timing_scope": "single full-sequence observations; no ABBA or attributed speed ratio",
        "frames": 5316, "sequences": 7,
        "deploy_engine_rebuilt": False,
        "deploy_engine_original_build_provenance_retained": True,
        "original_development_evidence": "unchanged; source_dirty=true and formal_acceptance=false retain their original meaning",
        "excluded_payloads": "binaries, libraries, packets, tensors, output payloads, nsys reports and sqlite retained locally",
        "local_artifacts_manifest": "local-artifacts.json",
    })
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    current = equivalence.get("current_revision", "unknown")
    report = f"""# F02 tile fusion／F03 N16 clean-source 整合補驗

2026-11-05。**PASS，限乾淨 `{REVISION}` 與 `v1-motion/1`。**
本輪以新的隔離 checkout 重新建置並執行原 TRACKING §14.5／§17.3 gate，
另補 F02／scan 的完整獨立 suite 四種 sanitizer；`source_dirty=false`、`formal_acceptance=true`。
最終判定由[獨立 postflight](control/methods/postflight.py)與[closeout.json](control/capture/closeout.json)支持。

受測 profile 為舊 `v1-motion/1`（cuFFT motion），不是現行 G1／`v1-motion/2`。
現行 G1 的[正式報告](../../tracking-session/2026-11-05-v1-motion-2-acceptance-2/REPORT.md)
仍是完整 session memcheck **OPEN／NOT A FULL PASS**；本輪不關閉它。
元件 hashes 與核對時的 main `{current}` 相同，範圍僅為
[equivalence 清單](control/capture/component-equivalence.json)，不將此結論擴大為兩個完整 runtime 相同。

| 項目 | 結果與證據 |
| --- | --- |
| Clean source／build | CPU CTest 16/16、CUDA CTest 59/59；[原 gate capture](acceptance/capture/acceptance.txt) |
| Session 與生命週期 | 原固定 gate 的 packet／serial／pipelined、HR MOT17-10、session sanitizer 範圍通過；命令、exit、summary 均保留於 [acceptance/capture](acceptance/capture/) |
| HR Python references | 四張 RGB／FP32 全元素 bitwise 相同；[log](control/capture/hr-python-reference.log)、[manifest](hr-references/manifest.json) |
| F02 standalone | 72 cases；memcheck／initcheck／racecheck／synccheck 均零 errors／hazards，沒有 kernel filter 或 launch-count 限制 |
| Scan standalone | 四工具通過；正常及其他三工具 368 checks，initcheck 360 checks；完整 groups、output／status／stream／graph 等既有測試 |
| Sanitizer 正控制 | 四工具各 exit 99 且診斷相符，postflight 逐個獨立核對；不只採信原 acceptance.txt |
| HR 七序列 | 5,316 幀；7 SCDR、7 MOT、7 offline MOT 與 M4 **21/21 相同**；[expected hashes](references/m4-outputs.sha256)、[新 outputs](evaluation/capture/outputs.sha256) |
| 保管 | 原接受 CUDA binaries 在補充工作前先封存；後續不重建；source、binaries、inputs 與 references hashes 經核對；[summary.json](summary.json) |

initcheck 依原 harness 契約使用 `--no-held-stream`，只略過 host callback 擋住 stream 的測項；
一般執行與其他三工具仍執行該項。Scan 的 shuffle mask 正確性仍依既有數值與 group-uniform 契約，
不由 synccheck 零報告單獨推導。原 gate 的正控制條件可能未阻擋非 99 狀態；
本輪另外以 postflight 明確驗四個 exit=99 及預期診斷後才定最終 PASS。

本輪**未重跑 ABBA、未讀 GT 或重新計分、未重跑 S02／F04 semantic mutations**。
這是原整合 gate 的補驗，不重新核發 S02 的完整原始 primitive 證書。
全序列時間僅為單次觀測，不宣稱新的加速比或品質結果。
原 11/04 dirty-source 加速量測與品質重測保留其 `source_dirty=true`、`formal_acceptance=false` 身分，
原始報告、失敗嘗試與 bytes 均不改寫。部署 engine 未重建，其原 build provenance 亦保留。

完整 HR session racecheck、TensorRT／nvJPEG／cuFFT 內部 racecheck／synccheck、
完整 session initcheck及跨 GPU／driver／SDK 支援不由本輪 standalone／scoped 結果關閉。

## 重跑與封存

外層方法為 [run.sh](control/methods/run.sh)，GPU lock 與獨立 clean checkout 的執行方式
保留於 [control capture](control/capture/)；原受測 scripts 及契約放在
[source/contracts-and-runners](source/contracts-and-runners/)。
Runner 的輸出目錄必須全新；不可直接覆用本次已存在的目錄。
Postflight 在 campaign 結束後驗 exits、精確 summary、source／binary custody 與 21 個 M4 outputs。
Archive helper 只在 closeout PASS 後封存，不執行 build、測試或 GPU 工作。

本機 checkout：`{ROOT}`。大型 binaries／packets／tensors／outputs／nsys／sqlite
不複製入 Git；位置及 hashes 見 [local-artifacts.json](local-artifacts.json)。
CTest logs、CMake cache、compile commands 保留於 [build-metadata](build-metadata/)。
每個封存檔案以 `SHA256SUMS` 核對；[archive-provenance.json](archive-provenance.json)
記錄來源位置、複製檔案與 helper hash。

```sh
cd evidence/f02-f03/2026-11-05-clean-23ade53
sha256sum -c SHA256SUMS
```
"""
    (staging / "REPORT.md").write_text(report)
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(staging).as_posix()}\n" for path in files))
    for path in files:
        require(path.is_file() and not path.is_symlink(), f"invalid archive member: {path}")
    require(not DEST.exists(), "destination appeared during archive")
    staging.rename(DEST)
    print(f"Archived {len(files)} files plus SHA256SUMS: {DEST}")


if __name__ == "__main__":
    main()
