#!/usr/bin/env python3
"""Read-only symlink and artifact-path inventory; never follow linked directories."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat


def scan(root: Path) -> dict:
    links, errors = [], []
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"scan root must be an existing real directory: {root}")
    for parent, dirs, files in os.walk(root, followlinks=False,
                                       onerror=lambda error: errors.append(str(error))):
        for name in sorted(dirs + files):
            path = Path(parent) / name
            if not path.is_symlink():
                continue
            try:
                target = os.readlink(path)
                resolved = Path(os.path.realpath(path))
                try:
                    mode = path.stat().st_mode
                    exists = True
                    kind = ("directory" if stat.S_ISDIR(mode) else
                            "file" if stat.S_ISREG(mode) else "device_or_other")
                except FileNotFoundError:
                    exists, kind = False, "missing"
                links.append({"path": str(path.relative_to(root)), "target": target,
                              "resolved": str(resolved), "exists": exists, "kind": kind,
                              "external": not resolved.is_relative_to(root),
                              "absolute": os.path.isabs(target)})
            except OSError as error:
                errors.append(f"{path}: {error}")
    links.sort(key=lambda link: link["path"])
    return {"root": str(root), "summary": {
        "total": len(links),
        "external": sum(link["external"] for link in links),
        "broken": sum(not link["exists"] for link in links),
        "absolute": sum(link["absolute"] for link in links),
        "external_targets": dict(sorted(Counter(
            link["resolved"] for link in links if link["external"]).items())),
    }, "links": links, "errors": errors}


def reference(name: str, path: Path, role: str = "input") -> dict:
    components = []
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            components.append({"path": str(part), "target": os.readlink(part)})
    return {"name": name, "path": str(path), "role": role,
            "exists": path.exists(), "resolved": os.path.realpath(path),
            "symlink_components": components}


def inventory(repo: Path, include_artifacts: bool) -> dict:
    home = Path.home()
    cache = home / ".cache/saccade-v1"
    t01 = Path(os.environ.get("SACCADE_T01_ROOT", str(cache / "t01"))).expanduser().absolute()
    legacy = Path(os.environ.get("SACCADE_LEGACY_ROOT",
                                 str(home / "developer/ai/saccade"))).expanduser().absolute()
    worktrees = home / "developer/ai/saccade_v1-worktrees"
    research = home / "developer/ai/saccade_v1_dev"
    sdk = Path(os.environ.get("SACCADE_TENSORRT_ROOT",
                             str(home / ".cache/saccade-native-trt/10.16.1.11/sdk"))).expanduser().absolute()
    deploy = Path(os.environ.get("SACCADE_HR_DEPLOY", str(cache / "hr-deploy"))).expanduser().absolute()
    scopes = {"repository": scan(repo)}
    if include_artifacts:
        for name, root in (("s04", cache / "s04"), ("t01", t01),
                           ("hr_deploy", deploy), ("tensorrt_sdk", sdk)):
            if root.is_dir() and not root.is_symlink():
                scopes[name] = scan(root)
            else:
                scopes[name] = {"root": str(root), "skipped": "missing or symlinked scan root"}
    refs = [("legacy", legacy), ("research", research), ("t01", t01), ("sdk", sdk),
            ("hr_engine", deploy / "model-hr-fp16.engine"),
            ("base_weights", legacy / "models/yolo/yolo26m.pt"),
            ("mot17", legacy / "datasets/MOT17"),
            ("personpath22", legacy / "datasets/PersonPath22/mot_test_kf"),
            ("trackeval", legacy / "third_party/TrackEval"),
            ("evaluation_python", legacy / ".venv/bin/python"),
            ("t01_engine", repo / "build/t01-f03-evidence/stage/model.engine"),
            ("detection_references", repo / "build/preprocess-evidence/refs"),
            ("t01_records", worktrees / "m1-data-v2"),
            ("hr_records", worktrees / "hr-records-v2"),
            ("m4_records", worktrees / "m4-formal/build-m4-evaluation/scdr"),
            ("flow_records", worktrees / "gmc-flow/build/gmc-flow/flow"),
            ("g1_mot", worktrees / "gmc-flow/build/gmc-flow/arms/G1"),
            ("pp22_mot", worktrees / "gmc-flow/build/gmc-flow2/pp22/arms/G1"),
            ("pp22_population", worktrees / "gmc-flow/build/gmc-flow2/pp22/included.txt")]
    references = [reference(name, path) for name, path in refs]
    references.append(reference("acceptance_baseline", worktrees / "g1-ofa-base",
                                "output_created_by_acceptance_step_8"))
    manifest_path = deploy / "manifest.json"
    history = {"manifest": str(manifest_path), "exists": manifest_path.is_file(),
               "scope": "historical provenance paths; absence is not runtime failure"}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        history["sources"] = [reference(path, Path(path), "historical_source")
                              for path in sorted(manifest.get("sources", {}))]
        references.extend(reference(name, Path(manifest[name]))
                          for name in ("teacher", "student") if name in manifest)
    return {"schema": "saccade-external-path-audit/1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "symlinks and named local dependencies; no payload hashing or acceptance",
            "scopes": scopes, "references": references, "hr_manifest": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-artifacts", action="store_true",
                        help="also scan S04, T01, HR deploy and TensorRT SDK roots")
    args = parser.parse_args()
    try:
        report = inventory(args.repo.resolve(strict=True), args.include_artifacts)
    except (OSError, ValueError) as error:
        parser.exit(2, f"audit_external_paths: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # Historical source paths and outputs that will be created are informational.
    # Input presence is reported separately; this is not an acceptance verdict.
    return int(any(scope.get("errors") or scope.get("summary", {}).get("broken")
                   for scope in report["scopes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
