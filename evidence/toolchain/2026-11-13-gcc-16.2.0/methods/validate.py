#!/usr/bin/env python3
"""Revalidate saccade_v1 built with the pinned GCC 16.2.0.

Phases (run in order, each writes receipts under build/gcc16-work):
  build  configure + build the full-feature config and the CPU ASan/UBSan config,
         then G0: every project object file was compiled by GCC 16.2.0
  ctest  full CTest suite (G1) and the ASan/UBSan CPU suite (G2)
  eval   eval_run.py on the five benchmarks of the 2026-11-12 validation:
         G3 MOT bytes, G4 metrics, G5 speed status and FPS against 2026-11-12
"""
import csv, datetime, gzip, hashlib, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "build/gcc16-work"
GCC = Path.home() / ".local/opt/gcc-16.2.0"
TOOLCHAIN = "cmake/toolchain-gcc-16.2.0.cmake"
TRT = "/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk"
ENGINE = Path("/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine")
ENGINE_SHA = "8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876"
REF = ROOT / "evidence/eval-run/2026-11-12-validation"
BUNDLE = ROOT / "evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle"
DATA = Path("/home/ray/developer/ai/dataset")
CASES = [
    ("MOT17", [DATA / "MOT17/train" / s for s in ("MOT17-10-SDP", "MOT17-02-SDP", "MOT17-04-SDP")]),
    ("MOT20", [DATA / "MOT20/MOT20/train/MOT20-01"]),
    ("MOT15", [DATA / "MOT15/MOT15/train/ADL-Rundle-6"]),
    ("DanceTrack", [DATA / "DanceTrack/val/dancetrack0004"]),
    ("PP22", [DATA / "PersonPath22/mot_test_kf/uid_vid_00008.mp4"]),
]
KEYS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")
FULL = ["-DSACCADE_ENABLE_" + n + "=ON" for n in (
    "CUDA", "PREPROCESS", "POSTPROCESS", "MODEL_PRODUCER", "TENSORRT", "MODEL_TRT",
    "DETECTION_SESSION", "FLOW", "VIDEO", "NVDEC", "STREAMING")]
SAN = "-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
JOBS = str(os.cpu_count())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(name, argv, env=None, check=True, timeout="3600s"):
    argv = ["timeout", "-k", "30s", timeout, *map(str, argv)]
    started, t0 = now(), time.monotonic()
    dump(WORK / f"{name}.command.json", {"argv": argv, "cwd": str(ROOT), "started_utc": started,
                                         "env_overrides": env or {}})
    print("START", name, flush=True)
    with (WORK / f"{name}.log").open("wb") as log:
        r = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, **(env or {})})
    dump(WORK / f"{name}.result.json", {"returncode": r.returncode, "wall_seconds": time.monotonic() - t0,
                                        "completed_utc": now()})
    print("DONE", name, r.returncode, f"{time.monotonic() - t0:.0f}s", flush=True)
    if check and r.returncode:
        raise SystemExit(f"{name} failed (exit {r.returncode}); see {WORK / (name + '.log')}")
    return r.returncode


def out(argv):
    return subprocess.run(list(map(str, argv)), capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"}).stdout


class Telemetry:
    def __init__(self, name):
        self.path = WORK / f"{name}.telemetry.csv"

    def __enter__(self):
        self.f = self.path.open("w")
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,"
                                   "utilization.gpu,memory.used", "--format=csv", "-l", "1"], stdout=self.f)
        return self

    def __exit__(self, *exc):
        self.p.send_signal(signal.SIGTERM)
        self.p.wait()
        self.f.close()


def phase_build():
    identity = {
        "g++": {"path": str(GCC / "bin/g++"), "sha256": sha(GCC / "bin/g++"),
                "version": out([GCC / "bin/g++", "-v"]).strip() or
                subprocess.run([GCC / "bin/g++", "-v"], capture_output=True, text=True,
                               env={**os.environ, "LC_ALL": "C"}).stderr.strip()},
        "cc1plus": sha(out([GCC / "bin/g++", "-print-prog-name=cc1plus"]).strip()),
        "nvcc": out(["/opt/cuda/bin/nvcc", "--version"]).strip().splitlines()[-2:],
        "cmake": out(["cmake", "--version"]).splitlines()[0],
        "system_g++": out(["/usr/bin/g++", "--version"]).splitlines()[0],
        "NVCC_CCBIN_env": os.environ.get("NVCC_CCBIN"),
        "git_head": out(["git", "-C", ROOT, "rev-parse", "HEAD"]).strip(),
        "git_status_porcelain": out(["git", "-C", ROOT, "status", "--porcelain"]),
    }
    dump(WORK / "identity.json", identity)
    if identity["git_status_porcelain"]:
        raise SystemExit("worktree is dirty; STOP")
    common = ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    run("configure-full", ["cmake", "-S", ".", "-B", "build/gcc16", *common, *FULL,
                           f"-DSACCADE_TENSORRT_ROOT={TRT}", "-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc",
                           "-DCMAKE_CUDA_ARCHITECTURES=120"])
    run("build-full", ["cmake", "--build", "build/gcc16", "--parallel", JOBS])
    run("configure-asan", ["cmake", "-S", ".", "-B", "build/gcc16-asan", "-DCMAKE_BUILD_TYPE=Debug",
                           f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                           f"-DCMAKE_CXX_FLAGS={SAN}", "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"])
    run("build-asan", ["cmake", "--build", "build/gcc16-asan", "--parallel", JOBS])
    phase_g0()


def phase_g0():
    # G0: which compiler produced every project object file, and what nvcc was told.
    g0 = {}
    for build in ("build/gcc16", "build/gcc16-asan"):
        # With a toolchain file CMake records the compilers here, not in CMakeCache.txt.
        detected = "".join(p.read_text() for p in (ROOT / build).glob("CMakeFiles/*/CMake*Compiler.cmake"))
        pick = lambda key: (re.search(rf'^set\({key} "([^"]*)"\)', detected, re.M) or [None, None])[1]
        comments, missing = {}, []
        objects = sorted((ROOT / build).glob("**/CMakeFiles/**/*.o"))
        for obj in objects:
            text = out(["readelf", "-p", ".comment", obj])
            found = sorted(set(re.findall(r"GCC: \([^)]*\) [^\n]+", text)))
            if not found:
                missing.append(str(obj.relative_to(ROOT)))
            for s in found:
                comments.setdefault(s.strip(), 0)
                comments[s.strip()] += 1
        cc = json.loads((ROOT / build / "compile_commands.json").read_text())
        cu = [c for c in cc if c["file"].endswith(".cu")]
        cpp = [c for c in cc if not c["file"].endswith(".cu")]
        ccbin_ok = all(f"-ccbin={GCC}/bin/g++" in c["command"] for c in cu)
        cxx_ok = all(c["command"].startswith(f"{GCC}/bin/g++ ") for c in cpp)
        g0[build] = {
            "CMAKE_CXX_COMPILER": pick("CMAKE_CXX_COMPILER"),
            "CMAKE_CUDA_HOST_COMPILER": pick("CMAKE_CUDA_HOST_COMPILER"),
            "object_files": len(objects), "comment_strings": comments, "objects_without_gcc_comment": missing,
            "compile_commands": {"cpp": len(cpp), "cu": len(cu), "cpp_use_pinned_gxx": cxx_ok,
                                 "cu_ccbin_pinned_gxx": ccbin_ok},
            "pass": set(comments) == {"GCC: (GNU) 16.2.0"} and not missing and cxx_ok and ccbin_ok
                    and pick("CMAKE_CXX_COMPILER") == f"{GCC}/bin/g++"
                    and (not cu or pick("CMAKE_CUDA_HOST_COMPILER") == f"{GCC}/bin/g++"),
        }
    track = ROOT / "build/gcc16/track_sequence"
    g0["runtime_libraries_track_sequence"] = [l.strip() for l in out(["ldd", track]).splitlines()
                                              if re.search(r"libstdc\+\+|libgcc_s|libnvinfer|libcudart", l)]
    # Effective defaults at the project's Release level must match the system GCC 16.2.1.
    defaults = {}
    for kind in ("target", "common", "optimizers", "c++", "warnings"):
        # GCC prints a random temporary file name for -o; mask it.
        q = lambda gxx: [re.sub(r"/tmp/cc\w+\.s", "/tmp/cc*.s", l)
                         for l in out([gxx, "-O3", "-Q", f"--help={kind}"]).splitlines()]
        mine, system = q(GCC / "bin/g++"), q("/usr/bin/g++")
        defaults[kind] = {"lines": len(mine), "system_lines": len(system),
                          "only_16_2_0": sorted(set(mine) - set(system)),
                          "only_system": sorted(set(system) - set(mine))}
    defaults["pass"] = all(v["lines"] > 0 and v["system_lines"] > 0 and not v["only_16_2_0"]
                           and not v["only_system"] for k, v in defaults.items() if k != "pass")
    g0["default_options_vs_system_O3"] = defaults
    g0["pass"] = defaults["pass"] and all(v["pass"] for k, v in g0.items() if k.startswith("build/"))
    dump(WORK / "G0-compiler-identity.json", g0)
    print("G0", "PASS" if g0["pass"] else "FAIL", flush=True)


def phase_ctest():
    with Telemetry("ctest-full"):
        rc = run("ctest-full", ["ctest", "--test-dir", "build/gcc16", "--output-on-failure", "--timeout", "1800",
                                "--output-junit", WORK / "ctest-full.junit.xml"], check=False, timeout="7200s")
    rc_asan = run("ctest-asan", ["ctest", "--test-dir", "build/gcc16-asan", "--output-on-failure",
                                 "--timeout", "1800", "--output-junit", WORK / "ctest-asan.junit.xml"],
                  env={"LD_LIBRARY_PATH": f"{GCC}/lib64"}, check=False, timeout="7200s")
    summarize_ctest()


def summarize_ctest():
    summary = {}
    for name in ("full", "asan"):
        rc = json.loads((WORK / f"ctest-{name}.result.json").read_text())["returncode"]
        log = (WORK / f"ctest-{name}.log").read_text(errors="replace")
        # CMake 4.x omits ", 0 tests failed" when nothing fails.
        m = re.search(r"(\d+)% tests passed(?:, (\d+) tests failed)? out of (\d+)", log)
        passed = re.findall(r"Test +#\d+: (\S+) \.+ +Passed", log)
        summary[name] = {"returncode": rc, "line": m.group(0) if m else None,
                         "total": int(m.group(3)) if m else None, "passed_tests": passed,
                         "failed": re.findall(r"^\s*\d+ - (\S+) \(", log, re.M),
                         "pass": rc == 0 and m is not None and m.group(1) == "100"
                                 and len(passed) == int(m.group(3))}
    dump(WORK / "G1-G2-ctest.json", summary)
    print("CTEST", json.dumps({k: v["line"] for k, v in summary.items()}), flush=True)


def phase_eval():
    ref_parity = json.loads((REF / "parity.json").read_text())
    expected_mot = {g["sequence"]: g["expected"] for g in ref_parity["G1"]}
    with (BUNDLE / "metrics/per-sequence.csv").open() as f:
        expected_metrics = {r["sequence"]: r for r in csv.DictReader(f) if r["arm"] == "main"}
    frozen = {s["id"]: s for s in json.loads((BUNDLE / "methods/scoring-tools/selection-frozen.json")
                                             .read_text())["sequences"]}
    checks = [("engine", sha(ENGINE), ENGINE_SHA),
              ("per-sequence.csv", sha(BUNDLE / "metrics/per-sequence.csv"),
               "92362c7de4698f2f70d872243d165756f9c5869e7b55f1d8820e117fba510f08")]
    for _, seqs in CASES:
        for seq in seqs:
            for key, rel in (("ground_truth", "gt/gt.txt"), ("sequence_info", "seqinfo.ini")):
                checks.append((f"{seq.name}/{key}", sha(seq / rel), frozen[seq.name][key]["sha256"]))
    driver = out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip()
    pre = {"checks": [{"input": a, "actual": b, "expected": c, "pass": b == c} for a, b, c in checks],
           "driver": {"actual": driver, "reference_2026_11_12": "621.14"},
           "git_status_porcelain": out(["git", "-C", ROOT, "status", "--porcelain"])}
    pre["pass"] = all(c["pass"] for c in pre["checks"]) and not pre["git_status_porcelain"]
    dump(WORK / "eval-preflight.json", pre)
    if not pre["pass"]:
        raise SystemExit("eval preflight failed; STOP")

    result = {"started_utc": now(), "G3_mot_bytes": [], "G4_metrics": [], "G5_speed": []}
    with Telemetry("eval"):
        for benchmark, seqs in CASES:
            dest = f"build/gcc16-eval-{benchmark}"
            run(f"eval-{benchmark}", ["python3", "-I", "-B", "scripts/eval_run.py", "--benchmark", benchmark,
                                      "--build", "build/gcc16", "--out", dest, *seqs])
            rows = {r["sequence"]: r for r in json.loads((ROOT / dest / "results.json").read_text())}
            env = json.loads((ROOT / dest / "environment.json").read_text())
            ref_rows = {r["sequence"]: r for r in json.loads((REF / "runs" / benchmark / "results.json").read_text())}
            for seq in seqs:
                actual = sha(ROOT / dest / "quality" / seq.name / "mot.txt")
                result["G3_mot_bytes"].append({"sequence": seq.name, "expected": expected_mot[seq.name],
                                               "actual": actual, "pass": actual == expected_mot[seq.name]})
                metrics = {}
                for k in KEYS:
                    exp = int(expected_metrics[seq.name][k]) if k == "IDSW" else float(expected_metrics[seq.name][k])
                    got = rows[seq.name][k]
                    metrics[k] = {"expected": exp, "actual": got, "pass": got == exp}
                result["G4_metrics"].append({"sequence": seq.name, "metrics": metrics,
                                             "pass": all(m["pass"] for m in metrics.values())})
                speed = env["speed"][seq.name]
                ref, cur = ref_rows[seq.name], rows[seq.name]
                result["G5_speed"].append({
                    "sequence": seq.name, "status": speed["status"],
                    "latency_samples": speed["graph"]["latency_samples"],
                    "compare_2026_11_12": {k: {"reference": ref[k], "gcc_16_2_0": cur[k],
                                               "difference_percent": 100 * (cur[k] - ref[k]) / ref[k]}
                                           for k in ("queued_fps", "single_mean_ms", "single_p95_ms", "single_p99_ms")},
                    "pass": speed["status"] == "PASS"})
            result.setdefault("environment", {})[benchmark] = {
                "git_head": env["git_head"], "git_status_porcelain": env["git_status_porcelain"],
                "files": env["files"], "status": env["status"],
                "source_dirty": sorted({d for c in env["captures"].values() for d in c["source_dirty"]})}
    for g in ("G3_mot_bytes", "G4_metrics", "G5_speed"):
        result[g + "_pass"] = all(x["pass"] for x in result[g])
    result["completed_utc"] = now()
    dump(WORK / "G3-G5-eval.json", result)
    print("EVAL", {g: result[g + "_pass"] for g in ("G3_mot_bytes", "G4_metrics", "G5_speed")}, flush=True)


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    {"build": phase_build, "g0": phase_g0, "ctest": phase_ctest, "ctest-summary": summarize_ctest,
     "eval": phase_eval}[sys.argv[1]]()
