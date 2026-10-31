"""Offline F01/F02 references and comparisons, t01-preprocess-fp32/1.

    venv-export/bin/python training/t01/t01_preprocess_ref.py generate --out build-preprocess-ref
    venv-export/bin/python training/t01/t01_preprocess_ref.py compare --refs build-preprocess-ref --dump NATIVE_DIR

The native runtime has no Python dependency. cases.tsv is UTF-8, one case per
line (comment header), with tab-separated label, input width, input height,
output width, output height, JPEG path or -, planar RGB-u8 path, CHW-f32 path.
Paths are relative to the reference directory. Native dumps use case-NNN.rgb
and case-NNN.f32. Artifacts include SHA-256 and the policy fixed before native
GPU comparisons. No native GPU result is used to select cases or tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import decode_jpeg

from t01_common import REPO, load_recipe, sha256_file, write_json

POLICY = "t01-preprocess-fp32/1"
REAL_FRAMES = (("MOT17-02-SDP", 1), ("MOT17-10-SDP", 1), ("MOT17-13-SDP", 400),
               ("MOT17-04-SDP", 1), ("MOT17-05-SDP", 1), ("MOT17-09-SDP", 1),
               ("MOT17-11-SDP", 1))
MUTATIONS = ("no_antialias", "bgr", "hwc_as_chw", "no_normalize", "uint8_rounding", "align_corners")


def tolerance(width: int, height: int, output_width: int, output_height: int) -> float:
    nx = math.ceil(2 * max(width / output_width, 1)) + 1
    ny = math.ceil(2 * max(height / output_height, 1)) + 1
    nu = (4 * (nx + ny) + 24) * 2.0**-24
    return 2 * nu / (1 - nu)


def resize(rgb: torch.Tensor, output_width: int, output_height: int) -> torch.Tensor:
    return F.interpolate(rgb.float().div(255)[None], size=(output_height, output_width),
                         mode="bilinear", antialias=True, align_corners=False)[0]


def ppm_rgb(payload: bytes) -> np.ndarray:
    """Parse djpeg's binary PPM without consuming whitespace-valued pixels."""
    pos = 0

    def token() -> bytes:
        nonlocal pos
        while pos < len(payload):
            if payload[pos] in b" \r\n\t":
                pos += 1
            elif payload[pos] == ord("#"):
                pos = payload.index(b"\n", pos) + 1
            else:
                break
        start = pos
        while pos < len(payload) and payload[pos] not in b" \r\n\t":
            pos += 1
        return payload[start:pos]

    if token() != b"P6":
        raise ValueError("djpeg did not produce binary RGB PPM")
    width, height, maximum = int(token()), int(token()), int(token())
    if maximum != 255 or pos >= len(payload) or payload[pos] not in b" \r\n\t":
        raise ValueError("invalid 8-bit PPM header")
    pos += 2 if payload[pos:pos + 2] == b"\r\n" else 1
    data = np.frombuffer(payload[pos:], dtype=np.uint8)
    if data.size != width * height * 3:
        raise ValueError("invalid PPM raster length")
    return data.reshape(height, width, 3).transpose(2, 0, 1).copy()


def synthetic_cases() -> list[tuple[str, np.ndarray, int, int]]:
    rng = np.random.default_rng(20261031)
    cases: list[tuple[str, np.ndarray, int, int]] = []
    cases.append(("zeros-singleton", np.zeros((3, 1, 1), np.uint8), 640, 640))
    cases.append(("ones-upsample", np.full((3, 41, 73), 255, np.uint8), 640, 640))
    y, x = np.indices((23, 37))
    ramp = np.stack(((x * 7 + y * 3) % 256, (x * 2 + 91) % 256, (y * 11 + 21) % 256)).astype(np.uint8)
    cases.append(("color-ramp", ramp, 640, 640))
    impulse = np.zeros((3, 17, 13), np.uint8)
    impulse[0, 0, 0], impulse[1, -1, -1], impulse[2, 8, 6] = 255, 201, 117
    cases.append(("edge-impulse", impulse, 7, 11))
    y, x = np.indices((721, 1281))
    checker = np.stack(((x + y) % 2 * 255, x % 2 * 239, y % 2 * 173)).astype(np.uint8)
    cases.append(("checker-odd-downsample", checker, 640, 640))
    y, x = np.indices((53, 81))
    stripe = np.stack((x % 3 * 97, y % 3 * 71, (x + 2 * y) % 5 * 51)).astype(np.uint8)
    cases.append(("stripes", stripe, 31, 19))
    cases.append(("random-downsample", rng.integers(0, 256, (3, 79, 103), np.uint8), 31, 29))
    cases.append(("identity", rng.integers(0, 256, (3, 11, 17), np.uint8), 17, 11))
    cases.append(("single-row", rng.integers(0, 256, (3, 1, 17), np.uint8), 7, 3))
    cases.append(("single-column", rng.integers(0, 256, (3, 19, 1), np.uint8), 3, 7))
    return cases


def mutation_outputs(rgb: torch.Tensor, ow: int, oh: int) -> dict[str, torch.Tensor]:
    x = rgb.float().div(255)[None]
    opts = {"size": (oh, ow), "mode": "bilinear", "align_corners": False, "antialias": True}
    result = {
        "no_antialias": F.interpolate(x, **{**opts, "antialias": False})[0],
        "bgr": F.interpolate(x[:, [2, 1, 0]], **opts)[0],
        "hwc_as_chw": F.interpolate(x.permute(0, 2, 3, 1).contiguous().reshape(x.shape), **opts)[0],
        "no_normalize": F.interpolate(rgb.float()[None], **opts)[0],
        "uint8_rounding": F.interpolate(rgb.float()[None], **opts).round().clamp(0, 255).div(255)[0],
        "align_corners": F.interpolate(x, **{**opts, "antialias": False, "align_corners": True})[0],
        "normalize_after_resize": F.interpolate(rgb.float()[None], **opts).div(255)[0],
    }
    return result


def generate(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"reference output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    policy_path = REPO / "docs" / "PREPROCESS_POLICY.md"
    shutil.copyfile(policy_path, out / "PREPROCESS_POLICY.md")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA reference generation requires the pinned T01 GPU environment")
    if not torch.__version__.startswith("2.11.0") or not torchvision.__version__.startswith("0.26.0"):
        raise SystemExit("expected pinned torch 2.11.0 and torchvision 0.26.0")
    djpeg = shutil.which(args.djpeg)
    if not djpeg:
        raise SystemExit("djpeg is required for the independent -dct int -nosmooth reference")
    djpeg_version = subprocess.run([djpeg, "-version"], check=True, capture_output=True, text=True)
    dataset = args.dataset or Path(load_recipe()["dataset"]["root"])
    paths = [dataset / "train" / seq / "img1" / f"{frame:06d}.jpg" for seq, frame in REAL_FRAMES]
    streams = [torch.from_numpy(np.frombuffer(p.read_bytes(), dtype=np.uint8).copy()) for p in paths]
    decoded = decode_jpeg(streams, device="cuda")
    repeated = decode_jpeg(streams, device="cuda")
    cases: list[dict[str, Any]] = []
    rows: list[str] = ["# label\twidth\theight\toutput_width\toutput_height\tjpeg\trgb\tf32"]
    detections = {name: 0 for name in MUTATIONS}

    def emit(label: str, rgb: torch.Tensor, ow: int, oh: int, jpeg: Path | None) -> None:
        index = len(cases)
        stem = f"case-{index:03d}"
        height, width = rgb.shape[1:]
        expected = resize(rgb, ow, oh)
        again = resize(rgb, ow, oh)
        if not torch.equal(expected, again) or not bool(torch.isfinite(expected).all()):
            raise AssertionError(f"reference not repeatable/finite: {label}")
        rgb_path, f32_path = f"{stem}.rgb", f"{stem}.f32"
        rgb.cpu().contiguous().numpy().tofile(out / rgb_path)
        expected.cpu().contiguous().numpy().astype("<f4").tofile(out / f32_path)
        jpeg_path = "-"
        if jpeg is not None:
            jpeg_path = f"{stem}.jpg"
            shutil.copyfile(jpeg, out / jpeg_path)
        atol = tolerance(width, height, ow, oh)
        mutations = {}
        for name, mutant in mutation_outputs(rgb, ow, oh).items():
            diff = (mutant - expected).abs()
            bad = int((diff > atol).sum())
            mutations[name] = {"max_abs": float(diff.max()), "above_bound": bad}
            if name in detections:
                detections[name] += bad
        payload = {"index": index, "label": label, "width": width, "height": height,
                   "output_width": ow, "output_height": oh, "jpeg": jpeg_path, "rgb": rgb_path,
                   "f32": f32_path, "atol": atol, "range": [float(expected.min()), float(expected.max())],
                   "source_jpeg": str(jpeg) if jpeg else None, "mutations": mutations,
                   "hashes": {path: sha256_file(out / path) for path in (rgb_path, f32_path)}}
        if jpeg is not None:
            payload["hashes"][jpeg_path] = sha256_file(out / jpeg_path)
        cases.append(payload)
        rows.append("\t".join(map(str, (label, width, height, ow, oh, jpeg_path, rgb_path, f32_path))))
        print(f"{stem} {label}: {width}x{height}->{ow}x{oh}, atol={atol:.9g}", flush=True)

    for (seq, frame), path, stream, rgb, other in zip(REAL_FRAMES, paths, streams, decoded, repeated, strict=True):
        single = decode_jpeg(stream, device="cuda")
        if not torch.equal(rgb, single) or not torch.equal(rgb, other):
            raise AssertionError(f"batched/single/repeat nvJPEG mismatch: {path}")
        cpu = ppm_rgb(subprocess.run([djpeg, "-dct", "int", "-nosmooth", "-rgb", str(path)],
                                     check=True, capture_output=True).stdout)
        if not np.array_equal(rgb.cpu().numpy(), cpu):
            raise AssertionError(f"nvJPEG vs libjpeg-turbo islow/nosmooth mismatch: {path}")
        emit(f"{seq}/{frame}", rgb, 640, 640, path)
    for label, rgb, ow, oh in synthetic_cases():
        emit(label, torch.from_numpy(rgb).cuda(), ow, oh, None)
    if any(count == 0 for count in detections.values()):
        raise AssertionError(f"undetected semantic mutations: {detections}")
    (out / "cases.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    libs = sorted((Path(torchvision.__file__).resolve().parent.parent / "torchvision.libs").glob("libnvjpeg*.so*"))
    manifest = {
        "schema": "saccade-v1-preprocess-reference/1", "policy": POLICY,
        "policy_sha256": sha256_file(policy_path), "reference_script_sha256": sha256_file(Path(__file__)),
        "cases_tsv_sha256": sha256_file(out / "cases.tsv"), "torch": torch.__version__,
        "torch_git": torch.version.git_version, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "nvjpeg_libraries": [{"path": str(p), "sha256": sha256_file(p)} for p in libs],
        "djpeg": {"path": djpeg, "sha256": sha256_file(Path(djpeg)),
                  "version": (djpeg_version.stdout + djpeg_version.stderr).strip()},
        "decode_checks": "all 7 JPEGs: batched == single == repeat == djpeg -dct int -nosmooth -rgb, bitwise",
        "mutation_elements_above_bound": detections, "cases": cases,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"reference": str(out), "cases": len(cases), "mutations": detections}, sort_keys=True))


def compare(args: argparse.Namespace) -> None:
    manifest = json.loads((args.refs / "manifest.json").read_text())
    if manifest["policy"] != POLICY:
        raise SystemExit("unsupported preprocessing policy")
    if sha256_file(args.refs / "PREPROCESS_POLICY.md") != manifest["policy_sha256"]:
        raise SystemExit("reference policy snapshot hash mismatch")
    results = []
    passed = True
    for case in manifest["cases"]:
        for path, expected_hash in case["hashes"].items():
            if sha256_file(args.refs / path) != expected_hash:
                raise SystemExit(f"reference artifact hash mismatch: {path}")
        rgb_native = args.dump / case["rgb"]
        rgb_exact = rgb_native.read_bytes() == (args.refs / case["rgb"]).read_bytes()
        native = np.fromfile(args.dump / case["f32"], dtype="<f4")
        expected = np.fromfile(args.refs / case["f32"], dtype="<f4")
        if native.shape != expected.shape:
            raise SystemExit(f"wrong native tensor length: {case['label']}")
        finite = bool(np.isfinite(native).all())
        diff = np.abs(native.astype(np.float64) - expected.astype(np.float64))
        bad = int(np.count_nonzero(diff > case["atol"]))
        ok = rgb_exact and finite and bad == 0
        passed &= ok
        worst = int(np.argmax(diff))
        row = {"label": case["label"], "pass": ok, "rgb_bitwise": rgb_exact, "finite": finite,
               "elements": int(expected.size), "above_bound": bad, "atol": case["atol"],
               "max_abs": float(np.max(diff)), "mean_abs": float(np.mean(diff)),
               "worst_chw": list(map(int, np.unravel_index(worst, (3, case["output_height"], case["output_width"])))),
               "native_range": [float(native.min()), float(native.max())],
               "native_rgb_sha256": sha256_file(rgb_native),
               "native_f32_sha256": sha256_file(args.dump / case["f32"])}
        results.append(row)
        print(json.dumps(row, sort_keys=True))
    report = {"policy": POLICY, "policy_sha256": manifest["policy_sha256"], "pass": passed, "cases": results}
    write_json(args.report or args.dump / "comparison.json", report)
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument("--dataset", type=Path)
    generate_parser.add_argument("--djpeg", default="djpeg")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--refs", type=Path, required=True)
    compare_parser.add_argument("--dump", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    {"generate": generate, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
