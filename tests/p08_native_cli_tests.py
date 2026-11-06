#!/usr/bin/env python3
"""P08 native CLI contract: static rejection, immutable metadata, receipts and byte equivalence.

Run under the project GPU lock. VIDEO is an existing nine-frame CPU-decodable
fixture from track_stream_cli_tests.py. No ground truth is opened.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import unquote


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("track_stream", "track_sequence", "tracking_config", "engine", "sequence", "video"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root.resolve()))
    print(f"P08 native artifacts: {output}", flush=True)
    results = []
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("SACCADE_CFG_")}

    def run(name, command, expected=0, overrides=None):
        env = clean_env | (overrides or {})
        result = subprocess.run(list(map(str, command)), env=env, capture_output=True, timeout=180)
        (output / f"{name}.stdout").write_bytes(result.stdout)
        (output / f"{name}.stderr").write_bytes(result.stderr)
        results.append({"case": name, "returncode": result.returncode})
        check(result.returncode == expected, f"{name}: exit {result.returncode}, see retained logs")
        return result

    def stream(name, *flags, source=None, expected=0, overrides=None):
        mot, packets, receipt = (output / f"{name}.{suffix}" for suffix in ("txt", "bin", "config"))
        result = run(name, [args.track_stream, args.engine, source or args.video, mot, "--decode", "cpu",
                           "--packets", packets, "--resolved-config", receipt, *flags], expected, overrides)
        if expected:
            check(not mot.exists() and not packets.exists() and not receipt.exists(),
                  f"{name}: static rejection created outputs")
        return result, mot, packets, receipt

    # These must fail before attempting to open the absent source.
    missing = output / "missing-video.mkv"
    for name, flags, diagnostic in (
        ("unknown-field", ["--set", "tracker.missing=1"], b"unknown"),
        ("duplicate-alias", ["--serial", "--set", "execution.mode=serial"], b"duplicate"),
        ("wrong-bool", ["--set", "execution.cuda_graph=1"], b"execution.cuda_graph"),
        ("cuda-events-unavailable", ["--set", "observer.event_dump=true"], b"observer.event_dump"),
        ("invalid-backend", ["--set", "implementation.backend=reference"], b"implementation"),
        ("invalid-dependency", ["--set", "tracker.assoc.high_thresh=0.01"], b"threshold"),
    ):
        result, *_ = stream(name, *flags, source=missing, expected=1)
        check(diagnostic in result.stderr, f"{name}: did not report typed configuration rejection")

    secret = "untrusted-env-value-must-not-appear"
    result, *_ = stream("unknown-environment", source=missing, expected=1,
                         overrides={"SACCADE_CFG_UNKNOWN": secret})
    check(b"undeclared" in result.stderr and secret.encode() not in result.stdout + result.stderr,
          "unknown environment rejection disclosed its value")
    bad = output / "missing-schema.conf"
    bad.write_text("execution.mode=serial\n")
    result, *_ = stream("required-schema", "--config", bad, source=missing, expected=1)
    check(b"schema" in result.stderr, "missing file schema not rejected before source open")

    protected = output / "existing-receipt.config"
    protected.write_text("must remain unchanged\n")
    absent_output = output / "must-not-create.txt"
    run("existing-receipt", [args.track_stream, args.engine, missing, absent_output,
                             "--resolved-config", protected], 1)
    check(protected.read_text() == "must remain unchanged\n" and not absent_output.exists(),
          "existing receipt was overwritten or created MOT")
    run("receipt-mot-collision", [args.track_stream, args.engine, missing, absent_output,
                                  "--resolved-config", absent_output], 1)
    check(not absent_output.exists(), "receipt/output collision created output")

    def receipt(path, executable):
        lines = [line.split("=", 1) for line in path.read_text().splitlines()]
        data = dict(lines)
        check(len(data) == len(lines), "duplicate receipt keys")
        check(data["record.schema"] == "saccade-resolved-record/1", "record schema")
        build = dict(line.split("=", 1) for line in unquote(data["build.identity"]).splitlines())
        check(build["build.engine_sha256"] == digest(args.engine), "engine identity separate from config")
        check(build["build.executable_sha256"] == digest(executable), "binary identity separate from config")
        check(build["build.source_dirty"] in ("true", "false", "unknown"), "source provenance missing")
        run(path.stem + "-record-self-compare", [args.tracking_config, "compare", path, path])
        return data

    _, baseline_mot, baseline_packets, baseline_receipt = stream("default")
    base = receipt(baseline_receipt, args.track_stream)
    check(base["implementation.model_profile"] == "hr-1280x768" and
          base["execution.result_slots"] == "1" and base["output.interpolate_max_gap"] == "0",
          "legacy stream defaults changed")
    check(base["origin.source.width"] == "derived:decoded-frame-metadata", "source metadata provenance")

    config = output / "precedence.conf"
    config.write_text("schema=saccade-tracking-config/1\nexecution.mode=pipelined\n")
    _, equivalent_mot, equivalent_packets, equivalent_receipt = stream(
        "precedence", "--config", config, "--set", "execution.mode=pipelined",
        overrides={"SACCADE_CFG_EXECUTION_MODE": "serial"})
    equivalent = receipt(equivalent_receipt, args.track_stream)
    check(base["config.sha256"] == equivalent["config.sha256"], "equivalent values changed config hash")
    check(equivalent["origin.execution.mode"] == "cli:execution.mode", "CLI precedence/provenance")
    check(baseline_packets.read_bytes() == equivalent_packets.read_bytes() and
          baseline_mot.read_bytes() == equivalent_mot.read_bytes(), "configuration observation changed output bytes")
    run("receipt-compare", [args.tracking_config, "compare", baseline_receipt, equivalent_receipt])

    # A source-owned width cannot silently be replaced by config overrides.
    wrong_width = str(int(base["source.width"]) + 8)
    stream("source-width-mismatch", "--set", "source.width=" + wrong_width, expected=1)

    # Check track_sequence with a private nine-frame metadata view. Image bytes
    # are borrowed read-only from the original dataset.
    short_sequence = output / "sequence"
    short_sequence.mkdir()
    (short_sequence / "img1").symlink_to((args.sequence / "img1").resolve(), target_is_directory=True)
    info = (args.sequence / "seqinfo.ini").read_text()
    info = re.sub(r"(?m)^seqLength\s*=.*$", "seqLength=9", info)
    (short_sequence / "seqinfo.ini").write_text(info)
    seq_mot, seq_packets, seq_receipt = (output / f"sequence.{suffix}" for suffix in ("txt", "bin", "config"))
    run("sequence-config", [args.track_sequence, args.engine, short_sequence, seq_mot,
                            "--profile", "hr-1280x768", "--packets", seq_packets,
                            "--resolved-config", seq_receipt])
    sequence_data = receipt(seq_receipt, args.track_sequence)
    check(sequence_data["output.profile"] == "v1-mot/1" and sequence_data["output.interpolate_max_gap"] == "35",
          "legacy offline output policy changed")
    check(sequence_data["origin.source.width"] == "derived:seqinfo.ini", "sequence metadata provenance")
    check(len(seq_packets.read_bytes()) == 9 * 12368, "sequence packet coverage")
    (output / "summary.json").write_text(json.dumps({"pass": True, "cases": results}, indent=2) + "\n")
    print(f"P08 native CLI PASS: {len(results)} cases; artifacts={output}")


if __name__ == "__main__":
    main()
