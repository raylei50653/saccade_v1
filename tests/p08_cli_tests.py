#!/usr/bin/env python3
"""Exercise the real P08 CLIs with public SCDR bytes and independent comparisons."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()
    bins = args.bin_dir.resolve()
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("SACCADE_CFG_")}
    checks = 0

    def check(value, label):
        nonlocal checks
        checks += 1
        if not value:
            raise AssertionError(label)

    def run(tool, *arguments, ok=True, env=None):
        result = subprocess.run([str(bins / tool), *map(str, arguments)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env or clean_env, timeout=30)
        check((result.returncode == 0) == ok,
              f"{tool} exit {result.returncode}: {result.stderr}\n{result.stdout[-2000:]}")
        return result

    def fixture(path, score=0.8, bad=False):
        frames = []
        for frame in range(6):
            header = struct.pack("<QQ12I", 17, frame, 1, 1, 300, 0,
                                 0 if bad else 0x34504F46, 1, 0, 0, 0, 0, 0, 0)
            row = struct.pack("<5f3I", 0, 0, 10, 20, score, 0, 7, 1)
            frames.append(header + row + bytes(299 * 32))
        path.write_bytes(b"SCDR" + struct.pack("<IIQQ", 1, 9664, 17, 6) + b"".join(frames))

    with tempfile.TemporaryDirectory(prefix="saccade-p08-cli-") as directory:
        root = Path(directory)
        data, other, failed = root / "input.scdr", root / "other.scdr", root / "failed.scdr"
        fixture(data); fixture(other, 0.9); fixture(failed, bad=True)
        run("tracking_config", "schema")
        config, receipt = root / "canonical.cfg", root / "receipt.txt"
        run("tracking_config", "resolve", "--output", config, "--record", receipt)
        run("tracking_config", "compare", config, receipt)
        equivalent = root / "equivalent.cfg"
        lines = config.read_text().splitlines()
        equivalent.write_text("# reordered and spaced\n" + "\n".join("  " + x.replace("=", " = ", 1) for x in reversed(lines)) + "\n")
        run("tracking_config", "compare", config, equivalent)
        override = root / "override.cfg"
        override.write_text("schema=saccade-tracking-config/1\ntracker.bridge.enabled=false\n")
        env = dict(clean_env, SACCADE_CFG_TRACKER_BRIDGE_ENABLED="false")
        result = run("tracking_config", "resolve", "--config", override,
                     "--set", "tracker.bridge.enabled=true", env=env)
        check("tracker.bridge.enabled=true\n" in result.stdout, "CLI must win over environment and file")
        for value in ("NaN", "inf", "2", "-1"):
            run("tracking_config", "resolve", "--set", f"tracker.life.confirm_score={value}", ok=False)
        run("tracking_config", "resolve", "--set", "unknown=true", ok=False)
        run("tracking_config", "resolve", "--set", "tracker.bridge.enabled=true", "--set", "tracker.bridge.enabled=false", ok=False)
        run("tracking_config", "resolve", env=dict(clean_env, SACCADE_CFG_UNKNOWN="hidden"), ok=False)
        run("tracking_config", "compare", config, override, "--allow", "tracker.bridge.enabled", ok=False)
        corrupted = root / "tampered.receipt"
        corrupted.write_text(receipt.read_text().replace("config.sha256=", "config.sha256=0", 1))
        run("tracking_config", "compare", receipt, corrupted, ok=False)
        for stem, source in (("plain", data), ("observed", data), ("repeat", data), ("other", other)):
            options = ["--packets", root / (stem + ".bin"), "--resolved-config", root / (stem + ".receipt")]
            if stem != "plain":
                options += ["--events", root / (stem + ".events")]
            run("track_offline", source, "-", root / (stem + ".mot"), *options)
        check((root / "plain.bin").read_bytes() == (root / "observed.bin").read_bytes(), "observer changed full packets")
        check((root / "plain.mot").read_bytes() == (root / "observed.mot").read_bytes(), "observer changed MOT")
        check((root / "observed.events").read_bytes() == (root / "repeat.events").read_bytes(), "event repeat bytes differ")
        run("tracking_config", "compare", root / "plain.receipt", root / "observed.receipt", ok=False)
        run("tracking_config", "compare", root / "plain.receipt", root / "observed.receipt", "--allow", "observer.event_dump")
        run("tracking_events", "validate", root / "observed.events", "--config", root / "observed.receipt")
        run("tracking_events", "validate", root / "observed.events", "--config", root / "observed.receipt", "--packets", root / "observed.bin")
        run("tracking_events", "validate", root / "observed.events", "--packets", root / "other.bin", ok=False)
        run("tracking_events", "validate", root / "observed.events", "--config", receipt, ok=False)
        run("tracking_events", "compare", root / "observed.events", root / "repeat.events")
        result = run("tracking_events", "compare", root / "observed.events", root / "other.events", ok=False)
        check("first divergence event=" in result.stdout and "field=" in result.stdout, "missing first divergence details")
        truncated = root / "truncated.events"
        truncated.write_bytes((root / "observed.events").read_bytes()[:-1])
        run("tracking_events", "validate", truncated, ok=False)
        run("track_offline", data, "-", root / "wrong.mot", "--set", "source.epoch=18", ok=False)
        check(not (root / "wrong.mot").exists(), "bad config created MOT")
        run("track_offline", data, "-", root / "plain.mot", "--resolved-config", data, ok=False)
        run("track_offline", data, "-", root / "same.mot", "--events", root / "same.mot", ok=False)
        run("track_offline", data, "-", root / "existing.mot", "--packets", root / "plain.bin", ok=False)
        scratch = root / "scratch.mot.tmp"
        scratch.write_text("do not replace")
        run("track_offline", data, "-", root / "scratch.mot", ok=False)
        check(scratch.read_text() == "do not replace", "MOT scratch was overwritten")
        run("track_offline", data, "-", root / "scratch2.mot", "--events", root / "scratch2.mot.tmp", ok=False)
        run("track_offline", failed, "-", root / "failure.mot", "--events", root / "failure.events", ok=False)
        check(not (root / "failure.mot").exists(), "failed frame became completed MOT")
        run("tracking_events", "validate", root / "failure.events")
    print(f"P08 CLI {checks} checks PASS")


if __name__ == "__main__":
    main()
