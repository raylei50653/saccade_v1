"""Run an old-source trainer script under the T01 import guard.

    python t01_exec.py SOURCE_ROOT SCRIPT [ARGS...]

Refuses to import any compiled extension of the old repository (the old CUDA
scan kernel lives in saccade_tracking_ext) and requires ``saccade`` to resolve
inside SOURCE_ROOT, so a stray ``.pth`` cannot swap in another tree.

With $SACCADE_T01_PRECISION set, the compute-precision policy is applied in
this process before the trainer starts, read back and checked (flags plus a
numeric probe), re-read every 10 s while it runs and once more at exit. The
record goes to $SACCADE_T01_PRECISION_RECORD; a deviation is recorded and the
process is stopped with SIGTERM.
"""

from __future__ import annotations

import atexit
import importlib.abc
import json
import os
import re
import runpy
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

_BLOCKED = re.compile(r"^(saccade_[a-z0-9_]*_ext|saccade_scan_torchop|libsaccade_.*)$")


class _BlockOldExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if _BLOCKED.match(fullname.split(".")[0]):
            raise ImportError(f"T01 import guard: {fullname} (old compiled extension) is not allowed")
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _guard_precision() -> None:
    import t01_precision  # noqa: PLC0415

    policy = os.environ["SACCADE_T01_PRECISION"]
    path = Path(os.environ["SACCADE_T01_PRECISION_RECORD"])
    t01_precision.apply(policy)
    record: dict = {"policy": policy, "declared": t01_precision.canonical(policy), "pid": os.getpid(),
                    "start": {"at": _now(), **t01_precision.readback()}, "samples": 0, "deviations": []}
    record["start"]["errors"] = t01_precision.check(policy, record["start"])
    lock = threading.Lock()
    stopped = threading.Event()

    def write() -> None:
        with lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(path)

    write()
    if record["start"]["errors"]:
        raise SystemExit(f"T01 precision policy {policy} not in effect: {record['start']['errors']}")

    def monitor() -> None:
        while not stopped.wait(10.0):
            rb = t01_precision.readback(probe=False)
            errors = t01_precision.check(policy, rb)
            record["samples"] += 1
            if errors:
                record["deviations"].append({"at": _now(), "errors": errors, **rb})
                write()
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def at_exit() -> None:
        stopped.set()
        try:
            rb = t01_precision.readback()
            record["exit"] = {"at": _now(), **rb, "errors": t01_precision.check(policy, rb)}
        except Exception as exc:  # noqa: BLE001 - recorded; the runner fails the attempt
            record["exit"] = {"at": _now(), "errors": [f"readback failed: {exc!r}"]}
        write()

    threading.Thread(target=monitor, daemon=True, name="t01-precision").start()
    atexit.register(at_exit)


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    script = (source_root / sys.argv[2]).resolve()
    sys.meta_path.insert(0, _BlockOldExtensions())
    sys.path[:0] = [str(source_root), str(source_root / "src")]
    import saccade  # noqa: PLC0415

    origin = Path(saccade.__file__).resolve()
    if source_root not in origin.parents:
        raise SystemExit(f"T01 import guard: saccade resolved to {origin}, outside {source_root}")
    if os.environ.get("SACCADE_T01_PRECISION"):
        _guard_precision()
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
