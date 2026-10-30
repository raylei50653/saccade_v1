"""Run an old-source trainer script under the T01 import guard.

    python t01_exec.py SOURCE_ROOT SCRIPT [ARGS...]

Refuses to import any compiled extension of the old repository (the old CUDA
scan kernel lives in saccade_tracking_ext) and requires ``saccade`` to resolve
inside SOURCE_ROOT, so a stray ``.pth`` cannot swap in another tree.
"""

from __future__ import annotations

import importlib.abc
import re
import runpy
import sys
from pathlib import Path

_BLOCKED = re.compile(r"^(saccade_[a-z0-9_]*_ext|saccade_scan_torchop|libsaccade_.*)$")


class _BlockOldExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if _BLOCKED.match(fullname.split(".")[0]):
            raise ImportError(f"T01 import guard: {fullname} (old compiled extension) is not allowed")
        return None


def main() -> None:
    source_root = Path(sys.argv[1]).resolve()
    script = (source_root / sys.argv[2]).resolve()
    sys.meta_path.insert(0, _BlockOldExtensions())
    sys.path[:0] = [str(source_root), str(source_root / "src")]
    import saccade  # noqa: PLC0415

    origin = Path(saccade.__file__).resolve()
    if source_root not in origin.parents:
        raise SystemExit(f"T01 import guard: saccade resolved to {origin}, outside {source_root}")
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
