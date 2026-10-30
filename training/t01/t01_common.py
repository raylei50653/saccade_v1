"""Shared helpers for the T01 training orchestration (saccade_v1).

Training only: nothing here is part of the v1 C++/CUDA/TensorRT runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "training" / "t01" / "recipe.json"


def load_recipe() -> dict[str, Any]:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def artifact_root(recipe: dict[str, Any] | None = None) -> Path:
    recipe = recipe or load_recipe()
    return Path(os.environ.get("SACCADE_T01_ROOT", recipe["artifact_root"])).expanduser()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_digest(root: Path) -> tuple[str, list[tuple[str, str]]]:
    """SHA-256 over the sorted "<sha256>  <relative path>\\n" lines of every file."""
    rows = sorted(
        (str(p.relative_to(root)), sha256_file(p))
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    text = "".join(f"{digest}  {rel}\n" for rel, digest in rows)
    return sha256_bytes(text.encode()), rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def v1_source_state() -> dict[str, Any]:
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status.splitlines()}


def split_sequences(recipe: dict[str, Any], role: str) -> list[str]:
    return list(recipe["split"][role])
