#!/usr/bin/env python3
"""Read-only verification of T01 archives or files restored into staging.

Use trusted receipts/manifests: hashes verify consistency, not authenticity.
--archive-dir checks the completed bundle set (all three mandatory bundles), archive/manifest
hashes and zstd integrity. --root checks every listed manifest entry after
restoring original absolute paths beneath a new staging root. It does not audit
unlisted staging files, restore ownership/modes, or follow symlink targets.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


BUNDLES = ("core", "formal-feature-cache", "negative-feature-cache")
BUNDLE_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def relative_name(name):
    require(isinstance(name, str) and name and "\0" not in name and "\\" not in name,
            f"invalid member name: {name!r}")
    parts = name.split("/")
    require(not name.startswith("/") and all(p not in ("", ".", "..") for p in parts),
            f"noncanonical or traversing member name: {name!r}")
    return parts


def manifest(path):
    data = read_json(path)
    require(data.get("schema") == "t01-lossless-archive/1", f"unknown manifest schema: {path}")
    require(isinstance(data.get("bundle"), str) and BUNDLE_NAME.fullmatch(data["bundle"]),
            f"invalid manifest bundle: {path}")
    entries = data.get("entries")
    require(isinstance(entries, list), f"manifest entries absent: {path}")
    names = set()
    regular_bytes = files = 0
    for entry in entries:
        name = entry.get("name")
        relative_name(name)
        require(name not in names, f"duplicate manifest member: {name}")
        names.add(name)
        require(entry.get("path") == "/" + name, f"absolute path/name disagree: {name}")
        require(entry.get("kind") in ("file", "directory", "symlink"), f"unknown member type: {name}")
        size = entry.get("stat", {}).get("size")
        require(type(size) is int and size >= 0, f"invalid member size: {name}")
        if entry["kind"] == "file":
            require(isinstance(entry.get("sha256"), str) and SHA256.fullmatch(entry["sha256"]),
                    f"invalid file hash: {name}")
            regular_bytes += size
            files += 1
        if entry["kind"] == "symlink":
            target = entry.get("target")
            require(isinstance(target, str) and "\0" not in target, f"invalid symlink target: {name}")
    require(data.get("file_count") == files and data.get("logical_bytes") == regular_bytes,
            f"manifest counts disagree: {path}")
    return data


def checked_hash(path, expected):
    require(isinstance(expected, str) and SHA256.fullmatch(expected), f"invalid expected hash: {path}")
    require(not path.is_symlink() and path.is_file(), f"missing or symlinked receipt artifact: {path}")
    require(sha256(path) == expected, f"SHA-256 mismatch: {path}")


def archive_set(directory):
    directory = directory.resolve(strict=True)
    require(directory.is_dir(), f"archive directory absent: {directory}")
    require(not (directory / "FAILED.json").exists(), "archive has a FAILED.json receipt")
    complete = read_json(directory / "COMPLETE.json")
    require(complete.get("pass") is True and complete.get("originals_unchanged") is True,
            "COMPLETE.json does not certify completion")
    archives = complete.get("archives")
    require(isinstance(archives, list) and len(archives) >= len(BUNDLES),
            "COMPLETE.json must list all three bundles")
    names = []
    for source in archives:
        require(isinstance(source, str) and source.startswith("/"), "invalid COMPLETE archive path")
        relative_name(source[1:])
        names.append(PurePosixPath(source).name)
    require(len(set(names)) == len(names) and all(n.endswith(".tar.zst") for n in names),
            "COMPLETE.json contains duplicate or invalid archive names")
    bundles = [name[:-len(".tar.zst")] for name in names]
    require(all(BUNDLE_NAME.fullmatch(b) for b in bundles) and set(BUNDLES) <= set(bundles),
            "COMPLETE.json is missing mandatory bundles or has invalid bundle names")
    results = []
    for bundle in bundles:
        receipt_path = directory / "receipts" / (bundle + "-receipt.json")
        receipt = read_json(receipt_path)
        require(receipt.get("bundle") == bundle and receipt.get("pass") is True,
                f"bundle receipt does not certify success: {bundle}")
        require(receipt.get("tar_exit") == 0 and receipt.get("zstd_exit") == 0
                and receipt.get("aborted") is None and receipt.get("source_stat_stable") is True,
                f"failed/incomplete archive production: {bundle}")
        archive = directory / (bundle + ".tar.zst")
        listing = directory / "receipts" / (bundle + "-manifest.json")
        checked_hash(listing, receipt.get("manifest_sha256"))
        data = manifest(listing)
        require(data["bundle"] == bundle, f"manifest bundle mismatch: {bundle}")
        comparison = receipt.get("full_tar_compare", {})
        require(comparison.get("decoder_exit") == 0 and comparison.get("missing_members") == []
                and comparison.get("verified_members") == len(data["entries"]) + 1,
                f"recorded full-member comparison incomplete: {bundle}")
        require(archive.stat().st_size == receipt.get("archive_bytes"), f"archive size mismatch: {bundle}")
        checked_hash(archive, receipt.get("archive_sha256"))
        process = subprocess.run(["zstd", "-q", "-t", "--", str(archive)], capture_output=True)
        require(process.returncode == 0,
                f"zstd integrity failed for {bundle}: {process.stderr.decode(errors='replace')[:400]}")
        results.append(dict(bundle=bundle, archive_bytes=archive.stat().st_size,
                            manifest_entries=len(data["entries"]), archive_sha256=receipt["archive_sha256"]))
        print(f"Verified archive integrity: {bundle}", file=sys.stderr, flush=True)
    return dict(pass_=True, mode="archive-integrity", bundles=results,
                scope="archive hashes/zstd integrity and completed producer receipts; verify restored contents with --root")


@contextmanager
def parent_fd(root_fd, name):
    """Walk inside staging via directory FDs, rejecting every symlink ancestor."""
    parts = relative_name(name)
    descriptor = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor, parts[-1]
    finally:
        os.close(descriptor)


def restored(root, paths):
    root = root.resolve(strict=True)
    require(root.is_dir(), f"staging directory absent: {root}")
    documents = [manifest(path) for path in paths]
    bundles = [data["bundle"] for data in documents]
    require(len(set(bundles)) == len(bundles), "duplicate bundle manifests")
    seen = set()
    counts = dict(file=0, directory=0, symlink=0)
    compared = 0
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for data in documents:
            for entry in data["entries"]:
                name = entry["name"]
                require(name not in seen, f"member duplicated across manifests: {name}")
                seen.add(name)
                with parent_fd(root_fd, name) as (descriptor, leaf):
                    before = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
                    kind = entry["kind"]
                    if kind == "directory":
                        require(stat.S_ISDIR(before.st_mode), f"directory type mismatch: {name}")
                    elif kind == "symlink":
                        require(stat.S_ISLNK(before.st_mode), f"symlink type mismatch: {name}")
                        require(os.readlink(leaf, dir_fd=descriptor) == entry["target"], f"symlink target mismatch: {name}")
                        require(before.st_size == entry["stat"]["size"], f"symlink size mismatch: {name}")
                    else:
                        require(stat.S_ISREG(before.st_mode), f"regular-file type mismatch: {name}")
                        require(before.st_size == entry["stat"]["size"], f"file size mismatch: {name}")
                        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                        with os.fdopen(fd, "rb") as source:
                            opened = os.fstat(source.fileno())
                            require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), f"file changed before hash: {name}")
                            h = hashlib.sha256()
                            for block in iter(lambda: source.read(4 * 1024**2), b""):
                                h.update(block)
                            after = os.fstat(source.fileno())
                        require((opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) ==
                                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"file changed during hash: {name}")
                        require(h.hexdigest() == entry["sha256"], f"file SHA-256 mismatch: {name}")
                        compared += before.st_size
                    counts[kind] += 1
    finally:
        os.close(root_fd)
    return dict(pass_=True, mode="restored-manifests", bundles=bundles, entries=len(seen),
                counts=counts, regular_bytes_hashed=compared,
                scope="listed restored entries only; symlink targets not followed; ownership/modes and unlisted paths not audited")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive-dir", type=Path)
    mode.add_argument("--root", type=Path, help="new staging root containing original paths without leading slash")
    parser.add_argument("--manifest", type=Path, nargs="+", action="extend", help="one or more trusted .json or .json.gz manifests")
    args = parser.parse_args()
    if args.archive_dir:
        if args.manifest:
            parser.error("--archive-dir uses its completion receipts; do not supply --manifest")
        result = archive_set(args.archive_dir)
    else:
        if not args.manifest:
            parser.error("--root requires --manifest")
        result = restored(args.root, args.manifest)
    result["pass"] = result.pop("pass_")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
