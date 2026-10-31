#!/usr/bin/env python3
"""Archive the approved T01 experiment in three bundles; never alter sources."""
import hashlib
import contextlib
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache/saccade-v1/t01"
DEST = Path.home() / ".cache/saccade-v1/archives/t01-experiment-final-2026-10-31"
RESERVE = 8 * 1024**3
MAIN = "2db332e96b7c1346d36808b61f366dcb9ca389d0"
TAG = "t01-experiment-final-2026-10-31"
EXTRAS = [REPO / p for p in ["build-t01-head-evidence", "build-t01-f03-evidence", "build-preprocess-evidence",
          "build-f01-investigation", "build-preprocess-initcheck-control", "build-f01-windows",
          "evidence/preprocess/2026-10-31-usage"]]
BASE_WEIGHT = Path.home() / "developer/ai/saccade/models/yolo/yolo26m.pt"
ACTIVE = set()
LIBC = ctypes.CDLL(None, use_errno=True)


def spawn(argv, **kwargs):
    parent = os.getpid()
    def parent_death():
        if LIBC.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            os._exit(127)
        if os.getppid() != parent:
            os.kill(os.getpid(), signal.SIGKILL)
    p = subprocess.Popen(argv, start_new_session=True, preexec_fn=parent_death, **kwargs)
    ACTIVE.add(p)
    return p


def cleanup_processes(processes=None):
    for p in list(ACTIVE if processes is None else processes):
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.wait()
        ACTIVE.discard(p)


@contextlib.contextmanager
def process_scope():
    existing = set(ACTIVE)
    try:
        yield
    finally:
        cleanup_processes(ACTIVE-existing)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def progress(phase, **values):
    value = dict(phase=phase, recorded_unix=time.time(), **values)
    save(DEST / "progress.json", value)
    print(json.dumps(value), flush=True)


def command(label, argv, cwd=REPO):
    started = time.time()
    p = spawn([str(a) for a in argv], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout,stderr = p.communicate()
    ACTIVE.discard(p)
    (DEST / "receipts" / (label+".stdout")).write_bytes(stdout)
    (DEST / "receipts" / (label+".stderr")).write_bytes(stderr)
    save(DEST / "receipts" / (label+".json"), dict(argv=[str(a) for a in argv], cwd=str(cwd),
         started_unix=started, ended_unix=time.time(), exit=p.returncode))
    if p.returncode:
        raise RuntimeError(f"{label} exit {p.returncode}")
    return stdout.decode(errors="replace")


def snapshot(path):
    s = path.lstat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                device=s.st_dev, inode=s.st_ino, mode=stat.S_IMODE(s.st_mode), uid=s.st_uid, gid=s.st_gid)


def walk(root):
    if root.is_symlink() or root.is_file():
        yield root
        return
    for base, dirs, files in os.walk(root, followlinks=False):
        yield Path(base)
        for name in dirs[:]:
            p = Path(base) / name
            if p.is_symlink():
                yield p
                dirs.remove(name)
        for name in files:
            yield Path(base) / name


def no_source_writers():
    writers = []
    roots = [str(CACHE)] + [str(p) for p in EXTRAS]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            for fd in (p / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if not any(target == root or target.startswith(root+"/") for root in roots):
                        continue
                    flags = int(next(r.split()[1] for r in (p / "fdinfo" / fd.name).read_text().splitlines()
                                     if r.startswith("flags:")), 8)
                    if flags & (os.O_WRONLY | os.O_RDWR):
                        writers.append(dict(pid=p.name, fd=fd.name, target=target, flags=oct(flags)))
                except (OSError, StopIteration):
                    pass
        except OSError:
            pass
    if writers:
        raise RuntimeError("source writers found: " + repr(writers))
    return dict(recorded_unix=time.time(), open_writable_source_fds=writers,
                limit="point-in-time same-user /proc FD scan; source stat guards additionally enforced")


def inventory():
    groups = {"core": {}, "formal-feature-cache": {}, "negative-feature-cache": {}}
    def add(group, p):
        s = p.lstat()
        kind = "symlink" if stat.S_ISLNK(s.st_mode) else "directory" if stat.S_ISDIR(s.st_mode) else "file" if stat.S_ISREG(s.st_mode) else None
        if kind is None:
            raise RuntimeError("unsupported source type: " + str(p))
        name = str(p).lstrip("/")
        groups[group][name] = dict(path=str(p), name=name, kind=kind, stat=snapshot(p),
                                   **({"target": os.readlink(p)} if kind == "symlink" else {}))
    for child in sorted(CACHE.iterdir()):
        if child.name in ("venv", "venv-export"):
            continue
        for p in walk(child):
            rel = p.relative_to(CACHE)
            payload = p.is_file() and not p.is_symlink() and p.suffix == ".pt" and any(x.startswith("cache_") for x in rel.parts)
            add("formal-feature-cache" if payload and rel.parts[0] == "runs" else "negative-feature-cache" if payload else "core", p)
    for root in EXTRAS + [BASE_WEIGHT]:
        if not root.exists():
            raise RuntimeError("approved source missing: " + str(root))
        for p in walk(root):
            add("core", p)
    # Restore installed environments from locks, but retain wheel identity records
    # and the tiny nvJPEG binary directly referenced by historical controls.
    for name in ("venv", "venv-export"):
        env = CACHE / name
        add("core", env / "pyvenv.cfg")
        for p in (env / "lib/python3.12/site-packages").glob("*.dist-info/*"):
            if p.is_file() and p.name in ("METADATA", "WHEEL", "RECORD", "INSTALLER", "direct_url.json"):
                add("core", p)
    old_nvjpeg = CACHE / "venv-export/lib/python3.12/site-packages/torchvision.libs/libnvjpeg.36e11081.so.13"
    add("core", old_nvjpeg)
    for p in walk(DEST / "metadata"):
        add("core", p)
    return {k: [v[n] for n in sorted(v)] for k, v in groups.items()}


def hash_group(name, entries):
    total = sum(e["stat"]["size"] for e in entries if e["kind"] == "file")
    done = 0
    last = 0
    for e in entries:
        p = Path(e["path"])
        if snapshot(p) != e["stat"]:
            raise RuntimeError("source drift before hash: " + str(p))
        if e["kind"] == "file":
            e["sha256"] = digest(p)
            done += e["stat"]["size"]
        if snapshot(p) != e["stat"]:
            raise RuntimeError("source drift during hash: " + str(p))
        if done-last >= 1024**3:
            progress("hash", bundle=name, bytes=done, total_bytes=total)
            last = done
    path = DEST / "receipts" / (name+"-manifest.json")
    save(path, dict(schema="t01-lossless-archive/1", bundle=name, entries=entries,
                    file_count=sum(e["kind"] == "file" for e in entries), logical_bytes=total))
    return path


def check_sources(entries):
    for e in entries:
        p = Path(e["path"])
        if snapshot(p) != e["stat"] or (e["kind"] == "symlink" and os.readlink(p) != e["target"]):
            raise RuntimeError("source identity drift: " + str(p))


def full_compare(path, entries, manifest):
    expected = {e["name"]: e for e in entries}
    manifest_name = str(manifest).lstrip("/")
    expected[manifest_name] = dict(kind="file", stat=dict(size=manifest.stat().st_size), sha256=digest(manifest))
    seen = set()
    verified = {}
    p = spawn(["zstd", "-q", "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    compared = 0
    last = 0
    try:
        with tarfile.open(fileobj=p.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name.rstrip("/")
                if name in seen or name not in expected:
                    raise RuntimeError("unexpected/duplicate archive member: " + name)
                e = expected[name]
                seen.add(name)
                if "mode" in e["stat"] and (member.mode != e["stat"]["mode"] or member.uid != e["stat"]["uid"] or member.gid != e["stat"]["gid"]):
                    raise RuntimeError("mode/owner metadata mismatch: " + name)
                if e["kind"] == "directory":
                    if not member.isdir():
                        raise RuntimeError("directory type mismatch: " + name)
                elif e["kind"] == "symlink":
                    if not member.issym() or member.linkname != e["target"]:
                        raise RuntimeError("symlink target/type mismatch: " + name)
                elif member.islnk():
                    target = member.linkname.rstrip("/")
                    if target not in verified or verified[target] != e["sha256"]:
                        raise RuntimeError("hardlink content target mismatch: " + name)
                    verified[name] = e["sha256"]
                elif member.isfile():
                    if member.size != e["stat"]["size"]:
                        raise RuntimeError("file size mismatch: " + name)
                    h = hashlib.sha256()
                    f = archive.extractfile(member)
                    for block in iter(lambda: f.read(4*1024**2), b""):
                        h.update(block)
                    if h.hexdigest() != e["sha256"]:
                        raise RuntimeError("full archive byte hash mismatch: " + name)
                    verified[name] = h.hexdigest()
                    compared += member.size
                else:
                    raise RuntimeError("unexpected file type: " + name)
                count += 1
                if compared-last >= 2*1024**3:
                    progress("full-tar-compare", bundle=path.name, bytes=compared, members=count)
                    last = compared
        # Drain trailing frame bytes so zstd can validate its checksum/exit.
        p.stdout.read()
        stderr = p.stderr.read()
        code = p.wait()
        if code or seen != set(expected):
            raise RuntimeError(f"archive decode/coverage failure {code}: {stderr[:400]!r}")
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        ACTIVE.discard(p)
    return dict(verified_members=count, full_regular_bytes_hashed=compared, missing_members=[], decoder_exit=code)


def pack(name, entries, manifest, level):
    no_source_writers()
    check_sources(entries)
    final = DEST / (name+".tar.zst")
    partial = DEST / (name+".tar.zst.partial")
    if final.exists() or partial.exists():
        raise RuntimeError("refuse archive overwrite: " + str(final))
    listing = DEST / "receipts" / (name+"-files.nul")
    listing.write_bytes(b"".join(e["name"].encode()+b"\0" for e in entries)+str(manifest).lstrip("/").encode()+b"\0")
    tar_argv = ["tar", "--format=pax", "--numeric-owner", "--no-recursion", "--null", "--verbatim-files-from",
                "-C", "/", "-T", str(listing), "-cf", "-"]
    zstd_argv = ["zstd", "-T2", f"-{level}", "--long=27", "--check", "-q", "-c"]
    record = dict(bundle=name, tar_argv=tar_argv, zstd_argv=zstd_argv, started_unix=time.time(), reserve_bytes=RESERVE)
    save(DEST / "receipts" / (name+"-pack-start.json"), record)
    with process_scope(), open(DEST / "receipts" / (name+"-tar.stderr"), "wb") as te, open(DEST / "receipts" / (name+"-zstd.stderr"), "wb") as ze, open(partial, "xb") as out:
        tar = spawn(tar_argv, stdout=subprocess.PIPE, stderr=te)
        zstd = spawn(zstd_argv, stdin=tar.stdout, stdout=out, stderr=ze)
        tar.stdout.close()
        last = 0
        aborted = None
        while tar.poll() is None or zstd.poll() is None:
            free = shutil.disk_usage(DEST).free
            if free < RESERVE+128*1024**2:
                aborted = "8 GiB reserve guard"
            if time.time()-record["started_unix"] > 4*3600:
                aborted = "4 hour per-bundle bound"
            if aborted:
                for proc in (tar,zstd):
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGKILL)
                break
            if time.time()-last > 10:
                progress("pack", bundle=name, archive_bytes=partial.stat().st_size, disk_free=free, reserve_bytes=RESERVE)
                last = time.time()
            time.sleep(.2)
        record.update(tar_exit=tar.wait(), zstd_exit=zstd.wait(), aborted=aborted, partial_bytes=partial.stat().st_size)
        out.flush()
        os.fsync(out.fileno())
    if record["tar_exit"] or record["zstd_exit"] or aborted:
        save(DEST / "receipts" / (name+"-failed.json"), record)
        # Only this runner's own partial, after a durable failure receipt.
        partial.unlink()
        raise RuntimeError(f"{name} archive aborted; originals untouched: {record}")
    command(name+"-zstd-test", ["zstd", "-q", "-t", str(partial)])
    record["full_tar_compare"] = full_compare(partial, entries, manifest)
    check_sources(entries)
    record.update(archive_bytes=partial.stat().st_size, archive_sha256=digest(partial), manifest_sha256=digest(manifest),
                  ended_unix=time.time(), source_stat_stable=True)
    record["pass"] = True
    os.replace(partial,final)
    fd = os.open(DEST,os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    save(DEST / "receipts" / (name+"-receipt.json"), record)
    progress("bundle-complete", bundle=name, archive_bytes=record["archive_bytes"], archive_sha256=record["archive_sha256"], disk_free=shutil.disk_usage(DEST).free)
    return final


def selected_restore(core, manifests):
    selected = [CACHE / "runs/t1/attempt-01/run/epoch_0001.ckpt", CACHE / "runs/teacher/attempt-01/run/epoch_0012.ckpt",
                CACHE / "export/yolo26m-backbone-t01.onnx", REPO / "build-t01-f03-evidence/trt/model_trt", BASE_WEIGHT,
                DEST / "metadata/repository.bundle"]
    engines = [e for e in manifests if e["name"].startswith(str(REPO / "build-t01-f03-evidence").lstrip("/")) and e["path"].endswith(".engine")]
    if engines:
        selected.append(Path(max(engines,key=lambda e:e["stat"]["size"])["path"]))
    if sum(p.stat().st_size for p in selected) > 1024**3:
        raise RuntimeError("selected staging exceeds 1 GiB bound")
    names = [str(p).lstrip("/") for p in selected]
    staged = DEST / "staging"
    reused_stage = staged.exists()
    if not reused_stage:
        staged.mkdir()
        command("selected-restore-command", ["tar", "--use-compress-program=zstd", "-xf", str(core), "-C", str(staged), "--no-same-owner", "--no-same-permissions", "--"]+names)
    expected = {e["name"]:e for e in manifests}
    if any((staged/name).is_symlink() for name in names):
        raise RuntimeError("selected restored regular file became symlink")
    restored = {name:digest(staged / name) for name in names}
    if any(restored[name] != expected[name]["sha256"] for name in names):
        raise RuntimeError("selected restore mismatch")
    bundle = staged / str(DEST / "metadata/repository.bundle").lstrip("/")
    if not (staged / "repository").exists():
        command("git-staged-clone", ["git", "clone", "--no-checkout", "--branch", "main", str(bundle), str(staged / "repository")])
    reference = "refs/remotes/origin/main" if reused_stage else "refs/heads/main"
    restored_main = command("git-staged-main-resume" if reused_stage else "git-staged-main", ["git", "-C", str(staged / "repository"), "rev-parse", reference]).strip()
    restored_tag = command("git-staged-tag-resume" if reused_stage else "git-staged-tag", ["git", "-C", str(staged / "repository"), "rev-parse", TAG+"^{}"]).strip()
    if restored_main != MAIN or not restored_tag.startswith("a47fb16"):
        raise RuntimeError("staged Git history identity mismatch")
    restore_receipt = dict(files=restored, git_main=restored_main, git_tag_peeled=restored_tag, git_main_reference=reference,
         reused_successful_extraction=reused_stage,
         scope="selected critical files plus Git history restored; full archive covered separately by streaming SHA comparison")
    restore_receipt["pass"] = True
    save(DEST / "receipts/selected-restore.json",restore_receipt)
    shutil.rmtree(staged)  # This run's own verified staging only; source never touched.


def resume_after_core():
    receipt = json.loads((DEST / "receipts/core-receipt.json").read_text())
    manifest = DEST / "receipts/core-manifest.json"
    entries = json.loads(manifest.read_text())["entries"]
    core = DEST / "core.tar.zst"
    if not receipt["pass"] or digest(core) != receipt["archive_sha256"] or digest(manifest) != receipt["manifest_sha256"]:
        raise RuntimeError("verified core archive/manifest drift")
    check_sources(entries)
    failed = DEST / "FAILED.json"
    if failed.exists():
        failed.rename(DEST / "receipts/FAILED-staged-main.json")
    selected_restore(core,entries)
    groups = inventory()
    archives = [str(core)]
    for name in ("formal-feature-cache", "negative-feature-cache"):
        path = hash_group(name,groups[name])
        archives.append(str(pack(name,groups[name],path,9)))
    check_sources(entries)
    save(DEST / "receipts/writers-after.json",no_source_writers())
    complete = dict(archives=archives, completed_unix=time.time(),originals_unchanged=True,
                    reserve_bytes=RESERVE, remaining_bytes=shutil.disk_usage(DEST).free)
    complete["pass"] = True
    save(DEST / "COMPLETE.json",complete)
    progress("complete", archives=archives, disk_free=shutil.disk_usage(DEST).free)


def main():
    if sys.argv[1:] == ["--resume-after-core"]:
        resume_after_core()
        return
    if DEST.exists():
        if sys.argv[1:] != ["--resume-pre-core"] or any(DEST.glob("*.tar.zst")) or any(DEST.glob("*.partial")):
            raise RuntimeError("refuse existing archive destination without bounded pre-core resume: " + str(DEST))
        prior = DEST / "interrupted-pre-cleanup-fix"
        prior.mkdir()
        for name in ("metadata","receipts","progress.json","FAILED.json"):
            p = DEST / name
            if p.exists():
                p.rename(prior / name)
    else:
        DEST.mkdir(parents=True)
    (DEST / "metadata").mkdir()
    (DEST / "receipts").mkdir()
    if shutil.disk_usage(DEST).free < 50*1024**3:
        raise RuntimeError("initial disk capacity changed")
    save(DEST / "metadata/writers-before.json", no_source_writers())
    command("tar-version", ["tar", "--version"])
    command("zstd-version", ["zstd", "--version"])
    head = command("git-main", ["git", "rev-parse", "main"]).strip()
    if head != MAIN:
        raise RuntimeError("approved main changed")
    scoped_bundle = Path("/tmp/saccade-v1-t01-closeout-scoped.bundle")
    if digest(scoped_bundle) != "340d1af88de611aa0793b8b652f9ebecf149a4c6735f5a6dd6175668dc35fbe9":
        raise RuntimeError("approved three-ref Git bundle identity mismatch")
    shutil.copyfile(scoped_bundle,DEST / "metadata/repository.bundle")
    command("git-bundle-verify", ["git", "bundle", "verify", str(DEST / "metadata/repository.bundle")])
    command("git-bundle-heads", ["git", "bundle", "list-heads", str(DEST / "metadata/repository.bundle")])
    for name in ("venv","venv-export"):
        value = command(name+"-freeze", ["uv", "pip", "freeze", "--python", str(CACHE / name / "bin/python")])
        (DEST / "metadata" / (name+"-freeze.txt")).write_text(value)
    for path in (REPO / "training/t01").glob("requirements*.txt"):
        shutil.copyfile(path, DEST / "metadata" / path.name)
    shutil.copyfile(__file__, DEST / "metadata/archive_t01_experiment.py")
    samples = []
    for label,root in [("formal",CACHE / "runs/cache_train/cache"),("negative",CACHE / "smoke-tf32-20261030/cache_train/cache")]:
        sample_path = next(root.rglob("*.pt"))
        raw = sample_path.read_bytes()
        for level in (3,7,9):
            p = spawn(["zstd","-T2",f"-{level}","--long=27","-q","-c"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            compressed,stderr = p.communicate(input=raw)
            ACTIVE.discard(p)
            if p.returncode:
                raise RuntimeError("read-only sample compression failed")
            samples.append(dict(label=label,path=str(sample_path),raw_bytes=len(raw),level=level,
                                compressed_bytes=len(compressed),ratio=len(compressed)/len(raw),exit=p.returncode))
    save(DEST / "metadata/compression-samples.json",samples)
    save(DEST / "metadata/restore-dependencies.json",dict(excluded_installed_venvs=True,
         included_venv_exceptions="pyvenv.cfg, distribution METADATA/WHEEL/RECORD/INSTALLER/direct_url, pinned torchvision nvJPEG 13.0.1",
         dataset="MOT17 external dataset is not bundled; fixed captured inputs and hashes are included",
         sdk="Installed CUDA/TensorRT/driver SDKs are restore dependencies, not bundled; historical binary/library identities are in formal captures",
         path_contract="member paths preserve original absolute path without leading slash; symlinks are preserved, never dereferenced",
         omissions="No feature cache omitted: formal and negative cache payloads are separate mandatory bundles"))
    groups = inventory()
    save(DEST / "metadata/plan.json",dict(groups={name:dict(entries=len(es), logical_bytes=sum(e["stat"]["size"] for e in es if e["kind"]=="file")) for name,es in groups.items()},
         reserve_bytes=RESERVE, source_main=MAIN, source_tag=TAG, threads=2, complete_requires_all_three=True))
    # Include the finalized plan in core after writing it, without rescanning sources.
    p = DEST / "metadata/plan.json"
    groups["core"].append(dict(path=str(p),name=str(p).lstrip("/"),kind="file",stat=snapshot(p)))
    # Metadata directory mtime changed when plan was written: snapshot that directory now.
    for e in groups["core"]:
        if e["path"] == str(DEST / "metadata"):
            e["stat"] = snapshot(DEST / "metadata")
    receipts = []
    for name in ("core", "formal-feature-cache", "negative-feature-cache"):
        entries = groups[name]
        manifest = hash_group(name,entries)
        archive = pack(name,entries,manifest,3 if name=="core" else 9)
        receipts.append(str(archive))
        if name=="core":
            selected_restore(archive,entries)
    save(DEST / "receipts/writers-after.json",no_source_writers())
    complete = dict(archives=receipts, completed_unix=time.time(),originals_unchanged=True,
                    reserve_bytes=RESERVE, remaining_bytes=shutil.disk_usage(DEST).free)
    complete["pass"] = True
    save(DEST / "COMPLETE.json",complete)
    progress("complete", archives=receipts, disk_free=shutil.disk_usage(DEST).free)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM,lambda signum,frame: sys.exit(128+signum))
    try:
        main()
    except BaseException as error:
        cleanup_processes()
        if DEST.exists():
            save(DEST / "FAILED.json",dict(error=str(error),recorded_unix=time.time(),originals_untouched=True))
        raise
    finally:
        cleanup_processes()
