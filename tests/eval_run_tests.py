#!/usr/bin/env python3
"""CPU integration tests: stub GPU executables, real repository TrackEval worker."""

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_run", REPO / "scripts/eval_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
EVAL_PYTHON = runner.eval_python()

COMMON = '''#!/usr/bin/python3
import configparser, fcntl, json, os, sys
from pathlib import Path
lock = (Path.home() / '.cache/saccade-v1/gpu.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    pass
else:
    raise RuntimeError('GPU binary called without lock')
seq, out = Path(sys.argv[2]), Path(sys.argv[3])
if 'STUB_SEQUENCE' in os.environ:
    assert seq == Path(os.environ['STUB_SEQUENCE'])
identity = ('source_commit=' + os.environ['STUB_COMMIT'] + '\\nsource_dirty='
            + os.environ.get('STUB_DIRTY', 'false') + '\\ntracker_config_sha256=capture-hash\\n')
'''
TRACK = COMMON + '''
assert sys.argv[4:10] == ['--profile', 'hr-1280x768', '--set',
    'tracker.kalman.nsa_enabled=false', '--set', 'tracker.rows.measurement_box=false']
assert sys.argv[10] == '--resolved-config'
out.write_bytes(Path(os.environ.get('STUB_PREDICTIONS', seq / 'predictions.txt')).read_bytes())
Path(sys.argv[11]).write_text(identity)
print(identity, end='')
print('capture stderr', file=sys.stderr)
'''
BENCH = COMMON + '''
assert sys.argv[4:] == ['--frames', '120', '--warmup', '20', '--rounds', '2', '--only', 'graph']
out.mkdir()
graph = {'mode': 'graph', 'latency_samples': 480, 'latency_mean_us': 2000.123456,
    'latency_p50_us': 1900.5, 'latency_p95_us': 2500.25, 'latency_p99_us': 3000.125,
    'queued_us_per_frame_mean': 1250.5, 'queued_us_per_frame_min': 1200.0,
    'queued_us_per_frame_max': 1300.0}
(out / 'summary.json').write_text(json.dumps({'status': 'PASS', 'groups': [graph]}))
(out / 'runs.csv').write_text('style\\nsingle\\nqueued\\n')
if os.environ.get('STUB_HASH') == 'mismatch':
    identity = identity.replace('capture-hash', 'different-hash')
if os.environ.get('STUB_HASH') == 'absent':
    identity = '\\n'.join(line for line in identity.splitlines() if not line.startswith('tracker_config_sha256='))
for index in range(8):
    (out / (str(index) + '.identity.txt')).write_text(identity)
'''
NVIDIA = '''#!/usr/bin/python3
import sys, time
if '-l' in sys.argv:
    print('timestamp, uuid, pstate, temperature.gpu, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], utilization.gpu [%], memory.used [MiB]', flush=True)
    while True:
        print('stub, stub, P0, 40, 1000, 1000, 10, 1, 1', flush=True)
        time.sleep(1)
else:
    print('name, driver_version\\nCPU stub, 621.14')
'''


def gt(frame):
    return f"{frame},1,10,20,30,40,1,1,1\n"


def prediction(frame, identity=11, x=10):
    return f"{frame},{identity},{x},20,30,40,0.9,-1,-1,-1\n"


class EvalRunTests(unittest.TestCase):
    def setUp(self):
        (REPO / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="eval-run-tests-", dir=REPO / "build")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "stub build"
        self.bin.mkdir()
        for name, content in (("track_sequence", TRACK), ("tracking_benchmark", BENCH), ("nvidia-smi", NVIDIA)):
            path = self.bin / name
            path.write_text(content)
            path.chmod(0o755)
        self.engine = self.root / "engine"
        self.engine.write_bytes(b"CPU fixture engine")
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        SACCADE_EVAL_PYTHON=EVAL_PYTHON,
                        STUB_COMMIT=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
        self.out = REPO / "build" / (self.root.name + "-output")
        self.addCleanup(lambda: shutil.rmtree(self.out, ignore_errors=True))

    def sequence(self, name="fixture", frames=2, keyframes=None):
        seq = self.root / "data" / name
        (seq / "gt").mkdir(parents=True)
        (seq / "images").mkdir()
        (seq / "images/000001.jpg").write_bytes(b"fixture image")
        (seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=images\nimExt=.jpg\nseqLength={frames}\n"
            "imWidth=640\nimHeight=480\nframeRate=30\n")
        (seq / "gt/gt.txt").write_text("".join(gt(frame) for frame in
                                               (range(1, frames + 1) if keyframes is None else keyframes)))
        (seq / "predictions.txt").write_text("".join(prediction(frame) for frame in range(1, frames + 1)))
        return seq

    def command(self, benchmark="MOT17"):
        return ["python3", "-I", "-B", str(REPO / "scripts/eval_run.py"), "--benchmark", benchmark,
                "--build", str(self.bin), "--engine", str(self.engine), "--out", str(self.out.relative_to(REPO))]

    def run_eval(self, *sequences, benchmark="MOT17", options=(), code=0):
        result = subprocess.run([*self.command(benchmark), *options, *map(str, sequences)],
                                cwd=self.root, env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def rows(self):
        return json.loads((self.out / "results.json").read_text())

    def test_argument_and_layout_validation(self):
        seq = self.sequence()
        for output in ("build/../escape", "build/nested/out", "build/-bad", "/tmp/out", "build/name with space"):
            with self.subTest(out=output):
                command = self.command()
                command[command.index("--out") + 1] = output
                result = subprocess.run([*command, str(seq)], env=self.env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
        for options in (("--profile", "hr-1280x768"), ("--benchmark", "SportsMOT")):
            self.run_eval(seq, options=options, code=2)
        self.run_eval(code=2)
        self.run_eval(seq, seq, code=2)
        other = self.sequence("other")
        moved = self.root / "different-parent" / other.name
        moved.parent.mkdir()
        other.rename(moved)
        self.run_eval(seq, moved, code=2)
        for relative in ("gt/gt.txt", "seqinfo.ini", "images"):
            path = seq / relative
            saved = path.with_name(path.name + ".saved")
            path.rename(saved)
            self.run_eval(seq, code=2)
            saved.rename(path)
        (self.bin / "tracking_benchmark").unlink()
        self.run_eval(seq, code=2)
        self.assertFalse(self.out.exists())

    def test_existing_out_is_preserved(self):
        seq = self.sequence()
        self.out.mkdir()
        (self.out / "marker").write_text("previous result")
        self.run_eval(seq, code=2)
        self.assertEqual(list(self.out.iterdir()), [self.out / "marker"])

    def test_eight_digit_image_view_for_capture_and_speed(self):
        seq = self.sequence("dance", frames=140)
        (seq / "images").rename(seq / "img1")
        (seq / "img1/000001.jpg").unlink()
        info = seq / "seqinfo.ini"
        info.write_text(info.read_text().replace("imDir=images", "imDir=img1"))
        for fid in range(1, 141):
            (seq / "img1" / f"{fid:08d}.jpg").write_bytes(b"fixture image")
        view = self.out / "inputs/dance"
        self.env.update(STUB_SEQUENCE=str(view), STUB_PREDICTIONS=str(seq / "predictions.txt"))
        self.run_eval(seq, benchmark="DanceTrack")
        self.assertEqual((view / "seqinfo.ini").read_bytes(), info.read_bytes())
        self.assertEqual(len(list((view / "img1").iterdir())), 140)
        for fid in range(1, 141):
            link = view / "img1" / f"{fid:06d}.jpg"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.readlink(), (seq / "img1" / f"{fid:08d}.jpg").resolve(strict=True))
        environment = json.loads((self.out / "environment.json").read_text())
        self.assertEqual(environment["gt_root"], str(seq.parent))
        self.assertEqual(environment["image_views"], {"dance": {
            "original_path": str(seq), "view_path": str(view), "source_digits": 8, "link_count": 140}})
        self.assertEqual((self.out / "scoring/gt/dance/seqinfo.ini").read_bytes(), info.read_bytes())
        self.assertEqual(self.rows()[0]["HOTA"], 100.0)

    def test_missing_images_exits_during_argument_validation(self):
        seq = self.sequence()
        (seq / "images/000001.jpg").unlink()
        result = self.run_eval(seq, code=2)
        self.assertIn("missing images", result.stderr)
        self.assertFalse(self.out.exists())

    def test_dancetrack_transform_real_worker(self):
        seq = self.sequence("dance")
        source = b"1,1,10,20,30,40,1,99,0.2,extra\r\n\n2,1,10,20,30,40,1,99,0.2\r\n"
        (seq / "gt/gt.txt").write_bytes(source)
        self.run_eval(seq, benchmark="DanceTrack", options=("--no-speed",))
        self.assertEqual((self.out / "scoring/gt/dance/gt/gt.txt").read_text(), gt(1) + gt(2))
        self.assertEqual((seq / "gt/gt.txt").read_bytes(), source)
        self.assertEqual(self.rows()[0]["HOTA"], 100.0)
        config = json.loads((self.out / "scoring/config.json").read_text())["dataset"]
        self.assertEqual(config["BENCHMARK"], "MOT17")

    def test_pp22_filter_preserves_original_frames_and_length(self):
        seq = self.sequence("pp22.mp4", frames=7, keyframes=(2, 5))
        original = (seq / "seqinfo.ini").read_bytes()
        self.run_eval(seq, benchmark="PP22")
        self.assertEqual((self.out / "scoring/trackers/v1/pp22.mp4.txt").read_text(), prediction(2) + prediction(5))
        self.assertEqual((self.out / "scoring/gt/pp22.mp4/seqinfo.ini").read_bytes(), original)
        self.assertEqual((seq / "seqinfo.ini").read_bytes(), original)
        self.assertEqual(self.rows()[0]["frames"], 7)
        self.assertEqual(self.rows()[0]["HOTA"], 100.0)
        self.assertEqual(json.loads((self.out / "scoring/keyframe-filter.json").read_text()),
                         [{"sequence": "pp22.mp4", "keyframes": 2, "input_rows": 7, "kept_rows": 2}])

    def test_short_sequence_skips_speed_and_tables_have_real_combined(self):
        first = self.sequence("first", frames=2)
        second = self.sequence("second", frames=2)
        with (second / "predictions.txt").open("a") as output:
            output.write(prediction(2, identity=12, x=500))
        self.run_eval(first, second)
        rows = self.rows()
        self.assertEqual([row["sequence"] for row in rows], ["first", "second", "COMBINED"])
        combined = rows[-1]
        self.assertEqual(combined["frames"], 4)
        self.assertEqual(combined["IDSW"], 0)
        self.assertAlmostEqual(combined["HOTA"], 100 * math.sqrt(4 / 5))
        self.assertNotAlmostEqual(combined["HOTA"], sum(row["HOTA"] for row in rows[:2]) / 2)
        for row in rows[:2]:
            self.assertEqual(row["speed_note"], "fewer than 140 frames")
            self.assertTrue(all(row[key] is None for key in runner.SPEED))
        self.assertEqual(list((self.out / "speed").iterdir()), [])
        with (self.out / "results.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(tuple(csv_rows[0]), runner.COLUMNS)
        self.assertEqual(csv_rows[-1]["single_mean_ms"], "")
        markdown = (self.out / "RESULTS.md").read_text()
        self.assertIn(f"{combined['HOTA']:.3f}", markdown)
        self.assertIn("COMBINED", markdown)
        self.assertTrue(all(sentence in markdown for sentence in runner.SCOPE))
        self.assertIn("capture stderr", (self.out / "quality/first/track.log").read_text())
        commands = json.loads((self.out / "command.json").read_text())
        self.assertEqual([row["step"] for row in commands["steps"]], ["capture first", "capture second", "scoring"])

    def test_speed_summary_units_identity_and_warning(self):
        seq = self.sequence(frames=140)
        self.env.update(STUB_COMMIT="older-commit", STUB_DIRTY="true")
        self.run_eval(seq)
        row = self.rows()[0]
        self.assertEqual(row["single_mean_ms"], 2.000123456)
        self.assertEqual(row["single_p95_ms"], 2.50025)
        self.assertEqual(row["queued_ms_per_frame"], 1.2505)
        self.assertEqual(row["queued_fps"], 1e6 / 1250.5)
        environment = json.loads((self.out / "environment.json").read_text())
        speed = environment["speed"]["fixture"]
        self.assertEqual(speed["tracker_config_check"], "PASS")
        self.assertEqual(len(speed["sessions"]), 8)
        self.assertEqual(speed["graph"]["queued_us_per_frame_min"], 1200)
        self.assertEqual(speed["graph"]["queued_us_per_frame_max"], 1300)
        self.assertEqual(environment["captures"]["fixture"]["source_commit"], ["older-commit"])
        self.assertTrue((self.out / "RESULTS.md").read_text().startswith("**WARNING:"))
        self.assertEqual(environment["files"]["engine"]["sha256"], runner.sha256(self.engine))
        self.assertIn("621.14", environment["gpus"][0].values())

    def test_speed_config_hash_mismatch_fails(self):
        seq = self.sequence(frames=140)
        self.env["STUB_HASH"] = "mismatch"
        self.run_eval(seq, code=1)
        environment = json.loads((self.out / "environment.json").read_text())
        self.assertEqual(environment["status"], "FAIL")
        self.assertEqual(environment["speed"]["fixture"]["tracker_config_check"], "FAIL")
        self.assertFalse((self.out / "scoring").exists())

    def test_speed_identity_without_hash_records_outcome(self):
        seq = self.sequence(frames=140)
        self.env["STUB_HASH"] = "absent"
        self.run_eval(seq)
        environment = json.loads((self.out / "environment.json").read_text())
        self.assertEqual(environment["speed"]["fixture"]["tracker_config_check"], "not exposed")

    def test_no_speed_and_mot15_mot20_adapters(self):
        seq = self.sequence(frames=140)
        for benchmark in ("MOT15", "MOT20"):
            with self.subTest(benchmark=benchmark):
                self.run_eval(seq, benchmark=benchmark, options=("--no-speed",))
                self.assertEqual(self.rows()[0]["speed_note"], "disabled (--no-speed)")
                config = json.loads((self.out / "scoring/config.json").read_text())["dataset"]
                self.assertEqual(config["BENCHMARK"], benchmark)
                self.assertEqual((self.out / "scoring/gt/fixture/gt/gt.txt").read_bytes(), (seq / "gt/gt.txt").read_bytes())
                shutil.rmtree(self.out)

    def test_eval_interpreter_resolution_matches_common_shell(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runner.eval_python(), str(Path.home() / ".cache/saccade-v1/eval/venv/bin/python"))
        with mock.patch.dict(os.environ, {"SACCADE_EVAL_ROOT": "/custom/eval", "SACCADE_EVAL_PYTHON": ""}, clear=True):
            self.assertEqual(runner.eval_python(), "/custom/eval/venv/bin/python")
        with mock.patch.dict(os.environ, {"SACCADE_EVAL_PYTHON": "/custom/python"}, clear=True):
            self.assertEqual(runner.eval_python(), "/custom/python")


if __name__ == "__main__":
    unittest.main(verbosity=2)
