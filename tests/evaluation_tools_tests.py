#!/usr/bin/env python3
"""Integration checks for the standalone MOT17 / PP22 evaluation tools.

Run with a Python environment containing the evaluation NumPy / SciPy versions:
    ~/.cache/saccade-v1/eval/venv/bin/python tests/evaluation_tools_tests.py
All ground truth and predictions below are synthetic and remain in a temporary
directory. No GPU, shared dataset, or legacy checkout is used.
"""

import csv
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
METRICS = {"HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"}


def gt_row(frame, identity=1):
    return f"{frame},{identity},10,20,30,40,1,1,1\n"


def mot_row(frame, identity=11, x=10):
    return f"{frame},{identity},{x},20,30,40,0.9,-1,-1,-1\n"


class EvaluationToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="saccade eval test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.datasets = self.root / "shared datasets"
        self.results = self.root / "MOT results"
        self.results.mkdir()
        self.work = self.root / "unrelated working directory"
        self.work.mkdir()
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("SACCADE_")}
        self.env.update({
            "SACCADE_EVAL_PYTHON": sys.executable,
            "SACCADE_DATASET_ROOT": str(self.datasets),
            "SACCADE_LEGACY_ROOT": str(self.root / "legacy does not exist"),
            "PYTHONDONTWRITEBYTECODE": "1",
        })

    def sequence(self, name, frames, *, benchmark="mot17", length=None):
        subdirectory = "MOT17/train" if benchmark == "mot17" else "PersonPath22/mot_test_kf"
        directory = self.datasets / subdirectory / name
        (directory / "gt").mkdir(parents=True)
        (directory / "gt/gt.txt").write_text("".join(gt_row(frame) for frame in frames))
        (directory / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={name}\n"
            "imDir=img1\nframeRate=30\n"
            f"seqLength={length if length is not None else max(frames)}\n"
            "imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
        return directory

    def predictions(self, name, rows):
        path = self.results / (name + ".txt")
        path.write_text("".join(rows))
        return path

    def run_eval(self, benchmark, *sequences, ok=True, env=None):
        result = subprocess.run(
            ["bash", str(REPO / "scripts" / f"eval_{benchmark}.sh"),
             str(self.results), *sequences],
            cwd=self.work, env=self.env if env is None else env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if ok:
            self.assertEqual(result.returncode, 0,
                             f"{result.stderr}\n{result.stdout[-2000:]}")
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def details(self):
        path = self.results / "trackeval/results/v1/pedestrian_detailed.csv"
        with path.open(newline="") as source:
            return {row["seq"]: row for row in csv.DictReader(source)}

    def summary(self, stdout):
        lines = stdout.splitlines()
        self.assertEqual(len(lines), 1, stdout)
        values = dict(field.split("=", 1) for field in lines[0].split())
        self.assertEqual(set(values), METRICS)
        return {key: float(value) for key, value in values.items()}

    def previous_output(self):
        directory = self.results / "trackeval"
        directory.mkdir()
        marker = directory / "previous evidence.txt"
        marker.write_text("preserve this output\n")
        return marker

    def assert_previous_output(self, marker):
        self.assertEqual(marker.read_text(), "preserve this output\n")
        self.assertEqual(list(marker.parent.iterdir()), [marker])

    def test_perfect_autodiscovery_combined_and_per_sequence(self):
        for name, frames in (("MOT17-10-SDP", (1, 2)), ("MOT17-11-SDP", (1, 2, 3))):
            self.sequence(name, frames)
            self.predictions(name, [mot_row(frame) for frame in frames])
        result = self.run_eval("mot17")
        self.assertEqual(self.summary(result.stdout),
                         {key: 0.0 if key == "IDSW" else 100.0 for key in METRICS})
        rows = self.details()
        self.assertEqual(set(rows), {"MOT17-10-SDP", "MOT17-11-SDP", "COMBINED"})
        self.assertEqual(float(rows["COMBINED"]["CLR_TP"]), 5)
        for row in rows.values():
            for metric in ("HOTA___AUC", "DetA___AUC", "AssA___AUC", "IDF1", "MOTA"):
                self.assertEqual(float(row[metric]), 1.0)
        self.assertFalse((self.root / "legacy does not exist").exists())

    def test_identity_switch_has_expected_association_scores(self):
        name = "MOT17-10-SDP"
        self.sequence(name, (1, 2))
        self.predictions(name, [mot_row(1, 11), mot_row(2, 12)])
        self.run_eval("mot17", name)
        row = self.details()[name]
        self.assertEqual(float(row["CLR_TP"]), 2)
        self.assertEqual(float(row["IDSW"]), 1)
        self.assertEqual(float(row["IDF1"]), 0.5)
        self.assertEqual(float(row["MOTA"]), 0.5)
        self.assertEqual(float(row["DetA___AUC"]), 1.0)
        self.assertEqual(float(row["AssA___AUC"]), 0.5)
        self.assertAlmostEqual(float(row["HOTA___AUC"]), math.sqrt(0.5))

    def test_false_positive_has_expected_detection_scores(self):
        name = "MOT17-10-SDP"
        self.sequence(name, (1, 2))
        self.predictions(name, [mot_row(1), mot_row(2), mot_row(2, 12, x=500)])
        self.run_eval("mot17", name)
        row = self.details()[name]
        self.assertEqual(float(row["CLR_TP"]), 2)
        self.assertEqual(float(row["CLR_FP"]), 1)
        self.assertEqual(float(row["IDF1"]), 0.8)
        self.assertEqual(float(row["MOTA"]), 0.5)
        self.assertAlmostEqual(float(row["DetA___AUC"]), 2 / 3)
        self.assertAlmostEqual(float(row["HOTA___AUC"]), math.sqrt(2 / 3))

    def test_combined_scores_are_recomputed_from_counts(self):
        first, second = "MOT17-10-SDP", "MOT17-11-SDP"
        for name in (first, second):
            self.sequence(name, (1, 2))
            self.predictions(name, [mot_row(1), mot_row(2)] +
                             ([mot_row(2, 12, x=500)] if name == second else []))
        self.run_eval("mot17", first, second)
        rows = self.details()
        combined = rows["COMBINED"]
        self.assertEqual(float(combined["CLR_TP"]), 4)
        self.assertEqual(float(combined["CLR_FP"]), 1)
        self.assertAlmostEqual(float(combined["DetA___AUC"]), 4 / 5)
        self.assertAlmostEqual(float(combined["HOTA___AUC"]), math.sqrt(4 / 5))
        self.assertAlmostEqual(float(combined["IDF1"]), 8 / 9)
        mean = sum(float(rows[name]["HOTA___AUC"]) for name in (first, second)) / 2
        self.assertNotAlmostEqual(float(combined["HOTA___AUC"]), mean)

    def test_pp22_filters_keyframes_and_preserves_frame_numbers(self):
        name = "uid_vid_00008.mp4"
        directory = self.sequence(name, (2, 5), benchmark="pp22", length=7)
        prediction = self.predictions(name, [mot_row(frame) for frame in range(1, 8)])
        original = prediction.read_bytes()
        self.run_eval("pp22")
        self.assertEqual(prediction.read_bytes(), original)
        output = self.results / "trackeval"
        filtered = (output / "trackers/v1" / (name + ".txt")).read_text()
        self.assertEqual(filtered, mot_row(2) + mot_row(5))
        with (output / "keyframe-filter.csv").open(newline="") as source:
            self.assertEqual(list(csv.DictReader(source)), [{
                "sequence": name, "keyframes": "2", "input_rows": "7", "kept_rows": "2"}])
        row = self.details()[name]
        self.assertEqual(float(row["CLR_Frames"]), 7)
        self.assertEqual(float(row["GT_Dets"]), 2)
        self.assertEqual(float(row["Dets"]), 2)
        self.assertEqual(float(row["HOTA___AUC"]), 1.0)
        self.assertIn("seqLength=7\n", (directory / "seqinfo.ini").read_text())

    def test_ground_truth_overrides_are_independent_of_dataset_root(self):
        for benchmark, name, variable in (("mot17", "MOT17-10-SDP", "SACCADE_MOT17_GT"),
                                          ("pp22", "uid_vid_00008.mp4", "SACCADE_PP22_GT")):
            with self.subTest(benchmark=benchmark):
                directory = self.sequence(name, (1,), benchmark=benchmark)
                self.predictions(name, [mot_row(1)])
                env = dict(self.env, SACCADE_DATASET_ROOT=str(self.root / "absent data"))
                env[variable] = str(directory.parent)
                result = self.run_eval(benchmark, name, env=env)
                self.assertEqual(self.summary(result.stdout)["HOTA"], 100)

    def test_empty_results_preserve_existing_output(self):
        marker = self.previous_output()
        for benchmark in ("mot17", "pp22"):
            with self.subTest(benchmark=benchmark):
                result = self.run_eval(benchmark, ok=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("no result files", result.stderr.lower())
                self.assert_previous_output(marker)

    def test_missing_ground_truth_preserves_existing_output(self):
        marker = self.previous_output()
        for benchmark, name in (("mot17", "MOT17-10-SDP"), ("pp22", "uid_vid_00008.mp4")):
            with self.subTest(benchmark=benchmark):
                self.predictions(name, [mot_row(1)])
                result = self.run_eval(benchmark, name, ok=False)
                self.assertEqual(result.returncode, 2)
                self.assert_previous_output(marker)

    def test_missing_python_preserves_existing_output(self):
        marker = self.previous_output()
        name = "MOT17-10-SDP"
        self.sequence(name, (1,))
        self.predictions(name, [mot_row(1)])
        env = dict(self.env, SACCADE_EVAL_PYTHON=str(self.root / "absent python"))
        result = self.run_eval("mot17", name, ok=False, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("python", result.stderr.lower())
        self.assert_previous_output(marker)

    def test_missing_python_packages_preserves_existing_output(self):
        marker = self.previous_output()
        name = "MOT17-10-SDP"
        self.sequence(name, (1,))
        self.predictions(name, [mot_row(1)])
        empty_venv = self.root / "venv without packages"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(empty_venv)],
                       check=True, text=True, timeout=30,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        env = dict(self.env, SACCADE_EVAL_PYTHON=str(empty_venv / "bin/python"))
        result = self.run_eval("mot17", name, ok=False, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("NumPy and SciPy", result.stderr)
        self.assertIn("setup_eval.sh", result.stderr)
        self.assert_previous_output(marker)

    def test_missing_seqinfo_preserves_existing_output(self):
        marker = self.previous_output()
        name = "MOT17-10-SDP"
        directory = self.sequence(name, (1,))
        (directory / "seqinfo.ini").unlink()
        self.predictions(name, [mot_row(1)])
        result = self.run_eval("mot17", name, ok=False)
        self.assertEqual(result.returncode, 2)
        self.assert_previous_output(marker)

    def test_score_rejects_a_different_ground_truth_root(self):
        # The synthetic receipt only reaches the root-identity rejection. It is
        # kept in a separate Git repository and never serves as formal evidence.
        checkout = self.root / "isolated score checkout"
        (checkout / "scripts").mkdir(parents=True)
        for name in ("score_mot17_evaluation.sh", "eval_common.sh"):
            shutil.copyfile(REPO / "scripts" / name, checkout / "scripts" / name)
        (checkout / ".gitignore").write_text("/build/\n")

        def git(*args):
            return subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                 *args], cwd=checkout, text=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

        git("init", "-q")
        git("add", "scripts", ".gitignore")
        git("-c", "user.name=Evaluation Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "synthetic scoring rejection fixture")
        output = checkout / "build/synthetic-rejection"
        capture = output / "capture"
        capture.mkdir(parents=True)
        evaluated_gt, other_gt = self.root / "evaluated sequences", self.root / "other sequences"
        evaluated_gt.mkdir()
        other_gt.mkdir()
        (capture / "gates.txt").write_text("PASS\n")
        (capture / "source.txt").write_text(
            f"allow_dirty=true\nrevision={git('rev-parse', 'HEAD')}\n"
            f"sequences={evaluated_gt}\nrecords={self.root / 'absent records'}\n")
        relative = "scripts/score_mot17_evaluation.sh"
        digest = hashlib.sha256((checkout / relative).read_bytes()).hexdigest()
        (capture / "source-files.sha256").write_text(f"{digest}  {relative}\n")
        env = dict(self.env, SACCADE_MOT17_GT=str(other_gt))
        result = subprocess.run(
            ["bash", "scripts/score_mot17_evaluation.sh", str(output), "--allow-dirty"],
            cwd=checkout, env=env, text=True, timeout=30,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("TrackEval GT root is not the evaluated sequence directory", result.stderr)
        self.assertFalse((output / "score").exists())
        self.assertFalse((capture / "scoring.txt").exists())
        self.assertFalse(list(capture.glob("*.command")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
