"""CPU checks of SCDR layout, kept filtering, original coordinates and T01 GT rules."""

from pathlib import Path
import struct
import tempfile

import torch

from eval_scdr import evaluate, FILE_HEADER, FRAME_HEADER, FRAME_BYTES


def main() -> None:
    checks = 0

    def check(value, message):
        nonlocal checks
        checks += 1
        if not value:
            raise RuntimeError(message)

    torch.set_num_threads(8)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        seq = root / "train" / "fixture"
        (seq / "img1").mkdir(parents=True)
        (seq / "gt").mkdir()
        (seq / "seqinfo.ini").write_text("[Sequence]\nimWidth=1280\nimHeight=720\n")
        (seq / "gt/gt.txt").write_text(
            "1,1,100,100,100,100,1,1,1\n2,1,100,100,100,100,1,1,1\n"
            "1,2,300,300,100,100,0,1,1\n1,3,500,300,100,100,1,2,1\n"
            "1,4,700,300,100,100,1,1,0.05\n")
        row = struct.Struct("<5f3I")

        def frame(fid, detections):
            kept = sum(d[-1] == 1 for d in detections)
            header = FRAME_HEADER.pack(1, fid, 1, len(detections), 300, 0, 0x34504F46,
                                       kept, len(detections) - kept, 0, 0, 0, 0, 0)
            return header + b"".join(row.pack(*d) for d in detections) + bytes((300 - len(detections)) * row.size)

        stream = FILE_HEADER.pack(b"SCDR", 1, FRAME_BYTES, 1, 2)
        stream += frame(0, [(100, 100, 200, 200, 0.9, 0, 0, 1),
                            (400, 400, 500, 500, 0.99, 0, 1, 2),
                            (300, 300, 400, 400, 0.8, 0, 2, 1)])
        stream += frame(1, [(100, 100, 200, 200, 0.7, 0, 0, 1)])
        path = root / "fixture.scdr"
        path.write_bytes(stream)
        result = evaluate(path, seq)
        check(result["frames"] == 2, "frame records")
        check(result["gt"] == 2, "marked person visibility GT rules")
        check(abs(result["ap50"] - (51 + 50 * 2 / 3) / 101) < 1e-14, "101-point AP with a ranked FP")
        check(result["thresholds"]["0.25"]["tp"] == 2, "original-pixel xyxy and zero-based frame identity")
        check(result["fp_at_0.25"] == 1, "suppressed rows are excluded")
        check(result["recall_at_0.25"] == 1, "recall")
        for payload in (FILE_HEADER.pack(b"SCDR", 2, FRAME_BYTES, 1, 2) + stream[FILE_HEADER.size:],
                        stream[:-1], stream + b"x"):
            path.write_bytes(payload)
            try:
                evaluate(path, seq)
            except ValueError:
                check(True, "invalid format rejected")
            else:
                check(False, "invalid format accepted")
    print(f"test_eval_scdr: {checks} checks passed")


if __name__ == "__main__":
    main()
