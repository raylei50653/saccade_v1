#!/usr/bin/env python3
"""Illustrate saved tracking observations; no inference, scoring, or GT access.

Reads the existing rendered demo and its packet-derived JSON. FFmpeg extracts
three exact frames and a fixed crop; SVG/PDF add only explanatory labels.
"""
import base64
import hashlib
import html
import json
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEMO = ROOT / "docs/demos/report1/assets"
VIDEO = DEMO / "tracking-demo.mp4"
DATA = DEMO / "data.json"
VIDEO_RECEIPT = DEMO / "video-render.json"
SVG = ROOT / "docs/reports/assets/report1-3cbaa518f0dc/story-example.svg"
PDF = HERE / "assets/story-example.pdf"
RECEIPT = HERE / "example-sources.json"
INDICES = (180, 195, 210)
OUTPUT_ID = 71
CROP = (1200, 240, 680, 800)  # In the already rendered 1920x1080 demo video.
FONT = "Noto Sans CJK TC, Noto Sans, sans-serif"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text(x, y, value, size=24, fill="#16324F", weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>')


def main():
    sources = [VIDEO, DATA, VIDEO_RECEIPT]
    before = {p: sha(p) for p in sources}
    data = json.loads(DATA.read_text())
    meta = data["meta"]
    assert (meta["width"], meta["height"], meta["fps"], meta["frameCount"]) == (1920, 1080, 30, 654)
    assert meta["sequence"] == "MOT17-10-SDP"
    rows = []
    for index in INDICES:
        row, = [r for r in data["frames"][index]["rows"] if r[0] == OUTPUT_ID]
        assert row[8] == 1, "only saved observed rows may be illustrated"
        rows.append(row)
    assert len({row[1] for row in rows}) == 1, "UID changed across selected snapshots"

    items = ['<rect width="1400" height="850" fill="#FFFFFF"/>',
             text(36, 49, "追蹤做的事：讓畫面裡的人，跨幀保有同一個 ID", 34, weight=700),
             text(36, 87, "下圖放大同一段畫面的右側；框與編號直接來自既有系統輸出。", 23, "#536578")]
    snapshots = []
    with tempfile.TemporaryDirectory(prefix="saccade-story-example-") as directory:
        for n, (index, row) in enumerate(zip(INDICES, rows)):
            image_path = Path(directory) / f"frame-{index + 1}.jpg"
            x, y, width, height = CROP
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(VIDEO),
                       "-vf", f"select=eq(n\\,{index}),crop={width}:{height}:{x}:{y}",
                       "-frames:v", "1", "-q:v", "2", "-threads", "1", "-y", str(image_path)]
            subprocess.run(command, check=True)
            payload = image_path.read_bytes()
            embedded = base64.b64encode(payload).decode()
            left = 36 + n * 450
            items += [f'<rect x="{left}" y="114" width="428" height="548" rx="10" fill="#EFF5F8"/>',
                      text(left + 18, 150, f"第 {index + 1} 幀 · {index / meta['fps']:.1f} 秒", 25, weight=700),
                      f'<image x="{left + 10}" y="166" width="408" height="480" '
                      f'href="data:image/jpeg;base64,{embedded}"/>',
                      f'<rect x="{left + 14}" y="608" width="137" height="36" rx="5" fill="#16324F"/>',
                      text(left + 25, 635, "輸出 ID 71", 24, "#FFFFFF", 700)]
            snapshots.append({"zero_based_index": index, "source_JPEG_frame": index + 1,
                              "media_time_seconds": index / meta["fps"],
                              "output_id": row[0], "uid": row[1], "original_xyxy": row[2:6],
                              "observed_flags": row[8], "cropped_JPEG_sha256": hashlib.sha256(payload).hexdigest(),
                              "cropped_JPEG_bytes": len(payload),
                              "command": command[:command.index(str(image_path))] + ["<temporary JPEG>"]})
    items += [text(36, 710, "看圖重點", 26, weight=700),
              text(36, 750, "連續畫面中，系統都輸出 ID 71；這正是追蹤要提供的跨幀關聯。", 26),
              text(36, 793, "展示系統輸出，非逐人正確性判定。三張局部快照也不代表整段身份都正確。", 22, "#536578"),
              text(36, 827, "來源：MOT17-10-SDP 既有回放｜固定裁切區域｜30 FPS 來源時間，不是推論速度", 20, "#536578")]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="850" viewBox="0 0 1400 850" '
           'role="img" aria-labelledby="title desc">\n'
           '<title id="title">真實追蹤輸出三幀示例</title>\n'
           '<desc id="desc">MOT17-10-SDP 第181、196、211幀的固定局部裁切。觀測輸出ID71與UID106在三幀均存在；'
           '此圖不使用GT，也不判定身份正確性。</desc>\n'
           f'<g font-family="{FONT}">\n' + "\n".join(items) + '\n</g>\n</svg>\n')
    SVG.parent.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text(svg)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(PDF), str(SVG)], check=True)
    assert all(sha(p) == before[p] for p in sources), "source changed during rendering"
    receipt = {"schema": "report-tracking-example/1", "sequence": meta["sequence"],
               "operation": "extract existing rendered demo frames; fixed crop and explanatory SVG labels",
               "new_inference": False, "new_quality_scoring": False, "GT_read": False,
               "scope": "saved observed output IDs, not GT identity or full-sequence identity correctness",
               "crop_rendered_video_xywh": CROP, "snapshots": snapshots,
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                            "sha256": before[p]} for p in sources],
               "outputs": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                            "sha256": sha(p)} for p in [SVG, PDF]],
               "tools": {"ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                         "rsvg-convert": subprocess.check_output(["rsvg-convert", "--version"], text=True).strip()}}
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print("Rendered tracking example: frames 181/196/211, saved output ID 71, same UID 106; no inference or GT.")


if __name__ == "__main__":
    main()
