#!/usr/bin/env python3
"""Draw the n/n+1 schedule from saved Nsight analysis, without running a GPU."""
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "evidence/report1/2026-11-12-schedule-trace/timeline.json"
SUMMARY = DATA.with_name("summary.json")
SVG = ROOT / "docs/reports/assets/report1-3cbaa518f0dc/story-schedule.svg"
PDF = HERE / "assets/story-schedule.pdf"
NAVY, TEAL, MUTED, INK, GRID = "#16324F", "#087F8C", "#536578", "#203247", "#DCE4EA"
BLUE, PALE_TEAL, AMBER, PURPLE = "#315C9A", "#B6DFDC", "#BB6A2B", "#7A68A6"


class Canvas:
    def __init__(self):
        self.items = []
        self.rect(0, 0, 1000, 930, "#FFFFFF")

    def rect(self, x, y, w, h, fill, stroke="none", radius=0, opacity=1):
        self.items.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}" stroke="{stroke}" rx="{radius}" opacity="{opacity}"/>')

    def text(self, x, y, content, size=23, color=INK, weight=400, anchor="start"):
        self.items.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(content))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1.5, dash=None):
        dotted = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="M{x1:.3f},{y1:.3f} L{x2:.3f},{y2:.3f}" stroke="{color}" stroke-width="{width}" fill="none"{dotted}/>')

    def path(self, path, color=NAVY, width=2, dash=None):
        dotted = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="{path}" stroke="{color}" stroke-width="{width}" fill="none"{dotted}/>')

    def arrow(self, x1, y1, x2, y2, color=NAVY, dash=None):
        self.line(x1, y1, x2, y2, color, 2.5, dash)
        angle = math.atan2(y2-y1, x2-x1)
        p1 = (x2-10*math.cos(angle-.45), y2-10*math.sin(angle-.45))
        p2 = (x2-10*math.cos(angle+.45), y2-10*math.sin(angle+.45))
        self.items.append(f'<path d="M{p1[0]},{p1[1]} L{x2},{y2} L{p2[0]},{p2[1]}" stroke="{color}" stroke-width="2.5" fill="none"/>')

    def save(self):
        title = "n／n+1：把能重疊的工作接起來"
        desc = "上半為依賴示意，下半為歷史 G1 Nsight trace 的三個相鄰影格階段區間。"
        SVG.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="930" viewBox="0 0 1000 930" role="img" aria-labelledby="title desc">'
                       f'<title id="title">{title}</title><desc id="desc">{desc}</desc>'
                       '<g font-family="Noto Sans CJK TC, Noto Sans, sans-serif">'+"\n".join(self.items)+"</g></svg>\n")


def conceptual(c):
    c.text(28, 43, "n／n+1：把能重疊的工作接起來", 31, NAVY, 700)
    c.text(28, 84, "D 偵測路徑可往下一幀前進；M 的身份狀態仍依幀順序提交。", 23, MUTED)
    c.rect(28, 108, 944, 256, "#F5F8FA", GRID, 10)
    c.text(48, 143, "依賴示意", 24, NAVY, 700)
    c.text(952, 143, "框寬不是時間", 21, MUTED, anchor="end")
    c.text(48, 196, "D 偵測", 23, NAVY, 700)
    c.text(48, 267, "M 追蹤", 23, TEAL, 700)
    c.rect(180, 164, 295, 47, BLUE, radius=7)
    c.text(327.5, 195, "前處理＋偵測 n", 24, "#FFFFFF", 600, "middle")
    c.arrow(480, 187.5, 495, 187.5)
    c.rect(500, 164, 305, 47, BLUE, radius=7)
    c.text(652.5, 195, "下一幀偵測路徑 n+1", 23, "#FFFFFF", 600, "middle")
    c.path("M475,211 V230 H500 V242", MUTED, 2, "5 4")
    c.text(230, 268, "等候候選 ready", 21, MUTED)
    c.rect(500, 242, 168, 45, TEAL, radius=7)
    c.text(584, 272, "關聯／提交 n", 22, "#FFFFFF", 600, "middle")
    c.arrow(675, 264, 811, 264, TEAL, "5 4")
    c.text(744, 246, "commit(n) 後", 18, TEAL, anchor="middle")
    c.rect(820, 242, 140, 45, TEAL, radius=7)
    c.text(890, 272, "追蹤 n+1", 21, "#FFFFFF", 600, "middle")
    c.text(48, 323, "可重疊的工作往前做；同一份狀態不能跨越前幀的 commit。", 23, INK)
    c.text(48, 350, "slot 以 ready／consumed 交接：最後讀取完成才重用，結果滿時背壓。", 20, MUTED)


def stage_color(stage):
    key = stage.lower()
    if key == "f03": return BLUE
    if key == "f06": return PALE_TEAL
    if "d2h" in key: return PURPLE
    if key in ["f02", "f05"]: return AMBER
    if "producer" in key or "f04" == key: return "#7D9BC0"
    return TEAL


def measured(c, data):
    segments = data["segments"]
    assert segments and all(s["end_ns"] > s["start_ns"] for s in segments)
    assert all(s["role"] in ["D", "M"] for s in segments)
    assert all(s["source_frame"] == s["frame"]+1 for s in segments)
    frames = sorted({s["frame"] for s in segments})
    assert len(frames) == 3 and frames == list(range(frames[0], frames[0]+3))
    sources = sorted({s["source_frame"] for s in segments})
    assert len(sources) == 3
    origin = min(s["start_ns"] for s in segments)
    end = max(s["end_ns"] for s in segments)
    span_ms = (end-origin)/1e6
    assert 0 < span_ms < 50
    left, width = 175, 770
    x = lambda stamp: left+(stamp-origin)/(end-origin)*width
    c.text(28, 410, f"Nsight 實測：第 {sources[0]}–{sources[-1]} 幀", 26, NAVY, 700)
    c.text(28, 447, "同幀 F03 偵測與 G1 運動估計也可重疊；下列為階段起訖區間。", 21, MUTED)
    for y, label, color in [(492, "D 偵測", NAVY), (590, "M 追蹤", TEAL)]:
        c.rect(28, y-12, 944, 88, "#F5F8FA", radius=6)
        c.text(48, y+32, label, 23, color, 700)
    step = 2 if span_ms < 20 else 5
    for value in range(0, math.floor(span_ms)+1, step):
        xx = left+value/span_ms*width
        c.line(xx, 477, xx, 677)
        c.text(xx, 710, value, 21, MUTED, anchor="middle")
    c.text(946, 710, "ms", 21, MUTED, anchor="end")
    major = []
    for s in segments:
        y = 499 if s["role"] == "D" else 597
        stage = s["stage"]
        a, b = x(s["start_ns"]), x(s["end_ns"])
        c.rect(a, y, b-a, 39, stage_color(stage), radius=2)
        if stage.lower() in ["f03", "f06"]:
            label = ("F03" if stage.lower() == "f03" else "G1") + f' {s["source_frame"]}'
            if b-a > 72:
                c.text((a+b)/2, y+27, label, 21, "#FFFFFF" if stage.lower()=="f03" else NAVY, 600, "middle")
            major.append(s)
        if stage.lower() == "tracking_tail":
            c.line((a+b)/2, y+41, (a+b)/2, y+61, TEAL, 1.5)
            c.text((a+b)/2, y+83, f'尾 {s["source_frame"]}', 19, TEAL, 600, "middle")
    c.text(28, 756, "圖例：", 21, MUTED)
    for start, fill, label in [(117, BLUE, "F03 偵測"), (322, PALE_TEAL, "G1 區間"), (520, TEAL, "追蹤工作"), (733, AMBER, "F02／F05")]:
        c.rect(start, 739, 23, 17, fill, radius=2)
        c.text(start+32, 756, label, 21, MUTED)
    c.text(28, 799, "長條包含階段內空檔；不表示 SM 持續忙碌或 OFA 純硬體耗時。", 22, MUTED)
    c.text(28, 839, "來源：2026-11-05／乾淨 fae5cb7／G1；原 trace 30 幀，取相鄰三幀。", 21, MUTED)
    c.text(28, 874, "OFA 本體未被直接觀察；此圖用既有 CUDA 活動重建，沒有新跑 GPU。", 21, MUTED)
    c.text(28, 909, "Profiler 會影響時間；這些區間不取代無 profiler 的產品延遲與吞吐。", 21, MUTED)
    return {"frames_zero_based": frames, "source_frames": sources, "origin_ns": origin,
            "end_ns": end, "display_span_ms": span_ms, "segment_count": len(segments),
            "bar_semantics": "stage envelopes, including gaps; not activity unions or hardware busy time"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(DATA.read_text())
    c = Canvas()
    conceptual(c)
    shown = measured(c, data)
    c.save()
    subprocess.run(["rsvg-convert", "--format=pdf", "--output", str(PDF), str(SVG)], check=True)
    fonts = subprocess.check_output(["pdffonts", str(PDF)], text=True).splitlines()[2:]
    assert fonts and all(line.split()[-5] == "yes" for line in fonts)
    extracted = subprocess.check_output(["pdftotext", str(PDF), "-"], text=True)
    assert "Nsight" in extracted and "n／n+1" in extracted
    sources = [DATA] + ([SUMMARY] if SUMMARY.exists() else [])
    sources += [p for p in [ROOT/"src/tracking_session.cpp", ROOT/"cuda/flow_motion.cu"] if p.exists()]
    receipt = {"schema": "report1-schedule-figure/1", "revision_date": "2026-11-12",
               "generator": "render_schedule.py", "generator_sha256": sha(Path(__file__)),
               "sources": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sources],
               "display": shown, "outputs": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in [SVG, PDF]],
               "fonts_embedded": True, "new_GPU_measurement": False,
               "concept_contract_commit": "3cbaa518f0dc37d691404e95ed85c0109c81130d",
               "scope": "current contract concept plus historical G1 stage-envelope visualization; OFA not directly observed"}
    (HERE / "schedule-sources.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n")
    print("Rendered n/n+1 schedule SVG/PDF from saved Nsight analysis; embedded fonts verified.")


if __name__ == "__main__":
    main()
