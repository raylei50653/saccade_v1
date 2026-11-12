#!/usr/bin/env python3
"""Draw the current G1 dataflow, state publication and buffer leases as vectors."""
import hashlib
import html
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ASSETS = HERE / "assets"
NAVY, BLUE, TEAL = "#16324F", "#315C9A", "#087F8C"
PURPLE, AMBER = "#7462A3", "#A9632B"
INK, MUTED, GRID = "#203247", "#536578", "#DCE4EA"
WHITE, PALE_BLUE, PALE_TEAL = "#FFFFFF", "#EFF4FB", "#EDF8F7"
SOURCES = ["src/tracking_session.cpp", "cuda/flow_motion.cu", "cuda/tracking_session.cu",
           "include/saccade/tracking_session.hpp", "include/saccade/model_profile.hpp",
           "include/saccade/protocol.hpp", "include/saccade/flow_motion_cuda.hpp",
           "include/saccade/flow_config.hpp", "src/mot_output.cpp"]


SOURCE_HASHES = {
    "src/tracking_session.cpp": "8681f7fd8c971560d5d8b39893648e9aa9436038cf5f55300e683b371fd9e538",
    "cuda/flow_motion.cu": "fe0d9f04cca3a7258dce9032b39ed9f027c6fbdf7116418cbab818f7a98344e3",
    "cuda/tracking_session.cu": "fea74953fc9e8678805b8377111fa9b3b7461f0139217fad07531ecc35df5b29",
    "include/saccade/tracking_session.hpp": "b62381ab6993863fbb62d07ea6f6313c2f9bfd78b2cd178522253f2dd0625844",
    "include/saccade/model_profile.hpp": "06e7058f61dc4cae75ca1f590c630cb0b222aaec62c6cd58345aa21558112dba",
    "include/saccade/protocol.hpp": "264c9062ad4eeb587ed348781d93ea6d826ad9c3c960fe435ec2eab44d7e885f",
    "include/saccade/flow_motion_cuda.hpp": "691c41d3830198b4007d7c04b255645abc90650187c345a98146856188ac74e4",
    "include/saccade/flow_config.hpp": "eca65824245439ece139c1466922ea8855d4f268fc337158e2a8e4c1d2f05492",
    "src/mot_output.cpp": "c037aaa37f45133ff42be534cdc07e5d13d05526a3e6cf49e5be1589d501275f"
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Canvas:
    def __init__(self, height, title, desc):
        self.height, self.title, self.desc, self.items = height, title, desc, []
        self.rect(0, 0, 1000, height, WHITE)

    def rect(self, x, y, w, h, fill=WHITE, stroke="none", radius=0, width=1.5):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, text, size=22, color=INK, weight=400, anchor="start"):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(text))}</text>')

    def lines(self, x, y, lines, size=21, color=MUTED, leading=29, anchor="start"):
        for i, line in enumerate(lines):
            self.text(x, y+i*leading, line, size, color, anchor=anchor)

    def line(self, x1, y1, x2, y2, color=GRID, width=2, dash=None, arrow=False):
        self.path(f'M{x1},{y1} L{x2},{y2}', color, width, dash, arrow)

    def path(self, d, color=BLUE, width=2.3, dash=None, arrow=True):
        marker = f' marker-end="url(#{color[1:]})"' if arrow else ""
        dotted = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none"{dotted}{marker}/>')

    def card(self, x, y, w, h, title, lines, color=BLUE, size=21, fill=WHITE):
        self.rect(x, y, w, h, fill, color, 10)
        self.text(x+16, y+32, title, 24, color, 700)
        self.lines(x+16, y+62, lines, size)

    def title_block(self, subtitle):
        self.text(28, 43, self.title, 31, NAVY, 700)
        self.text(28, 82, subtitle, 22, MUTED)

    def save(self, path):
        defs = '<defs>' + ''.join(f'<marker id="{c[1:]}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="{c}"/></marker>' for c in [BLUE, TEAL, PURPLE, MUTED, AMBER]) + '</defs>'
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{self.height}" viewBox="0 0 1000 {self.height}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(self.title)}</title><desc id="desc">{html.escape(self.desc)}</desc>{defs}<g font-family="Noto Sans CJK TC, Noto Sans, sans-serif">' + '\n'.join(self.items) + '</g></svg>\n')


def flow():
    c = Canvas(1200, "一幀進來，分成偵測與運動兩條資料路徑", "Current HR/G1 JPEG dataflow; blocks represent responsibilities and data dependencies, not measured time.")
    c.title_block("JPEG 主線・HR 1280×768・G1 NVOFA・CUDA Graph ON；按箭頭讀資料交接。")
    c.card(40, 112, 250, 100, "CPU：讀取 JPEG", ["壓縮 bytes", "pinned staging"], MUTED, 20)
    c.card(335, 112, 285, 100, "F01：nvJPEG 解碼", ["GPU hardware backend", "另有色彩轉換工作"], BLUE, 20)
    c.card(665, 112, 295, 100, "GPU：原圖 RGB8", ["planar R／G／B", "F02 與 F05 共用讀取"], BLUE, 20)
    c.line(292, 162, 331, 162, MUTED, arrow=True)
    c.line(622, 162, 661, 162, BLUE, arrow=True)

    c.items.append(f'<path d="M28,270 H972 V560 H510 V690 H28 Z" fill="{PALE_BLUE}" stroke="{GRID}" stroke-width="1.5"/>')
    c.rect(530, 560, 442, 130, PALE_TEAL, GRID, 12)
    c.text(45, 304, "D：偵測與灰階", 25, BLUE, 700)
    c.path('M812,212 V248 H240 V330', BLUE)
    c.path('M812,248 V330', BLUE)
    c.text(478, 249, "同一張原圖，兩種用途", 21, MUTED, anchor="middle")
    c.card(50, 330, 400, 95, "F02：模型 images", ["FP32 NCHW 1×3×768×1280", "正規化＋antialiased stretch"], BLUE, 21)
    c.card(550, 330, 400, 95, "F05：運動影像 MI", ["由原圖 RGB 取灰階／縮小", "獨立 MI slot；不是模型 tensor"], BLUE, 21)
    c.line(250, 427, 250, 448, BLUE, arrow=True)
    c.line(750, 427, 750, 448, BLUE, arrow=True)
    c.card(50, 450, 400, 100, "D Graph：偵測與檢查", ["F03 → producer → F04", "6 raw outputs／finite／9 statuses"], BLUE, 21)
    c.card(550, 450, 400, 100, "MI ready → G1 可先開始", ["M 等待 mi_ready", "F03 仍可在 D 上計算"], BLUE, 21)
    c.line(250, 552, 250, 573, BLUE, arrow=True)
    c.line(750, 552, 750, 573, TEAL, arrow=True)
    c.card(50, 575, 400, 100, "CS：本幀 candidates", ["kept ＋ suppressed", "candidates_ready 交給 M"], BLUE, 21)
    c.card(550, 575, 400, 100, "M：F06／G1 → MotionContext", ["上一 committed gray／mask ＋ MI", "NVOFA → cells → 分量中位數"], TEAL, 20)

    c.rect(28, 720, 944, 180, PALE_TEAL, GRID, 12)
    c.text(45, 748, "M：先估運動，再依同幀候選接續 ID（此列由右往左）", 23, TEAL, 700)
    c.path('M750,675 V696 H825 V766', TEAL)
    c.path('M250,675 V706 H500 V766', BLUE)
    c.card(690, 770, 260, 110, "F08a：本幀 prior", ["GMC＋一次 Kalman", "讀 committed TrackState"], TEAL, 20)
    c.card(370, 770, 260, 110, "等候 CS → current mask", ["kept person，score ≥ 0.45", "working mask 供下一幀用"], TEAL, 18)
    c.card(50, 770, 260, 110, "M Graph：追蹤提交", ["F04b → 關聯／update", "commit → 固定 packet"], TEAL, 20)
    c.line(687, 825, 633, 825, TEAL, arrow=True)
    c.line(367, 825, 313, 825, TEAL, arrow=True)
    c.line(180, 882, 180, 933, TEAL, arrow=True)

    c.card(50, 935, 400, 100, "M：依序兩次 D2H", ["MotionContext 56 B", "ResultPacket 12,368 B"], TEAL, 21)
    c.card(550, 935, 400, 100, "CPU：receive → F09", ["驗證／複製成 owned result", "JPEG 序列尾插值、ID 定稿、MOT"], MUTED, 20)
    c.line(452, 985, 547, 985, MUTED, arrow=True)

    c.rect(50, 1080, 900, 82, "#F3F0FA", PURPLE, 11)
    c.text(70, 1110, "跨幀回饋：TrackState ＋ gray ＋ mask", 25, PURPLE, 700)
    c.text(70, 1143, "成功 commit 才發布共同版本；gray／mask 回到 G1，TrackState 回到 predict。", 21, MUTED)
    c.path('M180,900 H15 V1120 H47', PURPLE, 2.2, '6 5')
    c.path('M951,1120 H986 V625 H953', PURPLE, 2.2, '6 5')
    c.path('M986,825 H953', PURPLE, 2.2, '6 5')
    c.text(28, 1191, "實線＝資料／必要次序；虛線＝上一已提交版本的回饋。色塊和箭頭不表示時間比例。", 20, MUTED)
    return c


def state():
    c = Canvas(880, "計算先寫 working，成功後才發布新版本", "Two payload slots share a single committed-slot index in Control. Failed work does not replace the last committed state.")
    c.title_block("TrackState、gray、mask 以同一個版本索引前進；Control 本身是一份提交控制。")
    c.card(40, 135, 400, 230, "committed：上次成功的版本", ["TrackState：身份、位置、生命週期", "gray：上次成功影像", "mask：上次成功人物遮罩", "本幀只讀；首幀從空狀態開始"], PURPLE, 21, "#F6F2FC")
    c.card(560, 135, 400, 230, "本幀暫存與 working 輸出", ["predict → 獨立 DevicePredicted", "update → working TrackState", "gray：本幀 MI 寫回", "mask：本幀 kept 人物建立", "PREDICTED 暫存不是 TS 版本"], TEAL, 21, PALE_TEAL)
    c.text(500, 218, "讀舊", 22, PURPLE, 700, "middle")
    c.text(500, 292, "算新", 22, TEAL, 700, "middle")
    c.line(444, 244, 556, 244, PURPLE, arrow=True)
    c.path('M760,365 V407 H500 V428', TEAL)
    c.card(180, 430, 640, 105, "Control：只有一個共同版本選擇", ["working_slot = 1 − committed_slot", "核對身份與錯誤；成功才更新 committed_slot"], NAVY, 21)
    c.path('M380,536 V565 H240 V589', TEAL)
    c.path('M620,536 V565 H760 V589', AMBER)
    c.text(185, 578, "成功", 22, TEAL, 700, "middle")
    c.text(815, 578, "失敗", 22, AMBER, 700, "middle")
    c.card(40, 590, 400, 175, "commit 發布 working 版本", ["committed_slot ← working_slot", "committed_frame 前進一幀", "TS／gray／mask 一起成為新參考", "下一幀讀取這一版"], TEAL, 21, PALE_TEAL)
    c.card(560, 590, 400, 175, "保留原 committed 版本", ["working 可以停留在部分計算", "鎖定首次 fail_frame／fail_code", "本幀 packet：FAILED", "已排入後幀：SKIPPED_AFTER_FAILURE"], AMBER, 20, "#FFF7EF")
    c.text(28, 810, "新狀態的發布由同一條 M stream 排序；沒有讓兩幀同時改寫已提交狀態。", 22, INK)
    c.text(28, 849, "FAILED／SKIPPED 的 count=0；成功但無人的 OK/count=0 是另一種結果。", 21, MUTED)
    return c


def lease_row(c, y, title, slot_text, labels, note, color):
    c.rect(28, y, 944, 151, "#F5F8FA", GRID, 10)
    c.text(44, y+36, title, 24, color, 700)
    c.text(44, y+72, slot_text, 20, MUTED)
    xs = [210, 365, 520, 675, 830]
    for i, (x, label) in enumerate(zip(xs, labels)):
        c.rect(x, y+22, 132, 66, WHITE, color, 7)
        for n, line in enumerate(label):
            c.text(x+66, y+48+n*24, line, 19, color if n==0 else MUTED, 600 if n==0 else 400, 'middle')
        if i<4: c.line(x+135,y+55,xs[i+1]-4,y+55,color,arrow=True)
    c.text(210, y+124, note, 20, MUTED)


def leases():
    c = Canvas(910, "誰讀完、誰歸還：slot 的可重用條件", "Graph-on MI/candidate/result ring lifetimes; host-owned result slots are released by receive, not a consumed event.")
    c.title_block("Graph ON 路徑；MI／candidate／result 容量為 2／2／4，header 不能取代同步。")
    lease_row(c, 125, "MI ring", "2 個 slot", [["D 寫灰階"],["mi_ready"],["M 讀取", "G1／gray copy"],["mi_consumed"],["D 再重用"]], "M 做完 flow_output 後才發布 consumed；下一個 producer 先 wait。", TEAL)
    lease_row(c, 301, "CS ring", "2 個 slot", [["D：F04 寫入"],["CS ready"],["M 讀取", "mask ＋ graph"],["CS consumed"],["D 再重用"]], "Graph 路徑等 M tail 最後讀取完成，才允許 D 覆寫該候選 slot。", BLUE)
    lease_row(c, 477, "Host results", "4 個 pinned slot", [["M：D2H", "56 B＋packet"],["done event"],["CPU receive", "等待＋取回複本"],["lease 歸還", "received 前進"],["M 再 D2H"]], "滿 4 份未接收結果時 enqueue 拒絕；host 先 receive，再繼續提交。", PURPLE)
    c.rect(28, 681, 944, 153, "#FFF8EF", "#D4B590", 10)
    c.text(48, 718, "共用 RGB 另有一條保護邊", 26, AMBER, 700)
    c.lines(48, 754, ["下一次 nvJPEG decode 前，host 等待 rgb_read_done。", "此事件接在 D 上最後的 F05 原圖讀取之後；RGB 不是 MI ring 的一格。", "nvJPEG 有私有工作，僅靠下一個 D 工作的提交順序不足以保護 RGB。"], 22, INK, 30)
    c.text(28, 878, "MI／CS 由 CUDA ready／consumed 交接；host result 由 receive 計數歸還，沒有額外 consumed event。", 19, MUTED)
    return c


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    sources = [{"path": name, "bytes": (ROOT/name).stat().st_size, "sha256": sha(ROOT/name)} for name in SOURCES]
    assert all(row["sha256"] == SOURCE_HASHES[row["path"]] for row in sources), "source differs from fixed 3cbaa518f0dc contract"
    figures = []
    for name, canvas in [("flow", flow()), ("state", state()), ("leases", leases())]:
        svg, pdf = ASSETS/f"{name}.svg", ASSETS/f"{name}.pdf"
        canvas.save(svg)
        subprocess.run(["rsvg-convert", "--format=pdf", "--output", str(pdf), str(svg)], check=True)
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True).splitlines()[2:]
        assert fonts and all(row.split()[-5] == "yes" for row in fonts), name
        extracted = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True).replace("\n", "")
        assert canvas.title in extracted, name
        figures.append({"name": name, "svg": svg.relative_to(ROOT).as_posix(), "svg_sha256": sha(svg),
                        "pdf": pdf.relative_to(ROOT).as_posix(), "pdf_sha256": sha(pdf),
                        "width": 1000, "height": canvas.height, "fonts_embedded": True, "vector_only": True})
    receipt = {"schema": "saccade-dataflow-figures/1", "date": "2026-11-12",
               "source_commit": "3cbaa518f0dc37d691404e95ed85c0109c81130d",
               "source_pin_verified": True, "generator": "render_figures.py", "generator_sha256": sha(Path(__file__)),
               "profile": "HR 1280x768/G1 v1-motion2/Graph ON; rings MI/CS/results=2/2/4",
               "sources": sources, "figures": figures, "new_GPU_measurement": False,
               "scope": "implementation-derived data dependencies, state publication and ownership; not timing or a new safety acceptance",
               "conversion_tool": subprocess.check_output(["rsvg-convert", "--version"], text=True).strip()}
    (HERE/"figures.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print("Rendered dataflow/state/lease SVG and PDF; embedded fonts and Chinese text verified.")


if __name__ == "__main__":
    main()
