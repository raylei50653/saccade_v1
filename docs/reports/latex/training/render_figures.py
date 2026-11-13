#!/usr/bin/env python3
"""Draw the training-report figures from saved CSV snapshots; no training, inference or scoring."""
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
NAVY, TEAL, GRAY, AMBER, INK, MUTED, GRID = "#16324F", "#087F8C", "#9AAABD", "#BB6A2B", "#203247", "#536578", "#DCE4EA"
LIGHT, PALE_TEAL, PALE_AMBER = "#F3F6F9", "#EFF7F8", "#FBF3EC"
INPUTS = {
    "teacher_ua2": DATA / "2026-11-02-hr-teacher-ua2.csv",
    "full25_epochs": DATA / "2026-11-03-full25-epochs.csv",
    "milestones": DATA / "milestones.csv",
}


class SVG:
    def __init__(self, width, height, title, desc):
        self.width, self.height, self.title, self.desc = width, height, title, desc
        self.items = []
        self.rect(0, 0, width, height, "#FFFFFF")

    def rect(self, x, y, w, h, fill, stroke="none", radius=0, width=1.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def text(self, x, y, text, size=22, color=INK, weight=400, anchor="start"):
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(text))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def poly(self, points, color, width=3, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')

    def dot(self, x, y, r, fill, stroke="#FFFFFF", width=2):
        self.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def arrow_h(self, x1, x2, y, color=NAVY):
        self.line(x1, y, x2 - 6, y, color, 3)
        self.items.append(f'<path d="M{x2-12:.1f},{y-7:.1f} L{x2:.1f},{y:.1f} L{x2-12:.1f},{y+7:.1f}" fill="none" stroke="{color}" stroke-width="3"/>')

    def arrow_v(self, x, y1, y2, color=NAVY, dash=None):
        self.line(x, y1, x, y2 - 6, color, 3, dash)
        self.items.append(f'<path d="M{x-7:.1f},{y2-12:.1f} L{x:.1f},{y2:.1f} L{x+7:.1f},{y2-12:.1f}" fill="none" stroke="{color}" stroke-width="3"/>')

    def save(self, path):
        body = "\n".join(self.items)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{html.escape(self.title)}</title><desc id="desc">{html.escape(self.desc)}</desc>'
            f'<g font-family="Noto Sans CJK TC, Noto Sans, sans-serif">{body}</g></svg>\n')


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def chain():
    c = SVG(1000, 640, "部署模型的兩半，各從哪個訓練階段來",
            "四個訓練階段的順序、各階段學什麼、選定 checkpoint 的 MOT17-10 val AP50，以及部署模型的 backbone 與 head 各來自哪一階段。")
    c.text(28, 42, "部署模型的兩半，各從哪個訓練階段來", 31, NAVY, 700)
    c.text(28, 80, "先把整個 YOLO26m 調成會找行人，再讓新的 Mamba head 模仿它，最後用標註答案修正。", 22, MUTED)
    c.rect(28, 104, 210, 46, LIGHT, GRID, 8)
    c.text(133, 134, "官方 YOLO26m 權重", 21, NAVY, 600, "middle")
    c.arrow_v(133, 150, 182)
    stages = [
        ("① Teacher", "微調整個 YOLO26m", "學會找 MOT 行人", "UA2 e4｜87.46%", "53.6 分鐘", NAVY),
        ("② Cache", "Teacher 凍結", "存下每幀特徵與輸出", "4,716 幀｜76 GB", "2.8 分鐘", MUTED),
        ("③ Distill", "Mamba head", "模仿 Teacher 的輸出", "e21｜81.94%", "24.8 分鐘", TEAL),
        ("④ GT1", "Mamba head", "改用標註答案修正", "Full25 e10｜86.01%", "26.7 分鐘", TEAL),
    ]
    for i, (name, who, what, result, minutes, color) in enumerate(stages):
        x = 28 + i * 240
        c.rect(x, 184, 210, 196, "#FFFFFF", color, 10, 2.5)
        c.text(x + 105, 220, name, 27, color, 700, "middle")
        c.text(x + 105, 254, who, 20, MUTED, anchor="middle")
        c.text(x + 105, 283, what, 21, INK, 600, "middle")
        c.line(x + 18, 302, x + 192, 302, GRID)
        c.text(x + 105, 333, result, 20, color, 700, "middle")
        c.text(x + 105, 362, minutes, 19, MUTED, anchor="middle")
        if i < 3:
            c.arrow_h(x + 213, x + 237, 282)
    c.arrow_v(133, 380, 434, NAVY, "6,5")
    c.arrow_v(853, 380, 434, TEAL, "6,5")
    c.text(493, 414, "部署時組合成一個模型", 19, MUTED, 600, "middle")
    c.rect(28, 436, 944, 112, PALE_TEAL, TEAL, 10)
    c.rect(48, 454, 380, 76, "#FFFFFF", NAVY, 8, 2)
    c.text(238, 486, "Backbone／neck（YOLO26m）", 22, NAVY, 700, "middle")
    c.text(238, 515, "取自 ① Teacher；③④ 期間凍結不動", 18, MUTED, anchor="middle")
    c.text(452, 500, "＋", 30, INK, 700, "middle")
    c.rect(478, 454, 290, 76, "#FFFFFF", TEAL, 8, 2)
    c.text(623, 486, "Mamba spatial head", 22, TEAL, 700, "middle")
    c.text(623, 515, "取自 ④ GT1 e10（EMA）", 18, MUTED, anchor="middle")
    c.arrow_h(772, 812, 492, NAVY)
    c.text(890, 484, "TensorRT", 21, NAVY, 700, "middle")
    c.text(890, 512, "FP16 engine", 21, NAVY, 700, "middle")
    c.text(28, 580, "資料：MOT17 五段訓練序列＋MOTSynth 合成影片（①、④）；③ 只讀 ② 的 MOT17 cache；④ 從 ③ 的選定權重接續。", 18, INK)
    c.text(28, 608, "百分比為 MOT17-10 val AP50（選模用的開發分數）；時間為單次正式 run 的牆鐘時間。部署 head 只看單一影格（T=1）。", 17, MUTED)
    return c


def data_split():
    c = SVG(1000, 640, "誰拿來訓練、誰拿來選、誰留到最後",
            "MOT17 七段依幀數比例分成訓練、選模與保留評估；MOTSynth 合成影片作為額外訓練資料；所有影像使用同一輸入前處理。")
    c.text(28, 42, "誰拿來訓練、誰拿來選、誰留到最後", 31, NAVY, 700)
    c.text(28, 80, "以「整段序列」為單位隔離：選模與最終評估的每一幀都沒有進入任何訓練階段。", 22, MUTED)
    seqs = [("02", 600, "eval"), ("04", 1050, "train"), ("05", 837, "train"), ("09", 525, "train"),
            ("10", 654, "val"), ("11", 900, "train"), ("13", 750, "train")]
    fills = {"train": TEAL, "val": AMBER, "eval": GRAY}
    total = sum(n for _, n, _ in seqs)
    x, x0, width = 28.0, 28, 944
    c.text(28, 122, "MOT17（真實街景，七段、5,316 幀；方塊寬度＝幀數）", 22, NAVY, 700)
    for seq, frames, role in seqs:
        w = width * frames / total
        c.rect(x + 1, 136, w - 2, 62, fills[role], radius=5)
        c.text(x + w / 2, 166, f"MOT17-{seq}", 19 if w > 110 else 16, "#FFFFFF", 700, "middle")
        c.text(x + w / 2, 189, f"{frames:,} 幀", 16, "#FFFFFF", anchor="middle")
        x += w
    legend = [("訓練", TEAL, "04／05／09／11／13：4,062 幀、73,762 個框"),
              ("選模", AMBER, "10：654 幀、11,117 個框（移動鏡頭、夜間）"),
              ("最後才開", GRAY, "02：600 幀、10,993 個框（固定鏡頭、白天）")]
    for i, (name, color, note) in enumerate(legend):
        y = 236 + i * 34
        c.rect(28, y - 17, 22, 18, color, radius=3)
        c.text(60, y, name, 21, color if color != GRAY else MUTED, 700)
        c.text(160, y, note, 20, INK)
    c.rect(28, 350, 944, 124, PALE_AMBER, "#E8D3BF", 10)
    c.text(48, 384, "MOTSynth（遊戲引擎合成的街景行人）", 22, AMBER, 700)
    c.text(48, 416, "764 支 1080p 影片 → NVDEC 解碼、每 20 幀取 1 → 68,760 張 JPEG；與 MOT17 同一套標註篩選。", 20, INK)
    c.text(48, 448, "Teacher：每個 MOT17 批次後接一個 MOTSynth 批次（1:1）；學生 GT1：每次更新 12＋4 張（3:1）。", 20, INK)
    c.text(28, 512, "所有影像同一條前處理（訓練、cache、選模、匯出參考共用；部署 F01／F02 實作同一規則）", 22, NAVY, 700)
    steps = ["JPEG", "nvJPEG 解成 RGB", "÷255 轉 float32", "抗鋸齒雙線性", "1280×768 輸入"]
    for i, step in enumerate(steps):
        bx = 28 + i * 192
        c.rect(bx, 530, 168, 54, LIGHT if i in (0, 4) else "#FFFFFF", GRID, 8)
        c.text(bx + 84, 564, step, 19, NAVY, 600, "middle")
        if i < 4:
            c.arrow_h(bx + 170, bx + 189, 557)
    c.text(28, 618, "直接拉伸（stretch），不補灰邊；不取整、不做 mean／std 正規化。前處理只要差一點，特徵就會差好幾個百分比。", 18, MUTED)
    return c


def curves(teacher_rows, epoch_rows):
    c = SVG(1000, 560, "為什麼不直接用最後一輪",
            "左：UA2 Teacher 每輪 val AP50，EMA 權重與原始權重；右：學生 GT1 加入 MOTSynth 與只用 MOT17 的每輪 val AP50。圓圈為選定 epoch。")
    c.text(28, 42, "為什麼不直接用最後一輪？", 31, NAVY, 700)
    c.text(28, 80, "每輪都在 MOT17-10 驗證；分數通常先升後降，所以依規則挑最佳一輪並提早停止。", 22, MUTED)

    def panel(x0, title, series, ymin, ymax, xmax, yticks):
        left, right, top, bottom = x0 + 64, x0 + 448, 156, 448
        c.text(x0 + 10, 126, title, 23, NAVY, 700)
        for v in yticks:
            y = bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
            c.line(left, y, right, y)
            c.text(left - 10, y + 7, f"{v:g}", 18, MUTED, anchor="end")
        for e in (range(1, xmax + 1) if xmax <= 10 else range(1, xmax + 1, 3)):
            x = left + (e - 1) / (xmax - 1) * (right - left)
            c.text(x, bottom + 28, e, 18, MUTED, anchor="middle")
        c.text((left + right) / 2, bottom + 58, "epoch", 18, MUTED, anchor="middle")
        c.text(x0 + 4, top - 14, "AP50 %", 17, MUTED)
        for points, color, width, selected in series:
            xy = [(left + (e - 1) / (xmax - 1) * (right - left), bottom - (v - ymin) / (ymax - ymin) * (bottom - top)) for e, v in points]
            c.poly(xy, color, width)
            for x, y in xy:
                c.dot(x, y, 4.5 if width >= 3 else 3.5, color, "#FFFFFF", 1.5)
            if selected is not None:
                sx, sy = xy[selected - 1]
                c.items.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="11" fill="none" stroke="{color}" stroke-width="3"/>')
                c.text(sx, sy - 20, f"選 e{selected}", 18, color, 700, "middle")
        return left, right, top, bottom

    ema = [(int(r["epoch"]), float(r["ap50"])) for r in teacher_rows if r["series"] == "ema"]
    raw = [(int(r["epoch"]), float(r["ap50"])) for r in teacher_rows if r["series"] == "raw"]
    selected = [int(r["epoch"]) for r in teacher_rows if r["series"] == "ema" and r["selected"] == "True"][0]
    left, right, top, bottom = panel(10, "Teacher（UA2）：EMA 與原始權重", [
        (raw, GRAY, 2.5, None),
        (ema, NAVY, 3.5, selected),
    ], 82, 88, 9, [82, 84, 86, 88])
    c.text(right - 6, top + 104, "EMA 0.996（選模）", 18, NAVY, 700, "end")
    c.text(right - 6, bottom - 8, "原始權重", 18, MUTED, 700, "end")

    full = [(int(r["epoch"]), 100 * float(r["ap50"])) for r in epoch_rows if r["arm"] == "full25"]
    ctrl = [(int(r["epoch"]), 100 * float(r["ap50"])) for r in epoch_rows if r["arm"] == "control"]
    best_full = max(full, key=lambda p: (p[1], p[0]))[0]
    left, right, top, bottom = panel(510, "學生 GT1：加不加 MOTSynth", [
        (ctrl, GRAY, 2.5, None),
        (full, TEAL, 3.5, best_full),
    ], 83.5, 86.5, 16, [84, 85, 86])
    c.text(right - 6, top + 112, "加 MOTSynth（部署）", 18, TEAL, 700, "end")
    c.text(right - 6, bottom - 8, "只用 MOT17（對照）", 18, MUTED, 700, "end")
    c.text(28, 534, "兩邊都是「連續 5 輪沒有更好就停」；左圖第 9 輪停止，右圖加 MOTSynth 第 15 輪、對照第 16 輪停止。", 19, MUTED)
    return c


def progress(rows):
    c = SVG(1000, 822, "一路改了什麼，分數各動了多少",
            "九個依序的訓練改動，Teacher 與部署學生的 MOT17-10 val AP50；虛線為官方 YOLO26m 未訓練時的參考。橫軸從 60 開始。")
    c.text(28, 42, "一路改了什麼，分數各動了多少", 31, NAVY, 700)
    c.text(28, 80, "MOT17-10 val AP50（%），依時間排列；Teacher 同配方只換 seed，選定點就差 0.5–0.65。", 22, MUTED)
    left, right, top = 400, 830, 160
    row_h = 62
    xmin, xmax = 60, 90
    bottom = top + row_h * len(rows)

    def xp(v):
        return left + (v - xmin) / (xmax - xmin) * (right - left)

    for v in range(60, 91, 5):
        c.line(xp(v), top - 14, xp(v), bottom - 10)
        c.text(xp(v), bottom + 16, v, 18, MUTED, anchor="middle")
    for v, label, anchor, dx in [(66.19, "官方未訓練 640", "end", -6), (72.71, "官方未訓練 1280×768", "start", 6)]:
        c.line(xp(v), top - 22, xp(v), bottom - 10, AMBER, 2, "5,5")
        c.text(xp(v) + dx, top - 20, label, 15, AMBER, 600, anchor)
    c.dot(36, 106, 7, NAVY)
    c.text(50, 112, "Teacher", 18, NAVY, 700)
    c.dot(146, 106, 7, TEAL)
    c.text(160, 112, "學生（部署用 Mamba head）", 18, TEAL, 700)
    c.text(912, 140, "T｜學生", 17, MUTED, 600, "middle")
    for i, row in enumerate(rows):
        y = top + i * row_h + 18
        if i % 2 == 0:
            c.rect(20, y - 30, 960, row_h, "#F7F9FB")
        c.text(28, y - 4, row["label"], 19, NAVY, 700)
        c.text(28, y + 21, row["date"][5:], 15, MUTED)
        t = float(row["teacher_ap50"]) if row["teacher_ap50"] else None
        s = float(row["student_ap50"]) if row["student_ap50"] else None
        if t is not None and s is not None:
            c.line(xp(min(t, s)), y, xp(max(t, s)), y, GRID, 4)
        if t is not None:
            c.dot(xp(t), y, 8, NAVY)
        if s is not None:
            c.dot(xp(s), y, 8, TEAL, "#FFFFFF", 2)
        tt = f"{t:.2f}" if t is not None else "—"
        ss = f"{s:.2f}" if s is not None else "—"
        c.text(890, y + 7, tt, 18, NAVY if t is not None else MUTED, 700, "end")
        c.text(904, y + 7, "｜", 18, MUTED, anchor="middle")
        c.text(918, y + 7, ss, 18, TEAL if s is not None else MUTED, 700, "start")
        if i == len(rows) - 1:
            c.rect(16, y - 32, 968, row_h + 4, "none", TEAL, 8, 2.5)
    c.text(28, bottom + 46, "640 兩列與 1280×768 列的輸入尺寸不同；T01 的 Teacher 依原規則固定 e12（它的高點 e2 為 75.82）。", 17, MUTED)
    c.text(28, bottom + 72, "「—」表示該步驟沒有重訓這一半；部署組合＝UA2 Teacher 的 backbone＋Full25 學生的 head。", 17, MUTED)
    return c


def deploy():
    c = SVG(1000, 520, "從訓練權重到 TensorRT engine",
            "匯出與部署驗收的四個步驟，以及每一步的檢查結果。")
    c.text(28, 42, "從訓練權重到 TensorRT engine", 31, NAVY, 700)
    c.text(28, 80, "PyTorch 只用在訓練與匯出；部署端是 C++／CUDA／TensorRT，每一步都先對數字再往下走。", 22, MUTED)
    steps = [
        ("① Backbone", "→ ONNX", ["特徵與訓練 cache 的", "51,118,080 個元素", "FP16 捨入後逐位元相同"]),
        ("② Head 權重", "→ SWB bundle", ["轉成原生格式，由", "自寫的 scan plugin", "在 TensorRT 內執行"]),
        ("③ FP32 engine", "native 錄製", ["AP50 86.0092%", "訓練紀錄 86.0093%", "門檻 ±0.1 pp：PASS"]),
        ("④ FP16 engine", "部署版本", ["AP50 86.0156%", "模型 25.7 → 6.4 ms", "同 engine 重跑一致"]),
    ]
    for i, (name, sub, lines) in enumerate(steps):
        x = 28 + i * 240
        color = TEAL if i == 3 else NAVY
        c.rect(x, 108, 210, 250, PALE_TEAL if i == 3 else "#FFFFFF", color, 10, 2.5)
        c.text(x + 105, 146, name, 25, color, 700, "middle")
        c.text(x + 105, 176, sub, 19, MUTED, anchor="middle")
        c.line(x + 18, 196, x + 192, 196, GRID)
        for j, text in enumerate(lines):
            c.text(x + 105, 234 + j * 34, text, 18, INK, 600 if j == 0 else 400, "middle")
        if i < 3:
            c.arrow_h(x + 213, x + 237, 232)
    c.rect(28, 384, 944, 106, LIGHT, radius=10)
    c.text(48, 418, "驗收用同一個尺：MOT17-10 的 AP50，必須落在訓練時記錄值 ±0.1 個百分點內。", 20, NAVY, 700)
    c.text(48, 450, "FP16 只用在卷積等運算；scan plugin、模型輸入與六個原始輸出維持 FP32。", 19, INK)
    c.text(48, 478, "部署 engine 以 SHA-256 8292cc23…7876 固定；之後重 build 就是新候選，要重做驗收。", 19, MUTED)
    return c


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def milestone_table(rows):
    """Write the appendix table from the same CSV the progress figure uses."""
    def tex(text):
        return text.replace("\\", "\\textbackslash{}").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")

    def cell(value):
        return f"{float(value):.2f}" if value else "—"
    lines = [r"\begin{tabularx}{\textwidth}{@{}l P{3.6cm} Y r r@{}}", r"\toprule",
             r"日期 & 步驟 & 比上一列改了什麼 & Teacher & 學生 \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['date'][5:]} & {tex(row['label'])} & {tex(row['change'])} & {cell(row['teacher_ap50'])} & {cell(row['student_ap50'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (HERE / "milestone-table.tex").write_text("% Generated by render_figures.py from data/milestones.csv\n" + "\n".join(lines) + "\n")


def main():
    ASSETS.mkdir(exist_ok=True)
    figures = {
        "train-chain": chain(),
        "train-data": data_split(),
        "train-curves": curves(read_csv(INPUTS["teacher_ua2"]), read_csv(INPUTS["full25_epochs"])),
        "train-progress": progress(read_csv(INPUTS["milestones"])),
        "train-deploy": deploy(),
    }
    outputs = {}
    for name, figure in figures.items():
        svg, pdf = ASSETS / f"{name}.svg", ASSETS / f"{name}.pdf"
        figure.save(svg)
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(svg)], check=True)
        fonts = subprocess.run(["pdffonts", str(pdf)], check=True, capture_output=True, text=True).stdout
        if any(line.split()[-5:-4] == ["no"] for line in fonts.splitlines()[2:]):
            raise SystemExit(f"{pdf} has a font that is not embedded")
        outputs[name] = {"svg": sha256(svg), "pdf": sha256(pdf)}
    milestone_table(read_csv(INPUTS["milestones"]))
    outputs["milestone-table.tex"] = sha256(HERE / "milestone-table.tex")
    receipt = {
        "schema": "training-report-figures/1",
        "scope": "Drawn from saved CSV snapshots; no training, inference, scoring or tuning.",
        "inputs": {key: {"path": str(path.relative_to(HERE)), "sha256": sha256(path)} for key, path in INPUTS.items()},
        "outputs": outputs,
    }
    (HERE / "figures.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
