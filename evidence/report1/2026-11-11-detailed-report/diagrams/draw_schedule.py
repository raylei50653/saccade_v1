#!/usr/bin/env python3
"""Draw the conceptual stream/ownership schedule as native editable SVG."""
from pathlib import Path
from svg_helpers import Canvas, BG, INK, MUTED, BLUE, TEAL, INDIGO, LINE, WHITE


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "docs/reports/assets/report1-3cbaa518f0dc/schedule.svg"
C = Canvas(1600, 1150, "Saccade v1｜執行順序與交付邊界",
           "基準 3cbaa518f0dc 的概念時序圖：D 偵測與 M 運動估計同幀重疊，"
           "候選完成後才擷取 mask 並執行 tracking tail，成功 commit 後兩次 D2H 再 receive。"
           "所有寬度僅表示依賴順序，不表示量測延遲。")

C.text(64, 42, "SACCADE V1  /  FIGURE 03", 16, BLUE, 700)
C.text(64, 90, "執行順序與交付邊界", 36, INK, 700)
C.text(64, 128, "以 ready／consumed 事件連接雙 stream；同幀運動估計與偵測可重疊。", 19, MUTED)
C.text(1536, 44, "BASELINE  3cbaa518f0dc", 16, MUTED, 500, "end")
C.rect(1178, 69, 358, 45, "#EAF0FB", rx=22)
C.text(1357, 99, "依賴示意，非量測時間比例", 18, BLUE, 600, "middle")

# Legend and the four aligned lanes.
for x, color, label in ((64, BLUE, "D：偵測"), (234, TEAL, "M：有序狀態更新"), (514, INDIGO, "Host：提交與交付")):
    C.rect(x, 156, 10, 10, color, rx=3)
    C.text(x + 21, 167, label, 17, MUTED)

C.rect(48, 196, 1504, 654, WHITE, LINE, 24)
C.text(274, 232, "時間順序", 17, MUTED, 600)
C.line(370, 226, 1480, 226, LINE, sw=2)

for y, h, fill in ((250, 104, "#F5F3FF"), (366, 142, "#EFF5FF"),
                   (548, 142, "#EDF9F7"), (730, 96, "#F5F3FF")):
    C.rect(64, y, 1472, h, fill, rx=14)

C.text(88, 287, "CPU caller", 22, INDIGO, 700)
C.text(88, 319, "提交與輸入控制", 17, MUTED)
C.text(88, 413, "Detect · D", 22, BLUE, 700)
C.text(88, 445, "F01–F05 / D Graph", 17, MUTED)
C.text(88, 595, "Main · M", 22, TEAL, 700)
C.text(88, 627, "依 frame 有序", 17, MUTED)
C.text(88, 768, "Host result", 22, INDIGO, 700)
C.text(88, 799, "接收與後處理", 17, MUTED)

# Edges go behind all cards.  Inter-lane events are orthogonal.
C.path("M 390 338 V 386", INDIGO, 2.3)
C.line(445, 437, 465, 437, BLUE, 2.3)
C.line(575, 437, 600, 437, BLUE, 2.3)
C.line(890, 437, 910, 437, BLUE, 2.3)
C.line(1120, 437, 1150, 437, BLUE, 2.3)
C.path("M 520 492 V 526 H 620 V 568", TEAL, 2.5)
C.path("M 1015 492 V 524 H 1172 V 568", BLUE, 2.5)
C.line(805, 619, 825, 619, TEAL, 2.3)
C.line(1005, 619, 1120, 619, TEAL, 2.3, dash="5 5")
C.line(1225, 619, 1245, 619, TEAL, 2.3)
C.path("M 1362 674 V 744", INDIGO, 2.5)

def card(x, y, w, h, color, title, lines, size=18):
    C.rect(x, y, w, h, WHITE, color, 12, 1.4)
    C.rect(x, y, 5, h, color, rx=2)
    C.text(x + w / 2, y + 31, title, 20, color, 700, "middle")
    for i, text in enumerate(lines):
        C.text(x + w / 2, y + 60 + i * 25, text, size, INK, 400, "middle")

card(270, 266, 240, 72, INDIGO, "enqueue(k)", ["JPEG bytes / staging"], 17)
C.text(566, 286, "下一次 JPEG decode 前，host 等 rgb_read_done", 18, INK, 500)
C.text(566, 318, "原圖 RGB 的最後 F02／F05 reader 完成後才可重用。", 17, MUTED)

card(270, 386, 175, 106, BLUE, "F01 / F02", ["decode · resize", "GPU RGB → images"], 17)
card(465, 386, 110, 106, BLUE, "F05", ["灰階 MI", "獨立 slot"], 17)
card(600, 386, 290, 106, BLUE, "F03 · TensorRT", ["YOLO26m + scan head", "D Graph"], 18)
card(910, 386, 210, 106, BLUE, "producer → F04", ["finite checks · NMS", "D Graph"], 17)
C.rect(1150, 386, 330, 106, "#E4EDFF", "#B2C9F5", 12)
C.text(1315, 419, "D(k+1) 可開始", 20, BLUE, 700, "middle")
C.text(1315, 449, "與 M(k) tracking tail 重疊", 17, BLUE, 400, "middle")
C.text(1315, 475, "依各 slot 的 consumed 保護重用", 17, MUTED, 400, "middle")

C.rect(604, 510, 104, 27, WHITE, rx=8)
C.text(616, 529, "mi_ready(k)", 17, TEAL, 600)
C.rect(958, 510, 219, 27, WHITE, rx=8)
C.text(970, 529, "candidates_ready(k)", 17, BLUE, 600)

card(600, 568, 205, 106, TEAL, "F06 · G1", ["NVOFA → median", "previous gray / mask"], 17)
card(825, 568, 180, 106, TEAL, "F08a", ["GMC control", "一次 Kalman predict"], 17)
C.text(1060, 597, "候選等待", 17, MUTED, 400, "middle")
card(1120, 568, 105, 106, TEAL, "Mask(k)", ["本幀遮罩", "供下幀用"], 17)
card(1245, 568, 235, 106, TEAL, "M tracking tail", ["F04b → F07 / F08b", "commit → packet"], 17)

card(1245, 744, 235, 70, INDIGO, "D2H → receive(k)", ["56 B + 12,368 B"], 17)
C.text(288, 767, "成功 commit 同時發布 TrackState、gray 與 mask 的新版本。", 18, INK, 500)
C.text(288, 799, "M tail 之後依序複製 MotionContext、ResultPacket；完成事件之後 host 才讀取。", 17, MUTED)

# Three concise ownership notes make the event semantics explicit without
# drawing return arrows across the timed lanes.
notes = [
    (64, "01", "同幀重疊", ["F05 發布獨立 MI 後，M 上的 F06／predict", "可與 D 上的 F03／F04 同時執行。", "OFA 是固定功能引擎，前後處理仍使用 CUDA。"], BLUE),
    (568, "02", "有限 slot 的交接", ["MI 與 candidates 各有 ready／consumed。", "M 完成最後讀取後發布 consumed，D 才重用。", "結果 ring 滿時，以背壓保留待接收結果。"], TEAL),
    (1072, "03", "跨幀狀態邊界", ["D(k+1) 可與 M(k) 的 tracking tail 重疊。", "M(k+1) 不得越過 commit(k)。", "失敗不發布 working state；首次錯誤保留。"], INDIGO),
]
for x, number, title, detail, color in notes:
    C.rect(x, 878, 464, 168, WHITE, LINE, 18)
    C.text(x + 24, 916, number, 17, color, 700)
    C.text(x + 65, 916, title, 22, INK, 700)
    C.lines(x + 24, 954, detail, 17, MUTED, 29)

C.line(64, 1082, 1536, 1082, LINE, 1, arrow=False)
C.text(64, 1116, "G1 / v1-motion/2   ·   HR 1280×768   ·   CUDA Graph ON", 17, MUTED)
C.text(1536, 1116, "來源：TrackingSession 實作；區塊寬度不代表延遲或硬體利用率。", 17, MUTED, 400, "end")

if __name__ == "__main__":
    C.save(OUTPUT)
    print(OUTPUT)
