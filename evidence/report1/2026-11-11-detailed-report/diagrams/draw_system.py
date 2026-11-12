"""Generate the report's manually laid out system architecture SVG."""
from pathlib import Path
from svg_helpers import Canvas, BG, WHITE, INK, MUTED, BLUE, TEAL, INDIGO, LINE

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/reports/assets/report1-3cbaa518f0dc/system.svg'
c = Canvas(1600, 1280, 'Saccade v1：原生 GPU 追蹤系統架構',
           '固定基準3cbaa518f0dc。JPEG輸入經D偵測與M追蹤兩條stream；G1 NVOFA使用上一提交的gray和mask；成功提交後輸出MotionContext和ResultPacket。')


def card(x, y, w, h, title, body, color=BLUE, tag=None):
    c.rect(x, y, w, h, WHITE, LINE, rx=14)
    c.rect(x, y + 18, 4, h - 36, color, rx=2)
    c.text(x + 18, y + 32, title, 21, INK, 650)
    c.lines(x + 18, y + 62, body, 17, MUTED, 26)
    if tag:
        c.text(x + 18, y + h - 15, tag, 15, color, 600)


c.text(64, 42, 'FIGURE 01  /  SYSTEM ARCHITECTURE', 16, BLUE, 650)
c.text(1536, 42, 'BASELINE  3cbaa518f0dc', 16, MUTED, 500, 'end')
c.text(64, 94, '原生 GPU 追蹤：從影像到穩定身份', 36, INK, 700)
c.text(64, 132, 'JPEG 主線  ·  HR 1280 × 768  ·  G1 NVOFA / v1-motion/2  ·  CUDA Graph ON', 20, MUTED)
c.rect(64, 378, 1472, 260, '#EDF3FF', '#CCDDFB', rx=20)
c.rect(64, 716, 1472, 308, '#EAF6F4', '#C1E4DD', rx=20)

# Input lane: the decode backend is separate from the D/M compute streams.
c.rect(64, 174, 1472, 160, WHITE, LINE, rx=20)
c.text(88, 205, '01  輸入與解碼', 19, INK, 650)
card(88, 222, 260, 88, 'CPU · JPEG 讀取', ['壓縮 bytes → pinned staging'], MUTED)
card(380, 222, 300, 88, 'F01 · nvJPEG', ['HARDWARE decode＋色彩轉換'], MUTED)
card(712, 222, 300, 88, 'GPU · planar RGB8', ['原圖尺寸；R／G／B 三平面'], MUTED)
c.line(348, 266, 378, 266, MUTED)
c.line(680, 266, 710, 266, MUTED)
c.text(1046, 249, '影像、候選與追蹤狀態留在 GPU', 21, INK, 600)
c.text(1046, 282, 'CPU 負責提交、接收與輸出定稿', 18, MUTED)

# The output RGB enters D before F02; a labelled return path avoids node overlap.
c.path('M 862 310 V 350 H 76 V 520 H 86', BLUE, 2.5)
c.rect(364, 337, 140, 27, BG, rx=8)
c.text(376, 357, 'RGB ready', 17, BLUE, 550)

# Detect lane and the bounded D Graph.
c.text(88, 413, '02  D · Detect stream', 23, BLUE, 650)
c.text(1512, 411, '模型與候選生成', 18, BLUE, 500, 'end')
c.rect(590, 434, 922, 180, '#E2ECFF', '#B8CDF7', rx=16)
c.text(610, 459, 'D CUDA GRAPH', 15, BLUE, 700)
c.line(318, 521, 348, 521, BLUE, 2.5)
c.line(565, 521, 608, 521, BLUE, 2.5)
c.line(970, 530, 998, 530, BLUE, 2.5)
c.line(1210, 530, 1238, 530, BLUE, 2.5)
card(88, 454, 230, 132, 'F02 · 模型前處理', ['正規化＋抗鋸齒縮放', 'FP32 NCHW', '1 × 3 × 768 × 1280'])
card(350, 454, 215, 132, 'F05 · 灰階 MI', ['直接取自原圖 RGB', '1080p 時 s = 1', '獨立 motion slot'])
card(610, 474, 360, 122, 'F03 · TensorRT 模型', ['YOLO26m backbone／neck', 'Spatial scan head → 6 raw outputs'])
card(1000, 474, 210, 122, 'Producer', ['finite checks', '9 plugin statuses'])
card(1240, 474, 248, 122, 'F04 · 候選整理', ['decode → top-300 → NMS', 'kept ＋ suppressed'])

# The two explicit handoff channels occupy their own gutters.
c.path('M 457 586 V 664 H 50 V 842 H 86', TEAL, 2.5)
c.rect(252, 648, 260, 30, BG, rx=8)
c.text(266, 669, 'mi_ready(k) → F06 可先開始', 17, TEAL, 550)
c.path('M 1364 596 V 686 H 732 V 782', BLUE, 2.5)
c.rect(856, 671, 328, 30, BG, rx=8)
c.text(870, 692, 'candidates_ready(k) → 等候同幀候選', 17, BLUE, 550)

# Main lane. Mask creation is outside the tracking graph.
c.text(88, 752, '03  M · 有序 main stream', 23, TEAL, 650)
c.text(1512, 750, '依 frame 順序推進狀態', 18, TEAL, 500, 'end')
c.rect(846, 770, 666, 204, '#DDF0EC', '#A9D7CE', rx=16)
c.text(866, 797, 'M CUDA GRAPH', 15, TEAL, 700)
c.line(358, 870, 386, 870, TEAL, 2.5)
c.line(613, 870, 640, 870, TEAL, 2.5)
c.line(822, 870, 864, 870, TEAL, 2.5)
c.line(1026, 876, 1048, 876, TEAL, 2.5)
c.line(1320, 876, 1342, 876, TEAL, 2.5)
card(88, 784, 270, 172, 'F06 · G1 NVOFA', ['上一 gray ＋ 本幀 MI', 'OFA → cells → 中位數', '排除上一幀 person mask'], TEAL, 'SLOW  /  grid 4  /  cell 8')
card(388, 784, 225, 172, 'F08a · 預測', ['GMC control', '一次 Kalman predict', '產生本幀 prior'], TEAL)
card(642, 784, 180, 172, '本幀 mask', ['kept person', 'score ≥ 0.45', '供下一幀使用'], TEAL)
card(866, 812, 160, 144, 'F04b', ['候選＋prior', 'suppressed：', '續追不 birth'], TEAL)
card(1050, 812, 270, 144, 'F07 / F08b · 追蹤', ['同幀核對＋auction', 'update／Bridge／birth'], TEAL)
card(1344, 812, 144, 144, '提交與封包', ['commit', 'packet'], TEAL)

# Two independent outgoing edges: version publication and host delivery.
c.path('M 1344 928 H 1332 V 994 H 350 V 1060', INDIGO, 2.2, '6 5')
c.rect(644, 979, 490, 29, '#EAF6F4', rx=8)
c.text(657, 1000, '成功 commit 才一起發布 TrackState／gray／mask', 17, INDIGO, 550)
c.path('M 1416 956 V 1040 H 910 V 1108', TEAL, 2.5)

# GPU state feedback is separate from output and uses dashed version edges.
c.path('M 64 1134 H 38 V 870 H 86', INDIGO, 2.2, '6 5')
c.rect(64, 1060, 620, 150, '#F0EDFF', '#D4CCF8', rx=20)
c.text(88, 1096, 'GPU 跨幀狀態', 23, INDIGO, 650)
c.text(88, 1130, 'committed ／ working 雙版本', 20, INK, 550)
c.text(88, 1164, 'TrackState ＋ gray ＋ mask ＋ Control', 18, MUTED)
c.text(88, 1190, '上一 committed 版本供 F06／F08a；失敗不發布新版本', 17, INDIGO)

c.rect(714, 1060, 822, 150, WHITE, LINE, rx=20)
c.text(738, 1094, '04  結果交付與 F09', 22, INK, 650)
card(738, 1108, 350, 80, 'M 上依序兩次 D2H', ['MotionContext 56 B ＋ packet 12,368 B'], TEAL)
card(1120, 1108, 392, 80, 'CPU receive → MOT', ['驗證 → 序列尾插值／ID 定稿 → 寫檔'], MUTED)
c.line(1088, 1148, 1118, 1148, MUTED, 2.5)

c.text(64, 1246, '同幀 F03 可與 F06／predict 重疊；D(k+1) 可與 M(k) tail 重疊。', 18, INK, 500)
c.text(1536, 1246, '色塊表示執行責任；連線表示資料或執行依賴，非時間比例。', 17, MUTED, 400, 'end')
c.save(OUT)
