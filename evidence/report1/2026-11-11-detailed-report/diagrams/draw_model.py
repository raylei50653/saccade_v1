#!/usr/bin/env python3
"""Draw the fixed report baseline's F03 model with editable SVG primitives."""
from pathlib import Path
from svg_helpers import BG, BLUE, INK, INDIGO, LINE, MUTED, TEAL, WHITE, Canvas


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / 'docs/reports/assets/report1-3cbaa518f0dc/model.svg'


def draw():
    c = Canvas(1600, 1190, 'F03 模型架構｜HR 三尺度 spatial head',
               '固定基準 3cbaa518f0dc。FP32 影像輸入經 YOLO26m backbone/neck 與三尺度 Mamba spatial head，'
               '輸出六個 FP32 raw tensors。TensorRT 使用 FP16 tactics，scan plugins 維持 FP32；T=1，每次 scan 零初始狀態。')
    c.text(64, 61, 'SACCADE V1  /  FIGURE 02', 17, BLUE, 700)
    c.text(64, 114, 'F03 模型架構', 36, INK, 700)
    c.text(64, 151, 'HR 1280×768 · 三尺度 spatial head · batch 1', 20, MUTED)
    c.rect(1228, 47, 308, 48, WHITE, LINE, rx=24)
    c.text(1382, 78, 'BASELINE  3cbaa518f0dc', 17, MUTED, 600, 'middle')

    # Main engine area. All tensor shapes use logical NCHW ordering.
    c.rect(40, 185, 1520, 555, '#EDF2FA', rx=24)
    c.text(64, 221, 'TENSORRT ENGINE', 16, BLUE, 700)
    c.text(1536, 221, 'D detect stream · CUDA Graph', 17, MUTED, 500, 'end')

    # Routed connectors precede foreground cards.
    c.line(308, 460, 346, 460, BLUE, 2.5)
    c.line(616, 460, 650, 460, BLUE, 2.5, arrow=False)
    c.line(650, 310, 650, 650, BLUE, 2.5, arrow=False)
    for center, color in ((310, TEAL), (480, BLUE), (650, INDIGO)):
        c.line(650, center, 680, center, color, 2.5)
        c.line(880, center, 918, center, color, 2.5)
        c.line(1182, center, 1220, center, color, 2.5)

    c.rect(64, 353, 244, 214, WHITE, LINE, rx=20)
    c.rect(84, 373, 66, 29, '#EAF1FF', rx=8)
    c.text(117, 394, 'INPUT', 14, BLUE, 700, 'middle')
    c.text(84, 438, '模型輸入', 24, INK, 700)
    c.text(84, 474, 'images · FP32', 18, BLUE, 600)
    c.text(84, 507, '[1,3,768,1280]', 20, INK, 500)
    c.text(84, 542, '由 F02 前處理交付', 17, MUTED)

    c.rect(346, 301, 270, 318, WHITE, LINE, rx=20)
    c.rect(366, 321, 230, 40, '#EAF1FF', rx=10)
    c.text(481, 347, '特徵擷取', 18, BLUE, 600, 'middle')
    c.text(370, 408, 'YOLO26m', 28, INK, 700)
    c.text(370, 446, 'backbone / neck', 23, INK, 500)
    c.line(370, 472, 592, 472, LINE, 1, arrow=False)
    c.lines(370, 507, ['TensorRT FP16 tactics', '多尺度特徵：P3 / P4 / P5'], 17, MUTED, 30)
    c.text(370, 588, '精度由實際 tactics 決定', 17, MUTED)

    branch_data = (
        ('P3', 246, TEAL, '#E9F7F4', '8', '[1,256,96,160]', '960', '[1,80,96,160]', '[1,4,96,160]'),
        ('P4', 416, BLUE, '#EBF2FF', '16', '[1,512,48,80]', '240', '[1,80,48,80]', '[1,4,48,80]'),
        ('P5', 586, INDIGO, '#EFEDFF', '32', '[1,512,24,40]', '60', '[1,80,24,40]', '[1,4,24,40]'),
    )
    for level, y, color, tint, stride, shape, length, cls, reg in branch_data:
        c.rect(680, y, 200, 128, WHITE, color, rx=16, sw=1.5)
        c.rect(694, y+14, 48, 28, tint, rx=8)
        c.text(718, y+34, level, 17, color, 700, 'middle')
        c.text(866, y+34, f'stride {stride}', 16, MUTED, 500, 'end')
        c.text(694, y+75, 'Feature map', 20, INK, 600)
        c.text(694, y+108, shape, 17, MUTED)

        c.rect(918, y, 264, 128, WHITE, LINE, rx=16)
        c.rect(918, y, 7, 128, color, rx=3)
        c.text(938, y+37, f'{level} Mamba spatial head', 20, INK, 700)
        c.text(938, y+72, f'4 directions · L={length}', 17, MUTED)
        c.text(938, y+106, 'FP32 scan · 3 plugin statuses', 16, color, 500)

        c.rect(1220, y, 316, 128, WHITE, LINE, rx=16)
        c.text(1240, y+35, 'FP32 raw outputs', 19, color, 700)
        c.text(1240, y+73, f'cls_{level.lower()}   {cls}', 18, INK, 500)
        c.text(1240, y+106, f'reg_{level.lower()}   {reg}', 18, INK, 500)

    # Summary band distinguishes output payload from status metadata.
    c.rect(64, 764, 694, 60, WHITE, LINE, rx=14)
    c.text(84, 801, '6 個 raw tensors · 20,160 positions · 6,773,760 B', 19, INK, 600)
    c.rect(778, 764, 758, 60, '#EAF7F6', '#BEE4E0', rx=14)
    c.text(800, 801, '9 個 plugin statuses ＋ input/raw finite checks → producer', 18, TEAL, 600)

    c.rect(64, 850, 1472, 253, WHITE, LINE, rx=22)
    c.text(88, 889, '每尺度 head 的內部路徑', 24, INK, 700)
    c.text(88, 921, '各尺度獨立權重 · 下圖省略 token 分支內的 gate、projection 與殘差細節', 17, MUTED)

    steps = (
        (88, 268, '01', '投影與降採樣', ['Conv1×1 → 128 channels', 'Conv4×4 / stride 4']),
        (376, 268, '02', '四方向 Cross-Scan', ['Reduced map → tokens', 'L = 960 / 240 / 60']),
        (664, 268, '03', 'Mamba selective scan', ['Causal DWConv1d + gate', 'state 16 · FP32 scan']),
        (952, 268, '04', 'Merge 與上採樣', ['Conv + PixelShuffle 4', '回到原 feature 尺寸']),
        (1240, 272, '05', 'Concat 與預測', ['與 input projection concat', 'Cls 80 / Reg 4 channels']),
    )
    for x, w, num, title, body in steps:
        if x != 88:
            c.line(x-20, 1000, x-3, 1000, '#A6B3C7', 2)
        c.rect(x, 943, w, 128, '#F5F7FB', rx=14)
        c.text(x+16, 972, num, 15, INDIGO, 700)
        c.text(x+16, 1002, title, 19, INK, 600)
        c.lines(x+16, 1031, body, 16, MUTED, 24)

    c.text(64, 1140, 'T=1：每次 scan 從零狀態開始；跨影格 TrackState 屬 F06／F08，非 Mamba temporal hidden state。', 17, MUTED)
    c.text(64, 1170, '精度註記：FP16 tactics 不代表全模型／全部中間量皆為 FP16；模型 I/O 與 scan plugins 維持 FP32。', 17, MUTED)
    c.save(OUTPUT)


if __name__ == '__main__':
    draw()
