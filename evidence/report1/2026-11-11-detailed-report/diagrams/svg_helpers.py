"""Small, dependency-free SVG drawing primitives for report figures."""
from html import escape
from pathlib import Path

BG = '#F5F7FB'
WHITE = '#FFFFFF'
INK = '#17263C'
MUTED = '#64748B'
BLUE = '#2563EB'
TEAL = '#0F8B8D'
INDIGO = '#4F46E5'
LINE = '#DCE3ED'
FONT = 'Noto Sans CJK TC, Noto Sans, sans-serif'


class Canvas:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.title, self.desc = title, desc
        self.items = []
        self.arrows = set()
        self.rect(0, 0, width, height, BG, rx=0)

    def rect(self, x, y, w, h, fill, stroke='', rx=16, sw=1):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def text(self, x, y, text, size=18, fill=INK, weight=400, anchor='start'):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{fill}" text-anchor="{anchor}">{escape(str(text))}</text>')

    def lines(self, x, y, lines, size=18, fill=MUTED, line_height=28, weight=400):
        for i, text in enumerate(lines):
            self.text(x, y + i * line_height, text, size, fill, weight)

    def path(self, d, color=LINE, sw=2, dash=None, arrow=True):
        marker = ''
        if arrow:
            self.arrows.add(color)
            marker = f' marker-end="url(#arrow-{color.lstrip("#")})"'
        dashed = f' stroke-dasharray="{dash}"' if dash else ''
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
                          f'stroke-linejoin="round" stroke-linecap="round"{dashed}{marker}/>')

    def line(self, x1, y1, x2, y2, color=LINE, sw=2, dash=None, arrow=True):
        self.path(f'M {x1} {y1} L {x2} {y2}', color, sw, dash, arrow)

    def save(self, path):
        markers = ''.join(
            f'<marker id="arrow-{color.lstrip("#")}" viewBox="0 0 10 10" '
            f'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            for color in sorted(self.arrows))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
               f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">\n'
               f'<title id="title">{escape(self.title)}</title>\n'
               f'<desc id="desc">{escape(self.desc)}</desc>\n'
               f'<defs>{markers}</defs>\n'
               f'<g font-family="{FONT}">\n' + '\n'.join(self.items) + '\n</g>\n</svg>\n')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(svg, encoding='utf-8')
