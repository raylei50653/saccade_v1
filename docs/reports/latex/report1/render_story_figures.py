#!/usr/bin/env python3
"""Draw explanatory SVG/PDF figures from saved results, without inference or scoring."""
import hashlib
import html
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SVG_DIR = ROOT / "docs/reports/assets/report1-3cbaa518f0dc"
PDF_DIR = HERE / "assets"
QUALITY = ROOT / "evidence/model-comparison/2026-11-11-yolo26m-primary156/bundle/metrics/summary.json"
SPEED = ROOT / "evidence/model-comparison/2026-11-11-yolo26m-trt-performance/performance/pipeline/summary.json"
NAVY, TEAL, GRAY, AMBER, INK, MUTED, GRID = "#16324F", "#087F8C", "#9AAABD", "#BB6A2B", "#203247", "#536578", "#DCE4EA"
ARMS = ["official", "finetuned", "main"]
COLORS = {"official": GRAY, "finetuned": NAVY, "main": TEAL}
LABELS = {"official": "官方 YOLO26m", "finetuned": "微調 YOLO26m", "main": "現行 HR Mamba"}


class SVG:
    def __init__(self, width, height, title, desc):
        self.width, self.height, self.title, self.desc = width, height, title, desc
        self.items = []
        self.rect(0, 0, width, height, "#FFFFFF")

    def rect(self, x, y, width, height, fill, stroke="none", radius=0):
        self.items.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def text(self, x, y, text, size=23, color=INK, weight=400, anchor="start"):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(text))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1.5):
        self.items.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def arrow(self, x1, x2, y):
        self.line(x1, y, x2-6, y, NAVY, 3)
        self.items.append(f'<path d="M{x2-12},{y-7} L{x2},{y} L{x2-12},{y+7}" fill="none" stroke="{NAVY}" stroke-width="3"/>')

    def save(self, path):
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(self.title)}</title><desc id="desc">{html.escape(self.desc)}</desc><g font-family="Noto Sans CJK TC, Noto Sans, sans-serif">'+"\n".join(self.items)+"</g></svg>\n")


def legend(c, y=91):
    for i, arm in enumerate(ARMS):
        x = 30 + i * 320
        c.rect(x, y-17, 24, 16, COLORS[arm], radius=3)
        c.text(x+34, y, LABELS[arm], 23)


def pipeline():
    c = SVG(1000, 415, "希望把『這裡有人』變成『這是同一個人』", "五步邏輯流程示意，非 GPU 計時或實際排程。")
    c.text(28, 43, "希望把『這裡有人』變成『這是同一個人』", 31, NAVY, 700)
    c.text(28, 82, "辨識每一幀的人，再把跨幀觀測接成持續的 ID。", 24, MUTED)
    c.rect(223, 117, 555, 161, "#EFF7F8", TEAL, 12)
    c.text(500, 144, "GPU：中間影像、候選與追蹤狀態留在裝置上", 22, TEAL, 600, "middle")
    titles = ["讀畫面", "找人", "估計相機位移", "接續 ID", "輸出"]
    subtitles = ["JPEG／串流", "人在哪裡？", "背景移了多少？", "與哪條軌跡相接？", "座標＋ID"]
    for i, (title, subtitle) in enumerate(zip(titles, subtitles)):
        x = 28 + i*194
        c.rect(x, 169, 168, 84, "#F3F6F9" if i in [0, 4] else "#FFFFFF", GRID, 9)
        c.text(x+84, 202, title, 25, NAVY, 700, "middle")
        c.text(x+84, 233, subtitle, 19, MUTED, anchor="middle")
        if i < 4: c.arrow(x+171, x+189, 211)
    c.text(28, 319, "CPU 負責讀入、排程、接收結果與寫檔；JPEG 解碼另走 GPU。", 23, INK)
    c.text(28, 354, "目的：減少反覆搬資料，讓偵測、相機運動補償與追蹤接在一起。", 23, INK)
    c.text(28, 391, "邏輯流程示意；框寬與順序不代表實際 GPU 耗時或跨 stream 排程。", 21, MUTED)
    return c


def datasets(quality):
    c = SVG(1000, 840, "現行方案在哪些資料集比較好？", "五資料集三模型合併 HOTA，橫軸從零開始，各模型使用共用追蹤流程與自身 detector-conditioned GMC。")
    c.text(28, 43, "現行方案在哪些資料集比較好？", 32, NAVY, 700)
    legend(c)
    x0, x1, maxval = 248, 864, 80
    for v in range(0, 81, 20):
        x = x0+(x1-x0)*v/maxval
        c.line(x, 129, x, 713)
        c.text(x, 746, v, 22, MUTED, anchor="middle")
    for i, name in enumerate(["MOT17", "MOT20", "MOT15", "DanceTrack", "PP22"]):
        y = 147+i*113
        row = quality["datasets"][name]
        c.text(28, y+22, name, 27, NAVY, 700)
        c.text(28, y+56, f'{row["sequence_count"]} 段／{row["frames"]:,} 幀', 20, MUTED)
        for j, arm in enumerate(ARMS):
            value = row["metrics"][arm]["HOTA"]
            width=(x1-x0)*value/maxval
            by=y+j*29
            c.rect(x0, by, width, 22, COLORS[arm], radius=3)
            c.text(x0+width+10, by+20, f"{value:.2f}", 22, MUTED if arm=="official" else COLORS[arm], 600)
    c.text(972, 746, "HOTA ↑", 23, INK, anchor="end")
    c.text(28, 789, "同一套追蹤流程；官方／微調為 raw6 one2many，非官方完整 tracker。", 21, MUTED)
    c.text(28, 821, "來源：primary156；156 段已曝露。PP22 幀數為完整序列，僅有 GT 影格計分。", 20, MUTED)
    return c


def group_deltas(quality):
    c = SVG(1000, 685, "相對微調原版：有進步，也有退步", "七個主要群組的 HR Mamba 減微調 YOLO26m 合併 HOTA 差值；正值較好，單位百分點。")
    c.text(28, 43, "相對微調原版：有進步，也有退步", 32, NAVY, 700)
    c.text(28, 86, "HR Mamba − 微調 YOLO26m；HOTA 差值，單位：百分點（pp）", 23, MUTED)
    x0, scale = 500, 44
    for tick in [-4,-2,0,2,4,6,8]:
        x=x0+tick*scale
        c.line(x, 122, x, 512, NAVY if tick==0 else GRID, 2.5 if tick==0 else 1.5)
        c.text(x, 548, f"{tick:+}" if tick else "0", 22, MUTED, anchor="middle")
    groups=["MOT17/all","MOT20/all","MOT15/moving","MOT15/other","DanceTrack/all","PP22/moving","PP22/static"]
    for i,name in enumerate(groups):
        y=138+i*54
        m=quality["primary_groups"][name]["metrics"]
        value=m["main"]["HOTA"]-m["finetuned"]["HOTA"]
        end=x0+value*scale
        color=TEAL if value>0 else AMBER
        c.text(28,y+24,name,24,NAVY)
        c.rect(min(x0,end),y,abs(value)*scale,32,color,radius=3)
        c.text(end+10 if value>0 else end-10,y+26,f"{value:+.2f}",24,color,700,"start" if value>0 else "end")
    c.text(320, 584, "← 較低",23,AMBER)
    c.text(852, 584, "較高 →",23,TEAL,anchor="end")
    c.rect(28, 606, 944, 48, "#F1F5F8", radius=8)
    c.text(50, 638, "七組：3 勝／4 負　｜　逐段：55 勝／101 負（共 156 段）", 25, NAVY, 600)
    c.text(28, 680, "來源：primary156；整組改善不表示每段都改善，既有資料已曝露。",20,MUTED)
    return c


def speed(data):
    c=SVG(1000,650,"換模型之後，速度差多少？","兩個 MOT17 前 140 幀的受控速度比較；左圖顯示單幀平均、P95、P99 延遲，右圖為排隊吞吐。分位數由同組 480 個原始樣本合併計算。")
    c.text(28,43,"換模型之後，速度差多少？",32,NAVY,700)
    legend(c)
    c.rect(28,119,23,13,MUTED,radius=2)
    c.text(61,133,"平均",21,MUTED)
    c.items.append(f'<circle cx="160" cy="126" r="5.5" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
    c.text(175,133,"P95",21,MUTED)
    c.items.append(f'<path d="M254,119 L261,126 L254,133 L247,126 Z" fill="{INK}"/>')
    c.text(270,133,"P99",21,MUTED)
    c.text(350,133,"（左圖，單位 ms）",20,MUTED)
    lefts=[(28,250,8,"單幀交付延遲（ms）↓"),(515,250,300,"排隊吞吐（FPS）↑")]
    rows=data["groups"]
    for panel,(start,width,limit,label) in enumerate(lefts):
        x0=start+100
        c.text(start,174,label,25,NAVY,700)
        ticks=[0,2,4,6,8] if panel==0 else [0,100,200,300]
        for value in ticks:
            x=x0+value/limit*width
            c.line(x,195,x,501)
            c.text(x,530,value,21,MUTED,anchor="middle")
        for seq_i,seq in enumerate(["MOT17-10-SDP","MOT17-02-SDP"]):
            top=216+seq_i*164
            c.text(start,top,seq.replace("-SDP",""),23,NAVY,600)
            for arm_i,arm in enumerate(ARMS):
                mode="single" if panel==0 else "queued"
                row,=[r for r in rows if r["path"]=="tracking" and r["sequence"]==seq and r["mode"]==mode and r["arm"]==arm]
                value=row["delivery_ms"]["mean"] if panel==0 else row["throughput_fps"]
                by=top+17+arm_i*38
                barwidth=value/limit*width
                c.rect(x0,by,barwidth,24,COLORS[arm],radius=3)
                if panel==0:
                    latency=row["delivery_ms"]
                    assert latency["samples"]==480
                    assert value<=latency["p95"]<=latency["p99"]<=limit
                    cy=by+12
                    p95x=x0+latency["p95"]/limit*width
                    p99x=x0+latency["p99"]/limit*width
                    c.line(x0+barwidth,cy,p99x,cy,INK,2)
                    c.line(p95x,cy,p95x,cy-6,INK,1.5)
                    c.items.append(f'<circle cx="{p95x}" cy="{cy-6}" r="5.5" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
                    c.line(p99x,cy,p99x,cy+6,INK,1.5)
                    c.items.append(f'<path d="M{p99x},{cy-1} L{p99x+7},{cy+6} L{p99x},{cy+13} L{p99x-7},{cy+6} Z" fill="{INK}"/>')
                    c.text(x0-12,by+21,f"{value:.2f}",22,MUTED if arm=="official" else COLORS[arm],600,"end")
                else:
                    c.text(x0+barwidth+8,by+21,f"{value:.1f}",23,MUTED if arm=="official" else COLORS[arm],600)
    c.text(28,572,"各前 140 幀；20 幀暖機；每組 4 sessions／480 樣本；JPEG 預載。",21,MUTED)
    c.text(28,607,"P95／P99 採合併樣本 nearest-rank；無 profiler，排除讀檔、初始化與寫檔。",20,MUTED)
    c.text(28,638,"來源：TRT 性能研究；不代表完整資料集或攝影機端到端速度。",20,MUTED)
    return c


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    quality=json.loads(QUALITY.read_text())
    performance=json.loads(SPEED.read_text())
    assert quality["result"]=="PASS" and quality["sequence_count"]==156
    assert quality["frames_per_arm"]==135503
    assert performance["tracking_graph"] and performance["samples_per_group"]==480
    sources=[]
    for path,manifest in [(QUALITY,QUALITY.parents[1]/"SHA256SUMS"),(SPEED,SPEED.parents[2]/"SHA256SUMS")]:
        expected={line.split(maxsplit=1)[1].lstrip("*").removeprefix("./"):line.split()[0] for line in manifest.read_text().splitlines()}
        assert sha(path)==expected[path.relative_to(manifest.parent).as_posix()]
        sources.append({"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path)})
    SVG_DIR.mkdir(parents=True,exist_ok=True);PDF_DIR.mkdir(parents=True,exist_ok=True)
    figures=[]
    for name,canvas in [("pipeline",pipeline()),("datasets",datasets(quality)),("group-deltas",group_deltas(quality)),("speed",speed(performance))]:
        svg=SVG_DIR/f"story-{name}.svg";pdf=PDF_DIR/f"story-{name}.pdf"
        canvas.save(svg)
        command=["rsvg-convert","--format=pdf","--output",str(pdf),str(svg)]
        subprocess.run(command,check=True)
        fonts=subprocess.check_output(["pdffonts",str(pdf)],text=True).splitlines()[2:]
        assert fonts and all(line.split()[-5]=="yes" for line in fonts),name
        extracted=subprocess.check_output(["pdftotext",str(pdf),"-"],text=True)
        assert canvas.title in extracted.replace("\n", ""),name
        figures.append({"name":name,"svg":svg.relative_to(ROOT).as_posix(),"svg_sha256":sha(svg),"pdf":pdf.relative_to(ROOT).as_posix(),"pdf_sha256":sha(pdf),"fonts_embedded":True,"vector_only":True,"data_scope":"existing metrics; no new inference/scoring" if name!="pipeline" else "logical explanatory diagram; not GPU scheduling or timing"})
    receipt={"schema":"saccade-report-story-figures/1","revision_date":"2026-11-12","generator":"render_story_figures.py","generator_sha256":sha(Path(__file__)),"sources":sources,"figures":figures,"new_inference_or_scoring":False,"conversion_tool":subprocess.check_output(["rsvg-convert","--version"],text=True).strip()}
    (HERE/"story-figures.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print("Rendered four explanatory SVG/PDF figures; source hashes and embedded fonts verified.")


if __name__=="__main__":main()
