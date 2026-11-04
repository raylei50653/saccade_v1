import struct, math, sys
scdr, scfl, frames = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
# SCDR: magic, u32 version, u32 sizeof(DetectionCandidates), u64 epoch, u64 count, then raw structs
d = open(scdr, 'rb')
magic = d.read(4); ver, size = struct.unpack('<II', d.read(8)); epoch, count = struct.unpack('<QQ', d.read(16))
base = d.tell()
def dets(k):
    d.seek(base + k*size)
    hdr = d.read(64)
    n = struct.unpack('<QQ7I5I', hdr)[3]
    out = []
    for i in range(n):
        x1,y1,x2,y2,s,c,cid,fl = struct.unpack('<5f3I', d.read(32))
        out.append((x1,y1,x2,y2,s,c,fl))
    return out
f = open(scfl, 'rb'); f.read(4); ver,W,H,cell,cols,rows,perf = struct.unpack('<I6i', f.read(28)); fc, = struct.unpack('<Q', f.read(8))
fbase = f.tell(); nf = rows*cols*2; fsize = 12 + 4*nf
for k in frames:
    f.seek(fbase + k*fsize); f.read(12); v = struct.unpack(f'<{nf}f', f.read(4*nf))
    prev = dets(k-1)   # mask source is frame k-1
    kept = [b for b in prev if b[6] & 1 and b[5] == 0]
    def covered(thr):
        boxes = [b for b in kept if b[4] >= thr]
        tot = cov = 0
        for i in range(rows*cols):
            if math.hypot(v[2*i], v[2*i+1]) <= 3: continue
            x, y = 8*(i % cols)+4, 8*(i//cols)+4
            tot += 1
            cov += any(b[0] <= x < b[2] and b[1] <= y < b[3] for b in boxes)
        return tot, cov
    tot, c45 = covered(0.45); _, c10 = covered(0.10); _, c05 = covered(0.05)
    # detections overlapping the near-camera region (left-centre, lower half)
    near = sorted([b for b in kept if b[2] > 340 and b[0] < 1000 and b[3] > 900 and (b[3]-b[1]) > 400], key=lambda b: -b[4])
    print(f'k={k}: cells>3px={tot}  masked@0.45={100*c45/tot:.0f}%  @0.10={100*c10/tot:.0f}%  @0.05={100*c05/tot:.0f}%')
    for b in near[:4]:
        print(f'    tall box score={b[4]:.3f} x={b[0]:.0f}-{b[2]:.0f} y={b[1]:.0f}-{b[3]:.0f}')
