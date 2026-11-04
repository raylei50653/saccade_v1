import csv, sys, math, collections
# For each arm/sequence: tail of |control| and correlation of control with the track's own
# box-centre motion (committed state at k-1 minus committed state at k-2, same uid).
for arm in sys.argv[2:]:
    for seq in sys.argv[1].split(','):
        rows = list(csv.DictReader(open(f'{arm}/MOT17-{seq}-SDP.csv')))
        prev = {}
        n = big = 0; sxy = sxx = syy = 0.0; mags = []
        for r in rows:
            k, uid = int(r['frame']), r['uid']
            cx = (float(r['x1'])+float(r['x2']))/2; cy = (float(r['y1'])+float(r['y2']))/2
            h = float(r['y2'])-float(r['y1'])
            dx, dy = float(r['dx']), float(r['dy'])
            m = math.hypot(dx, dy); mags.append(m)
            if m > 2: big += 1
            p = prev.get(uid)
            if p and p[0] == k-1:
                vx, vy = cx-p[1], cy-p[2]
                sxy += dx*vx + dy*vy; sxx += dx*dx+dy*dy; syy += vx*vx+vy*vy; n += 1
            prev[uid] = (k, cx, cy)
        mags.sort()
        corr = sxy/math.sqrt(sxx*syy) if sxx and syy else float('nan')
        print(f'{arm:3} {seq}: controls={len(mags)} p99={mags[int(0.99*len(mags))]:.2f}px >2px={100*big/len(mags):.2f}%  cos(control, own motion)={corr:+.3f}')
