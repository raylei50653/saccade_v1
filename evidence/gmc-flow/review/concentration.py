# For the 12 review tracks: median vs densest point (KDE mode) at several bandwidths,
# plus robust (MAD) and non-robust (std) spread of the band cells.
import sys, numpy as np
sys.argv = ['x', '/dev/null'] + sys.argv[1:]
src = open('gmc-flow/evidence/gmc-flow/review/band_distribution.py').read()
src = src[:src.index('fig, axes')]          # reuse loading/selection, skip plotting
exec(src)
for title, cells, g1, ctrl, ms in figs:
    med = np.median(cells, 0)
    out = []
    for bw in (0.5, 1.0, 2.0):
        d = np.linalg.norm(cells[:, None] - cells[None], axis=2)
        dens = np.exp(-0.5 * (d / bw) ** 2).sum(1)
        out.append(cells[np.argmax(dens)])
    mad = np.median(np.linalg.norm(cells - med, axis=1)); std = np.sqrt(((cells - cells.mean(0)) ** 2).sum(1).mean())
    f = lambda v: f'({v[0]:+6.1f},{v[1]:+5.1f})'
    print(f'{title:34} G1{f(g1)} median{f(med)} mode@0.5{f(out[0])} @1{f(out[1])} @2{f(out[2])}  MAD={mad:4.1f} std={std:4.1f}')
