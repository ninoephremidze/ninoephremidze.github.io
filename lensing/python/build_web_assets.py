"""Build the data files the interactive pages load.

  ../assets/galaxy.png         - the source galaxy cut from the JWST deep field
  ../assets/F_point_table.bin  - residual R = F - F_GO of the point-mass amplification
                                 factor on a (w, y) grid, float32, for the GW page

The GW page reconstructs F(w, y) = F_GO(w, y) + R(w, y), interpolating only the
small, slowly varying residual (the geometric-optics part is analytic), and uses
pure geometric optics above w = W_MAX.
"""
import json
import os
import numpy as np
from PIL import Image

from lensviz import gw as G
from lensviz import cluster as C
from lensviz import source as S

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, '..', 'assets')
W_MAX, DW = 100.0, 0.1
Y_MIN, Y_MAX, DY = 0.1, 3.0, 0.05


def galaxy():
    src = S.cutout(os.path.join(HERE, '..', '..', 'images', 'deep-field.jpg'), 672, 914, size=96, taper=0.6)
    img = (np.clip(src * 1.4, 0, 1) ** 0.6 * 255).astype('uint8')
    Image.fromarray(img).save(os.path.join(ASSETS, 'galaxy.png'))


def curves():
    """Critical curves and caustics of the cluster used on the interactive page."""
    cl = C.Cluster(b=18.0, q=0.5, s=5.0)
    crit, caus = C.critical_curves([cl], half=20, n=700)
    r = lambda ps: [np.round(p[::2], 3).tolist() for p in ps]
    json.dump(dict(cluster=dict(b=cl.b, q=cl.q, s=cl.s), critical=r(crit), caustic=r(caus),
                   kpc_per_arcsec=C.KPC_PER_ARCSEC, theta_e2_per_msun=C.THETA_E2_PER_MSUN),
              open(os.path.join(ASSETS, 'cluster_curves.json'), 'w'))


def table():
    ws = np.arange(0, W_MAX + DW / 2, DW)
    ys = np.round(np.arange(Y_MIN, Y_MAX + DY / 2, DY), 4)
    R = np.zeros((len(ys), len(ws), 2), dtype=np.float32)
    for j, y in enumerate(ys):
        F = np.array([G.F_point_exact(w, y) if w > 0 else 1 + 0j for w in ws])
        r = F - G.F_point_go(ws, y)
        R[j, :, 0], R[j, :, 1] = r.real, r.imag
        print(f'y = {y:.2f}', end='\r')
    R.tofile(os.path.join(ASSETS, 'F_point_table.bin'))
    meta = dict(w_max=W_MAX, dw=DW, y_min=Y_MIN, dy=DY, n_w=len(ws), n_y=len(ys),
                layout='float32 [n_y][n_w][re, im] of F - F_GO')
    json.dump(meta, open(os.path.join(ASSETS, 'F_point_table.json'), 'w'), indent=1)
    return ws, ys, R


def check(ws, ys, R, n=400, seed=0):
    """Accuracy of the bilinear-interpolated table against the exact F."""
    rng = np.random.default_rng(seed)
    err = []
    for _ in range(n):
        w, y = rng.uniform(0.01, W_MAX - 0.1), rng.uniform(Y_MIN, Y_MAX - 0.01)
        i, j = w / DW, (y - Y_MIN) / DY
        i0, j0 = int(i), int(j); fi, fj = i - i0, j - j0
        r = ((1 - fi) * (1 - fj) * R[j0, i0] + fi * (1 - fj) * R[j0, i0 + 1]
             + (1 - fi) * fj * R[j0 + 1, i0] + fi * fj * R[j0 + 1, i0 + 1])
        F = G.F_point_go(w, y) + r[0] + 1j * r[1]
        err.append(abs(F - G.F_point_exact(w, y)) / abs(G.F_point_exact(w, y)))
    err = np.array(err)
    print(f'\ntable accuracy: median {np.median(err):.1e}, 95% {np.percentile(err, 95):.1e}, max {err.max():.1e}')


if __name__ == '__main__':
    os.makedirs(ASSETS, exist_ok=True)
    galaxy()
    curves()
    if not os.path.exists(os.path.join(ASSETS, 'F_point_table.bin')) or '--table' in __import__('sys').argv:
        check(*table())
