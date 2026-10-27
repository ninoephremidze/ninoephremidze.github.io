"""Amplification-factor tables for the GW page, computed with FIONA.

For each lens profile (point mass, cored isothermal sphere, NFW, ULA soliton) and a few
values of its shape parameter, tabulate F(w, y) with FIONA on

  w: log-spaced on [1e-3, 1), then uniform with step DW up to W_MAX (plus node pairs at the
     Gauss-Legendre schedule's tier boundaries, as in fiona_w_table.make_w_grid)
  y: 0.2, 0.25, ..., 3.0 (the page's impact-parameter slider snaps to these nodes, so F is
     never interpolated in y, which would fail near caustics)

and store the residual R = F - s(w) F_GO, with s(w) = go_weight(w, 1) a smooth turn-on of
the geometric-optics sum over all images. Both F and F_GO are referred to the first-arriving
(minimum) image, F -> F exp(-i w phi_min), in FIONA's e^{+iwT} Fourier convention; the page
conjugates F for its waveform (LAL convention). The page reconstructs F = R + s(w) F_GO
(linear interpolation of R in w), and uses F_GO alone above W_MAX.

This is the scheme of fiona_pe_F.py / fiona_w_table.py in the FIONA inference code, and it
uses that code unchanged: the v4 Gauss-Legendre schedule, fiona_w_table.build_F_w_table and
fiona_w_table.image_set. NFW images (not covered there) use the analytic deflection below.
Lenses are FIONA's with psi0 = 1 (fisherWithFIONA._make_lens_3p): y and the shape parameters
are in units of the Einstein radius of a point mass M_Lz.

Usage (needs the FIONA environment, e.g. glow311; use a compute node: see submit script):
  python build_fiona_tables.py [--fiona-root DIR] [--workers N] [--skip-checks]

Writes ../assets/fiona/manifest.json and one <id>.bin (float32 [n_y][n_w][re, im] of R) per
lens, and prints three accuracy checks: table vs exact point-mass F (mpmath), table
interpolation vs direct FIONA at off-grid w, and direct FIONA vs FIONA with doubled GL nodes.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'assets', 'fiona')
DEFAULT_ROOT = '/n/netscratch/dvorkin_lab/Lab/nephremidze/2-LISA/0-dark_matter_inference'

W_MIN, W_SWITCH, W_MAX, DW, N_PER_DECADE = 1.0e-3, 1.0, 100.5, 0.1, 50
Y_GRID = np.round(np.arange(0.2, 3.0 + 1e-9, 0.05), 4)
GO_W_REF = 1.0

# (id, FIONA lens name, label, shape-parameter name, shape value)
LENSES = [
    ('point', 'PointLens', 'Point mass', None, None),
    ('cis_0.05', 'CIS', 'Cored isothermal sphere', 'x_c', 0.05),
    ('cis_0.2', 'CIS', 'Cored isothermal sphere', 'x_c', 0.2),
    ('cis_0.5', 'CIS', 'Cored isothermal sphere', 'x_c', 0.5),
    ('nfw_0.01', 'NFW', 'NFW halo', 'x_s', 0.01),
    ('nfw_0.1', 'NFW', 'NFW halo', 'x_s', 0.1),
    ('nfw_1', 'NFW', 'NFW halo', 'x_s', 1.0),
    ('ula_0.1', 'ULA', 'ULA soliton', 'r_s', 0.1),
    ('ula_0.3', 'ULA', 'ULA soliton', 'r_s', 0.3),
    ('ula_1', 'ULA', 'ULA soliton', 'r_s', 1.0),
]


# ------------------------------------------------------------------ NFW images (analytic)
def _nfw_F(u):
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    g, l = u > 1, u < 1
    s = np.sqrt(u[g] ** 2 - 1); out[g] = np.arctan(s) / s
    s = np.sqrt(1 - u[l] ** 2); out[l] = np.arctanh(s) / s
    return out


def nfw_alpha(r, xs):
    """Deflection of FIONA's NFW (psi0 = 1): d psi / dr = h(u) / r, h = ln(u/2) + F(u), u = r / xs.

    For u < 1 the two terms of h cancel to O(u^2); with s = sqrt(1 - u^2),
    h = log1p(-u^2 / (2 (1 + s))) + ln((1 + s) / u) u^2 / ((1 + s) s), which does not.
    (The direct form gives hundreds of spurious roots of the lens equation near r = 0.)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    u = r / xs
    h = np.log(u / 2) + _nfw_F(u)
    lo = u < 0.5
    ul = u[lo]
    s = np.sqrt(1 - ul * ul)
    h[lo] = np.log1p(-ul * ul / (2 * (1 + s))) + np.log((1 + s) / ul) * ul * ul / ((1 + s) * s)
    return h / r


def nfw_dalpha(r, xs):
    """d alpha / dr = 2 kappa - alpha / r, kappa = (1 - F(u)) / (2 xs^2 (u^2 - 1))."""
    r = np.asarray(r, dtype=float)
    u = r / xs
    d = u * u - 1
    k = np.full_like(u, 1 / 3)
    m = np.abs(d) > 1e-6
    k[m] = (1 - _nfw_F(u[m])) / d[m]
    return 2 * k / (2 * xs * xs) - nfw_alpha(r, xs) / r


def nfw_image_set(xs, y, psi, x_max=40.0, n_half=4000):
    """Same search as fiona_w_table.axisym_image_set, with the NFW deflection."""
    from scipy.optimize import brentq
    lens_eq = lambda x: x - np.sign(x) * nfw_alpha(np.abs(x), xs).reshape(np.shape(x)) - y
    g = np.geomspace(1e-9, x_max, n_half)
    xg = np.concatenate([-g[::-1], g])
    f = lens_eq(xg)
    phi, mu, nm = [], [], []
    for i in np.nonzero(f[:-1] * f[1:] < 0)[0]:
        if xg[i] < 0 < xg[i + 1]:
            continue
        x = brentq(lens_eq, xg[i], xg[i + 1], xtol=1e-15, rtol=1e-15)
        r = abs(x)
        f1 = 1 - float(nfw_alpha(np.array([r]), xs)[0]) / r
        f2 = 1 - float(nfw_dalpha(np.array([r]), xs)[0])
        phi.append(0.5 * (x - y) ** 2 - float(psi(np.array([r]))[0]))
        mu.append(1 / (f1 * f2))
        nm.append(0.0 if (f1 > 0 and f2 > 0) else (1.0 if (f1 < 0 and f2 < 0) else 0.5))
    return np.array(phi), np.array(mu), np.array(nm)


# ------------------------------------------------------------------ helpers
def shape_of(lens):
    _, model, _, _, val = lens
    return FWT.POINT_LENS_SOFTENING if model == 'PointLens' else val


def images(lens, y):
    _, model, _, _, _ = lens
    if model == 'NFW':
        return nfw_image_set(shape_of(lens), y, FWT_make_lens(lens).psi_r)
    return FWT.image_set(model, shape_of(lens), y)


def FWT_make_lens(lens):
    from fisherWithFIONA.lensing import _make_lens_3p
    return _make_lens_3p(lens[1], shape_of(lens))


def F_go_min(w, im):
    """GO sum referred to the minimum image (fiona_w_table.F_go with phi -> phi - phi_min)."""
    phi, mu, nm = im
    pmin = phi_min(im)
    return FWT.F_go(w, phi - pmin, mu, nm)


def phi_min(im):
    phi, mu, nm = im
    return float(phi[nm == 0].min()) if np.any(nm == 0) else float(phi.min())


def direct(lens, ys, ws, gl_dir, workers):
    """Direct FIONA F(w, y), shape (n_y, n_w), in FIONA's absolute time reference."""
    return FWT.build_F_w_table([shape_of(lens)], ys, ws, gl_dir, n_workers=workers,
                               lens_model=lens[1], verbose=False)[0]


def interp_R(w_grid, R_row, w):
    return np.interp(w, w_grid, R_row.real) + 1j * np.interp(w, w_grid, R_row.imag)


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fiona-root', default=os.environ.get('FIONA_ROOT', DEFAULT_ROOT))
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_NTASKS', os.cpu_count())))
    ap.add_argument('--skip-checks', action='store_true')
    ap.add_argument('--only', nargs='*', help='build only these lens ids')
    args = ap.parse_args()

    root = os.path.abspath(args.fiona_root)
    sys.path.insert(0, root)
    gl_dir = os.path.join(root, 'fiona_gl_cache')
    os.environ['FIONA_GL2D_DIR'] = gl_dir
    import finufft  # noqa: F401  (must precede the fiona imports)
    import fiona_w_table as FWT
    import fiona.axisym as AX
    GL = importlib.import_module('fiona_adaptive_gl_v4')
    GL.seed_gl_cache(gl_dir, n_values=tuple(GL.REQUIRED_N_GL))
    GL.apply()

    w_grid = FWT.make_w_grid(w_min=W_MIN, w_switch=W_SWITCH, w_max=W_MAX, n_per_decade=N_PER_DECADE, dw=DW)
    s = FWT.go_weight(w_grid, GO_W_REF)
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, 'manifest.json')
    old = {l['id']: l for l in json.load(open(man_path))['lenses']} if os.path.exists(man_path) else {}
    lenses_out = []
    rng = np.random.default_rng(1)
    print(f'w grid: {len(w_grid)} nodes on [{w_grid[0]:.3g}, {w_grid[-1]:.4g}]; y grid: {len(Y_GRID)} nodes; '
          f'{args.workers} workers', flush=True)

    for lens in LENSES:
        lid, model, label, pname, pval = lens
        if args.only and lid not in args.only:
            if lid in old:
                lenses_out.append(old[lid])
            continue
        t0 = time.time()
        ims = [images(lens, y) for y in Y_GRID]
        pm = np.array([phi_min(im) for im in ims])
        F = direct(lens, Y_GRID, w_grid, gl_dir, args.workers) * np.exp(-1j * np.outer(pm, w_grid))
        R = np.array([F[j] - s * F_go_min(w_grid, ims[j]) for j in range(len(Y_GRID))])
        if not np.all(np.isfinite(R)):
            raise RuntimeError(f'{lid}: non-finite table entries')
        R.astype(np.complex64).view(np.float32).tofile(os.path.join(OUT, f'{lid}.bin'))

        lens_obj = FWT_make_lens(lens)
        r_psi = np.geomspace(1e-3, 100, 240)
        entry = dict(
            id=lid, model=model, label=label, param=pname, value=pval, file=f'{lid}.bin',
            # images per y node: [phi - phi_min, mu, Morse index n]
            images=[[[round(float(p - pmin), 10), round(float(m), 10), float(n)]
                     for p, m, n in zip(*im)] for im, pmin in zip(ims, pm)],
            # lens potential psi(r) for the 2D wave picture
            psi_r=[round(float(v), 6) for v in r_psi], psi=[round(float(v), 8) for v in lens_obj.psi_r(r_psi)],
        )
        n_img = [len(im[0]) for im in ims]
        print(f'{lid}: images per y {min(n_img)}-{max(n_img)}, max |R| {np.abs(R).max():.3g}, '
              f'{time.time() - t0:.0f}s', flush=True)

        if not args.skip_checks:
            # 1) interpolation: direct FIONA at off-grid w vs the table, at every y node
            wt = np.sort(np.concatenate([rng.uniform(W_MIN, 1, 20), rng.uniform(1, W_MAX - 0.6, 60)]))
            Fd = direct(lens, Y_GRID, wt, gl_dir, args.workers) * np.exp(-1j * np.outer(pm, wt))
            st = FWT.go_weight(wt, GO_W_REF)
            Ft = np.array([interp_R(w_grid, R[j], wt) + st * F_go_min(wt, ims[j]) for j in range(len(Y_GRID))])
            e_int = np.abs(Ft - Fd)
            # 2) GL convergence: doubled node counts at a subset
            sub_w, sub_y = wt[::8], Y_GRID[::7]
            v4 = AX._adaptive_n_gl
            AX._adaptive_n_gl = lambda w: min(2 * v4(w), 1024000)
            F2 = direct(lens, sub_y, sub_w, gl_dir, args.workers)
            AX._adaptive_n_gl = v4
            F1 = direct(lens, sub_y, sub_w, gl_dir, args.workers)
            e_gl = np.abs(F2 - F1)
            entry['check'] = dict(interp_median=float(np.median(e_int)), interp_p95=float(np.percentile(e_int, 95)),
                                  interp_max=float(e_int.max()), gl_doubled_max=float(e_gl.max()))
            print(f'  |F_table - F_direct| at off-grid w: median {np.median(e_int):.1e}, 95% '
                  f'{np.percentile(e_int, 95):.1e}, max {e_int.max():.1e} (y = {Y_GRID[np.unravel_index(e_int.argmax(), e_int.shape)[0]]:.2f}); '
                  f'|F(2 n_gl) - F(n_gl)| max {e_gl.max():.1e}', flush=True)
            # 3) point mass: exact closed form
            if model == 'PointLens':
                sys.path.insert(0, HERE)
                from lensviz.gw import F_point_exact
                k = rng.integers(0, len(Y_GRID), 300)
                wx = rng.uniform(0.01, W_MAX - 0.6, 300)
                ex = np.array([F_point_exact(w, Y_GRID[j]) for w, j in zip(wx, k)])
                tb = np.array([interp_R(w_grid, R[j], w) + FWT.go_weight(w, GO_W_REF) * F_go_min(np.array([w]), ims[j])[0]
                               for w, j in zip(wx, k)])
                e = np.abs(tb - ex)
                entry['check'].update(exact_median=float(np.median(e)), exact_p95=float(np.percentile(e, 95)),
                                      exact_max=float(e.max()))
                print(f'  |F_table - F_exact| (point mass, mpmath): median {np.median(e):.1e}, 95% '
                      f'{np.percentile(e, 95):.1e}, max {e.max():.1e}', flush=True)
        lenses_out.append(entry)

    manifest = dict(
        source='FIONA (N. Ephremidze), v4 Gauss-Legendre schedule; built by python/build_fiona_tables.py',
        convention='R = F - s(w) F_GO, F in FIONA e^{+iwT} convention, times from the minimum image; '
                   's(w) = 0.5 (1 + tanh((w - w_ref) / (0.2 w_ref))); GO only above w_max',
        w_ref=GO_W_REF, w_max=float(w_grid[-1]), w=[float(v) for v in w_grid], y=[float(v) for v in Y_GRID],
        layout='float32 [n_y][n_w][re, im]', lenses=lenses_out)
    json.dump(manifest, open(man_path, 'w'), separators=(',', ':'))
    print(f'wrote {man_path} ({os.path.getsize(man_path) / 1e3:.0f} kB) and {len(lenses_out)} tables', flush=True)
