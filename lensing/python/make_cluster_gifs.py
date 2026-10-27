"""Animated GIFs: strong lensing of a JWST galaxy by a galaxy cluster.

Scenes (each written to its own GIF so they can be dropped into slides):
  1. cluster_raytracing.gif  - light from a background galaxy is bent by the cluster
                               into three images; ray tracing reveals them.
  2. cluster_subhalo_move.gif - a dark-matter subhalo near one arc; moving it moves
                               the perturbation it imprints on the image.
  3. cluster_subhalo_profile.gif - same subhalo, changing its density profile.
  4. cluster_wave_dm.gif     - wave-like dark matter granules instead of one subhalo:
                               a statistical imprint, summarised by the residual
                               power spectrum.

Usage:  python make_cluster_gifs.py [--scenes 1 2 3 4] [--source path/to/image.jpg X Y]
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage

from lensviz import cluster as C
from lensviz import source as S
from lensviz.style import (DIVERGING, fmt_mass, GifWriter, ease, PAPER, INK, MUTED, FAINT, LINE, ACCENT, CYAN, GOLD, MONO)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'gifs')
DPI = 150
FIGSIZE = (12.8, 7.2)          # 1920 x 1080 px (the GIF is downsampled to 960 x 540)

# ------------------------------------------------------------------ the lens system
CLUSTER = C.Cluster(b=18.0, q=0.5, s=5.0)
SRC_POS = (2.8, 0.7)           # source position [arcsec]
SRC_SIZE = 0.75                # source width [arcsec]  (~6 kpc at z = 1)
HALF_FULL = 16.0               # half width of the full image-plane view [arcsec]
HALF_ZOOM = 1.6                # half width of the zoom on one arc [arcsec]
N_FULL, N_ZOOM = 640, 420
R_T = 0.3                      # tidal truncation radius of the subhalo [arcsec] (~1.6 kpc)


def load_source(args):
    if args.source:
        path, x, y = args.source
        return S.cutout(path, int(x), int(y), size=96, taper=0.6)
    default = os.path.join(HERE, '..', '..', 'images', 'deep-field.jpg')
    if os.path.exists(default):
        return S.cutout(default, 672, 914, size=96, taper=0.6)   # a face-on spiral in the JWST deep field
    return S.synthetic_spiral()


def show(img, gain=1.4):
    return np.clip(img * gain, 0, 1) ** 0.6


def find_images(src):
    x, y = C.grid(0, 0, HALF_FULL, N_FULL)
    bx, by = C.ray_trace(x, y, [CLUSTER])
    im = C.sample(src, bx, by, SRC_POS, SRC_SIZE).sum(-1)
    lab, n = ndimage.label(ndimage.binary_closing(im > 0.05 * im.max(), iterations=3))
    sizes = ndimage.sum(im, lab, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:3] + 1
    pix = 2 * HALF_FULL / (N_FULL - 1)
    cents = [ndimage.center_of_mass(im, lab, k) for k in keep]
    pos = [(-HALF_FULL + cx * pix, -HALF_FULL + cy * pix) for cy, cx in cents]
    # order by magnification, most magnified first (that is the arc we zoom on)
    # order by how stretched each image is (tangential/radial eigenvalue ratio); the most arc-like is first
    def stretch(px, py, h=1e-3):
        x0, y0 = np.array([px, px + h, px]), np.array([py, py, py + h])
        bx, by = C.ray_trace(x0, y0, [CLUSTER])
        A = np.array([[bx[1] - bx[0], bx[2] - bx[0]], [by[1] - by[0], by[2] - by[0]]]) / h
        ev = np.abs(np.linalg.eigvals(A))
        return ev.max() / ev.min()
    st = [stretch(px, py) for px, py in pos]
    return [p for _, p in sorted(zip(st, pos), key=lambda t: -t[0])]


def axis_off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)


def header(fig, title, sub):
    fig.text(0.035, 0.935, title, fontsize=22, color=INK, ha='left', va='center')
    fig.text(0.035, 0.875, sub, fontsize=13, color=MUTED, ha='left', va='center')


def footer(fig, text):
    fig.text(0.035, 0.04, text, fontsize=9.5, color=FAINT, ha='left', va='bottom', fontdict=MONO)


# =================================================================== scene 1
def scene_raytracing(src, images):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    gw = GifWriter(f'{OUT}/cluster_raytracing.gif', fps=12)
    crit, caus = C.critical_curves([CLUSTER], half=20, n=500)

    # --- part A: 3D schematic, rays from the galaxy bend at the cluster toward us
    zO, zL, zS = 0.0, 1.0, 2.3
    beta = np.array(SRC_POS)
    n_frames = 72
    # cluster texture on the lens plane: convergence of the NIE
    t = np.linspace(-18, 18, 60)
    X, Y = np.meshgrid(t, t)
    q, s, b = CLUSTER.q, CLUSTER.s, CLUSTER.b
    kap = b / (2 * np.sqrt(q * q * (s * s + X * X) + Y * Y))
    kcol = plt.cm.bone(np.clip(kap / kap.max(), 0, 1) ** 0.8)
    kcol[..., 3] = 0.55
    # galaxy texture on the source plane (exaggerated x5 so it is visible)
    g = src[::3, ::3]
    gs = SRC_SIZE * 5
    gu = np.linspace(-gs / 2, gs / 2, g.shape[0])
    GX, GY = np.meshgrid(gu + beta[0], -gu + beta[1])
    gcol = np.concatenate([show(g), np.clip(g.sum(-1, keepdims=True) * 3, 0, 1)], -1)

    for f in range(n_frames):
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        header(fig, 'How a galaxy cluster lenses a background galaxy',
               'Light from one galaxy reaches us along three different paths')
        ax = fig.add_axes([-0.12, -0.02, 1.24, 0.9], projection='3d')
        ax.set_facecolor(PAPER)
        ax.set_axis_off()
        az = -70 + 25 * f / n_frames
        ax.view_init(elev=12, azim=az)
        ax.set_box_aspect((2.6, 1.3, 1.0), zoom=1.1)
        ax.set_xlim(zO - 0.1, zS + 0.1); ax.set_ylim(-19, 19); ax.set_zlim(-15, 15)
        # planes
        ax.plot_surface(np.full_like(X, zL), X, Y, facecolors=kcol, shade=False, rstride=1, cstride=1, linewidth=0)
        ax.plot_surface(np.full_like(GX, zS), GX, GY, facecolors=gcol, shade=False, rstride=1, cstride=1, linewidth=0)
        for z, h in ((zL, 18), (zS, 18)):
            ax.plot([z] * 5, [-h, h, h, -h, -h], [-h, -h, h, h, -h], color=LINE, lw=0.8)
        ax.scatter([zO], [0], [0], color=INK, s=40)
        ax.text(zO, 0, -4, 'observer', color=MUTED, fontsize=11, ha='center')
        ax.text(zL, 0, 20, 'galaxy cluster', color=MUTED, fontsize=11, ha='center')
        ax.text(zS, 0, 20, 'background galaxy', color=MUTED, fontsize=11, ha='center')
        # rays, growing from the source toward the observer
        prog = ease(f / (n_frames * 0.7))
        for k, (tx, ty) in enumerate(images):
            pk = np.clip(prog * 3 - k * 0.9, 0, 1)
            if pk <= 0:
                continue
            pts = np.array([[zS, beta[0], beta[1]], [zL, tx, ty], [zO, 0, 0]])
            # parametrise the polyline by length
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            Lr = pk * seg.sum()
            if Lr <= seg[0]:
                end = pts[0] + (pts[1] - pts[0]) * Lr / seg[0]
                path = np.array([pts[0], end])
            else:
                end = pts[1] + (pts[2] - pts[1]) * (Lr - seg[0]) / seg[1]
                path = np.array([pts[0], pts[1], end])
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=CYAN, lw=1.6, alpha=0.95)
            # dashed: where the observer *thinks* the light came from
            if pk >= 1:
                ext = np.array([zO, 0, 0]) + (np.array([zL, tx, ty]) - np.array([zO, 0, 0])) * zS / zL
                ax.plot([zL, ext[0]], [tx, ext[1]], [ty, ext[2]], color=ACCENT, lw=1, ls=(0, (3, 3)), alpha=0.8)
        footer(fig, 'Schematic: transverse distances exaggerated; cluster = cored isothermal ellipsoid, z_L = 0.4, z_S = 1.0')
        gw.add(fig)
        plt.close(fig)
    gw.hold(1.0)

    # --- part B: backward ray tracing fills the image plane, row by row
    x, y = C.grid(0, 0, HALF_FULL, N_FULL)
    bx, by = C.ray_trace(x, y, [CLUSTER])
    lensed = show(C.sample(src, bx, by, SRC_POS, SRC_SIZE))
    xs, ys = C.grid(SRC_POS[0] - 3, SRC_POS[1], 4.5, 300)
    srcimg = show(C.sample(src, xs, ys, SRC_POS, SRC_SIZE))
    n_frames = 60
    for f in range(n_frames + 1):
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        header(fig, 'Ray tracing: from the image plane back to the source',
               'Each pixel we observe maps to a point behind the cluster: β = θ − α(θ)')
        a1 = fig.add_axes([0.05, 0.11, 0.38, 0.68])
        a2 = fig.add_axes([0.52, 0.11, 0.43, 0.68])
        a1.imshow(srcimg, origin='lower', extent=[SRC_POS[0] - 7.5, SRC_POS[0] + 1.5, SRC_POS[1] - 4.5, SRC_POS[1] + 4.5])
        for p in caus:
            a1.plot(p[:, 0], p[:, 1], color=GOLD, lw=1)
        a1.set_xlim(SRC_POS[0] - 7.5, SRC_POS[0] + 1.5); a1.set_ylim(SRC_POS[1] - 4.5, SRC_POS[1] + 4.5)
        a1.set_title('Source plane (caustics in gold)', loc='left', color=MUTED, fontsize=12)
        axis_off(a1)
        frac = ease(f / n_frames)
        row = int(frac * N_FULL)
        shown = np.zeros_like(lensed)
        shown[N_FULL - row:] = lensed[N_FULL - row:]
        a2.imshow(shown, origin='lower', extent=[-HALF_FULL, HALF_FULL, -HALF_FULL, HALF_FULL])
        if 0 < row < N_FULL:
            yline = HALF_FULL - 2 * HALF_FULL * row / N_FULL
            a2.axhline(yline, color=ACCENT, lw=1, alpha=0.8)
        for p in crit:
            a2.plot(p[:, 0], p[:, 1], color=CYAN, lw=0.8, alpha=0.7)
        a2.set_xlim(-HALF_FULL, HALF_FULL); a2.set_ylim(-HALF_FULL, HALF_FULL)
        a2.set_title('Image plane (critical curves in blue)', loc='left', color=MUTED, fontsize=12)
        axis_off(a2)
        for k, (tx, ty) in enumerate(images):
            if HALF_FULL - 2 * HALF_FULL * row / N_FULL < ty:
                con = ConnectionPatch(xyA=(tx, ty), coordsA=a2.transData, xyB=SRC_POS, coordsB=a1.transData,
                                      color=CYAN, lw=0.9, alpha=0.6, ls=(0, (2, 3)))
                fig.add_artist(con)
                a2.add_patch(Circle((tx, ty), 1.3, fill=False, ec=ACCENT, lw=1.2))
                a2.text(tx + 1.6, ty + 0.4, f'image {k + 1}', color=ACCENT, fontsize=11)
        footer(fig, f'{len(images)} images of one galaxy · image plane ±{HALF_FULL:.0f}″ · source width {SRC_SIZE}″ (≈{SRC_SIZE * 8.0:.0f} kpc at z = 1)')
        gw.add(fig)
        plt.close(fig)
    gw.hold(2.5)
    gw.save()


# =================================================================== scenes 2-3
def zoom_center(images):
    # zoom on the most magnified image: the one closest to the critical curve, i.e. image 1
    return images[0]


def absorb_large_scale(ax, ay, x, y, weight):
    """Remove the part of a perturbing deflection that a smooth lens + source model would absorb.

    A real fit re-adjusts the source position and the local convergence/shear of the
    macro model, so only the part of the perturbation that is not constant or linear
    across the arc shows up in the residuals. We mimic this by subtracting a
    brightness-weighted linear fit to the perturbing deflection field."""
    w = weight.ravel() / weight.sum()
    A = np.c_[np.ones(x.size), x.ravel() - x.mean(), y.ravel() - y.mean()] * np.sqrt(w)[:, None]
    out = []
    for a in (ax, ay):
        coef = np.linalg.lstsq(A, a.ravel() * np.sqrt(w), rcond=None)[0]
        out.append(a - (coef[0] + coef[1] * (x - x.mean()) + coef[2] * (y - y.mean())))
    return out


def render_zoom(src, center, sub=None, extra=None, n=N_ZOOM):
    x, y = C.grid(center[0], center[1], HALF_ZOOM, n)
    bx0, by0 = C.ray_trace(x, y, [CLUSTER])
    if sub is None and extra is None:
        return C.sample(src, bx0, by0, SRC_POS, SRC_SIZE)
    weight = C.sample(src, bx0, by0, SRC_POS, SRC_SIZE).sum(-1) + 1e-6
    ax = np.zeros_like(x); ay = np.zeros_like(y)
    if sub is not None:
        dx, dy = sub.deflection(x, y); ax += dx; ay += dy
    if extra is not None:
        ax += extra[0]; ay += extra[1]
    ax, ay = absorb_large_scale(ax, ay, x, y, weight)
    return C.sample(src, bx0 - ax, by0 - ay, SRC_POS, SRC_SIZE)


def subhalo_frame(fig, src, center, smooth, sub, left_panel, title, subtitle, foot):
    header(fig, title, subtitle)
    img = render_zoom(src, center, sub)
    dev = (img - smooth).sum(-1) / smooth.sum(-1).max()
    a0 = fig.add_axes(getattr(left_panel, 'rect', [0.035, 0.12, 0.29, 0.66]))
    a1 = fig.add_axes([0.355, 0.12, 0.29, 0.66])
    a2 = fig.add_axes([0.675, 0.12, 0.29, 0.66])
    left_panel(a0)
    ext = [center[0] - HALF_ZOOM, center[0] + HALF_ZOOM, center[1] - HALF_ZOOM, center[1] + HALF_ZOOM]
    a1.imshow(show(img), origin='lower', extent=ext)
    rmark = max(0.12, np.sqrt(sub.mass * C.THETA_E2_PER_MSUN) * 1.5)
    a1.add_patch(Circle((sub.x0, sub.y0), rmark, fill=False, ec=ACCENT, lw=1.3, ls=(0, (2, 2))))
    a1.set_title('Lensed arc, zoomed in', loc='left', color=MUTED, fontsize=12)
    axis_off(a1); a1.set_xlim(ext[:2]); a1.set_ylim(ext[2:])
    a2.imshow(dev, origin='lower', extent=ext, cmap=DIVERGING, norm=TwoSlopeNorm(0, -0.25, 0.25))
    a2.add_patch(Circle((sub.x0, sub.y0), rmark, fill=False, ec=INK, lw=1, ls=(0, (2, 2)), alpha=0.7))
    a2.set_title('Change in the image (with − without subhalo)', loc='left', color=MUTED, fontsize=12)
    axis_off(a2); a2.set_xlim(ext[:2]); a2.set_ylim(ext[2:])
    footer(fig, foot)


def full_panel(src, images, center):
    x, y = C.grid(0, 0, HALF_FULL, 400)
    bx, by = C.ray_trace(x, y, [CLUSTER])
    lensed = show(C.sample(src, bx, by, SRC_POS, SRC_SIZE))
    crit, _ = C.critical_curves([CLUSTER], half=20, n=400)

    def draw(ax):
        ax.imshow(lensed, origin='lower', extent=[-HALF_FULL, HALF_FULL, -HALF_FULL, HALF_FULL])
        for p in crit:
            ax.plot(p[:, 0], p[:, 1], color=CYAN, lw=0.7, alpha=0.6)
        ax.add_patch(Rectangle((center[0] - HALF_ZOOM, center[1] - HALF_ZOOM), 2 * HALF_ZOOM, 2 * HALF_ZOOM,
                               fill=False, ec=ACCENT, lw=1.2))
        ax.set_title('Whole cluster field', loc='left', color=MUTED, fontsize=12)
        ax.set_xlim(-HALF_FULL, HALF_FULL); ax.set_ylim(-HALF_FULL, HALF_FULL)
        axis_off(ax)
    return draw


def scene_subhalo_move(src, images):
    gw = GifWriter(f'{OUT}/cluster_subhalo_move.gif', fps=12)
    c = zoom_center(images)
    smooth = render_zoom(src, c)
    left = full_panel(src, images, c)
    M = 5e9
    n = 96
    for f in range(n):
        ph = 2 * np.pi * f / n
        sx = c[0] + 0.55 * np.cos(ph) + 0.15 * np.sin(2 * ph)
        sy = c[1] + 0.95 * np.sin(ph)
        sub = C.Subhalo(sx, sy, mass=M, kind='nfw', shape=0.08, r_t=R_T)
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        subhalo_frame(fig, src, c, smooth, sub, left,
                      'A dark matter subhalo perturbs the arc',
                      'Moving the subhalo moves its imprint: it is strongest where the arc is bright',
                      f'Tidally truncated NFW subhalo, M = {fmt_mass(M)}, r_s = 0.08″ ({0.08 * C.KPC_PER_ARCSEC:.2f} kpc) · zoom ±{HALF_ZOOM}″ · large-scale part absorbed by the smooth lens model, as in a real fit')
        gw.add(fig)
        plt.close(fig)
    gw.save()


def scene_subhalo_profile(src, images):
    gw = GifWriter(f'{OUT}/cluster_subhalo_profile.gif', fps=12)
    c = zoom_center(images)
    smooth = render_zoom(src, c)
    sx, sy = c[0] + 0.15, c[1] + 0.35
    M = 5e9
    r = np.logspace(-2.3, 0.2, 200)
    # sequence of (label, kind, shape) keyframes
    seq = ([('NFW, diffuse (low concentration)', 'nfw', s_) for s_ in np.geomspace(0.5, 0.03, 40)]
           + [('Cored (SIDM-like)', 'cored', s_) for s_ in np.geomspace(0.03, 0.4, 40)]
           + [('Point mass (PBH-like)', 'point', 0.0)] * 24)

    def profile_panel(kind, shape, label):
        def draw(ax):
            sub = C.Subhalo(0, 0, mass=M, kind=kind, shape=shape, r_t=R_T)
            m = sub._enclosed(r) if kind != 'point' else np.ones_like(r)
            ax.plot(r * C.KPC_PER_ARCSEC, m, color=ACCENT, lw=2.2)
            ax.set_xscale('log'); ax.set_ylim(0, 1.05); ax.set_xlim(r[0] * C.KPC_PER_ARCSEC, r[-1] * C.KPC_PER_ARCSEC)
            ax.set_xlabel('radius [kpc]'); ax.set_ylabel('enclosed mass fraction')
            ax.set_title('Subhalo mass profile', loc='left', color=MUTED, fontsize=12)
            ax.text(0.04, 0.92, label, transform=ax.transAxes, color=INK, fontsize=12)
            for sp in ax.spines.values():
                sp.set_color(LINE)
        draw.rect = [0.075, 0.2, 0.25, 0.56]
        return draw

    for label, kind, shape in seq:
        sub = C.Subhalo(sx, sy, mass=M, kind=kind, shape=shape, r_t=R_T)
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        subhalo_frame(fig, src, c, smooth, sub, profile_panel(kind, shape, label),
                      'The imprint depends on how the mass is distributed',
                      'Same subhalo mass and position; only its density profile changes',
                      f'M = {fmt_mass(M)} in all frames · more centrally concentrated mass → sharper, stronger perturbation')
        gw.add(fig)
        plt.close(fig)
    gw.hold(1.5)
    gw.save()


# =================================================================== scene 4
def radial_power(img, box):
    n = img.shape[0]
    fk = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())) ** 2) / n ** 4 * box ** 2
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=box / n))
    kx, ky = np.meshgrid(k, k)
    kr = np.hypot(kx, ky)
    bins = np.geomspace(2 * np.pi / box, k.max(), 22)
    idx = np.digitize(kr.ravel(), bins)
    p = np.array([fk.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    kc = np.sqrt(bins[1:] * bins[:-1])
    return kc / C.KPC_PER_ARCSEC, p      # k in 1/kpc


def cdm_population(center, seed=3, n_sub=40):
    """Illustrative CDM subhalos around the arc: dN/dM ∝ M^-1.9 between 1e7 and 1e9 M_sun."""
    rng = np.random.default_rng(seed)
    a = -0.9
    u = rng.uniform(size=n_sub)
    lo, hi = 1e7 ** a, 1e9 ** a
    masses = (lo + u * (hi - lo)) ** (1 / a)
    xs = center[0] + rng.uniform(-HALF_ZOOM, HALF_ZOOM, n_sub)
    ys = center[1] + rng.uniform(-HALF_ZOOM, HALF_ZOOM, n_sub)
    return [C.Subhalo(x, y, mass=m, kind='nfw', shape=0.02 * (m / 1e8) ** 0.4, r_t=0.3) for x, y, m in zip(xs, ys, masses)]


def scene_wave_dm(src, images):
    gw = GifWriter(f'{OUT}/cluster_wave_dm.gif', fps=12)
    c = zoom_center(images)
    n = 256
    box = 2 * HALF_ZOOM
    x, y = C.grid(c[0], c[1], HALF_ZOOM, n)
    smooth = C.sample(src, *C.ray_trace(x, y, [CLUSTER]), SRC_POS, SRC_SIZE)
    # CDM reference
    cdm_img = C.sample(src, *C.ray_trace(x, y, [CLUSTER] + cdm_population(c)), SRC_POS, SRC_SIZE)
    k_cdm, p_cdm = radial_power((cdm_img - smooth).sum(-1) / smooth.sum(-1).max(), box)
    ext = [c[0] - HALF_ZOOM, c[0] + HALF_ZOOM, c[1] - HALF_ZOOM, c[1] + HALF_ZOOM]

    # keyframes: first sweep the de Broglie wavelength, then the amplitude
    lams = list(np.geomspace(0.04, 0.4, 60)) + [0.15] * 50
    amps = [0.06] * 60 + list(np.geomspace(0.015, 0.15, 50))
    history = []
    for f, (lam, amp) in enumerate(zip(lams, amps)):
        kap, ax_, ay_ = C.granule_field(n, box, lam, amp, seed=7)
        img = C.sample(src, *C.ray_trace(x, y, [CLUSTER], extra=(ax_, ay_)), SRC_POS, SRC_SIZE)
        dev = (img - smooth).sum(-1) / smooth.sum(-1).max()
        k, p = radial_power(dev, box)
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        header(fig, 'Wave-like dark matter: a statistical imprint',
               'Interference "granules" perturb the arc everywhere at once; we measure their power spectrum')
        A = [fig.add_axes([0.035, 0.12, 0.215, 0.62]), fig.add_axes([0.265, 0.12, 0.215, 0.62]),
             fig.add_axes([0.495, 0.12, 0.215, 0.62]), fig.add_axes([0.765, 0.14, 0.215, 0.58])]
        A[0].imshow(kap, origin='lower', extent=ext, cmap='PuOr_r', vmin=-0.3, vmax=0.3)
        A[0].set_title('Granule convergence δκ', loc='left', color=MUTED, fontsize=12)
        A[1].imshow(show(img), origin='lower', extent=ext)
        A[1].set_title('Lensed arc', loc='left', color=MUTED, fontsize=12)
        A[2].imshow(dev, origin='lower', extent=ext, cmap=DIVERGING, norm=TwoSlopeNorm(0, -0.1, 0.1))
        A[2].set_title('Residual vs. smooth lens', loc='left', color=MUTED, fontsize=12)
        for a in A[:3]:
            axis_off(a)
        a = A[3]
        a.loglog(k_cdm, p_cdm, color=FAINT, lw=1.5, ls='--', label='CDM subhalos')
        if f > 0 and f % 12 == 0:
            history.append((k, p))
        for kh, ph in history[-4:]:
            a.loglog(kh, ph, color=ACCENT, lw=0.8, alpha=0.3)
        a.loglog(k, p, color=ACCENT, lw=2.2, label='wave-like DM')
        a.set_xlabel('k [kpc⁻¹]'); a.set_ylabel('residual power P(k)')
        a.set_ylim(1e-9, 1e-3)
        a.legend(loc='lower left', fontsize=10, labelcolor=MUTED)
        a.set_title('Residual power spectrum', loc='left', color=MUTED, fontsize=12)
        for sp in a.spines.values():
            sp.set_color(LINE)
        lam_kpc = lam * C.KPC_PER_ARCSEC
        stage = 'lighter boson → larger granules' if f < 60 else 'stronger fluctuations → more power'
        fig.text(0.035, 0.80, f'λ_dB = {lam_kpc:.2f} kpc    δκ_rms = {amp:.3f}    ({stage})', fontsize=12, color=INK, fontdict=MONO)
        footer(fig, 'Illustrative amplitudes (exaggerated for visibility). λ_dB ∝ 1/(m v): at fixed cluster velocity, lighter bosons give larger granules.')
        gw.add(fig)
        plt.close(fig)
    gw.hold(1.5)
    gw.save()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', type=int, default=[1, 2, 3, 4])
    ap.add_argument('--source', nargs=3, metavar=('IMAGE', 'X', 'Y'),
                    help='cut the source galaxy from IMAGE centred on pixel (X, Y)')
    args = ap.parse_args()
    src = load_source(args)
    images = find_images(src)
    print('image positions [arcsec]:', [(round(a, 2), round(b, 2)) for a, b in images])
    for s in args.scenes:
        {1: scene_raytracing, 2: scene_subhalo_move, 3: scene_subhalo_profile, 4: scene_wave_dm}[s](src, images)
