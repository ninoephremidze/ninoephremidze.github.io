"""Strong lensing by a galaxy cluster, with dark-matter substructure.

All angles are in arcseconds on the sky. The cluster is a non-singular
isothermal ellipsoid (NIE, Keeton 2001); subhalos are NFW, cored
(pseudo-Jaffe) or point masses; wave-like dark matter is a Gaussian random
convergence field with a coherence length set by the de Broglie wavelength.
"""
import numpy as np

# ---- cosmology for z_lens = 0.4, z_source = 1.0 (flat LCDM, H0 = 70, Om = 0.3) ----
Z_LENS, Z_SOURCE = 0.4, 1.0
_ARCSEC = np.pi / 180 / 3600
_G_OVER_C2 = 4.786e-20          # G M_sun / c^2 in Mpc


def _comoving(z, n=2000):
    zz = np.linspace(0, z, n)
    Ez = np.sqrt(0.3 * (1 + zz) ** 3 + 0.7)
    return 2997.92458 / 0.7 * np.trapezoid(1 / Ez, zz)     # Mpc


_DC_L, _DC_S = _comoving(Z_LENS), _comoving(Z_SOURCE)
D_L = _DC_L / (1 + Z_LENS)
D_S = _DC_S / (1 + Z_SOURCE)
D_LS = (_DC_S - _DC_L) / (1 + Z_SOURCE)
KPC_PER_ARCSEC = D_L * 1e3 * _ARCSEC
# Einstein radius^2 of a 1 M_sun point mass, in arcsec^2
THETA_E2_PER_MSUN = 4 * _G_OVER_C2 * D_LS / (D_L * D_S) / _ARCSEC ** 2
# critical surface density in M_sun / arcsec^2
SIGMA_CRIT = 1 / (np.pi * THETA_E2_PER_MSUN)


# ------------------------------------------------------------------ main cluster
class Cluster:
    """Cored isothermal ellipsoid with external shear. Major axis along x."""

    def __init__(self, b=18.0, q=0.6, s=2.5, gamma=(0.0, 0.0)):
        self.b, self.q, self.s, self.gamma = b, q, s, gamma

    def deflection(self, x, y):
        b, q, s = self.b, self.q, self.s
        e = np.sqrt(1 - q * q)
        psi = np.sqrt(q * q * (s * s + x * x) + y * y)
        ax = b * q / e * np.arctan(e * x / (psi + s))
        ay = b * q / e * np.arctanh(e * y / (psi + q * q * s))
        g1, g2 = self.gamma
        return ax + g1 * x + g2 * y, ay + g2 * x - g1 * y


# ------------------------------------------------------------------ subhalos
class Subhalo:
    """A single dark-matter subhalo.

    kind: 'nfw'    - NFW, truncated at r_t; `shape` = concentration-like compactness
          'cored'  - pseudo-Jaffe with a core; `shape` = core radius [arcsec]
          'point'  - point mass (e.g. a primordial black hole)
    mass: total mass in M_sun.
    """

    def __init__(self, x0, y0, mass=3e9, kind='nfw', shape=0.15, r_t=1.0):
        self.x0, self.y0, self.mass, self.kind, self.shape, self.r_t = x0, y0, mass, kind, shape, r_t

    def _enclosed(self, r):
        """Projected enclosed mass fraction M(<r)/M_tot (dimensionless)."""
        rt = self.r_t
        if self.kind == 'point':
            return np.ones_like(r)
        if self.kind == 'cored':
            sc = max(self.shape, 1e-4)
            # pseudo-Jaffe: kappa ∝ 1/sqrt(r^2+s^2) - 1/sqrt(r^2+a^2), a = r_t
            f = (np.sqrt(r * r + sc * sc) - sc) - (np.sqrt(r * r + rt * rt) - rt)
            return f / (rt - sc)
        # NFW with scale radius r_s = shape, projected mass, smoothly truncated at r_t
        rs = max(self.shape, 1e-3)
        def m2d(R):
            u = np.maximum(R / rs, 1e-8)
            F = np.where(u < 1, np.arccosh(1 / np.minimum(u, 1 - 1e-12)) / np.sqrt(np.maximum(1 - u * u, 1e-24)),
                         np.where(u > 1, np.arccos(1 / np.maximum(u, 1 + 1e-12)) / np.sqrt(np.maximum(u * u - 1, 1e-24)), 1.0))
            return np.log(u / 2) + F
        m = m2d(r) / m2d(np.array(rt))
        return np.clip(np.where(r < rt, m, 1.0), 0, None)

    def deflection(self, x, y):
        dx, dy = x - self.x0, y - self.y0
        r2 = dx * dx + dy * dy + 1e-12
        r = np.sqrt(r2)
        amp = self.mass * THETA_E2_PER_MSUN * self._enclosed(r) / r2
        return amp * dx, amp * dy


# ------------------------------------------------------------------ wave dark matter
def granule_field(n, box, lam, amp, seed=0):
    """Gaussian random convergence field of psi-DM granules on an n x n grid.

    lam: de Broglie wavelength (granule size) in arcsec; amp: rms convergence.
    Returns kappa and its deflection field (ax, ay), all on the same grid.
    The mean is zero, so the mean surface density of the lens is unchanged.
    """
    rng = np.random.default_rng(seed)
    k = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx ** 2 + ky ** 2
    # granules: Gaussian power spectrum centred on k = 0 with width ~ 2 pi / lam
    pk = np.exp(-k2 * (lam / (2 * np.pi)) ** 2 * 2.0)
    white = np.fft.fft2(rng.standard_normal((n, n)))
    kap_k = white * np.sqrt(pk)
    kap_k[0, 0] = 0
    kappa = np.real(np.fft.ifft2(kap_k))
    norm = amp / kappa.std()
    kappa *= norm
    kap_k *= norm
    k2[0, 0] = 1
    # psi_k = -2 kappa_k / k^2 ; alpha = grad psi  ->  alpha_k = i k psi_k
    psi_k = -2 * kap_k / k2
    ax = np.real(np.fft.ifft2(1j * kx * psi_k))
    ay = np.real(np.fft.ifft2(1j * ky * psi_k))
    return kappa, ax, ay


# ------------------------------------------------------------------ ray tracing
def grid(cx, cy, half, n):
    t = np.linspace(-half, half, n)
    return np.meshgrid(cx + t, cy + t)


def ray_trace(x, y, lenses, extra=None):
    """Map image-plane positions to the source plane: beta = theta - alpha."""
    bx, by = x.copy(), y.copy()
    for L in lenses:
        ax, ay = L.deflection(x, y)
        bx -= ax
        by -= ay
    if extra is not None:
        bx -= extra[0]
        by -= extra[1]
    return bx, by


def magnification_det(x, y, lenses, h=1e-3):
    def a(xx, yy):
        return ray_trace(xx, yy, lenses)
    bxp, byp = a(x + h, y); bxm, bym = a(x - h, y)
    bxq, byq = a(x, y + h); bxn, byn = a(x, y - h)
    A11 = (bxp - bxm) / (2 * h); A21 = (byp - bym) / (2 * h)
    A12 = (bxq - bxn) / (2 * h); A22 = (byq - byn) / (2 * h)
    return A11 * A22 - A12 * A21


def critical_curves(lenses, half=40, n=800):
    """Critical curves (image plane) and caustics (source plane) as lists of (N,2) arrays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, y = grid(0, 0, half, n)
    det = magnification_det(x, y, lenses)
    fig = plt.figure()
    cs = plt.contour(x, y, det, levels=[0])
    plt.close(fig)
    crit = [p for p in cs.allsegs[0] if len(p) > 10]
    caus = []
    for p in crit:
        bx, by = ray_trace(p[:, 0], p[:, 1], lenses)
        caus.append(np.c_[bx, by])
    return crit, caus


def sample(img, bx, by, src_center, src_size):
    """Bilinear sample of a source image (H, W, C) placed at src_center with width src_size [arcsec]."""
    H, W = img.shape[:2]
    u = (bx - src_center[0]) / src_size * W + W / 2 - 0.5
    v = (src_center[1] - by) / src_size * H + H / 2 - 0.5      # image rows go down
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    fu = (u - u0)[..., None]; fv = (v - v0)[..., None]
    out = np.zeros(bx.shape + img.shape[2:], dtype=float)
    def px(uu, vv):
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        res = np.zeros(bx.shape + img.shape[2:])
        res[ok] = img[vv[ok], uu[ok]]
        return res
    out = (px(u0, v0) * (1 - fu) * (1 - fv) + px(u0 + 1, v0) * fu * (1 - fv)
           + px(u0, v0 + 1) * (1 - fu) * fv + px(u0 + 1, v0 + 1) * fu * fv)
    return out
