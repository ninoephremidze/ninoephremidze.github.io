"""Wave-optics lensing of gravitational waves by a compact dark-matter lens.

Physics conventions follow Takahashi & Nakamura (2003):
  w = 8 pi G M_Lz f / c^3   (dimensionless frequency, M_Lz the redshifted lens mass)
  y = impact parameter in units of the Einstein radius
  F(w, y) multiplies the frequency-domain waveform; arrival times are measured
  from the first (minimum) image.

For a point-mass lens F is known in closed form (a confluent hypergeometric
function). Above w ~ 100 we switch to geometric optics, where it is accurate to
better than ~1%. Waveforms use the LALSuite Fourier convention
(h(f) = int h(t) e^{-2 pi i f t} dt), so the lensed waveform is conj(F) * h(f).
"""
import numpy as np
import mpmath as mp

T_SUN = 4.925490947e-6          # G M_sun / c^3 in seconds
mp.mp.dps = 25


# ------------------------------------------------------------------ amplification factor
def _phi_min(y):
    xm = (y + np.sqrt(y * y + 4)) / 2
    return (xm - y) ** 2 / 2 - np.log(xm)


def F_point_exact(w, y):
    """Exact point-mass amplification factor F(w, y) for scalar w > 0."""
    if w <= 0:
        return 1.0 + 0j
    phim = _phi_min(y)
    val = (mp.exp(mp.pi * w / 4 + 1j * w / 2 * (mp.log(w / 2) - 2 * phim))
           * mp.gamma(1 - 1j * w / 2) * mp.hyp1f1(1j * w / 2, 1, 1j * w * y * y / 2))
    return complex(val)


def point_images(y):
    """Magnifications of the two images and their dimensionless time delay."""
    s = np.sqrt(y * y + 4)
    mu_p = 0.5 + (y * y + 2) / (2 * y * s)
    mu_m = 0.5 - (y * y + 2) / (2 * y * s)
    dT = y * s / 2 + np.log((s + y) / (s - y))
    return mu_p, mu_m, dT


def F_point_go(w, y):
    """Geometric-optics limit: two images, the second delayed by dT with a Morse phase -pi/2."""
    mu_p, mu_m, dT = point_images(y)
    return np.sqrt(abs(mu_p)) - 1j * np.sqrt(abs(mu_m)) * np.exp(1j * np.asarray(w) * dT)


def F_point(w, y, w_switch=100.0):
    """Vectorised F(w, y): exact below w_switch, geometric optics above."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    out = F_point_go(w, y).astype(complex)
    lo = w < w_switch
    out[lo] = [F_point_exact(wi, y) for wi in w[lo]]
    return out


def w_of_f(f, M_lz):
    return 8 * np.pi * T_SUN * M_lz * np.asarray(f)


def time_delay_seconds(M_lz, y):
    return 4 * T_SUN * M_lz * point_images(y)[2]


# ------------------------------------------------------------------ a toy BBH waveform
def toy_bbh(m1=30.0, m2=30.0, fs=4096, T=4.0, f_low=20.0, t_merge=3.0, f_rd=250.0, tau_rd=0.004):
    """Inspiral (leading-order post-Newtonian chirp) + smooth merger + ringdown.

    Not a precision waveform: it has the right shape, frequency evolution and
    duration for a GW150914-like system, which is what the illustrations need.
    Returns t [s], h(t) (peak-normalised).
    """
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    Mc = eta ** 0.6 * M * T_SUN                           # chirp mass in seconds
    t = np.arange(int(fs * T)) / fs
    f_isco = 1 / (6 ** 1.5 * np.pi * M * T_SUN)
    tau = np.clip(t_merge - t, 1e-6, None)
    f_insp = (5 / 256 / tau) ** 0.375 * Mc ** -0.625 / np.pi
    t_isco = t_merge - 5 / 256 * Mc ** (-5 / 3) * (np.pi * f_isco) ** (-8 / 3)
    # frequency: chirp up to ISCO, then blend smoothly to the ringdown frequency
    blend = 0.5 * (1 + np.tanh((t - t_isco) / 0.004 - 1.5))
    f = np.where(t < t_isco, f_insp, f_isco) * (1 - blend) + f_rd * blend
    f = np.where(f_insp < f_low, np.nan, f)
    start = np.nanargmin(np.abs(f_insp - f_low))
    f[:start] = f_low
    phase = 2 * np.pi * np.cumsum(f) / fs
    amp = (np.minimum(f, f_isco * 1.6) / f_low) ** (2 / 3)
    t_peak = t_isco + 0.012
    ring = np.exp(-np.clip(t - t_peak, 0, None) / tau_rd)
    amp = amp * ring
    taper = 0.5 * (1 - np.cos(np.pi * np.clip((t - t[start]) / 0.1, 0, 1)))
    h = amp * np.cos(phase) * taper
    h[:start] = 0
    return t, h / np.abs(h).max()


def lens_waveform(t, h, M_lz, y, f_max=2100.0, fast=True):
    """Apply point-mass lensing in the frequency domain; returns lensed h(t), f, F(f)."""
    n = len(t)
    fs = 1 / (t[1] - t[0])
    hf = np.fft.rfft(h)
    f = np.fft.rfftfreq(n, 1 / fs)
    Ff = np.ones_like(f, dtype=complex)
    band = (f > 0) & (f < f_max)
    Ff[band] = F_point_fast(w_of_f(f[band], M_lz), y) if fast else F_point(w_of_f(f[band], M_lz), y)
    hl = np.fft.irfft(np.conj(Ff) * hf, n)        # conj: LALSuite convention (later image arrives later)
    return hl, f, Ff


# ------------------------------------------------------------------ a 2D wave picture
def huygens_field(nx=360, ny=180, xlim=(0.0, 12.0), ylim=(-3.0, 3.0), source=(0.6, 0.0),
                  lens=(5.0, 0.0), wavelength=0.45, strength=0.6, core=0.03, n_lens=700, lens_half=7.0):
    """Monochromatic 2D wave from a point source, passing a thin 'gravitational' phase screen.

    A toy, two-dimensional version of the Fresnel–Kirchhoff integral: behind the
    lens plane the field is a Huygens sum over secondary sources on that plane,
    each delayed by the lens potential  strength * ln sqrt(l^2 + core^2)  (in units of
    path length), the 2D analogue of the Shapiro delay of a point mass.
    The interference pattern depends on w ~ (2 pi / wavelength) * strength.
    Returns the complex field U on the grid (x along the line of sight).
    """
    k = 2 * np.pi / wavelength
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X, dtype=complex)
    sx, sy = source
    lx, ly = lens
    # region before the lens plane: free cylindrical wave
    pre = X < lx
    r = np.hypot(X - sx, Y - sy) + 1e-3
    U[pre] = np.exp(1j * k * r[pre]) / np.sqrt(r[pre])
    # Huygens sum behind the lens plane
    l = np.linspace(ly - lens_half, ly + lens_half, n_lens)
    dl = l[1] - l[0]
    r1 = np.hypot(lx - sx, l - sy)
    psi = strength * np.log(np.sqrt((l - ly) ** 2 + core ** 2))
    a1 = np.exp(1j * k * (r1 - psi)) / np.sqrt(r1)
    win = np.cos(np.pi / 2 * np.clip(np.abs(l - ly) / lens_half, 0, 1)) ** 0.5   # soften the edges of the screen
    post = ~pre
    Xp, Yp = X[post], Y[post]
    out = np.zeros(Xp.size, dtype=complex)
    for i0 in range(0, Xp.size, 4000):
        xp = Xp[i0:i0 + 4000, None]; yp = Yp[i0:i0 + 4000, None]
        r2 = np.hypot(xp - lx, yp - l[None, :]) + 1e-3
        obl = (xp - lx) / r2                                   # obliquity factor
        out[i0:i0 + 4000] = (a1 * win * np.exp(1j * k * r2) * obl / np.sqrt(r2)).sum(1)
    U[post] = out * dl * np.sqrt(k / (2 * np.pi)) * np.exp(-1j * np.pi / 4)
    return xs, ys, U


# ------------------------------------------------------------------ fast tabulated F (same as the web page)
_TABLE = None


def _load_table():
    global _TABLE
    if _TABLE is None:
        import json, os
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets')
        meta = json.load(open(os.path.join(d, 'F_point_table.json')))
        R = np.fromfile(os.path.join(d, 'F_point_table.bin'), dtype=np.float32).reshape(meta['n_y'], meta['n_w'], 2)
        _TABLE = (meta, R[..., 0] + 1j * R[..., 1])
    return _TABLE


def F_point_fast(w, y):
    """F(w, y) = F_GO + interpolated residual (built by build_web_assets.py); GO above w_max."""
    meta, R = _load_table()
    w = np.atleast_1d(np.asarray(w, dtype=float))
    out = F_point_go(w, y).astype(complex)
    lo = w < meta['w_max'] - meta['dw']
    j = np.clip((y - meta['y_min']) / meta['dy'], 0, meta['n_y'] - 1.001)
    j0, fj = int(j), j - int(j)
    i = w[lo] / meta['dw']
    i0 = i.astype(int); fi = i - i0
    r = ((1 - fj) * ((1 - fi) * R[j0, i0] + fi * R[j0, i0 + 1])
         + fj * ((1 - fi) * R[j0 + 1, i0] + fi * R[j0 + 1, i0 + 1]))
    out[lo] += r
    return out
