"""Animated GIFs: wave-optics lensing of gravitational waves by a dark-matter subhalo.

Scenes:
  1. gw_lensing_intro.gif - a binary black hole merges; its gravitational waves pass a
                            compact dark-matter lens on the way to the detector, which
                            records an interference-modulated chirp.
  2. gw_lensing_scan.gif  - increasing the lens mass takes the signal from diffraction
                            (lens invisible) to interference (beating) to geometric optics
                            (two copies, an "echo"); then varying the impact parameter.

Left panel: a 2D Huygens–Fresnel toy model of the wave field (qualitative, not to scale).
Right panels: the actual point-mass amplification factor F(w, y) and the lensed waveform.

Usage:  python make_gw_gifs.py [--scenes 1 2]
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from lensviz import gw as G
from lensviz.style import GifWriter, ease, fmt_mass, PAPER, INK, MUTED, FAINT, LINE, ACCENT, CYAN, MONO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'gifs')
DPI = 150
FIGSIZE = (12.8, 7.2)          # 1920 x 1080 px (the GIF is downsampled to 960 x 540)

SRC = (0.6, 0.0)
LENS_X = 5.0
DET = (11.6, 0.0)
WAVELENGTH = 0.45
K = 2 * np.pi / WAVELENGTH
FIELD_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    'gwfield', ['#9fd0ff', '#2b6cb0', '#0E0F15', '#b5405a', '#ffc2cc'])

T_WAVE, H_WAVE = G.toy_bbh()


def strength_2d(M_lz):
    """Map the lens mass onto the 2D toy's phase-screen strength (monotonic, not to scale)."""
    w_ref = G.w_of_f(100.0, M_lz)
    return 0.3 * np.sqrt(w_ref / 3.7)


def einstein_2d(s):
    d_l, d_ls = LENS_X - SRC[0], DET[0] - LENS_X
    return np.sqrt(s * d_l * d_ls / (d_l + d_ls))


def field_for(M_lz, y):
    s = strength_2d(M_lz)
    lens_y = -y * einstein_2d(s)
    xs, ys, U = G.huygens_field(source=SRC, lens=(LENS_X, lens_y), wavelength=WAVELENGTH, strength=s,
                                  nx=540, ny=270)
    X = xs[None, :]
    U = U * np.sqrt(np.hypot(X - SRC[0], ys[:, None] - SRC[1]) + 0.3)   # undo 1/sqrt(r) spreading for display
    return xs, ys, U, lens_y, s


def draw_scene(ax, xs, ys, U, lens_y, s, phase, t_anim, front=None):
    vals = np.real(U * np.exp(-1j * phase))
    if front is not None:   # wave has only travelled `front` from the source so far
        R = np.hypot(xs[None, :] - SRC[0], ys[:, None] - SRC[1])
        vals = vals * np.clip((front - R) / 0.6, 0, 1)
    ax.imshow(vals, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap=FIELD_CMAP,
              vmin=-2.2, vmax=2.2, interpolation='bilinear', aspect='auto')
    # the binary: two black holes orbiting faster and faster
    om = 2 * np.pi * (0.6 + 2.5 * t_anim ** 3)
    a = 0.16 * (1 - 0.6 * t_anim)
    for sgn in (1, -1):
        ax.add_patch(Circle((SRC[0] + sgn * a * np.cos(om * t_anim * 6), SRC[1] + sgn * a * np.sin(om * t_anim * 6)),
                            0.07, color=INK, zorder=5))
    # the lens: a soft glow whose size grows with the lens strength
    for r, al in ((0.55, 0.10), (0.35, 0.18), (0.18, 0.35)):
        ax.add_patch(Circle((LENS_X, lens_y), r * (0.6 + 0.6 * s), color=ACCENT, alpha=al, lw=0, zorder=4))
    ax.plot([LENS_X, LENS_X], [ys[0], ys[-1]], color=INK, lw=0.6, ls=(0, (2, 4)), alpha=0.35)
    # detector: a small L-shaped interferometer
    dx, dy = DET
    ax.plot([dx, dx + 0.0], [dy, dy + 0.35], color=INK, lw=2, zorder=6)
    ax.plot([dx, dx - 0.35], [dy, dy], color=INK, lw=2, zorder=6)
    ax.text(SRC[0], ys[0] + 0.25, 'binary black hole', color=INK, fontsize=11, ha='left')
    ax.text(LENS_X + 0.1, ys[-1] - 0.45, 'dark matter subhalo', color=ACCENT, fontsize=11, ha='left')
    ax.text(dx - 0.1, ys[0] + 0.25, 'detector', color=INK, fontsize=11, ha='right')
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)


def draw_waveform(ax, hl, upto=None, t0=2.62, t1=3.06):
    m = (T_WAVE > t0) & (T_WAVE < t1)
    tt = (T_WAVE[m] - 3.0) * 1e3
    ax.plot(tt, H_WAVE[m], color=FAINT, lw=1.0, label='unlensed')
    if upto is None:
        ax.plot(tt, hl[m], color=ACCENT, lw=1.4, label='lensed')
    else:
        k = int(upto * m.sum())
        ax.plot(tt[:k], hl[m][:k], color=ACCENT, lw=1.4, label='lensed')
    ax.set_xlim(tt[0], tt[-1]); ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel('time to merger [ms]'); ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=10, labelcolor=MUTED, ncol=2)
    ax.set_title('Waveform at the detector', loc='left', color=MUTED, fontsize=12)
    for sp in ax.spines.values():
        sp.set_color(LINE)


def draw_F(ax, f, Ff, M_lz):
    m = (f > 5) & (f < 2000)
    ax.axvspan(20, 1000, color=INK, alpha=0.05, lw=0)
    ax.text(25, 2.75, 'LIGO band', color=FAINT, fontsize=9.5)
    ax.semilogx(f[m], np.abs(Ff[m]), color=CYAN, lw=1.3)
    ax.axhline(1, color=FAINT, lw=0.6, ls=':')
    # where w = 1: below it the wave diffracts around the lens
    f1 = 1 / (8 * np.pi * G.T_SUN * M_lz)
    if 5 < f1 < 2000:
        ax.axvline(f1, color=ACCENT, lw=0.8, ls=(0, (3, 3)))
        ax.text(f1 * 1.08, 0.25, 'w = 1', color=ACCENT, fontsize=10)
    ax.set_xlim(5, 2000); ax.set_ylim(0, 3.1)
    ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('|F(f)|')
    ax.set_title('Amplification factor', loc='left', color=MUTED, fontsize=12)
    for sp in ax.spines.values():
        sp.set_color(LINE)


def layout(title, sub):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.text(0.035, 0.935, title, fontsize=22, color=INK, va='center')
    fig.text(0.035, 0.875, sub, fontsize=13, color=MUTED, va='center')
    a_field = fig.add_axes([0.035, 0.12, 0.555, 0.68])
    a_wave = fig.add_axes([0.64, 0.56, 0.33, 0.24])
    a_F = fig.add_axes([0.64, 0.18, 0.33, 0.24])
    return fig, a_field, a_wave, a_F


def readout(fig, M_lz, y):
    dt = G.time_delay_seconds(M_lz, y) * 1e3
    fig.text(0.035, 0.04, f'M_Lz = {fmt_mass(M_lz):<12}  y = {y:.2f}   Δt = {dt:6.2f} ms   w(100 Hz) = {G.w_of_f(100, M_lz):5.2f}',
             fontsize=10.5, color=MUTED, fontdict=MONO)
    fig.text(0.97, 0.04, 'left: 2D wave toy, not to scale · right: point-mass lens, 30+30 M☉ binary',
             fontsize=9, color=FAINT, ha='right', fontdict=MONO)


def scene_intro():
    gwr = GifWriter(f'{OUT}/gw_lensing_intro.gif', fps=12)
    M, y = 300.0, 0.6
    xs, ys, U, ly, s = field_for(M, y)
    hl, f, Ff = G.lens_waveform(T_WAVE, H_WAVE, M, y)
    n = 84
    for i in range(n):
        u = i / (n - 1)
        fig, a1, a2, a3 = layout('Gravitational waves passing a dark matter subhalo',
                                 'When the wavelength is comparable to the lens, the waves diffract and interfere')
        front = 2 + 14 * ease(u / 0.5)
        draw_scene(a1, xs, ys, U, ly, s, phase=2 * np.pi * i / 6, t_anim=u, front=front)
        arrived = np.clip((u - 0.42) / 0.5, 0, 1)
        draw_waveform(a2, hl, upto=max(arrived, 1e-3))
        draw_F(a3, f, Ff, M)
        readout(fig, M, y)
        gwr.add(fig)
        plt.close(fig)
    gwr.hold(2.0)
    gwr.save()


def scene_scan():
    gwr = GifWriter(f'{OUT}/gw_lensing_scan.gif', fps=12)
    keys = [(M, 0.6, 'Heavier lens: diffraction → interference → two images (an echo)') for M in np.geomspace(10, 3000, 70)]
    keys += [(3000.0, y, 'Larger impact parameter: weaker second image, longer delay') for y in np.linspace(0.6, 1.8, 36)]
    keys += [(M, 1.8, 'Back to a light lens: the waves diffract around it') for M in np.geomspace(3000, 10, 30)]
    for i, (M, y, sub) in enumerate(keys):
        xs, ys, U, ly, s = field_for(M, y)
        hl, f, Ff = G.lens_waveform(T_WAVE, H_WAVE, M, y)
        fig, a1, a2, a3 = layout('From wave optics to geometric optics', sub)
        draw_scene(a1, xs, ys, U, ly, s, phase=2 * np.pi * i / 6, t_anim=0.5)
        draw_waveform(a2, hl)
        draw_F(a3, f, Ff, M)
        readout(fig, M, y)
        gwr.add(fig)
        plt.close(fig)
        print(f'frame {i + 1}/{len(keys)}', end='\r')
    gwr.hold(1.5)
    gwr.save()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', type=int, default=[1, 2])
    args = ap.parse_args()
    for s in args.scenes:
        {1: scene_intro, 2: scene_scan}[s]()
