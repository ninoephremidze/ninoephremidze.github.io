"""Shared look for all figures (matches ninoephremidze.github.io) and a GIF writer."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

PAPER = '#05060A'
INK = '#F4F2EC'
MUTED = '#BDB9B0'
FAINT = '#8E8B84'
LINE = '#3a3a40'
ACCENT = '#E8919E'      # light garnet, the site accent
CYAN = '#7FD3E6'        # critical curves / lensed light paths
GOLD = '#E8C27A'        # caustics

_here = os.path.dirname(os.path.abspath(__file__))
_fonts = os.path.join(os.path.dirname(_here), 'fonts')
for f in ('Newsreader.ttf', 'JetBrainsMono.ttf'):
    p = os.path.join(_fonts, f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)

plt.rcParams.update({
    'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
    'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': LINE,
    'xtick.color': FAINT, 'ytick.color': FAINT, 'axes.titlecolor': INK,
    'font.family': ['Newsreader', 'DejaVu Serif'], 'font.size': 13,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'mathtext.fontset': 'dejavuserif',
    'legend.frameon': False, 'axes.grid': False,
})
MONO = {'family': ['JetBrains Mono', 'DejaVu Sans Mono']}


def fig_to_array(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return a


class GifWriter:
    """Collect frames, then write a palette-optimised looping GIF."""

    def __init__(self, path, fps=12):
        self.path, self.fps, self.frames = path, fps, []

    def add(self, fig_or_array, repeat=1):
        a = fig_to_array(fig_or_array) if hasattr(fig_or_array, 'canvas') else fig_or_array
        self.frames += [a] * repeat

    def hold(self, seconds):
        self.frames += [self.frames[-1]] * int(round(seconds * self.fps))

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # one global palette avoids flicker between frames
        sample = np.concatenate([f[::4, ::4].reshape(-1, 3) for f in self.frames[::max(1, len(self.frames) // 24)]])
        pal = Image.fromarray(sample.reshape(1, -1, 3)).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in self.frames]
        ims[0].save(self.path, save_all=True, append_images=ims[1:], duration=int(1000 / self.fps),
                    loop=0, optimize=True, disposal=1)
        print(f'wrote {self.path}: {len(ims)} frames, {os.path.getsize(self.path) / 1e6:.1f} MB')


def ease(t):
    """Smooth 0->1 easing."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


from matplotlib.colors import LinearSegmentedColormap
# dark-centred diverging map: deficit in blue, excess in the site's garnet
DIVERGING = LinearSegmentedColormap.from_list('lens_div', ['#9fd0ff', '#2b6cb0', '#0E0F15', '#b5405a', '#ffc2cc'])


def fmt_mass(m):
    e = int(np.floor(np.log10(m)))
    a = m / 10 ** e
    sup = str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return (f'{a:.1f}×10{sup} M☉' if abs(a - 1) > 0.05 else f'10{sup} M☉')
