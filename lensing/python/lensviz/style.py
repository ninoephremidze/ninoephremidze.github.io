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


def _ffmpeg():
    import shutil
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which('ffmpeg')


def to_webm(mp4):
    """VP9 copy of an MP4, the page's fallback for browsers without H.264 (e.g. some Linux Chromium builds)."""
    import subprocess
    out = os.path.splitext(mp4)[0] + '.webm'
    subprocess.run([_ffmpeg(), '-y', '-loglevel', 'error', '-i', mp4, '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0',
                    '-row-mt', '1', '-pix_fmt', 'yuv420p', '-an', out], check=True)
    return out


class GifWriter:
    """Collect frames, then write a looping MP4 at full resolution and a smaller GIF.

    The page shows the MP4 (sharp, small); the GIF is for downloading. Frames are
    streamed to ffmpeg as they arrive, so only the downsampled GIF frames stay in memory.
    The MP4 goes to ../video/<name>.mp4 next to the GIF folder; it is skipped if
    ffmpeg is not available (pip install imageio-ffmpeg).
    """

    def __init__(self, path, fps=12, gif_width=960):
        self.path, self.fps, self.gif_width, self.frames = path, fps, gif_width, []
        gif_dir = os.path.dirname(os.path.abspath(path))
        name = os.path.splitext(os.path.basename(path))[0]
        self.video_path = os.path.join(os.path.dirname(gif_dir), 'video', name + '.mp4')
        self._proc, self._last = None, None

    def _write_video(self, a):
        if self._proc is None:
            exe = _ffmpeg()
            if exe is None:
                self._proc = False
                print('ffmpeg not found: skipping MP4 (pip install imageio-ffmpeg)')
                return
            import subprocess
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            h, w = a.shape[:2]
            self._proc = subprocess.Popen(
                [exe, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                 '-r', str(self.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                 '-tune', 'animation', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.video_path],
                stdin=subprocess.PIPE)
        if self._proc:
            self._proc.stdin.write(np.ascontiguousarray(a).tobytes())

    def add(self, fig_or_array, repeat=1):
        a = fig_to_array(fig_or_array) if hasattr(fig_or_array, 'canvas') else fig_or_array
        im = Image.fromarray(a)
        if im.width > self.gif_width:
            im = im.resize((self.gif_width, round(im.height * self.gif_width / im.width)), Image.Resampling.LANCZOS)
        small = np.asarray(im)
        for _ in range(repeat):
            self._write_video(a)
            self.frames.append(small)
        self._last = a

    def hold(self, seconds):
        for _ in range(int(round(seconds * self.fps))):
            self._write_video(self._last)
            self.frames.append(self.frames[-1])

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait()
            print(f'wrote {self.video_path}: {os.path.getsize(self.video_path) / 1e6:.1f} MB')
            webm = to_webm(self.video_path)
            print(f'wrote {webm}: {os.path.getsize(webm) / 1e6:.1f} MB')
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
