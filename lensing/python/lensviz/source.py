"""Source galaxies: cut a real galaxy out of a JWST image, or build a synthetic spiral."""
import numpy as np
from PIL import Image


def cutout(path, x, y, size=90, taper=0.85):
    """Cut a galaxy out of an RGB image, subtract the sky and taper the edges to zero.

    (x, y) is the galaxy centre in pixels; returns a float array (size, size, 3) in [0, 1].
    """
    im = np.asarray(Image.open(path).convert('RGB'), dtype=float) / 255
    h = size // 2
    c = im[y - h:y + h, x - h:x + h].copy()
    sky = np.median(np.concatenate([c[:5].reshape(-1, 3), c[-5:].reshape(-1, 3)]), axis=0)
    c = np.clip(c - sky, 0, None)
    t = np.linspace(-1, 1, c.shape[0])
    r = np.hypot(*np.meshgrid(t, t))
    w = np.clip((1 - r) / (1 - taper), 0, 1) ** 2
    c *= w[..., None]
    return c / c.max()


def synthetic_spiral(n=128, arms=2, seed=1):
    """A clumpy two-armed spiral, for when no real image is available."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, n)
    x, y = np.meshgrid(t, t)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    bulge = np.exp(-(r / 0.12) ** 2)
    disk = np.exp(-r / 0.3)
    spiral = (0.5 + 0.5 * np.cos(arms * th - 7 * r)) ** 3 * disk
    clumps = np.zeros_like(r)
    for _ in range(25):
        rr, tt = rng.uniform(0.15, 0.7), rng.uniform(0, 2 * np.pi)
        cx, cy = rr * np.cos(tt), rr * np.sin(tt)
        clumps += rng.uniform(.3, 1) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.0015)
    img = np.stack([1.0 * bulge + .6 * spiral + .5 * clumps,
                    .8 * bulge + .7 * spiral + .7 * clumps,
                    .6 * bulge + .9 * spiral + 1.0 * clumps], -1)
    img *= np.clip(1 - r, 0, 1)[..., None]
    return img / img.max()
