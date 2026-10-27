# Lensing visualizations

Animations and interactive pages for <https://ninoephremidze.github.io/lensing/>.

```
lensing/
  index.html            gallery of the GIFs, links to the interactive pages
  cluster.html          interactive: galaxy cluster lensing, subhalos, wave-like dark matter
  gw.html               interactive: wave-optics lensing of gravitational waves
  lensing.css           shared styles (same tokens as ../index.html)
  js/common.js          FFT, plotting, colour maps, dragging
  js/cluster.js         cluster lensing physics + UI
  js/gw.js              GW lensing physics + UI
  assets/               data the pages load (built by python/build_web_assets.py)
  gifs/                 animations (built by python/make_*_gifs.py)
  python/
    lensviz/cluster.py  lens models, ray tracing, wave-DM granule fields
    lensviz/gw.py       point-mass F(w, y), toy BBH waveform, 2D Huygens wave field
    lensviz/source.py   cut a galaxy out of a JWST image
    lensviz/style.py    figure style and GIF writer
    make_cluster_gifs.py
    make_gw_gifs.py
    build_web_assets.py
```

## Rebuilding

```bash
cd lensing/python
pip install -r requirements.txt
python build_web_assets.py            # galaxy.png, cluster curves; F table on first run (~3 min)
python make_cluster_gifs.py           # 4 GIFs, ~3 min
python make_gw_gifs.py                # 2 GIFs, ~6 min (needs the F table)
```

Options:

- `python make_cluster_gifs.py --scenes 2 3` renders only some scenes.
- `python make_cluster_gifs.py --source my_image.jpg X Y` lenses a different galaxy, cut from
  `my_image.jpg` around pixel (X, Y). The default is a face-on spiral in `../images/deep-field.jpg`.
- `python build_web_assets.py --table` rebuilds the F(w, y) table.

The pages are static, so they work as soon as the files are pushed. To preview locally run
`python -m http.server` in the repository root and open <http://localhost:8000/lensing/>.

## Physics and simplifications

**Cluster lensing.** A cored isothermal ellipsoid (q = 0.5, core 5″, Einstein radius ~18″) at
z_L = 0.4, source at z_S = 1 (flat ΛCDM, H0 = 70). Subhalos are tidally truncated NFW, cored
pseudo-Jaffe or point masses, normalised by total mass. Before showing a perturbation, the
brightness-weighted constant and linear parts of its deflection are removed, mimicking what a
smooth lens + source fit would absorb. Wave-like dark matter is a Gaussian random convergence
field with correlation length λ_dB and an adjustable rms amplitude (exaggerated for visibility).

**GW lensing.** Exact point-mass amplification factor (Takahashi & Nakamura 2003) via mpmath. The
web page and the GIFs use F = F_GO + R, with the residual R tabulated on w ≤ 100,
0.1 ≤ y ≤ 3 and interpolated bilinearly; above w = 100, geometric optics. Against the exact F the
table error is 4×10⁻⁴ (median) and 7×10⁻³ (95th percentile), largest at small y and large w.
Lensing multiplies the waveform by conj(F) (LALSuite Fourier convention). The waveform is a
leading-order chirp with a smooth merger and ringdown, not a precision template. The wave-field
picture is a 2D Huygens–Fresnel sum through a logarithmic phase screen; it is qualitative and
not to scale.
