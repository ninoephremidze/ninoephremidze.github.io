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
  js/gw.js              GW lensing physics + UI (amplification factors from assets/fiona/)
  assets/               data the pages load (built by python/build_web_assets.py)
  assets/fiona/         F(w, y) tables for four lens profiles (built by python/build_fiona_tables.py)
  video/                animations shown on the page, 1920x1080 MP4 (built by python/make_*_gifs.py)
  gifs/                 the same animations as 960x540 GIFs, for downloading
  python/
    lensviz/cluster.py  lens models, ray tracing, wave-DM granule fields
    lensviz/gw.py       point-mass F(w, y), toy BBH waveform, 2D Huygens wave field
    lensviz/source.py   cut a galaxy out of a JWST image
    lensviz/style.py    figure style and GIF writer
    make_cluster_gifs.py
    make_gw_gifs.py
    build_web_assets.py
    build_fiona_tables.py FIONA tables for the GW page (+ submit_fiona_tables.sh for SLURM)
```

## Rebuilding

```bash
cd lensing/python
pip install -r requirements.txt
python build_web_assets.py            # galaxy.png, cluster curves; F table on first run (~3 min)
python make_cluster_gifs.py           # 4 animations (MP4 + GIF)
python make_gw_gifs.py                # 2 animations (MP4 + GIF; needs the F table)
```

MP4s need ffmpeg, which `imageio-ffmpeg` (in requirements.txt) provides; without it only the GIFs
are written.

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

**GW lensing, interactive page.** F(w, y) is computed with FIONA for a point mass, a cored
isothermal sphere (x_c = 0.05, 0.2, 0.5), an NFW halo (x_s = 0.01, 0.1, 1) and a ULA soliton
(r_s = 0.1, 0.3, 1), all with FIONA's psi0 = 1, so lengths are in Einstein radii of a point mass
M_Lz. `build_fiona_tables.py` reuses the FIONA inference code unchanged (v4 Gauss-Legendre
schedule, `fiona_w_table.build_F_w_table`, `fiona_w_table.image_set`; NFW images from the
analytic deflection) and stores R = F - s(w) F_GO on 1178 w nodes in [10⁻³, 100.5] and
y = 0.2, 0.25, ..., 3, referred to the first-arriving image. The page snaps y to these nodes,
interpolates R linearly in w and adds s(w) F_GO back; above w = 100.5 it uses F_GO alone, which
overestimates |F| close to a caustic. Checks (printed by the build, stored in the manifest):
table vs direct FIONA at off-grid w, median 2×10⁻⁶ to 10⁻⁴, 95th percentile ≤ 2.3×10⁻³, worst
1.3×10⁻² (NFW x_s = 0.1 near its caustic); doubling the GL nodes changes F by ≤ 3×10⁻³; point
mass vs the exact solution, median 1.0×10⁻³, max 2.7×10⁻³. Building takes ~4 min on 48 cores
(`cd python && sbatch submit_fiona_tables.sh`; `--fiona-root` points at the FIONA code).

**GW lensing, GIFs.** Exact point-mass amplification factor (Takahashi & Nakamura 2003) via mpmath,
as F = F_GO + R with R tabulated on w ≤ 100, 0.1 ≤ y ≤ 3 (`build_web_assets.py --table`) and
interpolated bilinearly; above w = 100, geometric optics. Against the exact F the table error is
4×10⁻⁴ (median) and 7×10⁻³ (95th percentile).
Lensing multiplies the waveform by conj(F) (LALSuite Fourier convention). The waveform is a
leading-order chirp with a smooth merger and ringdown, not a precision template. The wave-field
picture is a 2D Huygens–Fresnel sum through a phase screen (logarithmic in the GIFs, the selected
profile's psi(r) on the page); it is qualitative and
not to scale.
