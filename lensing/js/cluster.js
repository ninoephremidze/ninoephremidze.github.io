// Interactive strong lensing by a galaxy cluster, with a subhalo or wave-like dark matter.
// Same physics as python/lensviz/cluster.py (angles in arcsec).
'use strict';

const S = {
  kpc: 5.373, te2: 3.898e-12,          // kpc per arcsec, Einstein radius^2 per M_sun [arcsec^2] (z_L=0.4, z_S=1)
  cl: { b: 18, q: 0.5, s: 5 },
  src: { x: 2.8, y: 0.7, size: 0.75 },
  srcView: { cx: 1.5, cy: 0, half: 5 },
  full: { half: 16, n: 320 },
  zoom: { cx: 7.47, cy: -7.16, half: 1.6, n: 256 },
  sub: { x: 7.62, y: -6.81, logM: 9.7, kind: 'nfw', shapeKpc: 0.43, rt: 0.3 },
  mode: 'subhalo',
  wave: { lamKpc: 0.8, amp: 0.06, seed: 7 },
  curves: null,
};

// ---------------------------------------------------------------- lens models
const E = Math.sqrt(1 - S.cl.q * S.cl.q);
function clusterDefl(x, y) {
  const { b, q, s } = S.cl;
  const psi = Math.sqrt(q * q * (s * s + x * x) + y * y);
  return [b * q / E * Math.atan(E * x / (psi + s)), b * q / E * Math.atanh(E * y / (psi + q * q * s))];
}

function nfwM2d(R, rs) {
  const u = Math.max(R / rs, 1e-8);
  let F;
  if (u < 1 - 1e-9) F = Math.acosh(1 / u) / Math.sqrt(1 - u * u);
  else if (u > 1 + 1e-9) F = Math.acos(1 / u) / Math.sqrt(u * u - 1);
  else F = 1;
  return Math.log(u / 2) + F;
}
function enclosedFraction(r, kind, shape, rt) {
  if (kind === 'point') return 1;
  if (r >= rt) return 1;
  if (kind === 'cored') {
    const sc = Math.max(shape, 1e-4);
    return ((Math.sqrt(r * r + sc * sc) - sc) - (Math.sqrt(r * r + rt * rt) - rt)) / (rt - sc);
  }
  const rs = Math.max(shape, 1e-3);
  return Math.max(0, nfwM2d(r, rs) / nfwM2d(rt, rs));
}
function subDefl(x, y) {
  const s = S.sub, dx = x - s.x, dy = y - s.y, r2 = dx * dx + dy * dy + 1e-12, r = Math.sqrt(r2);
  const shape = s.kind === 'cored' ? Math.min(s.shapeKpc / S.kpc, s.rt * 0.95) : s.shapeKpc / S.kpc;
  const a = Math.pow(10, s.logM) * S.te2 * enclosedFraction(r, s.kind, shape, s.rt) / r2;
  return [a * dx, a * dy];
}

// ---------------------------------------------------------------- source galaxy
let GAL = null;   // {w, h, data: Float32Array rgb}
function setGalaxy(img, taper) {
  const n = 128, c = document.createElement('canvas'); c.width = c.height = n;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0, n, n);
  const d = g.getImageData(0, 0, n, n).data, out = new Float32Array(n * n * 3);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const r = Math.hypot((x + .5) / n * 2 - 1, (y + .5) / n * 2 - 1);
    const w = taper ? Math.pow(Math.min(1, Math.max(0, (1 - r) / 0.4)), 2) : 1;
    for (let k = 0; k < 3; k++) out[3 * (y * n + x) + k] = d[4 * (y * n + x) + k] / 255 * w;
  }
  GAL = { w: n, h: n, data: out };
}
function sampleGal(bx, by, out, o) {
  const G = GAL, u = (bx - S.src.x) / S.src.size * G.w + G.w / 2 - 0.5, v = (S.src.y - by) / S.src.size * G.h + G.h / 2 - 0.5;
  out[o] = out[o + 1] = out[o + 2] = 0;
  if (u < -1 || v < -1 || u >= G.w || v >= G.h) return;
  const u0 = Math.floor(u), v0 = Math.floor(v), fu = u - u0, fv = v - v0;
  const add = (uu, vv, wgt) => {
    if (uu < 0 || vv < 0 || uu >= G.w || vv >= G.h || wgt === 0) return;
    const k = 3 * (vv * G.w + uu);
    out[o] += G.data[k] * wgt; out[o + 1] += G.data[k + 1] * wgt; out[o + 2] += G.data[k + 2] * wgt;
  };
  add(u0, v0, (1 - fu) * (1 - fv)); add(u0 + 1, v0, fu * (1 - fv)); add(u0, v0 + 1, (1 - fu) * fv); add(u0 + 1, v0 + 1, fu * fv);
}

function putRGB(ctx, n, rgb, gain) {
  const img = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) for (let k = 0; k < 3; k++)
    img.data[4 * i + k] = 255 * Math.pow(Math.min(1, rgb[3 * i + k] * (gain || 1)), 0.75);
  for (let i = 0; i < n * n; i++) img.data[4 * i + 3] = 255;
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------- grids (cached cluster mapping)
function makeGrid(cx, cy, half, n) {
  const bx = new Float32Array(n * n), by = new Float32Array(n * n), X = new Float32Array(n * n), Y = new Float32Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = cx - half + 2 * half * (i + .5) / n, y = cy + half - 2 * half * (j + .5) / n, k = j * n + i;
    const a = clusterDefl(x, y);
    X[k] = x; Y[k] = y; bx[k] = x - a[0]; by[k] = y - a[1];
  }
  return { X, Y, bx, by, n, cx, cy, half };
}
let FULL = null, ZOOM = null;

// ---------------------------------------------------------------- drawing helpers
function toPix(grid, x, y, cv) {
  return [(x - (grid.cx - grid.half)) / (2 * grid.half) * cv.width, ((grid.cy + grid.half) - y) / (2 * grid.half) * cv.height];
}
function drawCurves(ctx, cv, view, list, color) {
  ctx.strokeStyle = color; ctx.lineWidth = Math.max(1, cv.width / 300);
  for (const curve of list) {
    ctx.beginPath();
    curve.forEach((p, i) => { const q = toPix(view, p[0], p[1], cv); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
    ctx.closePath(); ctx.stroke();
  }
}

// ---------------------------------------------------------------- source & image planes
const cvSrc = document.getElementById('cv-source'), cvFull = document.getElementById('cv-image');
const cvZoom = document.getElementById('cv-zoom'), cvRes = document.getElementById('cv-resid');
const cvPk = document.getElementById('cv-pk'), cvKap = document.getElementById('cv-kappa');

function drawSource() {
  if (!GAL) return;
  const n = 256, half = S.srcView.half, ctx = cvSrc.getContext('2d');
  cvSrc.width = cvSrc.height = n;
  const rgb = new Float32Array(n * n * 3);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
    sampleGal(S.srcView.cx - half + 2 * half * (i + .5) / n, S.srcView.cy + half - 2 * half * (j + .5) / n, rgb, 3 * (j * n + i));
  putRGB(ctx, n, rgb, 1.0);
  const view = { cx: S.srcView.cx, cy: S.srcView.cy, half };
  if (S.curves) drawCurves(ctx, cvSrc, view, S.curves.caustic, 'rgba(232,194,122,.85)');
  const p = toPix(view, S.src.x, S.src.y, cvSrc);
  ctx.strokeStyle = 'rgba(232,145,158,.8)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(p[0], p[1], 11, 0, 7); ctx.stroke(); ctx.setLineDash([]);
}

function drawFull() {
  if (!FULL || !GAL) return;
  const g = FULL, n = g.n, ctx = cvFull.getContext('2d');
  cvFull.width = cvFull.height = n;
  const rgb = new Float32Array(n * n * 3);
  for (let k = 0; k < n * n; k++) sampleGal(g.bx[k], g.by[k], rgb, 3 * k);
  putRGB(ctx, n, rgb, 1.1);
  if (S.curves) drawCurves(ctx, cvFull, g, S.curves.critical, 'rgba(127,211,230,.55)');
  const a = toPix(g, S.zoom.cx - S.zoom.half, S.zoom.cy + S.zoom.half, cvFull), b = toPix(g, S.zoom.cx + S.zoom.half, S.zoom.cy - S.zoom.half, cvFull);
  ctx.strokeStyle = '#E8919E'; ctx.lineWidth = 1.2; ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
}

// ---------------------------------------------------------------- wave-like dark matter granules
let NOISE = null, GRAN = null;
function granules() {
  const n = S.zoom.n, box = 2 * S.zoom.half;
  if (!NOISE || NOISE.seed !== S.wave.seed) {
    const rnd = seededRandom(S.wave.seed), re = new Float64Array(n * n), im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) re[i] = gaussian(rnd);
    fft2(re, im, n, false);
    NOISE = { re, im, seed: S.wave.seed };
  }
  const lam = S.wave.lamKpc / S.kpc;
  const kr = new Float64Array(n * n), ki = new Float64Array(n * n), kxs = new Float64Array(n);
  for (let i = 0; i < n; i++) kxs[i] = 2 * Math.PI * (i < n / 2 ? i : i - n) / box;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k2 = kxs[i] * kxs[i] + kxs[j] * kxs[j], amp = Math.exp(-k2 * Math.pow(lam / (2 * Math.PI), 2));   // sqrt of exp(-2 k^2 (lam/2pi)^2)
    const q = j * n + i; kr[q] = NOISE.re[q] * amp; ki[q] = NOISE.im[q] * amp;
  }
  kr[0] = ki[0] = 0;
  const kapR = kr.slice(), kapI = ki.slice();
  fft2(kapR, kapI, n, true);
  let s2 = 0; for (let i = 0; i < n * n; i++) s2 += kapR[i] * kapR[i];
  const norm = S.wave.amp / Math.sqrt(s2 / (n * n));
  const axR = new Float64Array(n * n), axI = new Float64Array(n * n), ayR = new Float64Array(n * n), ayI = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const q = j * n + i, kx = kxs[i], ky = kxs[j], k2 = kx * kx + ky * ky || 1;
    const pr = -2 * kr[q] * norm / k2, pi = -2 * ki[q] * norm / k2;     // psi_k
    axR[q] = -kx * pi; axI[q] = kx * pr; ayR[q] = -ky * pi; ayI[q] = ky * pr;   // alpha_k = i k psi_k
  }
  fft2(axR, axI, n, true); fft2(ayR, ayI, n, true);
  const kap = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) kap[i] = kapR[i] * norm;
  // grid rows run from +y (top) to -y; the FFT field is periodic, so the orientation is arbitrary but consistent
  GRAN = { kap, ax: axR, ay: ayR };
}

// ---------------------------------------------------------------- zoom, residual, power spectrum
function absorb(ax, ay, X, Y, w) {
  // brightness-weighted linear fit of the perturbing deflection, removed: a smooth macro model absorbs it
  const n = ax.length; let sw = 0, mx = 0, my = 0;
  for (let i = 0; i < n; i++) { sw += w[i]; mx += w[i] * X[i]; my += w[i] * Y[i]; }
  mx /= sw; my /= sw;
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bx = [0, 0, 0], by = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const v = [1, X[i] - mx, Y[i] - my];
    for (let a = 0; a < 3; a++) { bx[a] += w[i] * v[a] * ax[i]; by[a] += w[i] * v[a] * ay[i]; for (let b = 0; b < 3; b++) M[a][b] += w[i] * v[a] * v[b]; }
  }
  const cx = solve3(M, bx), cy = solve3(M, by);
  for (let i = 0; i < n; i++) {
    const dx = X[i] - mx, dy = Y[i] - my;
    ax[i] -= cx[0] + cx[1] * dx + cx[2] * dy; ay[i] -= cy[0] + cy[1] * dx + cy[2] * dy;
  }
}
function solve3(A, b) {
  const d = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = d(A) || 1e-30;
  return [0, 1, 2].map(c => d(A.map((row, r) => row.map((v, k) => k === c ? b[r] : v))) / D);
}

function radialPower(res, n, box) {
  const re = Float64Array.from(res), im = new Float64Array(n * n);
  let mean = 0; for (const v of re) mean += v; mean /= n * n; for (let i = 0; i < n * n; i++) re[i] -= mean;
  fft2(re, im, n, false);
  const kmin = 2 * Math.PI / box, kmax = Math.PI * n / box, nb = 22;
  const edges = Array.from({ length: nb + 1 }, (_, i) => kmin * Math.pow(kmax / kmin, i / nb));
  const sum = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const kx = 2 * Math.PI * (i < n / 2 ? i : i - n) / box, ky = 2 * Math.PI * (j < n / 2 ? j : j - n) / box, k = Math.hypot(kx, ky);
    if (k < kmin || k >= kmax) continue;
    const b = Math.floor(Math.log(k / kmin) / Math.log(kmax / kmin) * nb);
    sum[b] += (re[j * n + i] ** 2 + im[j * n + i] ** 2) / Math.pow(n, 4) * box * box; cnt[b]++;
  }
  const x = [], y = [];
  for (let b = 0; b < nb; b++) if (cnt[b]) { x.push(Math.sqrt(edges[b] * edges[b + 1]) / S.kpc); y.push(sum[b] / cnt[b]); }
  return { x, y };
}

let PK_GHOST = null;
function drawZoom() {
  if (!ZOOM || !GAL) return;
  const g = ZOOM, n = g.n, N = n * n;
  const smooth = new Float32Array(N * 3), pert = new Float32Array(N * 3);
  for (let k = 0; k < N; k++) sampleGal(g.bx[k], g.by[k], smooth, 3 * k);
  const w = new Float32Array(N); let smax = 0;
  for (let k = 0; k < N; k++) { w[k] = smooth[3 * k] + smooth[3 * k + 1] + smooth[3 * k + 2] + 1e-6; smax = Math.max(smax, w[k]); }
  const ax = new Float64Array(N), ay = new Float64Array(N);
  if (S.mode === 'subhalo') { for (let k = 0; k < N; k++) { const a = subDefl(g.X[k], g.Y[k]); ax[k] = a[0]; ay[k] = a[1]; } }
  else { if (!GRAN) granules(); for (let k = 0; k < N; k++) { ax[k] = GRAN.ax[k]; ay[k] = -GRAN.ay[k]; } }   // grid rows run toward -y
  absorb(ax, ay, g.X, g.Y, w);
  for (let k = 0; k < N; k++) sampleGal(g.bx[k] - ax[k], g.by[k] - ay[k], pert, 3 * k);
  const res = new Float32Array(N);
  for (let k = 0; k < N; k++) res[k] = (pert[3 * k] + pert[3 * k + 1] + pert[3 * k + 2] - w[k]) / smax;

  cvZoom.width = cvZoom.height = n; cvRes.width = cvRes.height = n;
  const cz = cvZoom.getContext('2d'), cr = cvRes.getContext('2d');
  putRGB(cz, n, pert, 0.8);
  const vmax = S.mode === 'subhalo' ? 0.25 : 0.1;
  const img = cr.createImageData(n, n);
  for (let k = 0; k < N; k++) { const c = lerpStops(DIVERGING, 0.5 + 0.5 * res[k] / vmax); img.data[4 * k] = c[0]; img.data[4 * k + 1] = c[1]; img.data[4 * k + 2] = c[2]; img.data[4 * k + 3] = 255; }
  cr.putImageData(img, 0, 0);
  if (S.mode === 'subhalo') {
    const p = toPix(g, S.sub.x, S.sub.y, cvZoom), rad = Math.max(10, Math.sqrt(Math.pow(10, S.sub.logM) * S.te2) * 1.5 / (2 * g.half) * n);
    for (const [c, col] of [[cz, 'rgba(232,145,158,.9)'], [cr, 'rgba(244,242,236,.6)']]) {
      c.strokeStyle = col; c.setLineDash([3, 3]); c.lineWidth = 1.2; c.beginPath(); c.arc(p[0], p[1], rad, 0, 7); c.stroke(); c.setLineDash([]);
    }
  } else {
    cvKap.width = cvKap.height = n;
    drawField(cvKap.getContext('2d'), Array.from({ length: N }, (_, k) => GRAN.kap[(n - 1 - Math.floor(k / n)) * n + k % n]), n, 0.3, PUOR);
  }
  const pk = radialPower(res, n, 2 * g.half);
  const series = [];
  if (PK_GHOST) series.push({ x: PK_GHOST.x, y: PK_GHOST.y, color: 'rgba(232,145,158,.35)', width: 1.2, label: 'previous' });
  series.push({ x: pk.x, y: pk.y, color: ACCENT, width: 2, label: S.mode === 'subhalo' ? 'subhalo' : 'wave-like DM' });
  plot(cvPk, { series, xlog: true, ylog: true, xlim: [0.35, 45], ylim: [1e-10, 1e-3], xlabel: 'k [kpc⁻¹]', ylabel: 'P(k)' });
  drawZoom.last = pk;
  readout();
}

function readout() {
  const el = document.getElementById('readout');
  if (S.mode === 'subhalo') {
    const te = Math.sqrt(Math.pow(10, S.sub.logM) * S.te2);
    el.innerHTML = `<span>subhalo at <b>(${S.sub.x.toFixed(2)}″, ${S.sub.y.toFixed(2)}″)</b></span>` +
      `<span>point-mass Einstein radius <b>${(te * 1000).toFixed(0)} mas</b> (${(te * S.kpc * 1000).toFixed(0)} pc)</span>` +
      `<span>tidal radius <b>${(S.sub.rt * S.kpc).toFixed(1)} kpc</b></span>`;
  } else {
    el.innerHTML = `<span>granule size λ<sub>dB</sub> = <b>${S.wave.lamKpc.toFixed(2)} kpc</b> (${(S.wave.lamKpc / S.kpc).toFixed(3)}″)</span>` +
      `<span>rms convergence <b>${S.wave.amp.toFixed(3)}</b></span>`;
  }
}

// ---------------------------------------------------------------- interaction
const redrawAll = throttleFrame(() => { drawSource(); drawFull(); drawZoom(); });
const redrawZoom = throttleFrame(drawZoom);

onDrag(cvSrc, (u, v) => {
  const h = S.srcView.half;
  S.src.x = S.srcView.cx - h + 2 * h * u; S.src.y = S.srcView.cy + h - 2 * h * v;
  redrawAll();
});
onDrag(cvFull, (u, v, type) => {
  if (type !== 'pointerdown' && type !== 'pointermove') return;
  const h = S.full.half;
  const nx = -h + 2 * h * u, ny = h - 2 * h * v;
  S.sub.x += nx - S.zoom.cx; S.sub.y += ny - S.zoom.cy;   // the subhalo travels with the zoom box
  S.zoom.cx = nx; S.zoom.cy = ny;
  ZOOM = makeGrid(S.zoom.cx, S.zoom.cy, S.zoom.half, S.zoom.n);
  drawFull(); redrawZoom();
});
onDrag(cvZoom, (u, v) => {
  if (S.mode !== 'subhalo') return;
  const h = S.zoom.half;
  S.sub.x = S.zoom.cx - h + 2 * h * u; S.sub.y = S.zoom.cy + h - 2 * h * v;
  redrawZoom();
});

function bindRange(id, fmt, set) {
  const el = document.getElementById(id), out = document.querySelector(`output[for="${id}"]`);
  const f = () => { set(+el.value); out.textContent = fmt(+el.value); };
  el.addEventListener('input', () => { f(); });
  f();
  return el;
}
bindRange('mass', v => supMass(Math.pow(10, v)), v => { S.sub.logM = v; redrawZoom(); });
bindRange('shape', v => v.toFixed(2) + ' kpc', v => { S.sub.shapeKpc = v; redrawZoom(); });
bindRange('lam', v => v.toFixed(2) + ' kpc', v => { S.wave.lamKpc = v; GRAN = null; redrawZoom(); });
bindRange('amp', v => v.toFixed(3), v => { S.wave.amp = v; GRAN = null; redrawZoom(); });

const shapeLabel = document.getElementById('shape-label'), shapeCtl = document.getElementById('shape-ctl');
document.getElementById('profile').addEventListener('change', e => {
  S.sub.kind = e.target.value;
  shapeCtl.hidden = S.sub.kind === 'point';
  shapeLabel.textContent = S.sub.kind === 'cored' ? 'core radius' : 'NFW scale radius r_s';
  redrawZoom();
});
document.getElementById('reseed').addEventListener('click', () => { S.wave.seed++; GRAN = null; redrawZoom(); });
document.getElementById('keep').addEventListener('click', () => { PK_GHOST = drawZoom.last; redrawZoom(); });

for (const b of document.querySelectorAll('[data-mode]')) b.addEventListener('click', () => {
  S.mode = b.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(x => x.setAttribute('aria-pressed', x === b));
  document.querySelectorAll('.only-subhalo').forEach(x => x.hidden = S.mode !== 'subhalo');
  document.querySelectorAll('.only-wave').forEach(x => x.hidden = S.mode !== 'wave');
  PK_GHOST = null;
  redrawZoom();
});

document.getElementById('upload').addEventListener('change', e => {
  const f = e.target.files[0]; if (!f) return;
  const im = new Image(); im.onload = () => { setGalaxy(im, true); redrawAll(); }; im.src = URL.createObjectURL(f);
});

// ---------------------------------------------------------------- start
Promise.all([
  fetch('assets/cluster_curves.json').then(r => r.json()),
  new Promise(res => { const im = new Image(); im.onload = () => res(im); im.src = 'assets/galaxy.png'; }),
]).then(([curves, im]) => {
  S.curves = curves; S.kpc = curves.kpc_per_arcsec; S.te2 = curves.theta_e2_per_msun;
  setGalaxy(im, false);
  FULL = makeGrid(0, 0, S.full.half, S.full.n);
  ZOOM = makeGrid(S.zoom.cx, S.zoom.cy, S.zoom.half, S.zoom.n);
  drawSource(); drawFull(); drawZoom();
});
window.addEventListener('resize', throttleFrame(drawZoom));
