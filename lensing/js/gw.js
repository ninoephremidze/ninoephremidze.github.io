// Interactive wave-optics lensing of gravitational waves by a compact dark-matter lens.
// Same physics as python/lensviz/gw.py.
'use strict';

const T_SUN = 4.925490947e-6;               // G M_sun / c^3 [s]
const P = { logM: Math.log10(300), y: 0.6, playing: true };
let TABLE = null;

// ---------------------------------------------------------------- amplification factor
function images(y) {
  const s = Math.sqrt(y * y + 4);
  return { muP: 0.5 + (y * y + 2) / (2 * y * s), muM: 0.5 - (y * y + 2) / (2 * y * s), dT: y * s / 2 + Math.log((s + y) / (s - y)) };
}
function Fpoint(w, y) {
  // geometric optics (two images, Morse phase -pi/2 on the second) + tabulated wave-optics residual
  const im = images(y), a = Math.sqrt(Math.abs(im.muP)), b = Math.sqrt(Math.abs(im.muM));
  let re = a + b * Math.sin(w * im.dT), imag = -b * Math.cos(w * im.dT);
  const T = TABLE;
  if (T && w < T.w_max - T.dw) {
    const i = w / T.dw, i0 = Math.floor(i), fi = i - i0;
    let j = (y - T.y_min) / T.dy; j = Math.min(Math.max(j, 0), T.n_y - 1.001);
    const j0 = Math.floor(j), fj = j - j0;
    const at = (jj, ii, c) => T.R[2 * (jj * T.n_w + ii) + c];
    for (let c = 0; c < 2; c++) {
      const r = (1 - fj) * ((1 - fi) * at(j0, i0, c) + fi * at(j0, i0 + 1, c)) + fj * ((1 - fi) * at(j0 + 1, i0, c) + fi * at(j0 + 1, i0 + 1, c));
      if (c === 0) re += r; else imag += r;
    }
  }
  return [re, imag];
}
const wOf = (f, M) => 8 * Math.PI * T_SUN * M * f;

// ---------------------------------------------------------------- toy binary black hole waveform (30 + 30 M_sun)
const FS = 4096, N = 16384;
const WAVE = (() => {
  const m1 = 30, m2 = 30, M = m1 + m2, eta = m1 * m2 / (M * M), Mc = Math.pow(eta, 0.6) * M * T_SUN;
  const tMerge = 3.0, fLow = 20, fRd = 250, tauRd = 0.004;
  const fIsco = 1 / (Math.pow(6, 1.5) * Math.PI * M * T_SUN);
  const tIsco = tMerge - 5 / 256 * Math.pow(Mc, -5 / 3) * Math.pow(Math.PI * fIsco, -8 / 3);
  const t = new Float64Array(N), h = new Float64Array(N);
  let phase = 0, started = -1;
  for (let i = 0; i < N; i++) {
    t[i] = i / FS;
    const tau = Math.max(tMerge - t[i], 1e-6);
    const fIns = Math.pow(5 / 256 / tau, 0.375) * Math.pow(Mc, -0.625) / Math.PI;
    if (fIns < fLow) continue;
    if (started < 0) started = i;
    const blend = 0.5 * (1 + Math.tanh((t[i] - tIsco) / 0.004 - 1.5));
    const f = (t[i] < tIsco ? fIns : fIsco) * (1 - blend) + fRd * blend;
    phase += 2 * Math.PI * f / FS;
    let amp = Math.pow(Math.min(f, fIsco * 1.6) / fLow, 2 / 3) * Math.exp(-Math.max(t[i] - tIsco - 0.012, 0) / tauRd);
    amp *= 0.5 * (1 - Math.cos(Math.PI * Math.min((t[i] - t[started]) / 0.1, 1)));
    h[i] = amp * Math.cos(phase);
  }
  let mx = 0; for (const v of h) mx = Math.max(mx, Math.abs(v));
  for (let i = 0; i < N; i++) h[i] /= mx;
  const re = Float64Array.from(h), im = new Float64Array(N);
  fft(re, im, false);
  return { t, h, re, im };
})();

function lensed(M, y) {
  const re = new Float64Array(N), im = new Float64Array(N), fs = [], absF = [];
  for (let k = 0; k <= N / 2; k++) {
    const f = k * FS / N;
    let [Fr, Fi] = k === 0 ? [1, 0] : (f < 2100 ? Fpoint(wOf(f, M), y) : [1, 0]);
    if (k > 0 && f < 2100 && k % 2 === 0) { fs.push(f); absF.push(Math.hypot(Fr, Fi)); }
    Fi = -Fi;                                            // conj(F): LALSuite convention, later image arrives later
    const a = WAVE.re[k], b = WAVE.im[k];
    re[k] = a * Fr - b * Fi; im[k] = a * Fi + b * Fr;
    if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; }
  }
  fft(re, im, true);
  return { h: re, f: fs, absF };
}

// ---------------------------------------------------------------- 2D wave picture (Huygens–Fresnel toy)
const FIELD = { nx: 240, ny: 120, x0: 0, x1: 12, y0: -3, y1: 3, src: [0.6, 0], lensX: 5, det: [11.6, 0], lam: 0.45 };
const strength2d = M => 0.3 * Math.sqrt(wOf(100, M) / 3.7);
const einstein2d = s => { const dl = FIELD.lensX - FIELD.src[0], dls = FIELD.det[0] - FIELD.lensX; return Math.sqrt(s * dl * dls / (dl + dls)); };

function huygens(M, y) {
  const { nx, ny, x0, x1, y0, y1, src, lensX, lam } = FIELD, k = 2 * Math.PI / lam;
  const s = strength2d(M), ly = -y * einstein2d(s), core = 0.03, half = 7, nl = 420;
  const Ure = new Float32Array(nx * ny), Uim = new Float32Array(nx * ny);
  const lr = new Float64Array(nl), li = new Float64Array(nl), ll = new Float64Array(nl), dl = 2 * half / (nl - 1);
  for (let j = 0; j < nl; j++) {
    const l = ly - half + j * dl, r1 = Math.hypot(lensX - src[0], l - src[1]);
    const psi = s * Math.log(Math.sqrt((l - ly) ** 2 + core * core));
    const win = Math.sqrt(Math.cos(Math.PI / 2 * Math.min(Math.abs(l - ly) / half, 1)));
    const ph = k * (r1 - psi), a = win / Math.sqrt(r1);
    lr[j] = a * Math.cos(ph); li[j] = a * Math.sin(ph); ll[j] = l;
  }
  const norm = dl * Math.sqrt(k / (2 * Math.PI)), c45 = Math.SQRT1_2;
  for (let iy = 0; iy < ny; iy++) {
    const yy = y1 - (y1 - y0) * (iy + 0.5) / ny;
    for (let ix = 0; ix < nx; ix++) {
      const xx = x0 + (x1 - x0) * (ix + 0.5) / nx, q = iy * nx + ix;
      const rs = Math.hypot(xx - src[0], yy - src[1]) + 1e-3, disp = Math.sqrt(rs + 0.3);
      if (xx < lensX) { Ure[q] = Math.cos(k * rs) / Math.sqrt(rs) * disp; Uim[q] = Math.sin(k * rs) / Math.sqrt(rs) * disp; continue; }
      let sr = 0, si = 0;
      for (let j = 0; j < nl; j++) {
        const r2 = Math.hypot(xx - lensX, yy - ll[j]) + 1e-3, a = (xx - lensX) / (r2 * Math.sqrt(r2));
        const c = Math.cos(k * r2) * a, sn = Math.sin(k * r2) * a;
        sr += lr[j] * c - li[j] * sn; si += lr[j] * sn + li[j] * c;
      }
      // multiply by e^{-i pi/4}
      Ure[q] = (sr + si) * c45 * norm * disp; Uim[q] = (si - sr) * c45 * norm * disp;
    }
  }
  return { Ure, Uim, ly, s };
}

// ---------------------------------------------------------------- drawing
const cvField = document.getElementById('cv-field'), cvWave = document.getElementById('cv-wave'), cvF = document.getElementById('cv-F');
const off = document.createElement('canvas'); off.width = FIELD.nx; off.height = FIELD.ny;
let U = null, L = null, phase = 0, tAnim = 0;

function drawField() {
  if (!U) return;
  const { nx, ny } = FIELD, g = off.getContext('2d'), img = g.createImageData(nx, ny);
  const c = Math.cos(phase), s = Math.sin(phase);
  for (let q = 0; q < nx * ny; q++) {
    const v = U.Ure[q] * c + U.Uim[q] * s;                // Re[U e^{-i phase}]
    const col = lerpStops(DIVERGING, 0.5 + 0.5 * Math.max(-1, Math.min(1, v / 2.2)));
    img.data[4 * q] = col[0]; img.data[4 * q + 1] = col[1]; img.data[4 * q + 2] = col[2]; img.data[4 * q + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const { ctx, w, h } = setupCanvas(cvField);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, w, h);
  const X = x => (x - FIELD.x0) / (FIELD.x1 - FIELD.x0) * w, Y = y => (FIELD.y1 - y) / (FIELD.y1 - FIELD.y0) * h, sc = w / (FIELD.x1 - FIELD.x0);
  // lens glow
  for (const [r, a] of [[0.55, 0.10], [0.35, 0.18], [0.18, 0.35]]) {
    ctx.fillStyle = `rgba(232,145,158,${a})`; ctx.beginPath(); ctx.arc(X(FIELD.lensX), Y(U.ly), r * (0.6 + 0.6 * U.s) * sc, 0, 7); ctx.fill();
  }
  ctx.strokeStyle = 'rgba(244,242,236,.3)'; ctx.setLineDash([2, 5]); ctx.beginPath(); ctx.moveTo(X(FIELD.lensX), 0); ctx.lineTo(X(FIELD.lensX), h); ctx.stroke(); ctx.setLineDash([]);
  // binary black hole
  const a = 0.14, om = tAnim * 3;
  ctx.fillStyle = INK;
  for (const sg of [1, -1]) { ctx.beginPath(); ctx.arc(X(FIELD.src[0] + sg * a * Math.cos(om)), Y(FIELD.src[1] + sg * a * Math.sin(om)), 0.07 * sc, 0, 7); ctx.fill(); }
  // detector
  ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.beginPath();
  ctx.moveTo(X(FIELD.det[0]), Y(FIELD.det[1] + 0.35)); ctx.lineTo(X(FIELD.det[0]), Y(FIELD.det[1])); ctx.lineTo(X(FIELD.det[0] - 0.35), Y(FIELD.det[1])); ctx.stroke();
  ctx.font = '12px Newsreader, Georgia, serif'; ctx.fillStyle = INK;
  ctx.fillText('binary black hole', X(FIELD.src[0]), h - 10);
  ctx.textAlign = 'right'; ctx.fillText('detector', X(FIELD.det[0]), h - 10); ctx.textAlign = 'left';
  ctx.fillStyle = ACCENT; ctx.fillText('dark matter subhalo', X(FIELD.lensX) + 8, 18);
}

function drawPlots() {
  const M = Math.pow(10, P.logM), t = WAVE.t, i0 = Math.round(2.62 * FS), i1 = Math.round(3.06 * FS);
  const tt = [], h0 = [], hl = [];
  for (let i = i0; i < i1; i++) { tt.push((t[i] - 3) * 1e3); h0.push(WAVE.h[i]); hl.push(L.h[i]); }
  plot(cvWave, {
    series: [{ x: tt, y: h0, color: FAINT, width: 1, label: 'unlensed' }, { x: tt, y: hl, color: ACCENT, width: 1.5, label: 'lensed' }],
    xlim: [tt[0], tt[tt.length - 1]], ylim: [-2.4, 2.4], xlabel: 'time to merger [ms]', ylabel: 'strain (arb.)',
  });
  const f1 = 1 / (8 * Math.PI * T_SUN * M);
  plot(cvF, {
    series: [{ x: L.f, y: L.absF, color: CYAN, width: 1.4 }], xlog: true, xlim: [5, 2000], ylim: [0, 3.2],
    xlabel: 'frequency [Hz]', ylabel: '|F(f)|', bands: [{ x0: 20, x1: 1000, label: 'LIGO band' }],
    vlines: f1 > 5 && f1 < 2000 ? [{ x: f1, color: ACCENT, label: 'w = 1' }] : [],
  });
  const im = images(P.y), dt = 4 * T_SUN * M * im.dT * 1e3;
  document.getElementById('readout').innerHTML =
    `<span>time delay <b>${dt < 1 ? dt.toFixed(3) : dt.toFixed(1)} ms</b></span>` +
    `<span>magnifications <b>${im.muP.toFixed(2)}</b>, <b>${Math.abs(im.muM).toFixed(2)}</b></span>` +
    `<span>w at 100 Hz <b>${wOf(100, M).toFixed(2)}</b></span>` +
    `<span>regime <b>${wOf(100, M) < 0.3 ? 'diffraction: the lens is nearly invisible' : wOf(100, M) < 20 ? 'wave optics: interference' : 'geometric optics: two images'}</b></span>`;
}

const recompute = throttleFrame(() => {
  const M = Math.pow(10, P.logM);
  L = lensed(M, P.y); drawPlots();
  U = huygens(M, P.y); drawField();
});

// ---------------------------------------------------------------- controls & animation
function bind(id, fmt, set) {
  const el = document.getElementById(id), out = document.querySelector(`output[for="${id}"]`);
  const f = () => { set(+el.value); out.textContent = fmt(+el.value); };
  el.addEventListener('input', () => { f(); recompute(); }); f();
}
bind('mass', v => supMass(Math.pow(10, v)), v => { P.logM = v; });
bind('impact', v => v.toFixed(2), v => { P.y = v; });

const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const playBtn = document.getElementById('play');
P.playing = !reduce;
playBtn.textContent = P.playing ? 'Pause waves' : 'Play waves';
playBtn.addEventListener('click', () => { P.playing = !P.playing; playBtn.textContent = P.playing ? 'Pause waves' : 'Play waves'; if (P.playing) requestAnimationFrame(loop); });
let last = 0;
function loop(ts) {
  if (!P.playing) return;
  if (ts - last > 40) { phase += 0.35; tAnim += 0.12; drawField(); last = ts; }
  requestAnimationFrame(loop);
}

Promise.all([fetch('assets/F_point_table.json').then(r => r.json()), fetch('assets/F_point_table.bin').then(r => r.arrayBuffer())])
  .then(([meta, buf]) => { TABLE = Object.assign(meta, { R: new Float32Array(buf) }); recompute(); if (P.playing) requestAnimationFrame(loop); });
window.addEventListener('resize', throttleFrame(() => { drawField(); if (L) drawPlots(); }));
