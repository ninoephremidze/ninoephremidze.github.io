// Shared helpers for the lensing visualizations: FFT, colour maps, simple plots, dragging.
'use strict';

// ---------------------------------------------------------------- FFT (radix-2, in place)
function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = 2 * Math.PI / len * (inverse ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// 2D FFT on row-major n x n arrays
function fft2(re, im, n, inverse) {
  const r = new Float64Array(n), i_ = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { r[x] = re[y * n + x]; i_[x] = im[y * n + x]; }
    fft(r, i_, inverse);
    for (let x = 0; x < n; x++) { re[y * n + x] = r[x]; im[y * n + x] = i_[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; i_[y] = im[y * n + x]; }
    fft(r, i_, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = i_[y]; }
  }
}

// ---------------------------------------------------------------- colours
function lerpStops(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const s = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(s)), f = s - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
// dark-centred diverging map: deficit blue, excess garnet (same as the GIFs)
const DIVERGING = ['#9fd0ff', '#2b6cb0', '#0E0F15', '#b5405a', '#ffc2cc'].map(hex);
const PUOR = ['#e8a33a', '#7a4a10', '#0E0F15', '#4b3b8f', '#b8a8ff'].map(hex);

function drawField(ctx, data, n, vmax, stops) {
  const img = ctx.createImageData(n, n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const c = lerpStops(stops, 0.5 + 0.5 * data[(n - 1 - y) * n + x] / vmax), k = 4 * (y * n + x);
    img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------- line plots
const INK = '#F4F2EC', MUTED = '#BDB9B0', FAINT = '#8E8B84', ACCENT = '#E8919E', CYAN = '#7FD3E6', LINE = 'rgba(244,242,236,.16)';

function setupCanvas(cv) {
  // render at device resolution, draw in CSS pixels
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight || w * cv.height / cv.width;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function plot(cv, opts) {
  // opts: {series:[{x,y,color,width,dash}], xlog, ylog, xlim, ylim, xlabel, ylabel, vlines:[{x,color,label}], bands:[{x0,x1,label}]}
  const { ctx, w, h } = setupCanvas(cv);
  const m = { l: 46, r: 10, t: 10, b: 34 };
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#05060A'; ctx.fillRect(0, 0, w, h);
  const tx = v => opts.xlog ? Math.log10(v) : v, ty = v => opts.ylog ? Math.log10(Math.max(v, 1e-300)) : v;
  const [x0, x1] = opts.xlim.map(tx), [y0, y1] = opts.ylim.map(ty);
  const X = v => m.l + (tx(v) - x0) / (x1 - x0) * (w - m.l - m.r);
  const Y = v => h - m.b - (ty(v) - y0) / (y1 - y0) * (h - m.t - m.b);
  ctx.font = '11px "JetBrains Mono", monospace';
  for (const b of opts.bands || []) {
    ctx.fillStyle = 'rgba(244,242,236,.05)'; ctx.fillRect(X(b.x0), m.t, X(b.x1) - X(b.x0), h - m.t - m.b);
    ctx.fillStyle = FAINT; ctx.fillText(b.label, X(b.x0) + 4, m.t + 12);
  }
  // axes and ticks
  ctx.strokeStyle = LINE; ctx.lineWidth = 1; ctx.strokeRect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
  ctx.fillStyle = FAINT;
  const ticks = (lo, hi, log) => {
    const out = [];
    if (log) { for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) out.push([Math.pow(10, e), e === 0 ? '1' : (Math.abs(e) < 3 ? String(Math.pow(10, e)) : '1e' + e)]); }
    else { const step = niceStep((hi - lo) / 4); for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push([v, +v.toFixed(6) + '']); }
    return out;
  };
  for (const [v, s] of ticks(x0, x1, opts.xlog)) { const x = X(v); ctx.fillText(s, x - ctx.measureText(s).width / 2, h - m.b + 15); }
  for (const [v, s] of ticks(y0, y1, opts.ylog)) { const y = Y(v); ctx.fillText(s, m.l - 6 - ctx.measureText(s).width, y + 4); }
  ctx.fillStyle = MUTED;
  if (opts.xlabel) ctx.fillText(opts.xlabel, (w + m.l) / 2 - ctx.measureText(opts.xlabel).width / 2, h - 4);
  if (opts.ylabel) { ctx.save(); ctx.translate(11, (h - m.b + m.t) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(opts.ylabel, -ctx.measureText(opts.ylabel).width / 2, 0); ctx.restore(); }
  // data
  ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b); ctx.clip();
  for (const v of opts.vlines || []) {
    ctx.strokeStyle = v.color; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(v.x), m.t); ctx.lineTo(X(v.x), h - m.b); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = v.color; ctx.fillText(v.label, X(v.x) + 4, h - m.b - 6);
  }
  for (const s of opts.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.setLineDash(s.dash || []); ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const xv = s.x[i], yv = s.y[i];
      if (!isFinite(yv) || (opts.xlog && xv <= 0)) { pen = false; continue; }
      const px = X(xv), py = Y(yv);
      pen ? ctx.lineTo(px, py) : ctx.moveTo(px, py); pen = true;
    }
    ctx.stroke();
  }
  ctx.restore(); ctx.setLineDash([]);
  // legend
  let ly = m.t + 14;
  for (const s of opts.series) if (s.label) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
    ctx.beginPath(); ctx.moveTo(w - m.r - 120, ly - 4); ctx.lineTo(w - m.r - 100, ly - 4); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = MUTED; ctx.fillText(s.label, w - m.r - 94, ly); ly += 15;
  }
}
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }

// ---------------------------------------------------------------- pointer dragging
function onDrag(cv, fn) {
  // fn(u, v) with u, v in [0,1] across the canvas (v = 0 at the top)
  let down = false;
  const at = e => { const r = cv.getBoundingClientRect(); fn((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, e.type); };
  cv.addEventListener('pointerdown', e => { down = true; cv.setPointerCapture(e.pointerId); at(e); });
  cv.addEventListener('pointermove', e => { if (down) at(e); });
  cv.addEventListener('pointerup', () => { down = false; });
}

// run fn at most once per animation frame
function throttleFrame(fn) {
  let queued = false;
  return () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; fn(); }); } };
}

function seededRandom(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function gaussian(rand) { return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand()); }

function supMass(m) {
  const e = Math.floor(Math.log10(m)), a = m / Math.pow(10, e);
  const sup = String(e).replace(/[-0-9]/g, c => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]);
  return (Math.abs(a - 1) < 0.05 ? '' : a.toFixed(1) + '×') + '10' + sup + ' M☉';
}
