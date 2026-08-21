// latent-space concept page -- why generating images is affordable at all, and
// the hard ceiling that affordability buys.
//
// THE MECHANISM. Running a diffusion loop directly on pixels is prohibitive:
// every one of the tens of sampling steps has to touch every pixel, and the
// attention inside a transformer denoiser costs the square of the token count.
// Latent diffusion (Rombach et al., "High-Resolution Image Synthesis with
// Latent Diffusion Models", https://arxiv.org/abs/2112.10752) pays the pixel
// cost exactly twice -- encode once at the start, decode once at the end -- and
// runs the entire loop inside a compressed space in between. A typical image
// autoencoder downsamples by 8 in each spatial direction while widening the
// channel axis, so the element count collapses. This page computes that
// collapse from your own settings rather than quoting a number at you.
//
// THE POINT OF THE PAGE IS THE CEILING, NOT THE SAVING. Whatever the encoder
// cannot represent is gone BEFORE the diffusion loop starts, and no amount of
// sampler quality brings it back. So this page encodes and decodes with NO
// diffusion at all and shows the residual: the reconstruction floor. Set the
// sampling error to zero and the residual does not go to zero. That is the
// whole lesson, and it is set by a component that rarely gets discussed.
//
// AND IT IS CONTESTED, so the page says so rather than presenting the 8x /
// 4-channel recipe as settled. See the "contested" card and the README.
//
// THE TOY ENCODER, STATED PLAINLY: this page has NO trained VAE and does not
// pretend to. Its encoder is a block DCT-II -- an orthonormal strided linear
// transform over f x f blocks -- followed by keeping only the C lowest-frequency
// coefficients per block in zig-zag order. The decoder is the inverse transform
// with the discarded coefficients set to zero. That is a real encoder/decoder
// pair with a real, computable reconstruction floor, and it reproduces the
// classic casualties (fine texture, small text) for the same reason a trained
// autoencoder does: the discarded basis functions are where that detail lived.
// A trained VAE differs -- it learns its basis, it is not orthonormal, and its
// adversarial + perceptual losses trade measured error for perceived quality --
// so treat the shape of the trade here as the lesson, never the exact numbers.
//
// NOT THIS PAGE: the noising/denoising loop itself is diffusion-noise; step
// counts and samplers are diffusion-sampler; conditioning strength is guidance.
// This page only shows the arena that loop runs inside, and the cost of the
// door into it.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';
import { ramps, categorical, cellAt } from '../framework/render.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

const N = 64;                 // source image side, in pixels
const PIX = N * N;
const MONO = 'ui-monospace, monospace';
const FS = [2, 4, 8, 16];     // downsample factors that divide N
const REF_F = 8, REF_C = 4;   // the standard recipe, used as the NAMED baseline

// ------------------------------------------------------------ a 3x5 pixel font
// Small text is the classic casualty of a compressed latent, so the source image
// has to contain some. Drawn from a bitmap rather than the system font on
// purpose: it is deterministic, it needs no colour literal, and a 3x5 glyph is
// small enough to sit entirely inside one 8x8 block -- which is exactly the
// situation where a low-channel latent has nowhere to put it.
const GLYPH = {
  ' ': [0, 0, 0, 0, 0],
  A: [0b010, 0b101, 0b111, 0b101, 0b101], C: [0b011, 0b100, 0b100, 0b100, 0b011],
  D: [0b110, 0b101, 0b101, 0b101, 0b110], E: [0b111, 0b100, 0b110, 0b100, 0b111],
  F: [0b111, 0b100, 0b110, 0b100, 0b100], G: [0b011, 0b100, 0b101, 0b101, 0b011],
  H: [0b101, 0b101, 0b111, 0b101, 0b101], I: [0b111, 0b010, 0b010, 0b010, 0b111],
  L: [0b100, 0b100, 0b100, 0b100, 0b111], M: [0b101, 0b111, 0b111, 0b101, 0b101],
  N: [0b101, 0b111, 0b111, 0b111, 0b101], O: [0b010, 0b101, 0b101, 0b101, 0b010],
  P: [0b110, 0b101, 0b110, 0b100, 0b100], R: [0b110, 0b101, 0b110, 0b101, 0b101],
  S: [0b011, 0b100, 0b010, 0b001, 0b110], T: [0b111, 0b010, 0b010, 0b010, 0b010],
  U: [0b101, 0b101, 0b101, 0b101, 0b111], X: [0b101, 0b101, 0b010, 0b101, 0b101],
  Y: [0b101, 0b101, 0b010, 0b010, 0b010], Z: [0b111, 0b001, 0b010, 0b100, 0b111],
  4: [0b101, 0b101, 0b111, 0b001, 0b001], 8: [0b111, 0b101, 0b111, 0b101, 0b111],
};
function stampText(buf, str, x0, y0, v) {
  let x = x0;
  for (const ch of String(str).toUpperCase()) {
    const g = GLYPH[ch] || GLYPH[' '];
    for (let r = 0; r < 5; r++) for (let c = 0; c < 3; c++) {
      if ((g[r] >> (2 - c)) & 1) { const px = x + c, py = y0 + r; if (px >= 0 && px < N && py >= 0 && py < N) buf[py * N + px] = v; }
    }
    x += 4;
  }
}

// ------------------------------------------------------------- source scenes
// Four scenes chosen to separate what survives from what does not: smooth
// low-frequency structure survives almost any latent; 1-pixel gratings and 3x5
// glyphs are the first things to die.
const SCENES = ['mixed', 'text', 'grating', 'portrait'];
function buildSource(scene) {
  const im = new Float32Array(PIX);
  if (scene === 'grating') {
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const k = 0.02 + 0.46 * (c / (N - 1));            // spatial frequency sweep
      im[r * N + c] = 0.5 + 0.5 * Math.cos(2 * Math.PI * k * c) * (0.35 + 0.65 * (r / (N - 1)));
    }
    return im;
  }
  if (scene === 'portrait') {
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const x = (c + 0.5) / N - 0.5, y = (r + 0.5) / N - 0.5;
      let v = 0.82 * Math.exp(-((x * x) / 0.06 + (y * y) / 0.09));
      v += 0.35 * Math.exp(-(((x + 0.11) ** 2 + (y + 0.06) ** 2) / 0.004));
      v += 0.35 * Math.exp(-(((x - 0.11) ** 2 + (y + 0.06) ** 2) / 0.004));
      im[r * N + c] = Math.max(0, Math.min(1, 0.08 + v));
    }
    return im;
  }
  if (scene === 'text') {
    im.fill(0.08);
    const lines = ['TINY TEXT', 'DIES FIRST', 'IN A SMALL', 'LATENT', 'READ ME', 'AT 8X 4CH'];
    lines.forEach((s, i) => stampText(im, s, 2, 2 + i * 10, 0.95));
    return im;
  }
  // mixed: four quadrants, one per failure mode.
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    let v;
    if (c < 32 && r < 32) {                                  // smooth radial ramp
      const x = (c - 15.5) / 20, y = (r - 15.5) / 20;
      v = Math.max(0, 1 - Math.hypot(x, y));
    } else if (r < 32) {                                     // fine grating
      v = r < 16 ? ((c % 2) ? 0.95 : 0.05) : ((((c / 2) | 0) % 2) ? 0.9 : 0.1);
    } else if (c < 32) {                                     // small text
      v = 0.08;
    } else {                                                 // hard edges
      const inSq = c >= 36 && c < 50 && r >= 36 && r < 50;
      const inHole = c >= 41 && c < 45 && r >= 41 && r < 45;
      const inDisc = Math.hypot(c - 56, r - 54) < 6.5;
      v = (inSq && !inHole) || inDisc ? 0.95 : 0.06;
    }
    im[r * N + c] = v;
  }
  stampText(im, 'LATENT', 2, 35, 0.95);
  stampText(im, '8X 4CH', 2, 44, 0.95);
  stampText(im, 'FLOOR', 2, 53, 0.95);
  return im;
}

// ------------------------------------------------ the toy encoder / decoder
// Orthonormal DCT-II basis for one axis, plus the zig-zag order that says which
// coefficient is "channel 0", "channel 1", ... Orthonormal matters: it makes the
// transform energy-preserving, so the energy of the discarded coefficients IS
// the squared reconstruction error. The floor is not a metaphor here, it is a
// sum you can point at.
const _basis = {};
function basis(f) {
  if (_basis[f]) return _basis[f];
  const D = [];
  for (let k = 0; k < f; k++) {
    const row = new Float64Array(f), a = k === 0 ? Math.sqrt(1 / f) : Math.sqrt(2 / f);
    for (let n = 0; n < f; n++) row[n] = a * Math.cos(Math.PI * (2 * n + 1) * k / (2 * f));
    D.push(row);
  }
  const zz = [];
  for (let s = 0; s < 2 * f - 1; s++) {
    const lo = Math.max(0, s - f + 1), hi = Math.min(s, f - 1);
    if (s % 2 === 0) { for (let u = hi; u >= lo; u--) zz.push([u, s - u]); }
    else { for (let u = lo; u <= hi; u++) zz.push([u, s - u]); }
  }
  const rank = new Int32Array(f * f);                 // (u,v) -> its channel index
  zz.forEach(([u, v], i) => { rank[u * f + v] = i; });
  return (_basis[f] = { D, zz, rank });
}

// src -> per-block coefficients, laid out [block][u*f+v].
function forward(src, f) {
  const { D } = basis(f), nb = N / f, out = new Float32Array(PIX), tmp = new Float64Array(f * f);
  for (let br = 0; br < nb; br++) for (let bc = 0; bc < nb; bc++) {
    for (let u = 0; u < f; u++) { const Du = D[u];
      for (let n = 0; n < f; n++) { let s = 0; for (let m = 0; m < f; m++) s += Du[m] * src[(br * f + m) * N + bc * f + n]; tmp[u * f + n] = s; } }
    const base = (br * nb + bc) * f * f;
    for (let u = 0; u < f; u++) for (let v = 0; v < f; v++) { const Dv = D[v];
      let s = 0; for (let n = 0; n < f; n++) s += tmp[u * f + n] * Dv[n]; out[base + u * f + v] = s; }
  }
  return out;
}
function inverse(coef, f) {
  const { D } = basis(f), nb = N / f, out = new Float32Array(PIX), tmp = new Float64Array(f * f);
  for (let br = 0; br < nb; br++) for (let bc = 0; bc < nb; bc++) {
    const base = (br * nb + bc) * f * f;
    for (let m = 0; m < f; m++) for (let v = 0; v < f; v++) { let s = 0; for (let u = 0; u < f; u++) s += D[u][m] * coef[base + u * f + v]; tmp[m * f + v] = s; }
    for (let m = 0; m < f; m++) for (let n = 0; n < f; n++) { let s = 0; for (let v = 0; v < f; v++) s += tmp[m * f + v] * D[v][n]; out[(br * f + m) * N + bc * f + n] = s; }
  }
  return out;
}

// Keep the first C channels of every block; everything else is DISCARDED, and
// discarded is permanent -- nothing downstream can restore it.
// `noise` perturbs the KEPT channels only, standing in for the sampling loop
// landing near, but not exactly on, the right latent.
function encodeDecode(src, f, C, noise, seed) {
  const { zz } = basis(f), nb = N / f, nB = nb * nb, F2 = f * f;
  const coef = forward(src, f);
  const kept = new Float32Array(PIX);                 // truncated coefficient field
  const lat = [];                                     // C channels, each nb x nb
  const z = noise > 0 ? seededRandn(seed | 0, nB * C, { std: 1 }) : null;
  for (let k = 0; k < C; k++) {
    const [u, v] = zz[k], ch = new Float32Array(nB);
    for (let b = 0; b < nB; b++) {
      let val = coef[b * F2 + u * f + v];
      if (z) val += noise * z[k * nB + b];
      ch[b] = val; kept[b * F2 + u * f + v] = val;
    }
    lat.push(ch);
  }
  const rec = inverse(kept, f);
  let se = 0, mx = 0;
  const err = new Float32Array(PIX);
  for (let i = 0; i < PIX; i++) { const d = rec[i] - src[i]; err[i] = Math.abs(d); se += d * d; if (err[i] > mx) mx = err[i]; }
  return { coef, kept, lat, rec, err, rmse: Math.sqrt(se / PIX), maxErr: mx, nb, F2 };
}

// Per-basis energy across the whole image -- how much of the picture each
// coefficient carries. The sum over the DISCARDED entries is the floor.
function energy(coef, f, nb) {
  const F2 = f * f, e = new Float32Array(F2), nB = nb * nb;
  let tot = 0;
  for (let b = 0; b < nB; b++) for (let j = 0; j < F2; j++) { const c = coef[b * F2 + j]; e[j] += c * c; tot += c * c; }
  return { e, tot: tot || 1 };
}

const rmseOf = (src, f, C) => encodeDecode(src, f, C, 0, 0).rmse;

// ------------------------------------------------------------------- state
let src = buildSource('mixed');
let srcVersion = 0;                 // bumped on every paint / scene change
let panels = [];                    // hit-test rects, rebuilt each draw
let painting = false, dragTrade = false;
let selCell = null;                 // {panel, r, c} pinned by a click

const cacheRT = new Map(), cacheTrade = new Map(), cacheRef = new Map();
function roundTrip(f, C, noise, seed) {
  const key = `${srcVersion}|${f}|${C}|${noise}|${seed}`;
  let v = cacheRT.get(key);
  if (!v) { v = encodeDecode(src, f, C, noise, seed); if (cacheRT.size > 64) cacheRT.clear(); cacheRT.set(key, v); }
  return v;
}
function refFloor() {
  const key = `${srcVersion}`;
  let v = cacheRef.get(key);
  if (v == null) { v = rmseOf(src, REF_F, REF_C); cacheRef.clear(); cacheRef.set(key, v); }
  return v;
}
// Every (f, C) the reader can reach, as (compression ratio, floor) pairs. This
// is the curve the trade panel plots and the reader drags along.
const C_CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];
function tradeCurve() {
  const key = `${srcVersion}`;
  let v = cacheTrade.get(key);
  if (v) return v;
  v = [];
  for (const f of FS) for (const C of C_CANDIDATES) {
    if (C > f * f) continue;
    v.push({ f, C, ratio: (f * f) / C, rmse: rmseOf(src, f, C) });
  }
  cacheTrade.clear(); cacheTrade.set(key, v);
  return v;
}

// ------------------------------------------------------------- draw helpers
function frame(ctx, x, y, w, h, col) { ctx.save(); ctx.strokeStyle = col || T.n6; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); ctx.restore(); }
const f2 = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const fE = (v) => (v >= 1e-4 ? v.toFixed(5) : v > 0 ? v.toExponential(1) : '0');
const fBig = (v) => (v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : String(Math.round(v)));

const STEPS = [
  { key: 'source', label: 'x — the image at full resolution' },
  { key: 'encode', label: 'encode — one strided transform per f×f block' },
  { key: 'truncate', label: 'TRUNCATE — everything past channel C is discarded HERE' },
  { key: 'latent', label: 'z — the arena the whole sampling loop runs in' },
  { key: 'diffuse', label: 'diffuse — tens of steps, all of them inside z' },
  { key: 'decode', label: 'decode — one pass back to pixels' },
  { key: 'floor', label: 'the RECONSTRUCTION FLOOR — residual with no diffusion at all' },
];
const REVEAL = { src: 0, coef: 1, lat: 3, rec: 5, err: 6 };

mount({
  mount: 'body',
  title: 'latent-space — the compression that makes image generation affordable, and its ceiling',
  blurb: 'Diffusing directly on pixels is prohibitive: every sampling step touches every pixel, and attention costs the square of the token count. Latent diffusion pays the pixel price exactly twice — encode once, decode once — and runs the entire loop inside a compressed space in between. Drag the downsample factor and the channel count and watch the compute saving and the reconstruction error move in OPPOSITE directions under your hand. Then set the sampling error to zero: the residual panel does NOT go blank. Whatever the autoencoder cannot represent is gone before the first denoising step, and no sampler recovers it — the ceiling on the whole system is set by a component nobody talks about. The encoder here is a toy (a block DCT with a truncation, stated in full in the readout), not a trained VAE, so read the SHAPE of the trade, not the exact numbers. Paint into the source image and watch what survives the round trip; hover any latent cell for the region it covers.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  animate: true,
  compare: {
    key: 'chan',
    a: 4, b: 16,
    labelA: 'C = 4 channels — the original latent-diffusion recipe',
    labelB: 'C = 16 channels — the direction the field actually moved',
  },
  challenges: [
    {
      goal: 'Make the reconstruction floor vanish. Keep every channel of a block and read what compression you have left.',
      hint: 'the transform is orthonormal, so keeping all f² channels is lossless — and buys you exactly nothing.',
      check: (api) => ({ solved: (api.probe.rmse ?? 1) < 1e-6, detail: `floor RMSE = ${fE(api.probe.rmse ?? 1)} at ${f2(api.probe.ratio ?? 1)}× fewer elements (need RMSE < 1e-6)` }),
    },
    {
      goal: 'Turn the sampler off completely (sampling error = 0) while still throwing channels away, and confirm the residual survives.',
      hint: 'a perfect diffusion loop lands exactly on the right latent. The residual that is left is the part the encoder threw away.',
      check: (api) => ({ solved: (api.probe.noise ?? 1) === 0 && (api.probe.rmse ?? 0) > 1e-6, detail: `sampling error = ${(api.probe.noise ?? 0).toFixed(2)}, floor RMSE = ${fE(api.probe.rmse ?? 0)} (need error 0 AND a non-zero floor)` }),
    },
    {
      goal: 'Find a setting that saves more than 64× the elements, and read what that costs against the 8×/4-channel reference floor.',
      hint: 'drag the trade panel to the right, or push the downsample factor to 16 with a small channel count.',
      check: (api) => ({ solved: (api.probe.ratio ?? 0) > 64, detail: `saving = ${f2(api.probe.ratio ?? 0)}× fewer elements, floor = ${f2(api.probe.pct ?? 0)}% of the reference (need > 64×)` }),
    },
  ],
  controls: (c, page) => {
    c.select('scene', { label: 'source image', options: [
      { value: 'mixed', label: 'mixed — ramp, grating, text, edges' },
      { value: 'text', label: 'small text' },
      { value: 'grating', label: 'frequency sweep' },
      { value: 'portrait', label: 'smooth shapes (the easy case)' },
    ], value: 'mixed' });
    // Drag-first, but every factor has to divide the image side, so the value
    // snaps to the nearest of 2/4/8/16 rather than pretending 6× is available.
    c.slider('down', { label: 'downsample f (drag · snaps 2/4/8/16)', min: 2, max: 16, step: 1, value: 8 });
    c.slider('chan', { label: 'latent channels C (drag)', min: 1, max: 64, step: 1, value: 4, format: (v) => String(v | 0) });
    c.slider('noise', { label: 'sampling error in z', min: 0, max: 0.4, step: 0.01, value: 0 });
    c.select('brush', { label: 'brush (drag on x)', options: [
      { value: 'detail', label: 'fine detail — 1px checker' },
      { value: 'dot', label: 'bright dot' },
      { value: 'erase', label: 'erase to mid grey' },
    ], value: 'detail' });
    c.slider('seed', { label: 'sampler noise seed', min: 0, max: 99, step: 1, value: 3 });
    c.transport({ compute: () => STEPS.map((s, i) => ({ i, key: s.key, label: `${i} · ${s.label}` })), speed: 1.2, loop: true });
  },
  onPointer: (page, ev) => {
    const sp = panels.find((p) => p.key === 'src'), tp = panels.find((p) => p.key === 'trade');
    if (ev.type === 'up' || ev.type === 'leave') { painting = false; dragTrade = false; return; }
    if (ev.type === 'down') {
      if (tp && ev.x >= tp.rect.x - 10 && ev.x <= tp.rect.x + tp.rect.w + 10 && ev.y >= tp.rect.y - 10 && ev.y <= tp.rect.y + tp.rect.h + 10) {
        dragTrade = true; pickTrade(page, tp, ev.x, ev.y); return;
      }
      if (sp) { const hit = cellAt(sp.rect, N, N, ev.x, ev.y); if (hit) { painting = true; paintAt(hit.r, hit.c, page.state.brush); page.redraw(); return; } }
      for (const p of panels) { if (!p.grid) continue; const hit = cellAt(p.rect, p.grid.rows, p.grid.cols, ev.x, ev.y); if (hit) { selCell = { panel: p.key, r: hit.r, c: hit.c }; page.redraw(); return; } }
    } else if (ev.type === 'move' && page.pointer.down) {
      if (dragTrade && tp) pickTrade(page, tp, ev.x, ev.y);
      else if (painting && sp) { const hit = cellAt(sp.rect, N, N, ev.x, ev.y); if (hit) { paintAt(hit.r, hit.c, page.state.brush); page.redraw(); } }
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    if (st.scene !== src._scene) { src = buildSource(SCENES.includes(st.scene) ? st.scene : 'mixed'); src._scene = st.scene; srcVersion++; }

    const f = FS.reduce((a, b) => (Math.abs(b - (+st.down || 8)) < Math.abs(a - (+st.down || 8)) ? b : a));
    const C = Math.max(1, Math.min(st.chan | 0, f * f));
    const noise = +st.noise || 0;
    const si = page.step();
    const step = si ? si.i : 0;

    const rtFloor = roundTrip(f, C, 0, 0);                 // NO diffusion at all
    const rt = noise > 0 ? roundTrip(f, C, noise, st.seed | 0) : rtFloor;
    const { nb, F2 } = rtFloor;
    const { e: eBasis, tot: eTot } = energy(rtFloor.coef, f, nb);
    const { zz, rank } = basis(f);

    // ---- the numbers, all computed here from the reader's own settings -------
    const tokens = nb * nb;
    const latElems = tokens * C;
    const ratio = PIX / latElems;                           // = f² / C
    const attnRatio = (PIX / tokens) ** 2;                  // attention ∝ tokens²
    const ratioRGB = (PIX * 3) / latElems;                  // same latent, RGB source
    const ref = refFloor();
    const pct = ref > 0 ? (rtFloor.rmse / ref) * 100 : 0;
    let dropped = 0; for (let k = C; k < F2; k++) { const [u, v] = zz[k]; dropped += eBasis[u * f + v]; }
    const droppedPct = (dropped / eTot) * 100;
    page.probe = { rmse: rtFloor.rmse, total: rt.rmse, ratio, pct, noise, f, C, droppedPct };

    // ---- panel row -----------------------------------------------------------
    const pad = 16, gap = 12, top = 78;
    const pw = Math.min(112, Math.floor((W - 2 * pad - 4 * gap) / 5));
    const rowW = 5 * pw + 4 * gap, ox = pad + Math.max(0, Math.floor((W - 2 * pad - rowW) / 2));
    const latTiles = Math.min(C, 16), latCols = Math.ceil(Math.sqrt(latTiles)), latRows = Math.ceil(latTiles / latCols);
    // The latent, drawn as its C channel planes tiled into one square.
    const latImg = new Float32Array(latCols * nb * latRows * nb);
    const latW = latCols * nb;
    for (let k = 0; k < latTiles; k++) {
      const tr = (k / latCols) | 0, tc = k % latCols, ch = rtFloor.lat[k];
      for (let a = 0; a < nb; a++) for (let b = 0; b < nb; b++) latImg[(tr * nb + a) * latW + tc * nb + b] = ch[a * nb + b];
    }
    // Coefficient budget: the f×f basis grid, coloured by the energy it carries.
    const eImg = new Float32Array(F2);
    for (let j = 0; j < F2; j++) eImg[j] = Math.log10(Math.max(eBasis[j] / eTot, 1e-8));

    const spec = [
      { key: 'src', data: src, rows: N, cols: N, ramp: 'seq', dom: [0, 1], title: 'x  source', sub: `${N}×${N} = ${PIX} elements` },
      { key: 'coef', data: eImg, rows: f, cols: f, ramp: 'seq', dom: [-8, 0], title: 'basis energy', sub: `${F2} per block · keep ${C}` },
      { key: 'lat', data: latImg, rows: latRows * nb, cols: latW, ramp: 'div', dom: null, title: 'z  latent', sub: `${nb}×${nb}×${C} = ${latElems}` },
      { key: 'rec', data: rt.rec, rows: N, cols: N, ramp: 'seq', dom: [0, 1], title: 'x̂  decoded', sub: noise > 0 ? 'after a lossy loop' : 'no diffusion at all' },
      { key: 'err', data: rtFloor.err, rows: N, cols: N, ramp: 'seq', dom: [0, Math.max(0.02, rtFloor.maxErr)], title: '|x̂−x|  the FLOOR', sub: `max ${rtFloor.maxErr.toFixed(3)}` },
    ];
    panels = [];
    spec.forEach((p, i) => {
      const rect = { x: ox + i * (pw + gap), y: top, w: pw, h: pw };
      const live = step >= REVEAL[p.key];
      ctx.save(); ctx.globalAlpha = live ? 1 : 0.14;
      r.heatmap(p.data, { rows: p.rows, cols: p.cols, rect, ramp: p.ramp === 'seq' ? ramps.sequential : ramps.diverging, domain: p.dom || 'auto' });
      ctx.restore();
      frame(ctx, rect.x, rect.y, rect.w, rect.h, p.key === 'err' ? T.bad : T.n6);
      r.label(p.title, rect.x, rect.y - 20, { color: p.key === 'err' ? T.bad : live ? T.n13 : T.n9, font: `11px ${MONO}` });
      r.label(p.sub, rect.x, rect.y - 7, { color: T.n10, font: `9px ${MONO}` });
      panels.push({ key: p.key, rect, data: p.data, title: p.title, grid: { rows: p.rows, cols: p.cols } });

      // The truncation, drawn on the basis panel: kept channels outlined, the
      // rest washed out. This is the exact moment the information is lost.
      if (p.key === 'coef' && step >= 2) {
        const cw = rect.w / f;
        for (let u = 0; u < f; u++) for (let v = 0; v < f; v++) {
          const keptHere = rank[u * f + v] < C;
          ctx.save();
          if (!keptHere) { ctx.fillStyle = rgbaToken('n14', 0.55); ctx.fillRect(rect.x + v * cw, rect.y + u * cw, cw, cw); }
          else if (cw > 3) { ctx.strokeStyle = T.ok; ctx.lineWidth = 1; ctx.strokeRect(rect.x + v * cw + 0.5, rect.y + u * cw + 0.5, cw - 1, cw - 1); }
          ctx.restore();
        }
        r.label(`${droppedPct.toFixed(2)}% of the image energy discarded`, rect.x, rect.y + rect.h + 12, { color: T.bad, font: `9px ${MONO}` });
      }
      // The loop lives in z: a shimmer over the latent while the transport is on
      // the diffuse step. Illustrative motion only -- no sampler runs here.
      if (p.key === 'lat' && step === 4) {
        ctx.save();
        ctx.strokeStyle = alphaOf(T.violet, 0.55 + 0.35 * Math.sin((page.t || 0) * 4));
        ctx.lineWidth = 2.5; ctx.strokeRect(rect.x - 3.5, rect.y - 3.5, rect.w + 7, rect.h + 7);
        ctx.restore();
        r.label('every step, in here', rect.x, rect.y + rect.h + 12, { color: T.violet, font: `9px ${MONO}` });
      }
    });

    // ---- the ceiling bar: irreducible vs reducible ---------------------------
    const cy = top + pw + 30;
    // Wide enough that the two legends cannot collide at 9px mono.
    const cbW = Math.min(560, W - 2 * pad), cbX = ox;
    const totalE = Math.max(rt.rmse, 1e-9), floorE = Math.min(rtFloor.rmse, totalE);
    const scale = Math.max(totalE, ref * 1.6, 1e-6);
    ctx.save();
    ctx.fillStyle = T.n2; ctx.fillRect(cbX, cy, cbW, 13);
    ctx.fillStyle = T.bad; ctx.fillRect(cbX, cy, (floorE / scale) * cbW, 13);
    ctx.fillStyle = alphaOf(T.violet, 0.75); ctx.fillRect(cbX + (floorE / scale) * cbW, cy, ((totalE - floorE) / scale) * cbW, 13);
    ctx.strokeStyle = T.n11; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cbX + (ref / scale) * cbW + 0.5, cy - 3); ctx.lineTo(cbX + (ref / scale) * cbW + 0.5, cy + 16); ctx.stroke();
    ctx.restore();
    frame(ctx, cbX, cy, cbW, 13);
    r.label('■ irreducible — the encoder threw it away', cbX, cy - 5, { color: T.bad, font: `9px ${MONO}` });
    r.label('■ reducible — the sampling loop', cbX + cbW, cy - 5, { color: T.violet, font: `9px ${MONO}`, align: 'right' });
    r.label(`│ 8×/4ch reference floor (RMSE ${fE(ref)})`, cbX + (ref / scale) * cbW + 3, cy + 25, { color: T.n11, font: `9px ${MONO}` });

    // ---- bottom row: trade curve | arithmetic | contested --------------------
    const by = cy + 34, bh = Math.max(96, H - by - 14);
    // Three unequal columns: the arithmetic and the disagreement both need more
    // width than the plot, and at 0.30 each they overlapped.
    const avail = W - 2 * pad - 32;
    const w1 = Math.max(140, Math.floor(avail * 0.27)), w2 = Math.max(180, Math.floor(avail * 0.36));
    const x1 = pad, x2 = x1 + w1 + 16, x3 = x2 + w2 + 16;
    const colW = w1;

    // -- trade curve: every (f, C) the reader can reach ------------------------
    const tr = { x: x1, y: by + 14, w: colW, h: bh - 40 };
    r.label('cheaper ⟷ more error (drag me)', x1, by + 6, { color: T.n13, font: `11px ${MONO}` });
    frame(ctx, tr.x, tr.y, tr.w, tr.h);
    panels.push({ key: 'trade', rect: tr, data: null, title: 'trade' });
    const curve = tradeCurve();
    let rmMax = 0; for (const d of curve) if (d.rmse > rmMax) rmMax = d.rmse;
    rmMax = Math.max(rmMax, 1e-6);
    const tx = (d) => tr.x + (Math.log2(d.ratio) / 8) * tr.w;
    const ty = (d) => tr.y + tr.h - 4 - (d.rmse / rmMax) * (tr.h - 8);
    for (const d of curve) {
      const cc = categorical(FS.indexOf(d.f));
      ctx.save(); ctx.fillStyle = alphaOf(cc, 0.75);
      ctx.beginPath(); ctx.arc(tx(d), ty(d), 2.6, 0, 7); ctx.fill(); ctx.restore();
    }
    // reference + current markers
    const refPt = curve.find((d) => d.f === REF_F && d.C === REF_C);
    if (refPt) { ctx.save(); ctx.strokeStyle = T.n11; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(tx(refPt), ty(refPt), 6, 0, 7); ctx.stroke(); ctx.restore(); }
    const curPt = { ratio, rmse: rtFloor.rmse };
    ctx.save(); ctx.fillStyle = T.warn; ctx.beginPath(); ctx.arc(tx(curPt), ty(curPt), 4.5 + 0.8 * Math.sin((page.t || 0) * 3), 0, 7); ctx.fill(); ctx.restore();
    FS.forEach((ff, i) => r.label(`● f=${ff}`, tr.x + 4 + i * 40, tr.y + 11, { color: alphaOf(categorical(i), 0.95), font: `9px ${MONO}` }));
    r.label('1×', tr.x, tr.y + tr.h + 11, { color: T.n10, font: `9px ${MONO}` });
    r.label('256× fewer elements →', tr.x + tr.w - 118, tr.y + tr.h + 11, { color: T.n10, font: `9px ${MONO}` });
    r.label(`↑ floor RMSE, 0 … ${fE(rmMax)}`, tr.x, tr.y + tr.h + 22, { color: T.n10, font: `9px ${MONO}` });

    // -- arithmetic -----------------------------------------------------------
    r.label('the arithmetic, from your settings', x2, by + 6, { color: T.n13, font: `11px ${MONO}` });
    const A = [
      [`f = ${f}   C = ${C}   (C ≤ f² = ${F2})`, T.n12],
      [`pixels        ${N}×${N}×1        = ${fBig(PIX)}`, T.n13],
      [`latent        ${nb}×${nb}×${C}${nb < 10 ? '  ' : ''}      = ${fBig(latElems)}`, T.accent],
      [`elements      f²/C = ${f}²/${C}   = ${f2(ratio)}× FEWER`, T.ok],
      [`same on RGB   3f²/C            = ${f2(ratioRGB)}× FEWER`, T.ok],
      [`tokens        ${N}² → ${tokens}`, T.n12],
      [`attention ∝ tokens²            = ${fBig(attnRatio)}× FEWER`, T.okDeep],
      ['—', T.n6],
      [`floor RMSE    ${fE(rtFloor.rmse)}`, T.bad],
      [`  = ${f2(pct)}% of the 8×/4ch reference`, pct > 100 ? T.bad : T.ok],
      ['  (lower is better; 100% = parity)', T.n10],
      [`discarded energy               ${droppedPct.toFixed(2)}%`, T.bad],
      [`with a lossy loop (σ=${noise.toFixed(2)})  ${fE(rt.rmse)}`, T.violet],
    ];
    A.forEach((ln, i) => r.label(ln[0], x2, by + 22 + i * 12.4, { color: ln[1], font: `9.5px ${MONO}` }));

    // -- contested ------------------------------------------------------------
    r.label('this recipe is CONTESTED', x3, by + 6, { color: T.warnDeep, font: `11px ${MONO}` });
    const Ctd = [
      ['8× down / 4 channels came from the original', T.n12],
      ['latent-diffusion paper and stuck for years.', T.n12],
      ['Against it: the same team\'s rectified-flow', T.n11],
      ['work raised the channel count (4 → 16) and', T.n11],
      ['reported better reconstruction AND samples;', T.n11],
      ['2025–26 representation-autoencoder work', T.n11],
      ['replaces the VAE with a pretrained encoder,', T.n11],
      ['arguing a thin latent is the limiting factor.', T.n11],
      ['For it: deep-compression autoencoders push', T.n11],
      ['the OTHER way (32×, 64×) and still generate;', T.n11],
      ['and RMSE is not perception — a trained VAE', T.n11],
      ['spends adversarial + perceptual loss buying', T.n11],
      ['back exactly what this toy cannot.', T.n11],
      ['Nobody has settled where the knee is.', T.warnDeep],
    ];
    Ctd.forEach((ln, i) => r.label(ln[0], x3, by + 22 + i * 12.4, { color: ln[1], font: `9.5px ${MONO}` }));

    // ---- hover / pinned inspection ------------------------------------------
    const pt = page.pointer;
    if (pt.over && !painting && !dragTrade) {
      for (const p of panels) {
        if (!p.grid) continue;
        const hit = cellAt(p.rect, p.grid.rows, p.grid.cols, pt.x, pt.y);
        if (!hit) continue;
        page.setTip(tipFor(p, hit, { f, C, nb, latCols, rtFloor, rt, eBasis, eTot, rank, zz, noise }));
        break;
      }
      const tp = panels.find((q) => q.key === 'trade');
      if (tp && pt.x >= tp.rect.x && pt.x <= tp.rect.x + tp.rect.w && pt.y >= tp.rect.y && pt.y <= tp.rect.y + tp.rect.h) {
        const d = nearestTrade(tp, pt.x, pt.y, curve, rmMax);
        if (d) page.setTip(`f = ${d.f}, C = ${d.C}\n${f2(d.ratio)}× fewer elements\nfloor RMSE = ${fE(d.rmse)}\nclick to jump here`);
      }
    }
    if (selCell) {
      const p = panels.find((q) => q.key === selCell.panel);
      if (p && p.grid && selCell.r < p.grid.rows && selCell.c < p.grid.cols) {
        const cwx = p.rect.w / p.grid.cols, cwy = p.rect.h / p.grid.rows;
        ctx.save(); ctx.strokeStyle = T.warn; ctx.lineWidth = 1.5;
        ctx.strokeRect(p.rect.x + selCell.c * cwx + 0.5, p.rect.y + selCell.r * cwy + 0.5, Math.max(2, cwx - 1), Math.max(2, cwy - 1));
        ctx.restore();
      }
    }

    // ---- readout -------------------------------------------------------------
    let o = `step ${step}/${STEPS.length - 1} — ${STEPS[Math.max(0, Math.min(STEPS.length - 1, step))].label}.   tier:${r.name}\n`;
    o += `Encoder: f = ${f}× downsample, C = ${C} of ${F2} channels kept per block. `;
    o += `${N}×${N}×1 = ${PIX} pixel elements → ${nb}×${nb}×${C} = ${latElems} latent elements = ${f2(ratio)}× FEWER elements (f²/C). `;
    o += `On a 3-channel RGB source the same setting gives ${f2(ratioRGB)}× FEWER. `;
    o += `Token count falls ${PIX} → ${tokens}, so a quadratic-attention denoiser does ${fBig(attnRatio)}× FEWER attention units of work per step — and a latent-diffusion pipeline pays the pixel-resolution cost exactly twice, at encode and decode, instead of once per sampling step.\n`;
    const lossless = rtFloor.rmse < 1e-6;
    o += `THE CEILING: encode and decode with NO diffusion at all and the residual is RMSE ${fE(rtFloor.rmse)} = ${f2(pct)}% of the 8×/4-channel reference floor (RMSE ${fE(ref)}; lower is better, 100% = parity). ${droppedPct.toFixed(2)}% of the image's energy is discarded at the truncate step, and nothing downstream restores it. `;
    o += lossless
      ? `You are keeping every channel of every block, so this encoder is lossless and the floor is zero — and it has bought you ${f2(ratio)}× fewer elements, which is to say nothing. That is the trade in its degenerate corner: the saving IS the loss.`
      : noise > 0
        ? `With a lossy loop (σ = ${noise.toFixed(2)} on the kept channels) the total is RMSE ${fE(rt.rmse)} — and the red part of the bar is the part no sampler can remove, however good it gets.`
        : `The sampling error is set to 0 — a perfect loop — and the residual panel is still not blank. That is the point of this page.`;
    o += `\nTOY ENCODER, NOT A TRAINED VAE: an orthonormal block DCT-II over f×f blocks, keeping the C lowest-frequency coefficients per block in zig-zag order; the decoder is the inverse transform with the discarded coefficients set to zero. Because the transform is orthonormal, the discarded energy IS the squared error — the floor is a sum, not an estimate. A trained autoencoder learns its basis and optimises adversarial + perceptual objectives, so it buys back detail this toy cannot; read the SHAPE of the trade here, never the exact numbers. `;
    o += `CONTESTED: the 8×/4-channel recipe is not settled — later rectified-flow work raised the channel count to 16 and reported better reconstruction and samples, representation-autoencoder work replaces the VAE outright on the argument that a thin latent limits generation, and deep-compression autoencoders push compression far higher instead. Sources in the README. `;
    o += `The denoising loop itself is the diffusion-noise page; step counts and samplers are diffusion-sampler. Neither runs here — this page is the arena, and the price of the door.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__latentSpacePage = page;
  const q = new URLSearchParams(location.search);
  const tr = page.controls._transport;
  for (const k of ['scene', 'brush']) if (q.has(k)) page.controls.set(k, q.get(k));
  for (const k of ['down', 'chan', 'noise', 'seed']) if (q.has(k)) page.controls.set(k, +q.get(k));
  // ?stamp=r,c;r,c -- headless stand-in for painting with the brush (there is no
  // pointer under --screenshot), so an edited source is capture-verifiable.
  if (q.has('stamp')) { for (const s of q.get('stamp').split(';')) { const [rr, cc] = s.split(',').map(Number); paintAt(rr | 0, cc | 0, page.state.brush); } }
  // ?paint=i,v;i,v -- exact pixel values, same hook shape the other pages use.
  if (q.has('paint')) { for (const s of q.get('paint').split(';')) { const [i, v] = s.split(',').map(Number); if (i >= 0 && i < PIX) src[i] = Math.max(0, Math.min(1, v)); } bumpSrc(); }
  if (q.has('cell')) { const [pk, rr, cc] = q.get('cell').split(','); selCell = { panel: pk, r: +rr | 0, c: +cc | 0 }; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  // Any explicit handle pauses the autoplay so the transport cannot advance off
  // the requested step before a capture.
  if (q.has('step') || q.has('hover') || q.has('cell') || q.has('stamp') || q.has('paint')) { if (tr) tr.pause(); }
  if (q.has('step') && tr) tr.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && tr) tr.play();
  page.redraw();
});

// ------------------------------------------------------------------ helpers
function bumpSrc() { srcVersion++; cacheRT.clear(); cacheTrade.clear(); cacheRef.clear(); }
function paintAt(r, c, brush) {
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    const rr = r + dr, cc = c + dc;
    if (rr < 0 || rr >= N || cc < 0 || cc >= N) continue;
    src[rr * N + cc] = brush === 'erase' ? 0.5 : brush === 'dot' ? 0.98 : ((rr + cc) % 2 ? 0.98 : 0.02);
  }
  bumpSrc();
}
function nearestTrade(tp, x, y, curve, rmMax) {
  let best = null, bd = Infinity;
  for (const d of curve) {
    const px = tp.rect.x + (Math.log2(d.ratio) / 8) * tp.rect.w;
    const py = tp.rect.y + tp.rect.h - 4 - (d.rmse / rmMax) * (tp.rect.h - 8);
    const dd = (px - x) ** 2 + (py - y) ** 2;
    if (dd < bd) { bd = dd; best = d; }
  }
  return best;
}
function pickTrade(page, tp, x, y) {
  const curve = tradeCurve();
  let rmMax = 0; for (const d of curve) if (d.rmse > rmMax) rmMax = d.rmse;
  const d = nearestTrade(tp, x, y, curve, Math.max(rmMax, 1e-6));
  if (!d) return;
  page.controls.set('down', d.f, { silent: true });
  page.controls.set('chan', d.C);
}
// Hover text: value plus the derivation that produced it, per panel.
function tipFor(p, hit, ctxv) {
  const { f, C, nb, latCols, rtFloor, rt, eBasis, eTot, rank, zz, noise } = ctxv;
  const i = hit.r * p.grid.cols + hit.c;
  if (p.key === 'src') return `x  pixel (${hit.r}, ${hit.c})\nvalue = ${src[i].toFixed(4)}\nlives in block (${(hit.r / f) | 0}, ${(hit.c / f) | 0})\ndrag to paint detail in`;
  if (p.key === 'rec') return `x̂  pixel (${hit.r}, ${hit.c})\nvalue = ${rt.rec[i].toFixed(4)}   (x = ${src[i].toFixed(4)})\ninverse transform of ${C} kept channels${noise > 0 ? `, loop σ=${noise.toFixed(2)}` : ', no diffusion'}`;
  if (p.key === 'err') return `|x̂ − x| at (${hit.r}, ${hit.c})\n= |${rtFloor.rec[i].toFixed(4)} − ${src[i].toFixed(4)}| = ${rtFloor.err[i].toFixed(4)}\nno diffusion ran — this is the FLOOR`;
  if (p.key === 'coef') {
    const k = rank[hit.r * f + hit.c], share = (eBasis[hit.r * f + hit.c] / eTot) * 100;
    return `basis (u,v) = (${hit.r}, ${hit.c})\nchannel ${k} in zig-zag order\ncarries ${share.toFixed(3)}% of the image energy\n${k < C ? '✓ KEPT' : '✗ DISCARDED — gone before diffusion'}`;
  }
  if (p.key === 'lat') {
    const tr0 = (hit.r / nb) | 0, tc0 = (hit.c / nb) | 0, k = tr0 * latCols + tc0;
    if (k >= C || k >= rtFloor.lat.length) return 'latent tile — no channel here';
    const a = hit.r % nb, b = hit.c % nb, [u, v] = zz[k];
    return `z  channel ${k}, cell (${a}, ${b})\nvalue = ${rtFloor.lat[k][a * nb + b].toFixed(4)}\ncovers source rows ${a * f}–${a * f + f - 1}, cols ${b * f}–${b * f + f - 1}\n(one ${f}×${f} block = one latent position)\nbasis (u,v) = (${u}, ${v})`;
  }
  return null;
}
