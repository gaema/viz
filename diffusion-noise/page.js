// diffusion-noise concept page -- what a diffusion model is TRAINED to do.
//
// TRAINING, not sampling. One training step is: take a clean sample x0, draw a
// noise field eps ~ N(0, I), draw a timestep t, corrupt
//     x_t = a(t)*x0 + s(t)*eps
// and ask the network to regress a target (eps, or v, or x0 itself) from
// (x_t, t) alone. Two interpolants are offered side by side because they are
// the same construction with different coefficients:
//   diffusion (DDPM, https://arxiv.org/abs/2006.11239)
//       a = sqrt(abar_t),  s = sqrt(1 - abar_t)          (a^2 + s^2 = 1)
//   rectified flow (Liu, Gong & Liu, https://arxiv.org/abs/2209.03003 -- NOT
//       2210.02747, which is Lipman et al.'s Flow Matching; the two sibling
//       cards diffusion-sampler and few-step-distillation both cite 2209.03003
//       for this, and the README here already does)
//       a = 1 - tau,       s = tau                        (a + s = 1)
// The flow-matching paper's Gaussian probability paths subsume the diffusion
// paths as special cases, and rectified flow is what production image/video
// models train today -- so the toggle is not a footnote, it is the same page.
//
// THE MOMENT THIS PAGE EXISTS FOR: the fourth panel, x0-hat -- the model's
// current guess of the clean sample at this noise level. It is the dataset mean
// (blurry, structureless) at high noise and snaps to sharp texture at low
// noise, and watching that transition is what makes diffusion click.
//
// WHERE x0-hat COMES FROM (no weights are downloaded -- see README): the toy
// data distribution here is "pick one of K synthetic 16x16 templates, then add
// per-pixel spread", which is Gaussian enough that the Bayes-optimal denoiser
// -- the function a trained network approximates -- is available in closed form
// (see denoise() below). It is computed per frame, so every number on screen is
// real arithmetic you can hover and read.
//
// NOT THIS PAGE: integrating the trajectory, step counts and samplers belong to
// the diffusion-sampler page; conditioning and guidance scale belong to the
// guidance page. Both are named in the readout, neither is built here.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';
import { ramps, categorical, cellAt } from '../framework/render.js';
import { T, alphaOf } from '../framework/theme.js';

const N = 16;                 // image side (N*N pixels)
const PIX = N * N;
const BINS = 32;              // t axis resolution (the transport's steps)
const MONO = 'ui-monospace, monospace';

// ---------------------------------------------------------------- toy dataset
// Six deterministic 16x16 shapes in [0,1]. Small and synthetic on purpose: the
// closed-form denoiser above is only exact for a FINITE training set, and a
// finite set is also the honest way to show memorisation vs the blurry mean.
const SHAPES = ['ring', 'bars', 'cross', 'checker', 'blob', 'wedge'];
function makeRef(kind) {
  const im = new Float32Array(PIX);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const y = (r + 0.5) / N - 0.5, x = (c + 0.5) / N - 0.5, rad = Math.hypot(x, y);
    let v = 0;
    if (kind === 'ring') v = Math.exp(-((rad - 0.3) ** 2) / 0.008);
    else if (kind === 'bars') v = 0.5 + 0.5 * Math.cos(c * Math.PI / 2.0);
    else if (kind === 'cross') v = Math.exp(-(x * x) / 0.004) + Math.exp(-(y * y) / 0.004);
    else if (kind === 'checker') v = ((r >> 2) + (c >> 2)) % 2 ? 0.92 : 0.06;
    else if (kind === 'blob') v = Math.exp(-(((x + 0.12) ** 2 + (y - 0.1) ** 2)) / 0.02);
    else v = (x + y > 0 ? 1 : 0) * (0.35 + 0.65 * (0.5 - rad));
    im[r * N + c] = Math.max(0, Math.min(1, v));
  }
  return im;
}
function makeRefs(K) { return SHAPES.slice(0, K).map(makeRef); }

// A training SAMPLE is a template plus per-pixel spread: real data carries
// detail its class does not predict, and without that detail the closed-form
// denoiser below is perfect at every noise level and the whole eps-vs-x0 trade
// collapses to zero. `spread` is the knob for how much such detail exists.
function sampleOf(refs, si, sd) {
  const z = seededRandn(9001 + si, PIX, { std: 1 }), x0 = new Float32Array(PIX), mu = refs[si];
  for (let i = 0; i < PIX; i++) x0[i] = mu[i] + sd * z[i];
  return x0;
}

// ------------------------------------------------------------ the interpolant
// a(t), s(t) for both modes. `sched` reshapes the path: for diffusion it picks
// the beta/abar schedule; for rectified flow it picks the timestep SHIFT (the
// SD3-style reparameterisation tau = shift*t / (1 + (shift-1)*t), which spends
// more or less of the axis at high noise). Same control, same meaning: how fast
// signal is destroyed as t runs 0 -> 1.
const SHIFT = { linear: 1, cosine: 3, quadratic: 0.5 };
function coeffs(mode, sched, t) {
  if (mode === 'flow') {
    const shift = SHIFT[sched] ?? 1;
    const tau = shift * t / (1 + (shift - 1) * t);
    return { a: 1 - tau, s: tau, tau, shift, ab: null };
  }
  let ab;
  if (sched === 'cosine') { const f = (u) => Math.cos((u + 0.008) / 1.008 * Math.PI / 2) ** 2; ab = f(t) / f(0); }
  else if (sched === 'quadratic') ab = Math.exp(-10.1 * t ** 3);
  else ab = Math.exp(-1000 * (1e-4 * t + (0.02 - 1e-4) * t * t / 2));
  ab = Math.min(1 - 1e-7, Math.max(1e-7, ab));
  return { a: Math.sqrt(ab), s: Math.sqrt(1 - ab), ab, tau: null, shift: null };
}
const snrOf = (a, s) => (a * a) / Math.max(1e-12, s * s);

// ------------------------------------------------------------- the "network"
// Closed-form optimal denoiser for the data model above (see header). The prior
// is a mixture: pick one of the K templates, then add per-pixel spread sd -- so
// a training sample carries texture the templates do NOT explain, exactly like
// real data carries detail no class label predicts. Both stages stay Gaussian,
// so the posterior mean is exact:
//     x_t | k  ~  N(a*mu_k, (a^2*sd^2 + s^2) I)              -> the weights w_k
//     E[x0 | x_t, k] = (a*sd^2*x_t + s^2*mu_k) / (a^2*sd^2 + s^2)
// The second line is the whole "blurry mean early, texture late" effect in one
// expression: at high noise s^2 dominates and the estimate IS the template mean;
// at low noise the x_t term dominates and the un-templated texture comes back.
// The weights carry the other half -- near-uniform weights ARE the dataset mean.
function denoise(refs, xt, a, s, sd) {
  const vv = Math.max(a * a * sd * sd + s * s, 1e-8), inv = 1 / (2 * vv);
  const logw = refs.map((rf) => { let d = 0; for (let i = 0; i < PIX; i++) { const df = xt[i] - a * rf[i]; d += df * df; } return -d * inv; });
  let m = -Infinity; for (const l of logw) if (l > m) m = l;
  let z = 0; const w = logw.map((l) => { const e = Math.exp(l - m); z += e; return e; });
  for (let k = 0; k < w.length; k++) w[k] /= z;
  const cx = (a * sd * sd) / vv, cm = (s * s) / vv;          // x_t and template shares
  const x0h = new Float32Array(PIX);
  for (let k = 0; k < w.length; k++) {
    const wk = w[k]; if (wk < 1e-7) continue; const rf = refs[k];
    for (let i = 0; i < PIX; i++) x0h[i] += wk * (cx * xt[i] + cm * rf[i]);
  }
  return { x0h, w, share: cx * (a || 1e-9) };
}

// One training example at (t, eps): the corrupted input, the model's estimate,
// and every parameterisation of the regression target derived from them.
function example(refs, x0, eps, mode, sched, t, sd) {
  const { a, s, ab, tau, shift } = coeffs(mode, sched, t);
  const xt = new Float32Array(PIX);
  for (let i = 0; i < PIX; i++) xt[i] = a * x0[i] + s * eps[i];
  const { x0h, w } = denoise(refs, xt, a, s, sd);
  const epsH = new Float32Array(PIX), vTrue = new Float32Array(PIX), vHat = new Float32Array(PIX), err = new Float32Array(PIX);
  const si = 1 / Math.max(s, 1e-6);
  for (let i = 0; i < PIX; i++) {
    epsH[i] = (xt[i] - a * x0h[i]) * si;
    // diffusion v-parameterisation v = a*eps - s*x0 ; flow velocity u = eps - x0.
    vTrue[i] = mode === 'flow' ? eps[i] - x0[i] : a * eps[i] - s * x0[i];
    vHat[i] = mode === 'flow' ? epsH[i] - x0h[i] : a * epsH[i] - s * x0h[i];
    err[i] = Math.abs(x0h[i] - x0[i]);
  }
  return { a, s, ab, tau, shift, xt, x0h, w, epsH, vTrue, vHat, err };
}

// A fixed error in the predicted target costs this much in IMAGE space:
//   eps-pred: x0_hat = (x_t - s*eps_hat)/a   -> gain s/a  (= 1/sqrt(SNR))
//   v-pred:   x0_hat = a*x_t - s*v_hat       -> gain s
//   x0-pred:  x0_hat = x0_hat                -> gain 1
// This is the whole trade: the same network error is harmless at low noise and
// catastrophic at high noise, and which end you protect is the weighting you pick.
function x0Gain(target, a, s) { return target === 'eps' ? s / Math.max(a, 1e-6) : target === 'v' ? s : 1; }
const WEIGHTS = {
  uniform: () => 1,
  snr: (a, s) => snrOf(a, s),
  minsnr: (a, s) => { const q = snrOf(a, s); return Math.min(q, 5) / Math.max(q, 1e-9); },
};

// ------------------------------------------------------------- page state
let refs = makeRefs(4), refSig = '';
let curveCache = null, curveSig = '';
let panels = [];              // hit-test rects for hover/drag, rebuilt each draw
let dragPix = -1, dragT = false, selPix = 8 * N + 8;

const binT = (i) => (i + 1) / BINS;
function seekT(page, t) {
  const tr = page.controls._transport; if (!tr || !tr.steps.length) return;
  let bi = 0, bd = Infinity;
  for (let i = 0; i < tr.steps.length; i++) { const d = Math.abs(tr.steps[i].t - t); if (d < bd) { bd = d; bi = i; } }
  tr.seek(bi);
}

// loss(t) in the chosen target space, measured with a fresh noise draw per bin,
// plus the weighting curve and their product (where the training signal lands).
function curves(st, x0) {
  const sig = `${st.mode}|${st.sched}|${st.target}|${st.weight}|${st.seed}|${st.sample}|${st.K}|${st.spread}|${refSig}`;
  if (curveSig === sig && curveCache) return curveCache;
  const out = [];
  const DRAWS = 4;   // one noise draw per bin reads as spikes, not as a lesson
  for (let i = 0; i < BINS; i++) {
    const t = binT(i);
    let se = 0, a = 1, sg = 0;
    for (let d0 = 0; d0 < DRAWS; d0++) {
      const eps = seededRandn(((st.seed | 0) * 131 + i * 977 + d0 * 7919) % 100000, PIX, { std: 1 });
      const ex = example(refs, x0, eps, st.mode, st.sched, t, st.spread);
      a = ex.a; sg = ex.s;
      for (let p = 0; p < PIX; p++) {
        const d = st.target === 'eps' ? eps[p] - ex.epsH[p] : st.target === 'v' ? ex.vTrue[p] - ex.vHat[p] : x0[p] - ex.x0h[p];
        se += d * d;
      }
    }
    const loss = se / (PIX * DRAWS), w = WEIGHTS[st.weight](a, sg);
    out.push({ t, loss, w, gain: x0Gain(st.target, a, sg), budget: loss * w });
  }
  curveCache = out; curveSig = sig;
  return out;
}

// ------------------------------------------------------------- draw helpers
function frame(ctx, x, y, w, h) { ctx.save(); ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); ctx.restore(); }
function plot(ctx, rect, ys, color, opts = {}) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = opts.width || 1.6;
  if (opts.dash) ctx.setLineDash(opts.dash);
  ctx.beginPath();
  for (let i = 0; i < ys.length; i++) {
    const px = rect.x + (i / (ys.length - 1)) * rect.w, py = rect.y + rect.h - 3 - Math.max(0, Math.min(1, ys[i])) * (rect.h - 6);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke(); ctx.restore();
}
const norm = (arr) => { let m = 0; for (const v of arr) if (v > m) m = v; return arr.map((v) => v / (m || 1)); };
const f3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
// A loss that reads 0.0000 hides the whole point at the easy end of the axis.
const fL = (v) => (v >= 1e-4 ? v.toFixed(4) : v.toExponential(1));

mount({
  mount: 'body',
  title: 'diffusion-noise — what a diffusion model is trained to do',
  blurb: 'Training a diffusion model is one line: corrupt a clean sample by a known amount, then ask the network to undo exactly that much. Pick a timestep t, draw noise ε, build x_t = a(t)·x₀ + s(t)·ε, and regress a target — the noise ε, the velocity v, or the clean x₀ — from (x_t, t) alone. The fourth panel is the one to watch: the model’s current guess of the CLEAN sample. At high noise it is the blurry average of everything it was trained on; as t falls the posterior collapses onto one sample and texture appears. The interpolant toggle swaps the diffusion ᾱ schedule for RECTIFIED FLOW, x_t = (1−t)·x₀ + t·ε with target velocity ε − x₀ — the same construction with straight-line coefficients, and what production image and video models actually train. Drag the t handle on the schedule panel, drag a pixel of x₀ to edit the training data under your hand, and hover any pixel for the arithmetic that produced it.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  animate: true,
  compare: { key: 'mode', a: 'diffusion', b: 'flow', labelA: 'diffusion — ᾱ schedule, predict ε', labelB: 'rectified flow — (1−t)·x₀ + t·ε, predict velocity' },
  challenges: [
    { goal: 'Push the noise up until the model can no longer tell which sample it is looking at: get the top posterior weight below 0.45.', hint: 'drag the t handle right — at high noise every training sample explains x_t about equally well, and the guess becomes their mean.', check: (api) => ({ solved: (api.probe.wmax ?? 1) < 0.45, detail: `top weight = ${(api.probe.wmax ?? 1).toFixed(3)} (need < 0.45)` }) },
    { goal: 'Find a noise level where a fixed ε-error is amplified more than 3× in image space.', hint: 'switch the target to ε; the gain is s/a, which blows up once the signal coefficient a gets small.', check: (api) => ({ solved: (api.probe.target === 'eps') && (api.probe.gain ?? 0) > 3, detail: `target=${api.probe.target}, gain = ${(api.probe.gain ?? 0).toFixed(2)}× (need ε and > 3×)` }) },
  ],
  controls: (c, page) => {
    c.select('mode', { label: 'interpolant', options: [{ value: 'diffusion', label: 'diffusion — ᾱ schedule' }, { value: 'flow', label: 'rectified flow — (1−t)x₀+tε' }] , value: 'diffusion' });
    c.select('sched', { label: 'schedule shape', options: ['linear', 'cosine', 'quadratic'], value: 'linear' });
    c.select('target', { label: 'predict', options: [{ value: 'eps', label: 'ε — the noise' }, { value: 'v', label: 'v / velocity' }, { value: 'x0', label: 'x₀ — the clean sample' }], value: 'eps' });
    c.select('weight', { label: 'loss weighting', options: [{ value: 'uniform', label: 'uniform (simple loss)' }, { value: 'snr', label: 'SNR' }, { value: 'minsnr', label: 'min-SNR-5' }], value: 'uniform' });
    c.stepper('K', { label: 'training set', min: 2, max: 6, value: 4 });
    c.slider('spread', { label: 'data spread', min: 0.02, max: 0.4, step: 0.02, value: 0.18 });
    c.stepper('sample', { label: 'sample', min: 0, max: 5, value: 0 });
    c.slider('seed', { label: 'noise seed', min: 0, max: 99, step: 1, value: 7 });
    c.transport({ compute: () => Array.from({ length: BINS }, (_, i) => ({ i, t: binT(i), label: `t = ${binT(i).toFixed(3)}` })), speed: 6, loop: true });
  },
  onPointer: (page, ev) => {
    const x0p = panels.find((p) => p.key === 'x0'), sc = panels.find((p) => p.key === 'sched');
    if (ev.type === 'down') {
      dragPix = -1; dragT = false;
      if (sc && ev.x >= sc.rect.x - 8 && ev.x <= sc.rect.x + sc.rect.w + 8 && ev.y >= sc.rect.y - 8 && ev.y <= sc.rect.y + sc.rect.h + 18) {
        dragT = true; seekT(page, Math.max(0, Math.min(1, (ev.x - sc.rect.x) / sc.rect.w)));
        return;
      }
      for (const p of panels) {
        if (!p.rect || p.key === 'sched') continue;
        const hit = cellAt(p.rect, N, N, ev.x, ev.y);
        if (hit) { selPix = hit.r * N + hit.c; if (p.key === 'x0') dragPix = selPix; page.redraw(); return; }
      }
    } else if (ev.type === 'up' || ev.type === 'leave') { dragPix = -1; dragT = false; }
    else if (ev.type === 'move' && page.pointer.down) {
      if (dragT && sc) { seekT(page, Math.max(0, Math.min(1, (ev.x - sc.rect.x) / sc.rect.w))); }
      else if (dragPix >= 0) {
        const im = refs[page.state.sample % refs.length];
        im[dragPix] = Math.max(0, Math.min(1, im[dragPix] - ev.dy / 90));
        refSig = `paint:${Date.now()}`; page.redraw();
      }
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    if (refs.length !== (st.K | 0)) { refs = makeRefs(st.K | 0); refSig = `K${st.K}`; curveSig = ''; }
    const si = Math.min(st.sample | 0, refs.length - 1);
    const x0 = sampleOf(refs, si, st.spread);
    const s0 = page.step();
    const t = s0 ? s0.t : binT(0);
    const eps = seededRandn(st.seed | 0, PIX, { std: 1 });
    const ex = example(refs, x0, eps, st.mode, st.sched, t, st.spread);
    const { a, s } = ex;
    const gain = x0Gain(st.target, a, s);
    let wmax = 0; for (const w of ex.w) if (w > wmax) wmax = w;
    page.probe = { wmax, gain, target: st.target, t, a, s };

    // ---- panel row: x0 | eps | x_t | x0-hat | prediction | error -------------
    const pad = 18, gap = 10, top = 88;
    const pw = Math.min(104, Math.floor((W - 2 * pad - 5 * gap) / 6));
    const rowW = 6 * pw + 5 * gap, ox = pad + Math.max(0, Math.floor((W - 2 * pad - rowW) / 2));
    const predArr = st.target === 'eps' ? ex.epsH : st.target === 'v' ? ex.vHat : ex.x0h;
    const predName = st.target === 'eps' ? 'ε̂' : st.target === 'v' ? (st.mode === 'flow' ? 'û' : 'v̂') : 'x̂₀';
    const spec = [
      { key: 'x0', data: x0, ramp: 'seq', dom: [0, 1], title: 'x₀  clean', sub: 'template + spread' },
      { key: 'eps', data: eps, ramp: 'div', dom: null, title: 'ε  noise', sub: 'N(0, I)' },
      { key: 'xt', data: ex.xt, ramp: 'div', dom: null, title: 'x_t  input', sub: `a=${a.toFixed(3)} s=${s.toFixed(3)}` },
      { key: 'x0h', data: ex.x0h, ramp: 'seq', dom: [0, 1], title: 'x̂₀  model guess', sub: 'mean → texture' },
      { key: 'pred', data: predArr, ramp: st.target === 'x0' ? 'seq' : 'div', dom: st.target === 'x0' ? [0, 1] : null, title: `${predName}  prediction`, sub: `target: ${st.target === 'eps' ? 'ε' : st.target === 'v' ? 'v' : 'x₀'}` },
      { key: 'err', data: ex.err, ramp: 'seq', dom: [0, 1], title: '|x̂₀ − x₀|', sub: 'what is lost' },
    ];
    panels = [];
    spec.forEach((p, i) => {
      const rect = { x: ox + i * (pw + gap), y: top, w: pw, h: pw };
      r.heatmap(p.data, { rows: N, cols: N, rect, ramp: p.ramp === 'seq' ? ramps.sequential : ramps.diverging, domain: p.dom || 'auto' });
      frame(ctx, rect.x, rect.y, rect.w, rect.h);
      r.label(p.title, rect.x, rect.y - 20, { color: p.key === 'x0h' ? T.accent : T.n13, font: `11px ${MONO}` });
      r.label(p.sub, rect.x, rect.y - 7, { color: T.n10, font: `9px ${MONO}` });
      panels.push({ key: p.key, rect, data: p.data, title: p.title });
      // selected-pixel marker, so the arithmetic card names a visible cell
      const sr = (selPix / N) | 0, scl = selPix % N, cw = rect.w / N;
      ctx.save(); ctx.strokeStyle = T.warn; ctx.lineWidth = 1.5;
      ctx.strokeRect(rect.x + scl * cw + 0.5, rect.y + sr * cw + 0.5, cw - 1, cw - 1); ctx.restore();
    });

    // posterior weight bar under x0-hat: near-uniform weights ARE the blur.
    const hp = panels[3].rect, wy = hp.y + hp.h + 6;
    let acc = 0;
    for (let k = 0; k < ex.w.length; k++) {
      const seg = ex.w[k] * hp.w;
      ctx.save(); ctx.fillStyle = alphaOf(categorical(k), 0.85); ctx.fillRect(hp.x + acc, wy, Math.max(0, seg), 7); ctx.restore();
      acc += seg;
    }
    frame(ctx, hp.x, wy, hp.w, 7);
    r.label(`posterior over the ${refs.length} templates — top ${(wmax * 100).toFixed(0)}%`, hp.x, wy + 19, { color: T.n10, font: `9px ${MONO}` });

    // ---- the two interpolants, always both on screen -------------------------
    const eqY = top + pw + 38;
    const isFlow = st.mode === 'flow';
    r.label('x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε', ox, eqY, { color: isFlow ? T.n9 : T.accent, font: `${isFlow ? 11 : 12}px ${MONO}` });
    r.label('diffusion', ox, eqY + 13, { color: isFlow ? T.n9 : T.accent, font: `9px ${MONO}` });
    r.label('x_t = (1−t)·x₀ + t·ε   ·   target velocity = ε − x₀', ox + 232, eqY, { color: isFlow ? T.accent : T.n9, font: `${isFlow ? 12 : 11}px ${MONO}` });
    r.label('rectified flow  (what production models train)', ox + 232, eqY + 13, { color: isFlow ? T.accent : T.n9, font: `9px ${MONO}` });

    // ---- bottom row: schedule | where the loss goes | one training step ------
    const by = eqY + 26, bh = Math.max(104, H - by - 16);
    const avail = W - 2 * pad;
    const cw2 = Math.max(150, Math.floor(avail * 0.29));
    const cx1 = pad, cx2 = cx1 + cw2 + 20, cx3 = cx2 + cw2 + 20, cw3 = W - pad - cx3;

    // -- schedule panel (draggable t) --
    const sr = { x: cx1, y: by + 14, w: cw2, h: bh - 46 };
    r.label(isFlow ? `path — flow, shift ${ex.shift}` : `path — ${st.sched} ᾱ`, cx1, by + 6, { color: T.n13, font: `11px ${MONO}` });
    frame(ctx, sr.x, sr.y, sr.w, sr.h);
    panels.push({ key: 'sched', rect: sr, data: null, title: 'schedule' });
    const aS = [], sS = [], snrS = [];
    for (let i = 0; i < BINS; i++) { const cc = coeffs(st.mode, st.sched, binT(i)); aS.push(cc.a); sS.push(cc.s); snrS.push(Math.max(0, Math.min(1, (Math.log10(snrOf(cc.a, cc.s)) + 4) / 8))); }
    plot(ctx, sr, aS, T.ok, { width: 2 });
    plot(ctx, sr, sS, T.bad, { width: 2 });
    plot(ctx, sr, snrS, T.violet, { width: 1.2, dash: [3, 3] });
    const tx = sr.x + ((t * BINS - 1) / (BINS - 1)) * sr.w;
    ctx.save(); ctx.strokeStyle = T.warn; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(tx, sr.y); ctx.lineTo(tx, sr.y + sr.h); ctx.stroke();
    ctx.fillStyle = T.warn; ctx.beginPath(); ctx.arc(tx, sr.y + sr.h - a * sr.h, 3.5 + 0.8 * Math.sin((page.t || 0) * 3), 0, 7); ctx.fill(); ctx.restore();
    r.label('— a signal', sr.x, sr.y + sr.h + 12, { color: T.ok, font: `9px ${MONO}` });
    r.label('— s noise', sr.x + 66, sr.y + sr.h + 12, { color: T.bad, font: `9px ${MONO}` });
    r.label('⋯ log SNR', sr.x + 128, sr.y + sr.h + 12, { color: T.violet, font: `9px ${MONO}` });
    r.label('t: 0 ↔ 1   drag here to set the noise level', sr.x, sr.y + sr.h + 24, { color: T.accent, font: `9px ${MONO}` });

    // -- loss / weighting panel --
    const cv = curves(st, x0);
    const lr = { x: cx2, y: by + 14, w: cw2, h: bh - 46 };
    r.label(`where the loss goes — w(t) ${st.weight}`, cx2, by + 6, { color: T.n13, font: `11px ${MONO}` });
    frame(ctx, lr.x, lr.y, lr.w, lr.h);
    plot(ctx, lr, norm(cv.map((d) => d.loss)), T.teal, { width: 2 });
    plot(ctx, lr, norm(cv.map((d) => d.w)), T.gold, { width: 1.2, dash: [4, 3] });
    plot(ctx, lr, norm(cv.map((d) => d.budget)), T.violetDeep, { width: 2.4 });
    ctx.save(); ctx.strokeStyle = T.warn; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(tx - cx1 + cx2, lr.y); ctx.lineTo(tx - cx1 + cx2, lr.y + lr.h); ctx.stroke(); ctx.restore();
    r.label(`— loss(${st.target === 'eps' ? 'ε' : st.target === 'v' ? 'v' : 'x₀'})`, lr.x, lr.y + lr.h + 12, { color: T.teal, font: `9px ${MONO}` });
    r.label('⋯ w(t)', lr.x + 66, lr.y + lr.h + 12, { color: T.gold, font: `9px ${MONO}` });
    r.label('— w·loss budget', lr.x + 120, lr.y + lr.h + 12, { color: T.violetDeep, font: `9px ${MONO}` });
    r.label('t: 0 ↔ 1   normalised per curve', lr.x, lr.y + lr.h + 24, { color: T.n10, font: `9px ${MONO}` });

    // -- one training step (arithmetic card) --
    const p = selPix, pr = (p / N) | 0, pc = p % N;
    const cur = cv[Math.max(0, Math.min(BINS - 1, Math.round(t * BINS) - 1))];
    r.label('one training step', cx3, by + 6, { color: T.n13, font: `11px ${MONO}` });
    const lines = [
      [`t = ${t.toFixed(3)}   boxed pixel (${pr}, ${pc})`, T.n12],
      [isFlow ? `τ = ${ex.tau.toFixed(3)}   a = 1−τ = ${a.toFixed(3)}   s = τ = ${s.toFixed(3)}` : `ᾱ = ${ex.ab.toFixed(4)}   a = √ᾱ = ${a.toFixed(3)}   s = √(1−ᾱ) = ${s.toFixed(3)}`, T.n12],
      [`SNR = a²/s² = ${f3(snrOf(a, s))}`, T.violet],
      [`x_t = ${a.toFixed(3)}·${x0[p].toFixed(3)} + ${s.toFixed(3)}·${eps[p].toFixed(3)} = ${ex.xt[p].toFixed(3)}`, T.n13],
      [`${predName} = ${predArr[p].toFixed(3)}   vs target ${st.target === 'eps' ? eps[p].toFixed(3) : st.target === 'v' ? ex.vTrue[p].toFixed(3) : x0[p].toFixed(3)}`, T.teal],
      [`loss at this t = ${fL(cur.loss)}   ×  w(t) = ${f3(cur.w)}`, T.gold],
      [`a fixed ${st.target === 'eps' ? 'ε' : st.target === 'v' ? 'v' : 'x₀'} error is ×${gain.toFixed(2)} in image space`, gain > 3 ? T.bad : T.n13],
      ['—', T.n6],
      ['the model never sees clean data at inference:', T.n11],
      ['its own output is the next input, so these errors', T.n11],
      ['compound — that is the sampler page, not this one.', T.n11],
    ];
    lines.forEach((ln, i) => r.label(ln[0], cx3, by + 22 + i * 13, { color: ln[1], font: `${i === 3 ? 10.5 : 9.5}px ${MONO}` }));

    // ---- hover: the arithmetic that produced the pixel under the cursor ------
    if (page.pointer.over && dragPix < 0 && !dragT) {
      const pt = page.pointer;
      for (const pn of panels) {
        if (pn.key === 'sched') continue;
        const hit = cellAt(pn.rect, N, N, pt.x, pt.y);
        if (!hit) continue;
        const i = hit.r * N + hit.c;
        let tip = `${pn.title}  pixel (${hit.r}, ${hit.c})\nvalue = ${pn.data[i].toFixed(4)}\n`;
        if (pn.key === 'xt') tip += `x_t = a·x₀ + s·ε = ${a.toFixed(3)}·${x0[i].toFixed(3)} + ${s.toFixed(3)}·${eps[i].toFixed(3)}`;
        else if (pn.key === 'x0h') tip += `E[x₀ | x_t] over the ${refs.length} templates\ntop weight ${(wmax * 100).toFixed(0)}% — ${wmax > 0.9 ? 'one template explains x_t' : 'a blend: the blurry mean'}\ntemplate share s²/(a²sd²+s²) = ${((s * s) / (a * a * st.spread * st.spread + s * s)).toFixed(3)}`;
        else if (pn.key === 'pred') tip += st.target === 'eps' ? `ε̂ = (x_t − a·x̂₀)/s = (${ex.xt[i].toFixed(3)} − ${a.toFixed(3)}·${ex.x0h[i].toFixed(3)})/${s.toFixed(3)}\ntarget ε = ${eps[i].toFixed(3)}` : st.target === 'v' ? `${predName} = ${isFlow ? 'ε̂ − x̂₀' : 'a·ε̂ − s·x̂₀'} = ${predArr[i].toFixed(3)}\ntarget = ${ex.vTrue[i].toFixed(3)}` : `x̂₀ = ${ex.x0h[i].toFixed(3)}   target x₀ = ${x0[i].toFixed(3)}`;
        else if (pn.key === 'err') tip += `|x̂₀ − x₀| = |${ex.x0h[i].toFixed(3)} − ${x0[i].toFixed(3)}|`;
        else if (pn.key === 'x0') tip += `template ${refs[si][i].toFixed(3)} + spread ${(x0[i] - refs[si][i]).toFixed(3)}\ndrag ↕ to edit this sample's template`;
        else tip += 'one draw from N(0, 1) — resampled every training step';
        page.setTip(tip);
        break;
      }
    }

    // ---- readout -------------------------------------------------------------
    const modeTxt = isFlow
      ? `rectified flow: x_t = (1−τ)·x₀ + τ·ε with τ = ${ex.tau.toFixed(3)} (shift ${ex.shift}); the regression target is the velocity ε − x₀, constant along the straight path from data to noise.`
      : `diffusion: x_t = √ᾱ·x₀ + √(1−ᾱ)·ε with ᾱ = ${ex.ab.toFixed(4)} on the ${st.sched} schedule; a² + s² = 1, so x_t keeps unit variance at every t.`;
    let o = `${modeTxt}   t = ${t.toFixed(3)}, a = ${a.toFixed(3)}, s = ${s.toFixed(3)}, SNR = ${f3(snrOf(a, s))}.   tier:${r.name}\n`;
    o += `x̂₀ is the exact posterior mean E[x₀ | x_t, t] over the ${refs.length}-template training distribution — top weight ${(wmax * 100).toFixed(0)}%, so the guess is ${wmax > 0.9 ? 'one template, with its texture recovered' : wmax < 0.45 ? 'their blurry average' : 'a blend of a few samples'}. `;
    o += `Predicting ${st.target === 'eps' ? 'ε' : st.target === 'v' ? 'v' : 'x₀'}: a fixed error costs ×${gain.toFixed(2)} in image space (loss ${fL(cur.loss)}, weight ${f3(cur.w)}). `;
    o += st.target === 'eps'
      ? 'Note the inversion: predicting ε is nearly TRIVIAL at high noise, where x_t is almost all ε and a network can copy its input (watch the loss curve fall towards zero), and hardest at low noise — while the COST runs the other way, since a fixed ε-error is multiplied by s/a = 1/√SNR. Easy exactly where mistakes are most expensive. The loss weighting is what re-balances that, so the weighting you pick IS the decision about where image quality goes. '
      : 'Compare with ε-prediction: its loss falls towards zero at high noise (x_t is almost all ε — copy the input) while its image-space gain s/a = 1/√SNR blows up there, so it is easy exactly where mistakes are most expensive. The loss weighting is what re-balances that, so the weighting you pick IS the decision about where image quality goes. ';
    o += 'Integrating this into a picture — step counts and samplers — is the diffusion-sampler page; conditioning and guidance scale are the guidance page. Neither is shown here.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__diffNoisePage = page;
  const q = new URLSearchParams(location.search);
  const tr = page.controls._transport;
  for (const k of ['mode', 'sched', 'target', 'weight']) if (q.has(k)) page.controls.set(k, q.get(k));
  for (const k of ['K', 'sample', 'seed', 'spread']) if (q.has(k)) page.controls.set(k, +q.get(k));
  if (q.has('pix')) selPix = Math.max(0, Math.min(PIX - 1, parseInt(q.get('pix'), 10) || 0));
  // ?paint=i,v;i,v -- headless stand-in for dragging pixels of x₀ (no pointer
  // under --screenshot), so an edited training sample is capture-verifiable.
  if (q.has('paint')) {
    const im = refs[Math.min(page.state.sample | 0, refs.length - 1)];
    for (const pr of q.get('paint').split(';')) { const [i, v] = pr.split(',').map(Number); if (i >= 0 && i < PIX) im[i] = Math.max(0, Math.min(1, v)); }
    refSig = 'painted'; curveSig = '';
  }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  // Deterministic frame for capture: any explicit handle pauses the autoplay so
  // the transport cannot advance off the requested t before the snapshot.
  if (q.has('step') || q.has('t') || q.has('hover') || q.has('paint') || q.has('pix')) { if (tr) tr.pause(); }
  if (q.has('step') && tr) tr.seek(parseInt(q.get('step'), 10));
  if (q.has('t')) seekT(page, parseFloat(q.get('t')));
  if (q.get('play') === '1' && tr) tr.play();
  page.redraw();
});
