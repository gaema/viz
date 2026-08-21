// dit concept page -- the transformer that replaced the U-Net in image and
// video generation, and the one structural idea that made it work.
//
// A DiT patchifies the NOISY LATENT into tokens and runs an ordinary
// transformer over them. The interesting part is HOW the conditioning enters.
// A U-Net concatenated its conditioning or cross-attended to it; a DiT's
// best-performing variant instead MODULATES THE NORMALISATION: the timestep
// and class embedding are projected into per-block scale / shift / gate
// vectors applied around each sublayer (adaLN-Zero), with the GATE initialised
// to ZERO so every block starts as an exact identity and learns its way in.
//
// Everything on screen is arithmetic this file just did: the latent, the patch
// projection, the sin-cos position embedding, the timestep embedding, the adaLN
// projection, layer norm, single-head attention over every token, the MLP, and
// the two gated residual adds. Nothing is staged.
//
// Source: Peebles & Xie, "Scalable Diffusion Models with Transformers",
// https://arxiv.org/abs/2212.09748 . The MMDiT panel: https://arxiv.org/abs/2403.03206 .
//
// Companion pages own the rest of the diffusion story and are NOT re-taught
// here: diffusion-noise (the schedule), diffusion-sampler (the integrator),
// guidance (the conditioning knob). This page owns the ARCHITECTURE.

import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn, silu, gelu, softmax, layernorm } from '../framework/tensor.js';
import { T, alphaOf, signedColor, rgbaToken, inkOn } from '../framework/theme.js';

// The drawn block carries 8 features per token so every number fits on screen.
// The cost panel prices a REAL configuration from its own d slider -- the two
// are labelled apart, never conflated.
const DTOY = 8;
const CLASSES = ['tabby cat', 'volcano', 'sports car', 'sunflower'];
const STAGES = [
  { key: 'x', name: 'token in', expr: 'x' },
  { key: 'h1', name: 'modulate', expr: 'h = LN(x)·(1+γ₁) + β₁' },
  { key: 'a', name: 'attend', expr: 'a = Attention(h) over all N tokens' },
  { key: 'x2', name: 'gate + add', expr: 'x ← x + α₁·a' },
  { key: 'h2', name: 'modulate', expr: 'h = LN(x)·(1+γ₂) + β₂' },
  { key: 'm', name: 'MLP', expr: 'm = W₂·GELU(W₁h)' },
  { key: 'out', name: 'gate + add', expr: 'out ← x + α₂·m' },
];
const MODNAMES = ['γ₁ scale', 'β₁ shift', 'α₁ gate', 'γ₂ scale', 'β₂ shift', 'α₂ gate'];
const MODWHY = [
  'scales the normalised token before attention',
  'shifts the normalised token before attention',
  'GATES the attention branch — zero here means the branch contributes nothing',
  'scales the normalised token before the MLP',
  'shifts the normalised token before the MLP',
  'GATES the MLP branch — zero here means the branch contributes nothing',
];

// ---------------------------------------------------------------------------
// small linear-algebra helpers over {data, rows, cols}
// ---------------------------------------------------------------------------
const mv = (W, v) => {                      // W (rows x cols) · v (cols) -> rows
  const out = new Float32Array(W.rows);
  for (let i = 0; i < W.rows; i++) { let s = 0; for (let j = 0; j < W.cols; j++) s += W.data[i * W.cols + j] * v[j]; out[i] = s; }
  return out;
};
const scaled = (M, s) => { const d = Float32Array.from(M.data); for (let i = 0; i < d.length; i++) d[i] *= s; return { data: d, rows: M.rows, cols: M.cols }; };
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };
const norm2 = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };

// Frozen 2-D sin-cos position embedding: half the dims encode the patch row,
// half encode the patch column. This is what a DiT actually uses -- the
// position is not learned, it is a fixed function of where the patch sat.
function posEmbed2D(pi, pj, d) {
  const out = new Float32Array(d), half = d / 2;
  for (let k = 0; k < half / 2; k++) {
    const f = 1 / Math.pow(10000, (2 * k) / half);
    out[2 * k] = Math.sin(pi * f); out[2 * k + 1] = Math.cos(pi * f);
    out[half + 2 * k] = Math.sin(pj * f); out[half + 2 * k + 1] = Math.cos(pj * f);
  }
  return out;
}

// Sinusoidal timestep embedding -- the same construction transformers use for
// position, applied to the diffusion timestep instead.
function timestepEmbed(t, d) {
  const out = new Float32Array(d), half = d / 2;
  for (let k = 0; k < half; k++) {
    const f = Math.exp(-Math.log(10000) * k / half);
    out[k] = Math.sin(t * f); out[half + k] = Math.cos(t * f);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The model. Recomputed only when its inputs change (draw() runs on every
// pointer move; the attention is O(N²·d) and must not run per frame).
// ---------------------------------------------------------------------------
let M = null, sig = '';

function compute(st) {
  const d = DTOY, G = st.G | 0, p = st.p | 0;
  const side = Math.max(1, Math.floor(G / p)), N = side * side, seed = st.seed | 0;

  // --- the noisy latent, and its patchification into tokens ----------------
  const Z = seededRandn(seed * 7 + 1, [G, G]);
  const Wp = scaled(seededRandn(seed * 7 + 2, [d, p * p]), 1 / Math.sqrt(p * p));
  const X = new Float32Array(N * d), patchOf = [];
  for (let pi = 0; pi < side; pi++) for (let pj = 0; pj < side; pj++) {
    const px = new Float32Array(p * p);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) px[a * p + b] = Z.data[(pi * p + a) * G + (pj * p + b)];
    const proj = mv(Wp, px), pe = posEmbed2D(pi, pj, d), i = pi * side + pj;
    for (let k = 0; k < d; k++) X[i * d + k] = proj[k] + pe[k];
    patchOf.push({ pi, pj, px });
  }

  // --- the conditioning vector: timestep + class ---------------------------
  const tEmb = timestepEmbed(st.t, d);
  const Wt = scaled(seededRandn(seed * 7 + 3, [d, d]), 1 / Math.sqrt(d));
  const tProj = silu(mv(Wt, tEmb));
  const yTable = seededRandn(seed * 7 + 4, [CLASSES.length, d]);
  const cls = Math.max(0, Math.min(CLASSES.length - 1, st.cls | 0));
  const yEmb = new Float32Array(d);
  for (let k = 0; k < d; k++) yEmb[k] = yTable.data[cls * d + k];
  const c = new Float32Array(d);
  for (let k = 0; k < d; k++) c[k] = tProj[k] + st.cscale * yEmb[k];

  // --- adaLN: ONE linear layer turns c into six per-block vectors ----------
  // adaLN-ZERO: this layer's weight AND bias start at exactly zero, so every
  // one of the six comes out zero and the block is an identity. `init` is the
  // page's stand-in for training progress: 0 = the initialisation, 1 = trained.
  const Wa = scaled(seededRandn(seed * 7 + 5, [6 * d, d]), 1 / Math.sqrt(d));
  const ba = seededRandn(seed * 7 + 6, 6 * d);   // a NUMBER shape -> a flat vector
  const sc = silu(c), raw = mv(Wa, sc);
  const mod = new Float32Array(6 * d);
  for (let i = 0; i < 6 * d; i++) mod[i] = (raw[i] + 0.3 * ba[i]) * st.init;
  const slice = (k) => mod.subarray(k * d, k * d + d);
  const g1 = slice(0), b1 = slice(1), a1 = slice(2), g2 = slice(3), b2 = slice(4), a2 = slice(5);

  // --- the block itself ----------------------------------------------------
  const H1 = new Float32Array(N * d);
  for (let i = 0; i < N; i++) {
    const ln = layernorm(X.subarray(i * d, i * d + d));
    for (let k = 0; k < d; k++) H1[i * d + k] = ln[k] * (1 + g1[k]) + b1[k];
  }

  const Wq = scaled(seededRandn(seed * 7 + 7, [d, d]), 1 / Math.sqrt(d));
  const Wk = scaled(seededRandn(seed * 7 + 8, [d, d]), 1 / Math.sqrt(d));
  const Wv = scaled(seededRandn(seed * 7 + 9, [d, d]), 1 / Math.sqrt(d));
  const Wo = scaled(seededRandn(seed * 7 + 10, [d, d]), 1 / Math.sqrt(d));
  const Q = new Float32Array(N * d), K = new Float32Array(N * d), V = new Float32Array(N * d);
  for (let i = 0; i < N; i++) {
    const h = H1.subarray(i * d, i * d + d);
    Q.set(mv(Wq, h), i * d); K.set(mv(Wk, h), i * d); V.set(mv(Wv, h), i * d);
  }
  const A = new Float32Array(N * d), inv = 1 / Math.sqrt(d);
  const tok = Math.max(0, Math.min(N - 1, st.tok | 0));
  let attnRow = null;
  for (let i = 0; i < N; i++) {
    const logits = new Float32Array(N);
    for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < d; k++) s += Q[i * d + k] * K[j * d + k]; logits[j] = s * inv; }
    const pw = softmax(logits);
    if (i === tok) attnRow = pw;
    const ctx = new Float32Array(d);
    for (let j = 0; j < N; j++) { const w = pw[j]; for (let k = 0; k < d; k++) ctx[k] += w * V[j * d + k]; }
    A.set(mv(Wo, ctx), i * d);
  }

  const X2 = new Float32Array(N * d);
  for (let i = 0; i < N; i++) for (let k = 0; k < d; k++) X2[i * d + k] = X[i * d + k] + a1[k] * A[i * d + k];

  const H2 = new Float32Array(N * d);
  for (let i = 0; i < N; i++) {
    const ln = layernorm(X2.subarray(i * d, i * d + d));
    for (let k = 0; k < d; k++) H2[i * d + k] = ln[k] * (1 + g2[k]) + b2[k];
  }

  const W1 = scaled(seededRandn(seed * 7 + 11, [4 * d, d]), 1 / Math.sqrt(d));
  const W2 = scaled(seededRandn(seed * 7 + 12, [d, 4 * d]), 1 / Math.sqrt(4 * d));
  const MO = new Float32Array(N * d), OUT = new Float32Array(N * d), delta = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const hid = gelu(mv(W1, H2.subarray(i * d, i * d + d)));
    const m = mv(W2, hid);
    MO.set(m, i * d);
    for (let k = 0; k < d; k++) OUT[i * d + k] = X2[i * d + k] + a2[k] * m[k];
    let s = 0; for (let k = 0; k < d; k++) { const dd = OUT[i * d + k] - X[i * d + k]; s += dd * dd; }
    delta[i] = Math.sqrt(s);
  }

  const vecAt = (buf) => buf.subarray(tok * d, tok * d + d);
  return {
    d, G, p, side, N, tok, cls, Z, patchOf, X, H1, A, X2, H2, MO, OUT, delta, attnRow,
    tEmb, yEmb, c, mod, g1, b1, a1, g2, b2, a2,
    stages: { x: vecAt(X), h1: vecAt(H1), a: vecAt(A), x2: vecAt(X2), h2: vecAt(H2), m: vecAt(MO), out: vecAt(OUT) },
    maxDelta: maxAbs(delta),
  };
}

function ensure(st) {
  const s = [st.seed, st.G, st.p, st.t, st.cscale, st.init, st.cls, st.tok].join('|');
  if (s !== sig || !M) { sig = s; M = compute(st); }
  return M;
}

// ---------------------------------------------------------------------------
// The trade, computed live from the reader's settings. No FLOP number in this
// file is hard-coded: every one is arithmetic over (N, d).
// ---------------------------------------------------------------------------
function costs(N, d) {
  const attn = 4 * N * N * d;          // scores + weighted sum, 2 flops per MAC
  const proj = 8 * N * d * d;          // q, k, v and the output projection
  const mlp = 16 * N * d * d;          // d -> 4d -> d
  const ada = 12 * d * d;              // once per SAMPLE, not per token
  const total = attn + proj + mlp + ada;
  const pAttn = 4 * d * d, pMlp = 8 * d * d, pAda = 6 * d * d + 6 * d;
  return { N, d, attn, proj, mlp, ada, total, quadShare: attn / total,
           pCore: pAttn + pMlp, pAda, pTotal: pAttn + pMlp + pAda };
}
const fmtN = (v) => (v >= 1e12 ? (v / 1e12).toFixed(2) + ' T' : v >= 1e9 ? (v / 1e9).toFixed(2) + ' G' : v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' k' : String(Math.round(v)));

// ---------------------------------------------------------------------------
// drawing helpers
// ---------------------------------------------------------------------------
function panel(ctx, x, y, w, h, title) {
  ctx.save();
  ctx.fillStyle = T.n1; ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.fill(); ctx.stroke(); }
  else { ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); }
  if (title) { ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText(title, x + 7, y + 12); }
  ctx.restore();
}

// One d-wide vector as a row of signed cells. Returns the rect for hit-testing.
function vecRow(ctx, vec, x, y, w, h, dom, opts = {}) {
  const n = vec.length, cw = w / n;
  ctx.save();
  for (let k = 0; k < n; k++) {
    ctx.fillStyle = signedColor(vec[k] / (dom || 1));
    ctx.fillRect(x + k * cw, y, cw, h);
  }
  ctx.strokeStyle = alphaOf('n14', 0.16); ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k <= n; k++) { ctx.moveTo(x + k * cw, y); ctx.lineTo(x + k * cw, y + h); }
  ctx.moveTo(x, y); ctx.lineTo(x + w, y); ctx.moveTo(x, y + h); ctx.lineTo(x + w, y + h);
  ctx.stroke();
  if (opts.values && cw >= 26) {
    ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let k = 0; k < n; k++) { ctx.fillStyle = inkOn(signedColor(vec[k] / (dom || 1))); ctx.fillText(vec[k].toFixed(2), x + k * cw + cw / 2, y + h / 2); }
  }
  if (opts.dim) { ctx.fillStyle = rgbaToken('n0', 0.55); ctx.fillRect(x, y, w, h); }
  ctx.restore();
  return { x, y, w, h };
}

// A draggable horizontal track. Returns its rect so onPointer can hit-test it.
function track(ctx, x, y, w, frac, label, value, col) {
  ctx.save();
  ctx.fillStyle = T.n3; ctx.fillRect(x, y + 7, w, 5);
  ctx.fillStyle = col; ctx.fillRect(x, y + 7, w * frac, 5);
  const hx = x + w * frac;
  ctx.beginPath(); ctx.arc(hx, y + 9.5, 5.5, 0, 6.2832); ctx.fillStyle = col; ctx.fill();
  ctx.strokeStyle = T.n0; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = T.n11; ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(label, x, y + 2);
  ctx.textAlign = 'right'; ctx.fillStyle = T.n13; ctx.fillText(value, x + w, y + 2);
  ctx.restore();
  return { x, y, w, h: 19 };
}

function arrowRight(ctx, x, y, len, col, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha == null ? 1 : alpha;
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + len - 5, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x + len, y); ctx.lineTo(x + len - 6, y - 3.5); ctx.lineTo(x + len - 6, y + 3.5); ctx.closePath(); ctx.fill();
  ctx.restore();
}

// ---------------------------------------------------------------------------
// hit-test rects, captured in draw() and read by onPointer()
// ---------------------------------------------------------------------------
let R = {};
let grab = null;   // 't' | 'cscale' while a canvas track is being dragged

mount({
  mount: 'body',
  title: 'DiT — a transformer over latent patches, conditioned by modulating the norm',
  blurb: 'A DiT chops the noisy LATENT into patches, embeds each as a token, and runs a plain transformer over them. What replaces the U-Net\'s concatenation / cross-attention is how the conditioning gets in: the timestep and class embedding go through ONE linear layer that emits six per-block vectors — scale, shift and gate around each sublayer (adaLN-Zero). Drag the timestep and the conditioning strength on the canvas and watch all six move. Then drag "training progress" to 0: that is the initialisation, both gates are exactly zero, and the whole block collapses to an identity — which is what makes very deep DiTs trainable. Patch size is the cost dial: halving it quadruples the tokens and multiplies the quadratic term by sixteen, priced live below.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  autoplay: true,
  compare: { key: 'init', a: 0, b: 1, labelA: 'init: gates = 0 (block ≡ identity)', labelB: 'trained: gates open' },
  controls: (c, page) => {
    c.stepper('G', { label: 'latent grid (G×G)', min: 8, max: 16, step: 8, value: 16 });
    c.select('p', { label: 'patch size p', value: '2', options: [{ value: '1', label: '1  (finest, most tokens)' }, { value: '2', label: '2' }, { value: '4', label: '4' }, { value: '8', label: '8  (coarsest)' }], onInput: (v, st) => { const side = Math.max(1, Math.floor(st.G / +v)); if (st.tok >= side * side) c.set('tok', 0, { silent: true }); } });
    c.slider('t', { label: 'timestep t (noise level)', min: 0, max: 1000, step: 1, value: 700, format: (v) => String(Math.round(v)) });
    c.select('cls', { label: 'class condition', value: '0', options: CLASSES.map((n, i) => ({ value: String(i), label: n })) });
    c.slider('cscale', { label: 'conditioning strength', min: 0, max: 3, step: 0.05, value: 1 });
    c.slider('init', { label: 'training progress (0 = adaLN-Zero init)', min: 0, max: 1, step: 0.01, value: 1 });
    c.button('⤓ zero the gate (initialisation)', () => c.set('init', 0));
    c.stepper('tok', { label: 'tracked token', min: 0, max: 255, value: 0 });
    c.stepper('seed', { label: 'seed', min: 1, max: 40, value: 3 });
    c.slider('dm', { label: 'cost panel: real d_model', min: 128, max: 2048, step: 64, value: 1152, format: (v) => String(Math.round(v)) });
    c.transport({ compute: () => STAGES.map((s, i) => ({ i, label: `${i + 1}. ${s.name} — ${s.expr}` })), speed: 0.9, loop: true });
  },

  // Direct manipulation: drag the timestep track or the conditioning-strength
  // track on the canvas and every one of the six modulation vectors moves under
  // your hand. Click a patch to follow a different token through the block.
  onPointer: (page, ev) => {
    const setFromTrack = (r, key, min, max, step) => {
      const f = Math.max(0, Math.min(1, (ev.x - r.x) / r.w));
      let v = min + f * (max - min);
      if (step) v = Math.round(v / step) * step;
      page.controls.set(key, +v.toFixed(4));
    };
    if (ev.type === 'down') {
      grab = null;
      const inR = (r) => r && ev.x >= r.x - 6 && ev.x <= r.x + r.w + 6 && ev.y >= r.y && ev.y <= r.y + r.h;
      if (inR(R.tTrack)) { grab = 't'; setFromTrack(R.tTrack, 't', 0, 1000, 1); }
      else if (inR(R.cTrack)) { grab = 'cscale'; setFromTrack(R.cTrack, 'cscale', 0, 3, 0.05); }
      else if (inR(R.iTrack)) { grab = 'init'; setFromTrack(R.iTrack, 'init', 0, 1, 0.01); }
      else if (M && R.latent) {
        const h = cellAt(R.latent, M.side, M.side, ev.x, ev.y);
        if (h) page.controls.set('tok', h.r * M.side + h.c);
      }
    } else if (ev.type === 'up' || ev.type === 'leave') { grab = null; }
    else if (ev.type === 'move' && grab && page.pointer.down) {
      if (grab === 't') setFromTrack(R.tTrack, 't', 0, 1000, 1);
      else if (grab === 'cscale') setFromTrack(R.cTrack, 'cscale', 0, 3, 0.05);
      else setFromTrack(R.iTrack, 'init', 0, 1, 0.01);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    const m = ensure(st);
    const { d, G, p, side, N, tok } = m;
    const zeroed = st.init === 0;
    R = {};

    const PAD = 10;
    // ---------------- row 1: latent -> tokens | conditioning | modulation ---
    const r1y = 22, r1h = Math.max(150, H * 0.315);

    // (a) the latent, cut into patches
    const gW = Math.min(r1h - 54, 132);
    panel(ctx, PAD, r1y, gW + 76, r1h, 'noisy latent → patches → tokens');
    const gx = PAD + 9, gy = r1y + 22;
    r.heatmap(m.Z, { rows: G, cols: G, rect: { x: gx, y: gy, w: gW, h: gW }, ramp: ramps.diverging });
    R.latent = { x: gx, y: gy, w: gW, h: gW };
    const pc = gW / side;
    ctx.save();
    ctx.strokeStyle = alphaOf('n14', 0.45); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k <= side; k++) { ctx.moveTo(gx + k * pc, gy); ctx.lineTo(gx + k * pc, gy + gW); ctx.moveTo(gx, gy + k * pc); ctx.lineTo(gx + gW, gy + k * pc); }
    ctx.stroke();
    const tr = Math.floor(tok / side), tcv = tok % side;
    ctx.strokeStyle = T.accent; ctx.lineWidth = 2.5;
    ctx.strokeRect(gx + tcv * pc - 1, gy + tr * pc - 1, pc + 2, pc + 2);
    ctx.restore();
    r.label(`${G}×${G} latent · p=${p} → N = ${N} tokens`, gx, gy + gW + 13, { color: T.n11, font: '9.5px ui-monospace, monospace' });
    r.label('click a patch to follow it', gx, gy + gW + 25, { color: T.n9, font: '9px ui-monospace, monospace' });

    // (b) how much this block CHANGED each token -- flat when the gates are zero
    const dW = 52, dx = gx + gW + 10;
    r.heatmap(m.delta, { rows: side, cols: side, rect: { x: dx, y: gy, w: dW, h: dW }, ramp: ramps.sequential, domain: [0, Math.max(1e-6, m.maxDelta)] });
    R.delta = { x: dx, y: gy, w: dW, h: dW };
    ctx.save(); ctx.strokeStyle = alphaOf('n14', 0.2); ctx.strokeRect(dx, gy, dW, dW); ctx.restore();
    r.label('‖Δ‖/token', dx, gy + dW + 11, { color: T.n11, font: '9px ui-monospace, monospace' });
    r.label(zeroed ? 'all 0' : `max ${m.maxDelta.toFixed(2)}`, dx, gy + dW + 22, { color: zeroed ? T.ok : T.n9, font: '9px ui-monospace, monospace' });

    // (c) conditioning: timestep + class -> c
    const cx = PAD + gW + 88, cW = 150;
    panel(ctx, cx, r1y, cW, r1h, 'conditioning c');
    R.tTrack = track(ctx, cx + 9, r1y + 24, cW - 18, st.t / 1000, 'timestep t ↔', String(Math.round(st.t)), T.violet);
    R.cTrack = track(ctx, cx + 9, r1y + 50, cW - 18, st.cscale / 3, 'cond. strength ↔', st.cscale.toFixed(2), T.teal);
    R.iTrack = track(ctx, cx + 9, r1y + 76, cW - 18, st.init, 'train progress ↔', st.init.toFixed(2), zeroed ? T.warn : T.gold);
    r.label('t-embed (sin-cos)', cx + 9, r1y + 100, { color: T.n11, font: '9px ui-monospace, monospace' });
    R.tEmb = vecRow(ctx, m.tEmb, cx + 9, r1y + 104, cW - 18, 12, 1);
    r.label(`class “${CLASSES[m.cls]}” × ${st.cscale.toFixed(2)}`, cx + 9, r1y + 128, { color: T.n11, font: '9px ui-monospace, monospace' });
    const yScaled = Float32Array.from(m.yEmb, (v) => v * st.cscale);
    R.yEmb = vecRow(ctx, yScaled, cx + 9, r1y + 132, cW - 18, 12, Math.max(1, maxAbs(yScaled)));
    r.label('c = t-embed + class', cx + 9, r1y + 156, { color: T.violetDeep, font: '9px ui-monospace, monospace' });
    R.cVec = vecRow(ctx, m.c, cx + 9, r1y + 160, cW - 18, 13, Math.max(1, maxAbs(m.c)));

    // (d) the six modulation vectors adaLN emits
    const mx = cx + cW + 12, mW = Math.max(160, W - mx - PAD);
    panel(ctx, mx, r1y, mW, r1h, 'adaLN:  SiLU(c) → one Linear → six per-block vectors');
    const labW = 62, sW = Math.min(mW - labW - 22, 40 * d), sx = mx + labW + 8;
    const rowH = Math.min(17, (r1h - 46) / 6), gap = (r1h - 40 - rowH * 6) / 5;
    const modV = [m.g1, m.b1, m.a1, m.g2, m.b2, m.a2];
    const modDom = Math.max(0.35, maxAbs(m.mod));
    R.mod = [];
    for (let k = 0; k < 6; k++) {
      const y = r1y + 24 + k * (rowH + gap);
      const isGate = k === 2 || k === 5;
      r.label(MODNAMES[k], mx + 8, y + rowH / 2 + 3, { color: isGate ? (zeroed ? T.warn : T.goldDeep) : T.n12, font: '10px ui-monospace, monospace' });
      R.mod.push({ ...vecRow(ctx, modV[k], sx, y, sW, rowH, modDom, { values: true }), k });
      if (isGate) { ctx.save(); ctx.strokeStyle = zeroed ? T.warn : T.goldLine; ctx.lineWidth = 1.5; ctx.strokeRect(sx - 2, y - 2, sW + 4, rowH + 4); ctx.restore(); }
    }
    if (zeroed) r.label('every one is EXACTLY zero — this is what adaLN-Zero initialises to', sx, r1y + r1h - 6, { color: T.warn, font: '9.5px ui-monospace, monospace' });
    else r.label(`drag t or the conditioning strength — all six move · ‖mod‖ = ${norm2(m.mod).toFixed(2)}`, sx, r1y + r1h - 6, { color: T.n9, font: '9.5px ui-monospace, monospace' });

    // ---------------- row 2: one block, stage by stage ----------------------
    const r2y = r1y + r1h + 16, r2h = Math.max(128, H * 0.25);
    panel(ctx, PAD, r2y, W - 2 * PAD, r2h, `one DiT block, applied to token ${tok}  (${d} of d features drawn)`);
    const s = page.step(), si = s ? s.i : STAGES.length - 1;
    const nS = STAGES.length, bw = (W - 2 * PAD - 22 - 12 * (nS - 1)) / nS;
    const by = r2y + 50, bh = 20;
    R.stage = [];
    const centers = [];
    for (let k = 0; k < nS; k++) {
      const x = PAD + 11 + k * (bw + 12), stg = STAGES[k], vec = m.stages[stg.key];
      const on = k <= si;
      const isGate = stg.key === 'x2' || stg.key === 'out';
      const isMod = stg.key === 'h1' || stg.key === 'h2';
      const col = isGate ? (zeroed ? T.warn : T.goldDeep) : isMod ? T.violet : k === 0 ? T.n9 : T.teal;
      ctx.save();
      ctx.globalAlpha = on ? 1 : 0.32;
      ctx.fillStyle = k === si ? alphaOf(col, 0.18) : T.n2;
      ctx.strokeStyle = k === si ? col : T.n5; ctx.lineWidth = k === si ? 2 : 1;
      ctx.fillRect(x, by, bw, bh); ctx.strokeRect(x, by, bw, bh);
      ctx.fillStyle = k === si ? col : T.n12; ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(stg.name, x + bw / 2, by + bh / 2);
      ctx.restore();
      const rect = vecRow(ctx, vec, x, by + bh + 8, bw, 22, Math.max(0.6, maxAbs(m.stages.x)), { dim: !on });
      R.stage.push({ ...rect, k });
      ctx.save();
      ctx.globalAlpha = on ? 1 : 0.32;
      ctx.fillStyle = T.n10; ctx.font = '8.5px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText(`‖·‖ ${norm2(vec).toFixed(2)}`, x + bw / 2, by + bh + 42);
      ctx.restore();
      centers.push(x + bw / 2);
      if (k < nS - 1) arrowRight(ctx, x + bw + 1, by + bh / 2, 10, T.n8, on ? 1 : 0.35);
    }
    // the two residual arcs: what survives when the gate is zero
    ctx.save();
    ctx.strokeStyle = zeroed ? T.ok : T.okDeep; ctx.lineWidth = zeroed ? 2.4 : 1.4;
    ctx.setLineDash(zeroed ? [] : [4, 3]);
    for (const [from, to] of [[0, 3], [3, 6]]) {
      ctx.beginPath();
      ctx.moveTo(centers[from], by - 3);
      ctx.bezierCurveTo(centers[from], by - 20, centers[to], by - 20, centers[to], by - 3);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = zeroed ? T.ok : T.okDeep; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('residual — the ONLY path when the gate is 0', (centers[0] + centers[6]) / 2, by - 24);
    ctx.restore();

    // ---------------- row 3: the two trades, and MMDiT ----------------------
    const r3y = r2y + r2h + 14, r3h = Math.max(96, H - r3y - PAD);
    const cw3 = (W - 2 * PAD - 12) * 0.56;
    panel(ctx, PAD, r3y, cw3, r3h, 'the trade: patch size is the whole cost story');
    const dm = Math.round(st.dm), C = costs(N, dm), Ch = costs(N * 4, dm), Cl = costs(Math.max(1, N / 4), dm);
    const lines = [
      [`N = (G/p)² = (${G}/${p})² = ${N} tokens`, T.n13],
      [`N² = ${fmtN(N * N)} attention pairs · 4·N²·d = ${fmtN(C.attn)} FLOP`, T.n12],
      [`block @ d=${dm}: ${fmtN(C.total)} FLOP — its N² term is ${(C.quadShare * 100).toFixed(1)}% of that`, T.n12],
      [`p → p/2:  N ×4 = ${N * 4},  N² term ×16 = ${fmtN(Ch.attn)},`, T.warnDeep],
      [`          block ${fmtN(C.total)} → ${fmtN(Ch.total)} = ${(Ch.total / C.total * 100).toFixed(0)}% of here. Finer detail,`, T.warnDeep],
      [`          quadratically dearer. That is the whole dial.`, T.warnDeep],
      [`p → 2p:  N = ${Math.round(N / 4)}, block → ${fmtN(Cl.total)} = ${(Cl.total / C.total * 100).toFixed(0)}% of here`, T.n11],
      ['', T.n11],
      [`adaLN is not free either: 6·d² + 6·d = ${fmtN(C.pAda)} params/block,`, T.n13],
      [`vs attention + MLP 12·d² = ${fmtN(C.pCore)} — so adaLN is`, T.goldDeep],
      [`${(C.pAda / C.pTotal * 100).toFixed(1)}% of the block's parameters. Its FLOPs are tiny`, T.goldDeep],
      [`(${fmtN(C.ada)}, ${(C.ada / C.total * 100).toFixed(3)}% of the block): once per SAMPLE, not per token.`, T.n11],
    ];
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    for (let k = 0; k < lines.length; k++) { ctx.fillStyle = lines[k][1]; ctx.fillText(lines[k][0], PAD + 9, r3y + 26 + k * 11.6); }
    ctx.restore();

    // MMDiT: one comparative panel, not the whole page.
    const qx = PAD + cw3 + 12, qw = W - qx - PAD;
    panel(ctx, qx, r3y, qw, r3h, 'MMDiT (Stable Diffusion 3) — one comparative panel');
    ctx.save();
    ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    const laneY = r3y + 34, lw = qw - 20;
    const box = (x, y, w, h, txt, col) => {
      ctx.fillStyle = alphaOf(col, 0.15); ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.fillText(txt, x + w / 2, y + h / 2); ctx.textAlign = 'left';
    };
    ctx.fillStyle = T.n11; ctx.fillText('DiT', qx + 9, laneY + 7);
    box(qx + 38, laneY, lw * 0.34, 14, 'image tokens', T.accent);
    arrowRight(ctx, qx + 40 + lw * 0.34, laneY + 7, 12, T.n8);
    box(qx + 54 + lw * 0.34, laneY, lw * 0.4, 14, 'attention + MLP', T.teal);
    ctx.fillStyle = T.violet; ctx.fillText('↑ text is POOLED into c: it enters only as modulation', qx + 38, laneY + 24);

    const l2 = laneY + 42;
    ctx.fillStyle = T.n11; ctx.fillText('MMDiT', qx + 9, l2 + 14);
    box(qx + 46, l2, lw * 0.3, 13, 'image tokens', T.accent);
    box(qx + 46, l2 + 17, lw * 0.3, 13, 'text tokens', T.violet);
    arrowRight(ctx, qx + 48 + lw * 0.3, l2 + 7, 10, T.n8);
    arrowRight(ctx, qx + 48 + lw * 0.3, l2 + 24, 10, T.n8);
    box(qx + 60 + lw * 0.3, l2, lw * 0.42, 30, 'JOINT attention', T.tealDeep);
    ctx.fillStyle = T.n11;
    const mmLines = [
      'Same modulation idea, two streams. Each modality keeps',
      'its OWN qkv / MLP / adaLN weights (≈2× the block params)',
      '— but the tokens are concatenated for ONE attention, so',
      'text and image attend to each other directly rather than',
      'through a pooled vector. Attention pairs are then',
      `(N_img + N_txt)², not the ${fmtN(N * N)} above.`,
    ];
    for (let k = 0; k < mmLines.length; k++) ctx.fillText(mmLines[k], qx + 9, l2 + 44 + k * 10.5);
    ctx.restore();

    // ---------------- hover-to-inspect --------------------------------------
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      let tip = null;
      const inR = (rr) => rr && pt.x >= rr.x && pt.x <= rr.x + rr.w && pt.y >= rr.y && pt.y <= rr.y + rr.h;
      const cellIdx = (rr) => Math.max(0, Math.min(d - 1, Math.floor((pt.x - rr.x) / (rr.w / d))));
      const lh = R.latent && cellAt(R.latent, side, side, pt.x, pt.y);
      const dh = R.delta && cellAt(R.delta, side, side, pt.x, pt.y);
      if (lh) {
        const i = lh.r * side + lh.c, pinfo = m.patchOf[i];
        tip = `patch (${pinfo.pi},${pinfo.pj}) → token ${i}\n${p}×${p} latent cells → ${d}-dim token\nx[0] = ${m.X[i * d].toFixed(3)}  (projection + sin-cos pos-embed)\nclick to follow this token`;
      } else if (dh) {
        const i = dh.r * side + dh.c;
        tip = `token ${i}: ‖out − in‖ = ${m.delta[i].toFixed(4)}\n${zeroed ? 'zero — the block is an identity here' : 'how much this one block moved the token'}`;
      } else {
        for (const rr of R.mod || []) if (inR(rr)) {
          const k = cellIdx(rr), v = modV[rr.k][k];
          tip = `${MODNAMES[rr.k]}[${k}] = ${v.toFixed(4)}\nfrom: row ${rr.k * d + k} of the block's adaLN Linear,\napplied to SiLU(c) where c = t-embed(t=${Math.round(st.t)}) + ${st.cscale.toFixed(2)}·class\n${MODWHY[rr.k]}${st.init < 1 ? `\n× training progress ${st.init.toFixed(2)}` : ''}`;
        }
        if (!tip) for (const rr of R.stage || []) if (inR(rr)) {
          const k = cellIdx(rr), stg = STAGES[rr.k], v = m.stages[stg.key][k];
          tip = `${stg.name}: ${stg.expr}\ncomponent ${k} of token ${tok} = ${v.toFixed(4)}\n‖vector‖ = ${norm2(m.stages[stg.key]).toFixed(3)}`;
        }
        if (!tip && inR(R.cVec)) { const k = cellIdx(R.cVec); tip = `c[${k}] = ${m.c[k].toFixed(4)}\n= SiLU-projected t-embed[${k}] (${m.tEmb[k].toFixed(3)})\n  + ${st.cscale.toFixed(2)} × class-embed[${k}] (${m.yEmb[k].toFixed(3)})`; }
        if (!tip && inR(R.tEmb)) { const k = cellIdx(R.tEmb); tip = `t-embed[${k}] = ${m.tEmb[k].toFixed(4)}\nsinusoid of t = ${Math.round(st.t)} at frequency band ${k % (d / 2)}`; }
        if (!tip && inR(R.yEmb)) { const k = cellIdx(R.yEmb); tip = `class-embed[${k}] × strength = ${(m.yEmb[k] * st.cscale).toFixed(4)}\nclass “${CLASSES[m.cls]}”, raw ${m.yEmb[k].toFixed(3)} × ${st.cscale.toFixed(2)}`; }
        if (!tip && inR(R.tTrack)) tip = `timestep t = ${Math.round(st.t)}\ndrag ↔ — every modulation vector is a function of it`;
        if (!tip && inR(R.cTrack)) tip = `conditioning strength = ${st.cscale.toFixed(2)}\ndrag ↔ — scales the class embedding inside c`;
        if (!tip && inR(R.iTrack)) tip = `training progress = ${st.init.toFixed(2)}\ndrag ↔ to 0 for the adaLN-Zero initialisation`;
      }
      if (tip) page.setTip(tip);
    }

    // probes for challenge mode
    page.probe = { N, init: st.init, maxDelta: m.maxDelta, quadShare: C.quadShare, p };

    // ---------------- readout ------------------------------------------------
    const stg = STAGES[si] || STAGES[STAGES.length - 1];
    let o = `${s ? `stage ${si + 1}/${nS}` : '(whole block)'}  ${stg.name}:  ${stg.expr}    token ${tok} of ${N}    tier:${r.name}\n`;
    o += `adaLN from c = t-embed(${Math.round(st.t)}) + ${st.cscale.toFixed(2)}·“${CLASSES[m.cls]}”  →  `;
    o += `α₁[0]=${m.a1[0].toFixed(3)}  α₂[0]=${m.a2[0].toFixed(3)}  γ₁[0]=${m.g1[0].toFixed(3)}  β₁[0]=${m.b1[0].toFixed(3)}\n`;
    o += zeroed
      ? `GATES ARE ZERO: out = x + 0·a + 0·m = x exactly. max‖out−in‖ over all ${N} tokens = ${m.maxDelta.toFixed(6)}. Every block is an identity at initialisation, so a 28-block DiT starts as a clean residual pass-through and each block learns its way in.\n`
      : `gates open: max‖out−in‖ over all ${N} tokens = ${m.maxDelta.toFixed(4)}. Drag "training progress" to 0 to watch this collapse to exactly 0.\n`;
    o += `cost @ d_model=${dm}: N=${N}, N²=${fmtN(N * N)} attention pairs, ${fmtN(C.total)} FLOP/block — quadratic term ${(C.quadShare * 100).toFixed(1)}% of it; adaLN ${(C.pAda / C.pTotal * 100).toFixed(1)}% of the block's parameters.`;
    page.setReadout(o);
  },

  challenges: [
    {
      goal: 'Make the block an exact identity — ‖out − in‖ = 0 for every token.',
      hint: 'The gate is what multiplies each branch before it is added back.',
      check: (api) => ({ solved: api.probe.maxDelta === 0, detail: `max‖out−in‖ = ${(api.probe.maxDelta || 0).toFixed(4)} — not zero yet` }),
    },
    {
      goal: 'Push attention\'s quadratic N² term past 10% of the block\'s FLOPs.',
      hint: 'The N² term grows with the square of the token count, and p is what sets the token count.',
      check: (api) => ({ solved: api.probe.quadShare > 0.10, detail: `quadratic term is ${((api.probe.quadShare || 0) * 100).toFixed(1)}% — smaller patches, or a smaller d_model` }),
    },
  ],
}).then((page) => {
  window.__ditPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k, cast) => { if (q.has(k)) page.controls.set(k, cast(q.get(k))); };
  // Every control is restorable, so one URL reproduces one exact frame.
  num('G', (v) => parseInt(v, 10));
  if (q.has('p')) page.controls.set('p', q.get('p'));
  if (q.has('cls')) page.controls.set('cls', q.get('cls'));
  num('t', parseFloat); num('cscale', parseFloat); num('init', parseFloat);
  num('tok', (v) => parseInt(v, 10)); num('seed', (v) => parseInt(v, 10)); num('dm', parseFloat);
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover, since
  // --screenshot has no pointer). ?drag=t,850 / ?drag=cscale,2.4 are the
  // headless stand-ins for the two canvas drags.
  if (q.has('drag')) {
    const [key, val] = q.get('drag').split(',');
    if (key === 't' || key === 'cscale' || key === 'init') page.controls.set(key, parseFloat(val));
  }
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause for any of these hooks.
  if (q.has('step') || q.has('hover') || q.has('drag')) {
    // Pause on a REAL stage, not on index -1: the transport's "(start)" state and
    // the readout's fallback-to-last would otherwise disagree in the capture.
    if (t) { t.pause(); if (t.index < 0 && t.steps.length) t.seek(t.steps.length - 1); }
  }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
