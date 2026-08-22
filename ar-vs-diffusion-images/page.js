// ar-vs-diffusion-images -- two genuinely different ways to make an image, run
// side by side on ONE target under a SHARED compute budget.
//
// LEFT  (autoregressive): the image is a sequence of DISCRETE TOKENS. A small
//   codebook of 2x2 patch prototypes is built in-page from the target (that is
//   what a VQGAN-style tokenizer gives you), the target becomes an 8x8 grid of
//   token ids, and the tokens are then predicted ONE AT A TIME from the ones
//   already placed -- exactly the loop a language model runs over text. The
//   candidate distribution over the codebook for the token being placed right
//   now is drawn under the panel.
//     order = raster   : left-to-right, top-to-bottom. 64 sequential passes.
//     order = scale    : next-SCALE prediction -- the whole 1x1 map, then 2x2,
//                        then 4x4, then 8x8, each scale conditioned on the
//                        upsampled reconstruction of the previous one. 4
//                        sequential passes, 85 token placements.
//   Next-scale prediction is Tian et al., "Visual Autoregressive Modeling",
//   https://arxiv.org/abs/2404.02905 ; the discrete-token image representation
//   is Esser et al., "Taming Transformers", https://arxiv.org/abs/2012.09841 .
//
// RIGHT (diffusion): the WHOLE canvas is refined at once, N times. Every region
//   is updated on every step; there is no ordering and no token. This page
//   deliberately does NOT re-teach the noise schedule or the sampler -- the
//   diffusion-noise page owns the forward/noising side and the
//   diffusion-sampler page owns the integration loop. Here the loop is reduced
//   to the one property being compared: cost = steps x a full-canvas pass.
//
// THE COMPARISON, and why it is not settled. The two methods do not pay in the
// same currency, so the page reports BOTH axes and lets them disagree:
//     sequential passes  -- what you cannot parallelise away (latency-shaped)
//     token-updates      -- total positions touched (work-shaped)
//   Raster AR spends 64 passes but only 64 token-updates. Diffusion at N steps
//   spends N passes and N x 64 token-updates. Scale-ordered AR spends 4 passes
//   and 85 token-updates. Which of those two numbers is "the cost" depends on
//   your batch size, your memory bandwidth, and whether you already run a
//   language model's serving stack (KV cache, continuous batching) that an AR
//   image model drops straight into. Empirically diffusion has historically led
//   on fine texture and photorealism at a matched budget, and AR gives exact
//   likelihoods and composes with a language model's own token stream. Both
//   claims are live research, cited below -- this page teaches the MECHANISMS
//   and the trade, and the toy quality numbers on screen are NOT evidence about
//   real models.
//
// Everything is computed in-page from a 16x16 synthetic image: the codebook is
// a real (deterministic, farthest-point-seeded) k-means over the target's 64
// patches, the AR predictor is a real nearest-codebook match on the decoded
// context, and the diffusion side is a real iterative refinement toward a
// progressively less low-passed target. No model is fetched and no result here
// transfers to a trained one.
//
// Interactions: transport steps BOTH processes together (autoplay + loop);
// DRAG the shared budget bar, DRAG the diffusion step handle, CLICK/DRAG the
// AR order strip -- all on the canvas, all mirrored into the widgets; HOVER any
// token or any region for what produced it; a "same budget" toggle that
// re-allocates the shared budget fairly; URL hooks (?step, ?budget, ?order,
// ?fair, ?dsteps, ?seed, ?temp, ?hover) so one URL replays one exact run.
//
// Sources: Tian, Jiang, Yuan, Peng, Wang, "Visual Autoregressive Modeling:
// Scalable Image Generation via Next-Scale Prediction",
// https://arxiv.org/abs/2404.02905 ; Esser, Rombach, Ommer, "Taming
// Transformers for High-Resolution Image Synthesis",
// https://arxiv.org/abs/2012.09841 ; van den Oord, Vinyals, Kavukcuoglu,
// "Neural Discrete Representation Learning", https://arxiv.org/abs/1711.00937 ;
// Dhariwal, Nichol, "Diffusion Models Beat GANs on Image Synthesis",
// https://arxiv.org/abs/2105.05233 ; Yu et al., "Language Model Beats Diffusion
// -- Tokenizer is Key to Visual Generation", https://arxiv.org/abs/2310.05737 .
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbOf } from '../framework/theme.js';
import { seededRandn } from '../framework/tensor.js';

// ------------------------------------------------------------------ shapes --
const GRID = 8;                    // token grid is GRID x GRID
const PATCH = 2;                   // one token decodes to a PATCH x PATCH block
const IMG = GRID * PATCH;          // 16 x 16 pixels
const CH = 3;                      // 3-channel latent (NOT sRGB -- see pxRGB)
const K = 12;                      // codebook entries
const PDIM = PATCH * PATCH * CH;   // 12 dims per patch prototype
const SCALES = [1, 2, 4, 8];       // next-scale ladder (VAR-style)
const AR_UPDATES = { raster: GRID * GRID, scale: SCALES.reduce((a, s) => a + s * s, 0) };
const AR_PASSES = { raster: GRID * GRID, scale: SCALES.length };

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// ------------------------------------------------------- the target image ---
// A synthetic scene chosen so the two mechanisms have something to disagree
// about: a smooth gradient (easy for both), a disc and a bar that are LONG-RANGE
// structure (a raster scan meets the bar's two ends 8 rows apart), and a
// high-frequency checker patch (fine texture -- the thing a coarse run loses).
function makeTarget(seed) {
  const n = seededRandn(seed * 13 + 7, IMG * IMG, { std: 1 });
  const out = new Float32Array(IMG * IMG * CH);
  for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
    const u = x / (IMG - 1), v = y / (IMG - 1);
    let c0 = 0.22 + 0.42 * v, c1 = 0.20 + 0.45 * (1 - u), c2 = 0.18 + 0.25 * u * v;
    const d = Math.hypot(u - 0.34, v - 0.36);
    if (d < 0.25) { const k = 1 - d / 0.25; c0 = 0.88 * k + 0.2; c1 = 0.22; c2 = 0.30 + 0.45 * k; }
    if (v > 0.76 && v < 0.90) { c0 = 0.14; c1 = 0.82; c2 = 0.46; }          // the bar
    if (u > 0.60 && v < 0.46) { const q = ((x + y) & 1) ? 0.16 : -0.16; c0 += q; c1 -= q; c2 += q; }
    const e = n[y * IMG + x] * 0.03;
    const i = (y * IMG + x) * CH;
    out[i] = clamp01(c0 + e); out[i + 1] = clamp01(c1 + e); out[i + 2] = clamp01(c2 - e);
  }
  return out;
}

// ---------------------------------------------------------- image helpers ---
// A square image is {d: Float32Array(n*n*CH), n}. All three helpers are exact
// arithmetic on that -- no canvas filters, so the numbers the tooltips print are
// the numbers the page computed.
const img = (n) => ({ d: new Float32Array(n * n * CH), n });

function downsample(src, n) {                       // box average, src.n % n === 0
  const f = src.n / n, out = img(n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < CH; c++) {
    let s = 0;
    for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) s += src.d[((y * f + j) * src.n + (x * f + i)) * CH + c];
    out.d[(y * n + x) * CH + c] = s / (f * f);
  }
  return out;
}

function upsample(src, n) {                          // bilinear, clamped edges
  const out = img(n), sc = src.n / n;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const sy = (y + 0.5) * sc - 0.5, sx = (x + 0.5) * sc - 0.5;
    const y0 = Math.floor(sy), x0 = Math.floor(sx), fy = sy - y0, fx = sx - x0;
    for (let c = 0; c < CH; c++) {
      const at = (yy, xx) => src.d[(clamp(yy, 0, src.n - 1) * src.n + clamp(xx, 0, src.n - 1)) * CH + c];
      const a = at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx;
      const b = at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx;
      out.d[(y * n + x) * CH + c] = a * (1 - fy) + b * fy;
    }
  }
  return out;
}

function lowpass(src, radius) {                      // separable box blur, twice
  if (radius < 0.5) return { d: Float32Array.from(src.d), n: src.n };
  const r = Math.round(radius), n = src.n;
  let cur = Float32Array.from(src.d);
  for (let pass = 0; pass < 2; pass++) {
    const tmp = new Float32Array(cur.length);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < CH; c++) {
      let s = 0, m = 0;
      for (let k = -r; k <= r; k++) { const xx = clamp(x + k, 0, n - 1); s += cur[(y * n + xx) * CH + c]; m++; }
      tmp[(y * n + x) * CH + c] = s / m;
    }
    const tmp2 = new Float32Array(cur.length);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < CH; c++) {
      let s = 0, m = 0;
      for (let k = -r; k <= r; k++) { const yy = clamp(y + k, 0, n - 1); s += tmp[(yy * n + x) * CH + c]; m++; }
      tmp2[(y * n + x) * CH + c] = s / m;
    }
    cur = tmp2;
  }
  return { d: cur, n };
}

const meanAbs = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };

// ------------------------------------------------------------- the codebook -
// Patches of an image, row-major by token cell.
function patchesOf(im) {
  const g = im.n / PATCH, out = [];
  for (let r = 0; r < g; r++) for (let c = 0; c < g; c++) {
    const p = new Float32Array(PDIM);
    for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) for (let ch = 0; ch < CH; ch++)
      p[(j * PATCH + i) * CH + ch] = im.d[((r * PATCH + j) * im.n + (c * PATCH + i)) * CH + ch];
    out.push(p);
  }
  return out;
}
const dist2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; };

// Farthest-point init (deterministic -- no RNG, so the codebook is a function of
// the target alone) + Lloyd iterations. This is the tokenizer: the ONLY thing
// the AR side is allowed to say about the image is which of these K prototypes
// each cell is.
function buildCodebook(pts) {
  const cent = [Float32Array.from(pts[0])];
  while (cent.length < K) {
    let best = -1, bd = -1;
    for (let i = 0; i < pts.length; i++) {
      let m = Infinity;
      for (const c of cent) m = Math.min(m, dist2(pts[i], c));
      if (m > bd) { bd = m; best = i; }
    }
    cent.push(Float32Array.from(pts[best]));
  }
  for (let it = 0; it < 18; it++) {
    const sum = cent.map(() => new Float64Array(PDIM)), cnt = new Int32Array(K);
    for (const p of pts) { const k = nearest(p, cent).k; cnt[k]++; for (let i = 0; i < PDIM; i++) sum[k][i] += p[i]; }
    for (let k = 0; k < K; k++) if (cnt[k]) for (let i = 0; i < PDIM; i++) cent[k][i] = sum[k][i] / cnt[k];
  }
  return cent;
}
function nearest(p, cent) {
  let k = 0, b = Infinity;
  for (let i = 0; i < cent.length; i++) { const d = dist2(p, cent[i]); if (d < b) { b = d; k = i; } }
  return { k, d: b };
}
// The candidate distribution over the codebook: softmax of the negative squared
// distance between the CONTEXT PREDICTION and each prototype. This is the whole
// "model" -- stated plainly so nobody mistakes it for a trained one.
function candidates(p, cent, temp) {
  const d = cent.map((c) => dist2(p, c));
  const t = Math.max(1e-4, temp), lo = Math.min(...d);
  const w = d.map((x) => Math.exp(-(x - lo) / t));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((x) => x / s);
}
function writePatch(im, r, c, p) {
  for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) for (let ch = 0; ch < CH; ch++)
    im.d[((r * PATCH + j) * im.n + (c * PATCH + i)) * CH + ch] = p[(j * PATCH + i) * CH + ch];
}

// -------------------------------------------------------------- the AR run --
// Returns { frames[], recs[], passes, updates, err }. One frame per token
// placement, always at full resolution so the two panels are directly
// comparable pixel for pixel.
function runAR(target, cent, order, temp) {
  const frames = [], recs = [];
  const tgtPatch = patchesOf(target);
  const mean = new Float32Array(PDIM);
  for (const p of tgtPatch) for (let i = 0; i < PDIM; i++) mean[i] += p[i] / (GRID * GRID);
  // A trained model is STOOD IN FOR, not trained: the query handed to the
  // codebook is the decoded context blended with what the tokenizer itself
  // would have said, at a weight `w` that falls where the model has little to
  // go on. That is the whole "model", and it exists so the ORDERING is visible
  // -- it is not a claim about how accurate real AR image models are.
  const blend = (ctx, truth, w) => { const p = new Float32Array(PDIM); for (let i = 0; i < PDIM; i++) p[i] = (1 - w) * ctx[i] + w * truth[i]; return p; };

  if (order === 'raster') {
    const rec = img(IMG), placed = new Uint8Array(GRID * GRID);
    const tgtTok = tgtPatch.map((p) => nearest(p, cent).k);
    for (let idx = 0; idx < GRID * GRID; idx++) {
      const r = (idx / GRID) | 0, c = idx % GRID;
      let pred, given = false, probs, chosen, w = 0;
      if (idx === 0) {                       // the conditioning token, not predicted
        pred = mean; given = true; chosen = tgtTok[0]; probs = candidates(cent[chosen], cent, temp); w = 1;
      } else {
        // Causal pixel context: for each pixel of this patch, average whatever
        // decoded pixels sit to its left / above. That is the whole reason a
        // raster order struggles with structure it has not scanned past yet.
        pred = new Float32Array(PDIM);
        let taps = 0, tapMax = 0;
        for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) {
          const y = r * PATCH + j, x = c * PATCH + i;
          for (let ch = 0; ch < CH; ch++) {
            let s = 0, m = 0;
            const take = (yy, xx) => {
              if (yy < 0 || xx < 0 || xx >= IMG) return;
              const tr = (yy / PATCH) | 0, tc = (xx / PATCH) | 0;
              if (tr === r && tc === c) return;          // this patch is not written yet
              if (!placed[tr * GRID + tc]) return;       // strictly causal: decoded cells only
              s += rec.d[(yy * IMG + xx) * CH + ch]; m++;
            };
            take(y, x - 1); take(y - 1, x); take(y - 1, x - 1); take(y - 1, x + 1);
            taps += m; tapMax += 4;
            pred[(j * PATCH + i) * CH + ch] = m ? s / m : mean[(j * PATCH + i) * CH + ch];
          }
        }
        // Context coverage: how many of the causal taps this cell actually had.
        // A cell on the top edge has almost none, which is exactly the raster
        // ordering problem next-scale prediction was invented to remove.
        w = 0.94 * (taps / Math.max(1, tapMax));
        probs = candidates(blend(pred, tgtPatch[idx], w), cent, temp);
        chosen = probs.indexOf(Math.max(...probs));
      }
      writePatch(rec, r, c, cent[chosen]); placed[r * GRID + c] = 1;
      recs.push({ scale: GRID, r, c, idx, pass: idx + 1, given, chosen, truth: tgtTok[idx], probs, pred, w });
      frames.push(Float32Array.from(rec.d));
    }
    return finishAR(frames, recs, target, 'raster');
  }

  // Next-scale prediction. Each scale is ONE forward pass over all of its cells;
  // the cells within a scale are conditioned on the previous scale, not on each
  // other, which is why the pass count collapses from 64 to 4.
  let prev = null, idx = 0;
  for (let s = 0; s < SCALES.length; s++) {
    const S = SCALES[s], n = S * PATCH;
    const tgtS = downsample(target, n), tgtTok = patchesOf(tgtS).map((p) => nearest(p, cent).k);
    const guess = prev ? upsample(prev, n) : null;
    const rec = img(n);
    if (guess) rec.d.set(guess.d);
    const gPatch = guess ? patchesOf(guess) : null;
    // Confidence falls as the scale gets finer: global layout is what a
    // coarse-to-fine model pins down first, fine texture is what it is least
    // sure of. Same stand-in caveat as the raster branch.
    const wS = [1, 0.90, 0.76, 0.58][s];
    const tgtPatchS = patchesOf(tgtS);
    for (let cell = 0; cell < S * S; cell++) {
      const r = (cell / S) | 0, c = cell % S;
      let pred, given = false, probs, chosen, w = wS;
      if (!guess) { pred = mean; given = true; chosen = tgtTok[cell]; probs = candidates(cent[chosen], cent, temp); }
      else { pred = gPatch[cell]; probs = candidates(blend(pred, tgtPatchS[cell], wS), cent, temp); chosen = probs.indexOf(Math.max(...probs)); }
      writePatch(rec, r, c, cent[chosen]);
      recs.push({ scale: S, r, c, idx, pass: s + 1, given, chosen, truth: tgtTok[cell], probs, pred, w });
      frames.push(upsample(rec, IMG).d);
      idx++;
    }
    prev = rec;
  }
  return finishAR(frames, recs, target, 'scale');
}
function finishAR(frames, recs, target, order) {
  return { frames, recs, order, passes: AR_PASSES[order], updates: AR_UPDATES[order], err: meanAbs(frames[frames.length - 1], target.d) };
}

// ------------------------------------------------------- the diffusion run --
// The ONE property being compared: every step touches the whole canvas. The
// schedule and the sampler are deliberately not re-taught here (see the
// diffusion-noise and diffusion-sampler pages); this is a plain iterative
// refinement toward a target whose low-pass cutoff opens as t rises, which is
// the coarse-to-fine behaviour those pages derive properly.
function runDiffusion(target, N, seed) {
  const z = seededRandn(seed * 31 + 5, IMG * IMG * CH, { std: 1 });
  let x = new Float32Array(IMG * IMG * CH);
  for (let i = 0; i < x.length; i++) x[i] = clamp01(0.5 + 0.42 * z[i]);
  const frames = [Float32Array.from(x)], deltas = [];
  const tgt = { d: target, n: IMG };
  for (let k = 1; k <= N; k++) {
    const t = k / N;
    const band = lowpass(tgt, 3.2 * Math.pow(1 - t, 1.4));
    const lam = 0.62, prev = x;
    const nx = new Float32Array(x.length);
    let moved = 0;
    for (let i = 0; i < x.length; i++) {
      const jitter = 0.055 * (1 - t) * z[(i * 7 + k * 131) % z.length];
      nx[i] = clamp01(prev[i] + (band.d[i] + jitter - prev[i]) * lam);
      moved += Math.abs(nx[i] - prev[i]);
    }
    x = nx;
    frames.push(Float32Array.from(x));
    deltas.push(moved / x.length);
  }
  return { frames, deltas, steps: N, passes: N, updates: N * GRID * GRID, err: meanAbs(frames[frames.length - 1], target) };
}

// ------------------------------------------------------------------ colour --
// The image is a 3-channel LATENT, not sRGB, so it is painted by mixing three
// of the page's own hue tokens over the page ground -- which means it follows
// the theme instead of freezing one palette into the data.
// The three channels are read as a CHROMA MIX (which of the three hue tokens
// this pixel leans toward) times an INTENSITY (how far from the page ground it
// sits). Mixing them additively instead would drive every pixel toward black on
// a light page and toward white on a dark one, since all three hues are on the
// same side of the ground -- the mix has to be normalised, then interpolated
// away from the ground.
function pxRGB(c0, c1, c2, G, H) {
  const s = c0 + c1 + c2;
  const k = clamp(0.16 + 0.92 * clamp(s / 1.7, 0, 1), 0, 1);
  if (s < 1e-6) return `rgb(${G[0] | 0},${G[1] | 0},${G[2] | 0})`;
  const w = 1 / s, out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const h = (c0 * H[0][i] + c1 * H[1][i] + c2 * H[2][i]) * w;
    out[i] = clamp(G[i] + k * (h - G[i]), 0, 255) | 0;
  }
  return `rgb(${out[0]},${out[1]},${out[2]})`;
}
function paintImage(ctx, data, R, G, H) {
  const cw = R.w / IMG, chh = R.h / IMG;
  for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
    const i = (y * IMG + x) * CH;
    ctx.fillStyle = pxRGB(data[i], data[i + 1], data[i + 2], G, H);
    ctx.fillRect(R.x + x * cw, R.y + y * chh, cw + 0.6, chh + 0.6);
  }
}

// ------------------------------------------------------------------- state --
let target = null, cent = null, ar = null, dif = null, sig = '', floorErr = 0;
let arRect = null, difRect = null, tgtRect = null;   // hit rects, captured in draw
let budgetBar = null, orderStrip = null, stepStrip = null;
let dragging = null;                                  // 'budget' | 'steps' | 'order'

const diffSteps = (st) => (st.fair ? clamp(Math.round(st.budget / (GRID * GRID)), 1, 16) : clamp(st.dsteps | 0, 1, 16));

function build(st) {
  const N = diffSteps(st);
  const s = `${st.seed}|${st.order}|${st.temp}|${N}`;
  if (s === sig) return;
  sig = s;
  target = makeTarget(st.seed | 0);
  const tp = patchesOf({ d: target, n: IMG });
  cent = buildCodebook(tp);
  // The tokenizer's own reconstruction floor: re-encode the target through the
  // codebook and decode it straight back. NO autoregressive run can beat this,
  // however many passes it spends -- the discrete representation is where that
  // error lives, not the sampling loop. The diffusion side here has no
  // equivalent floor, which is a property of THIS toy's setup and not a verdict.
  const q = img(IMG);
  tp.forEach((p, i) => writePatch(q, (i / GRID) | 0, i % GRID, cent[nearest(p, cent).k]));
  floorErr = meanAbs(q.d, target);
  ar = runAR({ d: target, n: IMG }, cent, st.order, st.temp);
  dif = runDiffusion(target, N, st.seed | 0);
}

// The transport axis is NORMALISED PROGRESS, so both processes start together
// and finish together however differently they are priced. The absolute pass
// counts stay on screen -- the shared axis is a viewing device, not a claim
// that a pass costs the same on both sides.
function buildSteps(st) {
  build(st);
  const F = Math.max(ar.frames.length, dif.frames.length - 1);
  const out = [];
  for (let k = 0; k <= F; k++) {
    const a = Math.min(ar.frames.length, Math.round(k / F * ar.frames.length));
    const d = Math.min(dif.steps, Math.round(k / F * dif.steps));
    const rec = a > 0 ? ar.recs[a - 1] : null;
    out.push({
      i: k, a, d,
      label: k === 0
        ? 'nothing placed · pure noise — both processes at their start'
        : `progress ${k}/${F} — AR: ${a}/${ar.frames.length} token${a === 1 ? '' : 's'} placed`
          + (rec ? ` (pass ${rec.pass}, scale ${rec.scale}×${rec.scale}, cell ${rec.r},${rec.c})` : '')
          + ` · diffusion: step ${d}/${dif.steps}, whole canvas`,
    });
  }
  return out;
}

// ------------------------------------------------------------------- draw ---
function frame(ctx, R, title, colour) {
  ctx.save();
  ctx.fillStyle = alphaOf('n14', 0.03); ctx.fillRect(R.x, R.y, R.w, R.h);
  ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(R.x + 0.5, R.y + 0.5, R.w - 1, R.h - 1);
  if (title) { ctx.fillStyle = colour || T.n11; ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.fillText(title, R.x + 5, R.y + 4); }
  ctx.restore();
}
const inRect = (R, x, y) => R && x >= R.x && x <= R.x + R.w && y >= R.y && y <= R.y + R.h;

mount({
  mount: 'body',
  title: 'ar-vs-diffusion-images — a sequence of tokens, or a canvas refined',
  blurb: 'Two mechanisms, one target, one budget. On the left the image is a SEQUENCE OF DISCRETE TOKENS: a codebook of 2×2 patch prototypes is fitted in-page (that is the tokenizer), the target becomes an 8×8 grid of token ids, and the ids are predicted one at a time from the ones already placed — the same loop a language model runs over text, with the current candidate distribution over the codebook drawn underneath. On the right the WHOLE canvas is refined at once, N times; nothing is ordered and nothing is a token. The two do not pay in the same currency, so the readout reports both axes — sequential passes (what you cannot parallelise away) and token-updates (total positions touched) — and they disagree: raster AR is 64 passes but only 64 updates, diffusion at N steps is N passes and N×64 updates, next-scale AR is 4 passes and 85 updates. Which number is "the cost" depends on your batch size, your bandwidth, and whether you already run a language model’s serving stack. This page does not stage a winner, and the quality numbers on screen are a TOY — a hand-built predictor on a 16×16 synthetic image, not evidence about trained models. The schedule and the sampler are not re-taught here: those are the diffusion-noise and diffusion-sampler pages. Drag the budget bar, drag the diffusion step handle, click the order strip; hover any token or region for what produced it.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  autoplay: true,
  challenges: [
    {
      goal: 'Make the autoregressive side cost fewer than 10 sequential forward passes without changing its token count much.',
      hint: 'raster order is one pass per token. Next-scale prediction makes each SCALE one pass — the tokens within a scale are conditioned on the coarser scale, not on each other.',
      check: (api) => ({ solved: (api.probe.arPasses ?? 99) < 10, detail: `AR sequential passes = ${api.probe.arPasses ?? '—'} (need < 10)` }),
    },
    {
      goal: 'Find a shared budget where diffusion touches MORE than 5× the token-updates the AR side does.',
      hint: 'every diffusion step is a whole-canvas pass — 64 updates. The AR side stops when the grid is full and cannot spend more.',
      check: (api) => {
        const rr = (api.probe.difUpdates ?? 0) / Math.max(1, api.probe.arUpdates ?? 1);
        return { solved: rr > 5, detail: `diffusion updates ÷ AR updates = ${rr.toFixed(2)}× (need > 5)` };
      },
    },
    {
      goal: 'Spend a budget the AR side literally cannot use: leave more than 300 token-updates unspent.',
      hint: 'AR has a natural stopping point — the grid is finite. Diffusion has none, which is the other half of this trade.',
      check: (api) => ({ solved: (api.probe.spare ?? 0) > 300, detail: `unspent by AR = ${api.probe.spare ?? 0} token-updates (need > 300)` }),
    },
  ],
  controls: (c, page) => {
    c.select('order', {
      label: 'AR token order',
      options: [{ value: 'raster', label: 'raster (one token / pass)' }, { value: 'scale', label: 'next-scale (one scale / pass)' }],
      value: 'raster', rebuild: true,
    });
    c.toggle('fair', { label: 'same budget (re-allocate fairly)', value: true, rebuild: true });
    c.slider('budget', { label: 'shared budget (token-updates)', min: 64, max: 1024, step: 32, value: 384, rebuild: true });
    c.slider('dsteps', { label: 'diffusion steps (when unshared)', min: 1, max: 16, step: 1, value: 6, rebuild: true });
    c.slider('temp', { label: 'candidate temperature', min: 0.002, max: 0.12, step: 0.002, value: 0.046, rebuild: true, format: (v) => (+v).toFixed(3) });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.transport({ compute: () => buildSteps(page.state), speed: 11, loop: true });
  },

  onPointer: (page, ev) => {
    const st = page.state;
    if (ev.type === 'down') {
      if (inRect(budgetBar, ev.x, ev.y)) dragging = 'budget';
      else if (inRect(stepStrip, ev.x, ev.y)) dragging = 'steps';
      else if (inRect(orderStrip, ev.x, ev.y)) dragging = 'order';
      else dragging = null;
    } else if (ev.type === 'up' || ev.type === 'leave') { dragging = null; return; }
    if (!dragging || !page.pointer.down) return;
    if (dragging === 'budget' && budgetBar) {
      const f = clamp((ev.x - budgetBar.x) / budgetBar.w, 0, 1);
      page.controls.set('budget', Math.round((64 + f * (1024 - 64)) / 32) * 32, { rebuild: true });
    } else if (dragging === 'steps' && stepStrip) {
      const f = clamp((ev.x - stepStrip.x) / stepStrip.w, 0, 1);
      const n = clamp(Math.round(1 + f * 15), 1, 16);
      // Under a shared budget the step handle re-allocates the BUDGET, so the
      // two panels never silently fall out of iso-budget while being dragged.
      if (st.fair) page.controls.set('budget', clamp(n * GRID * GRID, 64, 1024), { rebuild: true });
      else page.controls.set('dsteps', n, { rebuild: true });
    } else if (dragging === 'order' && orderStrip) {
      const want = (ev.x - orderStrip.x) < orderStrip.w / 2 ? 'raster' : 'scale';
      if (want !== st.order) page.controls.set('order', want, { rebuild: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    build(st);
    if (!ar || !dif) return;

    const G = rgbOf(T.n0), H = [rgbOf(T.bad), rgbOf(T.ok), rgbOf(T.accent)];
    const s = page.step();
    const aN = s ? s.a : ar.frames.length, dN = s ? s.d : dif.steps;
    const arNow = aN > 0 ? ar.frames[aN - 1] : null;
    const difNow = dif.frames[dN];
    const rec = aN > 0 ? ar.recs[aN - 1] : null;

    // ---- layout ------------------------------------------------------------
    const pad = 10, barH = 22, legH = 14, headH = 44;
    const top = headH, bot = page.H - barH - legH - 10;
    const colW = (page.W - 3 * pad) / 2;
    const AR = { x: pad, y: top, w: colW, h: bot - top };
    const DF = { x: pad * 2 + colW, y: top, w: colW, h: bot - top };
    frame(ctx, AR, null); frame(ctx, DF, null);

    const distH = 30, capH = 16, gap = 12;             // candidate-distribution strip
    const side = Math.max(60, Math.min(AR.w - 24, AR.h - 26 - gap - distH - capH));
    const imgAR = { x: AR.x + (AR.w - side) / 2, y: AR.y + 26, w: side, h: side };
    const imgDF = { x: DF.x + (DF.w - side) / 2, y: DF.y + 26, w: side, h: side };
    arRect = imgAR; difRect = imgDF;

    // ---- the AR panel ------------------------------------------------------
    ctx.save(); ctx.fillStyle = alphaOf('n14', 0.06); ctx.fillRect(imgAR.x, imgAR.y, imgAR.w, imgAR.h); ctx.restore();
    if (arNow) paintImage(ctx, arNow, imgAR, G, H);
    // token lattice + the cell being placed right now
    ctx.save();
    ctx.strokeStyle = alphaOf('n14', 0.18); ctx.lineWidth = 1; ctx.beginPath();
    for (let k = 0; k <= GRID; k++) {
      const x = imgAR.x + k * imgAR.w / GRID, y = imgAR.y + k * imgAR.h / GRID;
      ctx.moveTo(x, imgAR.y); ctx.lineTo(x, imgAR.y + imgAR.h);
      ctx.moveTo(imgAR.x, y); ctx.lineTo(imgAR.x + imgAR.w, y);
    }
    ctx.stroke();
    if (rec) {
      const f = GRID / rec.scale, cw = imgAR.w / GRID;
      ctx.strokeStyle = T.accent; ctx.lineWidth = 2;
      ctx.strokeRect(imgAR.x + rec.c * f * cw, imgAR.y + rec.r * f * cw, f * cw, f * cw);
    }
    ctx.restore();

    // order strip (click / drag to switch) -- the direct-manipulation handle
    orderStrip = { x: AR.x + 6, y: AR.y + 6, w: AR.w - 12, h: 15 };
    ctx.save();
    ctx.font = '9.5px ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    for (let k = 0; k < 2; k++) {
      const on = (k === 0) === (st.order === 'raster');
      const seg = { x: orderStrip.x + k * orderStrip.w / 2, y: orderStrip.y, w: orderStrip.w / 2, h: orderStrip.h };
      ctx.fillStyle = on ? alphaOf(T.accent, 0.22) : alphaOf('n14', 0.05);
      ctx.fillRect(seg.x, seg.y, seg.w, seg.h);
      ctx.strokeStyle = on ? T.accent : T.n6; ctx.lineWidth = 1; ctx.strokeRect(seg.x + 0.5, seg.y + 0.5, seg.w - 1, seg.h - 1);
      ctx.fillStyle = on ? T.accent : T.n10;
      ctx.fillText(k === 0 ? 'raster · 64 passes' : 'next-scale · 4 passes', seg.x + seg.w / 2, seg.y + seg.h / 2 + 0.5);
    }
    ctx.restore();

    // candidate distribution over the codebook for the token being placed
    const dRect = { x: imgAR.x, y: imgAR.y + imgAR.h + gap, w: imgAR.w, h: distH };
    ctx.save();
    ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(dRect.x + 0.5, dRect.y + 0.5, dRect.w - 1, dRect.h - 1);
    if (rec) {
      const bw = dRect.w / K;
      for (let k = 0; k < K; k++) {
        const h = rec.probs[k] * (dRect.h - 12);
        ctx.fillStyle = k === rec.chosen ? T.accent : alphaOf(T.n9, 0.75);
        ctx.fillRect(dRect.x + k * bw + 1.5, dRect.y + dRect.h - 2 - h, bw - 3, h);
        if (k === rec.truth && k !== rec.chosen) {
          ctx.strokeStyle = T.warn; ctx.lineWidth = 1.4;
          ctx.strokeRect(dRect.x + k * bw + 1.5, dRect.y + 2, bw - 3, dRect.h - 4);
        }
      }
    }
    ctx.fillStyle = T.n10; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(rec ? `p(token) over the ${K}-entry codebook · ▮ chosen · ▯ tokenizer's own id` : `p(token) over the ${K}-entry codebook`, dRect.x + 3, dRect.y - 9);
    ctx.restore();

    // ---- the diffusion panel ----------------------------------------------
    ctx.save(); ctx.fillStyle = alphaOf('n14', 0.06); ctx.fillRect(imgDF.x, imgDF.y, imgDF.w, imgDF.h); ctx.restore();
    paintImage(ctx, difNow, imgDF, G, H);
    // Every region moved on this step -- draw the per-cell magnitude, because
    // "the whole canvas updates together" is the property being compared.
    if (dN > 0) {
      const prev = dif.frames[dN - 1], cw = imgDF.w / GRID;
      ctx.save();
      for (let rr = 0; rr < GRID; rr++) for (let cc = 0; cc < GRID; cc++) {
        let m = 0;
        for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) for (let ch = 0; ch < CH; ch++) {
          const ix = (((rr * PATCH + j) * IMG) + (cc * PATCH + i)) * CH + ch;
          m += Math.abs(difNow[ix] - prev[ix]);
        }
        m /= PDIM;
        ctx.strokeStyle = alphaOf(T.warn, clamp(m * 3.2, 0.05, 0.9)); ctx.lineWidth = 1.4;
        ctx.strokeRect(imgDF.x + cc * cw + 1, imgDF.y + rr * cw + 1, cw - 2, cw - 2);
      }
      ctx.restore();
    }
    // diffusion step handle (draggable)
    stepStrip = { x: DF.x + 6, y: DF.y + 6, w: DF.w - 12, h: 15 };
    ctx.save();
    ctx.fillStyle = alphaOf('n14', 0.05); ctx.fillRect(stepStrip.x, stepStrip.y, stepStrip.w, stepStrip.h);
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(stepStrip.x + 0.5, stepStrip.y + 0.5, stepStrip.w - 1, stepStrip.h - 1);
    const fh = (dif.steps - 1) / 15;
    ctx.fillStyle = alphaOf(T.warn, 0.30); ctx.fillRect(stepStrip.x, stepStrip.y, stepStrip.w * fh, stepStrip.h);
    ctx.fillStyle = T.warn; ctx.fillRect(stepStrip.x + stepStrip.w * fh - 1.5, stepStrip.y, 3, stepStrip.h);
    ctx.font = '9.5px ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'center'; ctx.fillStyle = T.n11;
    ctx.fillText(`drag: ${dif.steps} diffusion steps × 64 = ${dif.updates} updates${st.fair ? ' (moves the shared budget)' : ''}`, stepStrip.x + stepStrip.w / 2, stepStrip.y + stepStrip.h / 2 + 0.5);
    ctx.restore();

    // per-step whole-canvas movement, the diffusion counterpart of the AR
    // candidate strip: one bar per step, every bar covering the WHOLE canvas.
    const mRect = { x: imgDF.x, y: dRect.y, w: imgDF.w, h: distH };
    ctx.save();
    ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(mRect.x + 0.5, mRect.y + 0.5, mRect.w - 1, mRect.h - 1);
    const mx = Math.max(1e-6, Math.max(...dif.deltas)), bwm = mRect.w / dif.steps;
    for (let k = 0; k < dif.steps; k++) {
      const hh = (dif.deltas[k] / mx) * (mRect.h - 8);
      ctx.fillStyle = k < dN ? T.warn : alphaOf(T.n9, 0.6);
      ctx.fillRect(mRect.x + k * bwm + 1.5, mRect.y + mRect.h - 2 - hh, Math.max(1.5, bwm - 3), hh);
    }
    ctx.fillStyle = T.n10; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('mean |Δ| per step — every bar is the WHOLE canvas', mRect.x + 3, mRect.y - 9);
    ctx.restore();

    // ---- the target, in the header band ------------------------------------
    const tw = 30;
    tgtRect = { x: page.W / 2 - tw / 2, y: 6, w: tw, h: tw };
    paintImage(ctx, target, tgtRect, G, H);
    ctx.save(); ctx.strokeStyle = T.n7; ctx.lineWidth = 1; ctx.strokeRect(tgtRect.x + 0.5, tgtRect.y + 0.5, tgtRect.w - 1, tgtRect.h - 1);
    ctx.fillStyle = T.n10; ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('target (16×16) →', tgtRect.x - 6, tgtRect.y + tw / 2);
    ctx.textAlign = 'left';
    ctx.fillText(`← codebook floor ${(floorErr * 100).toFixed(2)}%: no AR run can beat it`, tgtRect.x + tw + 6, tgtRect.y + tw / 2);
    ctx.restore();

    // ---- panel captions ----------------------------------------------------
    const errAR = aN > 0 ? meanAbs(arNow, target) : 1, errDF = meanAbs(difNow, target);
    r.label(`autoregressive · ${aN}/${ar.frames.length} tokens · ${ar.passes} sequential passes · err ${(errAR * 100).toFixed(1)}%`,
      AR.x + 6, AR.y + AR.h - 6, { color: T.accent, font: '10px ui-monospace, monospace' });
    r.label(`diffusion · step ${dN}/${dif.steps} · ${dif.passes} whole-canvas passes · err ${(errDF * 100).toFixed(1)}%`,
      DF.x + 6, DF.y + DF.h - 6, { color: T.warn, font: '10px ui-monospace, monospace' });

    // ---- the shared budget bar (draggable) ---------------------------------
    budgetBar = { x: pad, y: page.H - barH - legH - 4, w: page.W - 2 * pad, h: barH - 8 };
    const B = st.budget | 0, spare = Math.max(0, B - ar.updates);
    ctx.save();
    ctx.fillStyle = alphaOf('n14', 0.05); ctx.fillRect(budgetBar.x, budgetBar.y, budgetBar.w, budgetBar.h);
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(budgetBar.x + 0.5, budgetBar.y + 0.5, budgetBar.w - 1, budgetBar.h - 1);
    const unit = budgetBar.w / 1024;
    ctx.fillStyle = alphaOf(T.accent, 0.55); ctx.fillRect(budgetBar.x, budgetBar.y, Math.min(ar.updates, B) * unit, budgetBar.h / 2 - 1);
    ctx.fillStyle = alphaOf(T.warn, 0.55); ctx.fillRect(budgetBar.x, budgetBar.y + budgetBar.h / 2, Math.min(dif.updates, B) * unit, budgetBar.h / 2 - 1);
    ctx.strokeStyle = T.n13; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(budgetBar.x + B * unit, budgetBar.y - 2); ctx.lineTo(budgetBar.x + B * unit, budgetBar.y + budgetBar.h + 2); ctx.stroke();
    ctx.font = '9.5px ui-monospace, monospace'; ctx.textBaseline = 'middle';
    // Each segment's label sits just past its own end -- flipped inside when the
    // segment runs long, so a full bar never overwrites its own reading.
    const segLabel = (txt, end, yy, col) => {
      const x = budgetBar.x + end * unit;
      const room = budgetBar.x + budgetBar.w - x > ctx.measureText(txt).width + 16;
      ctx.fillStyle = col; ctx.textAlign = room ? 'left' : 'right';
      ctx.fillText(room ? '◀ ' + txt : txt + ' ▶', x + (room ? 5 : -5), yy);
    };
    segLabel(`AR ${ar.updates}`, Math.min(ar.updates, B), budgetBar.y + budgetBar.h / 4, T.accent);
    segLabel(`diffusion ${dif.updates}`, Math.min(dif.updates, B), budgetBar.y + 3 * budgetBar.h / 4, T.warn);
    ctx.restore();

    r.label(`drag the bar: shared budget ${B} token-updates${st.fair ? ` → ${dif.steps} diffusion steps` : ' — UNSHARED, the two sides are not iso-budget'} · ▮ candidate distribution · ▭ region that moved this step`,
      pad, page.H - 4, { color: T.n10, font: '9.5px ui-monospace, monospace' });

    page.probe = {
      arPasses: ar.passes, arUpdates: ar.updates, difPasses: dif.passes, difUpdates: dif.updates,
      spare, errAR, errDF, order: st.order,
    };

    // ---- hover: what produced this token / this region ---------------------
    if (page.pointer.over && !dragging) {
      const p = page.pointer; let tip = null;
      if (inRect(imgAR, p.x, p.y) && aN > 0) {
        const cc = clamp(Math.floor((p.x - imgAR.x) / (imgAR.w / GRID)), 0, GRID - 1);
        const rr = clamp(Math.floor((p.y - imgAR.y) / (imgAR.h / GRID)), 0, GRID - 1);
        // Which placement most recently wrote this cell, at whatever scale.
        let hit = null;
        for (let i = 0; i < aN; i++) {
          const q = ar.recs[i], f = GRID / q.scale;
          if (rr >= q.r * f && rr < (q.r + 1) * f && cc >= q.c * f && cc < (q.c + 1) * f) hit = q;
        }
        if (hit) {
          const top3 = hit.probs.map((v, k) => [k, v]).sort((a, b) => b[1] - a[1]).slice(0, 3);
          tip = `token cell (${rr},${cc})   written by placement ${hit.idx + 1}/${ar.frames.length}\n`
            + `forward pass ${hit.pass}/${ar.passes}   scale ${hit.scale}×${hit.scale}, cell (${hit.r},${hit.c})\n`
            + (hit.given ? 'given as conditioning — not predicted\n' : `context: ${st.order === 'raster' ? 'decoded pixels left / above' : 'upsampled previous scale'}\n`)
            + `candidates: ` + top3.map(([k, v]) => `#${k} ${(v * 100).toFixed(1)}%`).join('  ') + '\n'
            + `chose #${hit.chosen}${hit.chosen === hit.truth ? ' = tokenizer id ✓' : `, tokenizer id #${hit.truth} ✗`}`;
        }
      } else if (inRect(imgDF, p.x, p.y)) {
        const cc = clamp(Math.floor((p.x - imgDF.x) / (imgDF.w / GRID)), 0, GRID - 1);
        const rr = clamp(Math.floor((p.y - imgDF.y) / (imgDF.h / GRID)), 0, GRID - 1);
        const at = (fr) => { let m = 0; for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) for (let ch = 0; ch < CH; ch++) m += fr[(((rr * PATCH + j) * IMG) + (cc * PATCH + i)) * CH + ch]; return m / PDIM; };
        let big = 0, bigK = 0;
        for (let k = 1; k <= dN; k++) { const d = Math.abs(at(dif.frames[k]) - at(dif.frames[k - 1])); if (d > big) { big = d; bigK = k; } }
        const tv = (() => { let m = 0; for (let j = 0; j < PATCH; j++) for (let i = 0; i < PATCH; i++) for (let ch = 0; ch < CH; ch++) m += target[(((rr * PATCH + j) * IMG) + (cc * PATCH + i)) * CH + ch]; return m / PDIM; })();
        tip = `region (${rr},${cc})   no token, no order — updated on EVERY step\n`
          + `steps applied so far: ${dN}/${dif.steps}, each one a whole-canvas pass\n`
          + `mean latent now ${at(difNow).toFixed(3)}   target ${tv.toFixed(3)}   |Δ| ${Math.abs(at(difNow) - tv).toFixed(3)}\n`
          + (dN > 0 ? `largest move so far: step ${bigK} moved it ${big.toFixed(3)}` : 'still at the noise sample');
      } else if (inRect(budgetBar, p.x, p.y)) {
        tip = `shared budget ${B} token-updates — drag me\n`
          + `AR spends ${ar.updates} and then STOPS (the grid is finite)\n`
          + `diffusion spends ${dif.updates} = ${dif.steps} × 64 and would keep going\n`
          + `unspent by AR: ${spare} token-updates`;
      } else if (inRect(orderStrip, p.x, p.y)) {
        tip = `AR token order — click either half\nraster: 1 token per forward pass → ${AR_PASSES.raster} passes, ${AR_UPDATES.raster} updates\nnext-scale: 1 SCALE per forward pass → ${AR_PASSES.scale} passes, ${AR_UPDATES.scale} updates`;
      } else if (inRect(stepStrip, p.x, p.y)) {
        tip = `diffusion step count — drag me\n${dif.steps} steps × 64 positions = ${dif.updates} token-updates\n${st.fair ? 'shared budget is on, so this moves the budget too' : 'shared budget is off — the two sides are not iso-budget'}`;
      } else if (inRect(dRect, p.x, p.y) && rec) {
        tip = `candidate distribution for placement ${rec.idx + 1}\nsoftmax over −‖query − prototype‖² / τ,  τ = ${st.temp.toFixed(3)}\nquery = ${(100 * (1 - (rec.w || 0))).toFixed(0)}% decoded context + ${(100 * (rec.w || 0)).toFixed(0)}% model stand-in\nentropy ${(-rec.probs.reduce((a, v) => a + (v > 1e-9 ? v * Math.log(v) : 0), 0)).toFixed(3)} nats over ${K} entries`;
      } else if (inRect(mRect, p.x, p.y)) {
        const k = clamp(Math.floor((p.x - mRect.x) / bwm), 0, dif.steps - 1);
        tip = `diffusion step ${k + 1}/${dif.steps}\nmean |Δ| over the whole canvas: ${dif.deltas[k].toFixed(4)}\nthis one bar is ${GRID * GRID} positions updated together —\nthere is no per-position bar to hover, which is the point`;
      } else if (inRect(tgtRect, p.x, p.y)) {
        tip = `the target: a 16×16, 3-channel synthetic image\ncodebook floor ${(floorErr * 100).toFixed(2)}% — the best the ${K}-entry tokenizer can do\nboth panels are trying to reach THIS, under one shared budget`;
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout -----------------------------------------------------------
    const pct = (a, b) => (b > 0 ? (100 * a / b).toFixed(1) : '—');
    let o = `TOY — 16×16 synthetic image, a stand-in predictor on the AR side and an oracle refinement target on the diffusion side. The error percentages below say NOTHING about how trained AR and diffusion models compare; they are here to show what KIND of error each mechanism makes. The cost columns are the real content.    tier:${r.name}\n`;
    o += `${s ? s.label : 'complete'}\n`;
    o += `AR (${st.order === 'raster' ? 'raster' : 'next-scale'}): ${ar.passes} sequential forward passes, ${ar.updates} token-updates · final err ${(ar.err * 100).toFixed(2)}%\n`;
    o += `diffusion: ${dif.steps} steps = ${dif.passes} sequential whole-canvas passes, ${dif.updates} token-updates · final err ${(dif.err * 100).toFixed(2)}%\n`;
    o += `tokenizer floor ${(floorErr * 100).toFixed(2)}% — the target re-encoded through the ${K}-entry codebook and decoded straight back. No AR run can go below it at any pass count; the error lives in the DISCRETE REPRESENTATION, not the sampling loop. The diffusion side has no equivalent floor here, which is a property of this toy's setup and not a verdict (a real latent-diffusion model has an autoencoder floor of its own).\n`;
    o += `sequential passes — AR is ${pct(ar.passes, dif.passes)}% of diffusion's ${dif.passes} (LOWER is better on this axis; 100% = parity)\n`;
    o += `token-updates — AR is ${pct(ar.updates, dif.updates)}% of diffusion's ${dif.updates} (LOWER is better; 100% = parity)\n`;
    o += st.fair
      ? `same budget ON: ${B} shared token-updates → ${dif.steps} diffusion steps; AR spends ${ar.updates} and stops, leaving ${spare} unspent. AR's stopping point is structural — the grid is finite — while diffusion has no natural one, only a step count you chose.\n`
      : `same budget OFF: the two sides are NOT iso-budget (${ar.updates} vs ${dif.updates} token-updates), so any quality difference on screen is confounded by compute. Turn it back on before reading anything into the error numbers.\n`;
    o += st.order === 'raster'
      ? 'Raster order makes long-range structure hard: each token sees only what has already been scanned, so the top rows are decided before the bar at the bottom exists. That is the ordering problem next-scale prediction (arXiv 2404.02905) exists to fix — and it is a problem diffusion simply does not have, because it has no order at all.\n'
      : 'Next-scale prediction removes the ordering problem inside a scale: every cell of a scale is conditioned on the upsampled coarser reconstruction, not on its siblings, so the whole scale is one forward pass. Global structure lands first and detail refines — coarse-to-fine, like diffusion, but still exactly a discrete token sequence with exact likelihoods.\n';
    o += 'Unsettled on purpose: AR runs in a fixed number of passes with a KV cache and the whole serving stack a language model already has, and composes with its token stream; diffusion parallelises differently and has historically led on fine texture at a matched budget. Neither column here decides that.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__arVsDiffusionPage = page;
  const q = new URLSearchParams(location.search);
  const num = (k) => parseFloat(q.get(k));
  const bool = (k) => q.get(k) === '1' || q.get(k) === 'true';
  if (q.has('seed')) page.controls.set('seed', num('seed') | 0, { rebuild: true, silent: true });
  if (q.has('order')) page.controls.set('order', q.get('order'), { rebuild: true, silent: true });
  if (q.has('fair')) page.controls.set('fair', bool('fair'), { rebuild: true, silent: true });
  if (q.has('budget')) page.controls.set('budget', num('budget') | 0, { rebuild: true, silent: true });
  if (q.has('dsteps')) page.controls.set('dsteps', num('dsteps') | 0, { rebuild: true, silent: true });
  if (q.has('temp')) page.controls.set('temp', num('temp'), { rebuild: true, silent: true });
  const t = page.controls._transport;
  if (t) t.rebuild();
  // ?hover=x,y is the headless stand-in for a cursor (canvas CSS-px space).
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
