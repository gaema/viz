// few-step-distillation -- how fifty sampling steps become one to four, and
// what you pay for it.
//
// The sibling page `diffusion-sampler` treats step count as a DISCRETISATION
// dial on a FIXED model: same field, coarser integration. This page is the
// other thing entirely -- the model is CHANGED so that few steps suffice. A
// teacher sampler walks many small steps along a trajectory; a distilled
// student is trained to jump most of the way in one.
//
// Everything on screen is arithmetic this page just did. The toy data
// distribution is a two-moons mixture of narrow Gaussians, for which the
// probability-flow velocity is closed form, so the TEACHER trajectory is exact
// and the student's error against it is a measurement rather than a drawing.
//
//   interpolant     x_t = a(t)*x_data + s(t)*x_noise,  a = sin(pi t/2), s = cos(pi t/2)
//   posterior mean  E[x_data | x_t = x] = (1 - c*a)*M(x) + c*x,   c = a*var/tau^2
//   velocity        u(x,t) = a'*E1 + (s'/s)*(x - a*E1)
//   teacher step    x <- x + u*h,  h = 1/N        (N of them, N large)
//
// THE STUDENT IS A DIFFERENT MODEL, NOT A COARSER SAMPLER. It is fitted here,
// in the page, by distillation from the teacher:
//
//   1. draw M noise samples z_j and run the teacher to convergence: y_j
//   2. the student is a finite-capacity map fitted to those pairs --
//      Nadaraya-Watson kernel regression with bandwidth h_bw:
//
//        f(x, t) = sum_j w_j y_j / sum_j w_j,
//        w_j     = exp(-|x - (a(t) y_j + s(t) z_j)|^2 / (2 h(t)^2)),
//        h(t)    = h0 * (s(t) + 0.06)
//
//      The bandwidth is scaled by the NOISE LEVEL s(t), because that is what
//      sets how hard the jump is: from pure noise at t=0 the student must guess
//      the whole endpoint, while from a mostly-denoised point at t=0.9 it barely
//      has to move. A bandwidth fixed in x-space instead makes every jump
//      equally lossy -- built that way first, and the page's own spread number
//      then said more jumps made diversity WORSE, which is not what shipped
//      few-step samplers do. The noise-scaled form is both the physically right
//      one and the one whose measurements behave.
//
//      The regression targets are indexed by the point the trajectory occupies
//      at time t, which IS the consistency training signal: every point on one
//      trajectory must map to that trajectory's endpoint.
//   3. K-step sampling is the multistep consistency procedure -- jump to the
//      endpoint estimate, re-noise back to the next time, jump again.
//
// WHY THIS SHAPE MEASURES THE COST INSTEAD OF ASSERTING IT. Capacity is the
// bandwidth. A wide kernel averages many endpoints together, so different noise
// samples produce the SAME averaged output -- diversity collapse, arising from
// the fit rather than painted on. The effective sample size (ESS = 1 / sum of
// squared normalised weights) says how many teacher endpoints each output is a
// blend of, and the strip below sweeps the seed with everything else held fixed
// so the collapse is visible and counted.
//
// Two shipped families appear as MODES of the one widget:
//   consistency   -- the fit above, trained on self-consistency pairs, drawn as
//                    the tie-lines from several points of one trajectory to
//                    their single shared endpoint.
//   adversarial   -- the same fit plus a term that keeps output on the data
//                    manifold (a discriminator in ADD, a distribution-matching
//                    loss in DMD2), modelled here as a projection of the raw
//                    output toward the nearest data mode and drawn as the
//                    correction arrow. Whether it also buys back DIVERSITY on
//                    this toy is not asserted -- it is in the readout.
//
// Sources: Song et al., "Consistency Models", https://arxiv.org/abs/2303.01469 ;
// Luo et al., "Latent Consistency Models", https://arxiv.org/abs/2310.04378 ;
// Sauer et al., "Adversarial Diffusion Distillation" (SDXL-Turbo),
// https://arxiv.org/abs/2311.17042 ; Yin et al., "Improved Distribution Matching
// Distillation" (DMD2), https://arxiv.org/abs/2405.14867 . No number from any of
// those papers is reproduced here; every figure on screen is measured in-page on
// the toy.
import { mount } from '../framework/layout.js';
import { T, alphaOf, mixColor } from '../framework/theme.js';
import { seededRandn } from '../framework/tensor.js';

// ---------------------------------------------------------------- toy data --
const MODE_SD = 0.06, MODE_VAR = MODE_SD * MODE_SD;
const MOONS = (() => {
  const pts = [], K = 48;
  for (let i = 0; i < K; i++) { const th = Math.PI * i / (K - 1); pts.push([Math.cos(th), Math.sin(th)]); }
  for (let i = 0; i < K; i++) { const th = Math.PI * i / (K - 1); pts.push([1 - Math.cos(th), 0.5 - Math.sin(th)]); }
  return pts.map(([x, y]) => [(x - 0.5) * 1.25, (y - 0.25) * 1.25]);
})();
const HALF = 2.5;          // math-space half-extent drawn in the big panels
const TEACHER_REF = 120;   // steps in the converged teacher run (the distillation target)
const NTRAIN = 48;         // distillation pairs the student is fitted on
const NSTRIP = 16;         // seed sweep: columns in the sample strip
const DUP_EPS = 0.08;      // "these two samples are the same picture" radius
const ADV_PULL = 0.9;      // how hard the manifold term pulls, in adversarial mode

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// Interpolant coefficients and derivatives (the diffusion path; the interpolant
// dial itself belongs to the diffusion-sampler page, so it is fixed here).
function coefs(t) {
  const h = Math.PI / 2;
  return { a: Math.sin(h * t), s: Math.cos(h * t), ad: h * Math.cos(h * t), sd: -h * Math.sin(h * t) };
}

// Exact probability-flow velocity of the toy mixture at (x, y, t).
function field(x, y, t) {
  const { a, s, ad, sd } = coefs(t);
  const sig = Math.max(s, 1e-3);
  const tau2 = a * a * MODE_VAR + sig * sig;
  let best = Infinity;
  const d = new Float64Array(MOONS.length);
  for (let k = 0; k < MOONS.length; k++) {
    const dx = x - a * MOONS[k][0], dy = y - a * MOONS[k][1];
    d[k] = dx * dx + dy * dy; if (d[k] < best) best = d[k];
  }
  let wsum = 0, mx = 0, my = 0;
  for (let k = 0; k < MOONS.length; k++) {
    const w = Math.exp(-(d[k] - best) / (2 * tau2));
    wsum += w; mx += w * MOONS[k][0]; my += w * MOONS[k][1];
  }
  mx /= wsum; my /= wsum;
  const c = a * MODE_VAR / tau2;
  const e1x = (1 - c * a) * mx + c * x, e1y = (1 - c * a) * my + c * y;
  const rx = x - a * e1x, ry = y - a * e1y;
  return { a, s: sig, e1x, e1y, ux: ad * e1x + (sd / sig) * rx, uy: ad * e1y + (sd / sig) * ry };
}

// The teacher: N deterministic Euler steps of the exact field, t: 0 -> 1.
function integrate(x0, y0, N) {
  const out = new Float64Array(2 * (N + 1));
  let x = x0, y = y0; out[0] = x; out[1] = y;
  const h = 1 / N;
  for (let i = 0; i < N; i++) {
    const f = field(x, y, i * h);
    x = clamp(x + f.ux * h, -6, 6); y = clamp(y + f.uy * h, -6, 6);
    out[2 * i + 2] = x; out[2 * i + 3] = y;
  }
  return out;
}
function teacherEnd(x0, y0) {
  const tr = integrate(x0, y0, TEACHER_REF);
  return [tr[2 * TEACHER_REF], tr[2 * TEACHER_REF + 1]];
}

// Deterministic re-noise draws, so one URL replays one run exactly.
function gauss2(seed, si, k) {
  const mix = (a, b, c) => { let h = (Math.imul(a, 374761393) + Math.imul(b, 668265263) + Math.imul(c, 2246822519)) >>> 0; h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0; return ((h ^ (h >>> 16)) >>> 8) / 16777216; };
  const u = Math.max(1e-7, mix(seed + 1, si + 17, k + 3)), v = mix(seed + 101, si + 5, k + 61);
  const r = Math.sqrt(-2 * Math.log(u));
  return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
}

function nearestMode(x, y) {
  let best = Infinity, bi = 0;
  for (let k = 0; k < MOONS.length; k++) {
    const dx = x - MOONS[k][0], dy = y - MOONS[k][1], d = dx * dx + dy * dy;
    if (d < best) { best = d; bi = k; }
  }
  return { i: bi, d: Math.sqrt(best) };
}

// ------------------------------------------------------- the fitted student --
// bandwidth at t=0: strength 0 -> 0.05 (near nearest-neighbour, high capacity),
//                   strength 1 -> 3.00 (one averaged answer for every input).
// The working bandwidth is this scaled by the noise level still present.
const bwOf = (strength) => 0.05 * Math.pow(60, clamp(strength, 0, 1));
const hAt = (h0, t) => h0 * (coefs(t).s + 0.06);

// f(x, t) over the distilled pairs, plus the diagnostics the tooltip prints.
function studentJump(D, x, y, t, bw) {
  const { a, s } = coefs(t), hh = hAt(bw, t), h2 = 2 * hh * hh;
  const n = NTRAIN, d = new Float64Array(n);
  let best = Infinity;
  for (let j = 0; j < n; j++) {
    const px = a * D.ys[2 * j] + s * D.zs[2 * j], py = a * D.ys[2 * j + 1] + s * D.zs[2 * j + 1];
    const dx = x - px, dy = y - py;
    d[j] = dx * dx + dy * dy; if (d[j] < best) best = d[j];
  }
  let wsum = 0, ox = 0, oy = 0;
  const w = new Float64Array(n);
  for (let j = 0; j < n; j++) { w[j] = Math.exp(-(d[j] - best) / h2); wsum += w[j]; ox += w[j] * D.ys[2 * j]; oy += w[j] * D.ys[2 * j + 1]; }
  let sq = 0; for (let j = 0; j < n; j++) { const p = w[j] / wsum; sq += p * p; }
  const order = Array.from({ length: n }, (_, j) => j).sort((p, q) => w[q] - w[p]).slice(0, 3);
  return {
    x: ox / wsum, y: oy / wsum, h: hh,
    ess: 1 / sq,                                   // how many endpoints got averaged
    top: order.map((j) => ({ j, w: w[j] / wsum, y: [D.ys[2 * j], D.ys[2 * j + 1]] })),
  };
}

// One student sample: K jumps, re-noising between them (multistep consistency).
function studentSample(D, z0, z1, K, bw, mode, seed, si) {
  let x = z0, y = z1;
  const hops = [];
  for (let k = 0; k < K; k++) {
    const t = k / K;
    const raw = studentJump(D, x, y, t, bw);
    let ox = raw.x, oy = raw.y;
    if (mode === 'adversarial') {
      const nm = nearestMode(ox, oy);
      ox += ADV_PULL * (MOONS[nm.i][0] - ox); oy += ADV_PULL * (MOONS[nm.i][1] - oy);
    }
    const hop = { t, from: [x, y], raw: [raw.x, raw.y], out: [ox, oy], ess: raw.ess, h: raw.h, top: raw.top, renoise: null };
    if (k < K - 1) {
      const tn = (k + 1) / K, c = coefs(tn), zz = gauss2(seed, si, k);
      x = c.a * ox + c.s * zz[0]; y = c.a * oy + c.s * zz[1];
      hop.renoise = [x, y, tn];
    } else { x = ox; y = oy; }
    hops.push(hop);
  }
  return { end: [x, y], hops };
}

// ------------------------------------------------------------------ metrics --
// RMS distance from the set's own centroid: how much variety a sample set has.
function spread(pts) {
  const n = pts.length; if (!n) return 0;
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p[0]; cy += p[1]; }
  cx /= n; cy /= n;
  let s = 0; for (const p of pts) s += (p[0] - cx) ** 2 + (p[1] - cy) ** 2;
  return Math.sqrt(s / n);
}
// Fraction of samples that have a near-identical twin elsewhere in the set.
function dupFraction(pts) {
  const n = pts.length; let dup = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < DUP_EPS) { dup++; break; }
    }
  }
  return n ? dup / n : 0;
}
function modesHit(pts) { const s = new Set(); for (const p of pts) s.add(nearestMode(p[0], p[1]).i); return s.size; }
function meanOff(pts) { let s = 0; for (const p of pts) s += nearestMode(p[0], p[1]).d; return pts.length ? s / pts.length : 0; }

// ------------------------------------------------------------------- state ---
let D = null;        // { sig, zs, ys } the distillation dataset
let S = null;        // { sig, z, tEnd } the seed sweep (strip) noise + teacher samples
let M = null;        // { sig, starts, trajSig, trajs, tEnd } the main-panel particles
let U = null;        // { sig, runs, strip, m } everything student-side
let plotT = null, plotS = null;   // panel rects + math<->px maps
let grab = null;     // {kind:'start', i} | {kind:'K'} | {kind:'bw'}
let railK = null, railBw = null, stripCells = [];

function ensure(st) {
  const seed = st.seed | 0, P = st.P | 0;

  const dsig = `${seed}`;
  if (!D || D.sig !== dsig) {
    const z = seededRandn(seed * 13 + 7, 2 * NTRAIN, { std: 1 });
    const zs = new Float64Array(2 * NTRAIN), ys = new Float64Array(2 * NTRAIN);
    for (let j = 0; j < NTRAIN; j++) {
      zs[2 * j] = z[2 * j]; zs[2 * j + 1] = z[2 * j + 1];
      const e = teacherEnd(zs[2 * j], zs[2 * j + 1]);
      ys[2 * j] = e[0]; ys[2 * j + 1] = e[1];
    }
    D = { sig: dsig, zs, ys };
  }

  if (!S || S.sig !== dsig) {
    const z = seededRandn(seed * 31 + 5, 2 * NSTRIP, { std: 1 });
    const zz = [], tEnd = [];
    for (let c = 0; c < NSTRIP; c++) { zz.push([z[2 * c], z[2 * c + 1]]); tEnd.push(teacherEnd(z[2 * c], z[2 * c + 1])); }
    S = { sig: dsig, z: zz, tEnd };
  }

  const msig = `${seed}|${P}`;
  if (!M || M.sig !== msig) {
    const z = seededRandn(seed * 7 + 1, 2 * P, { std: 1 });
    const starts = new Float64Array(2 * P);
    for (let i = 0; i < 2 * P; i++) starts[i] = z[i];
    M = { sig: msig, starts, trajSig: '', trajs: null, tEnd: null };
  }
  const tsig = `${msig}|${st.tsteps}|${M.dirty || 0}`;
  if (M.trajSig !== tsig) {
    M.trajSig = tsig; M.trajs = []; M.tEnd = [];
    for (let p = 0; p < P; p++) {
      M.trajs.push(integrate(M.starts[2 * p], M.starts[2 * p + 1], Math.max(1, st.tsteps | 0)));
      M.tEnd.push(teacherEnd(M.starts[2 * p], M.starts[2 * p + 1]));
    }
  }

  const bw = bwOf(st.bw);
  const usig = `${tsig}|${st.K}|${st.bw}|${st.mode}`;
  if (!U || U.sig !== usig) {
    const runs = [];
    for (let p = 0; p < P; p++) runs.push(studentSample(D, M.starts[2 * p], M.starts[2 * p + 1], st.K | 0, bw, st.mode, seed, 1000 + p));
    const strip = [];
    for (let c = 0; c < NSTRIP; c++) strip.push(studentSample(D, S.z[c][0], S.z[c][1], st.K | 0, bw, st.mode, seed, c));
    const sEnds = strip.map((r) => r.end), tEnds = S.tEnd;
    let resid = 0, ess = 0;
    for (let c = 0; c < NSTRIP; c++) {
      resid += Math.hypot(sEnds[c][0] - tEnds[c][0], sEnds[c][1] - tEnds[c][1]);
      ess += strip[c].hops[0].ess;
    }
    const spT = spread(tEnds), spS = spread(sEnds);
    // Paired control for the jump rail: the SAME student, same fit, same seeds,
    // at one jump. Without it "more jumps helps" would be an assertion.
    let sp1 = spS, off1 = 0;
    if ((st.K | 0) === 1) { off1 = meanOff(sEnds); } else {
      const one = [];
      for (let c = 0; c < NSTRIP; c++) one.push(studentSample(D, S.z[c][0], S.z[c][1], 1, bw, st.mode, seed, c).end);
      sp1 = spread(one); off1 = meanOff(one);
    }
    U = {
      sig: usig, bw, runs, strip,
      m: {
        sp1Pct: spT > 1e-9 ? 100 * sp1 / spT : 0, off1,
        spT, spS, spPct: spT > 1e-9 ? 100 * spS / spT : 0,
        dupT: dupFraction(tEnds), dupS: dupFraction(sEnds),
        modeT: modesHit(tEnds), modeS: modesHit(sEnds),
        offT: meanOff(tEnds), offS: meanOff(sEnds),
        resid: resid / NSTRIP, ess: ess / NSTRIP,
      },
    };
  }
}

// A sample's position as a colour, so a repeated sample is a repeated swatch.
function sampleColor(x, y) {
  const u = clamp((x + 1.6) / 3.2, 0, 1), v = clamp((y + 1.3) / 2.6, 0, 1);
  return mixColor(mixColor(T.teal, T.violet, u), T.gold, 0.55 * v);
}

function panelFrame(ctx, R, title, colour) {
  ctx.save();
  ctx.fillStyle = alphaOf('n14', 0.03); ctx.fillRect(R.x, R.y, R.w, R.h);
  ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(R.x + 0.5, R.y + 0.5, R.w - 1, R.h - 1);
  if (title) { ctx.fillStyle = colour || T.n11; ctx.font = '10.5px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.fillText(title, R.x + 6, R.y + 5); }
  ctx.restore();
}
function drawManifold(ctx, map, r, alpha) {
  ctx.save(); ctx.fillStyle = alphaOf(T.ok, alpha);
  for (let k = 0; k < MOONS.length; k++) { const p = map(MOONS[k][0], MOONS[k][1]); ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.2832); ctx.fill(); }
  ctx.restore();
}
function mkPlot(R, top) {
  const sc = Math.min(R.w - 10, R.h - top - 26) / (2 * HALF);
  const cx = R.x + R.w / 2, cy = R.y + top + (R.h - top - 16) / 2;
  return {
    R, sc,
    map: (mx, my) => ({ x: cx + mx * sc, y: cy - my * sc }),
    unmap: (px, py) => ({ x: (px - cx) / sc, y: -(py - cy) / sc }),
  };
}

// A draggable rail: returns its rect + knob position for hit-testing.
function drawRail(ctx, r, x, y, w, value, lo, hi, label, tint) {
  const f = (value - lo) / (hi - lo), kx = x + f * w;
  ctx.save();
  ctx.strokeStyle = T.n6; ctx.lineWidth = 3; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + w, y); ctx.stroke();
  ctx.strokeStyle = tint; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(kx, y); ctx.stroke();
  ctx.fillStyle = tint; ctx.beginPath(); ctx.arc(kx, y, 5.5, 0, 6.2832); ctx.fill();
  ctx.strokeStyle = T.n0; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(kx, y, 5.5, 0, 6.2832); ctx.stroke();
  ctx.restore();
  r.label(label, x, y - 8, { color: T.n11, font: '9.5px ui-monospace, monospace' });
  return { x, y, w, kx };
}

mount({
  mount: 'body',
  title: 'few-step-distillation — changing the model so four steps are enough',
  blurb: 'A teacher sampler takes fifty small steps along a trajectory. A distilled student is trained to jump most of the way in ONE — and one to four jumps is what shipped few-step image models actually run. This is not the same lever as lowering the step count on a fixed model (that is the diffusion-sampler page); the model itself is different. The student here is fitted in the page from the teacher: draw noise samples, run the exact teacher to convergence, and fit a finite-capacity map to those pairs, indexed by where the trajectory is at time t — which IS the consistency training signal, that every point on one trajectory maps to the same endpoint. THE COST IS DIVERSITY. Hold everything fixed and sweep only the seed: the teacher strip is varied, the distilled strip repeats, because a low-capacity student averages many teacher endpoints into one answer. The strip counts it — sample spread as a percent of the teacher\'s, distinct data modes reached, and how many samples have a near-identical twin. Drag the two rails in the student panel (jump count 1..8, distillation strength) and watch variety drain. The mode switch changes what the student is trained against: pure self-consistency, or self-consistency plus a term that keeps output on the data manifold — the adversarial / distribution-matching family. Drag a hollow start marker to re-run both models under your hand; hover any step, jump or strip cell for its arithmetic.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  autoplay: true,
  challenges: [
    {
      goal: 'Give the variety back: get the student\'s sample spread above 70% of the teacher\'s.',
      hint: 'the collapse is capacity, not step count — lower the distillation strength so each answer is a blend of fewer teacher endpoints, and watch the repeated swatches in the strip separate.',
      check: (api) => ({ solved: (api.probe.spPct ?? 0) > 70, detail: `student spread = ${(api.probe.spPct ?? 0).toFixed(1)}% of the teacher's (need > 70%; lower = less variety)` }),
    },
    {
      goal: 'Show the collapse belongs to the MODEL, not the step count: keep spread under 25% of the teacher\'s while running 4 or more jumps.',
      hint: 'more jumps do move the number, in whichever direction the paired 1-jump control in the readout says — a re-noise puts randomness back and then the averaging is applied again. Raise the strength until the model swallows it either way. That is the point: a fixed model sampled coarsely is the diffusion-sampler page; here the model itself is what threw the variety away.',
      check: (api) => ({ solved: (api.probe.K ?? 0) >= 4 && (api.probe.spPct ?? 999) < 25, detail: `${api.probe.K ?? 0} jumps, spread ${(api.probe.spPct ?? 0).toFixed(1)}% of the teacher's (need >= 4 jumps and < 25%)` }),
    },
    {
      goal: 'Land a one-or-two-jump sample ON the data manifold: mean off-manifold distance under 0.05.',
      hint: 'that is what the adversarial / distribution-matching family is for — switch the training signal. Then read the diversity numbers again before deciding it was free.',
      check: (api) => ({ solved: (api.probe.K ?? 9) <= 2 && (api.probe.offS ?? 9) < 0.05, detail: `${api.probe.K ?? 0} jumps, off-manifold ${(api.probe.offS ?? 0).toFixed(3)} (need <= 2 jumps and < 0.05); spread is ${(api.probe.spPct ?? 0).toFixed(1)}% of the teacher's` }),
    },
  ],
  controls: (c, page) => {
    c.select('mode', {
      label: 'training signal',
      options: [{ value: 'consistency', label: 'consistency (self-consistency only)' }, { value: 'adversarial', label: 'adversarial / distribution-matching' }],
      value: 'consistency',
    });
    c.slider('K', { label: 'student jumps (drag the rail too)', min: 1, max: 8, step: 1, value: 1 });
    c.slider('bw', { label: 'distillation strength', min: 0, max: 1, step: 0.01, value: 0.88 });
    c.slider('tsteps', { label: 'teacher steps', min: 10, max: 80, step: 1, value: 50, rebuild: true });
    c.stepper('P', { label: 'trajectories shown', min: 3, max: 12, value: 6 });
    c.toggle('signal', { label: 'draw the training signal', value: true });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.transport({
      compute: () => {
        const st = page.state; ensure(st);
        const N = Math.max(1, st.tsteps | 0);
        return Array.from({ length: N + 1 }, (_, i) => ({
          i, t: i / N,
          label: i === 0 ? `t=0 — the same noise sample enters both models` : `teacher step ${i}/${N} (x ← x + u·h), student jump ${Math.min(st.K | 0, Math.floor(i * (st.K | 0) / N) + 1)}/${st.K | 0}`,
        }));
      },
      speed: 12, loop: true,
    });
  },

  onPointer: (page, ev) => {
    const st = page.state;
    if (ev.type === 'down') {
      grab = null;
      if (railK && Math.hypot(ev.x - railK.kx, ev.y - railK.y) < 13) grab = { kind: 'K' };
      else if (railBw && Math.hypot(ev.x - railBw.kx, ev.y - railBw.y) < 13) grab = { kind: 'bw' };
      else if (plotT && M) {
        for (let p = 0; p < (st.P | 0); p++) {
          const q = plotT.map(M.starts[2 * p], M.starts[2 * p + 1]);
          if (Math.hypot(ev.x - q.x, ev.y - q.y) < 11) { grab = { kind: 'start', i: p }; break; }
        }
      }
      if (grab) page.redraw();
    } else if (ev.type === 'up' || ev.type === 'leave') { grab = null; }
    else if (ev.type === 'move' && grab && page.pointer.down) {
      if (grab.kind === 'K' && railK) {
        const v = clamp(Math.round(1 + 7 * (ev.x - railK.x) / railK.w), 1, 8);
        if (v !== (st.K | 0)) page.controls.set('K', v);
      } else if (grab.kind === 'bw' && railBw) {
        const v = clamp((ev.x - railBw.x) / railBw.w, 0, 1);
        page.controls.set('bw', +v.toFixed(3));
      } else if (grab.kind === 'start' && plotT && M) {
        const m = plotT.unmap(ev.x, ev.y);
        M.starts[2 * grab.i] = clamp(m.x, -3.2, 3.2);
        M.starts[2 * grab.i + 1] = clamp(m.y, -3.2, 3.2);
        M.dirty = (M.dirty || 0) + 1;      // invalidates both trajectories and student runs
        page.redraw();
      }
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    ensure(st);
    const P = st.P | 0, N = Math.max(1, st.tsteps | 0), K = st.K | 0;
    const s = page.step(), si = s ? s.i : N, tcur = si / N;
    const h = 1 / N;
    const m = U.m;

    // ---- layout: two panels on top, the seed-sweep strip underneath ---------
    const pad = 12, top = 26, stripH = 128, bot = page.H - 6;
    const panelH = Math.max(120, (bot - top) - stripH);
    const pw = (page.W - 3 * pad) / 2;
    const RT = { x: pad, y: top, w: pw, h: panelH };
    const RS = { x: pad * 2 + pw, y: top, w: pw, h: panelH };
    plotT = mkPlot(RT, 20); plotS = mkPlot(RS, 20);

    panelFrame(ctx, RT, `TEACHER — ${N} steps of the exact field`, T.n12);
    panelFrame(ctx, RS, `STUDENT — ${K} jump${K === 1 ? '' : 's'}, ${st.mode}`, T.accent);
    drawManifold(ctx, plotT.map, 1.9, 0.30);
    drawManifold(ctx, plotS.map, 1.9, st.mode === 'adversarial' && st.signal ? 0.55 : 0.30);

    // ---- teacher: the field at this t, then the trajectories so far ---------
    const nx = 11, ny = Math.max(5, Math.round(11 * RT.h / RT.w));
    const stepX = (RT.w - 8) / nx, stepY = (RT.h - 30) / ny;
    const cells = []; let vmax = 1e-6;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const px = RT.x + 4 + (i + 0.5) * stepX, py = RT.y + 22 + (j + 0.5) * stepY;
      const q = plotT.unmap(px, py), f = field(q.x, q.y, tcur);
      const mag = Math.hypot(f.ux, f.uy); if (mag > vmax) vmax = mag;
      cells.push({ px, py, mx: q.x, my: q.y, f, mag });
    }
    const alen = Math.min(stepX, stepY) * 0.85;
    for (const c of cells) {
      const k = alen * Math.min(1, c.mag / vmax) / Math.max(c.mag, 1e-6);
      r.arrow({ x: c.px, y: c.py }, { x: c.px + c.f.ux * k, y: c.py - c.f.uy * k },
        { color: T.violet, width: 1, head: 3.5, alpha: 0.10 + 0.35 * Math.min(1, c.mag / vmax) });
    }
    ctx.save(); ctx.lineWidth = 1.8; ctx.strokeStyle = alphaOf(T.n12, 0.85);
    for (let p = 0; p < P; p++) {
      const tr = M.trajs[p]; ctx.beginPath();
      for (let i = 0; i <= si; i++) { const q = plotT.map(tr[2 * i], tr[2 * i + 1]); if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y); }
      ctx.stroke();
    }
    ctx.restore();
    ctx.save();
    for (let p = 0; p < P; p++) {
      const q0 = plotT.map(M.starts[2 * p], M.starts[2 * p + 1]);
      const held = grab && grab.kind === 'start' && grab.i === p;
      ctx.strokeStyle = held ? T.accent : T.n9; ctx.lineWidth = held ? 2 : 1.2;
      ctx.beginPath(); ctx.arc(q0.x, q0.y, 5, 0, 6.2832); ctx.stroke();
      const tr = M.trajs[p], q = plotT.map(tr[2 * si], tr[2 * si + 1]);
      ctx.fillStyle = T.n13; ctx.beginPath(); ctx.arc(q.x, q.y, 3.2, 0, 6.2832); ctx.fill();
    }
    ctx.restore();
    r.label(`t = ${tcur.toFixed(3)}   step ${si}/${N}   h = 1/${N} = ${h.toFixed(4)}   ↗ field u(x,t)`,
      RT.x + 6, RT.y + RT.h - 6, { color: T.n10, font: '9.5px ui-monospace, monospace' });

    // ---- student: the training signal, then the jumps -----------------------
    // consistency: several points of ONE trajectory tied to their shared endpoint.
    if (st.signal && st.mode === 'consistency') {
      ctx.save();
      ctx.lineWidth = 1;
      for (let j = 0; j < 3; j++) {
        const idx = (j * 13 + 2) % NTRAIN;
        const z = [D.zs[2 * idx], D.zs[2 * idx + 1]], y = [D.ys[2 * idx], D.ys[2 * idx + 1]];
        const qe = plotS.map(y[0], y[1]);
        ctx.strokeStyle = alphaOf(T.gold, 0.55);
        if (ctx.setLineDash) ctx.setLineDash([3, 3]);
        for (const tt of [0.0, 0.25, 0.5, 0.75]) {
          const c = coefs(tt), qx = c.a * y[0] + c.s * z[0], qy = c.a * y[1] + c.s * z[1];
          const q = plotS.map(qx, qy);
          ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(qe.x, qe.y); ctx.stroke();
          ctx.fillStyle = alphaOf(T.gold, 0.9); ctx.beginPath(); ctx.arc(q.x, q.y, 2.2, 0, 6.2832); ctx.fill();
        }
        if (ctx.setLineDash) ctx.setLineDash([]);
        ctx.fillStyle = T.goldDeep; ctx.beginPath(); ctx.arc(qe.x, qe.y, 3.4, 0, 6.2832); ctx.fill();
      }
      ctx.restore();
    }
    // adversarial: the correction from the raw regression output onto the data.
    if (st.signal && st.mode === 'adversarial') {
      for (let p = 0; p < P; p++) {
        for (const hop of U.runs[p].hops) {
          const qa = plotS.map(hop.raw[0], hop.raw[1]), qb = plotS.map(hop.out[0], hop.out[1]);
          if (Math.hypot(qa.x - qb.x, qa.y - qb.y) < 2) continue;
          ctx.save(); ctx.strokeStyle = alphaOf(T.bad, 0.75); ctx.lineWidth = 1.1;
          ctx.beginPath(); ctx.arc(qa.x, qa.y, 3, 0, 6.2832); ctx.stroke(); ctx.restore();
          r.arrow(qa, qb, { color: T.bad, width: 1.1, head: 4, alpha: 0.7 });
        }
      }
    }

    // the jumps themselves, revealed with the transport
    const revealed = si >= N ? K : clamp(Math.floor(si * K / N) + (si > 0 ? 1 : 0), 0, K);
    ctx.save();
    for (let p = 0; p < P; p++) {
      const run = U.runs[p];
      const q0 = plotS.map(M.starts[2 * p], M.starts[2 * p + 1]);
      ctx.strokeStyle = T.n9; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(q0.x, q0.y, 5, 0, 6.2832); ctx.stroke();
      for (let k = 0; k < Math.min(revealed, run.hops.length); k++) {
        const hop = run.hops[k];
        r.arrow(plotS.map(hop.from[0], hop.from[1]), plotS.map(hop.out[0], hop.out[1]),
          { color: T.accent, width: 1.9, head: 6, alpha: 0.9 });
        if (hop.renoise && k + 1 < revealed) {
          const a = plotS.map(hop.out[0], hop.out[1]), b = plotS.map(hop.renoise[0], hop.renoise[1]);
          ctx.strokeStyle = alphaOf(T.warn, 0.8); ctx.lineWidth = 1.4;
          if (ctx.setLineDash) ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          if (ctx.setLineDash) ctx.setLineDash([]);
        }
      }
      // where the teacher's many steps ended, and the residual to it
      const te = M.tEnd[p], qt = plotS.map(te[0], te[1]), qs = plotS.map(run.end[0], run.end[1]);
      ctx.strokeStyle = alphaOf(T.ok, 0.9); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(qt.x, qt.y, 5, 0, 6.2832); ctx.stroke();
      if (revealed >= K) {
        ctx.strokeStyle = alphaOf(T.n10, 0.8); ctx.lineWidth = 1;
        if (ctx.setLineDash) ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(qs.x, qs.y); ctx.lineTo(qt.x, qt.y); ctx.stroke();
        if (ctx.setLineDash) ctx.setLineDash([]);
        ctx.fillStyle = T.accent; ctx.beginPath(); ctx.arc(qs.x, qs.y, 3.6, 0, 6.2832); ctx.fill();
      }
    }
    ctx.restore();

    // the two drag rails
    const railW = Math.min(150, RS.w * 0.38), railY = RS.y + RS.h - 16;
    railK = drawRail(ctx, r, RS.x + 10, railY, railW, K, 1, 8, `jumps K = ${K}`, T.accent);
    railBw = drawRail(ctx, r, RS.x + RS.w - railW - 10, railY, railW, st.bw, 0, 1,
      `strength ${st.bw.toFixed(2)} → h₀ = ${U.bw.toFixed(3)}`, T.violet);

    // ---- the seed sweep: everything fixed but the noise sample --------------
    const sy = top + panelH + 22, rowH = 30, lab = 62;
    const cw = (page.W - 2 * pad - lab) / NSTRIP;
    r.label('SWEEP THE SEED, HOLD EVERYTHING ELSE FIXED — each column is one noise sample',
      pad, sy - 8, { color: T.n11, font: '10px ui-monospace, monospace' });
    stripCells = [];
    const rows = [
      { key: 'teacher', pts: S.tEnd, y: sy, tag: `teacher ${N}` },
      { key: 'student', pts: U.strip.map((x) => x.end), y: sy + rowH + 4, tag: `student ${K}` },
    ];
    for (const row of rows) {
      r.label(row.tag, pad, row.y + rowH * 0.62, { color: T.n11, font: '9.5px ui-monospace, monospace' });
      for (let c = 0; c < NSTRIP; c++) {
        const p = row.pts[c], x = pad + lab + c * cw;
        const R2 = { x: x + 1, y: row.y, w: cw - 2, h: rowH };
        let dup = false;
        for (let q = 0; q < NSTRIP; q++) if (q !== c && Math.hypot(p[0] - row.pts[q][0], p[1] - row.pts[q][1]) < DUP_EPS) { dup = true; break; }
        ctx.save();
        ctx.fillStyle = alphaOf(sampleColor(p[0], p[1]), 0.72);
        ctx.fillRect(R2.x, R2.y, R2.w, R2.h);
        ctx.strokeStyle = dup ? T.bad : T.n5; ctx.lineWidth = dup ? 1.6 : 1;
        ctx.strokeRect(R2.x + 0.5, R2.y + 0.5, R2.w - 1, R2.h - 1);
        // a micro-plot of where in the data space this sample landed
        const msc = Math.min(R2.w, R2.h) / (2 * 1.9), mcx = R2.x + R2.w / 2, mcy = R2.y + R2.h / 2;
        ctx.fillStyle = alphaOf(T.n14, 0.20);
        for (let k = 0; k < MOONS.length; k += 3) ctx.fillRect(mcx + MOONS[k][0] * msc, mcy - MOONS[k][1] * msc, 1, 1);
        ctx.fillStyle = T.n14;
        ctx.beginPath(); ctx.arc(mcx + p[0] * msc, mcy - p[1] * msc, 2.4, 0, 6.2832); ctx.fill();
        ctx.restore();
        stripCells.push({ ...R2, c, p, dup, row: row.key });
      }
    }
    const my = sy + 2 * rowH + 22;
    r.label(`spread ${m.spS.toFixed(3)} vs teacher ${m.spT.toFixed(3)} = ${m.spPct.toFixed(1)}% of the teacher's (lower = LESS variety)`
      + `   ·   repeats ${(100 * m.dupS).toFixed(0)}% vs ${(100 * m.dupT).toFixed(0)}%   ·   modes ${m.modeS} vs ${m.modeT} of ${NSTRIP}`,
      pad, my, { color: m.spPct < 60 ? T.bad : T.n12, font: '10.5px ui-monospace, monospace' });
    r.label(`red outline = a near-identical twin elsewhere in the row (within ${DUP_EPS})   ·   swatch colour = the sample's position`,
      pad, my + 13, { color: T.n10, font: '9.5px ui-monospace, monospace' });

    page.probe = { spPct: m.spPct, spS: m.spS, spT: m.spT, K, bw: st.bw, mode: st.mode, offS: m.offS, offT: m.offT, resid: m.resid, ess: m.ess, dupS: m.dupS, modeS: m.modeS };

    // ---- hover: the arithmetic under the cursor -----------------------------
    if (page.pointer.over && !grab) {
      const pt = page.pointer; let tip = null;
      // a teacher step
      for (let p = 0; p < P && !tip; p++) {
        const tr = M.trajs[p], q = plotT.map(tr[2 * si], tr[2 * si + 1]);
        if (Math.hypot(pt.x - q.x, pt.y - q.y) < 9) {
          const x = tr[2 * si], y = tr[2 * si + 1], f = field(x, y, tcur);
          tip = `teacher · trajectory ${p} · step ${si}/${N}\n`
            + `x = (${x.toFixed(3)}, ${y.toFixed(3)})   t = ${tcur.toFixed(3)}\n`
            + `u(x,t) = (${f.ux.toFixed(3)}, ${f.uy.toFixed(3)})   h = 1/${N} = ${h.toFixed(4)}\n`
            + `x + u·h = (${(x + f.ux * h).toFixed(3)}, ${(y + f.uy * h).toFixed(3)})\n`
            + `${N - si} step${N - si === 1 ? '' : 's'} still to run`;
        }
      }
      // a student jump
      for (let p = 0; p < P && !tip; p++) {
        const run = U.runs[p];
        for (let k = 0; k < run.hops.length && !tip; k++) {
          const hop = run.hops[k], qb = plotS.map(hop.out[0], hop.out[1]);
          if (Math.hypot(pt.x - qb.x, pt.y - qb.y) < 9) {
            const te = M.tEnd[p];
            tip = `student jump ${k + 1}/${K} · trajectory ${p} · t = ${hop.t.toFixed(3)}\n`
              + `f(x,t) = Σ w_j·y_j over ${NTRAIN} distilled pairs,  h(t) = h₀·(σ+0.06) = ${hop.h.toFixed(3)}\n`
              + hop.top.map((e) => `  w[${e.j}] = ${e.w.toFixed(3)} → y = (${e.y[0].toFixed(2)}, ${e.y[1].toFixed(2)})`).join('\n') + '\n'
              + `ESS = ${hop.ess.toFixed(1)} of ${NTRAIN} endpoints averaged\n`
              + `raw f = (${hop.raw[0].toFixed(3)}, ${hop.raw[1].toFixed(3)})`
              + (st.mode === 'adversarial' ? `  →  manifold term → (${hop.out[0].toFixed(3)}, ${hop.out[1].toFixed(3)})\n` : '\n')
              + (hop.renoise ? `re-noise to t=${hop.renoise[2].toFixed(3)}: x = α·f + σ·z = (${hop.renoise[0].toFixed(3)}, ${hop.renoise[1].toFixed(3)})`
                : `teacher endpoint (${te[0].toFixed(3)}, ${te[1].toFixed(3)}) · residual ${Math.hypot(run.end[0] - te[0], run.end[1] - te[1]).toFixed(3)}`);
          }
        }
      }
      // a strip cell
      if (!tip) for (const cell of stripCells) {
        if (pt.x >= cell.x && pt.x <= cell.x + cell.w && pt.y >= cell.y && pt.y <= cell.y + cell.h) {
          const nm = nearestMode(cell.p[0], cell.p[1]);
          const other = cell.row === 'student' ? S.tEnd[cell.c] : U.strip[cell.c].end;
          tip = `${cell.row} · seed column ${cell.c} (same distribution, same fit, different noise sample)\n`
            + `sample = (${cell.p[0].toFixed(3)}, ${cell.p[1].toFixed(3)})   off-manifold ${nm.d.toFixed(3)}\n`
            + `the other model's sample from this same noise: (${other[0].toFixed(3)}, ${other[1].toFixed(3)})\n`
            + `distance between them ${Math.hypot(cell.p[0] - other[0], cell.p[1] - other[1]).toFixed(3)}\n`
            + (cell.dup ? `REPEAT — within ${DUP_EPS} of another column in this row` : 'distinct from every other column in this row');
          break;
        }
      }
      if (!tip && railK && Math.hypot(pt.x - railK.kx, pt.y - railK.y) < 14) tip = `drag: student jump count, 1..8\nfewer jumps = fewer re-noise draws = less variety`;
      if (!tip && railBw && Math.hypot(pt.x - railBw.kx, pt.y - railBw.y) < 14) tip = `drag: distillation strength → h₀ = ${U.bw.toFixed(3)}, working width h(t) = h₀·(σ(t)+0.06)\nwider h averages more teacher endpoints per answer (ESS now ${m.ess.toFixed(1)} of ${NTRAIN} at the first jump)`;
      if (tip) page.setTip(tip);
    }

    // ---- readout ------------------------------------------------------------
    let o = `${st.mode} student · ${K} jump${K === 1 ? '' : 's'} vs a ${N}-step teacher · distillation strength ${st.bw.toFixed(2)} (kernel h₀ = ${U.bw.toFixed(3)}, h(t) = h₀·(σ(t)+0.06)) · ${NTRAIN} distilled pairs · seed ${st.seed}    tier:${r.name}\n`;
    o += `step ${si}/${N}  t=${tcur.toFixed(3)}  ${s ? s.label : ''}\n`;
    o += `COST — sample spread ${m.spS.toFixed(4)} = ${m.spPct.toFixed(1)}% of the teacher's ${m.spT.toFixed(4)} (lower is worse here; 100% = teacher parity) · `
      + `${(100 * m.dupS).toFixed(0)}% of student samples have a near-identical twin vs ${(100 * m.dupT).toFixed(0)}% of the teacher's · `
      + `${m.modeS} of ${NSTRIP} distinct data modes reached vs the teacher's ${m.modeT}\n`;
    o += `WHY — each student answer is a weighted blend of ESS ${m.ess.toFixed(1)} of ${NTRAIN} teacher endpoints (ESS ${NTRAIN} = one answer for every input; ESS 1 = a copy of one training endpoint)\n`;
    o += `ACCURACY — mean |student − teacher| over the ${NSTRIP} swept seeds: ${m.resid.toFixed(4)} · off-manifold ${m.offS.toFixed(4)} vs the teacher's ${m.offT.toFixed(4)} (floor ≈ the data's own mode width ${MODE_SD.toFixed(2)})\n`;
    o += K === 1
      ? `JUMPS — this IS the one-jump run. Drag the K rail: the same fitted student is re-run at 1 jump as a paired control, so the two numbers below are measured side by side rather than argued.\n`
      : `JUMPS — paired control, the SAME student and the same seeds at 1 jump: spread ${m.sp1Pct.toFixed(1)}% of the teacher's and off-manifold ${m.off1.toFixed(4)}; at ${K} jumps, ${m.spPct.toFixed(1)}% and ${m.offS.toFixed(4)}. So going to ${K} jumps ${m.spPct > m.sp1Pct ? 'RAISED' : m.spPct < m.sp1Pct ? 'LOWERED' : 'left'} the spread and ${m.offS < m.off1 ? 'moved the output CLOSER to' : m.offS > m.off1 ? 'moved the output FURTHER from' : 'left the output the same distance from'} the manifold. Each extra jump re-noises before jumping again, which puts randomness back AND re-applies the averaging; which of the two wins is not a slogan, it is the pair of numbers above, and it changes with the strength.\n`;
    o += st.mode === 'adversarial'
      ? `The manifold term pulls each raw output ${(100 * ADV_PULL).toFixed(0)}% of the way to the nearest data mode — the stand-in for a discriminator (ADD) or a distribution-matching loss (DMD2). It is aimed at off-manifold error, which the number above prices directly; whether it also moves the spread is measured, not claimed — switch the training signal back and compare the percent.`
      : `Self-consistency alone is a REGRESSION onto teacher endpoints, and a regression under capacity pressure averages. That averaging is the whole cost: read the spread percent, then move the strength rail and read it again.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__fewStepDistillationPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k) => parseFloat(q.get(k));
  if (q.has('seed')) page.controls.set('seed', num('seed') | 0, { rebuild: true, silent: true });
  if (q.has('P')) page.controls.set('P', num('P') | 0, { silent: true });
  if (q.has('mode')) page.controls.set('mode', q.get('mode'), { silent: true });
  if (q.has('K')) page.controls.set('K', clamp(num('K') | 0, 1, 8), { silent: true });
  if (q.has('bw')) page.controls.set('bw', clamp(num('bw'), 0, 1), { silent: true });
  if (q.has('tsteps')) page.controls.set('tsteps', num('tsteps') | 0, { rebuild: true, silent: true });
  if (q.has('signal')) page.controls.set('signal', q.get('signal') === '1' || q.get('signal') === 'true', { silent: true });
  if (t) t.rebuild();
  // ?drag=i,x,y is the headless stand-in for dragging trajectory i's start.
  if (q.has('drag')) {
    const [i, x, y] = q.get('drag').split(',').map(Number);
    if (M && i >= 0 && 2 * i + 1 < M.starts.length) {
      M.starts[2 * i] = clamp(x, -3.2, 3.2); M.starts[2 * i + 1] = clamp(y, -3.2, 3.2);
      M.dirty = (M.dirty || 0) + 1;
    }
  }
  // ?hover=x,y fakes the cursor (canvas-space px) so the tooltip path renders.
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('hover') || q.has('drag')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
