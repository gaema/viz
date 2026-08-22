// verifier-selection -- three ways to spend ONE fixed budget of N generations,
// and the one that gets WORSE the more you spend.
//
// The three strategies, all given the same budget on the same benchmark:
//   best-of-N        sample N, score each with a verifier / reward model, keep
//                    the top-scoring one.
//   majority vote    sample N, return the most common ANSWER. No verifier.
//                    (self-consistency)
//   beam search      spend the budget on breadth at INTERMEDIATE steps under a
//                    process reward model, instead of on N complete attempts.
//
// Everything on screen is computed here, from a model of one benchmark, never
// transcribed from a paper's plot:
//
//   per problem j:  difficulty d_j,  solve probability  s_j = sigmoid(k(a - d_j))
//                   answers 0..A-1, exactly one correct;
//                   p[correct] = s_j, the rest of the mass spread over the wrong
//                   answers with a concentration g (large g = one wrong answer
//                   the model keeps giving).
//   verifier:       a FIXED score per answer, v_i = z_i + mu*[i correct], with
//                   mu = sqrt(2)*Phi^-1(q) so that q is exactly the probability
//                   the verifier ranks a random correct answer above a random
//                   wrong one. q < 1 means some wrong answer somewhere outscores
//                   the right one -- and the more you sample, the likelier you
//                   are to find it.
//
//   best-of-N   P(pick i) = C_i^N - (C_i - p_i)^N,  C_i = total probability of
//               the answers scoring at or below v_i.  (Exact: best-of-N returns
//               the highest-scoring answer that got sampled at all.)
//   majority    P(i wins) summed over count vectors, ties broken uniformly --
//               exact, by a dynamic program over the wrong answers' counts.
//   beam        per-step correctness r = s_j^(1/D); each step generates
//               c = floor(N/D) continuations from the live beams and keeps b of
//               them; the process reward model ranks perfectly with probability
//               2q-1 and at random otherwise. Exact DP over the number of
//               still-on-track beams.
//
// So the crossings are read off the arithmetic. Drag the verifier accuracy down
// and the best-of-N curve turns over -- it is optimising the verifier's score,
// not correctness, and a bigger N is a wider search for whatever the verifier
// wrongly loves. The majority-vote curve cannot fail that way -- it never
// consults a verifier, so nothing there is gameable. It fails a DIFFERENT way:
// it converges on the model's most common answer, so when the model is
// confidently and consistently wrong, more votes make the wrong consensus
// firmer. Raise "wrong-answer consistency" and majority vote can end LOWER than
// it started while best-of-N is untouched -- two failure modes with no overlap.
//
// Empirical sources for the claims this mechanism explains (this page shows its
// OWN model's behaviour, not their numbers):
//   Cobbe et al., "Training Verifiers to Solve Math Word Problems",
//     arXiv:2110.14168 (2021) -- sample-and-rank with a trained verifier.
//   Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in
//     Language Models", arXiv:2203.11171 (2022) -- majority vote over samples.
//   Lightman et al., "Let's Verify Step by Step", arXiv:2305.20050 (2023) --
//     process supervision, i.e. rewarding intermediate steps.
//   Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective
//     than Scaling Model Parameters", arXiv:2408.03314 (2024) -- which strategy
//     wins depends on problem difficulty and budget; none dominates.
import { mount } from '../framework/layout.js';
import { rng, seededRandn } from '../framework/tensor.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

const NS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const NMAX = NS[NS.length - 1];
const MAXP = 12, MAXA = 7;
const pct = (x) => (x * 100).toFixed(1) + '%';
const f3 = (x) => x.toFixed(3);
/** A ratio as a PERCENT OF the named baseline. Higher is better on this axis. */
const rel = (x, base) => (base > 1e-9 ? (x / base * 100).toFixed(0) + '%' : 'n/a');

// log-factorial table, for the exact multinomial arithmetic below
const LF = new Float64Array(NMAX + 2);
for (let i = 1; i <= NMAX + 1; i++) LF[i] = LF[i - 1] + Math.log(i);
const lnBinom = (n, k) => LF[n] - LF[k] - LF[n - k];

/** Inverse normal CDF (Acklam). Turns the verifier's stated pairwise accuracy
 *  into the score separation that produces it. */
function probit(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  let q, r;
  if (p < 0.02425) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 0.97575) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
const sepFor = (q) => Math.SQRT2 * probit(Math.min(0.9985, Math.max(0.5001, q)));

// ------------------------------------------------------------------ the bench
// A problem carries only its LATENTS: how much harder than average it is, which
// answer is correct, how the wrong mass is shaped, and the verifier's noise per
// answer. Everything the sliders control is applied on top, per frame, so a drag
// never rebuilds the benchmark under the reader.
let bench = { key: '', probs: [] };
let geom = null;      // rects captured in draw(), read by onPointer
let grab = null;      // which track is being dragged
let memo = { key: '' };

function buildBench(seed, M, A) {
  const out = [];
  for (let m = 0; m < M; m++) {
    const r = rng(seed * 7919 + m * 131 + 3);
    const dz = Array.from(seededRandn(seed * 1013 + m * 97 + 1, 1))[0];
    const wz = Array.from(seededRandn(seed * 3301 + m * 211 + 5, A));
    const vz = Array.from(seededRandn(seed * 6151 + m * 307 + 11, A));
    out.push({ dz, wz, vz, ci: Math.floor(r() * A) % A });
  }
  return out;
}
function ensure(st) {
  const key = `${st.seed | 0}|${st.probs | 0}|${st.cands | 0}`;
  if (bench.key !== key) bench = { key, probs: buildBench(st.seed | 0, st.probs | 0, st.cands | 0) };
  return bench.probs;
}

/** The model's answer distribution for one problem. */
function answerDist(pr, A, macc, diff, spread, wcon) {
  const d = Math.max(0.01, Math.min(0.99, diff + spread * pr.dz));
  const s = 1 / (1 + Math.exp(-6 * (macc - d)));
  const w = new Array(A).fill(0);
  let Z = 0;
  for (let i = 0; i < A; i++) if (i !== pr.ci) { w[i] = Math.exp(wcon * pr.wz[i]); Z += w[i]; }
  const p = new Array(A);
  for (let i = 0; i < A; i++) p[i] = i === pr.ci ? s : (Z ? (1 - s) * w[i] / Z : 0);
  return { p, s, d };
}
/** The verifier's FIXED score per answer. Same answer, same score, every draw --
 *  which is exactly why more samples cannot fix a wrong one. */
const verifierScores = (pr, A, mu) => pr.vz.slice(0, A).map((z, i) => z + (i === pr.ci ? mu : 0));

// ----------------------------------------------------------- the three spends
/** best-of-N, exact. Returns the selection probability of every answer.
 *  Best-of-N returns the highest-scoring answer PRESENT in the N draws, so
 *  P(pick i) = P(all N draws score <= v_i) - P(all N draws score < v_i). */
function bestOfNSel(p, v, N) {
  const A = p.length, order = p.map((_, i) => i).sort((x, y) => v[x] - v[y]);
  const sel = new Array(A).fill(0);
  let cum = 0;
  for (const i of order) { const lo = cum; cum += p[i]; sel[i] = Math.pow(cum, N) - Math.pow(lo, N); }
  return sel;
}

/** majority vote, exact, ties broken uniformly at random among the tied answers.
 *  P(answer `focus` wins) = SUM over k of P(focus drawn k times) *
 *  P(every other answer drawn < k times, weighted 1/(1+ties)) -- the inner term
 *  by a DP over the other answers' counts, carrying the number of exact ties. */
function majoritySel(p, N, focus) {
  const A = p.length, pc = p[focus];
  if (pc <= 0) return 0;
  const rest = 1 - pc, wq = [];
  for (let i = 0; i < A; i++) if (i !== focus) wq.push(rest > 1e-12 ? p[i] / rest : 0);
  const W = wq.length;
  let total = 0;
  for (let k = 1; k <= N; k++) {
    const m = N - k;
    let lp = lnBinom(N, k) + k * Math.log(pc);
    if (m > 0) { if (rest <= 1e-12) continue; lp += m * Math.log(rest); }
    const pk = Math.exp(lp);
    if (!(pk > 1e-15)) continue;
    let f = new Float64Array((m + 1) * (W + 1)); f[0] = 1;
    for (let j = 0; j < W; j++) {
      const g = new Float64Array((m + 1) * (W + 1));
      const lq = wq[j] > 0 ? Math.log(wq[j]) : -Infinity;
      for (let mm = 0; mm <= m; mm++) for (let t = 0; t <= W; t++) {
        const val = f[mm * (W + 1) + t];
        if (!val) continue;
        const cmax = Math.min(k, m - mm);
        for (let cnt = 0; cnt <= cmax; cnt++) {
          const term = cnt === 0 ? 1 : (wq[j] > 0 ? Math.exp(cnt * lq - LF[cnt]) : 0);
          if (cnt > 0 && term === 0) break;
          g[(mm + cnt) * (W + 1) + t + (cnt === k ? 1 : 0)] += val * term;
        }
      }
      f = g;
    }
    let G = 0;
    for (let t = 0; t <= W; t++) G += f[m * (W + 1) + t] / (1 + t);
    total += pk * G * Math.exp(LF[m]);
  }
  return total;
}

const hyper = (c, K, b, x) => (x < 0 || x > K || x > b || b - x > c - K) ? 0
  : Math.exp(lnBinom(K, x) + lnBinom(c - K, b - x) - lnBinom(c, b));

/** beam / stepwise search, exact. The budget buys BREADTH at each of the D
 *  steps instead of N whole attempts, and the process reward model decides what
 *  survives. Returns {acc, live[]} where live[t] = E[on-track beams] / b after
 *  step t -- the quantity a process reward model is actually gambling with. */
function beamSearch(s, N, D, bWant, q) {
  const r = Math.pow(Math.max(1e-12, s), 1 / D);
  const c = Math.max(1, Math.floor(N / D));
  const b = Math.max(1, Math.min(bWant, c));
  const qr = Math.max(0, Math.min(1, 2 * q - 1));
  let dist = new Float64Array(b + 1); dist[1] = 1;
  let live = 1;
  const trace = [1];
  for (let step = 0; step < D; step++) {
    const nd = new Float64Array(b + 1);
    for (let n = 0; n <= b; n++) {
      const P = dist[n];
      if (!P) continue;
      if (n === 0) { nd[0] += P; continue; }
      const rho = Math.min(1, n / live) * r;
      for (let K = 0; K <= c; K++) {
        const pk = Math.exp(lnBinom(c, K) + (K ? K * Math.log(rho) : 0) + (c - K ? (c - K) * Math.log(1 - rho) : 0));
        if (!(pk > 1e-14)) continue;
        nd[Math.min(K, b)] += P * pk * qr;
        if (qr < 1) for (let x = 0; x <= Math.min(K, b); x++) nd[x] += P * pk * (1 - qr) * hyper(c, K, b, x);
      }
    }
    dist = nd; live = b;
    let e = 0;
    for (let n = 0; n <= b; n++) e += dist[n] * n;
    trace.push(e / b);
  }
  let acc = 0;
  for (let n = 1; n <= b; n++) acc += dist[n] * (qr + (1 - qr) * n / b);
  return { acc, trace, c, b, r, qr };
}

// ------------------------------------------------------------------- the sweep
function evaluate(st) {
  const A = st.cands | 0, M = st.probs | 0, mu = sepFor(st.vacc);
  const rows = ensure(st).slice(0, M).map((pr) => {
    const { p, s, d } = answerDist(pr, A, st.macc, st.diff, st.spread, st.wcon);
    return { p, s, d, v: verifierScores(pr, A, mu), ci: pr.ci };
  });
  // The named baseline: ONE generation, no budget and no verifier -- the model's
  // single most likely answer. Every percent on this page is a percent of this.
  let greedy = 0;
  for (const x of rows) { let bi = 0; for (let i = 1; i < A; i++) if (x.p[i] > x.p[bi]) bi = i; if (bi === x.ci) greedy++; }
  greedy /= M || 1;

  const bo = [], mv = [], be = [];
  for (const N of NS) {
    let a = 0, b = 0, c = 0;
    for (const x of rows) {
      a += bestOfNSel(x.p, x.v, N)[x.ci];
      b += majoritySel(x.p, N, x.ci);
      c += beamSearch(x.s, N, st.depth | 0, st.beam | 0, st.vacc).acc;
    }
    bo.push(a / M); mv.push(b / M); be.push(c / M);
  }
  // Where best-of-N turns over: its peak, and how much it has given back by the
  // largest budget on the ladder. That is the page's claim, in two numbers.
  let pk = 0;
  for (let i = 1; i < NS.length; i++) if (bo[i] > bo[pk]) pk = i;
  const giveback = bo[pk] - bo[NS.length - 1];
  return { rows, greedy, bo, mv, be, peakI: pk, giveback, mu, M, A };
}

// -------------------------------------------------------------------- drawing
function drawTrack(page, rect, val, min, max, label, color, fmt) {
  const ctx = page.ctx, r = page.renderer;
  const t = (val - min) / (max - min), hx = rect.x + t * rect.w;
  r.label(label, rect.x - 9, rect.y + 4, { color: T.n11, font: '10.5px ui-monospace, monospace', align: 'right' });
  ctx.save();
  ctx.strokeStyle = T.n6; ctx.lineWidth = 3; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(rect.x, rect.y); ctx.lineTo(rect.x + rect.w, rect.y); ctx.stroke();
  ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(rect.x, rect.y); ctx.lineTo(hx, rect.y); ctx.stroke();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(hx, rect.y, 5.6, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = T.n0; ctx.beginPath(); ctx.arc(hx, rect.y, 2.2, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
  r.label(fmt(val), rect.x + rect.w + 8, rect.y + 4, { color, font: '10.5px ui-monospace, monospace' });
}

mount({
  mount: 'body',
  title: 'verifier selection — three ways to spend the same test-time budget',
  blurb: 'You are allowed N generations per problem. You can sample N and keep whichever one a verifier scores highest (best-of-N); sample N and return the most common answer, consulting nothing (majority vote / self-consistency); or spend the same N on breadth at intermediate steps under a process reward model (beam search). Every curve here is computed in-page from one benchmark of problems — a distribution over candidate answers per problem, plus a verifier whose accuracy you control. Drag the verifier accuracy down and best-of-N TURNS OVER: more samples is a wider search for whatever a flawed verifier wrongly loves. Majority vote cannot fail that way — it never asks a verifier — and fails its own way instead, when the model is confidently and consistently wrong. Empirical grounding: Cobbe et al. arXiv:2110.14168, Wang et al. arXiv:2203.11171, Lightman et al. arXiv:2305.20050, Snell et al. arXiv:2408.03314.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  compare: {
    key: 'vacc', a: 0.95, b: 0.62,
    labelA: 'accurate verifier — best-of-N keeps paying', labelB: 'flawed verifier — best-of-N gets WORSE with N',
  },
  challenges: [
    {
      goal: 'Make best-of-N peak and then LOSE at least 5 points of accuracy by the largest budget, while majority vote keeps rising.',
      hint: 'drag the verifier-accuracy track left. Below roughly 0.75 the search starts finding wrong answers the verifier likes better than the right one.',
      check: (api) => {
        const g = api.probe.giveback ?? 0, up = (api.probe.mvEnd ?? 0) > (api.probe.mvStart ?? 0);
        return {
          solved: g >= 0.05 && up,
          detail: `best-of-N gives back ${pct(g)} after its peak (need ≥ 5.0%)`
            + (up ? '' : ' — and majority vote is not rising here, so both are failing at once'),
        };
      },
    },
    {
      goal: 'Break majority vote instead: make MORE budget make it WORSE — accuracy at N = 64 below accuracy at N = 1 — with the verifier left accurate, so this is not the verifier\'s fault.',
      hint: 'raise "wrong-answer consistency" toward 3. One wrong answer the model keeps repeating outvotes a correct answer it reaches several different ways, and extra votes only make the consensus firmer. Keep verifier accuracy at 0.85 or above so best-of-N stays healthy.',
      check: (api) => {
        const d = (api.probe.mvEnd ?? 0) - (api.probe.mvStart ?? 0);
        if ((api.probe.vacc ?? 0) < 0.85) return { solved: false, detail: 'verifier accuracy is below 0.85 — that is the OTHER failure mode' };
        return { solved: d < -0.005, detail: `majority vote goes ${d < 0 ? 'DOWN' : 'up'} ${pct(Math.abs(d))} from N = 1 to N = ${NMAX} (need it down)` };
      },
    },
  ],

  controls: (c) => {
    c.slider('vacc', { label: 'verifier accuracy — P(scores a correct answer above a wrong one)', min: 0.5, max: 0.99, step: 0.01, value: 0.78, format: (v) => (+v).toFixed(2) });
    c.slider('macc', { label: 'model accuracy — the generator\'s own ability', min: 0.05, max: 0.95, step: 0.01, value: 0.55, format: (v) => (+v).toFixed(2) });
    c.slider('diff', { label: 'problem difficulty (benchmark mean)', min: 0.05, max: 0.95, step: 0.01, value: 0.5, format: (v) => (+v).toFixed(2) });
    c.slider('spread', { label: 'difficulty spread across the benchmark', min: 0, max: 0.5, step: 0.01, value: 0.28, format: (v) => (+v).toFixed(2) });
    c.slider('wcon', { label: 'wrong-answer consistency — how much the wrong mass piles on ONE answer', min: 0, max: 3, step: 0.05, value: 1.4, format: (v) => (+v).toFixed(2) });
    c.stepper('depth', { label: 'reasoning steps per solution D (beam only)', min: 2, max: 6, value: 3 });
    c.stepper('beam', { label: 'beam width b (beam only)', min: 1, max: 6, value: 4 });
    c.stepper('cands', { label: 'distinct candidate answers per problem', min: 3, max: MAXA, value: 6 });
    c.stepper('probs', { label: 'problems in the benchmark', min: 4, max: MAXP, value: 10 });
    c.stepper('focus', { label: 'problem shown on the left', min: 1, max: MAXP, value: 1 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7 });
    c.transport({ compute: () => NS.map((n) => ({ n, label: `budget N = ${n} generation${n > 1 ? 's' : ''} per problem` })), speed: 2.0, loop: true });
  },

  // Direct manipulation: the three tracks across the top are the three axes the
  // page is about. Grab one and every curve recomputes under your hand.
  onPointer: (page, ev) => {
    const st = page.state;
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!geom) return;
    const { tracks, strip } = geom;
    if (ev.type === 'down') {
      grab = null;
      for (const tk of tracks) {
        if (ev.y > tk.y - 9 && ev.y < tk.y + 9 && ev.x > tk.x - 10 && ev.x < tk.x + tk.w + 10) { grab = tk; break; }
      }
      if (!grab && strip && ev.x >= strip.x && ev.x <= strip.x + strip.w && ev.y >= strip.y && ev.y <= strip.y + strip.h) {
        const j = Math.floor((ev.y - strip.y) / strip.rowH);
        if (j >= 0 && j < (st.probs | 0)) page.controls.set('focus', j + 1);
      }
    }
    if (grab && page.pointer.down) {
      const t = Math.max(0, Math.min(1, (ev.x - grab.x) / grab.w));
      const raw = grab.min + t * (grab.max - grab.min);
      page.controls.set(grab.key, Math.round(raw / grab.step) * grab.step);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const A = st.cands | 0, M = st.probs | 0;

    // Memoized on every input that can move the arithmetic: the sweep is twelve
    // budgets x M problems of exact multinomial DP, and the page redraws on
    // every hover.
    const mkey = `${bench.key}|${st.vacc}|${st.macc}|${st.diff}|${st.spread}|${st.wcon}|${st.depth}|${st.beam}|${M}|${A}`;
    if (memo.key !== mkey) memo = Object.assign({ key: mkey }, evaluate(st));
    const { rows, greedy, bo, mv, be, peakI, giveback, mu } = memo;

    const sIdx = Math.max(0, NS.indexOf(page.step() ? page.step().n : 1));
    const N = NS[sIdx];
    const fi = Math.max(0, Math.min(rows.length - 1, (st.focus | 0) - 1));
    const F = rows[fi];

    page.probe = { giveback, mvStart: mv[0], mvEnd: mv[NS.length - 1], boEnd: bo[NS.length - 1], vacc: st.vacc, peakN: NS[peakI] };

    // ------------------------------------------------------------ layout
    const pad = 14;
    const colW = (page.W - pad * 2) * 0.43;
    const leftX = pad, rightX = pad + colW + 30, rightW = page.W - pad - rightX;
    const tX = leftX + 96, tW = Math.max(60, colW - 96 - 32);
    const tracks = [
      { key: 'vacc', x: tX, y: 14, w: tW, min: 0.5, max: 0.99, step: 0.01, label: 'verifier ⇄', color: T.accent, fmt: (v) => v.toFixed(2) },
      { key: 'macc', x: tX, y: 31, w: tW, min: 0.05, max: 0.95, step: 0.01, label: 'model acc ⇄', color: T.ok, fmt: (v) => v.toFixed(2) },
      { key: 'diff', x: tX, y: 48, w: tW, min: 0.05, max: 0.95, step: 0.01, label: 'difficulty ⇄', color: T.warn, fmt: (v) => v.toFixed(2) },
    ];
    const topY = 68;
    const cands = { x: leftX + 4, y: topY + 14, w: colW - 8, h: Math.max(48, page.H * 0.215) };
    const base = cands.y + cands.h;
    const beamY = base + 74;
    const stripY = beamY + 58;
    const rowH = Math.min(15, Math.max(8, (page.H - 10 - stripY) / Math.max(1, M)));
    const strip = { x: leftX + 4, y: stripY, w: colW - 8, h: rowH * M, rowH };
    geom = { tracks, cands, strip, A };

    for (const tk of tracks) drawTrack(page, tk, st[tk.key], tk.min, tk.max, tk.label, tk.color, tk.fmt);

    // ============================================ left: one problem, in detail
    const boSel = bestOfNSel(F.p, F.v, N);
    const mvSel = F.p.map((_, i) => majoritySel(F.p, N, i));
    const bm = beamSearch(F.s, N, st.depth | 0, st.beam | 0, st.vacc);
    r.label(`problem ${fi + 1}/${M} · difficulty ${F.d.toFixed(2)} · one sample: ${pct(F.s)}`,
      leftX, topY, { color: T.n13, font: '10.5px ui-monospace, monospace' });

    const cw = cands.w / A, bw = Math.min(26, cw * 0.5);
    const domP = Math.max(1e-6, ...F.p);
    const vLo = Math.min(...F.v), vHi = Math.max(...F.v), vSpan = Math.max(0.4, vHi - vLo);
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cands.x, base); ctx.lineTo(cands.x + cands.w, base); ctx.stroke();
    for (let i = 0; i < A; i++) {
      const cx = cands.x + i * cw + cw / 2, ok = i === F.ci;
      // how often the model produces this answer
      const h = (F.p[i] / domP) * (cands.h - 22);
      ctx.fillStyle = alphaOf(ok ? T.ok : T.n8, ok ? 0.8 : 0.5);
      ctx.fillRect(cx - bw / 2, base - h, bw, h);
      // the verifier's fixed score for it, as a tick on the same column
      const vy = cands.y + (1 - (F.v[i] - vLo) / vSpan) * (cands.h - 26) + 2;
      ctx.strokeStyle = T.accent; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx - bw / 2 - 3, vy); ctx.lineTo(cx + bw / 2 + 3, vy); ctx.stroke();
      ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillStyle = ok ? T.ok : T.n9;
      ctx.fillText(ok ? '✓' : '✗', cx, base + 11);
      ctx.fillStyle = T.n9; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(String(i), cx, base + 21);
      // who picks it, at this budget: best-of-N (blue) and majority (violet)
      const pw = Math.max(3, bw + 4);
      ctx.fillStyle = rgbaToken('n14', 0.07); ctx.fillRect(cx - pw / 2, base + 26, pw, 9);
      ctx.fillStyle = alphaOf(T.accent, 0.9); ctx.fillRect(cx - pw / 2, base + 26, pw * boSel[i], 4);
      ctx.fillStyle = alphaOf(T.violet, 0.9); ctx.fillRect(cx - pw / 2, base + 31, pw * mvSel[i], 4);
    }
    ctx.restore();
    r.label('bar = P(model gives this answer) · tick = verifier score', cands.x, cands.y - 5, { color: T.n10, font: '9px ui-monospace, monospace' });
    r.label(`P(picks it) at N = ${N}:`, cands.x, base + 47, { color: T.n11, font: '9px ui-monospace, monospace' });
    ctx.save();
    ctx.fillStyle = T.accent; ctx.fillRect(cands.x + 128, base + 43, 9, 3);
    ctx.fillStyle = T.violet; ctx.fillRect(cands.x + 200, base + 43, 9, 3);
    ctx.restore();
    r.label('best-of-N', cands.x + 141, base + 47, { color: T.accent, font: '9px ui-monospace, monospace' });
    r.label('majority', cands.x + 213, base + 47, { color: T.violet, font: '9px ui-monospace, monospace' });

    // ---- beam is a different SHAPE of spend: it never sees a whole answer, so
    // it has no bar in the panel above. What it gambles with is how many of its
    // b beams are still on a correct path after each step.
    r.label(`beam — keep b = ${bm.b} of c = ${bm.c} continuations per step, ${st.depth} steps`,
      leftX, beamY - 10, { color: T.teal, font: '9.5px ui-monospace, monospace' });
    ctx.save();
    const bwid = (colW - 16) / (bm.trace.length || 1);
    for (let t = 0; t < bm.trace.length; t++) {
      const x = leftX + 4 + t * bwid, hh = 18 * bm.trace[t];
      ctx.fillStyle = rgbaToken('n14', 0.07); ctx.fillRect(x, beamY, bwid - 5, 18);
      ctx.fillStyle = alphaOf(T.teal, 0.85); ctx.fillRect(x, beamY + 18 - hh, bwid - 5, hh);
      ctx.fillStyle = T.n10; ctx.font = '8.5px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText(t === 0 ? 'start' : 's' + t, x + (bwid - 5) / 2, beamY + 27);
    }
    ctx.restore();
    r.label('fraction of beams still on a correct path — beam never scores a whole answer',
      leftX + 4, beamY + 38, { color: T.n10, font: '8.5px ui-monospace, monospace' });

    // ============================================ left: the whole benchmark
    r.label(`per-problem accuracy at N = ${N} — click a row to open it`,
      leftX, strip.y - 7, { color: T.n11, font: '9px ui-monospace, monospace' });
    ctx.save();
    const labW = 24, barW = strip.w - labW - 4, hh = Math.max(1.6, rowH * 0.26);
    for (let m = 0; m < M; m++) {
      const y = strip.y + m * rowH, x = rows[m];
      if (m === fi) { ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(strip.x - 3, y - 1, strip.w + 6, rowH); }
      ctx.fillStyle = m === fi ? T.n13 : T.n10; ctx.font = '8.5px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText(`P${m + 1}`, strip.x, y + rowH * 0.6);
      const a = bestOfNSel(x.p, x.v, N)[x.ci], b = majoritySel(x.p, N, x.ci), c = beamSearch(x.s, N, st.depth | 0, st.beam | 0, st.vacc).acc;
      ctx.fillStyle = alphaOf(T.accent, 0.8); ctx.fillRect(strip.x + labW, y + 0.5, barW * a, hh);
      ctx.fillStyle = alphaOf(T.violet, 0.8); ctx.fillRect(strip.x + labW, y + 1 + hh, barW * b, hh);
      ctx.fillStyle = alphaOf(T.teal, 0.8); ctx.fillRect(strip.x + labW, y + 1.5 + hh * 2, barW * c, hh);
    }
    ctx.restore();

    // ============================================ right: the three curves
    const ch = { x: rightX + 32, y: 42, w: rightW - 38, h: page.H - 42 - 42 };
    const xFor = (n) => ch.x + (Math.log2(Math.max(1, n)) / Math.log2(NMAX)) * ch.w;
    const yFor = (v) => ch.y + ch.h - v * ch.h;
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) { const y = yFor(g / 4); ctx.beginPath(); ctx.moveTo(ch.x, y); ctx.lineTo(ch.x + ch.w, y); ctx.stroke(); }
    ctx.restore();
    for (let g = 0; g <= 4; g++) r.label((g / 4).toFixed(2), ch.x - 6, yFor(g / 4) + 3.5, { color: T.n10, font: '9px ui-monospace, monospace', align: 'right' });
    for (const n of NS) r.label(String(n), xFor(n), ch.y + ch.h + 13, { color: n === N ? T.n13 : T.n10, font: (n === N ? 'bold ' : '') + '9px ui-monospace, monospace', align: 'center' });
    r.label('N  (generations of budget per problem, log scale)', ch.x + ch.w / 2, ch.y + ch.h + 28, { color: T.n11, font: '10px ui-monospace, monospace', align: 'center' });
    r.label('benchmark accuracy', ch.x, ch.y - 6, { color: T.n13, font: '11.5px ui-monospace, monospace' });

    // the named baseline: one generation, no verifier
    ctx.save();
    ctx.strokeStyle = alphaOf(T.n9, 0.9); ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(ch.x, yFor(greedy)); ctx.lineTo(ch.x + ch.w, yFor(greedy)); ctx.stroke();
    ctx.restore();
    r.label(`baseline: greedy single sample = ${pct(greedy)} = 100%`,
      ch.x + 4, yFor(greedy) - 5, { color: T.n10, font: '9px ui-monospace, monospace' });

    const curve = (tab, color) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      for (let i = 0; i < NS.length; i++) { const x = xFor(NS[i]), y = yFor(tab[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke();
      for (let i = 0; i < NS.length; i++) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(xFor(NS[i]), yFor(tab[i]), 2.6, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
    };
    curve(be, T.teal);
    curve(mv, T.violet);
    curve(bo, T.accent);

    // the turnover marker -- the page's claim, drawn where it happens
    if (giveback > 1e-3 && peakI < NS.length - 1) {
      const x = xFor(NS[peakI]);
      ctx.save();
      ctx.strokeStyle = alphaOf(T.bad, 0.8); ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, ch.y); ctx.lineTo(x, ch.y + ch.h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = T.bad; ctx.beginPath(); ctx.arc(x, yFor(bo[peakI]), 5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      const right = x > ch.x + ch.w - 150;
      r.label(`best-of-N peaks at N = ${NS[peakI]}, then gives back ${pct(giveback)}`,
        x + (right ? -7 : 7), ch.y + 14, { color: T.bad, font: 'bold 10px ui-monospace, monospace', align: right ? 'right' : 'left' });
    }

    // current budget from the transport
    ctx.save();
    ctx.strokeStyle = alphaOf(T.n14, 0.32); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xFor(N), ch.y); ctx.lineTo(xFor(N), ch.y + ch.h); ctx.stroke();
    for (const [tab, col] of [[bo, T.accent], [mv, T.violet], [be, T.teal]]) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(xFor(N), yFor(tab[sIdx]), 4.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    // legend, reported as a percent of the named baseline
    const ly = ch.y + ch.h - 16;
    ctx.save();
    ctx.fillStyle = alphaOf(T.n0, 0.82); ctx.fillRect(ch.x + 4, ly - 34, 268, 42);
    ctx.fillStyle = T.accent; ctx.fillRect(ch.x + 10, ly - 26, 13, 3);
    ctx.fillStyle = T.violet; ctx.fillRect(ch.x + 10, ly - 14, 13, 3);
    ctx.fillStyle = T.teal; ctx.fillRect(ch.x + 10, ly - 2, 13, 3);
    ctx.restore();
    r.label(`best-of-N  ${pct(bo[sIdx])} = ${rel(bo[sIdx], greedy)} of baseline`, ch.x + 29, ly - 22, { color: T.accent, font: '10px ui-monospace, monospace' });
    r.label(`majority   ${pct(mv[sIdx])} = ${rel(mv[sIdx], greedy)} of baseline`, ch.x + 29, ly - 10, { color: T.violet, font: '10px ui-monospace, monospace' });
    r.label(`beam       ${pct(be[sIdx])} = ${rel(be[sIdx], greedy)} of baseline`, ch.x + 29, ly + 2, { color: T.teal, font: '10px ui-monospace, monospace' });

    // ------------------------------------------------------ hover-to-inspect
    if (page.pointer.over && !grab) {
      const px = page.pointer.x, py = page.pointer.y;
      if (px >= ch.x - 14 && px <= ch.x + ch.w + 14 && py >= ch.y - 12 && py <= ch.y + ch.h + 12) {
        let bi = 0, bd = Infinity;
        for (let i = 0; i < NS.length; i++) { const d = Math.abs(xFor(NS[i]) - px); if (d < bd) { bd = d; bi = i; } }
        const n = NS[bi], ex = rows[fi];
        const sel = bestOfNSel(ex.p, ex.v, n);
        const order = ex.p.map((_, i) => i).sort((x, y) => ex.v[x] - ex.v[y]);
        let cum = 0, cAt = 0;
        for (const i of order) { cum += ex.p[i]; if (i === ex.ci) cAt = cum; }
        const bmh = beamSearch(ex.s, n, st.depth | 0, st.beam | 0, st.vacc);
        page.setTip(
          `N = ${n} generations per problem\n` +
          `best-of-N  ${pct(bo[bi])}   ${rel(bo[bi], greedy)} of greedy single sample\n` +
          `majority   ${pct(mv[bi])}   ${rel(mv[bi], greedy)} of greedy single sample\n` +
          `beam       ${pct(be[bi])}   ${rel(be[bi], greedy)} of greedy single sample\n` +
          `— on problem ${fi + 1}, the arithmetic —\n` +
          `best-of-N picks ✓ with  C^N − (C−p)^N\n` +
          `  C = ${f3(cAt)} (mass scoring ≤ the ✓ answer), p = ${f3(ex.p[ex.ci])}\n` +
          `  ${f3(Math.pow(cAt, n))} − ${f3(Math.pow(cAt - ex.p[ex.ci], n))} = ${f3(sel[ex.ci])}\n` +
          `majority picks ✓ with P(✓ is the plurality) = ${f3(majoritySel(ex.p, n, ex.ci))}\n` +
          `beam keeps b = ${bmh.b} of c = ${bmh.c}/step, r = s^(1/${st.depth}) = ${f3(bmh.r)}\n` +
          `  → ${f3(bmh.acc)}   (process reward ranks correctly ${pct(bmh.qr)} of the time)`);
      } else if (px >= cands.x && px <= cands.x + cands.w && py >= cands.y - 8 && py <= base + 40) {
        const i = Math.floor((px - cands.x) / cw);
        if (i >= 0 && i < A) {
          const ok = i === F.ci;
          const order = F.p.map((_, j) => j).sort((x, y) => F.v[x] - F.v[y]);
          let cum = 0, cAt = 0;
          for (const j of order) { cum += F.p[j]; if (j === i) cAt = cum; }
          page.setTip(
            `answer ${i} — ${ok ? 'CORRECT' : 'wrong'}\n` +
            `model gives it   p = ${f3(F.p[i])}\n` +
            `verifier scores it  v = ${F.v[i].toFixed(3)}${ok ? `  (= noise ${(F.v[i] - mu).toFixed(3)} + separation ${mu.toFixed(3)})` : ''}\n` +
            `best-of-N picks it: C^N − (C−p)^N with C = ${f3(cAt)}\n` +
            `  = ${f3(Math.pow(cAt, N))} − ${f3(Math.pow(cAt - F.p[i], N))} = ${f3(boSel[i])}\n` +
            `majority picks it:  ${f3(mvSel[i])}   (most common of ${N} draws, ties split)\n` +
            (ok ? 'more budget widens the search — including for wrong answers scoring above this one'
                : (F.v[i] > F.v[F.ci] ? '⚠ this WRONG answer outscores the correct one — best-of-N converges onto it'
                                      : 'scores below the correct answer, so best-of-N passes over it')));
        }
      } else if (px >= strip.x && px <= strip.x + strip.w && py >= strip.y && py <= strip.y + strip.h) {
        const j = Math.min(M - 1, Math.max(0, Math.floor((py - strip.y) / rowH)));
        const x = rows[j];
        const a = bestOfNSel(x.p, x.v, N)[x.ci];
        page.setTip(
          `problem ${j + 1} — difficulty ${x.d.toFixed(2)}, single sample solves it ${pct(x.s)}\n` +
          `at N = ${N}:  best-of-N ${pct(a)} · majority ${pct(majoritySel(x.p, N, x.ci))} · beam ${pct(beamSearch(x.s, N, st.depth | 0, st.beam | 0, st.vacc).acc)}\n` +
          `verifier's top-scoring answer here is ${x.v.indexOf(Math.max(...x.v)) === x.ci ? 'the CORRECT one' : 'a WRONG one — best-of-N converges onto it'}\n` +
          `click to open this problem on the left`);
      } else if (px >= leftX && px <= leftX + colW && py >= beamY - 14 && py <= beamY + 42) {
        page.setTip(
          `beam search, problem ${fi + 1}\n` +
          `per-step correctness r = s^(1/${st.depth}) = ${f3(bm.r)}\n` +
          `budget N = ${N} → c = floor(N/${st.depth}) = ${bm.c} continuations per step, keep b = ${bm.b}\n` +
          `process reward model ranks correctly ${pct(bm.qr)} of the time, at random otherwise\n` +
          `on-track beams after each step: ${bm.trace.map((v) => f3(v)).join(' → ')}\n` +
          `final accuracy ${pct(bm.acc)} — beam never scores a whole answer, it prunes prefixes`);
      }
    }

    // ------------------------------------------------------------- readout
    const bestNow = bo[sIdx] >= mv[sIdx] && bo[sIdx] >= be[sIdx] ? 'best-of-N'
      : mv[sIdx] >= be[sIdx] ? 'majority vote' : 'beam search';
    let out = `budget N = ${N} per problem, ${M} problems, ${A} candidate answers each · verifier accuracy ${st.vacc.toFixed(2)} (score separation μ = ${mu.toFixed(3)}) · tier:${r.name}\n`;
    out += `best-of-N ${pct(bo[sIdx])} (${rel(bo[sIdx], greedy)} of greedy single sample) · majority ${pct(mv[sIdx])} (${rel(mv[sIdx], greedy)}) · beam ${pct(be[sIdx])} (${rel(be[sIdx], greedy)}) · baseline greedy single sample ${pct(greedy)} = 100%. Higher is better; 100% = parity with one generation.\n`;
    out += `at N = ${N} the best spend of this budget is ${bestNow}.\n`;
    const mvA = mv[0], mvB = mv[NS.length - 1];
    if (giveback > 1e-3 && peakI < NS.length - 1) {
      out += `TURNOVER: best-of-N peaks at N = ${NS[peakI]} (${pct(bo[peakI])}) and falls to ${pct(bo[NS.length - 1])} at N = ${NMAX} — it gives back ${pct(giveback)}, `
        + `because a verifier that is right ${pct(st.vacc)} of the time is a TARGET, and ${NMAX} samples is a wide search for whatever it wrongly scores highest.\n`;
    } else {
      out += `no turnover on this ladder — best-of-N rises to ${pct(bo[NS.length - 1])} at N = ${NMAX}. Drag verifier accuracy below about 0.75 and it will peak and then decline.\n`;
    }
    out += `majority vote goes ${pct(mvA)} → ${pct(mvB)} over the same budget. It cannot be gamed the way best-of-N is, because it consults no verifier — `
      + (mvB < mvA - 1e-3
        ? `but it is DOWN ${pct(mvA - mvB)} here anyway, for the other reason: more votes converge on the model's most common answer, and on these problems that answer is often wrong.`
        : `and here it is UP ${pct(mvB - mvA)}. Raise "wrong-answer consistency" until the model's most common answer is the wrong one, and this curve falls too — from consensus, not from a bad verifier.`);
    out += `\nWhere each one lands: as N → ∞ majority vote converges on the MODAL answer, so its limit is exactly the greedy single-sample baseline (${pct(greedy)}) — at a finite N it can sit either side of that. This is a property of THIS model, where sampling is i.i.d. over whole answers; a real language model's greedy decoding is per-token argmax and need not produce the modal answer, which is why self-consistency does beat greedy decoding in practice.`;
    page.setReadout(out);
  },
}).then((page) => {
  window.__verifierSelectionPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  for (const k of ['vacc', 'macc', 'diff', 'spread', 'wcon', 'seed']) if (q.has(k)) page.controls.set(k, parseFloat(q.get(k)));
  for (const k of ['depth', 'beam', 'cands', 'probs', 'focus']) if (q.has(k)) page.controls.set(k, parseInt(q.get(k), 10));
  // ?hover=x,y fakes the cursor (headless stand-in: --screenshot has no pointer).
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
