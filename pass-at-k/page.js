// pass-at-k concept page -- why sharpening a distribution buys single-sample
// accuracy and sells the tail, and why that makes the two pass@k curves CROSS.
//
// Everything on screen is computed here from two distributions over candidate
// answers, never transcribed from a paper's plot:
//   base:  p_i        = softmax(logits_i)                    (broad)
//   RL:    q_i        = p_i^e * exp(beta * reinforced_i)  / Z (peaked)
//                       e = 1 + alpha,  reinforced_i = correct_i AND p_i >= floor
//   per problem:  s = SUM over correct i of that model's probability
//   pass@k     = mean over problems of  1 - (1 - s)^k        (k i.i.d. samples)
// The crossover is then read off the arithmetic, not asserted.
//
// The reason a crossover is possible at all is worth stating: for ONE problem,
// 1-(1-s)^k is monotone in s, so a model with the larger s wins at every k and
// the curves can never cross. The crossing lives in the AVERAGE over problems.
// Sharpening raises s where the model's favourite answer happens to be right
// and drives s toward 0 where it is wrong -- and a problem with s = 0 is never
// solved, at any k. Coverage is what the average is made of.
//
// Empirical source for the result this teaches: Yue et al., "Does Reinforcement
// Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base
// Model?", arXiv:2504.13837 (2025).
import { mount } from '../framework/layout.js';
import { softmax, seededRandn, rng } from '../framework/tensor.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

const KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const KMAX = KS[KS.length - 1];
const MAXN = 14, MAXM = 10;
const fmt3 = (x) => x.toFixed(3);
const pct = (x) => (x * 100).toFixed(1) + '%';

// ---------------------------------------------------------------- the data
// One "problem" = a set of candidate answers with base-model logits and a
// verifiable correct/wrong label per candidate. Deterministic in the seed, so a
// URL replays exactly one picture.
let cache = { key: '', probs: [] };
let geom = null;                 // rects captured in draw(), used by onPointer
let grab = null;                 // 'sharp' | {cand:i} while dragging
let edits = 0;                   // bumped by a bar drag, so the memo below misses
let memo = { key: '' };          // the whole model, recomputed only when an input moves

function buildProblems(seed, N, M) {
  const out = [];
  for (let m = 0; m < M; m++) {
    const logits = Array.from(seededRandn(seed * 1013 + m * 97 + 1, N, { std: 1.25 }));
    const r = rng(seed * 7919 + m * 131 + 3);
    const correct = new Array(N).fill(0).map(() => (r() < 0.32 ? 1 : 0));
    if (!correct.some((x) => x)) correct[Math.floor(r() * N)] = 1;   // every problem is solvable
    out.push({ logits, correct });
  }
  return out;
}

function ensure(st) {
  const key = `${st.seed | 0}|${st.n | 0}|${st.m | 0}`;
  if (cache.key !== key) cache = { key, probs: buildProblems(st.seed | 0, st.n | 0, st.m | 0) };
  return cache.probs;
}

/** RL distribution: sharpen the base by exponent e, reinforcing the correct
 *  candidates the base actually SAMPLES often enough to be discovered. */
function rlProbs(p, correct, e, beta, floor) {
  const w = p.map((pi, i) => Math.pow(pi, e) * Math.exp(beta * (correct[i] && pi >= floor ? 1 : 0)));
  const Z = w.reduce((a, b) => a + b, 0) || 1;
  return w.map((x) => x / Z);
}

const successOf = (probs, correct) => { let s = 0; for (let i = 0; i < probs.length; i++) if (correct[i]) s += probs[i]; return s; };

/** pass@k for every integer k in [1, KMAX], averaged over problems.
 *  Built incrementally: (1-s)^k from (1-s)^(k-1), so no pow() in the loop. */
function passCurve(ss) {
  const M = ss.length || 1, miss = ss.map(() => 1), out = new Float64Array(KMAX + 1);
  for (let k = 1; k <= KMAX; k++) {
    let acc = 0;
    for (let m = 0; m < ss.length; m++) { miss[m] *= (1 - ss[m]); acc += 1 - miss[m]; }
    out[k] = acc / M;
  }
  return out;
}

/** The headline: the smallest k at which the base model catches the RL model,
 *  plus where the RL model is furthest behind.
 *  EPS matters. Both curves saturate at 1.0 once k is large enough to find any
 *  answer with s > 0, so a bare `cr <= cb` test fires on a floating-point TIE at
 *  the top of the chart and reports a "crossing" that is worth 0.0% -- which is
 *  not a crossing, it is two curves finishing together. A real crossing has to
 *  cost something, so it must open a gap of at least EPS. */
const EPS = 5e-4;
function crossoverK(cb, cr) {
  let worstK = 1, worst = 0;
  for (let k = 1; k <= KMAX; k++) { const d = cb[k] - cr[k]; if (d > worst) { worst = d; worstK = k; } }
  if (cr[1] <= cb[1]) return { k: 0, why: 'rl-never-leads', worst, worstK };
  for (let k = 2; k <= KMAX; k++) if (cr[k] < cb[k] - EPS) return { k, why: 'crossed', worst, worstK };
  return { k: 0, why: 'no-crossing', worst, worstK };
}

// ------------------------------------------------------------------ drawing
function drawTrack(page, rect, val, min, max, label) {
  const ctx = page.ctx, r = page.renderer;
  const t = (val - min) / (max - min), hx = rect.x + t * rect.w;
  r.label(label, rect.x - 10, rect.y + 4, { color: T.n11, font: '11.5px ui-monospace, monospace', align: 'right' });
  ctx.save();
  ctx.strokeStyle = T.n6; ctx.lineWidth = 3; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(rect.x, rect.y); ctx.lineTo(rect.x + rect.w, rect.y); ctx.stroke();
  ctx.strokeStyle = T.warn; ctx.beginPath(); ctx.moveTo(rect.x, rect.y); ctx.lineTo(hx, rect.y); ctx.stroke();
  ctx.fillStyle = T.warn; ctx.beginPath(); ctx.arc(hx, rect.y, 6.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = T.n0; ctx.beginPath(); ctx.arc(hx, rect.y, 2.6, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
  r.label(`α = ${val.toFixed(2)}  (drag ↔)`, rect.x + rect.w + 9, rect.y + 4, { color: T.warn, font: '11.5px ui-monospace, monospace' });
}

mount({
  mount: 'body',
  title: 'pass@k — sharpening buys the first sample and sells the tail',
  blurb: 'Reinforcement learning on a verifiable reward concentrates probability mass onto reasoning paths the base model could already produce. That is a real gain at k = 1 — and a loss of coverage, which only shows up when you are allowed several tries. Left: the candidate answers for one problem, base (outline) vs RL (filled), with the verifier\'s ✓/✗. Right: pass@k = mean over problems of 1 − (1 − s)^k, computed live from those distributions for both models. Drag the α track, or drag any candidate bar, and watch where the curves cross. The empirical result this mechanism explains is Yue et al., arXiv:2504.13837.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  compare: {
    key: 'sharp', a: 0.3, b: 3.2,
    labelA: 'weak sharpening — RL leads at every k', labelB: 'strong sharpening — the base overtakes',
  },
  challenges: [
    {
      goal: 'Make the curves cross at k ≤ 8 — RL wins the first sample and loses by the eighth.',
      hint: 'push the α track right (strong sharpening) and raise the discovery floor, so some correct answers are never reinforced.',
      check: (api) => { const c = api.probe.cross ?? 0; return { solved: c >= 2 && c <= 8, detail: c ? `crossing at k = ${c} (need ≤ 8)` : 'no crossing yet — RL leads at every k, or never leads' }; },
    },
    {
      goal: 'Find sharpening that helps at EVERY k — no crossing anywhere up to k = 1024.',
      hint: 'drop the discovery floor to 0 (every correct path gets reinforced) and keep α small. Then RL adds mass to correct answers without pruning any.',
      check: (api) => ({ solved: (api.probe.cross ?? 1) === 0 && (api.probe.rlLeads === true), detail: (api.probe.cross ? `still crosses at k = ${api.probe.cross}` : (api.probe.rlLeads ? 'no crossing — RL leads throughout' : 'RL does not lead at k = 1')) }),
    },
  ],
  controls: (c, page) => {
    c.slider('sharp', { label: 'RL sharpening α  (exponent e = 1+α)', min: 0, max: 4, step: 0.05, value: 2.2, format: (v) => (+v).toFixed(2) });
    c.slider('tilt', { label: 'reward tilt β  (reinforcement strength)', min: 0, max: 6, step: 0.1, value: 1.5, format: (v) => (+v).toFixed(1) });
    c.slider('floor', { label: 'discovery floor — a correct path below it is never sampled in training, so never reinforced', min: 0, max: 0.25, step: 0.005, value: 0.06, format: (v) => (+v).toFixed(3) });
    c.stepper('n', { label: 'candidate answers per problem', min: 3, max: MAXN, value: 8 });
    c.stepper('m', { label: 'problems in the benchmark', min: 1, max: MAXM, value: 6 });
    c.stepper('focus', { label: 'problem shown on the left', min: 1, max: MAXM, value: 1 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7 });
    c.transport({ compute: () => KS.map((k) => ({ k, label: `k = ${k} independent sample${k > 1 ? 's' : ''} per problem` })), speed: 2.2, loop: true });
  },

  // Direct manipulation: drag the α track, or drag a candidate bar to change
  // that answer's base probability (in logit space; the rest renormalize).
  onPointer: (page, ev) => {
    const st = page.state;
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!geom) return;
    const { track, cands, strip, N } = geom;
    if (ev.type === 'down') {
      grab = null;
      if (ev.y > track.y - 14 && ev.y < track.y + 14 && ev.x > track.x - 12 && ev.x < track.x + track.w + 12) grab = 'sharp';
      else if (ev.x >= cands.x && ev.x <= cands.x + cands.w && ev.y >= cands.y && ev.y <= cands.y + cands.h) {
        const i = Math.floor((ev.x - cands.x) / (cands.w / N));
        if (i >= 0 && i < N) grab = { cand: i };
      } else if (strip && ev.x >= strip.x && ev.x <= strip.x + strip.w && ev.y >= strip.y && ev.y <= strip.y + strip.h) {
        const j = Math.floor((ev.y - strip.y) / strip.rowH);
        if (j >= 0 && j < (st.m | 0)) page.controls.set('focus', j + 1);
      }
    }
    if (grab === 'sharp' && page.pointer.down) {
      const t = Math.max(0, Math.min(1, (ev.x - track.x) / track.w));
      page.controls.set('sharp', Math.round(t * 4 * 20) / 20);
    } else if (grab && grab.cand != null && page.pointer.down) {
      const pr = cache.probs[Math.min(cache.probs.length - 1, (st.focus | 0) - 1)];
      if (pr) { pr.logits[grab.cand] = Math.max(-6, Math.min(6, pr.logits[grab.cand] - ev.dy * 0.035)); edits++; }
      page.redraw();
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const problems = ensure(st);
    const N = st.n | 0, M = Math.min(problems.length, st.m | 0);
    const e = 1 + st.sharp, beta = st.tilt, floor = st.floor;
    const fi = Math.max(0, Math.min(M - 1, (st.focus | 0) - 1));

    // ---- the math, for every problem.
    // Memoized on every input that can move it (including the drag counter):
    // the curves are 1024 k-values wide and the page redraws on hover, so
    // recomputing them per frame would spend the whole budget on arithmetic
    // nothing asked to change.
    const mkey = `${cache.key}|${e}|${beta}|${floor}|${M}|${edits}`;
    if (memo.key !== mkey) {
      const rowsN = problems.slice(0, M).map((pr) => {
        const pArr = Array.from(softmax(Float32Array.from(pr.logits)));
        const q = rlProbs(pArr, pr.correct, e, beta, floor);
        return { p: pArr, q, correct: pr.correct, sB: successOf(pArr, pr.correct), sR: successOf(q, pr.correct) };
      });
      const cbN = passCurve(rowsN.map((x) => x.sB)), crN = passCurve(rowsN.map((x) => x.sR));
      memo = { key: mkey, rows: rowsN, cb: cbN, cr: crN, cross: crossoverK(cbN, crN) };
    }
    const rows = memo.rows, cb = memo.cb, cr = memo.cr, cross = memo.cross;
    page.probe = { cross: cross.k, rlLeads: cr[1] > cb[1], why: cross.why };

    const s = page.step();
    const kNow = s ? s.k : 1;

    // ---- layout
    const pad = 14;
    const track = { x: pad + 168, y: 22, w: Math.max(80, page.W * 0.42 - 30) };
    const topY = 52;
    const colW = (page.W - pad * 2) * 0.44;
    const leftX = pad, rightX = pad + colW + 26, rightW = page.W - pad - rightX;
    const cands = { x: leftX + 4, y: topY + 26, w: colW - 8, h: Math.max(66, page.H * 0.32) };
    const stripY = cands.y + cands.h + 82;
    const rowH = Math.min(18, Math.max(11, (page.H - 12 - stripY) / Math.max(1, M)));
    const strip = { x: leftX + 4, y: stripY, w: colW - 8, h: rowH * M, rowH };
    geom = { track, cands, strip, N };

    drawTrack(page, track, st.sharp, 0, 4, 'RL sharpening');

    // ================================================== left: one problem
    const F = rows[fi];
    r.label(`problem ${fi + 1} of ${M} — ${N} candidate answers`, leftX, topY + 2, { color: T.n13, font: '12px ui-monospace, monospace' });
    const dom = Math.max(1e-6, ...F.p, ...F.q);
    const colWi = cands.w / N, bw = Math.min(30, colWi * 0.56);
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cands.x, cands.y + cands.h); ctx.lineTo(cands.x + cands.w, cands.y + cands.h); ctx.stroke();
    for (let i = 0; i < N; i++) {
      const cx = cands.x + i * colWi + colWi / 2, base = cands.y + cands.h;
      const hb = (F.p[i] / dom) * (cands.h - 6), hq = (F.q[i] / dom) * (cands.h - 6);
      const ok = !!F.correct[i], seen = F.p[i] >= floor;
      // base: hollow outline (the broad distribution)
      ctx.strokeStyle = alphaOf(T.accent, 0.9); ctx.lineWidth = 1.4;
      ctx.strokeRect(cx - bw / 2, base - hb, bw, hb);
      ctx.fillStyle = rgbaToken('n14', 0.05); ctx.fillRect(cx - bw / 2, base - hb, bw, hb);
      // RL: filled, inset (the peaked distribution)
      ctx.fillStyle = alphaOf(ok ? T.warn : T.n9, ok ? 0.85 : 0.55);
      ctx.fillRect(cx - bw / 2 + 4, base - hq, bw - 8, hq);
      // verifier mark
      ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillStyle = ok ? (seen ? T.ok : T.n9) : T.n8;
      ctx.fillText(ok ? (seen ? '✓' : '✓') : '✗', cx, base + 13);
      if (ok && !seen) { ctx.strokeStyle = T.n8; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx - 6, base + 9.5); ctx.lineTo(cx + 6, base + 9.5); ctx.stroke(); }
      ctx.fillStyle = T.n9; ctx.font = '9.5px ui-monospace, monospace';
      ctx.fillText(String(i), cx, base + 25);
    }
    ctx.restore();
    r.label('outline = base p', cands.x, cands.y - 6, { color: T.accent, font: '10.5px ui-monospace, monospace' });
    r.label('filled = RL q', cands.x + cands.w, cands.y - 6, { color: T.warn, font: '10.5px ui-monospace, monospace', align: 'right' });

    // per-problem success probabilities for the focused problem
    r.label(`s_base = ${fmt3(F.sB)}    s_RL = ${fmt3(F.sR)}    (Σ over ✓)`,
      cands.x, cands.y + cands.h + 40, { color: T.n12, font: '11px ui-monospace, monospace' });
    r.label('✓ correct · struck = below the floor · drag bars ↕',
      cands.x, cands.y + cands.h + 55, { color: T.n10, font: '9.5px ui-monospace, monospace' });

    // ================================================== left: problem strip
    r.label('success probability s, per problem', leftX, strip.y - 10, { color: T.n11, font: '11px ui-monospace, monospace' });
    ctx.save();
    const labW = 26, barW = strip.w - labW - 4;
    for (let m = 0; m < M; m++) {
      const y = strip.y + m * rowH;
      if (m === fi) { ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(strip.x - 3, y - 1, strip.w + 6, rowH); }
      ctx.fillStyle = m === fi ? T.n13 : T.n10; ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText(`P${m + 1}`, strip.x, y + rowH * 0.62);
      const hh = Math.max(2, rowH * 0.32);
      ctx.fillStyle = alphaOf(T.accent, 0.75); ctx.fillRect(strip.x + labW, y + 1, barW * rows[m].sB, hh);
      ctx.fillStyle = alphaOf(T.warn, 0.85); ctx.fillRect(strip.x + labW, y + 2 + hh, barW * rows[m].sR, hh);
      if (rows[m].sR < 1e-4) { ctx.fillStyle = T.bad; ctx.font = '9px ui-monospace, monospace'; ctx.fillText('s_RL ≈ 0 — unreachable at any k', strip.x + labW + 4, y + rowH * 0.92); }
    }
    ctx.restore();

    // ================================================== right: the curves
    const ch = { x: rightX + 34, y: topY + 14, w: rightW - 40, h: page.H - topY - 60 };
    const xFor = (k) => ch.x + (Math.log2(Math.max(1, k)) / Math.log2(KMAX)) * ch.w;
    const yFor = (v) => ch.y + ch.h - v * ch.h;
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) { const y = yFor(g / 4); ctx.beginPath(); ctx.moveTo(ch.x, y); ctx.lineTo(ch.x + ch.w, y); ctx.stroke(); }
    ctx.restore();
    for (let g = 0; g <= 4; g++) r.label((g / 4).toFixed(2), ch.x - 6, yFor(g / 4) + 3.5, { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'right' });
    for (const k of KS) {
      const x = xFor(k);
      r.label(String(k), x, ch.y + ch.h + 14, { color: k === kNow ? T.n13 : T.n10, font: (k === kNow ? 'bold ' : '') + '9.5px ui-monospace, monospace', align: 'center' });
    }
    r.label('k  (samples drawn per problem, log scale)', ch.x + ch.w / 2, ch.y + ch.h + 30, { color: T.n11, font: '10.5px ui-monospace, monospace', align: 'center' });
    r.label('pass@k', ch.x, ch.y - 6, { color: T.n13, font: '12px ui-monospace, monospace' });

    const curve = (tab, color) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      const STEPS = 160;
      for (let i = 0; i <= STEPS; i++) {
        const k = Math.max(1, Math.min(KMAX, Math.round(Math.pow(2, (i / STEPS) * Math.log2(KMAX)))));
        const x = xFor(k), y = yFor(tab[k]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.restore();
    };
    curve(cb, T.accent);
    curve(cr, T.warn);
    ctx.save();
    for (const k of KS) {
      ctx.fillStyle = T.accent; ctx.beginPath(); ctx.arc(xFor(k), yFor(cb[k]), 2.6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = T.warn; ctx.beginPath(); ctx.arc(xFor(k), yFor(cr[k]), 2.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();

    // crossover marker -- the closing readout, drawn where it happens
    if (cross.k) {
      const x = xFor(cross.k), pulse = 5;
      ctx.save();
      ctx.strokeStyle = alphaOf(T.bad, 0.75); ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, ch.y); ctx.lineTo(x, ch.y + ch.h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = T.bad; ctx.beginPath(); ctx.arc(x, yFor(cb[cross.k]), pulse, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      const lx = x + 6 > ch.x + ch.w - 96 ? x - 6 : x + 6;
      r.label(`cross at k = ${cross.k}`, lx, ch.y + 26, { color: T.bad, font: 'bold 11px ui-monospace, monospace', align: x + 6 > ch.x + ch.w - 96 ? 'right' : 'left' });
    }

    // current k from the transport
    ctx.save();
    ctx.strokeStyle = alphaOf(T.n14, 0.35); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xFor(kNow), ch.y); ctx.lineTo(xFor(kNow), ch.y + ch.h); ctx.stroke();
    ctx.fillStyle = T.accent; ctx.beginPath(); ctx.arc(xFor(kNow), yFor(cb[kNow]), 4.6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = T.warn; ctx.beginPath(); ctx.arc(xFor(kNow), yFor(cr[kNow]), 4.6, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    // legend
    const lgy = ch.y + ch.h - 6;
    ctx.save();
    ctx.fillStyle = T.accent; ctx.fillRect(ch.x + 8, lgy - 20, 14, 3);
    ctx.fillStyle = T.warn; ctx.fillRect(ch.x + 8, lgy - 8, 14, 3);
    ctx.restore();
    // The practical consequence, and why the trade is usually worth taking: at
    // k = 1 there is nothing to pick WITH. Everything right of k = 1 assumes a
    // verifier good enough to recognize the right answer among the k you drew --
    // which is exactly what best-of-N selection needs and what shipping lacks.
    r.label('k = 1 is what ships: no verifier, no choosing. k > 1 assumes one.',
      ch.x + 8, lgy - 32, { color: T.n10, font: '10px ui-monospace, monospace' });
    r.label(`base model — pass@${kNow} = ${pct(cb[kNow])}`, ch.x + 28, lgy - 15, { color: T.accent, font: '11px ui-monospace, monospace' });
    r.label(`RL model   — pass@${kNow} = ${pct(cr[kNow])}`, ch.x + 28, lgy - 3, { color: T.warn, font: '11px ui-monospace, monospace' });

    // ---- hover-to-inspect
    if (page.pointer.over && !grab) {
      const px = page.pointer.x, py = page.pointer.y;
      if (px >= cands.x && px <= cands.x + cands.w && py >= cands.y - 6 && py <= cands.y + cands.h + 28) {
        const i = Math.floor((px - cands.x) / colWi);
        if (i >= 0 && i < N) {
          const ok = !!F.correct[i], seen = F.p[i] >= floor, boost = ok && seen ? beta : 0;
          const w = Math.pow(F.p[i], e) * Math.exp(boost);
          page.setTip(
            `candidate ${i} — ${ok ? 'CORRECT' : 'wrong'}${ok && !seen ? ', below the discovery floor' : ''}\n` +
            `base  p = ${fmt3(F.p[i])}\n` +
            `w = p^${e.toFixed(2)} · e^${boost.toFixed(2)} = ${w.toExponential(3)}\n` +
            `RL    q = w / Σw = ${fmt3(F.q[i])}\n` +
            `drag ↕ to change this answer's base probability`);
        }
      } else if (px >= ch.x - 12 && px <= ch.x + ch.w + 12 && py >= ch.y - 10 && py <= ch.y + ch.h + 10) {
        let bk = KS[0], bd = Infinity;
        for (const k of KS) { const d = Math.abs(xFor(k) - px); if (d < bd) { bd = d; bk = k; } }
        const terms = rows.slice(0, Math.min(3, M)).map((x) => `1−(1−${fmt3(x.sB)})^${bk}`).join(' + ');
        const termsR = rows.slice(0, Math.min(3, M)).map((x) => `1−(1−${fmt3(x.sR)})^${bk}`).join(' + ');
        const tail = M > 3 ? ' + …' : '';
        page.setTip(
          `k = ${bk}\n` +
          `base pass@${bk} = [ ${terms}${tail} ] / ${M} = ${fmt3(cb[bk])}\n` +
          `RL   pass@${bk} = [ ${termsR}${tail} ] / ${M} = ${fmt3(cr[bk])}\n` +
          `${cr[bk] > cb[bk] ? 'RL ahead' : cr[bk] < cb[bk] ? 'base ahead' : 'tied'} by ${pct(Math.abs(cr[bk] - cb[bk]))}`);
      } else if (px >= strip.x && px <= strip.x + strip.w && py >= strip.y && py <= strip.y + strip.h) {
        const j = Math.min(M - 1, Math.max(0, Math.floor((py - strip.y) / rowH)));
        const row = rows[j], nCorrect = row.correct.reduce((a, b) => a + b, 0);
        page.setTip(
          `problem ${j + 1} — ${nCorrect} of ${N} candidates correct\n` +
          `s_base = ${fmt3(row.sB)}   →  pass@1024 = ${fmt3(1 - Math.pow(1 - row.sB, KMAX))}\n` +
          `s_RL   = ${fmt3(row.sR)}   →  pass@1024 = ${fmt3(1 - Math.pow(1 - row.sR, KMAX))}\n` +
          `click to show this problem's candidates`);
      }
    }

    // ---- readout
    const dead = rows.filter((x) => x.sR < 1e-4 && x.sB > 1e-4).length;
    let out = `pass@k = (1/${M}) Σ over problems of  1 − (1 − s)^k      s = Σ p over the ✓ candidates      q ∝ p^${e.toFixed(2)} · e^(β·reinforced)      tier:${r.name}\n`;
    out += `k = ${kNow}:  base ${pct(cb[kNow])}   ·   RL ${pct(cr[kNow])}   ·   ${cr[kNow] >= cb[kNow] ? 'RL ahead' : 'BASE ahead'} by ${pct(Math.abs(cr[kNow] - cb[kNow]))}\n`;
    out += (cross.worst < EPS && Math.abs(cr[1] - cb[1]) < EPS)
      ? `α = ${st.sharp.toFixed(2)}, β = ${beta.toFixed(1)} — the two models are the same distribution, so there is nothing to trade yet. Sharpen to buy k = 1.`
      : cross.why === 'crossed'
      ? `the curves CROSS at k = ${cross.k} — RL is ${pct(cr[1] - cb[1])} ahead at k = 1, and ${pct(cross.worst)} behind at its worst (k = ${cross.worstK}).`
      : cross.why === 'no-crossing'
        ? `no crossing up to k = ${KMAX} — this sharpening helps at every k (nothing correct was pruned).`
        : `RL does not lead even at k = 1 here — nothing is being bought, so there is nothing to trade.`;
    if (dead) out += `   ${dead} of ${M} problem${dead > 1 ? 's are' : ' is'} now unreachable for the RL model at ANY k (s_RL ≈ 0) while the base still solves ${dead > 1 ? 'them' : 'it'}.`;
    page.setReadout(out);
  },
}).then((page) => {
  window.__passkPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  for (const k of ['sharp', 'tilt', 'floor', 'seed']) if (q.has(k)) page.controls.set(k, parseFloat(q.get(k)));
  for (const k of ['n', 'm', 'focus']) if (q.has(k)) page.controls.set(k, parseInt(q.get(k), 10));
  // ?drag=i,logit sets one candidate's base logit (headless stand-in for a bar drag).
  if (q.has('drag')) {
    const [i, val] = q.get('drag').split(',').map(Number);
    const pr = cache.probs[Math.max(0, (page.state.focus | 0) - 1)];
    if (pr && i >= 0 && i < pr.logits.length) pr.logits[i] = val;
  }
  // ?hover=x,y fakes the cursor (headless stand-in: --screenshot has no pointer).
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('hover') || q.has('drag')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
