// speculative-decoding concept page -- how a small draft model buys tokens from
// a big one without changing what the big one would have said.
//
// Plain decode emits ONE token per target forward pass: a single token's matmul
// is a skinny matrix-vector product, so the math units mostly wait on memory.
// Speculation trades that idle width for work: a cheap draft proposes k tokens,
// the target runs ONE forward over all k+1 positions (they batch the way a
// prompt does), and the verifier walks the proposals left to right, keeping them
// while the target agrees and stopping at the FIRST disagreement -- where it
// emits the target's OWN token instead. So a round yields at least 1 and at most
// k+1 tokens, and the rejected tail is thrown away: that discarded draft compute
// is the price of the trade.
//
// Everything on screen is really computed here: the target and draft
// distributions, the accept test min(1, p(x)/q(x)) against a seeded uniform, the
// residual distribution the corrected token is drawn from, and the token
// accounting. A seeded generator means one URL replays one exact run.
//
// Interactive: the transport steps the four phases of a round (propose /
// verify / accept test / commit) and loops; drag the draft-quality and
// draft-cost handles on the canvas; k and rounds are steppers; hover any
// proposal for the comparison that decided it.
import { mount } from '../framework/layout.js';
import { softmax, seededRandn, rng } from '../framework/tensor.js';
import { T, alphaOf, inkOn } from '../framework/theme.js';

const VOCAB = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'dog', 'ran', 'far', 'then', 'slept'];
const V = VOCAB.length;
const PHASES = ['propose', 'verify', 'accept test', 'commit'];
const PHASE_TEXT = [
  'PROPOSE — the small draft model runs k times, cheaply, guessing the next k tokens.',
  'VERIFY — one target forward covers all k+1 positions at once (they batch like a prompt does).',
  'ACCEPT TEST — walk left to right: keep while the target agrees, stop at the FIRST disagreement.',
  'COMMIT — keep the accepted prefix plus the target’s own token; discard the rejected tail.',
];

// --- the toy two-model setup ------------------------------------------------
// Both "models" are deterministic functions of (seed, round, slot), so a URL
// replays a run exactly. The draft is the target blended with its own
// idiosyncratic beliefs: agreement=1 makes it the target (every proposal is
// accepted), agreement=0 makes it a different model that agrees only by luck.
const targetDist = (seed, r, j) => softmax(seededRandn(seed * 7919 + r * 97 + j * 13 + 1, V, { std: 1.35 }));
const draftBias = (seed, r, j) => softmax(seededRandn(seed * 7919 + r * 97 + j * 13 + 500003, V, { std: 1.35 }));
const sampleIdx = (dist, u) => { let c = 0; for (let i = 0; i < dist.length; i++) { c += dist[i]; if (u <= c) return i; } return dist.length - 1; };

let run = null, runSig = '';
let geom = null;                  // rects captured in draw() for hit-testing
let grab = null;                  // 'quality' | 'cost' while dragging a handle

// Build the whole run up front. The uniforms come from one per-round stream and
// every slot draws the same number of them, so changing the draft's agreement
// re-tests the SAME luck (common random numbers) instead of reshuffling the
// run -- which is what makes sweeping the handle read as a clean cause-effect.
function buildRun(st) {
  const K = st.k | 0, R = st.rounds | 0, a = st.quality, seed = st.seed | 0;
  const rounds = [];
  let tokens = 0, wasted = 0;
  for (let r = 0; r < R; r++) {
    const next = rng(seed * 131071 + r * 7 + 3);
    const slots = [];
    for (let j = 0; j < K; j++) {
      const p = targetDist(seed, r, j), b = draftBias(seed, r, j);
      const q = new Float32Array(V);
      for (let i = 0; i < V; i++) q[i] = a * p[i] + (1 - a) * b[i];
      const u1 = next(), u2 = next(), u3 = next();
      const x = sampleIdx(q, u1);
      const ratio = q[x] > 1e-12 ? p[x] / q[x] : 0;
      const acc = u2 <= Math.min(1, ratio);
      // On a rejection the target does NOT just take its argmax: it draws from
      // the residual normalize(max(0, p - q)). That is the detail which makes
      // the whole scheme produce exactly the target's own distribution.
      const res = new Float32Array(V); let s = 0;
      for (let i = 0; i < V; i++) { const d = Math.max(0, p[i] - q[i]); res[i] = d; s += d; }
      if (s > 1e-9) { for (let i = 0; i < V; i++) res[i] /= s; } else res.set(p);
      slots.push({ p, q, x, ratio, acc, u: u2, fix: sampleIdx(res, u3) });
    }
    let firstRej = -1;
    for (let j = 0; j < K; j++) if (!slots[j].acc) { firstRej = j; break; }
    const nAcc = firstRej < 0 ? K : firstRej;
    const pBonus = targetDist(seed, r, K), u4 = next();
    const out = [];
    for (let j = 0; j < nAcc; j++) out.push({ tok: slots[j].x, kind: 'accept' });
    // Every round ends with one token the TARGET produced: either the correction
    // at the first disagreement, or -- when the whole draft survived -- the free
    // bonus token from the extra position the same forward already covered.
    out.push(firstRej >= 0 ? { tok: slots[firstRej].fix, kind: 'fix' } : { tok: sampleIdx(pBonus, u4), kind: 'bonus' });
    tokens += out.length; wasted += K - nAcc;
    rounds.push({ slots, firstRej, nAcc, out });
  }
  run = { rounds, K, R, tokens, wasted };
  return run;
}

function ensureRun(st) {
  const sig = `${st.seed | 0}|${st.k | 0}|${st.rounds | 0}|${(+st.quality).toFixed(3)}`;
  if (sig !== runSig) { buildRun(st); runSig = sig; }
  return run;
}

// A rounded token chip. Returns its rect so the caller can hit-test it.
function chip(ctx, x, y, w, h, text, fill, ink, opts = {}) {
  ctx.save();
  ctx.fillStyle = fill;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, 5); ctx.fill(); } else ctx.fillRect(x, y, w, h);
  if (opts.dash) { ctx.setLineDash([3, 3]); ctx.strokeStyle = opts.stroke || T.n8; ctx.lineWidth = 1; if (ctx.roundRect) ctx.stroke(); else ctx.strokeRect(x, y, w, h); ctx.setLineDash([]); }
  else if (opts.stroke) { ctx.strokeStyle = opts.stroke; ctx.lineWidth = 2; if (ctx.roundRect) ctx.stroke(); else ctx.strokeRect(x, y, w, h); }
  ctx.fillStyle = ink; ctx.font = opts.font || '11px ui-monospace, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, x + w / 2, y + h / 2 + 0.5);
  if (opts.strike) { ctx.strokeStyle = ink; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x + 5, y + h / 2); ctx.lineTo(x + w - 5, y + h / 2); ctx.stroke(); }
  ctx.restore();
  return { x, y, w, h };
}

// A horizontal drag track with a handle -- the direct-manipulation widget.
function track(ctx, rect, frac, label, hue, active) {
  const { x, y, w, h } = rect;
  ctx.save();
  ctx.fillStyle = T.n3;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, h / 2); ctx.fill(); } else ctx.fillRect(x, y, w, h);
  ctx.fillStyle = alphaOf(hue, 0.5);
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, Math.max(h, w * frac), h, h / 2); ctx.fill(); } else ctx.fillRect(x, y, w * frac, h);
  ctx.fillStyle = hue; ctx.strokeStyle = T.n0; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x + w * frac, y + h / 2, active ? 9 : 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(label, x, y - 6);
  ctx.restore();
  return rect;
}

mount({
  mount: 'body',
  title: 'speculative-decoding — draft k tokens, verify them in one pass',
  blurb: 'Plain decode emits one token per forward pass of the big model, one after another — and a single token is a skinny matrix-vector product, so the hardware spends most of its time waiting on memory rather than doing math. Speculation spends that idle width instead: a small, cheap DRAFT model proposes k tokens, and the big TARGET model verifies all k+1 positions in ONE forward pass (they batch exactly the way a prompt does). The verifier then walks the proposals left to right, keeping them while the target agrees, and stops at the FIRST disagreement — where it emits its own token instead. So every round yields at least 1 and at most k+1 tokens, and the rejected tail is thrown away. Drag the draft-agreement handle and watch the accepted lengths, the tokens-per-forward counter and the verdict move; drag the draft cost up and a weak draft turns out slower than not speculating at all.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  animate: true,
  autoplay: true,
  compare: { key: 'quality', a: 0.25, b: 0.95, labelA: 'weak draft — agreement 0.25', labelB: 'strong draft — agreement 0.95' },
  challenges: [
    {
      goal: 'Get above 2.00 tokens per target forward.',
      hint: 'drag the draft-agreement handle right — a draft that agrees more often survives further into each round.',
      check: (api) => ({ solved: (api.probe.tpf ?? 0) >= 2, detail: `${(api.probe.tpf ?? 0).toFixed(2)} tokens per target forward (need ≥ 2.00)` }),
    },
    {
      goal: 'Make speculation LOSE — drive the net rate below plain decode once draft cost is counted.',
      hint: 'drop the draft agreement and raise the draft cost: the k draft passes are paid for whether or not their tokens survive.',
      check: (api) => ({ solved: (api.probe.net ?? 2) < 1, detail: `net ${((api.probe.net ?? 1) * 100).toFixed(0)}% of plain decode (need < 100%)` }),
    },
  ],
  controls: (c, page) => {
    c.stepper('k', { label: 'draft length k', min: 1, max: 6, value: 4, rebuild: true });
    c.slider('quality', { label: 'draft agreement', min: 0, max: 1, step: 0.01, value: 0.7 });
    c.slider('cost', { label: 'draft cost per token (target forwards)', min: 0, max: 0.5, step: 0.01, value: 0.1 });
    c.stepper('rounds', { label: 'rounds', min: 3, max: 10, value: 6, rebuild: true });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7 });
    c.transport({
      compute: () => Array.from({ length: (page.state.rounds | 0) * 4 }, (_, i) => ({ round: (i / 4) | 0, phase: i % 4, label: `round ${((i / 4) | 0) + 1} · ${PHASES[i % 4]}` })),
      speed: 1.4, loop: true,
    });
  },
  onPointer: (page, ev) => {
    if (ev.type === 'up' || ev.type === 'leave') {
      // Commit the dragged value non-silently on release so the deep link syncs.
      if (grab) page.controls.set(grab, page.state[grab]);
      grab = null; return;
    }
    if (!geom) return;
    const near = (rect) => ev.y >= rect.y - 14 && ev.y <= rect.y + rect.h + 14 && ev.x >= rect.x - 14 && ev.x <= rect.x + rect.w + 14;
    if (ev.type === 'down') grab = near(geom.qTrack) ? 'quality' : near(geom.cTrack) ? 'cost' : null;
    if (!grab || !page.pointer.down) return;
    const rect = grab === 'quality' ? geom.qTrack : geom.cTrack;
    const f = Math.max(0, Math.min(1, (ev.x - rect.x) / rect.w));
    const max = grab === 'quality' ? 1 : 0.5;
    page.controls.set(grab, Math.round(f * max * 100) / 100, { silent: true });
  },
  draw: (page) => {
    const ctx = page.ctx, r = page.renderer, st = page.state, W = page.W, H = page.H;
    const data = ensureRun(st);
    const K = data.K, R = data.R;
    r.clear(T.n0);

    const tp = page.controls._transport;
    const idx = tp ? tp.index : -1;
    const roundIdx = idx < 0 ? 0 : Math.min(R - 1, (idx / 4) | 0);
    const phase = idx < 0 ? -1 : idx % 4;
    const done = idx < 0 ? 0 : ((idx / 4) | 0) + (idx % 4 === 3 ? 1 : 0);   // rounds fully committed
    const rd = data.rounds[roundIdx];

    // ---- the economics of the whole configured run -------------------------
    const tpf = data.tokens / R;                     // tokens per TARGET forward
    const net = tpf / (1 + K * st.cost);             // ... once the draft passes are paid for
    const soFar = data.rounds.slice(0, done).reduce((s, x) => s + x.out.length, 0);
    page.probe = { tpf, net, meanAcc: data.rounds.reduce((s, x) => s + x.nAcc, 0) / R };

    // ---- vertical anchors ---------------------------------------------------
    const pad = 16;
    const ctlY = H - 96;                                  // handles + gauge block
    const yHistLab = ctlY - 44, yHist = ctlY - 36, hHist = 24;
    const yKeptChip = yHistLab - 36, yKeptLab = yKeptChip - 7;
    const yColLab = 76, yFwdLab = 88, brY = 94, brH = 42, chipY = brY + 5, chipH = 32;
    const by = brY + brH + 22;
    const bh = Math.max(26, Math.min(66, yKeptLab - 74 - by));

    // ---- committed output so far -------------------------------------------
    r.label(`committed output — ${soFar} token${soFar === 1 ? '' : 's'} from ${done} target forward${done === 1 ? '' : 's'}`, pad, 16, { color: T.n11, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.font = '11px ui-monospace, monospace';
    const chips = [];
    for (let i = 0; i < done; i++) for (const o of data.rounds[i].out) chips.push(o);
    const shown = chips.slice(Math.max(0, chips.length - 30));
    let cx = pad;
    if (shown.length < chips.length) { r.label('…', cx, 40, { color: T.n9, font: '11px ui-monospace, monospace' }); cx += 14; }
    for (const o of shown) {
      const w = ctx.measureText(VOCAB[o.tok]).width + 14;
      if (cx + w > W - pad) break;
      const fill = alphaOf(o.kind === 'accept' ? T.ok : o.kind === 'fix' ? T.violet : T.teal, 0.72);
      chip(ctx, cx, 26, w, 20, VOCAB[o.tok], fill, inkOn(fill));
      cx += w + 4;
    }
    ctx.restore();

    // ---- the current round --------------------------------------------------
    const cols = K + 1, panelW = W - 2 * pad, cw = panelW / cols;
    geom = { slotRects: [] };
    r.label(`round ${roundIdx + 1} / ${R}   ·   ${phase < 0 ? 'press ▶ to run a round' : PHASE_TEXT[phase]}`, pad, 62, { color: T.n14, font: '12px ui-monospace, monospace' });

    // the ONE target forward, drawn as a bracket over every position at once
    if (phase >= 1) {
      ctx.save();
      ctx.strokeStyle = T.accent; ctx.lineWidth = phase === 1 ? 2.5 : 1.2;
      if (phase === 1) { ctx.fillStyle = alphaOf(T.accent, 0.14); ctx.fillRect(pad - 2, brY, (panelW + 4) * ((page.t * 0.9) % 1), brH); }
      ctx.strokeRect(pad - 2, brY, panelW + 4, brH);
      ctx.restore();
      r.label('one target forward · all k+1 positions in a single batch', pad, yFwdLab, { color: T.accent, font: '10px ui-monospace, monospace' });
    }

    for (let j = 0; j < cols; j++) {
      const cxm = pad + j * cw + cw / 2;
      const isFree = j === K;
      const slot = isFree ? null : rd.slots[j];
      const rejected = !isFree && rd.firstRej === j;
      const discarded = !isFree && rd.firstRej >= 0 && j > rd.firstRej;
      const accepted = !isFree && !rejected && !discarded;

      r.label(isFree ? 'free position' : `proposal ${j + 1}`, cxm, yColLab, { color: T.n10, font: '10px ui-monospace, monospace', align: 'center' });

      // row 1: what the draft proposed
      const label = isFree ? '(no draft)' : VOCAB[slot.x];
      const w = Math.min(cw - 14, Math.max(48, ctx.measureText(label).width + 24));
      let fill = alphaOf(T.accent, 0.42), stroke = null, strike = false;
      if (isFree) fill = T.n2;
      else if (phase >= 2) {
        if (accepted) { fill = alphaOf(T.ok, 0.75); stroke = T.ok; }
        else if (rejected) { fill = alphaOf(T.bad, 0.8); stroke = T.bad; }
        else { fill = T.n3; strike = true; }
      }
      const rect = chip(ctx, cxm - w / 2, chipY, w, chipH, label, fill, isFree ? T.n9 : inkOn(fill), { stroke, strike, dash: isFree, font: '12px ui-monospace, monospace' });
      geom.slotRects.push({ ...rect, j, isFree });

      if (phase < 1) continue;

      // row 2: the two probabilities the accept test compares
      if (!isFree) {
        const pv = slot.p[slot.x], qv = slot.q[slot.x], bw = Math.min(18, cw / 7), gap = Math.max(bw + 8, Math.min(26, cw / 5));
        const pairs = [[qv, T.warn, 'q'], [pv, T.accent, 'p']];
        ctx.save(); ctx.textAlign = 'center'; ctx.font = '9px ui-monospace, monospace';
        for (let i = 0; i < 2; i++) {
          const [val, hue, nm] = pairs[i], bx = cxm + (i === 0 ? -gap : gap) - bw / 2;
          ctx.fillStyle = T.n3; ctx.fillRect(bx, by, bw, bh);
          ctx.fillStyle = alphaOf(hue, 0.8); ctx.fillRect(bx, by + bh * (1 - val), bw, bh * val);
          ctx.fillStyle = T.n11; ctx.fillText(`${nm} ${val.toFixed(2)}`, bx + bw / 2, by + bh + 12);
        }
        ctx.restore();
        r.label(`p/q = ${slot.ratio.toFixed(2)}`, cxm, by + bh + 27, { color: T.n12, font: '10px ui-monospace, monospace', align: 'center' });
        if (phase >= 2) {
          const vc = accepted ? T.ok : rejected ? T.bad : T.n9;
          r.label(accepted ? '✓ accepted' : rejected ? '✗ rejected' : 'discarded', cxm, by + bh + 43, { color: vc, font: '11px ui-monospace, monospace', align: 'center' });
          r.label(discarded ? '(after the stop)' : `draw ${slot.u.toFixed(2)} ${slot.u <= Math.min(1, slot.ratio) ? '≤' : '>'} min(1, p/q)`, cxm, by + bh + 57, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
        }
      } else if (phase >= 2) {
        const lines = rd.firstRej < 0 ? ['already covered by the', 'same forward — free token'] : ['not reached this round —', 'the run stopped earlier'];
        r.label(lines[0], cxm, by + 12, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
        r.label(lines[1], cxm, by + 24, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
      }

      // row 3: what this position contributes to the output
      if (phase >= 3) {
        const o = rd.out[j];
        if (o) {
          const oh = alphaOf(o.kind === 'accept' ? T.ok : o.kind === 'fix' ? T.violet : T.teal, 0.8);
          const ow = Math.min(cw - 14, Math.max(48, ctx.measureText(VOCAB[o.tok]).width + 24));
          r.label(o.kind === 'accept' ? 'kept' : o.kind === 'fix' ? 'target’s own token' : 'free bonus token', cxm, yKeptLab, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
          chip(ctx, cxm - ow / 2, yKeptChip, ow, 22, VOCAB[o.tok], oh, inkOn(oh), { font: '11px ui-monospace, monospace' });
        } else {
          r.label('thrown away', cxm, yKeptChip + 15, { color: T.n9, font: '10px ui-monospace, monospace', align: 'center' });
        }
      }
    }

    // ---- tokens produced per round (the whole run at a glance) --------------
    r.label('tokens per round (accepted + 1)', pad, yHistLab, { color: T.n11, font: '10px ui-monospace, monospace' });
    const hbw = Math.min(26, (W * 0.4) / R);
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < R; i++) {
      const bx = pad + i * hbw, ro = data.rounds[i], f = ro.out.length / (K + 1);
      const fillC = i < done ? alphaOf(T.ok, 0.3 + 0.5 * f) : i === roundIdx ? alphaOf(T.accent, 0.32) : T.n3;
      ctx.fillStyle = fillC; ctx.fillRect(bx, yHist, hbw - 3, hHist);
      if (i === roundIdx) { ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5; ctx.strokeRect(bx - 1, yHist - 1, hbw - 1, hHist + 2); }
      ctx.fillStyle = i < done || i === roundIdx ? inkOn(fillC) : T.n9;
      ctx.fillText(String(ro.out.length), bx + (hbw - 3) / 2, yHist + hHist / 2);
    }
    ctx.restore();

    // ---- direct-manipulation handles ---------------------------------------
    const tw = Math.min(230, W * 0.27);
    geom.qTrack = { x: pad, y: ctlY + 10, w: tw, h: 10 };
    geom.cTrack = { x: pad, y: ctlY + 48, w: tw, h: 10 };
    track(ctx, geom.qTrack, st.quality, `draft agreement ${(+st.quality).toFixed(2)}   (drag ↔)`, T.ok, grab === 'quality');
    track(ctx, geom.cTrack, st.cost / 0.5, `draft cost ${(+st.cost).toFixed(2)} per drafted token   (drag ↔)`, T.warn, grab === 'cost');

    // ---- the tokens-per-forward gauge, against the baseline of 1.0 ----------
    const gx = pad + tw + 44, gw = Math.max(120, W - gx - pad), gy = ctlY + 20, gmax = K + 1;
    ctx.save();
    ctx.fillStyle = T.n3; ctx.fillRect(gx, gy, gw, 16);
    ctx.fillStyle = alphaOf(net >= 1 ? T.ok : T.bad, 0.45); ctx.fillRect(gx, gy, gw * Math.min(1, tpf / gmax), 16);
    ctx.fillStyle = net >= 1 ? T.okDeep : T.bad; ctx.fillRect(gx, gy, gw * Math.min(1, net / gmax), 16);
    const bx1 = gx + gw / gmax;
    ctx.strokeStyle = T.n14; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx1, gy - 5); ctx.lineTo(bx1, gy + 21); ctx.stroke();
    ctx.fillStyle = T.n11; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('1.00 — plain decode', bx1, gy + 32);
    ctx.textAlign = 'left'; ctx.fillStyle = T.n14; ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(`${tpf.toFixed(2)} tokens per target forward`, gx, gy - 10);
    ctx.fillStyle = net >= 1 ? T.okDeep : T.bad; ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(`net ${(net * 100).toFixed(0)}% of plain decode ${net >= 1 ? '— worth it' : '— SLOWER than not speculating'}`, gx, gy + 50);
    ctx.restore();

    // ---- hover-to-inspect ---------------------------------------------------
    if (page.pointer.over && !grab) {
      const px = page.pointer.x, py = page.pointer.y;
      for (const s of geom.slotRects) {
        if (px < s.x || px > s.x + s.w || py < s.y || py > s.y + s.h) continue;
        if (s.isFree) {
          page.setTip(rd.firstRej < 0
            ? 'free position\nthe target forward already covered k+1 positions, so when every\nproposal survives this extra token costs nothing at all'
            : 'free position\ncollected only when the whole draft is accepted; this round\nstopped at a disagreement, so it is never reached');
          break;
        }
        const sl = rd.slots[s.j], thr = Math.min(1, sl.ratio);
        const verdict = rd.firstRej === s.j ? 'REJECTED' : (rd.firstRej >= 0 && s.j > rd.firstRej) ? 'DISCARDED' : 'ACCEPTED';
        const why = verdict === 'DISCARDED'
          ? `proposal ${rd.firstRej + 1} was rejected first, so everything after it is\nthrown away untested — that is the wasted draft compute`
          : `draw ${sl.u.toFixed(3)} ${sl.u <= thr ? '≤' : '>'} min(1, p/q) = ${thr.toFixed(3)}  →  ${verdict.toLowerCase()}`;
        const tail = verdict === 'REJECTED' ? `\nthe target emits its own "${VOCAB[sl.fix]}" here, drawn from the\nresidual normalize(max(0, p − q))` : '';
        page.setTip(`proposal ${s.j + 1}: "${VOCAB[sl.x]}"\ndraft q = ${sl.q[sl.x].toFixed(3)} · target p = ${sl.p[sl.x].toFixed(3)} · p/q = ${sl.ratio.toFixed(3)}\n${why}${tail}`);
        break;
      }
    }

    let o = `k=${K} · draft agreement ${(+st.quality).toFixed(2)} · draft cost ${(+st.cost).toFixed(2)} → mean accepted ${page.probe.meanAcc.toFixed(2)} of ${K}, so ${tpf.toFixed(2)} tokens per target forward (plain decode = 1.00).    tier:${r.name}\n`;
    o += `${R} rounds: ${data.tokens} tokens for ${R} target forwards + ${R * K} draft forwards; ${data.wasted} drafted tokens discarded. `;
    o += `Charging each drafted token ${(+st.cost).toFixed(2)} of a target forward: net ${(net * 100).toFixed(0)}% of plain decode${net >= 1 ? '.' : ' — the draft is not earning its keep.'}`;
    page.setReadout(o);
  },
}).then((page) => {
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  for (const key of ['k', 'rounds']) if (q.has(key)) page.controls.set(key, parseInt(q.get(key), 10), { rebuild: true });
  for (const key of ['quality', 'cost', 'seed']) if (q.has(key)) page.controls.set(key, parseFloat(q.get(key)));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (t && (q.has('step') || q.has('hover'))) t.pause();
  if (t && q.has('step')) t.seek(parseInt(q.get('step'), 10));
  if (t && q.get('play') === '1') t.play();
  page.redraw();
});
