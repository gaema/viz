// sampling concept page -- how the next token is chosen from the logits:
// temperature -> softmax -> a truncation strategy (greedy / top-k / top-p) ->
// a draw. Bars sorted by probability; kept set colored, cut tail greyed; the
// cumulative curve + p-line (top-p) / divider (top-k) show the kept set. A live
// sampler draws from the kept, renormalized distribution and tallies the picks
// so the empirical frequency converges to the probabilities. Interactive: drag
// the k divider / p line (routes through controls.set so the slider tracks),
// hover any bar, the sampler animates.
import { mount } from '../framework/layout.js';
import { softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', RED = '#d1242f', GREY = '#9aa4ad';
const VOCAB = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast', 'and', 'slept'];

let cur = null, builtSeed = null;
let geom = null;                 // {leftX, bw, yBase, barsH, V} captured in draw
let grab = null;                 // 'k' | 'p' while dragging a threshold
let tally = null, total = 0, lastSig = '', lastSampleT = 0, flash = { idx: -1, ft: -2 };
let rngState = 1;
function nextRand() {             // mulberry32 (browser page; deterministic per seed)
  rngState = (rngState + 0x6D2B79F5) | 0;
  let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function buildData(st) {
  const V = VOCAB.length;
  cur = { logits: seededRandn(st.seed | 0, V, { std: 1.1 }), V };
  return [];
}

mount({
  mount: 'body',
  title: 'sampling — greedy / top-k / top-p / temperature',
  blurb: 'How the next token is chosen from the lm-head logits. Temperature T scales the logits before softmax (T<1 sharpens → greedy, T>1 flattens → random). Then a truncation strategy keeps a subset: greedy = the argmax; top-k = the k most likely; top-p = the smallest top set whose cumulative probability reaches p. The kept set is renormalized and sampled. Bars are sorted by probability; the live sampler draws tokens and tallies them so the empirical frequency converges to the distribution. Drag the k divider or the p line to resize the kept set; hover any bar.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  compare: { key: 'strat', a: 'full', b: 'greedy', labelA: 'full distribution', labelB: 'greedy — argmax only' },
  animate: true,                 // the live sampler runs off api.t
  challenges: [
    { goal: 'Make decoding deterministic — always the same token.', hint: 'set strategy to "greedy", OR top-k with k=1, OR lower the temperature until the top token ≥ 99%.', check: (api) => { const s = api.state; const det = s.strat === 'greedy' || (s.strat === 'top-k' && (s.k | 0) === 1) || (api.probe.maxP ?? 0) >= 0.99; return { solved: det, detail: `top token p = ${((api.probe.maxP ?? 0) * 100).toFixed(0)}%` }; } },
    { goal: 'Make it nearly uniform — flatten the distribution (top token ≤ 12%).', hint: 'raise the temperature toward 2 with the "full" strategy.', check: (api) => ({ solved: (api.probe.maxP ?? 1) <= 0.12, detail: `top token p = ${((api.probe.maxP ?? 1) * 100).toFixed(0)}% (need ≤ 12%)` }) },
  ],
  controls: (c, page) => {
    c.slider('temp', { label: 'temperature T', min: 0.1, max: 2, step: 0.1, value: 1 });
    c.select('strat', { label: 'strategy', options: ['full', 'greedy', 'top-k', 'top-p'], value: 'full' });
    c.slider('k', { label: 'top-k: k', min: 1, max: VOCAB.length, step: 1, value: 3 });
    c.slider('p', { label: 'top-p: p', min: 0.1, max: 1, step: 0.05, value: 0.9 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 6, rebuild: true });
  },
  onPointer: (page, ev) => {
    const st = page.state;
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!geom) return;
    const { leftX, bw, yBase, barsH, V } = geom;
    if (ev.type === 'down') {
      grab = null;
      if (st.strat === 'top-k' && Math.abs(ev.x - (leftX + st.k * bw)) < 16) grab = 'k';
      else if (st.strat === 'top-p' && Math.abs(ev.y - (yBase - st.p * barsH)) < 16) grab = 'p';
    }
    if (grab === 'k' && page.pointer.down) page.controls.set('k', Math.max(1, Math.min(V, Math.round((ev.x - leftX) / bw))), { silent: true });
    else if (grab === 'p' && page.pointer.down) page.controls.set('p', Math.max(0.1, Math.min(1, Math.round(((yBase - ev.y) / barsH) * 20) / 20)), { silent: true });
  },
  draw: (page) => {
    const ctx = page.ctx, st = page.state, r = page.renderer;
    if (!cur || builtSeed !== (st.seed | 0)) { buildData(st); builtSeed = st.seed | 0; }   // no transport: build logits here
    const { logits, V } = cur;
    r.clear('#ffffff');
    const T = Math.max(0.05, st.temp), strat = st.strat, K = st.k | 0, P = st.p;

    // temperature softmax + descending sort
    const probs = softmax(Float32Array.from(logits, (z) => z / T));
    const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
    page.probe = { maxP: probs[order[0]] };
    const cum = new Float32Array(V); let acc = 0;
    for (let r2 = 0; r2 < V; r2++) { acc += probs[order[r2]]; cum[r2] = acc; }

    // kept set by strategy (over the sorted order)
    let keptRank = V;                                    // ranks [0..keptRank) are kept
    if (strat === 'greedy') keptRank = 1;
    else if (strat === 'top-k') keptRank = Math.min(K, V);
    else if (strat === 'top-p') { keptRank = 1; for (let r2 = 0; r2 < V; r2++) { if (cum[r2] >= P) { keptRank = r2 + 1; break; } keptRank = r2 + 1; } }
    const kept = new Array(V).fill(false); for (let r2 = 0; r2 < keptRank; r2++) kept[order[r2]] = true;
    let ksum = 0; for (let i = 0; i < V; i++) if (kept[i]) ksum += probs[i];
    const renorm = Float32Array.from(probs, (p, i) => (kept[i] ? p / ksum : 0));

    // layout
    const pad = 16, leftX = pad + 8, barsW = page.W * 0.66, yBase = page.H - pad - 40, barsH = page.H - 96;
    const bw = barsW / V, pmax = probs[order[0]] || 1;
    geom = { leftX, bw, yBase, barsH, V };

    // live sampler: draw from `renorm` each ~0.34s, tally; reset on param change
    const sig = `${T}|${strat}|${K}|${P}|${st.seed}|${V}`;
    if (sig !== lastSig) { tally = new Float64Array(V); total = 0; lastSig = sig; rngState = (st.seed | 0) + 1; lastSampleT = page.t; flash = { idx: -1, ft: -2 }; }
    if (page.t - lastSampleT > 0.34) {
      lastSampleT = page.t; const u = nextRand(); let c = 0, pick = order[0];
      if (strat === 'greedy') pick = order[0]; else for (const idx of order) { c += renorm[idx]; if (u <= c) { pick = idx; break; } }
      tally[pick]++; total++; flash = { idx: pick, ft: page.t };
    }

    // axis baseline
    ctx.save(); ctx.strokeStyle = '#e3e6ea'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(leftX, yBase); ctx.lineTo(leftX + barsW, yBase); ctx.stroke(); ctx.restore();
    r.label('p(token)  — sorted, after temperature', leftX, yBase - barsH - 14, { color: INK, font: '12px ui-monospace, monospace' });

    // bars (sorted desc): kept colored, cut greyed; empirical-frequency outline
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
    for (let rk = 0; rk < V; rk++) {
      const idx = order[rk], x = leftX + rk * bw, h = (probs[idx] / pmax) * barsH;
      const isKept = kept[idx], isTop = rk === 0;
      ctx.fillStyle = !isKept ? '#eceef0' : (strat === 'greedy' && isTop) ? RED : 'rgba(31,111,235,0.72)';
      ctx.fillRect(x + 2, yBase - h, bw - 4, h);
      if (flash.idx === idx && page.t - flash.ft < 0.3) { ctx.fillStyle = 'rgba(44,160,44,0.9)'; ctx.fillRect(x + 2, yBase - h - 8, bw - 4, 5); }   // sampled flash
      if (total > 0) { const eh = (tally[idx] / total) * (barsH * (pmax)); ctx.strokeStyle = '#1a1d21'; ctx.lineWidth = 1.4; ctx.strokeRect(x + bw / 2 - 4, yBase - eh, 8, eh); }  // empirical freq
      ctx.fillStyle = isKept ? '#3a4047' : GREY; ctx.textBaseline = 'top';
      ctx.fillText(`"${VOCAB[idx]}"`, x + bw / 2, yBase + 4);
      ctx.fillText(probs[idx].toFixed(2), x + bw / 2, yBase + 16);
    }
    ctx.restore();

    // cumulative curve + the top-p line / top-k divider
    ctx.save();
    ctx.strokeStyle = 'rgba(137,87,229,0.8)'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let rk = 0; rk < V; rk++) { const x = leftX + rk * bw + bw / 2, y = yBase - cum[rk] * barsH; if (rk === 0) ctx.moveTo(leftX, yBase); ctx.lineTo(x, y); }
    ctx.stroke();
    r.label('Σ cumulative', leftX + barsW + 6, yBase - barsH + 8, { color: '#8957e5', font: '10px ui-monospace, monospace' });
    if (strat === 'top-p') {
      const py = yBase - P * barsH;
      ctx.strokeStyle = RED; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(leftX, py); ctx.lineTo(leftX + barsW, py); ctx.stroke(); ctx.setLineDash([]);
      r.label(`p = ${P.toFixed(2)}  (drag ↕)`, leftX + barsW + 6, py, { color: RED, font: '10px ui-monospace, monospace' });
    } else if (strat === 'top-k') {
      const kx = leftX + K * bw;
      ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(kx, yBase - barsH); ctx.lineTo(kx, yBase + 4); ctx.stroke();
      r.label(`k = ${K}  (drag ↔)`, kx + 4, yBase - barsH + 8, { color: RED, font: '10px ui-monospace, monospace' });
    }
    ctx.restore();

    // hover-to-inspect
    if (page.pointer.over && !grab) {
      const rk = Math.floor((page.pointer.x - leftX) / bw);
      if (rk >= 0 && rk < V && page.pointer.y <= yBase && page.pointer.y >= yBase - barsH) {
        const idx = order[rk];
        page.setTip(`"${VOCAB[idx]}"  (rank ${rk + 1})\nlogit ${logits[idx].toFixed(2)} · p ${probs[idx].toFixed(3)} · Σ ${cum[rk].toFixed(3)}\n${kept[idx] ? 'kept → renorm p ' + renorm[idx].toFixed(3) : 'cut (not sampled)'}`);
      }
    }

    const keptToks = order.slice(0, keptRank).map((i) => `"${VOCAB[i]}"`).join(' ');
    let o = `softmax(logits / T=${T.toFixed(1)}) → ${strat}.    empirical draws: ${total} (outline bars converge to p).    tier:${r.name}\n`;
    o += strat === 'greedy' ? `greedy: always argmax = "${VOCAB[order[0]]}" (p ${probs[order[0]].toFixed(3)}).`
      : strat === 'full' ? `full distribution: sample from all ${V} tokens.`
      : `${strat}: keep ${keptRank} token${keptRank > 1 ? 's' : ''} {${keptToks}} (Σp=${cum[keptRank - 1].toFixed(3)}), renormalize, sample.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__smpPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('temp')) page.controls.set('temp', parseFloat(q.get('temp')));
  if (q.has('strat')) page.controls.set('strat', q.get('strat'));
  if (q.has('k')) page.controls.set('k', parseInt(q.get('k'), 10));
  if (q.has('p')) page.controls.set('p', parseFloat(q.get('p')));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
