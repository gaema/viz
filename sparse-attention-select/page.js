// sparse-attention-select concept page -- how a long-context model chooses WHAT
// to attend to instead of attending to everything.
//
// The mechanism, in four modes the reader switches between:
//   full  dense causal attention -- the baseline every other mode is scored against.
//   dsa   a cheap side network (the "indexer") scores EVERY past token for the
//         current query; the top-k scoring tokens are kept; real attention runs
//         only on those. This is the shape DeepSeek-V3.2 introduced.
//   csa   compress the KV cache along the SEQUENCE dimension first, then apply
//         the same top-k selection over the compressed entries.
//   hca   compress much more heavily, then run DENSE attention over the short
//         compressed sequence -- no selection at all.
// The V3.2 successor's model card describes exactly that last pair as two
// attention TYPES mixed inside one model. Sources are listed in README.md; the
// only quantities this page shows are computed live from the reader's own
// settings (k, N, head dim, indexer dim, compression factor), never quoted.
//
// Interactive per the shared render framework's contract:
//   - TRANSPORT steps the query position along the sequence, auto-playing +
//     looping (?step=N pins one query).
//   - DIRECT MANIPULATION: drag the top-k cut line across the ranked-score bar
//     to change k; drag the right edge of the first compressed block to change
//     the compression factor; drag the needle marker along the sequence strip.
//     Every counter moves under your hand.
//   - HOVER any past token (strip, ranked bar, or attention matrix) for its
//     indexer score, its rank, and whether it made the cut.
//   - MODE switch between sparse-select, compress-then-dense, and the dense
//     baseline.
// Both of the trades are on screen and findable by dragging:
//   1. the indexer still scores every past token, so it is ITSELF quadratic --
//      there is a crossover sequence length below which this costs MORE than
//      plain attention, and the panel prints where it is;
//   2. top-k is a hard recall cliff -- hide a needle at position p and watch it
//      fall out of the selected set as k shrinks, taking its whole share of the
//      true attention mass with it.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, mixColor } from '../framework/theme.js';

const MODES = [
  { key: 'full', name: 'full attention (baseline)', desc: 'dense causal attention — every query reads every past KV entry. Cost grows with the square of the sequence.' },
  { key: 'dsa', name: 'indexer + top-k select', desc: 'a cheap side network scores every past token for this query; only the k highest-scoring are read by real attention.' },
  { key: 'csa', name: 'compress, then select', desc: 'the KV cache is first compressed along the sequence dimension; the same top-k selection then runs over the compressed entries.' },
  { key: 'hca', name: 'compress harder, then dense', desc: 'much heavier compression, then DENSE attention over the short compressed sequence — no selection at all.' },
];
const modeOf = (k) => MODES.find((m) => m.key === k) || MODES[0];

// ---------------------------------------------------------------------------
// Cost model. Every figure the page prints comes from these closed forms
// evaluated on the reader's settings -- multiply counts for one causal prefill
// of N tokens. 2*d per attended entry = the QK score plus the PV accumulation.
// ---------------------------------------------------------------------------
const S1 = (N) => (N * (N + 1)) / 2;                       // causal (query,key) pairs
// sum over m=1..N of min(k, m)
function sumMin(N, k) { if (k >= N) return S1(N); return (k * (k - 1)) / 2 + k * (N - k + 1); }
// sum over m=1..N of ceil(m/c)  -- how many compressed entries each query sees
function sumCeil(N, c) { if (N <= 0) return 0; const q = Math.floor(N / c), r = N - q * c; return c * (q * (q + 1)) / 2 + r * (q + 1); }
// sum over m=1..N of min(k, ceil(m/c))
function sumMinCeil(N, c, k) { const M = Math.min(N, (k - 1) * c); return sumCeil(M, c) + (N - M) * k; }

// {comp, idx, attn, total} multiply counts for a whole N-token prefill.
function costs(mode, N, k, d, di, c, ch) {
  if (mode === 'full') return { comp: 0, idx: 0, attn: 2 * d * S1(N), total: 2 * d * S1(N) };
  if (mode === 'dsa') { const idx = di * S1(N), attn = 2 * d * sumMin(N, k); return { comp: 0, idx, attn, total: idx + attn }; }
  if (mode === 'csa') { const comp = N * d, idx = di * sumCeil(N, c), attn = 2 * d * sumMinCeil(N, c, k); return { comp, idx, attn, total: comp + idx + attn }; }
  const comp = N * d, attn = 2 * d * sumCeil(N, ch);
  return { comp, idx: 0, attn, total: comp + attn };
}

// Smallest sequence length at which this mode costs LESS than full attention.
// Below it the indexer's own quadratic term dominates and the "sparse" path is
// the more expensive one -- the first of the two trades this page is about.
let _xCache = null;
function crossover(mode, k, d, di, c, ch, cap = 20000) {
  if (mode === 'full') return 0;
  const sig = `${mode}|${k}|${d}|${di}|${c}|${ch}`;
  if (_xCache && _xCache.sig === sig) return _xCache.v;
  let v = -1;
  for (let n = 2; n <= cap; n++) if (costs(mode, n, k, d, di, c, ch).total < costs('full', n, k, d, di, c, ch).total) { v = n; break; }
  _xCache = { sig, v };
  return v;
}

const fmt = (x) => (x >= 1e9 ? (x / 1e9).toFixed(2) + ' G' : x >= 1e6 ? (x / 1e6).toFixed(2) + ' M' : x >= 1e3 ? (x / 1e3).toFixed(1) + ' k' : String(Math.round(x)));
const pct = (x) => (100 * x).toFixed(1) + '%';

// ---------------------------------------------------------------------------
// Synthetic-but-real tensors. Deterministic per seed, so the picture reloads
// identically. The indexer is a genuine low-rank projection of the same query /
// key vectors real attention uses: its score APPROXIMATES the true score, and
// the approximation gets worse as the indexer dim shrinks -- which is why
// dragging `di` down visibly costs top-k recall.
// ---------------------------------------------------------------------------
let cache = null;
function tensors(N, d, di, seed) {
  const sig = `${N}|${d}|${di}|${seed}`;
  if (cache && cache.sig === sig) return cache;
  const Q = seededRandn(seed, [N, d]), K = seededRandn(seed + 1, [N, d]);
  // JL-style projection: entries ~ N(0, 1/di) so E[(Pᵀq)·(Pᵀk)] = q·k.
  const P = seededRandn(seed + 2, [d, di], { std: 1 / Math.sqrt(di) });
  const Qp = new Float32Array(N * di), Kp = new Float32Array(N * di);
  for (let n = 0; n < N; n++) for (let t = 0; t < di; t++) {
    let a = 0, b = 0;
    for (let e = 0; e < d; e++) { const w = P.data[e * di + t]; a += Q.data[n * d + e] * w; b += K.data[n * d + e] * w; }
    Qp[n * di + t] = a; Kp[n * di + t] = b;
  }
  const inv = 1 / Math.sqrt(d);
  const trueS = new Float32Array(N * N), idxS = new Float32Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
    let s = 0; for (let e = 0; e < d; e++) s += Q.data[i * d + e] * K.data[j * d + e];
    let g = 0; for (let t = 0; t < di; t++) g += Qp[i * di + t] * Kp[j * di + t];
    trueS[i * N + j] = s * inv; idxS[i * N + j] = g * inv;
  }
  cache = { sig, N, trueS, idxS };
  return cache;
}

// ---------------------------------------------------------------------------
// Per-query selection. Returns, for query i: the candidate entries (raw tokens
// or compressed blocks), their true + indexer scores, the top-k chosen set, the
// softmax over the FULL prefix (to price what selection throws away), and the
// needle's standing.
// ---------------------------------------------------------------------------
function selectFor(st, tn, i) {
  const { N, trueS, idxS } = tn;
  const mode = st.mode, k = st.k, p = Math.min(st.p, N - 1), nb = st.nb;
  const grp = mode === 'csa' ? st.c : mode === 'hca' ? st.hc : 1;   // tokens per entry
  const nEnt = Math.ceil((i + 1) / grp);
  const ent = [];
  for (let b = 0; b < nEnt; b++) {
    const lo = b * grp, hi = Math.min((b + 1) * grp, i + 1);
    let ts = 0, is = 0, holdsNeedle = false;
    for (let j = lo; j < hi; j++) {                                  // sequence-dim pooling
      ts += trueS[i * N + j] + (j === p ? nb : 0);
      is += idxS[i * N + j] + (j === p ? nb : 0);
      if (j === p) holdsNeedle = true;
    }
    ent.push({ b, lo, hi, ts: ts / (hi - lo), is: is / (hi - lo), holdsNeedle });
  }
  // top-k by INDEXER score (that is the whole point: the cheap score decides).
  const byIdx = ent.slice().sort((a, b2) => b2.is - a.is);
  byIdx.forEach((e, r) => { e.rank = r; });
  const keep = mode === 'full' || mode === 'hca' ? nEnt : Math.min(k, nEnt);
  const sel = new Set(byIdx.slice(0, keep).map((e) => e.b));
  // what a dense pass would have paid attention to, so we can price the loss
  let mx = -Infinity; for (const e of ent) mx = Math.max(mx, e.ts);
  let z = 0; for (const e of ent) { e.w = Math.exp(e.ts - mx); z += e.w; }
  let kept = 0; for (const e of ent) { e.w /= z; if (sel.has(e.b)) kept += e.w; }
  // top-k recall: how many of the TRUE top-k the cheap indexer actually found
  const byTrue = ent.slice().sort((a, b2) => b2.ts - a.ts).slice(0, keep);
  let recall = 0; for (const e of byTrue) if (sel.has(e.b)) recall++;
  const nEnt2 = ent.find((e) => e.holdsNeedle);
  return {
    ent, sel, keep, nEnt, grp, discarded: Math.max(0, 1 - kept), recall, recallOf: byTrue.length,
    needle: nEnt2 ? { rank: nEnt2.rank, inSet: sel.has(nEnt2.b), mass: nEnt2.w, entry: nEnt2 } : null,
    ranked: byIdx,
  };
}

// --- geometry captured in draw() for hover + drag ---------------------------
let rStrip = null, rBlocks = null, rRank = null, rMat = null;
let rankN = 0, grab = null;   // candidate count behind the ranked bar (the k-drag scale)

mount({
  mount: 'body',
  title: 'sparse-attention-select — choosing what to attend to',
  blurb: 'Long-context attention gets unaffordable because every query reads every past token. So a cheap side network — the indexer — scores every past token for the current query, and only the top-k scoring ones are handed to real attention. Switch modes to compress the KV along the sequence first and select over the compressed entries, or to compress much harder and run dense attention over the short result. Drag the k cut, the compression factor, and the needle; every counter recomputes under your hand. Two costs are the whole story: the indexer scores EVERY past token, so it is itself quadratic (find the sequence length where it stops paying), and top-k is a hard cliff — a needle outside the k contributes exactly nothing. The indexer drawn here is an UNTRAINED low-rank projection of the real query and key vectors, so the recall it reports is a worst case: a trained one agrees with full attention far more often, at the same cost.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  autoplay: true,
  controls: (c, page) => {
    c.select('mode', { label: 'attention type', options: MODES.map((m) => ({ value: m.key, label: m.name })) , value: 'dsa' });
    c.stepper('N', { label: 'sequence length N', min: 12, max: 48, value: 32, rebuild: true });
    c.slider('k', { label: 'top-k kept (drag the cut)', min: 1, max: 24, step: 1, value: 6 });
    c.slider('c', { label: 'compression factor (compress→select)', min: 1, max: 8, step: 1, value: 3 });
    c.slider('hc', { label: 'heavy compression factor (→dense)', min: 2, max: 16, step: 1, value: 8 });
    c.slider('d', { label: 'head dim d', min: 8, max: 64, step: 8, value: 32 });
    c.slider('di', { label: 'indexer dim (cheaper ⇒ noisier)', min: 2, max: 32, step: 2, value: 16 });
    c.slider('p', { label: 'needle position p (drag it)', min: 0, max: 47, step: 1, value: 6 });
    c.slider('nb', { label: 'needle relevance', min: 0, max: 4, step: 0.25, value: 2 });
    c.slider('seed', { label: 'seed', min: 1, max: 40, step: 1, value: 7, rebuild: false });
    c.transport({ compute: () => new Array(page.state.N).fill(0).map((_, i) => ({ label: `query q = ${i}` })), speed: 2.5, loop: true });
  },

  // Three drag bands, one per row of the picture: the sequence strip moves the
  // needle, the compressed-block row resizes the compression factor, the ranked
  // bar moves the top-k cut.
  onPointer: (page, ev) => {
    const st = page.state, N = st.N;
    const inRect = (r) => r && ev.x >= r.x - 6 && ev.x <= r.x + r.w + 6 && ev.y >= r.y - 6 && ev.y <= r.y + r.h + 6;
    if (ev.type === 'down') {
      grab = inRect(rStrip) ? 'needle' : inRect(rBlocks) ? 'comp' : inRect(rRank) ? 'k' : null;
    } else if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!grab || !page.pointer.down) return;
    if (grab === 'needle' && rStrip) {
      const j = Math.floor((ev.x - rStrip.x) / (rStrip.w / N));
      page.controls.set('p', Math.max(0, Math.min(N - 1, j)), { silent: true });
    } else if (grab === 'comp' && rBlocks) {
      const g = Math.round((ev.x - rBlocks.x) / (rBlocks.w / N));
      if (st.mode === 'hca') page.controls.set('hc', Math.max(2, Math.min(16, g)), { silent: true });
      else page.controls.set('c', Math.max(1, Math.min(8, g)), { silent: true });
    } else if (grab === 'k' && rRank) {
      const r = Math.round((ev.x - rRank.x) / (rRank.w / Math.max(1, rankN || N)));
      page.controls.set('k', Math.max(1, Math.min(24, r)), { silent: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const N = st.N, mode = st.mode, m = modeOf(mode);
    const tn = tensors(N, st.d, st.di, st.seed | 0);
    const p = Math.min(st.p, N - 1);

    // query position: the transport index sweeps it, looping over the sequence
    const tr = page.controls._transport;
    const qi = tr && tr.index >= 0 ? Math.min(tr.index, N - 1) : N - 1;
    const sl = selectFor(st, tn, qi);

    const pad = 14, W = page.W, H = page.H;
    const stripX = pad, stripW = W - 2 * pad, cw = stripW / N;
    const mono = (px) => `${px}px ui-monospace, monospace`;
    const txt = (s, x, y, col, px, align) => { ctx.save(); ctx.fillStyle = col; ctx.font = mono(px || 11); ctx.textAlign = align || 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText(s, x, y); ctx.restore(); };

    let hdr = `${m.name} — ${m.desc}`;
    ctx.save(); ctx.font = mono(11.5);
    while (hdr.length > 12 && ctx.measureText(hdr).width > W - 2 * pad) hdr = hdr.slice(0, -2) + '…';
    ctx.restore();
    txt(hdr, pad, 13, T.n12, 11.5);

    // --- row 1: the sequence, tinted by this query's INDEXER score -----------
    const sy = 42, sh = 18;   // a marker row (query ▼ / needle ▼) sits just above
    rStrip = { x: stripX, y: sy, w: stripW, h: sh };
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j <= qi; j++) { const v = tn.idxS[qi * N + j] + (j === p ? st.nb : 0); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const norm = (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5);
    txt('sequence — indexer score of every past token for this query', stripX, sy - 16, T.n11, 10.5);
    txt('▬ read by real attention', stripX + stripW, sy - 16, T.accent, 10.5, 'right');
    for (let j = 0; j < N; j++) {
      const x = stripX + j * cw;
      if (j > qi) ctx.fillStyle = alphaOf('n12', 0.10);
      else ctx.fillStyle = mixColor(T.n0, T.violet, 0.12 + 0.82 * norm(tn.idxS[qi * N + j] + (j === p ? st.nb : 0)));
      ctx.fillRect(x, sy, Math.max(1, cw - 1), sh);
    }
    // the entries this query actually reads, marked under the strip
    ctx.save(); ctx.fillStyle = T.accent;
    for (const e of sl.ent) if (sl.sel.has(e.b)) ctx.fillRect(stripX + e.lo * cw, sy + sh + 2, Math.max(1.5, (e.hi - e.lo) * cw - 1), 3);
    ctx.restore();
    // current query + needle markers
    ctx.save(); ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5; ctx.strokeRect(stripX + qi * cw - 0.5, sy - 2.5, Math.max(2, cw - 1) + 1, sh + 5); ctx.restore();
    txt('q▼', stripX + qi * cw + cw / 2, sy - 3, T.n14, 10, 'center');
    txt('▼p', stripX + p * cw + cw / 2, sy - 3, T.warn, 10, 'center');

    // --- row 2: compressed blocks (the sequence-dimension compression) -------
    const by = sy + sh + 20, bh = 13;
    if (mode === 'csa' || mode === 'hca') {
      const g = mode === 'hca' ? st.hc : st.c;
      rBlocks = { x: stripX, y: by, w: stripW, h: bh };
      txt(`KV compressed along the sequence: ${g} tokens per entry, ${Math.ceil(N / g)} entries for N=${N}  (drag an edge ↔)`, stripX, by - 4, T.n11, 10.5);
      for (let b = 0; b * g < N; b++) {
        const x0 = stripX + b * g * cw, wpx = Math.min(g, N - b * g) * cw - 2;
        const visible = b * g <= qi;
        const chosen = sl.sel.has(b) && visible;
        ctx.fillStyle = chosen ? mixColor(T.n0, T.accent, 0.55) : visible ? alphaOf('n12', 0.34) : alphaOf('n12', 0.10);
        ctx.fillRect(x0, by, Math.max(2, wpx), bh);
      }
      ctx.save(); ctx.strokeStyle = T.warn; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(stripX + g * cw - 1, by - 3); ctx.lineTo(stripX + g * cw - 1, by + bh + 3); ctx.stroke(); ctx.restore();
    } else {
      rBlocks = null;
      txt(mode === 'full' ? 'no compression, no selection — the dense baseline' : 'no compression: selection runs over the raw tokens', stripX, by + 11, T.n10, 10.5);
    }

    // --- row 3: ranked indexer scores + the draggable top-k cut --------------
    const ry = by + bh + 26, rh = 52;
    rRank = { x: stripX, y: ry, w: stripW, h: rh };
    const ranked = sl.ranked, nR = ranked.length, bw = stripW / Math.max(1, nR);
    rankN = nR;                                       // ranked-bar hit-test scale
    let rlo = Infinity, rhi = -Infinity;
    for (const e of ranked) { rlo = Math.min(rlo, e.is); rhi = Math.max(rhi, e.is); }
    txt(`indexer scores, ranked (${nR} candidate ${sl.grp > 1 ? 'entries' : 'tokens'})  —  drag the cut ↔`, stripX, ry - 5, T.n11, 10.5);
    for (let i2 = 0; i2 < nR; i2++) {
      const e = ranked[i2], t = rhi > rlo ? (e.is - rlo) / (rhi - rlo) : 0.5;
      const h2 = 4 + t * (rh - 6), x = stripX + i2 * bw;
      const inSet = sl.sel.has(e.b);
      ctx.fillStyle = e.holdsNeedle ? T.warn : inSet ? T.accent : alphaOf('n12', 0.38);
      ctx.fillRect(x, ry + rh - h2, Math.max(1, bw - 1), h2);
    }
    if (mode === 'dsa' || mode === 'csa') {
      const cx = stripX + Math.min(nR, sl.keep) * bw - 1;
      ctx.save(); ctx.strokeStyle = T.bad; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(cx, ry - 4); ctx.lineTo(cx, ry + rh + 4); ctx.stroke(); ctx.restore();
      txt(`k=${sl.keep} cut`, cx + 4, ry + 8, T.bad, 10);
      txt('kept', stripX + 2, ry + rh + 12, T.accent, 10);
      txt('discarded →', cx + 4, ry + rh + 12, T.n10, 10);
    }

    // --- row 4: the attention matrix, selected columns only -----------------
    const my = ry + rh + 30;
    const cols = mode === 'full' || mode === 'dsa' ? N : Math.ceil(N / (mode === 'hca' ? st.hc : st.c));
    const cell = Math.max(3, Math.min(13, Math.min((W * 0.40) / Math.max(cols, 1), (H - my - 14) / N)));
    rMat = { x: pad, y: my, w: cols * cell, h: N * cell };
    txt(`attention matrix — queries ↓ × ${cols === N ? 'keys' : 'compressed entries'} →`, pad, my - 9, T.n11, 10.5);
    for (let i2 = 0; i2 < N; i2++) {
      const row = selectFor(st, tn, i2);
      for (let b = 0; b < cols; b++) {
        const x = rMat.x + b * cell, y = rMat.y + i2 * cell;
        const e = row.ent[b];
        if (!e) { ctx.fillStyle = alphaOf('n12', 0.07); }
        else if (row.sel.has(b)) ctx.fillStyle = mixColor(T.n0, T.accent, 0.18 + 0.82 * Math.min(1, e.w * 3));
        else ctx.fillStyle = alphaOf('n12', 0.30);
        ctx.fillRect(x, y, Math.max(1, cell - 1), Math.max(1, cell - 1));
      }
    }
    ctx.save(); ctx.strokeStyle = T.n14; ctx.lineWidth = 1.4;
    ctx.strokeRect(rMat.x - 1, rMat.y + qi * cell - 1, cols * cell + 1, cell + 1); ctx.restore();
    if (mode === 'full' || mode === 'dsa') { ctx.save(); ctx.fillStyle = T.warn; ctx.fillRect(rMat.x + p * cell, rMat.y - 4, Math.max(1, cell - 1), 2.5); ctx.restore(); }

    // --- the live counters, all computed from the reader's settings ---------
    const px0 = pad + cols * cell + 26;
    let y = my + 5;
    const line = (s, col, sz) => { txt(s, px0, y, col || T.n12, sz || 11); y += (sz || 11) + 3; };
    const C = costs(mode, N, st.k, st.d, st.di, st.c, st.hc);
    const F = costs('full', N, st.k, st.d, st.di, st.c, st.hc);
    const ratio = C.total / F.total;
    const xN = crossover(mode, st.k, st.d, st.di, st.c, st.hc);
    const attendedTokens = mode === 'full' ? qi + 1 : Math.min((qi + 1), sl.keep * sl.grp);
    const idxScoresThisQuery = mode === 'full' ? 0 : mode === 'hca' ? 0 : sl.nEnt;

    line(`this query  q = ${qi}   (${qi + 1} past token${qi ? 's' : ''})`, T.n14, 12);
    line(`KV entries read        ${sl.keep}${sl.grp > 1 ? ` × ${sl.grp} tok` : ''}`, T.accent);
    line(`tokens reached         ${attendedTokens} of ${qi + 1}  (${pct(attendedTokens / (qi + 1))})`);
    line(`indexer scores         ${idxScoresThisQuery}${idxScoresThisQuery === qi + 1 && idxScoresThisQuery ? '  ← every past token' : ''}`, T.violet);
    if (mode !== 'full') line(`true attention mass lost   ${pct(sl.discarded)}`, sl.discarded > 0.25 ? T.bad : T.n12);
    if (mode === 'dsa' || mode === 'csa') line(`indexer found ${sl.recall}/${sl.recallOf} of the true top-k`, sl.recall === sl.recallOf ? T.ok : T.warn);
    y += 6;
    line(`whole prefill of N = ${N}`, T.n14, 12);
    line(`full attention   ${fmt(F.total)} mult`, T.n11);
    const rcol = mode === 'full' ? T.n11 : ratio < 1 ? T.ok : T.bad;
    line(`this mode        ${fmt(C.total)} mult`, rcol);
    line(`  = ${pct(ratio)} of full attention (lower is better; 100% = parity)`, rcol, 10.5);
    if (C.idx) line(`  indexer ${fmt(C.idx)} · attention ${fmt(C.attn)}${C.comp ? ` · compress ${fmt(C.comp)}` : ''}`, T.n10, 10.5);
    else if (C.comp) line(`  compress ${fmt(C.comp)} · attention ${fmt(C.attn)}`, T.n10, 10.5);
    if (mode !== 'full') {
      const above = xN > 0 && N >= xN;
      line(xN < 0 ? 'crossover: NEVER at these settings' : `crossover N* = ${xN} → N=${N} is ${above ? 'ABOVE it' : 'BELOW it'}`, above ? T.ok : T.bad, 11);
      if (!above) line('  here this LOSES to plain attention', T.bad, 10.5);
    }

    // --- the needle: the recall cliff --------------------------------------
    y += 8;
    if (sl.needle && p <= qi) {
      const nd = sl.needle;
      line(`needle @ p = ${p}`, T.warn, 12);
      if (mode === 'full') line('read like every other token — nothing is selected away', T.ok, 10.5);
      else if (mode === 'hca') line(`pooled into entry ${nd.entry.b} with ${nd.entry.hi - nd.entry.lo - 1} other token(s) — present, but blurred`, T.n12, 10.5);
      else {
        line(`indexer rank ${nd.rank + 1} of ${sl.nEnt}`, T.n12);
        if (nd.inSet) line('IN the selected set — it contributes', T.ok, 11);
        else { line('DROPPED — contributes exactly nothing', T.bad, 11); line(`  it held ${pct(nd.mass)} of the true mass`, T.bad, 10.5); }
      }
    } else if (sl.needle) {
      line(`needle @ p = ${p} is still in the future`, T.n10, 11);
    }

    // --- hover-to-inspect ---------------------------------------------------
    if (page.pointer.over && !grab) {
      const px = page.pointer.x, py = page.pointer.y;
      let tip = null;
      if (rStrip && px >= rStrip.x && px <= rStrip.x + rStrip.w && py >= rStrip.y - 8 && py <= rStrip.y + rStrip.h + 8) {
        const j = Math.max(0, Math.min(N - 1, Math.floor((px - rStrip.x) / cw)));
        tip = tipFor(st, tn, sl, qi, j, p);
      } else if (rRank && px >= rRank.x && px <= rRank.x + rRank.w && py >= rRank.y && py <= rRank.y + rRank.h) {
        const i2 = Math.max(0, Math.min(nR - 1, Math.floor((px - rRank.x) / bw)));
        const e = ranked[i2];
        tip = `rank ${i2 + 1} of ${nR}\nentry ${e.lo}${e.hi - e.lo > 1 ? `–${e.hi - 1}` : ''}  indexer ${e.is.toFixed(3)}\n${sl.sel.has(e.b) ? 'kept (rank ≤ k)' : 'discarded (rank > k)'}${e.holdsNeedle ? '\nholds the needle' : ''}`;
      } else if (rMat) {
        const hit = cellAt(rMat, N, cols, px, py);
        if (hit) {
          const row = selectFor(st, tn, hit.r), e = row.ent[hit.c];
          tip = e ? `q${hit.r} → ${sl.grp > 1 ? `entry ${hit.c} (tok ${e.lo}–${e.hi - 1})` : `k${hit.c}`}\nindexer ${e.is.toFixed(3)} · rank ${e.rank + 1}\n${row.sel.has(hit.c) ? `selected · weight ${pct(e.w)}` : 'not selected · weight 0'}`
            : `q${hit.r} → ${hit.c}: future (masked)`;
        }
      }
      if (tip) page.setTip(tip);
    }

    // --- probe (challenge checks) + readout --------------------------------
    page.probe = { ratio, xN, aboveCrossover: xN > 0 && N >= xN, needleIn: sl.needle ? sl.needle.inSet : null, needleVisible: !!sl.needle && p <= qi, recall: sl.recallOf ? sl.recall / sl.recallOf : 1, discarded: sl.discarded, keep: sl.keep, mode };
    let o = `${m.name}: ${m.desc}\n`;
    o += `q=${qi} of N=${N} · k=${st.k} · d=${st.d} · indexer dim=${st.di}` + (mode === 'csa' ? ` · compression ${st.c}` : mode === 'hca' ? ` · heavy compression ${st.hc}` : '') + '\n';
    o += `reads ${sl.keep} KV ${sl.grp > 1 ? 'compressed entries' : 'entries'} (${attendedTokens}/${qi + 1} tokens reached), scores ${idxScoresThisQuery} with the indexer; loses ${pct(sl.discarded)} of the true attention mass`;
    o += (mode === 'dsa' || mode === 'csa') ? `; the cheap score recovers ${sl.recall}/${sl.recallOf} of the true top-k.\n` : '.\n';
    o += `prefill cost ${fmt(C.total)} vs ${fmt(F.total)} mult for full attention = ${pct(ratio)} of full attention (lower is better; 100% = parity). ` + (mode === 'full' ? '' : xN < 0 ? 'No crossover: the indexer never pays for itself at these settings.' : `Crossover N* = ${xN}; N=${N} is ${N >= xN ? 'above' : 'below'} it.`) + '\n';
    o += !sl.needle || p > qi ? `needle @ p=${p} is ahead of the query — step the transport past it.`
      : mode === 'full' ? `needle @ p=${p}: read like every other token — this mode selects nothing away.`
      : mode === 'hca' ? `needle @ p=${p}: pooled into compressed entry ${sl.needle.entry.b} — present, but blurred together with its neighbours.`
      : `needle @ p=${p}: indexer rank ${sl.needle.rank + 1} of ${sl.nEnt} — ${sl.needle.inSet ? 'inside the top-k, it contributes' : `OUTSIDE the top-k: it contributes nothing, and ${pct(sl.needle.mass)} of the true attention mass goes with it`}.`;
    page.setReadout(o);
  },

  challenges: [
    {
      goal: 'Drop the needle: shrink k until the token at p falls out of the selected set.',
      hint: 'drag the k cut left, or drag the needle further from the query',
      check: (api) => ({ solved: api.probe.needleVisible && api.probe.needleIn === false, detail: api.probe.needleVisible ? 'the needle is still inside the top-k' : 'step the transport until the query is past p' }),
    },
    {
      goal: 'Find the crossover: make sparse selection actually cost LESS than full attention.',
      hint: 'the indexer is quadratic too — raise N, or lower k / the indexer dim',
      check: (api) => ({ solved: api.probe.mode !== 'full' && api.probe.ratio < 1, detail: api.probe.mode === 'full' ? 'switch off the dense baseline first' : `still ${(100 * api.probe.ratio).toFixed(1)}% of full attention` }),
    },
    {
      goal: 'Make the indexer lie: get its top-k recall below 60% of the true top-k.',
      hint: 'a cheaper indexer is a noisier one — drag the indexer dim down',
      check: (api) => ({ solved: (api.probe.mode === 'dsa' || api.probe.mode === 'csa') && api.probe.recall < 0.6, detail: `recall is ${(100 * api.probe.recall).toFixed(0)}%` }),
    },
  ],
}).then((page) => {
  window.__sasPage = page;
  const q = new URLSearchParams(location.search);
  const num = (key, lo, hi, rebuild) => { if (q.has(key)) page.controls.set(key, Math.max(lo, Math.min(hi, +q.get(key))), { silent: true, rebuild }); };
  if (q.has('mode') && MODES.some((m) => m.key === q.get('mode'))) page.controls.set('mode', q.get('mode'), { silent: true });
  num('N', 12, 48, true); num('k', 1, 24); num('c', 1, 8); num('hc', 2, 16);
  num('d', 8, 64); num('di', 2, 32); num('p', 0, 47); num('nb', 0, 4); num('seed', 1, 40);
  const t = page.controls._transport;
  // ?hover=x,y fakes the cursor so the hover-inspect path is screenshot-verifiable.
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') && t) { t.pause(); t.seek(parseInt(q.get('step'), 10)); }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});

// Hover text for one past token in the sequence strip: its indexer score, its
// rank, whether it made the cut, and what real attention would have given it.
function tipFor(st, tn, sl, qi, j, p) {
  const N = tn.N;
  if (j > qi) return `token ${j}: future — masked (j > q)`;
  const e = sl.ent.find((x) => j >= x.lo && j < x.hi);
  const is = tn.idxS[qi * N + j] + (j === p ? st.nb : 0);
  const ts = tn.trueS[qi * N + j] + (j === p ? st.nb : 0);
  const lines = [`token ${j}${j === p ? '  (the needle)' : ''}`, `indexer ${is.toFixed(3)} · true ${ts.toFixed(3)}`];
  if (e) {
    lines.push(`${sl.grp > 1 ? `in entry ${e.b} (tok ${e.lo}–${e.hi - 1}), ` : ''}rank ${e.rank + 1} of ${sl.nEnt}`);
    lines.push(sl.sel.has(e.b) ? `MADE THE CUT (k=${sl.keep}) · weight ${pct(e.w)}` : `outside the top-k → contributes nothing`);
  }
  return lines.join('\n');
}
