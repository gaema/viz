// kv-quant concept page -- quantizing the KV CACHE itself, not the weights.
//
// The sibling `quantization` page quantizes WEIGHTS: same arithmetic, different
// tensor, different consequence. Weights are a FIXED cost -- one file, sized
// once, and quantizing them buys model size. The KV cache is the term that
// GROWS: every decoded token appends one K and one V vector per layer per
// kv-head, forever, so at long context it is the memory that decides how much
// context fits. Quantizing it buys CONTEXT LENGTH.
//
// Mechanism shown here: store cached K and V as small integer codes (4-bit or
// 8-bit) plus a SCALE (and integer zero-point) shared by each GROUP of
// elements, and dequantize on read:
//     s = (max - min) / (2^bits - 1)         per group
//     z = round(-min / s)
//     q = clamp(round((x - min) / s), 0, 2^bits - 1)      <- what is stored
//     x' = min + s*q                                       <- what attention reads
// The gap x - x' is the reconstruction error.
//
// The part worth a page: WHICH ELEMENTS SHARE A SCALE. A group can run down a
// column (all channels of one token -- "per-token") or along a row (many tokens
// of one channel -- "per-channel"), and because K and V do not carry their large
// values in the same layout, the axis you pick moves the error a lot. Published
// KV-quant work treats the two halves separately for exactly this reason (see
// README.md for sources). Switch the axis and watch the error move; drag one
// cached value into an extreme and watch its group's scale stretch and coarsen
// every neighbour that shares it.
//
// The tensors here are SYNTHETIC and deliberately structured (a channel-aligned
// large-value band in K, a token-aligned one in V) so the axis effect is visible
// in a 10x20 slice. Which axis wins for a given real model is a property of that
// model's tensors and is something you measure, not something you assume.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

// ---- fixed model shape for the memory accounting (a generic mid-size open LM;
// the point is the SHAPE of the growth, not one particular checkpoint) --------
const L_LAYERS = 32, H_KV = 8, HEAD_DIM = 128;   // 8 kv-heads (grouped-query)
const PARAMS = 8e9, W_BITS = 4.5;                 // 8B params at ~4.5-bit weights
const SCALE_BITS = 16, ZP_BITS = 8;               // stored per group
const GiB = 1073741824;

const D = 10;            // channels drawn (a slice of head_dim)
const VLIM = 3.2;        // drag clamp

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const fmtB = (b) => (b >= GiB ? (b / GiB).toFixed(2) + ' GiB' : (b / 1048576).toFixed(1) + ' MiB');
const fmtTok = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : Math.round(n) + '');

// ---- the cache slice --------------------------------------------------------
// K carries its large values in a CHANNEL-aligned band (a few channels are big
// across every token); V carries them in a TOKEN-aligned one (a few tokens are
// big across every channel). Same generator, transposed structure -- which is
// precisely what makes the grouping axis a real choice rather than a detail.
function buildCache(seed, N) {
  const n = D * N, rk = seededRandn(seed, n, { std: 1 }), rv = seededRandn(seed + 7, n, { std: 1 });
  const K = new Float32Array(n), V = new Float32Array(n);
  for (let r = 0; r < D; r++) {
    const bigChan = (r % 4 === 2);
    for (let c = 0; c < N; c++) {
      const i = r * N + c, bigTok = (c % 6 === 3);
      K[i] = clamp(rk[i] * (bigChan ? 0.85 : 0.30) + (bigChan ? (r % 8 === 2 ? 1.9 : -1.9) : 0), -VLIM, VLIM);
      V[i] = clamp(rv[i] * (bigTok ? 0.95 : 0.32) + (bigTok ? (c % 12 === 3 ? 1.8 : -1.8) : 0), -VLIM, VLIM);
    }
  }
  return { K, V, N };
}

// ---- group-wise affine quantization of the filled part of the cache ---------
// axis 'token'   -> a group is G channels of ONE token   (down a column)
// axis 'channel' -> a group is G tokens of ONE channel   (along a row)
// Only the first `seq` columns exist: the rest of the cache has not been written
// yet, so it is neither quantized nor scored.
function groupsOf(N, seq, axis, G) {
  const out = [];
  if (axis === 'token') {
    for (let c = 0; c < seq; c++) for (let g0 = 0; g0 < D; g0 += G) {
      const idx = []; for (let r = g0; r < Math.min(g0 + G, D); r++) idx.push(r * N + c);
      out.push({ idx, tag: `token ${c}, chan ${g0}..${Math.min(g0 + G, D) - 1}` });
    }
  } else {
    for (let r = 0; r < D; r++) for (let g0 = 0; g0 < seq; g0 += G) {
      const idx = []; for (let c = g0; c < Math.min(g0 + G, seq); c++) idx.push(r * N + c);
      out.push({ idx, tag: `chan ${r}, tokens ${g0}..${Math.min(g0 + G, seq) - 1}` });
    }
  }
  return out;
}

function quantize(X, N, seq, axis, bits, G) {
  const levels = (1 << bits) - 1;
  const q = new Int32Array(D * N), deq = new Float32Array(D * N), gid = new Int32Array(D * N).fill(-1);
  const gs = groupsOf(N, seq, axis, G), meta = [];
  let se = 0, maxE = 0, n = 0;
  for (let g = 0; g < gs.length; g++) {
    const idx = gs[g].idx;
    let lo = Infinity, hi = -Infinity;
    for (const i of idx) { const v = X[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const s = (hi - lo) / levels || 1e-9, z = Math.round(-lo / s);
    for (const i of idx) {
      const code = clamp(Math.round((X[i] - lo) / s), 0, levels), x2 = lo + s * code;
      q[i] = code; deq[i] = x2; gid[i] = g;
      const e = X[i] - x2; se += e * e; if (Math.abs(e) > maxE) maxE = Math.abs(e); n++;
    }
    meta.push({ s, lo, hi, z, levels, tag: gs[g].tag, n: idx.length });
  }
  return { q, deq, gid, meta, levels, rmse: n ? Math.sqrt(se / n) : 0, maxE, nGroups: gs.length, nElem: n };
}
const rmseOf = (X, N, seq, axis, bits, G) => quantize(X, N, seq, axis, bits, G).rmse;

// ---- module state shared between build / draw / pointer ---------------------
let cur = null, sig = '';
let rOrig = null, rDeq = null, rErr = null;   // panel rects, captured in draw
let dragCell = null, sel = { r: 2, c: 3 }, shownTensor = 'K', seqNow = 1, nCols = 20;

mount({
  mount: 'body',
  title: 'kv-quant — quantizing the KV cache, not the weights',
  blurb: 'Weight quantization shrinks a fixed cost. The KV cache is the term that GROWS: one K and one V vector per token, per layer, per kv-head, held for the whole conversation — so past a few tens of thousands of tokens it outweighs the entire quantized weight file, and it is what decides how much context fits. So quantize the cache: store cached K and V as 4-bit (or 8-bit) codes with a scale + zero-point per GROUP of elements, and dequantize on read. Same arithmetic as weight quant, different tensor, different payoff — context length instead of model size. The non-obvious knob is WHICH elements share a scale: a group can run down a column (per-token) or along a row (per-channel), and because K and V hold their large values in different layouts, switching the axis moves the reconstruction error a long way. Press play to decode (the cache grows, the memory bars move), switch the axis, and DRAG any cached value ↕ to push it into an outlier — its whole group’s scale stretches and every neighbour sharing that scale gets coarser. Hover any cell for the full derivation: original, code bits, group scale, reconstruction, error.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  autoplay: true,
  compare: { key: 'axis', a: 'token', b: 'channel', labelA: 'per-token groups', labelB: 'per-channel groups' },
  challenges: [
    {
      goal: 'Show that the axis matters: find a setting where one grouping axis has at least 1.5× the error of the other.',
      hint: 'K hides a channel-aligned band and V a token-aligned one — pick a tensor, then flip the axis. Smaller groups blunt the difference.',
      check: (api) => {
        const a = api.probe.rTok ?? 0, b = api.probe.rCh ?? 0, hi = Math.max(a, b), lo = Math.min(a, b) || 1e-9;
        return { solved: hi / lo >= 1.5, detail: `per-token RMSE ${a.toFixed(4)} vs per-channel ${b.toFixed(4)} → ${(hi / lo).toFixed(2)}× (need ≥ 1.5×)` };
      },
    },
    {
      goal: 'Reach 3× more context than fp16 KV in the same memory budget, with RMSE under 0.08.',
      hint: 'fewer bits buys context; a bigger group cuts the scale overhead but raises the error.',
      check: (api) => ({
        solved: (api.probe.ctxGain ?? 0) >= 3 && (api.probe.rmse ?? 1) < 0.08,
        detail: `${(api.probe.ctxGain ?? 0).toFixed(2)}× the fp16 context, RMSE ${(api.probe.rmse ?? 1).toFixed(4)}`,
      }),
    },
  ],
  controls: (c, page) => {
    c.select('tensor', { label: 'show tensor', options: ['K', 'V'], value: 'K' });
    c.select('axis', {
      label: 'group axis',
      value: 'token',
      options: [{ value: 'token', label: 'per-token (down a column)' }, { value: 'channel', label: 'per-channel (along a row)' }],
    });
    c.slider('bits', { label: 'bits per element', min: 2, max: 8, step: 1, value: 4 });
    c.select('G', { label: 'group size', options: ['4', '8', '16', '32'], value: '8' });
    c.stepper('ctx', { label: 'context (tokens shown)', min: 8, max: 24, value: 20, rebuild: true });
    c.slider('budget', { label: 'memory budget (GiB)', min: 8, max: 96, step: 1, value: 24 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({
      compute: () => {
        const N = page.state.ctx | 0, steps = [];
        for (let t = 0; t < N; t++) steps.push({ t, label: `decode token ${t}: K and V column ${t} written to the cache and quantized in place — ${t + 1} of ${N} slots filled` });
        return steps;
      },
      speed: 3, loop: true,
    });
  },

  // Direct manipulation: press a cached cell and drag ↕ to change its value.
  // One dragged outlier stretches its group's [min,max], which stretches s, which
  // coarsens every OTHER element sharing that scale -- the whole lesson in one
  // gesture, and the reason group size and grouping axis exist as knobs at all.
  onPointer: (page, ev) => {
    if (!cur || !rOrig) return;
    const N = cur.N;
    if (ev.type === 'down') {
      const h = cellAt(rOrig, D, N, ev.x, ev.y);
      dragCell = (h && h.c < seqNow) ? h : null;
      if (dragCell) { sel = { r: dragCell.r, c: dragCell.c }; page.redraw(); }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragCell = null;
    } else if (ev.type === 'move' && dragCell && page.pointer.down) {
      const X = cur[shownTensor], i = dragCell.r * N + dragCell.c;
      X[i] = clamp(X[i] - ev.dy * 0.03, -VLIM, VLIM);
      page.redraw();
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    const N = st.ctx | 0;
    if (`${st.seed}|${N}` !== sig) { cur = buildCache(st.seed | 0, N); sig = `${st.seed}|${N}`; }
    nCols = N;
    r.clear(T.n0);

    const s = page.step(), t = s ? s.t : N - 1, seq = t + 1;
    seqNow = seq;
    const bits = st.bits | 0, G = +st.G, axis = st.axis, tn = st.tensor;
    shownTensor = tn;
    const X = cur[tn];
    const Q = quantize(X, N, seq, axis, bits, G);
    const levels = Q.levels;
    if (sel.c >= seq) sel = { r: sel.r, c: seq - 1 };

    // ===================== row 1: three panels ==============================
    const gut = 50, gap = 62, pad = 14;
    const cs = clamp(Math.floor(Math.min((W - gut - 2 * gap - 2 * pad) / (3 * N), (H * 0.30) / D)), 6, 18);
    const pw = N * cs, ph = D * cs;
    const y1 = 56;
    const px = [gut, gut + pw + gap, gut + 2 * (pw + gap)];
    rOrig = { x: px[0], y: y1, w: pw, h: ph };
    rDeq = { x: px[1], y: y1, w: pw, h: ph };
    rErr = { x: px[2], y: y1, w: pw, h: ph };

    const err = new Float32Array(D * N);
    for (let i = 0; i < D * N; i++) err[i] = Q.gid[i] >= 0 ? Math.abs(X[i] - Q.deq[i]) : 0;
    const dom = VLIM, eDom = Math.max(1e-6, Q.maxE);

    r.heatmap(X, { rows: D, cols: N, rect: rOrig, ramp: ramps.diverging, domain: [-dom, dom] });
    r.heatmap(Q.deq, { rows: D, cols: N, rect: rDeq, ramp: ramps.diverging, domain: [-dom, dom] });
    r.heatmap(err, { rows: D, cols: N, rect: rErr, ramp: ramps.sequential, domain: [0, eDom] });

    // veil the slots this decode step has not written yet
    for (const rect of [rOrig, rDeq, rErr]) {
      if (seq < N) { ctx.save(); ctx.fillStyle = alphaOf(T.n2, 0.9); ctx.fillRect(rect.x + seq * cs, rect.y, rect.w - seq * cs, rect.h); ctx.restore(); }
      ctx.save(); ctx.strokeStyle = T.n6; ctx.strokeRect(rect.x, rect.y, rect.w, rect.h); ctx.restore();
    }

    // group dividers on the original + reconstruction, drawn along the grouping axis
    const divider = (rect) => {
      ctx.save(); ctx.strokeStyle = rgbaToken('n14', 0.35); ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      if (axis === 'token') {
        for (let g0 = G; g0 < D; g0 += G) { const y = rect.y + g0 * cs; ctx.beginPath(); ctx.moveTo(rect.x, y); ctx.lineTo(rect.x + seq * cs, y); ctx.stroke(); }
        for (let c = 1; c < seq; c++) { const x = rect.x + c * cs; ctx.beginPath(); ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h); ctx.stroke(); }
      } else {
        for (let g0 = G; g0 < seq; g0 += G) { const x = rect.x + g0 * cs; ctx.beginPath(); ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h); ctx.stroke(); }
        for (let rr = 1; rr < D; rr++) { const y = rect.y + rr * cs; ctx.beginPath(); ctx.moveTo(rect.x, y); ctx.lineTo(rect.x + seq * cs, y); ctx.stroke(); }
      }
      ctx.setLineDash([]); ctx.restore();
    };
    divider(rOrig); divider(rDeq);

    // an ambient scan over the groups, so the "who shares a scale" unit is legible
    const scanG = Q.nGroups ? Math.floor((page.t || 0) * 1.6) % Q.nGroups : -1;
    if (scanG >= 0) {
      ctx.save(); ctx.fillStyle = rgbaToken('goldDeep', 0.20);
      const gsList = groupsOf(N, seq, axis, G)[scanG];
      for (const i of gsList.idx) {
        const rr = (i / N) | 0, cc = i % N;
        ctx.fillRect(rOrig.x + cc * cs, rOrig.y + rr * cs, cs, cs);
        ctx.fillRect(rDeq.x + cc * cs, rDeq.y + rr * cs, cs, cs);
      }
      ctx.restore();
    }

    // selection outline
    const outline = (rect, cell, col) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.strokeRect(rect.x + cell.c * cs + 1, rect.y + cell.r * cs + 1, cs - 2, cs - 2); ctx.restore(); };
    for (const rect of [rOrig, rDeq, rErr]) outline(rect, sel, T.n14);

    // panel titles + axis labels
    const mono = (n) => `${n}px ui-monospace, monospace`;
    r.label(`cached ${tn} — original (fp16)`, rOrig.x, y1 - 22, { color: T.accent, font: mono(11) });
    r.label(`reconstruction ${tn}′ = min + s·q  (${bits}-bit)`, rDeq.x, y1 - 22, { color: T.violet, font: mono(11) });
    r.label('|error| = |x − x′|', rErr.x, y1 - 22, { color: T.bad, font: mono(11) });
    r.label('drag a cell ↕ to make an outlier', rOrig.x, y1 - 8, { color: T.n10, font: mono(9) });
    r.label(`stored: ${Q.nGroups} scales + ${Q.nElem} codes`, rDeq.x, y1 - 8, { color: T.n10, font: mono(9) });
    r.label(`RMSE ${Q.rmse.toFixed(4)}  ·  max ${Q.maxE.toFixed(4)}`, rErr.x, y1 - 8, { color: T.n10, font: mono(9) });
    r.label('chan', 6, y1 + 10, { color: T.n10, font: mono(9) });
    r.label('↓', 6, y1 + 22, { color: T.n10, font: mono(9) });
    r.label('token →', rOrig.x, y1 + ph + 12, { color: T.n10, font: mono(9) });

    // ---- the per-group scales, drawn explicitly -----------------------------
    // Same orientation as the grouping: a strip UNDER the reconstruction for
    // per-token groups (one cell per token per channel-group), a strip to its
    // RIGHT for per-channel groups (one cell per channel per token-group).
    const sVals = Q.meta.map((m) => m.s), sMax = Math.max(1e-9, ...sVals, 1e-9);
    const scaleCell = Math.max(6, Math.min(cs, 12));
    let strip;
    if (axis === 'token') {
      const rows = Math.ceil(D / G);
      strip = { x: rDeq.x, y: rDeq.y + ph + 20, w: seq * cs, h: rows * scaleCell, rows, cols: seq };
      for (let g = 0; g < Q.nGroups; g++) {
        const c = (g / Math.ceil(D / G)) | 0, gr = g % Math.ceil(D / G);
        ctx.save(); ctx.fillStyle = alphaOf(T.violet, 0.18 + 0.72 * (sVals[g] / sMax));
        ctx.fillRect(strip.x + c * cs, strip.y + gr * scaleCell, cs, scaleCell);
        ctx.strokeStyle = rgbaToken('n14', 0.10); ctx.strokeRect(strip.x + c * cs, strip.y + gr * scaleCell, cs, scaleCell); ctx.restore();
      }
      r.label(`per-group scale s  (${Math.ceil(D / G)} per token)`, strip.x, strip.y - 6, { color: T.violet, font: mono(9) });
    } else {
      const cols = Math.ceil(seq / G);
      strip = { x: rDeq.x + pw + 8, y: rDeq.y, w: cols * scaleCell, h: ph, rows: D, cols };
      for (let g = 0; g < Q.nGroups; g++) {
        const rr = (g / Math.ceil(seq / G)) | 0, gc = g % Math.ceil(seq / G);
        ctx.save(); ctx.fillStyle = alphaOf(T.violet, 0.18 + 0.72 * (sVals[g] / sMax));
        ctx.fillRect(strip.x + gc * scaleCell, strip.y + rr * cs, scaleCell, cs);
        ctx.strokeStyle = rgbaToken('n14', 0.10); ctx.strokeRect(strip.x + gc * scaleCell, strip.y + rr * cs, scaleCell, cs); ctx.restore();
      }
      r.label(`s  (${Math.ceil(seq / G)}/chan)`, strip.x, strip.y - 6, { color: T.violet, font: mono(9) });
    }
    // highlight the selected element's own group in the scale strip
    const selIdx = sel.r * N + sel.c, selG = Q.gid[selIdx];
    if (selG >= 0) {
      ctx.save(); ctx.strokeStyle = T.goldDeep; ctx.lineWidth = 2;
      if (axis === 'token') { const c = (selG / Math.ceil(D / G)) | 0, gr = selG % Math.ceil(D / G); ctx.strokeRect(strip.x + c * cs, strip.y + gr * scaleCell, cs, scaleCell); }
      else { const rr = (selG / Math.ceil(seq / G)) | 0, gc = selG % Math.ceil(seq / G); ctx.strokeRect(strip.x + gc * scaleCell, strip.y + rr * cs, scaleCell, cs); }
      ctx.restore();
    }

    // ===================== row 2: the bits of one element ====================
    const y2 = Math.max(rOrig.y + ph + 62, strip.y + strip.h + 26);
    const m = selG >= 0 ? Q.meta[selG] : null;
    const xv = X[selIdx], code = Q.q[selIdx], rec = Q.deq[selIdx];
    r.label(`one cached value → ${bits} stored bits`, gut, y2 - 8, { color: T.n14, font: mono(11) });
    const bw = 20, bh = 22, bx0 = gut;
    const bstr = m ? code.toString(2).padStart(bits, '0') : '—'.repeat(bits);
    for (let b = 0; b < bits; b++) {
      const on = bstr[b] === '1', bx = bx0 + b * (bw + 3);
      ctx.save();
      ctx.fillStyle = on ? alphaOf(T.teal, 0.85) : alphaOf(T.n5, 0.9);
      ctx.fillRect(bx, y2, bw, bh);
      ctx.strokeStyle = T.n7; ctx.strokeRect(bx, y2, bw, bh);
      ctx.fillStyle = on ? T.n0 : T.n11; ctx.font = mono(12); ctx.textAlign = 'center';
      ctx.fillText(bstr[b], bx + bw / 2, y2 + bh - 6);
      ctx.restore();
      r.label(`${bits - 1 - b}`, bx + bw / 2 - 3, y2 + bh + 11, { color: T.n9, font: mono(8) });
    }
    const dx0 = bx0 + bits * (bw + 3) + 16;
    if (m) {
      r.label(`x  = ${xv.toFixed(4)}   (${tn}[chan ${sel.r}][token ${sel.c}], fp16)`, dx0, y2 + 9, { color: T.n14, font: mono(10) });
      r.label(`group ${m.tag}:  min ${m.lo.toFixed(3)}  max ${m.hi.toFixed(3)}  →  s = (max−min)/${levels} = ${m.s.toFixed(4)}   z = ${m.z}`, dx0, y2 + 23, { color: T.violet, font: mono(10) });
      r.label(`q  = clamp(round((x−min)/s), 0, ${levels}) = ${code}   →   x′ = min + s·q = ${rec.toFixed(4)}   err = ${(xv - rec).toFixed(4)}`, dx0, y2 + 37, { color: T.n12, font: mono(10) });
    }

    // ===================== row 3: memory + tradeoff ==========================
    // Effective bits: the codes, plus the per-group scale + zero-point amortised
    // over the group. The ACTUAL count is used for this slice (a half-full
    // per-channel group is genuinely more overhead per element); the nominal
    // group size is used for the steady-state projection bars.
    const effActual = bits + (SCALE_BITS + ZP_BITS) * (Q.nGroups / Math.max(1, Q.nElem));
    const effAt = (b) => b + (SCALE_BITS + ZP_BITS) / G;
    const bytesPerTok = (eb) => 2 * L_LAYERS * H_KV * HEAD_DIM * eb / 8;
    const wBytes = PARAMS * W_BITS / 8;
    const budget = st.budget * GiB, kvBudget = Math.max(0, budget - wBytes);
    const ctxAt = (eb) => kvBudget / bytesPerTok(eb);
    const ctxFp16 = ctxAt(16), ctxNow = ctxAt(effAt(bits)), ctxGain = ctxNow / ctxFp16;
    const crossover = wBytes / bytesPerTok(16);

    const y3 = y2 + 62, leftW = Math.min(430, W * 0.42);
    r.label(`memory budget ${st.budget} GiB — weights are fixed, the KV cache is not`, gut, y3 - 8, { color: T.n14, font: mono(11) });
    // stacked budget bar: weights | KV at the reachable context | free
    const sbY = y3 + 4, sbH = 16;
    const wFrac = Math.min(1, wBytes / budget);
    ctx.save();
    ctx.fillStyle = alphaOf(T.n9, 0.55); ctx.fillRect(gut, sbY, leftW * wFrac, sbH);
    ctx.fillStyle = alphaOf(T.accent, 0.65); ctx.fillRect(gut + leftW * wFrac, sbY, leftW * (1 - wFrac), sbH);
    ctx.strokeStyle = T.n6; ctx.strokeRect(gut, sbY, leftW, sbH);
    ctx.restore();
    r.label(`weights ${fmtB(wBytes)}`, gut + 3, sbY + 12, { color: T.n14, font: mono(9) });
    r.label(`KV cache — grows per token`, gut + leftW * wFrac + 5, sbY + 12, { color: T.n0, font: mono(9) });
    r.label(`at fp16 KV, the cache passes the whole weight file at ~${fmtTok(crossover)} tokens`, gut, sbY + sbH + 13, { color: T.n11, font: mono(9) });

    // context reachable at each precision, in this budget
    const rows = [
      { lab: 'fp16 KV', eb: 16, col: T.n9 },
      { lab: '8-bit KV', eb: effAt(8), col: T.teal },
      { lab: '4-bit KV', eb: effAt(4), col: T.ok },
      { lab: `${bits}-bit now`, eb: effAt(bits), col: T.warn },
    ];
    const maxCtx = Math.max(...rows.map((q) => ctxAt(q.eb)));
    const barX = gut + 86, barW = leftW - 86, rowY = sbY + sbH + 24;
    for (let i = 0; i < rows.length; i++) {
      const q = rows[i], y = rowY + i * 17, c = ctxAt(q.eb);
      r.label(q.lab, gut, y + 10, { color: T.n12, font: mono(9) });
      ctx.save(); ctx.fillStyle = alphaOf(q.col, 0.7); ctx.fillRect(barX, y, barW * (c / maxCtx), 12);
      ctx.strokeStyle = T.n5; ctx.strokeRect(barX, y, barW, 12); ctx.restore();
      r.label(`${fmtTok(c)} tokens  (${q.eb.toFixed(2)} eff. bits/elem)`, barX + 6, y + 10, { color: T.n13, font: mono(9) });
    }

    // ---- error vs bits, one line per axis -----------------------------------
    const cx = gut + leftW + 40, cw = Math.max(150, W - cx - 190), chY = y3 + 6, chH = 78;
    r.label(`RMSE vs bits — both axes, ${tn}, group ${G}`, cx, chY - 8, { color: T.n14, font: mono(10) });
    const eLo = -3.2, eHi = 0.2;
    const EX = (b) => cx + (b - 2) / 6 * cw;
    const EY = (e) => chY + chH - (clamp(Math.log10(e + 1e-9), eLo, eHi) - eLo) / (eHi - eLo) * chH;
    ctx.save(); ctx.strokeStyle = T.n4; ctx.strokeRect(cx, chY, cw, chH); ctx.restore();
    for (const ax of ['token', 'channel']) {
      ctx.save();
      ctx.strokeStyle = ax === 'token' ? T.accent : T.warn;
      ctx.lineWidth = ax === axis ? 2.2 : 1.2;
      ctx.globalAlpha = ax === axis ? 1 : 0.55;
      ctx.beginPath();
      for (let b = 2; b <= 8; b++) { const e = rmseOf(X, N, seq, ax, b, G), pxx = EX(b), pyy = EY(e); if (b === 2) ctx.moveTo(pxx, pyy); else ctx.lineTo(pxx, pyy); }
      ctx.stroke(); ctx.restore();
    }
    ctx.save(); ctx.fillStyle = T.bad; ctx.beginPath(); ctx.arc(EX(bits), EY(Q.rmse), 4, 0, 7); ctx.fill(); ctx.restore();
    r.label('2 bits', cx, chY + chH + 11, { color: T.n10, font: mono(8) });
    r.label('8 bits', cx + cw - 30, chY + chH + 11, { color: T.n10, font: mono(8) });
    r.label('per-token', cx + 4, chY + 11, { color: T.accent, font: mono(9) });
    r.label('per-channel', cx + 4, chY + 23, { color: T.warn, font: mono(9) });

    // ---- the axis table: both tensors x both axes, at the current bits/G ----
    const rTokK = rmseOf(cur.K, N, seq, 'token', bits, G), rChK = rmseOf(cur.K, N, seq, 'channel', bits, G);
    const rTokV = rmseOf(cur.V, N, seq, 'token', bits, G), rChV = rmseOf(cur.V, N, seq, 'channel', bits, G);
    const tx = cx + cw + 24, ty = chY - 8;
    r.label('RMSE by axis', tx, ty, { color: T.n14, font: mono(10) });
    r.label('per-token', tx + 48, ty + 15, { color: T.accent, font: mono(9) });
    r.label('per-chan', tx + 108, ty + 15, { color: T.warn, font: mono(9) });
    const cellTxt = (v, best) => ({ txt: v.toFixed(4), col: best ? T.ok : T.n12 });
    const kRow = [cellTxt(rTokK, rTokK <= rChK), cellTxt(rChK, rChK < rTokK)];
    const vRow = [cellTxt(rTokV, rTokV <= rChV), cellTxt(rChV, rChV < rTokV)];
    r.label('K', tx, ty + 31, { color: T.n13, font: mono(9) });
    r.label(kRow[0].txt, tx + 48, ty + 31, { color: kRow[0].col, font: mono(9) });
    r.label(kRow[1].txt, tx + 108, ty + 31, { color: kRow[1].col, font: mono(9) });
    r.label('V', tx, ty + 46, { color: T.n13, font: mono(9) });
    r.label(vRow[0].txt, tx + 48, ty + 46, { color: vRow[0].col, font: mono(9) });
    r.label(vRow[1].txt, tx + 108, ty + 46, { color: vRow[1].col, font: mono(9) });
    r.label('green = the better axis', tx, ty + 62, { color: T.n10, font: mono(8) });
    r.label('for THIS slice', tx, ty + 74, { color: T.n10, font: mono(8) });

    page.probe = { rmse: Q.rmse, rTok: tn === 'K' ? rTokK : rTokV, rCh: tn === 'K' ? rChK : rChV, ctxGain, t, N };

    // ---- hover: the full derivation ----------------------------------------
    if (page.pointer.over && !dragCell) {
      const p = page.pointer;
      const h = cellAt(rOrig, D, N, p.x, p.y) || cellAt(rDeq, D, N, p.x, p.y) || cellAt(rErr, D, N, p.x, p.y);
      if (h) {
        const i = h.r * N + h.c, g = Q.gid[i];
        if (g >= 0) {
          const gm = Q.meta[g];
          page.setTip(
            `${tn}[chan ${h.r}][token ${h.c}]\n` +
            `x  = ${X[i].toFixed(4)}  (fp16)\n` +
            `group: ${gm.tag}  (${gm.n} elem share one scale)\n` +
            `min ${gm.lo.toFixed(3)}  max ${gm.hi.toFixed(3)}  s = ${gm.s.toFixed(4)}  z = ${gm.z}\n` +
            `q  = ${Q.q[i]} of 0..${levels}  = 0b${Q.q[i].toString(2).padStart(bits, '0')}\n` +
            `x′ = ${Q.deq[i].toFixed(4)}   err = ${(X[i] - Q.deq[i]).toFixed(4)}\n` +
            `click + drag ↕ to change x`);
        } else {
          page.setTip(`${tn}[chan ${h.r}][token ${h.c}] — not written yet (decode step ${h.c})`);
        }
      }
    }

    // ---- readout ------------------------------------------------------------
    const better = tn === 'K' ? (rTokK <= rChK ? 'per-token' : 'per-channel') : (rTokV <= rChV ? 'per-token' : 'per-channel');
    const ratio = tn === 'K' ? Math.max(rTokK, rChK) / Math.max(1e-9, Math.min(rTokK, rChK)) : Math.max(rTokV, rChV) / Math.max(1e-9, Math.min(rTokV, rChV));
    let o = `${s ? s.label : `cache full: ${N} of ${N} slots`}    tier:${r.name}\n`;
    o += `${tn} cache, ${bits}-bit, groups of ${G} ${axis === 'token' ? 'channels within one token' : 'tokens within one channel'}: `;
    o += `s=(max−min)/${levels}, z=round(−min/s), q=clamp(round((x−min)/s),0,${levels}), x′=min+s·q.  `;
    o += `RMSE ${Q.rmse.toFixed(4)}, max|err| ${Q.maxE.toFixed(4)} over ${Q.nElem} live elements in ${Q.nGroups} groups → ${effActual.toFixed(2)} effective bits/element on this slice.\n`;
    o += `axis: ${better} wins here by ${ratio.toFixed(2)}× on ${tn} — the large values in this slice line up with that axis, so a group cut along it spans a narrow range and its scale stays small. `;
    o += `Cut across it and one large value stretches its group's scale and coarsens every neighbour sharing it (drag one to see it happen).\n`;
    o += `memory: ${st.budget} GiB budget − ${fmtB(wBytes)} weights = ${fmtB(kvBudget)} for KV → ${fmtTok(ctxNow)} tokens at ${effAt(bits).toFixed(2)} eff. bits vs ${fmtTok(ctxFp16)} at fp16 = ${(ctxGain).toFixed(2)}× the context. `;
    o += `Weight quant (see the quantization page) buys model SIZE once; this buys CONTEXT, every token, for as long as the conversation runs.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__kvqPage = page;
  const q = new URLSearchParams(location.search);
  for (const k of ['tensor', 'axis', 'G']) if (q.has(k)) page.controls.set(k, q.get(k));
  for (const k of ['bits', 'budget', 'seed']) if (q.has(k)) page.controls.set(k, +q.get(k));
  if (q.has('ctx')) page.controls.set('ctx', +q.get('ctx'), { rebuild: true });
  const tr = page.controls._transport;
  if (q.has('sel')) { const [rr, cc] = q.get('sel').split(',').map(Number); sel = { r: rr | 0, c: cc | 0 }; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  const posKey = q.has('step') ? 'step' : (q.has('pos') ? 'pos' : null);
  if ((posKey || q.has('hover') || q.has('drag')) && tr) tr.pause();
  if (posKey && tr) tr.seek(parseInt(q.get(posKey), 10));
  if (q.get('play') === '1' && tr) tr.play();
  page.redraw();
  // ?drag=r,c,v[;...] is the headless stand-in for a pointer drag (--screenshot has
  // no cursor): it sets one cached value directly, so the outlier-stretches-the-
  // group state is reproducible from a URL.
  //
  // It needs `cur`, and draw() builds the cache on the FIRST paint. This handler
  // runs as a microtask off mount()'s promise -- before any frame -- and the
  // page.redraw() that used to sit above it is rAF-coalesced, not synchronous, so
  // `cur` was still null and the guard dropped the hook without a word. Apply it
  // after the first paint instead, then repaint.
  if (q.has('drag')) requestAnimationFrame(() => requestAnimationFrame(() => {
    if (!cur) return;
    for (const part of q.get('drag').split(';')) {
      const [rr, cc, vv] = part.split(',').map(Number);
      if (rr >= 0 && rr < D && cc >= 0 && cc < cur.N) {
        cur[q.get('tensor') || 'K'][rr * cur.N + cc] = Math.max(-VLIM, Math.min(VLIM, vv));
        if (!q.has('sel')) sel = { r: rr, c: cc };   // an explicit ?sel still wins
      }
    }
    page.redraw();
  }));
});
