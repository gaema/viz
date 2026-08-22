// context-parallelism concept page -- the SERVING CHOICE on top of a sharded
// sequence: pass the KV, or pass the queries?
//
// The sequence is already split across devices and attention is already
// computed by rotating blocks around a ring with an online-softmax
// accumulation. That machinery -- the rotation, the rescale recurrence, the
// compute/transfer overlap -- is the neighbouring `ring-attention` page's
// subject and is NOT re-taught here. This page asks the one question that
// mechanism leaves open, and that a real serving system answers differently at
// different moments:
//
//   Both devices need every (query, key) pair to meet. Exactly one of the two
//   operands has to travel. Which one?
//
//   PASS-KV : keys and values circulate; each device keeps its queries put.
//             The wire carries the KV -- and with grouped-query attention there
//             are far fewer KV heads than query heads, so that payload is small
//             per token.
//   PASS-Q  : the query block circulates against resident KV. The wire carries
//             the queries, and with them the running partial output and the two
//             softmax statistics they have accumulated so far, because a query
//             that moves must take its unfinished answer with it.
//
// Both compute the SAME attention output, bit-for-bit-equivalent maths. Only
// the traffic differs.
//
// WHY IT FLIPS BY PHASE. Pass-KV's payload is the resident KV shard: it scales
// with how much context the device holds and not at all with how many queries
// are in flight. Pass-Q's payload scales with the queries. During prefill a
// device has a whole prompt shard of queries live at once, so pass-Q's payload
// is enormous and pass-KV wins. During decode there is one query token per
// sequence and the KV cache is the biggest object in the system, so pass-Q's
// payload is tiny and it wins by orders of magnitude. The two cost curves
// therefore CROSS, and the crossing is what this page draws.
//
// Every byte figure on screen is computed live from the head counts, sequence
// length, device count and phase the reader sets -- nothing is a stored
// constant, so the arithmetic in each tooltip can be checked by hand.
//
// Public sources:
//   Liu, Zaharia, Abbeel, "Ring Attention with Blockwise Transformers for
//     Near-Infinite Context", https://arxiv.org/abs/2310.01889
//   Liu et al., "World Model on Million-Length Video and Language with
//     Blockwise RingAttention", https://arxiv.org/abs/2402.08268
//   Grattafiori et al., "The Llama 3 Herd of Models",
//     https://arxiv.org/abs/2407.21783 -- describes long-context serving with
//     the key/value tensors gathered rather than the queries, on the grounds
//     that grouped-query attention makes K,V much smaller than Q.
//   Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
//     from Multi-Head Checkpoints", https://arxiv.org/abs/2305.13245
//   Brandon et al., "Striped Attention: Faster Ring Attention for Causal
//     Transformers", https://arxiv.org/abs/2311.09431

import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';

// ---------------------------------------------------------------------------
// The cost model. Bytes on the wire per ring hop, per device.
// ---------------------------------------------------------------------------
const D = 128;          // head dim, held fixed so the head COUNTS are the story
const EL = 2;           // Q, K, V elements are 16-bit
const ACC = 4;          // the running partial output is fp32, as in online softmax
const STAT = 8;         // the two softmax statistics (running max, running sum), fp32 each

// Bytes one query token costs to move, per hop: the query itself, plus the
// partial answer it has accumulated so far, plus its two softmax statistics.
// A moving query cannot leave its unfinished work behind.
const qTokenBytes = (Hq) => Hq * (EL * D + ACC * D + STAT);
// Bytes one token's worth of cache costs to move, per hop: K and V, over the
// KV heads only. This is the term grouped-query attention shrinks.
const kvTokenBytes = (Hkv) => 2 * EL * Hkv * D;

function derive(st) {
  const P = Math.max(2, st.devices | 0);
  const S = Math.pow(2, st.seqPow | 0);
  const Hq = Math.max(1, st.qheads | 0);
  const Hkv = Math.max(1, Math.min(st.kvheads | 0, Hq));
  const shard = Math.max(1, Math.round(S / P));      // tokens of context per device
  const persist = !!st.persist;
  const qTok = qTokenBytes(Hq);
  const kvTok = kvTokenBytes(Hkv);
  const kvResident = kvTok * shard;                  // the whole KV shard, one hop

  // pass-KV per hop: the resident shard -- unless the shards are PERSISTED, in
  // which case a step only has to move the KV of the tokens produced this step.
  const passKV = (q) => (persist ? kvTok * q : kvResident);
  const passQ = (q) => qTok * q;

  // Where the two curves meet, in query tokens per device. With persisted KV
  // shards both costs are linear in q, so they are parallel and never cross.
  const cross = persist ? null : kvResident / qTok;

  // The operating point: prefill has the whole prompt shard of queries live;
  // decode has one query token per sequence in the batch.
  const qOp = st.phase === 'decode'
    ? Math.max(1, Math.min(st.batch | 0, shard))
    : shard;

  const bKV = passKV(qOp), bQ = passQ(qOp);
  return {
    P, S, Hq, Hkv, shard, persist, qTok, kvTok, kvResident,
    passKV, passQ, cross, qOp, bKV, bQ,
    group: Hq / Hkv,
    winner: bQ < bKV ? 'pass-Q' : 'pass-KV',
    // A LOWER-is-better axis: this is pass-Q's bytes as a percent of pass-KV's,
    // so >100% means pass-Q is WORSE. Direction is always stated at the callsite.
    pctQofKV: (bQ / bKV) * 100,
  };
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
const UNITS = [[1024 ** 3, 'GiB'], [1024 ** 2, 'MiB'], [1024, 'KiB']];
function fmtB(n) {
  for (const [k, u] of UNITS) if (n >= k) return (n / k).toFixed(n / k < 10 ? 2 : 1) + ' ' + u;
  return Math.round(n) + ' B';
}
const fmtN = (n) => Math.round(n).toLocaleString('en-US');
function fmtTok(n) {
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(n % (1024 * 1024) ? 2 : 0) + 'M';
  if (n >= 1024) return (n / 1024).toFixed(n % 1024 ? 1 : 0) + 'K';
  return String(Math.round(n));
}
// Percent with the direction spelled out, because a bare ratio on a
// lower-is-better axis reads equally well as "better" and "worse".
function pctPhrase(pct) {
  if (pct < 100) return `${pct < 0.01 ? pct.toFixed(4) : pct < 1 ? pct.toFixed(3) : pct.toFixed(1)}% of pass-KV's bytes — ${(100 / pct).toFixed(pct < 1 ? 0 : 1)}x LESS traffic`;
  if (pct > 100) return `${pct.toFixed(1)}% of pass-KV's bytes — ${(pct / 100).toFixed(2)}x MORE traffic`;
  return '100% of pass-KV\'s bytes — exact parity';
}

// Divisors of Hq, so a dragged grouping lands on an equal-sized group.
function nearestDivisor(Hq, want) {
  let best = 1, bd = Infinity;
  for (let k = 1; k <= Hq; k++) if (Hq % k === 0) { const dd = Math.abs(k - want); if (dd < bd) { bd = dd; best = k; } }
  return best;
}

// ---------------------------------------------------------------------------
// Geometry captured each draw(), reused by onPointer() and the hover paths.
// ---------------------------------------------------------------------------
let geom = null;
let grab = null;      // 'cross' | 'op' | null -- which handle is held

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// Drag the crossing: solve for the KV-head count that would put it at qStar.
// This is the whole reason grouped-query attention changes this decision --
// moving the crossing IS choosing a grouping.
function setCrossTo(page, qStar) {
  const m = derive(page.state);
  if (m.persist) return;
  const want = (qStar * m.qTok) / (2 * EL * D * m.shard);
  page.controls.set('kvheads', nearestDivisor(m.Hq, Math.max(1, Math.round(want))), { rebuild: true });
}

// Drag the operating point: below a full prompt shard that is a decode batch;
// at the shard it is a full prefill.
function setOpTo(page, q) {
  const m = derive(page.state);
  if (q >= m.shard) { page.controls.set('phase', 'prefill', { rebuild: true }); return; }
  page.controls.set('phase', 'decode', { rebuild: true, silent: true });
  page.controls.set('batch', Math.max(1, Math.min(64, Math.round(q))), { rebuild: true });
}

mount({
  mount: 'body',
  title: 'context-parallelism — pass the KV, or pass the queries?',
  blurb: 'The sequence is already sharded across devices and attention is already computed by rotating blocks around a ring (that mechanism is the ring-attention page). This page is the serving decision left over: for every (query, key) pair to meet, exactly one operand must travel. Pass-KV circulates keys and values and keeps queries put; pass-Q circulates queries against resident KV. Both compute the same attention output — only the traffic differs. Drag the sequence length, device count and head counts and watch the two byte-per-hop curves CROSS: pass-KV is flat in the number of live queries, pass-Q is linear in it, so prefill and decode land on opposite sides of the crossing. Drag the ✕ itself to re-solve the crossing for a KV-head grouping.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  compare: { key: 'phase', a: 'prefill', b: 'decode', labelA: 'prefill — a whole prompt shard of queries is live', labelB: 'decode — one query token per sequence, enormous resident KV' },

  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const inPlot = ev.x >= g.px0 - 18 && ev.x <= g.px1 + 18 && ev.y >= g.py0 - 12 && ev.y <= g.py1 + 12;
    if (ev.type === 'down') {
      grab = null;
      if (!inPlot) return;
      if (g.crossX != null && Math.abs(ev.x - g.crossX) < 18 && Math.abs(ev.y - g.crossY) < 22) grab = 'cross';
      else if (Math.abs(ev.x - g.opX) < 16) grab = 'op';
      else grab = 'op';                                  // a bare click on the plot moves the operating point
      if (grab === 'cross') setCrossTo(page, g.qAt(ev.x));
      else setOpTo(page, g.qAt(ev.x));
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      if (grab === 'cross') setCrossTo(page, g.qAt(ev.x));
      else setOpTo(page, g.qAt(ev.x));
    }
  },

  challenges: [
    {
      goal: 'Make pass-Q the cheaper pattern — get it under 1% of pass-KV\'s bytes per hop.',
      hint: 'switch to decode: one query token per sequence against a whole resident KV shard.',
      check: (api) => ({ solved: (api.probe.pct ?? 1e9) < 1, detail: `pass-Q is at ${(api.probe.pct ?? 0).toFixed(2)}% of pass-KV (need < 1%)` }),
    },
    {
      goal: 'Without leaving prefill, push the crossing past 8,000 query tokens per device.',
      hint: 'the crossing moves with the KV-head count — fewer groups means a smaller KV payload, so pass-Q has further to climb.',
      check: (api) => ({ solved: api.probe.phase === 'prefill' && (api.probe.cross ?? 0) > 8000, detail: api.probe.phase !== 'prefill' ? 'switch back to prefill' : `crossing at ${fmtN(api.probe.cross ?? 0)} query tokens/device` }),
    },
  ],

  controls: (c, page) => {
    // n, not the token count: the value field is click-to-type and must round-trip
    // a plain number. The token count itself is on the canvas header and readout.
    c.slider('seqPow', { label: 'sequence length: 2^n tokens', min: 12, max: 21, step: 1, value: 17, rebuild: true });
    c.stepper('devices', { label: 'devices in the ring', min: 2, max: 16, value: 8 });
    c.stepper('qheads', { label: 'query heads', min: 4, max: 64, step: 4, value: 32 });
    c.stepper('kvheads', { label: 'KV heads (grouping)', min: 1, max: 64, value: 8 });
    c.select('phase', { label: 'phase', options: [{ value: 'prefill', label: 'prefill (prompt)' }, { value: 'decode', label: 'decode (one token)' }], value: 'prefill', rebuild: true });
    c.stepper('batch', { label: 'decode: seqs per device', min: 1, max: 64, value: 1 });
    c.toggle('persist', { label: 'KV shards persist (only new KV moves)', value: false, rebuild: true });
    c.select('show', { label: 'ring shows', options: [{ value: 'auto', label: 'auto — the winner' }, { value: 'kv', label: 'pass-KV' }, { value: 'q', label: 'pass-Q' }], value: 'auto' });
    c.transport({ compute: () => { const m = derive(page.state); return Array.from({ length: m.P }, (_, k) => ({ k, label: `hop ${k + 1} of ${m.P} — every device holds a block it did not start with` })); }, speed: 1.1, loop: true });
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const m = derive(st);
    const W = page.W, H = page.H;
    const s = page.step();
    const hop = s ? s.k : 0;
    const mode = st.show === 'auto' ? (m.winner === 'pass-Q' ? 'q' : 'kv') : st.show;
    const movingIsKV = mode === 'kv';
    page.probe = { pct: m.pctQofKV, cross: m.cross, phase: st.phase, kvheads: m.Hkv, winner: m.winner };

    const KVC = T.accent, QC = T.warn;

    // ---------------------------------------------------------------- header
    // Kept compact deliberately: at 12px monospace a longer line overruns the
    // canvas on a narrow stage and gets clipped mid-word.
    r.label(`${m.S >= 1024 ? fmtTok(m.S) : m.S} tok / ${m.P} dev  ·  ${fmtN(m.shard)} tok ctx per device  ·  H_q ${m.Hq}  ·  H_kv ${m.Hkv} (${m.group % 1 ? m.group.toFixed(2) : m.group}:1)  ·  d=${D}`,
      14, 22, { color: T.n12, font: '12px ui-monospace, monospace' });

    // ------------------------------------------------------------- the ring
    // One row of devices. Whichever operand is travelling is drawn as a chip
    // riding the arrow between neighbours; the other stays pinned in its box.
    const ringY = 44, boxH = 58, pad = 14;
    const slot = (W - 2 * pad) / m.P, boxW = Math.min(96, slot - 16);
    const cx = (i) => pad + i * slot + slot / 2;
    r.label(movingIsKV ? 'PASS-KV — the K,V blocks travel, the queries stay put' : 'PASS-Q — the query blocks travel (carrying their partial answers), the K,V stay put',
      14, ringY - 8, { color: movingIsKV ? KVC : QC, font: '600 12px ui-monospace, monospace' });

    const devRects = [];
    for (let i = 0; i < m.P; i++) {
      const x = cx(i) - boxW / 2;
      const held = (i - hop + m.P * 4) % m.P;         // which block index device i currently holds
      const col = categorical(held);
      ctx.save();
      roundRect(ctx, x, ringY + 6, boxW, boxH, 7);
      ctx.fillStyle = alphaOf(T.n2, 1); ctx.fill();
      ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.stroke();
      ctx.restore();
      r.label(`dev ${i}`, cx(i), ringY + 21, { color: T.n11, font: '10px ui-monospace, monospace', align: 'center' });
      // the resident operand (never moves) and the travelling operand (does).
      const resident = movingIsKV ? `Q${i}` : `KV${i}`;
      const travelling = movingIsKV ? `KV${held}` : `Q${held}`;
      ctx.save();
      roundRect(ctx, x + 6, ringY + 27, boxW / 2 - 9, 20, 4);
      ctx.fillStyle = alphaOf(movingIsKV ? QC : KVC, 0.18); ctx.fill();
      ctx.strokeStyle = alphaOf(movingIsKV ? QC : KVC, 0.75); ctx.lineWidth = 1; ctx.stroke();
      ctx.restore();
      r.label(resident, x + 6 + (boxW / 2 - 9) / 2, ringY + 41, { color: movingIsKV ? QC : KVC, font: '10px ui-monospace, monospace', align: 'center' });
      ctx.save();
      roundRect(ctx, x + boxW / 2 + 3, ringY + 27, boxW / 2 - 9, 20, 4);
      ctx.fillStyle = alphaOf(col, 0.85); ctx.fill();
      ctx.strokeStyle = alphaOf(col, 1); ctx.lineWidth = 1; ctx.stroke();
      ctx.restore();
      r.label(travelling, x + boxW / 2 + 3 + (boxW / 2 - 9) / 2, ringY + 41, { color: inkOn(col), font: '10px ui-monospace, monospace', align: 'center' });
      devRects.push({ x, y: ringY + 6, w: boxW, h: boxH, i, held });
    }
    // hop arrows: block i moves to its right-hand neighbour (last wraps around).
    const perHop = movingIsKV ? m.passKV(m.qOp) : m.passQ(m.qOp);
    for (let i = 0; i < m.P - 1; i++) {
      r.arrow({ x: cx(i) + boxW / 2 + 2, y: ringY + 35 }, { x: cx(i + 1) - boxW / 2 - 3, y: ringY + 35 },
        { color: movingIsKV ? KVC : QC, width: 2, head: 7, alpha: 0.85 });
    }
    // the wrap edge, drawn as a return path under the row
    ctx.save();
    ctx.strokeStyle = alphaOf(movingIsKV ? KVC : QC, 0.55); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(cx(m.P - 1), ringY + 6 + boxH); ctx.lineTo(cx(m.P - 1), ringY + boxH + 22);
    ctx.lineTo(cx(0), ringY + boxH + 22); ctx.lineTo(cx(0), ringY + 6 + boxH);
    ctx.stroke(); ctx.restore();
    r.label(`each arrow carries ${fmtB(perHop)} this hop`, W / 2, ringY + boxH + 36, { color: T.n11, font: '11px ui-monospace, monospace', align: 'center' });

    // -------------------------------------------------------------- the plot
    const px0 = 76, px1 = Math.max(300, Math.min(W * 0.60, W - 302));
    const py0 = 188, py1 = H - 56;
    const qMax = Math.max(8, m.shard);
    const lx0 = 0, lx1 = Math.log10(qMax);
    const X = (q) => px0 + ((Math.log10(Math.max(1, q)) - lx0) / (lx1 - lx0 || 1)) * (px1 - px0);
    const qAt = (x) => Math.pow(10, lx0 + ((x - px0) / (px1 - px0)) * (lx1 - lx0));

    const cand = [m.passQ(1), m.passQ(qMax), m.passKV(1), m.passKV(qMax)];
    const yLo = Math.log10(Math.max(1, Math.min(...cand))) - 0.25;
    const yHi = Math.log10(Math.max(...cand)) + 0.25;
    const Y = (b) => py1 - ((Math.log10(Math.max(1, b)) - yLo) / (yHi - yLo || 1)) * (py1 - py0);

    // frame + decade gridlines
    ctx.save();
    ctx.strokeStyle = rgbaToken('n14', 0.10); ctx.lineWidth = 1;
    for (let e = Math.ceil(yLo); e <= Math.floor(yHi); e++) {
      const y = Y(Math.pow(10, e));
      ctx.beginPath(); ctx.moveTo(px0, y); ctx.lineTo(px1, y); ctx.stroke();
      r.label(fmtB(Math.pow(10, e)), px0 - 6, y + 3, { color: T.n10, font: '9px ui-monospace, monospace', align: 'right' });
    }
    for (let e = 0; e <= Math.floor(lx1); e++) {
      const x = X(Math.pow(10, e));
      ctx.beginPath(); ctx.moveTo(x, py0); ctx.lineTo(x, py1); ctx.stroke();
      r.label(fmtN(Math.pow(10, e)), x, py1 + 14, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
    }
    ctx.strokeStyle = T.n7; ctx.strokeRect(px0, py0, px1 - px0, py1 - py0);
    ctx.restore();
    r.label('bytes on the wire, per hop, per device', px0, py0 - 10, { color: T.n12, font: '600 11px ui-monospace, monospace' });
    r.label('query tokens live per device this step  →', px0, py1 + 30, { color: T.n11, font: '10px ui-monospace, monospace' });

    // the two curves. Both are straight on this log-log frame; pass-KV is flat
    // in q unless the shards persist, pass-Q is always linear in q.
    const curve = (fn, col, width) => {
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = width; ctx.beginPath();
      for (let i = 0; i <= 64; i++) {
        const q = Math.pow(10, lx0 + (i / 64) * (lx1 - lx0));
        const x = X(q), y = Y(fn(q));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.restore();
    };
    curve(m.passKV, KVC, 2.4);
    curve(m.passQ, QC, 2.4);
    r.label('pass-KV', px0 + 8, Y(m.passKV(Math.pow(10, lx1 * 0.10))) - 7, { color: KVC, font: '600 11px ui-monospace, monospace' });
    r.label('pass-Q', px1 - 8, Y(m.passQ(qMax)) + (Y(m.passQ(qMax)) < py0 + 22 ? 16 : -8), { color: QC, font: '600 11px ui-monospace, monospace', align: 'right' });

    // the crossing
    let crossX = null, crossY = null;
    if (m.cross != null && m.cross >= 1 && m.cross <= qMax) {
      crossX = X(m.cross); crossY = Y(m.passQ(m.cross));
      ctx.save();
      ctx.strokeStyle = T.n13; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(crossX - 7, crossY - 7); ctx.lineTo(crossX + 7, crossY + 7);
      ctx.moveTo(crossX + 7, crossY - 7); ctx.lineTo(crossX - 7, crossY + 7); ctx.stroke();
      ctx.strokeStyle = grab === 'cross' ? T.violet : rgbaToken('n14', 0.35); ctx.lineWidth = grab === 'cross' ? 2 : 1;
      ctx.beginPath(); ctx.arc(crossX, crossY, 13, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
      // Below-right of the ✕: that quadrant is empty (pass-Q descends to the
      // left there), whereas above it collides with the pass-Q curve label.
      const capRight = crossX + 18 > px1 - 150;
      r.label(`↔ crossing: ${fmtN(m.cross)} query tokens`, capRight ? crossX - 18 : crossX + 18, crossY + 24,
        { color: T.n13, font: '10px ui-monospace, monospace', align: capRight ? 'right' : 'left' });
    }

    // the operating point
    const opX = X(m.qOp);
    ctx.save();
    ctx.strokeStyle = grab === 'op' ? T.violet : T.violetDeep; ctx.lineWidth = grab === 'op' ? 2.5 : 1.6; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(opX, py0); ctx.lineTo(opX, py1); ctx.stroke();
    ctx.restore();
    for (const [b, col] of [[m.bKV, KVC], [m.bQ, QC]]) {
      ctx.save(); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(opX, Y(b), 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = T.n0; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore();
    }
    // Right-align the operating-point caption once the line is near the plot's
    // right edge, so it does not spill over the verdict card beside it.
    const opNearEdge = opX > px1 - 60;
    r.label(`${st.phase} · q = ${fmtN(m.qOp)}`, opNearEdge ? opX - 4 : opX, py0 - 24,
      { color: T.violetDeep, font: '600 10px ui-monospace, monospace', align: opNearEdge ? 'right' : 'center' });
    r.label('↔ drag anywhere on the plot to move the operating point', px0, py0 - 24, { color: T.n10, font: '9px ui-monospace, monospace' });

    // -------------------------------------------------------- the verdict card
    const cx0 = px1 + 30, cw = Math.max(180, W - cx0 - 12);
    ctx.save();
    roundRect(ctx, cx0, py0 - 34, cw, (py1 + 34) - (py0 - 34), 8);
    ctx.fillStyle = alphaOf(m.winner === 'pass-Q' ? QC : KVC, 0.10); ctx.fill();
    ctx.strokeStyle = alphaOf(m.winner === 'pass-Q' ? QC : KVC, 0.55); ctx.lineWidth = 1.2; ctx.stroke();
    ctx.restore();
    let ty = py0 - 12;
    const line = (txt, col, font) => { r.label(txt, cx0 + 12, ty, { color: col, font: font || '11px ui-monospace, monospace' }); ty += 16; };
    line(`VERDICT — ${st.phase}`, T.n11, '600 11px ui-monospace, monospace');
    line(m.winner.toUpperCase(), m.winner === 'pass-Q' ? QC : KVC, '700 17px ui-monospace, monospace');
    ty += 4;
    line(`pass-KV  ${fmtB(m.bKV)}`, KVC);
    line(`pass-Q   ${fmtB(m.bQ)}`, QC);
    ty += 6;
    line('pass-Q traffic, as a percent of', T.n11, '10px ui-monospace, monospace');
    line('pass-KV (lower is better;', T.n11, '10px ui-monospace, monospace');
    line('100% = parity):', T.n11, '10px ui-monospace, monospace');
    const pct = m.pctQofKV;
    line(pct < 1 ? pct.toFixed(3) + '%' : pct.toFixed(1) + '%', pct < 100 ? QC : KVC, '700 15px ui-monospace, monospace');
    ty += 6;
    if (m.cross != null) {
      line('crossing at', T.n11, '10px ui-monospace, monospace');
      line(`${fmtN(m.cross)} query tokens/device`, T.n13);
      ty += 2;
      line(m.qOp < m.cross ? 'you are LEFT of it → pass-Q' : 'you are RIGHT of it → pass-KV', T.n12, '10px ui-monospace, monospace');
    } else {
      line('no crossing: with persisted', T.n11, '10px ui-monospace, monospace');
      line('shards BOTH costs are linear', T.n11, '10px ui-monospace, monospace');
      line('in q, so the curves are', T.n11, '10px ui-monospace, monospace');
      line('parallel and one pattern', T.n11, '10px ui-monospace, monospace');
      line('wins at every q.', T.n11, '10px ui-monospace, monospace');
    }
    ty += 8;
    line('same attention output either', T.n10, '9px ui-monospace, monospace');
    line('way — only the traffic differs.', T.n10, '9px ui-monospace, monospace');

    geom = { px0, px1, py0, py1, qAt, X, Y, opX, crossX, crossY, devRects, ringY, boxH };

    // ------------------------------------------------------ hover-to-inspect
    if (page.pointer.over && !grab) {
      const p = page.pointer; let tip = null;
      for (const d of devRects) {
        if (p.x >= d.x && p.x <= d.x + d.w && p.y >= d.y && p.y <= d.y + d.h) {
          tip = movingIsKV
            ? `device ${d.i} — hop ${hop + 1}/${m.P}\nHOLDS (pinned): query block Q${d.i}, ${fmtN(m.qOp)} query token(s)\nHOLDS (in transit): K,V block ${d.held} — ${fmtN(m.shard)} tokens of cache\nSENDING to device ${(d.i + 1) % m.P}: K,V block ${d.held} = ${fmtB(m.passKV(m.qOp))}\nits queries never move; the cache does`
            : `device ${d.i} — hop ${hop + 1}/${m.P}\nHOLDS (pinned): K,V block ${d.i} — ${fmtN(m.shard)} tokens of cache\nHOLDS (in transit): query block Q${d.held} + its partial output + softmax stats\nSENDING to device ${(d.i + 1) % m.P}: query block ${d.held} = ${fmtB(m.passQ(m.qOp))}\nthe cache never moves; the queries do, and carry their unfinished answer`;
          break;
        }
      }
      if (!tip && p.x >= px0 - 6 && p.x <= px1 + 6 && p.y >= py0 - 6 && p.y <= py1 + 6) {
        const q = Math.max(1, Math.round(qAt(p.x)));
        const yq = Y(m.passQ(q)), ykv = Y(m.passKV(q));
        if (Math.abs(p.y - ykv) < Math.abs(p.y - yq) && Math.abs(p.y - ykv) < 34) {
          tip = m.persist
            ? `pass-KV @ q = ${fmtN(q)} query tokens\n  q × 2 (K and V) × ${EL} B × H_kv × d\n  = ${fmtN(q)} × 2 × ${EL} × ${m.Hkv} × ${D}\n  = ${fmtB(m.passKV(q))} per hop\nshards persist: only the NEW tokens' K,V move`
            : `pass-KV @ q = ${fmtN(q)} query tokens\n  shard × 2 (K and V) × ${EL} B × H_kv × d\n  = ${fmtN(m.shard)} × 2 × ${EL} × ${m.Hkv} × ${D}\n  = ${fmtB(m.passKV(q))} per hop\nFLAT in q — the resident shard moves whatever the query count`;
        } else if (Math.abs(p.y - yq) < 34) {
          tip = `pass-Q @ q = ${fmtN(q)} query tokens\n  q × H_q × ( ${EL}·d [Q] + ${ACC}·d [partial out] + ${STAT} [softmax m,l] )\n  = ${fmtN(q)} × ${m.Hq} × (${EL * D} + ${ACC * D} + ${STAT})\n  = ${fmtB(m.passQ(q))} per hop\nLINEAR in q — a moving query carries its unfinished answer`;
        } else if (crossX != null && Math.abs(p.x - crossX) < 20 && Math.abs(p.y - crossY) < 24) {
          tip = `crossing: q* = ${fmtN(m.cross)} query tokens/device\n  q* = (shard × 2 × ${EL} × H_kv × d) / (H_q × (${EL}·d + ${ACC}·d + ${STAT}))\n↔ drag this ✕ to re-solve it for a KV-head grouping\nfewer KV heads → smaller KV payload → the crossing moves LEFT`;
        } else {
          tip = `q = ${fmtN(q)} query tokens live per device\npass-KV ${fmtB(m.passKV(q))}   ·   pass-Q ${fmtB(m.passQ(q))}\n↔ drag to move the operating point`;
        }
      }
      if (tip) page.setTip(tip);
    }

    // ------------------------------------------------------------- readout
    const cheaper = m.winner, dearer = m.winner === 'pass-Q' ? 'pass-KV' : 'pass-Q';
    const lo = Math.min(m.bQ, m.bKV), hi = Math.max(m.bQ, m.bKV);
    let o = `phase=${st.phase}  q=${fmtN(m.qOp)} query tokens/device  ·  ${fmtN(m.shard)} tokens of context/device  ·  H_q=${m.Hq} H_kv=${m.Hkv}  ·  ${m.P} devices  ·  tier:${r.name}\n`;
    o += `WINNER: ${cheaper} — ${fmtB(lo)}/hop vs ${dearer} ${fmtB(hi)}/hop. pass-Q moves ${pctPhrase(m.pctQofKV)}.\n`;
    o += m.cross != null
      ? `Crossing at q* = ${fmtN(m.cross)} query tokens/device: left of it pass-Q is cheaper, right of it pass-KV is. Prefill puts q at the full shard (${fmtN(m.shard)}); decode puts it at ${fmtN(st.phase === 'decode' ? m.qOp : Math.max(1, st.batch | 0))} — opposite sides, which is why a real system picks per phase.\n`
      : `KV shards persist, so only the newly produced K,V moves: both costs are linear in q, the curves never cross, and ${cheaper} wins at every q. Persisting the shards is exactly how a system removes the phase flip.\n`;
    o += `Both patterns compute the SAME attention output — the maths is identical, only which operand crosses the wire changes. Grouped-query attention is what makes the KV payload small (${m.Hkv} KV heads vs ${m.Hq} query heads, ${(m.group % 1 ? m.group.toFixed(2) : m.group)}:1), which is why it moves this decision at all. Mechanism this sits on: the ring-attention page.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__cpPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Restore every control from the query string (the framework mirrors state
  // into the URL on change, so a copied link reproduces the whole view).
  for (const k of ['seqPow', 'devices', 'qheads', 'kvheads', 'batch']) {
    if (q.has(k)) page.controls.set(k, parseInt(q.get(k), 10), { rebuild: true, silent: true });
  }
  for (const k of ['phase', 'show']) if (q.has(k)) page.controls.set(k, q.get(k), { rebuild: true, silent: true });
  if (q.has('persist')) page.controls.set('persist', q.get('persist') === '1' || q.get('persist') === 'true', { rebuild: true, silent: true });
  // ?opq=N is the headless stand-in for dragging the operating point, and
  // ?crossq=N for dragging the ✕ (a screenshot has no pointer to drag with).
  if (q.has('opq')) setOpTo(page, Math.max(1, parseFloat(q.get('opq')) || 1));
  if (q.has('crossq')) setCrossTo(page, Math.max(1, parseFloat(q.get('crossq')) || 1));
  // ?hover=x,y fakes the cursor so the tooltip path is screenshot-verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: any of these hooks pauses the transport so
  // autoplay cannot advance off the requested hop before the snapshot.
  if (q.has('step') || q.has('hover') || q.has('opq') || q.has('crossq')) { if (t) t.pause(); }
  if (q.has('step') && t) { t.rebuildIfDirty(); t.seek(parseInt(q.get('step'), 10)); }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
