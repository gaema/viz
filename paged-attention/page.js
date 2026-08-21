// paged-attention concept page -- how a serving runtime stores the KV cache.
//
// The lesson, in three moves:
//   1. CONTIGUOUS. Each sequence reserves one slab the size of the LONGEST
//      answer it might produce. Everything past its current length is dead
//      memory, and the tail left over between slabs is too small to hold
//      another sequence -- fragmentation you can point at.
//   2. PAGED. The same sequences stored in fixed-size BLOCKS (a handful of
//      tokens each) plus a per-sequence BLOCK TABLE mapping logical position ->
//      physical block. Blocks are handed out from a free list, so they need
//      not be adjacent; only the LAST block of a sequence is partly empty.
//   3. SHARED. Two sequences that begin with the same prompt point their first
//      block-table entries at the SAME physical blocks. The prefix is stored
//      once; the block where they diverge is private (copy-on-write).
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: auto-plays + loops a decode. Each step appends one token to one
//    sequence, and a new block is allocated only when the current one fills.
//  - DIRECT MANIPULATION: drag the ◂▸ grip on the block-size bar to resize the
//    blocks, and drag a reservation boundary in the contiguous strip to resize
//    the per-sequence slab. Both counters recompute under your hand.
//  - HOVER: any cell reports which sequence and logical position it holds (or
//    "free" / "reserved but unused"); any block-table entry reports the physical
//    block it points at and whether that block is shared.
//  - URL hooks mirror the drag state: ?blocksize=8&seqs=3&reserve=16&share=1,
//    plus ?step=N / ?hover=x,y / ?play=1.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

// The memory pool, in token slots. Fixed so the two schemes are scored against
// the same denominator: "wasted cells / total cells" means the same thing in
// both strips. Every allowed block size divides it exactly.
const TOTAL = 96;
const ALLOWED_BS = [2, 3, 4, 6, 8];
// Prompt lengths per sequence. Sequences 0 and 1 are two answers to the SAME
// prompt (the sampling / beam case), which is what makes prefix sharing legal.
const PROMPTS = [9, 9, 5, 7];
const DECODE_STEPS = 16;

const snapBS = (v) => ALLOWED_BS.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), ALLOWED_BS[0]);
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const seqColor = (i) => categorical(i);
const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);

// Deterministic free list. Handing blocks out in address order would hide the
// point, so walk the pool with a stride coprime to every allowed block count
// (pool/bs is always 2^a·3^b) -- consecutive allocations land far apart, which
// is exactly what a real allocator's free list does after some churn.
function freeList(nb) {
  const out = [];
  for (let k = 0; k < nb; k++) out.push((k * 7) % nb);
  return out;
}

// Shared between compute(), draw() and onPointer(): the model parameters the
// snapshots were built with, so the hover tooltips can name real quantities.
let cur = { steps: null, bs: 4, n: 3, nb: 24, reserve: 16, sharedBlocks: 0 };
// Hit-test geometry, captured in draw().
let geom = null;
let drag = null;    // 'bs' | 'reserve' while a handle is grabbed

// Simulate prefill + decode and snapshot the allocation after every token.
// A snapshot is {lens, alloc, label}: lens[i] = tokens sequence i holds,
// alloc[i][L] = the PHYSICAL block backing sequence i's logical block L.
function buildModel(st) {
  const bs = snapBS(st.blocksize | 0);
  const n = clamp(st.seqs | 0, 2, 4);
  const nb = TOTAL / bs;
  const reserve = clamp(st.reserve | 0, 6, Math.floor(TOTAL / n));
  const order = freeList(nb);
  let next = 0;
  const take = () => (next < nb ? order[next++] : -1);

  const lens = [];
  const alloc = [];
  // Prefill sequence 0, then let sequence 1 REUSE its whole prompt blocks when
  // sharing is on. Only WHOLE blocks can be shared -- the block holding the
  // first token they could ever disagree on stays private to each sequence.
  const shareOn = !!st.share && n >= 2;
  const sharedBlocks = shareOn ? Math.floor(Math.min(PROMPTS[0], PROMPTS[1]) / bs) : 0;
  for (let i = 0; i < n; i++) {
    lens[i] = PROMPTS[i];
    alloc[i] = [];
    const need = Math.ceil(lens[i] / bs);
    for (let L = 0; L < need; L++) {
      alloc[i][L] = (i === 1 && L < sharedBlocks) ? alloc[0][L] : take();
    }
  }

  const snap = (label) => ({ lens: lens.slice(), alloc: alloc.map((a) => a.slice()), label });
  const steps = [snap(`prompts loaded — ${n} sequences, ${lens.reduce((a, b) => a + b, 0)} tokens${sharedBlocks ? `; S0 and S1 share ${sharedBlocks} prefix block${sharedBlocks > 1 ? 's' : ''}` : ''}`)];
  for (let s = 0; s < DECODE_STEPS; s++) {
    const i = s % n;
    lens[i]++;
    const need = Math.ceil(lens[i] / bs);
    let grew = false;
    while (alloc[i].length < need) { alloc[i].push(take()); grew = true; }
    steps.push(snap(grew
      ? `decode step ${s + 1}: S${i} token ${lens[i] - 1} filled its block — allocate physical block ${alloc[i][alloc[i].length - 1]}`
      : `decode step ${s + 1}: S${i} token ${lens[i] - 1} lands in block ${alloc[i][need - 1]} slot ${(lens[i] - 1) % bs} — no allocation`));
  }
  cur = { steps, bs, n, nb, reserve, sharedBlocks };
  return steps;
}

// owners[p] = [{seq, L}, ...] for physical block p under this snapshot.
function ownersOf(alloc, nb) {
  const owners = Array.from({ length: nb }, () => []);
  for (let i = 0; i < alloc.length; i++) {
    for (let L = 0; L < alloc[i].length; L++) {
      const p = alloc[i][L];
      if (p >= 0 && p < nb) owners[p].push({ seq: i, L });
    }
  }
  return owners;
}

// Diagonal hatch inside a rect -- the page's one visual code for "committed
// memory holding no token".
function hatch(ctx, x, y, w, h, color) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let d = -h; d < w + h; d += 4) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke(); }
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'paged-attention — a KV cache in blocks, not slabs',
  blurb: 'A serving runtime cannot give every sequence a contiguous slab sized for its longest possible answer: the unused tail is dead memory and the leftovers fragment. Paging stores the same keys and values in fixed-size blocks, with a per-sequence block table mapping logical position → physical block, so blocks need not be adjacent and only the last one is partly empty — and two sequences sharing a prompt prefix can point at the SAME blocks. It auto-plays a decode; drag the ◂▸ block-size grip or a reservation boundary, and watch the wasted-cell counters move.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  compare: { key: 'blocksize', a: 8, b: 2, labelA: 'block = 8 tokens (few, coarse blocks)', labelB: 'block = 2 tokens (many, fine blocks)', rebuild: true },
  challenges: [
    { goal: 'Get paged waste down to 3 cells or fewer.', hint: 'drag the ◂▸ block-size grip to the left — a smaller block leaves a smaller tail in each sequence.', check: (api) => ({ solved: (api.probe.pagedWaste ?? 99) <= 3, detail: `paged waste = ${api.probe.pagedWaste ?? '–'} cells (need ≤ 3)` }) },
    { goal: 'Store a prompt prefix once: get 2 or more shared blocks.', hint: 'turn "share prompt prefix" on; a bigger block holds more tokens but fewer WHOLE blocks can be shared.', check: (api) => ({ solved: (api.probe.sharedBlocks ?? 0) >= 2, detail: `${api.probe.sharedBlocks ?? 0} shared block(s)` }) },
  ],
  controls: (c, page) => {
    c.slider('blocksize', { label: 'block size (tokens)', min: 2, max: 8, step: 1, value: 4, rebuild: true, format: (v) => String(snapBS(v)) });
    c.stepper('seqs', { label: 'sequences', min: 2, max: 4, value: 3, rebuild: true });
    c.slider('reserve', { label: 'contiguous reservation', min: 6, max: 32, step: 1, value: 26, rebuild: true });
    c.toggle('share', { label: 'share prompt prefix', value: true, rebuild: true });
    c.transport({ compute: () => buildModel(page.state), speed: 2.5, loop: true });
  },

  // Direct manipulation: the two operands that decide the whole picture are the
  // BLOCK SIZE and the per-sequence RESERVATION, so both are grabbable on the
  // canvas. Dragging rebuilds the model, so the allocation and both counters
  // recompute under the cursor.
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const onBS = (x, y) => x >= g.bsBar.x - 12 && x <= g.bsBar.x + g.bsBar.w + 12 && y >= g.bsBar.y - 10 && y <= g.bsBar.y + g.bsBar.h + 10;
    const onRes = (x, y) => y >= g.aY - 16 && y <= g.aY + g.aH && g.resHandles.some((hx) => Math.abs(x - hx) <= 8);
    const setBS = (x) => {
      const frac = clamp((x - g.bsBar.x) / g.bsBar.w, 0, 1);
      const v = ALLOWED_BS[Math.round(frac * (ALLOWED_BS.length - 1))];
      if (v !== cur.bs) page.controls.set('blocksize', v, { rebuild: true });
    };
    const setRes = (x) => {
      const cells = (x - g.stripX) / g.cwA;
      const v = clamp(Math.round(cells / Math.max(1, drag.idx + 1)), 6, Math.floor(TOTAL / cur.n));
      if (v !== cur.reserve) page.controls.set('reserve', v, { rebuild: true });
    };
    if (ev.type === 'down') {
      if (onBS(ev.x, ev.y)) { drag = { kind: 'bs' }; setBS(ev.x); }
      else if (onRes(ev.x, ev.y)) {
        let idx = 0, best = Infinity;
        g.resHandles.forEach((hx, i) => { const d = Math.abs(ev.x - hx); if (d < best) { best = d; idx = i; } });
        drag = { kind: 'reserve', idx };
        setRes(ev.x);
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      if (drag.kind === 'bs') setBS(ev.x); else setRes(ev.x);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    if (!cur.steps) return;
    r.clear(T.n0);
    const s = page.step() || cur.steps[0];
    const { lens, alloc } = s;
    const { bs, n, nb, reserve, sharedBlocks } = cur;
    const owners = ownersOf(alloc, nb);

    // ---- geometry (anchored to the canvas height so it fills any aspect) ----
    const pad = 16, gutter = 108, H = page.H;
    const stripX = pad + gutter, stripW = Math.max(120, page.W - stripX - pad);
    const cwA = stripW / TOTAL;
    const aY = Math.round(H * 0.14), aH = Math.round(H * 0.075);   // contiguous strip
    const bY = Math.round(H * 0.40), bH = Math.round(H * 0.08);    // paged strip
    const gap = Math.min(3, Math.max(1, cwA * bs * 0.12));
    const blockW = (stripW - (nb - 1) * gap) / nb, cwB = blockW / bs;
    const bsBar = { x: stripX, y: Math.round(H * 0.60), w: Math.min(240, stripW * 0.45), h: 12 };
    const tabY = Math.round(H * 0.70), rowH = Math.min(32, Math.max(16, (H - tabY - 10) / n));

    // ---- scheme 1: one contiguous slab per sequence ------------------------
    r.label('contiguous — one max-length slab per sequence', pad, aY - 26, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label('slab memory →', pad, aY + aH / 2 + 4, { color: T.n11, font: '11px ui-monospace, monospace' });

    // Per-cell ownership under the slab scheme.
    const cellA = new Array(TOTAL).fill(null);
    let contigWaste = 0, contigLive = 0, overflow = 0;
    for (let i = 0; i < n; i++) {
      const start = i * reserve;
      const held = Math.min(lens[i], reserve);
      overflow += Math.max(0, lens[i] - reserve);
      contigLive += held;
      contigWaste += reserve - held;
      for (let k = 0; k < reserve; k++) cellA[start + k] = { seq: i, pos: k, live: k < held };
    }
    const tail = TOTAL - n * reserve;
    // A leftover smaller than one reservation cannot host another sequence: it
    // is fragmentation, not free memory.
    const tailWaste = tail > 0 && tail < reserve ? tail : 0;
    contigWaste += tailWaste;
    for (let k = n * reserve; k < TOTAL; k++) cellA[k] = { frag: tailWaste > 0 };

    ctx.save();
    for (let k = 0; k < TOTAL; k++) {
      const x = stripX + k * cwA, c = cellA[k];
      ctx.fillStyle = c && c.seq != null ? (c.live ? alphaOf(seqColor(c.seq), 0.9) : alphaOf(seqColor(c.seq), 0.12)) : T.n1;
      ctx.fillRect(x, aY, cwA, aH);
      if (c && c.seq != null && !c.live) hatch(ctx, x, aY, cwA, aH, alphaOf(T.bad, 0.5));
      if (c && c.frag) hatch(ctx, x, aY, cwA, aH, alphaOf(T.bad, 0.35));
    }
    ctx.strokeStyle = rgbaToken('n14', 0.18); ctx.lineWidth = 1;
    ctx.strokeRect(stripX, aY, stripW, aH);
    ctx.restore();

    // Reservation boundaries, each a drag handle for the reservation size.
    const resHandles = [];
    for (let i = 0; i < n; i++) {
      const x0 = stripX + i * reserve * cwA, x1 = stripX + (i + 1) * reserve * cwA;
      resHandles.push(x1);
      ctx.save();
      ctx.strokeStyle = alphaOf(seqColor(i), 0.95); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x0 + 1, aY - 8); ctx.lineTo(x0 + 1, aY + aH + 4); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1 - 1, aY - 8); ctx.lineTo(x1 - 1, aY + aH + 4); ctx.stroke();
      ctx.fillStyle = drag && drag.kind === 'reserve' ? T.warn : alphaOf(seqColor(i), 0.95);
      ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText('◂▸', x1 - 1, aY - 12);
      ctx.fillStyle = alphaOf(seqColor(i), 0.95); ctx.textAlign = 'left';
      ctx.fillText(`S${i} ${lens[i]}/${reserve}`, x0 + 3, aY + aH + 14);
      ctx.restore();
    }

    // ---- scheme 2: fixed-size blocks + block tables -------------------------
    r.label(`paged — ${nb} blocks of ${bs} tokens, allocated on demand`, pad, bY - 26, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label('blocks →', pad, bY + bH / 2 + 4, { color: T.n11, font: '11px ui-monospace, monospace' });

    const blockX = (p) => stripX + p * (blockW + gap);
    const tokensIn = (p) => {
      const o = owners[p];
      if (!o.length) return 0;
      if (o.length > 1) return bs;                       // a shared prefix block is full by construction
      return clamp(lens[o[0].seq] - o[0].L * bs, 0, bs);
    };
    let pagedWaste = 0, pagedLive = 0, usedBlocks = 0, sharedCount = 0;
    ctx.save();
    for (let p = 0; p < nb; p++) {
      const x = blockX(p), o = owners[p], tk = tokensIn(p);
      if (o.length) { usedBlocks++; pagedLive += tk; pagedWaste += bs - tk; }
      if (o.length > 1) sharedCount++;
      for (let k = 0; k < bs; k++) {
        const cx = x + k * cwB;
        ctx.fillStyle = o.length ? (k < tk ? alphaOf(seqColor(o[0].seq), 0.9) : alphaOf(seqColor(o[0].seq), 0.12)) : T.n1;
        ctx.fillRect(cx, bY, cwB, bH);
        if (o.length && k >= tk) hatch(ctx, cx, bY, cwB, bH, alphaOf(T.bad, 0.5));
      }
      ctx.strokeStyle = o.length > 1 ? T.violet : (o.length ? alphaOf(seqColor(o[0].seq), 0.9) : rgbaToken('n14', 0.16));
      ctx.lineWidth = o.length > 1 ? 2 : 1;
      ctx.strokeRect(x, bY, blockW, bH);
      if (o.length > 1) {
        ctx.fillStyle = T.violet; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
        ctx.fillText('⇄', x + blockW / 2, bY - 4);
      }
      if (blockW >= 20) { ctx.fillStyle = T.n9; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.fillText(String(p), x + 2, bY + bH + 9); }
    }
    ctx.restore();
    const savedCells = sharedCount * bs;

    // Block tables: logical position -> physical block, one row per sequence.
    const boxW = Math.min(34, Math.max(20, (stripW - 60) / Math.max(4, Math.max(...alloc.map((a) => a.length)))));
    const tabX = stripX;
    const entryRects = [];
    for (let i = 0; i < n; i++) {
      const y = tabY + i * rowH;
      r.label(`S${i} block table`, pad, y + rowH * 0.62, { color: alphaOf(seqColor(i), 0.95), font: '11px ui-monospace, monospace' });
      for (let L = 0; L < alloc[i].length; L++) {
        const p = alloc[i][L], shared = owners[p] && owners[p].length > 1;
        const x = tabX + L * (boxW + 4), h = rowH - 6;
        ctx.save();
        ctx.fillStyle = alphaOf(seqColor(i), shared ? 0.32 : 0.8);
        ctx.fillRect(x, y, boxW, h);
        ctx.strokeStyle = shared ? T.violet : alphaOf(seqColor(i), 0.9); ctx.lineWidth = shared ? 2 : 1;
        ctx.strokeRect(x, y, boxW, h);
        ctx.fillStyle = shared ? T.violet : inkOn(seqColor(i));
        ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(p), x + boxW / 2, y + h / 2);
        ctx.restore();
        entryRects.push({ x, y, w: boxW, h, seq: i, L, p, shared });
        // Arrow into the physical block. Shared entries are drawn strongly --
        // two tables converging on one block is the whole payoff.
        r.arrow({ x: x + boxW / 2, y }, { x: blockX(p) + blockW / 2, y: bY + bH + 12 },
          { color: shared ? T.violet : alphaOf(seqColor(i), 0.55), width: shared ? 1.6 : 1, head: 5, alpha: shared ? 0.9 : 0.32 });
      }
    }

    // Counter text and the block-size bar are painted LAST, each on a plate of
    // page ground, so the block-table arrow fan passes behind them and never
    // strikes through a number.
    const fnt = '11px ui-monospace, monospace';
    const plate = (x, y, w, h) => { ctx.save(); ctx.fillStyle = T.n0; ctx.fillRect(x, y, w, h); ctx.restore(); };
    // Plate exactly the width of the text, never the whole strip -- the arrow
    // fan stays readable everywhere the text is not.
    const plated = (text, x, y, color, font) => {
      ctx.save(); ctx.font = font; const w = ctx.measureText(text).width; ctx.restore();
      plate(x - 3, y - 11, w + 6, 15);
      r.label(text, x, y, { color, font });
    };
    plated('block tables — logical block → physical block', pad, tabY - 12, T.n14, '13px ui-monospace, monospace');
    plate(bsBar.x - 8, bsBar.y - 8, bsBar.w + 16, bsBar.h + 16);

    // Block-size bar: the second grabbable operand.
    ctx.save();
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(bsBar.x, bsBar.y, bsBar.w, bsBar.h);
    const bsFrac = ALLOWED_BS.indexOf(bs) / (ALLOWED_BS.length - 1);
    ctx.fillStyle = alphaOf(T.accent, 0.45); ctx.fillRect(bsBar.x, bsBar.y, bsBar.w * bsFrac, bsBar.h);
    const hx = bsBar.x + bsBar.w * bsFrac;
    ctx.fillStyle = drag && drag.kind === 'bs' ? T.warn : T.accent;
    ctx.fillRect(hx - 4, bsBar.y - 5, 8, bsBar.h + 10);
    ctx.restore();
    plated(`◂▸ block size = ${bs} tokens (drag)`, bsBar.x + bsBar.w + 10, bsBar.y + bsBar.h, T.accent, fnt);

    plated(`wasted ${contigWaste}/${TOTAL} cells (${pct(contigWaste, TOTAL)}%) = ${contigWaste - tailWaste} reserved-but-unused + ${tailWaste} fragmented tail · live ${contigLive}`,
      stripX, aY + aH + 32, T.n12, fnt);
    if (overflow > 0) plated(`⚠ ${overflow} token(s) past the reservation — a full slab has nowhere to grow`, stripX, aY + aH + 46, T.bad, fnt);
    plated(`wasted ${pagedWaste}/${TOTAL} cells (${pct(pagedWaste, TOTAL)}%) — only the last block of each sequence is partial`, stripX, bY + bH + 24, T.n12, fnt);
    plated(`live ${pagedLive} · ${usedBlocks} blocks in use, ${nb - usedBlocks} free · ${sharedCount} shared (${savedCells} cells stored once)`, stripX, bY + bH + 38, T.n12, fnt);

    geom = { stripX, stripW, cwA, aY, aH, bY, bH, blockW, cwB, gap, bsBar, resHandles, entryRects, blockX, owners, tokensIn, lens, cellA };
    page.probe = { pagedWaste, contigWaste, sharedBlocks: sharedCount, bs };

    // ---- hover-to-inspect --------------------------------------------------
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      let tip = null;
      if (p.y >= aY && p.y <= aY + aH && p.x >= stripX && p.x <= stripX + stripW) {
        const k = clamp(Math.floor((p.x - stripX) / cwA), 0, TOTAL - 1), c = cellA[k];
        if (c && c.seq != null) {
          tip = c.live
            ? `slab cell ${k} — S${c.seq} logical position ${c.pos}\nreservation starts at cell ${c.seq * reserve}, length ${reserve}`
            : `slab cell ${k} — reserved by S${c.seq}, EMPTY\nS${c.seq} holds ${lens[c.seq]} of ${reserve} reserved cells; the rest is dead memory`;
        } else {
          tip = tailWaste
            ? `slab cell ${k} — leftover tail (${tailWaste} cells)\ntoo small for another ${reserve}-cell reservation: fragmentation`
            : `slab cell ${k} — free (${tail} cells left, room for another reservation)`;
        }
      } else if (p.y >= bY && p.y <= bY + bH && p.x >= stripX) {
        const idx = Math.floor((p.x - stripX) / (blockW + gap));
        if (idx >= 0 && idx < nb && p.x - blockX(idx) <= blockW) {
          const o = owners[idx], tk = tokensIn(idx);
          const slot = clamp(Math.floor((p.x - blockX(idx)) / cwB), 0, bs - 1);
          if (!o.length) tip = `physical block ${idx} — FREE\nany sequence can take it on its next allocation`;
          else if (o.length > 1) tip = `physical block ${idx} — SHARED by ${o.map((e) => `S${e.seq}`).join(', ')}\nslot ${slot} = logical position ${o[0].L * bs + slot} of each\nstored once; a write here would copy the block first`;
          else tip = slot < tk
            ? `physical block ${idx} — S${o[0].seq} logical block ${o[0].L}\nslot ${slot} = logical position ${o[0].L * bs + slot}`
            : `physical block ${idx} — S${o[0].seq} logical block ${o[0].L}, slot ${slot} EMPTY\nthe only partly-filled block a sequence has: ${bs - tk} cell(s) of slack`;
        }
      } else {
        for (const e of entryRects) {
          if (p.x >= e.x && p.x <= e.x + e.w && p.y >= e.y && p.y <= e.y + e.h) {
            const o = owners[e.p];
            tip = `S${e.seq} block table[${e.L}] → physical block ${e.p}\nlogical positions ${e.L * bs}..${e.L * bs + bs - 1}\n${e.shared ? `SHARED with ${o.filter((q) => q.seq !== e.seq).map((q) => `S${q.seq}`).join(', ')} — one copy in memory` : 'private to this sequence'}`;
            break;
          }
        }
        if (!tip && p.x >= bsBar.x - 12 && p.x <= bsBar.x + bsBar.w + 12 && p.y >= bsBar.y - 10 && p.y <= bsBar.y + bsBar.h + 10) {
          tip = `block size = ${bs} tokens → ${nb} blocks in the pool\n◂▸ drag: smaller blocks waste less tail, at more block-table entries`;
        }
      }
      if (tip) page.setTip(tip);
    }

    let o = `${s.label}    tier:${r.name}\n`;
    o += `contiguous: ${contigWaste}/${TOTAL} cells wasted (${pct(contigWaste, TOTAL)}%)   ·   paged: ${pagedWaste}/${TOTAL} wasted (${pct(pagedWaste, TOTAL)}%) in ${usedBlocks} blocks of ${bs}\n`;
    o += sharedCount
      ? `${sharedCount} block(s) carry a prompt prefix that ${cur.sharedBlocks ? 'S0 and S1' : 'two sequences'} both point at — ${savedCells} cells stored once instead of twice. Diverging tokens land in a private block (copy-on-write).`
      : 'No blocks are shared right now — turn on "share prompt prefix" (or grow the block count) to store a common prompt once for both sequences.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__pagedPage = page;
  const q = new URLSearchParams(location.search);
  // URL hooks mirroring the drag state, so a manipulated view is reproducible
  // without a pointer: ?blocksize=8&seqs=3&reserve=16&share=0
  if (q.has('blocksize')) page.controls.set('blocksize', snapBS(parseInt(q.get('blocksize'), 10) || 4), { rebuild: true });
  if (q.has('seqs')) page.controls.set('seqs', clamp(parseInt(q.get('seqs'), 10) || 3, 2, 4), { rebuild: true });
  if (q.has('reserve')) page.controls.set('reserve', clamp(parseInt(q.get('reserve'), 10) || 16, 6, 48), { rebuild: true });
  if (q.has('share')) page.controls.set('share', q.get('share') !== '0', { rebuild: true });
  const t = page.controls._transport;
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover) so the
  // tooltip path is verifiable. Canvas-space px.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause for any of these hooks.
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
