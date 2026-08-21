// hybrid-cache-allocator concept page -- ONE serving memory pool, TWO memory
// shapes, and the bug class that lives between them.
//
// A hybrid model interleaves attention layers with recurrent / linear-attention
// layers. Their per-sequence memory could not be more different:
//
//   ATTENTION layer  -> a KV cache that GROWS one slot per token, without bound.
//                       Naturally paged: hand out a fixed-size page whenever the
//                       current one fills. Cost is proportional to CONTEXT.
//   RECURRENT layer  -> a FIXED-SIZE state matrix per sequence. Allocated once at
//                       admission, never grows, freed at completion. Cost is
//                       proportional to CONCURRENCY, and is paid UP FRONT even by
//                       a sequence that only ever emits ten tokens.
//
// A serving runtime holds both in one pool, under one capacity line. Split that
// pool and the two shapes fail in OPPOSITE directions:
//
//   (a) size it for the state slabs -> long sequences exhaust the paged side
//       MID-GENERATION and get preempted.
//   (b) size it for the pages       -> you cannot ADMIT more sequences, because
//       each new one needs its fixed slab before it emits a single token, and the
//       page pool sits half empty while sequences wait.
//
// So maximum CONCURRENCY is set by the fixed part and maximum CONTEXT by the
// growing part. One knob cannot satisfy both -- and when demand is high enough,
// NO position of the knob satisfies both, which the feasible band on the ceiling
// chart shows as an empty strip.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: auto-plays + loops an admission phase followed by generation
//    rounds. Each step admits, allocates, completes or preempts, and says which.
//  - DIRECT MANIPULATION: drag the grip on the capacity bar to move the
//    split; drag the two demand bars (sequences, target context); drag any
//    individual sequence's row to change that one sequence's target length.
//  - HOVER: any page -> owning sequence, its logical page range, size, and that
//    it belongs to the attention layers. Any slab -> owning sequence, size, and
//    that it is the recurrent layers' fixed state. Any sequence row -> target,
//    emitted, status and the reason.
//  - URL hooks mirror every drag: ?split=25&seqs=6&ctx=4096&page=128&attnevery=4
//    &lens=512,4096,... plus ?step=N / ?hover=x,y / ?play=1.
//
// The sibling pages own the neighbouring mechanisms and are not re-taught here:
// paged-attention owns block-table paging for a pure-attention model, and
// hybrid-by-layer owns which layers are which.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

// ---------------------------------------------------------------------------
// Model shape. Plain, published-arithmetic constants -- a mid-size hybrid whose
// numbers are easy to hold in your head, not a measurement of any product.
// ---------------------------------------------------------------------------
const LAYERS = 24;
// 2 (K and V) x 16 kv-heads x 128 head_dim x 2 bytes = 8 KiB per token, per
// attention layer. This is the quantity the paged side scales with.
const KV_KIB_PER_TOKEN_PER_ATTN_LAYER = 8;
// 32 heads x 128 head_dim x 128 state_dim x 2 bytes = 1 MiB per recurrent layer,
// per sequence. Independent of how many tokens that sequence has seen.
const STATE_MIB_PER_RECURRENT_LAYER = 1;
const POOL_MIB = 384;
const MAX_ROUNDS = 40;
const ATTN_EVERY = [2, 4, 8];        // 1 attention layer every k layers
const PAGE_TOKENS = [64, 128, 256];
const MAX_SEQS = 12;
const CTX_MIN = 256, CTX_MAX = 8192;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const snapTo = (v, list) => list.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), list[0]);
const seqColor = (i) => categorical(i);
const mib = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' MiB';
// Exact tokens unless the value is a whole multiple of 1024 -- a ceiling
// rounded to "1.3k" is a ceiling the reader cannot check against the page.
const tok = (v) => (v >= 1024 && v % 1024 === 0 ? v / 1024 + 'k' : String(v));

// Shared between compute(), draw() and onPointer().
let cur = null;
let geom = null;
let drag = null;

// ---------------------------------------------------------------------------
// Derived pool shape for the current control state.
// ---------------------------------------------------------------------------
function shapeOf(st) {
  const attnEvery = snapTo(st.attnevery | 0, ATTN_EVERY);
  const nAttn = LAYERS / attnEvery;
  const nRec = LAYERS - nAttn;
  const pageTokens = snapTo(st.page | 0, PAGE_TOKENS);
  const pageMiB = (pageTokens * nAttn * KV_KIB_PER_TOKEN_PER_ATTN_LAYER) / 1024;
  const slabMiB = nRec * STATE_MIB_PER_RECURRENT_LAYER;
  const split = clamp(Math.round(st.split), 0, 100);
  const pageBudget = (POOL_MIB * split) / 100;
  const slabBudget = POOL_MIB - pageBudget;
  const nPages = Math.floor(pageBudget / pageMiB);
  const nSlabs = Math.floor(slabBudget / slabMiB);
  // What the split could not spend on either shape: a page needs pageMiB and a
  // slab needs slabMiB, so the remainder is stranded whichever way you lean.
  const residueMiB = POOL_MIB - nPages * pageMiB - nSlabs * slabMiB;
  return { attnEvery, nAttn, nRec, pageTokens, pageMiB, slabMiB, split, nPages, nSlabs, residueMiB };
}

// Per-sequence target lengths: the `lens` control is a comma list; blanks and
// missing entries fall back to the single "target context" slider.
function targetsOf(st, pageTokens) {
  const n = clamp(st.seqs | 0, 1, MAX_SEQS);
  const base = clamp(st.ctx | 0, CTX_MIN, CTX_MAX);
  const raw = String(st.lens || '').split(',').map((s) => parseInt(s.trim(), 10));
  const out = [];
  for (let i = 0; i < n; i++) {
    const v = Number.isFinite(raw[i]) ? raw[i] : base;
    out.push(clamp(Math.round(v / pageTokens) * pageTokens, pageTokens, CTX_MAX));
  }
  return out;
}

// The ceilings, as pure functions of the split -- used both for the live readout
// and to sweep the feasible band on the chart.
const pagesAt = (sh, split) => Math.floor((POOL_MIB * split) / 100 / sh.pageMiB);
const slabsAt = (sh, split) => Math.floor((POOL_MIB * (100 - split)) / 100 / sh.slabMiB);
const ctxCeil = (sh, nPages, n) => sh.pageTokens * Math.floor(nPages / Math.max(1, n));

// ---------------------------------------------------------------------------
// Simulation: admission, then generation rounds. Snapshot after every event so
// the transport can walk it. Everything the page reports is computed here.
// ---------------------------------------------------------------------------
function buildModel(st) {
  const sh = shapeOf(st);
  const targets = targetsOf(st, sh.pageTokens);
  const pageOwner = new Array(sh.nPages).fill(-1);
  const slabOwner = new Array(sh.nSlabs).fill(-1);
  // 'queued' = the scheduler has not looked at it yet; 'waiting' = it was looked
  // at and REFUSED. Keeping them apart is what makes the admission-side failure
  // countable instead of blurring into "not started".
  const seqs = targets.map((t, i) => ({ i, target: t, emitted: 0, pages: [], slab: -1, status: 'queued', why: 'queued — the scheduler has not reached it yet' }));
  const steps = [];
  const snap = (label, focus) => steps.push({
    label, focus,
    pageOwner: pageOwner.slice(),
    slabOwner: slabOwner.slice(),
    seqs: seqs.map((s) => ({ ...s, pages: s.pages.slice() })),
  });

  const freeSlab = () => slabOwner.indexOf(-1);
  const freePage = () => pageOwner.indexOf(-1);

  // Admission charges the WHOLE fixed slab up front, plus one page for the
  // prompt. The slab is the gate: a sequence that will emit ten tokens pays the
  // same state as one that will emit eight thousand.
  const admit = (s) => {
    const sl = freeSlab();
    if (sl < 0) {
      s.status = 'waiting';
      s.why = `no free state slab — all ${sh.nSlabs} are held. The fixed part of the pool sets the concurrency ceiling, and it is reached before a single extra token is emitted.`;
      return null;
    }
    const pg = freePage();
    if (pg < 0) { s.status = 'waiting'; s.why = 'no free KV page for the prompt — the paged part is full.'; return null; }
    slabOwner[sl] = s.i; s.slab = sl;
    pageOwner[pg] = s.i; s.pages = [pg];
    s.emitted = Math.min(s.target, sh.pageTokens);
    s.status = 'live'; s.why = 'admitted — slab allocated once, pages allocated as it grows';
    return `S${s.i} admitted: state slab ${sl} (${mib(sh.slabMiB)}, fixed forever) + KV page ${pg} (${mib(sh.pageMiB)}, first of many)`;
  };

  snap(`empty pool — ${POOL_MIB} MiB split ${sh.split}/${100 - sh.split} into ${sh.nPages} KV pages of ${sh.pageTokens} tok and ${sh.nSlabs} state slabs of ${mib(sh.slabMiB)}`, -1);

  for (let i = 0; i < seqs.length; i++) {
    const msg = admit(seqs[i]);
    snap(msg || `S${i} REFUSED at admission — ${seqs[i].why.split('.')[0]}`, i);
  }

  let preempted = 0, finished = 0, peakLive = 0;
  for (let r = 0; r < MAX_ROUNDS; r++) {
    const notes = [];
    // A freed slab lets a waiting sequence in: the waiting queue is retried
    // every round, which is why the fixed part is a THROUGHPUT limit and not
    // only a hard refusal.
    for (const s of seqs) if (s.status === 'waiting' || s.status === 'queued') { const m = admit(s); if (m) notes.push(m); }
    let progressed = false;
    for (const s of seqs) {
      if (s.status !== 'live') continue;
      const next = Math.min(s.target, s.emitted + sh.pageTokens);
      if (next <= s.emitted) continue;
      const need = Math.ceil(next / sh.pageTokens);
      let ok = true;
      while (s.pages.length < need) {
        const p = freePage();
        if (p < 0) { ok = false; break; }
        pageOwner[p] = s.i; s.pages.push(p);
      }
      if (!ok) {
        for (const p of s.pages) pageOwner[p] = -1;
        if (s.slab >= 0) slabOwner[s.slab] = -1;
        s.pages = []; s.slab = -1; s.status = 'preempted';
        s.why = `KV page pool exhausted at ${s.emitted} of ${s.target} tok — every one of the ${sh.nPages} pages was taken, so this sequence was preempted MID-GENERATION. Its fixed slab was fine; the growing side ran out.`;
        notes.push(`S${s.i} PREEMPTED at ${s.emitted}/${s.target} tok — no free KV page`);
        preempted++; progressed = true; continue;
      }
      s.emitted = next; progressed = true;
      if (s.emitted >= s.target) {
        s.status = 'done'; s.why = `reached its ${s.target} tok target; slab and pages returned to the pool`;
        for (const p of s.pages) pageOwner[p] = -1;
        if (s.slab >= 0) slabOwner[s.slab] = -1;
        s.pages = []; s.slab = -1;
        notes.push(`S${s.i} finished at ${s.target} tok — ${mib(sh.slabMiB)} slab + its pages returned`);
        finished++;
      }
    }
    const live = seqs.filter((s) => s.status === 'live').length;
    peakLive = Math.max(peakLive, live);
    if (!progressed && !live) break;
    snap(`round ${r + 1}: ${live} live, +${sh.pageTokens} tok each${notes.length ? ' — ' + notes.join(' · ') : ''}`, -1);
    if (!seqs.some((s) => s.status === 'live' || s.status === 'waiting' || s.status === 'queued')) break;
  }

  cur = { sh, targets, steps, preempted, finished, peakLive };
  return steps;
}

// Diagonal hatch: this page's one visual code for "capacity committed, nothing
// in it". Same code on both pools so the two are directly comparable.
function hatch(ctx, x, y, w, h, color) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let d = -h; d < w + h; d += 4) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke(); }
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'hybrid-cache-allocator — two memory shapes, one pool',
  blurb: 'A hybrid model interleaves attention layers with recurrent (linear-attention) layers, and their memory could not be more different. An attention layer’s KV cache GROWS one slot per token, without bound, so it is stored in pages handed out on demand — its cost scales with CONTEXT. A recurrent layer’s state is a FIXED-SIZE matrix per sequence, allocated once at admission and never growing — its cost scales with CONCURRENCY, and it is charged up front even for a sequence that emits ten tokens. A serving runtime holds both under one capacity line, and the two shapes fail in opposite directions: size the pool for the slabs and long sequences exhaust the paged side mid-generation; size it for the pages and you cannot admit more sequences while the page pool sits half empty. Drag the ⇔ grip on the capacity bar and watch the two ceilings move against each other — and against the green band of splits that satisfy BOTH, which for a demanding workload is empty.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  compare: {
    key: 'split', a: 22, b: 92, rebuild: true,
    labelA: 'split 22% pages — sized for the state slabs: sequences admit, then starve for pages mid-generation',
    labelB: 'split 92% pages — sized for the KV pages: pages go spare while sequences wait for a state slab',
  },
  challenges: [
    {
      goal: 'Starve the GROWING side: get at least one sequence preempted mid-generation.',
      hint: 'drag the ⇔ grip left (fewer pages) and the target-context bar right — admission still succeeds, the pages run out later.',
      check: (api) => ({ solved: (api.probe.preempted ?? 0) >= 1, detail: `${api.probe.preempted ?? 0} preempted (need ≥ 1)` }),
    },
    {
      goal: 'Starve the FIXED side: leave 3+ sequences waiting while the page pool is over half free.',
      hint: 'drag the ⇔ grip right (few slabs), raise the sequence count, and keep the target context short.',
      check: (api) => ({
        solved: (api.probe.waiting ?? 0) >= 3 && (api.probe.freePageFrac ?? 0) > 0.5,
        detail: `${api.probe.waiting ?? 0} waiting, page pool ${Math.round(100 * (api.probe.freePageFrac ?? 0))}% free (need ≥ 3 and > 50%)`,
      }),
    },
    {
      goal: 'Find a split that satisfies BOTH ceilings at once (the green band).',
      hint: 'if the band is empty, no split can — lower the sequence count or the target context until one appears, then drag into it.',
      check: (api) => ({ solved: !!api.probe.inBand, detail: api.probe.bandText || '—' }),
    },
  ],

  controls: (c, page) => {
    c.slider('split', { label: 'pool split → KV pages (%)', min: 0, max: 100, step: 1, value: 60, rebuild: true });
    c.stepper('seqs', { label: 'sequences requested', min: 1, max: MAX_SEQS, value: 6, rebuild: true });
    c.slider('ctx', { label: 'target context (tok)', min: CTX_MIN, max: CTX_MAX, step: 128, value: 2048, rebuild: true });
    c.slider('page', { label: 'KV page (tokens)', min: 64, max: 256, step: 1, value: 128, rebuild: true, format: (v) => String(snapTo(v, PAGE_TOKENS)) });
    c.slider('attnevery', { label: 'attention every k layers', min: 2, max: 8, step: 1, value: 4, rebuild: true, // Just k: the value field is a few characters wide, and the resulting
// attention/recurrent counts are already spelled out on both pool headers.
format: (v) => String(snapTo(v, ATTN_EVERY)) });
    c.text('lens', { label: 'per-seq targets (csv)', value: '', placeholder: 'blank = all at target', rebuild: true });
    c.transport({ compute: () => buildModel(page.state), speed: 3, loop: true });
  },

  // Direct manipulation. Four grabbable operands, all of which rebuild the
  // simulation under the cursor: the pool SPLIT, the sequence COUNT, the target
  // CONTEXT, and any one sequence's own target.
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const near = (v, a, b) => v >= a && v <= b;
    const setSplit = (x) => {
      const v = clamp(Math.round(((x - g.cap.x) / g.cap.w) * 100), 0, 100);
      if (v !== cur.sh.split) page.controls.set('split', v, { rebuild: true });
    };
    const setSeqs = (x) => {
      const v = clamp(Math.round(((x - g.dSeq.x) / g.dSeq.w) * MAX_SEQS), 1, MAX_SEQS);
      if (v !== cur.targets.length) page.controls.set('seqs', v, { rebuild: true });
    };
    const setCtx = (x) => {
      const f = clamp((x - g.dCtx.x) / g.dCtx.w, 0, 1);
      const v = clamp(Math.round((CTX_MIN + f * (CTX_MAX - CTX_MIN)) / 128) * 128, CTX_MIN, CTX_MAX);
      if (v !== (page.state.ctx | 0)) page.controls.set('ctx', v, { rebuild: true });
    };
    const setOne = (i, x) => {
      const f = clamp((x - g.seqBarX) / g.seqBarW, 0, 1);
      const v = clamp(Math.round((f * CTX_MAX) / cur.sh.pageTokens) * cur.sh.pageTokens, cur.sh.pageTokens, CTX_MAX);
      const next = cur.targets.slice();
      next[i] = v;
      page.controls.set('lens', next.join(','), { rebuild: true });
    };
    if (ev.type === 'down') {
      if (near(ev.y, g.cap.y - 12, g.cap.y + g.cap.h + 12) && near(ev.x, g.cap.x - 10, g.cap.x + g.cap.w + 10)) { drag = { k: 'split' }; setSplit(ev.x); }
      else if (near(ev.y, g.dSeq.y - 7, g.dSeq.y + g.dSeq.h + 7) && near(ev.x, g.dSeq.x - 6, g.dSeq.x + g.dSeq.w + 6)) { drag = { k: 'seqs' }; setSeqs(ev.x); }
      else if (near(ev.y, g.dCtx.y - 7, g.dCtx.y + g.dCtx.h + 7) && near(ev.x, g.dCtx.x - 6, g.dCtx.x + g.dCtx.w + 6)) { drag = { k: 'ctx' }; setCtx(ev.x); }
      else {
        const row = g.seqRows.find((rw) => near(ev.y, rw.y, rw.y + rw.h) && near(ev.x, g.seqBarX - 4, g.seqBarX + g.seqBarW + 4));
        if (row) { drag = { k: 'one', i: row.i }; setOne(row.i, ev.x); }
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      if (drag.k === 'split') setSplit(ev.x);
      else if (drag.k === 'seqs') setSeqs(ev.x);
      else if (drag.k === 'ctx') setCtx(ev.x);
      else setOne(drag.i, ev.x);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    if (!cur) return;
    r.clear(T.n0);
    const s = page.step() || cur.steps[0];
    const sh = cur.sh;
    const { pageOwner, slabOwner, seqs } = s;
    const W = page.W, H = page.H;
    const fnt = '11px ui-monospace, monospace';
    const hdr = '12.5px ui-monospace, monospace';
    const sml = '9px ui-monospace, monospace';

    const pad = 16, gutter = 104;
    const stripX = pad + gutter, stripW = Math.max(160, W - stripX - pad);

    // ---- shared capacity line: pages | slabs, with a draggable divider -------
    const cap = { x: stripX, y: Math.round(H * 0.055), w: stripW, h: Math.round(H * 0.05) };
    const divX = cap.x + (cap.w * sh.split) / 100;
    ctx.save();
    ctx.fillStyle = alphaOf(T.accent, 0.22); ctx.fillRect(cap.x, cap.y, divX - cap.x, cap.h);
    ctx.fillStyle = alphaOf(T.violet, 0.22); ctx.fillRect(divX, cap.y, cap.x + cap.w - divX, cap.h);
    // the residue the split can spend on neither shape
    const resW = (sh.residueMiB / POOL_MIB) * cap.w;
    if (resW > 0.5) { hatch(ctx, cap.x + cap.w - resW, cap.y, resW, cap.h, alphaOf(T.bad, 0.45)); }
    ctx.strokeStyle = rgbaToken('n14', 0.22); ctx.lineWidth = 1; ctx.strokeRect(cap.x, cap.y, cap.w, cap.h);
    // The grip lives ON the bar, not above it: the two side captions own the
    // line above and a glyph there collided with whichever one the split is near.
    ctx.strokeStyle = drag && drag.k === 'split' ? T.warn : T.n14; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(divX, cap.y - 3); ctx.lineTo(divX, cap.y + cap.h + 3); ctx.stroke();
    ctx.fillStyle = drag && drag.k === 'split' ? T.warn : T.n14;
    ctx.fillRect(divX - 4, cap.y + cap.h / 2 - 7, 8, 14);
    ctx.fillStyle = inkOn(drag && drag.k === 'split' ? T.warn : T.n14);
    ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('⇔', divX, cap.y + cap.h / 2);
    ctx.textBaseline = 'alphabetic';
    ctx.restore();
    r.label(`ONE POOL — ${POOL_MIB} MiB`, pad, cap.y + cap.h / 2 + 4, { color: T.n14, font: fnt });
    r.label(`KV pages ${mib((POOL_MIB * sh.split) / 100)} (grows with context)`, cap.x + 6, cap.y - 8, { color: T.accent, font: sml });
    const rLbl = `state slabs ${mib(POOL_MIB - (POOL_MIB * sh.split) / 100)} (grows with concurrency)`;
    ctx.save(); ctx.font = sml; const rw = ctx.measureText(rLbl).width; ctx.restore();
    r.label(rLbl, cap.x + cap.w - rw - 4, cap.y - 8, { color: T.violet, font: sml });

    // ---- the GROWING shape: a page pool that fills as sequences lengthen -----
    const gY = Math.round(H * 0.185), gH = Math.round(H * 0.155);
    r.label(`GROWING — ${sh.nPages} KV pages × ${sh.pageTokens} tok (${mib(sh.pageMiB)} each, ${sh.nAttn} attention layers)`, pad, gY - 10, { color: T.n14, font: hdr });
    r.label('attention →', pad, gY + gH / 2, { color: T.accent, font: sml });
    const cols = Math.min(40, Math.max(8, sh.nPages || 8));
    const rows = Math.max(1, Math.ceil(sh.nPages / cols));
    const cw = stripW / cols, chh = clamp(gH / rows, 3.5, 15);
    const pageRect = (p) => ({ x: stripX + (p % cols) * cw, y: gY + Math.floor(p / cols) * chh, w: cw, h: chh });
    let usedPages = 0;
    ctx.save();
    for (let p = 0; p < sh.nPages; p++) {
      const q = pageRect(p), o = pageOwner[p];
      ctx.fillStyle = o >= 0 ? alphaOf(seqColor(o), 0.9) : T.n1;
      ctx.fillRect(q.x, q.y, Math.max(1, q.w - 1), Math.max(1, q.h - 1));
      if (o >= 0) usedPages++;
      else hatch(ctx, q.x, q.y, Math.max(1, q.w - 1), Math.max(1, q.h - 1), rgbaToken('n14', 0.10));
    }
    ctx.strokeStyle = rgbaToken('n14', 0.18); ctx.lineWidth = 1;
    ctx.strokeRect(stripX, gY, stripW, rows * chh);
    ctx.restore();
    const freePages = sh.nPages - usedPages;
    const freePageFrac = sh.nPages ? freePages / sh.nPages : 1;

    // ---- the FIXED shape: a slab per sequence, allocated once, never growing -
    const sY = Math.round(H * 0.40), sH = Math.round(H * 0.055);
    r.label(`FIXED — ${sh.nSlabs} state slabs × ${mib(sh.slabMiB)} (${sh.nRec} recurrent layers, one per sequence, never grows)`, pad, sY - 10, { color: T.n14, font: hdr });
    r.label('recurrent →', pad, sY + sH / 2 + 4, { color: T.violet, font: sml });
    const sw = sh.nSlabs ? stripW / sh.nSlabs : stripW;
    const slabRect = (k) => ({ x: stripX + k * sw, y: sY, w: sw, h: sH });
    let usedSlabs = 0;
    ctx.save();
    for (let k = 0; k < sh.nSlabs; k++) {
      const q = slabRect(k), o = slabOwner[k];
      ctx.fillStyle = o >= 0 ? alphaOf(seqColor(o), 0.85) : T.n1;
      ctx.fillRect(q.x + 1, q.y, Math.max(1, q.w - 2), q.h);
      ctx.strokeStyle = o >= 0 ? alphaOf(seqColor(o), 0.95) : rgbaToken('n14', 0.2);
      ctx.strokeRect(q.x + 1, q.y, Math.max(1, q.w - 2), q.h);
      if (o >= 0) { usedSlabs++; if (sw > 22) { ctx.fillStyle = inkOn(seqColor(o)); ctx.font = sml; ctx.textAlign = 'center'; ctx.fillText(`S${o}`, q.x + q.w / 2, q.y + q.h / 2 + 3); } }
      else hatch(ctx, q.x + 1, q.y, Math.max(1, q.w - 2), q.h, rgbaToken('n14', 0.12));
    }
    if (!sh.nSlabs) { ctx.fillStyle = alphaOf(T.bad, 0.15); ctx.fillRect(stripX, sY, stripW, sH); }
    ctx.restore();
    const freeSlabs = sh.nSlabs - usedSlabs;

    // ---- demand: the two knobs that set what the pool is being asked for -----
    const Lw = Math.round(stripW * 0.52);
    const dY = Math.round(H * 0.535);
    const dSeq = { x: stripX, y: dY, w: Math.round(Lw * 0.42), h: 9 };
    const dCtx = { x: stripX + Math.round(Lw * 0.52), y: dY, w: Math.round(Lw * 0.42), h: 9 };
    const nSeq = cur.targets.length;
    const ctxTarget = clamp(page.state.ctx | 0, CTX_MIN, CTX_MAX);
    const maxTarget = Math.max(...cur.targets);
    const bar = (b, frac, col, on) => {
      ctx.save();
      ctx.fillStyle = T.n2; ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.fillStyle = alphaOf(col, 0.55); ctx.fillRect(b.x, b.y, b.w * frac, b.h);
      ctx.fillStyle = on ? T.warn : col; ctx.fillRect(b.x + b.w * frac - 2.5, b.y - 3, 5, b.h + 6);
      ctx.strokeStyle = rgbaToken('n14', 0.2); ctx.lineWidth = 1; ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.restore();
    };
    bar(dSeq, nSeq / MAX_SEQS, T.violet, drag && drag.k === 'seqs');
    bar(dCtx, (ctxTarget - CTX_MIN) / (CTX_MAX - CTX_MIN), T.accent, drag && drag.k === 'ctx');
    r.label(`◂▸ ${nSeq} sequences asked for`, dSeq.x, dSeq.y - 6, { color: T.violet, font: sml });
    r.label(`◂▸ target context ${tok(ctxTarget)} tok`, dCtx.x, dCtx.y - 6, { color: T.accent, font: sml });

    // ---- per-sequence rows: each one's own target is grabbable ---------------
    const seqTop = Math.round(H * 0.60);
    const rowH = clamp((H - seqTop - 34) / Math.max(1, nSeq), 9, 20);
    const seqBarX = stripX + 30, seqBarW = Lw - 118;
    const seqRows = [];
    let waiting = 0, live = 0, done = 0, preemptedNow = 0, queued = 0;
    r.label('sequences — drag a row', pad, seqTop - 7, { color: T.n14, font: hdr });
    ctx.save();
    for (let i = 0; i < nSeq; i++) {
      const q = seqs[i], y = seqTop + i * rowH, h = Math.max(5, rowH - 3);
      seqRows.push({ i, y, h });
      if (q.status === 'waiting') waiting++;
      else if (q.status === 'queued') queued++;
      else if (q.status === 'live') live++;
      else if (q.status === 'done') done++;
      else if (q.status === 'preempted') preemptedNow++;
      ctx.fillStyle = alphaOf(seqColor(i), 0.95); ctx.font = sml; ctx.textAlign = 'left';
      ctx.fillText(`S${i}`, stripX, y + h - 1);
      // target extent (ghost) + emitted extent (solid)
      const tW = (q.target / CTX_MAX) * seqBarW, eW = (q.emitted / CTX_MAX) * seqBarW;
      ctx.fillStyle = T.n2; ctx.fillRect(seqBarX, y, seqBarW, h);
      ctx.fillStyle = alphaOf(seqColor(i), 0.2); ctx.fillRect(seqBarX, y, tW, h);
      ctx.fillStyle = alphaOf(seqColor(i), 0.9); ctx.fillRect(seqBarX, y, eW, h);
      ctx.strokeStyle = drag && drag.k === 'one' && drag.i === i ? T.warn : alphaOf(seqColor(i), 0.55);
      ctx.lineWidth = 1; ctx.strokeRect(seqBarX + 0.5, y + 0.5, tW, h - 1);
      const badge = q.status === 'preempted' ? '⚠ preempted' : q.status === 'waiting' ? '… no slab' : q.status === 'queued' ? '· queued' : q.status === 'done' ? '✓ done' : '● live';
      ctx.fillStyle = q.status === 'preempted' ? T.bad : q.status === 'waiting' ? T.warn : q.status === 'queued' ? T.n9 : q.status === 'done' ? T.ok : T.n12;
      ctx.font = sml; ctx.fillText(`${badge}  ${tok(q.emitted)}/${tok(q.target)}`, seqBarX + seqBarW + 6, y + h - 1);
    }
    ctx.restore();

    // ---- the two ceilings, side by side, both moving with the split ---------
    const chX = stripX + Lw + 22, chY = Math.round(H * 0.56);
    const chW = Math.max(90, stripW - Lw - 22), chH = Math.round(H * 0.34);
    const concCeil = sh.nSlabs;
    const contCeil = ctxCeil(sh, sh.nPages, nSeq);
    // NAMED baseline: the same pool run by a pure-attention allocator, which has
    // no recurrent state and therefore needs no slab at all.
    const purePages = Math.floor(POOL_MIB / sh.pageMiB);
    const pureCont = ctxCeil(sh, purePages, nSeq);
    const pureConc = purePages;                     // one page each is all it needs
    // A little headroom so a curve that reaches its own maximum draws inside the
    // frame instead of riding the border.
    const concMax = Math.max(1, Math.floor(POOL_MIB / sh.slabMiB)) * 1.06;
    const contMax = Math.max(1, sh.pageTokens * Math.floor(purePages / Math.max(1, nSeq))) * 1.06;
    const SX = (sp) => chX + (sp / 100) * chW;
    const CY = (v, m) => chY + chH - (clamp(v / m, 0, 1)) * chH;

    // feasible band: splits at which BOTH ceilings clear the demand.
    let band = [], inBand = false;
    {
      let start = -1;
      for (let sp = 0; sp <= 100; sp++) {
        const okSp = slabsAt(sh, sp) >= nSeq && ctxCeil(sh, pagesAt(sh, sp), nSeq) >= maxTarget;
        if (okSp && start < 0) start = sp;
        if ((!okSp || sp === 100) && start >= 0) { band.push([start, okSp ? sp : sp - 1]); start = -1; }
        if (okSp && sp === sh.split) inBand = true;
      }
    }
    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(chX, chY, chW, chH);
    for (const [a, b] of band) { ctx.fillStyle = alphaOf(T.ok, 0.22); ctx.fillRect(SX(a), chY, Math.max(1, SX(b) - SX(a)), chH); }
    ctx.strokeStyle = rgbaToken('n14', 0.22); ctx.lineWidth = 1; ctx.strokeRect(chX, chY, chW, chH);
    // concurrency ceiling (violet, left scale) and context ceiling (accent, right)
    ctx.strokeStyle = T.violet; ctx.lineWidth = 2; ctx.beginPath();
    for (let sp = 0; sp <= 100; sp++) { const px = SX(sp), py = CY(slabsAt(sh, sp), concMax); if (!sp) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke();
    ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath();
    for (let sp = 0; sp <= 100; sp++) { const px = SX(sp), py = CY(ctxCeil(sh, pagesAt(sh, sp), nSeq), contMax); if (!sp) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke();
    // demand lines
    ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.strokeStyle = alphaOf(T.violet, 0.75); ctx.beginPath(); ctx.moveTo(chX, CY(nSeq, concMax)); ctx.lineTo(chX + chW, CY(nSeq, concMax)); ctx.stroke();
    ctx.strokeStyle = alphaOf(T.accent, 0.75); ctx.beginPath(); ctx.moveTo(chX, CY(maxTarget, contMax)); ctx.lineTo(chX + chW, CY(maxTarget, contMax)); ctx.stroke();
    ctx.setLineDash([]);
    // where we are
    ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(SX(sh.split), chY); ctx.lineTo(SX(sh.split), chY + chH); ctx.stroke();
    ctx.restore();
    r.label('ceilings vs split — both, side by side', chX, chY - 20, { color: T.n14, font: hdr });
    const cLbl = `concurrency ${concCeil} seq`;
    r.label(cLbl, chX, chY - 7, { color: T.violet, font: sml });
    ctx.save(); ctx.font = sml; const cLw = ctx.measureText(cLbl + '  ').width; ctx.restore();
    r.label(`context ${tok(contCeil)} tok/seq`, chX + cLw, chY - 7, { color: T.accent, font: sml });
    r.label('0% pages', chX, chY + chH + 11, { color: T.n10, font: sml });
    ctx.save(); ctx.font = sml; const e100 = ctx.measureText('100%').width; ctx.restore();
    r.label('100%', chX + chW - e100, chY + chH + 11, { color: T.n10, font: sml });
    const bandText = band.length
      ? `both ceilings met at split ${band.map(([a, b]) => `${a}–${b}%`).join(', ')}`
      : `NO split meets both — ${nSeq} seq × ${tok(maxTarget)} tok does not fit ${POOL_MIB} MiB`;
    r.label(bandText, chX, chY + chH + 24, { color: band.length ? T.ok : T.bad, font: sml });

    // ---- wasted memory, per regime -----------------------------------------
    const idlePage = freePages * sh.pageMiB, idleSlab = freeSlabs * sh.slabMiB;
    const slabTokenEquiv = Math.round((sh.slabMiB / sh.pageMiB) * sh.pageTokens);
    const plate = (x, y, w, h) => { ctx.save(); ctx.fillStyle = T.n0; ctx.fillRect(x, y, w, h); ctx.restore(); };
    const plated = (text, x, y, color, font) => {
      ctx.save(); ctx.font = font; const w = ctx.measureText(text).width; ctx.restore();
      plate(x - 3, y - 11, w + 6, 15);
      r.label(text, x, y, { color, font });
    };
    plated(`${usedPages}/${sh.nPages} pages in use · ${mib(idlePage)} idle on the growing side`, stripX, gY + rows * chh + 13, freePages ? T.n12 : T.bad, sml);
    plated(`${usedSlabs}/${sh.nSlabs} slabs held · ${mib(idleSlab)} idle on the fixed side · each slab = ${tok(slabTokenEquiv)} tok of KV never bought`, stripX, sY + sH + 13, freeSlabs ? T.n12 : T.bad, sml);
    if (sh.residueMiB > 0.01) plated(`${mib(sh.residueMiB)} stranded — too small for another ${mib(sh.pageMiB)} page or ${mib(sh.slabMiB)} slab`, stripX, cap.y + cap.h + 13, T.bad, sml);

    geom = { cap, dSeq, dCtx, seqRows, seqBarX, seqBarW, pageRect, slabRect, cols, rows, cw, chh, gY, sY, sH, chart: { x: chX, y: chY, w: chW, h: chH } };
    // Everything reported is read off THIS snapshot, never off the end of the
    // run: a step-3 frame must not quote a preemption that happens at step 20.
    page.probe = {
      preempted: preemptedNow, waiting, queued, live, done, freePageFrac,
      concCeil, contCeil, inBand, bandText, split: sh.split,
    };

    // ---- hover-to-inspect ---------------------------------------------------
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      let tip = null;
      if (p.y >= gY && p.y <= gY + rows * chh && p.x >= stripX && p.x <= stripX + stripW) {
        const c0 = Math.floor((p.x - stripX) / cw), r0 = Math.floor((p.y - gY) / chh), idx = r0 * cols + c0;
        if (idx >= 0 && idx < sh.nPages) {
          const o = pageOwner[idx];
          if (o < 0) tip = `KV page ${idx} — FREE (${mib(sh.pageMiB)})\nGROWING shape: the attention layers' cache.\nAny live sequence takes it the moment its current page fills.`;
          else {
            const q = seqs[o], L = q.pages.indexOf(idx);
            tip = `KV page ${idx} — S${o}, logical page ${L}\ncovers tokens ${L * sh.pageTokens}..${(L + 1) * sh.pageTokens - 1}\n${mib(sh.pageMiB)} = ${sh.pageTokens} tok × ${sh.nAttn} attention layers × ${KV_KIB_PER_TOKEN_PER_ATTN_LAYER} KiB\nS${o} holds ${q.pages.length} page(s); it will take another every ${sh.pageTokens} tokens.`;
          }
        }
      } else if (p.y >= sY && p.y <= sY + sH && p.x >= stripX && p.x <= stripX + stripW) {
        const k = Math.floor((p.x - stripX) / sw);
        if (k >= 0 && k < sh.nSlabs) {
          const o = slabOwner[k];
          tip = o < 0
            ? `state slab ${k} — FREE (${mib(sh.slabMiB)})\nFIXED shape: the recurrent layers' state.\nOne is charged in full the instant a sequence is admitted.`
            : `state slab ${k} — S${o} (${mib(sh.slabMiB)})\n${sh.nRec} recurrent layers × ${STATE_MIB_PER_RECURRENT_LAYER} MiB, allocated ONCE at admission\nsize does NOT depend on length: S${o} has emitted ${tok(seqs[o].emitted)} tok and pays the same as one at ${tok(CTX_MAX)}\nsame bytes would have bought ${tok(slabTokenEquiv)} tok of KV pages`;
        }
      } else {
        const row = seqRows.find((rw) => p.y >= rw.y && p.y <= rw.y + rw.h && p.x >= seqBarX - 4 && p.x <= seqBarX + seqBarW + 90);
        if (row) {
          const q = seqs[row.i];
          tip = `S${row.i} — ${q.status.toUpperCase()}\n${tok(q.emitted)} of ${tok(q.target)} tok emitted, ${q.pages.length} page(s) + ${q.slab >= 0 ? `slab ${q.slab}` : 'no slab'}\n${q.why}\n◂▸ drag this row to change only this sequence's target`;
        } else if (p.y >= cap.y - 12 && p.y <= cap.y + cap.h + 12) {
          tip = `pool split — ${sh.split}% to KV pages, ${100 - sh.split}% to state slabs\n${sh.nPages} pages × ${mib(sh.pageMiB)} + ${sh.nSlabs} slabs × ${mib(sh.slabMiB)}\nconcurrency ceiling ${concCeil} seq (set by the FIXED side)\ncontext ceiling ${tok(contCeil)} tok at ${nSeq} seq (set by the GROWING side)\n⇔ drag: one knob, two ceilings, opposite directions`;
        } else if (p.x >= chX && p.x <= chX + chW && p.y >= chY && p.y <= chY + chH) {
          const sp = clamp(Math.round(((p.x - chX) / chW) * 100), 0, 100);
          tip = `at split ${sp}%: concurrency ${slabsAt(sh, sp)} seq, context ${tok(ctxCeil(sh, pagesAt(sh, sp), nSeq))} tok/seq\ndemand is ${nSeq} seq × ${tok(maxTarget)} tok\ngreen band = splits that clear BOTH dashed demand lines`;
        }
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout ------------------------------------------------------------
    const contPct = pureCont ? (100 * contCeil) / pureCont : 0;
    let o = `${s.label}    tier:${r.name}\n`;
    o += `CEILINGS at split ${sh.split}% → concurrency ${concCeil} sequences (fixed side: ${sh.nSlabs} slabs × ${mib(sh.slabMiB)}) · context ${tok(contCeil)} tok/seq at ${nSeq} sequences (growing side: ${sh.nPages} pages × ${sh.pageTokens} tok). `;
    o += `Baseline — a PURE-ATTENTION allocator on the same ${POOL_MIB} MiB pool has no recurrent state and so needs no slab at all: all ${purePages} pages are KV, its context ceiling at ${nSeq} sequences is ${tok(pureCont)} tok/seq, and it can admit ${pureConc} sequences before it runs out of room for their prompts. `;
    o += `This allocator's context ceiling is ${contPct.toFixed(1)}% of that pure-attention baseline (higher is better; 100% = parity), and its concurrency is hard-capped at ${concCeil} where the baseline's is not capped by fixed state at all.\n`;
    o += `NOW (this step): ${live} live, ${waiting} refused for want of a slab, ${queued} not yet reached, ${done} finished, ${preemptedNow} preempted mid-generation. Idle ${mib(idlePage)} of pages + ${mib(idleSlab)} of slabs${sh.residueMiB > 0.01 ? ` + ${mib(sh.residueMiB)} stranded` : ''}. `;
    o += waiting && freePageFrac > 0.4
      ? `REGIME (b): ${waiting} sequence(s) cannot be admitted while ${Math.round(100 * freePageFrac)}% of the page pool sits free — the FIXED side is the wall, and each refused sequence needs its whole ${mib(sh.slabMiB)} slab before it emits one token.`
      : preemptedNow
        ? `REGIME (a): ${preemptedNow} sequence(s) were admitted and then preempted MID-GENERATION when the pages ran out — the GROWING side is the wall, and no amount of state slab helps.`
        : `Neither wall has been hit at this step. ${bandText}.`;
    o += `\n${bandText}. One knob sets both ceilings, and they move in opposite directions: every page you add costs a slab, and every slab costs pages.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__hybridAllocPage = page;
  const q = new URLSearchParams(location.search);
  const num = (k, lo, hi) => clamp(parseInt(q.get(k), 10), lo, hi);
  if (q.has('split')) page.controls.set('split', num('split', 0, 100), { rebuild: true });
  if (q.has('seqs')) page.controls.set('seqs', num('seqs', 1, MAX_SEQS), { rebuild: true });
  if (q.has('ctx')) page.controls.set('ctx', num('ctx', CTX_MIN, CTX_MAX), { rebuild: true });
  if (q.has('page')) page.controls.set('page', snapTo(num('page', 64, 256), PAGE_TOKENS), { rebuild: true });
  if (q.has('attnevery')) page.controls.set('attnevery', snapTo(num('attnevery', 2, 8), ATTN_EVERY), { rebuild: true });
  if (q.has('lens')) page.controls.set('lens', q.get('lens'), { rebuild: true });
  const t = page.controls._transport;
  // ?hover=x,y fakes the cursor so the tooltip path is verifiable headlessly.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) { t.rebuildIfDirty(); t.seek(parseInt(q.get('step'), 10)); }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
