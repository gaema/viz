// batch-invariance -- your reply depends on who else was batched with you.
//
// THE CLAIM THIS PAGE MAKES, AND WHY IT IS ARITHMETIC RATHER THAN A BUG:
// floating-point addition is not associative. (a+b)+c and a+(b+c) are different
// numbers in general, because each `+` rounds. A GPU reduction kernel does not
// add left-to-right; it splits the row across some number of parallel lanes,
// sums each lane, then combines the lane totals in a tree. HOW MANY lanes it
// picks depends on the SHAPE of the launch -- and the shape depends on how many
// requests are in flight. So the same prompt, the same weights, the same seed
// and greedy decoding can land on a different token because somebody else's
// request was batched alongside yours.
//
// NOTHING ON THIS PAGE IS A HARD-CODED "DIFFERENCE". Every number is computed
// here, in real IEEE-754 binary32: JavaScript numbers are binary64, and
// Math.fround() rounds to binary32 after each operation, so the accumulator
// carries exactly the bits a float32 kernel would carry. The two sums are
// produced by two real loops, and the bit strips are read out of the same
// Float32Array/Uint32Array pair, so the strips cannot disagree with the values.
// When a configuration produces NO difference the page says so -- that is a
// result too, and it is what the batch-invariant toggle is for.
//
// Source for the mechanism and for the fix: Thinking Machines, "Defeating
// Nondeterminism in LLM Inference" (2025-09),
// https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
// -- which identifies batch-size-dependent reduction order as the cause of
// run-to-run nondeterminism in a server that is otherwise fully deterministic,
// and proposes batch-invariant kernels (a fixed reduction split, independent of
// batch size) as the fix, at a throughput cost.
//
// WHAT THIS PAGE CANNOT DO: it cannot measure a real kernel's slowdown. The
// throughput cost of batch invariance is a claim taken from the source above,
// and it is labelled as such wherever it appears. What the page DOES show about
// cost is arithmetic and checkable: a fixed split leaves lanes idle at batch
// sizes the adaptive split would have filled, and that idle fraction is drawn
// from the same lane counts the reduction actually used.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';
import { T, alphaOf, signedColor, inkOn, rgbaToken } from '../framework/theme.js';

// ---- real binary32 -------------------------------------------------------
// One scratch pair, aliased: writing f32[0] and reading u32[0] IS the bit
// pattern of the float32, so a displayed bit string is never a re-derivation.
const fr = Math.fround;
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const bitsOf = (x) => { f32[0] = x; return u32[0] >>> 0; };
const fromBits = (b) => { u32[0] = b >>> 0; return f32[0]; };
/** Move k representable float32 steps away from zero (k may be negative). */
function addUlps(x, k) {
  const b = bitsOf(x);
  return fromBits((b >>> 31) ? (b - k) >>> 0 : (b + k) >>> 0);
}
/** Signed distance in representable steps, for two same-sign finite floats. */
const ulpDist = (a, b) => bitsOf(b) - bitsOf(a);
/** Position (0 = sign bit, 31 = last mantissa bit) of the FIRST bit that differs, or -1. */
function firstDiffBit(a, b) {
  const d = (bitsOf(a) ^ bitsOf(b)) >>> 0;
  return d === 0 ? -1 : Math.clz32(d);
}
const bitStr = (x) => bitsOf(x).toString(2).padStart(32, '0');
/** Shortest round-trip decimal of the value -- the number itself, not a rounding of it. */
const exact = (x) => String(x);

/** Trim to fit `maxW` px in the context's CURRENT font. Measured, not counted. */
function ellipsize(ctx, txt, maxW) {
  if (maxW <= 0) return '';
  if (ctx.measureText(txt).width <= maxW) return txt;
  let lo = 0, hi = txt.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(txt.slice(0, mid) + '…').width <= maxW) lo = mid; else hi = mid - 1;
  }
  return lo > 0 ? txt.slice(0, lo) + '…' : '';
}

function fmt(x) {
  if (x == null || Number.isNaN(x)) return '—';
  if (!Number.isFinite(x)) return x > 0 ? '+Inf' : '−Inf';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(2);
  return String(Number(x.toPrecision(6)));
}

// ---- the split a kernel picks --------------------------------------------
// A reduction kernel has a fixed pool of parallel lanes to spend. With B rows in
// flight it can only give each row a share of them, so the per-row split -- and
// therefore the per-row ADDITION ORDER -- is a function of the batch size. That
// is the whole mechanism, and this is the smallest honest model of it: take the
// largest power of two that fits in POOL/B. It is a model of shape-driven split
// selection, not a transcription of any particular vendor's kernel.
const POOL = 64;
// The batch-invariant kernel commits to ONE split and uses it at every batch
// size. It is not "the sequential order" -- it is A fixed order, chosen once.
const FIXED_LANES = 4;
const lanesFor = (batch, n) => Math.max(1, Math.min(n, Math.pow(2, Math.max(0, Math.floor(Math.log2(POOL / Math.max(1, batch)))))));
const lanesUsed = (st, n) => (st.invariant ? Math.min(n, FIXED_LANES) : lanesFor(st.batch, n));

// ---- the row of values ----------------------------------------------------
// Four deterministic rows, so a screenshot reproduces. `mixed` spans about six
// decades, which is where non-associativity bites hardest; `uniform` is all
// ones, where every order gives the same answer -- kept precisely because a page
// that can only ever show a difference is not showing arithmetic, it is showing
// a rigged example.
const PRESETS = [
  { value: 'mixed', label: 'mixed magnitudes (wide dynamic range)' },
  { value: 'cancel', label: 'catastrophic cancellation (±1024 + crumbs)' },
  { value: 'heavy', label: 'heavy tail (one huge, many tiny)' },
  { value: 'uniform', label: 'all ones (no rounding to lose)' },
];
function genValue(i, preset) {
  if (preset === 'cancel') return (i % 2 === 0 ? 1024 : -1024) + Math.sin(i * 2.3) * 7e-4 + 1.3e-3;
  if (preset === 'heavy') return i === 0 ? 4096 : Math.sin(i * 1.7) * 9e-4 + 1.1e-3;
  if (preset === 'uniform') return 1;
  return (Math.sin(i * 1.7) * 1.3 + Math.cos(i * 0.41) * 0.6) * Math.pow(6, (i % 6) - 2);
}
/** Parse the `edit` control ("3:-1.25,7:0.5") into {index: value}. */
function parseEdits(s) {
  const out = {};
  for (const tok of String(s || '').split(/[,;]/)) {
    const m = /^\s*(\d+)\s*:\s*(-?[0-9.eE+-]+)\s*$/.exec(tok);
    if (m) { const v = parseFloat(m[2]); if (Number.isFinite(v)) out[+m[1]] = fr(v); }
  }
  return out;
}
const editStr = (e) => Object.keys(e).map((k) => +k).sort((a, b) => a - b).map((k) => `${k}:${Number(e[k].toPrecision(6))}`).join(',');

function valuesFor(st) {
  const n = Math.max(2, st.n | 0), ed = parseEdits(st.edit);
  return Array.from({ length: n }, (_, i) => (ed[i] != null ? ed[i] : fr(genValue(i, st.preset))));
}

// ---- the two reductions ---------------------------------------------------
/** Left-to-right, one accumulator. The textbook order, and what a single-lane kernel does. */
function seqTrace(vals) {
  let s = 0; const running = [];
  for (let i = 0; i < vals.length; i++) { s = fr(s + vals[i]); running.push(s); }
  return { sum: s, running };
}
/**
 * Split into `lanes` contiguous chunks, sum each chunk left-to-right, then
 * combine the lane totals pairwise up a tree. Every add is rounded to float32.
 * `span` records which ELEMENTS each node covers, so the drawing can place a
 * node over the values it is made of and the tree visibly re-shapes with lanes.
 */
function splitTrace(vals, lanes) {
  const n = vals.length, chunk = Math.ceil(n / lanes);
  const partials = [];
  for (let a = 0; a < n; a += chunk) {
    const b = Math.min(n, a + chunk);
    let s = 0; const trace = [];
    for (let i = a; i < b; i++) { s = fr(s + vals[i]); trace.push({ i, x: vals[i], s }); }
    partials.push({ lo: a, hi: b, sum: s, trace });
  }
  const levels = [partials.map((p) => ({ v: p.sum, lo: p.lo, hi: p.hi }))];
  const combines = [];
  while (levels[levels.length - 1].length > 1) {
    const cur = levels[levels.length - 1], next = [];
    for (let i = 0; i < cur.length; i += 2) {
      if (i + 1 < cur.length) {
        const v = fr(cur[i].v + cur[i + 1].v);
        combines.push({ lvl: levels.length - 1, out: next.length, a: cur[i].v, b: cur[i + 1].v, v, lo: cur[i].lo, hi: cur[i + 1].hi });
        next.push({ v, lo: cur[i].lo, hi: cur[i + 1].hi });
      } else next.push(cur[i]);          // odd node rides up untouched
    }
    levels.push(next);
  }
  return { sum: levels[levels.length - 1][0].v, partials, levels, combines, chunk };
}

function scene(st) {
  const vals = valuesFor(st), n = vals.length;
  const lanes = lanesUsed(st, n);
  const seq = seqTrace(vals), spl = splitTrace(vals, lanes);
  return { vals, n, lanes, seq, spl, diffBit: firstDiffBit(seq.sum, spl.sum), ulps: ulpDist(seq.sum, spl.sum) };
}

// ---- the amplifier: close logits, greedy argmax ---------------------------
// A last-bit difference is invisible until something THRESHOLDS on it. Greedy
// decoding is exactly that: take the largest logit. Four candidates share the
// scene; the runner-up sits a controllable number of representable steps above
// the sequential sum, so whether it wins is decided by bits the reduction order
// moved. The two offset distractors only exist to make the softmax a real
// distribution rather than a two-horse race.
// Read the query ONCE, at module load. The framework mirrors control state back
// into location.search on every change, and the transport's autoplay can fire
// its first seek before mount()'s promise resolves -- so a deep link read inside
// .then() is read from a string the page has already overwritten with its own
// defaults. Measured: ?view=argmax was dropped on some loads and honoured on
// others, purely on rAF timing.
const Q = (typeof location !== 'undefined') ? new URLSearchParams(location.search) : new URLSearchParams();

const TOKENS = ['alpha', 'beta', 'gamma', 'delta'];
function logitsFor(sum, ref, rivalUlps) {
  return [sum, addUlps(ref, rivalUlps | 0), fr(ref - 0.35), fr(ref - 0.9)];
}
function softmax(xs) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const z = e.reduce((p, q) => p + q, 0);
  return e.map((x) => x / z);
}
const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

// ---- transport steps ------------------------------------------------------
// Lane-local accumulation first (lane-major, which for contiguous chunks visits
// the elements in index order -- so the SEQUENTIAL running total can advance on
// the same counter and both orders are watchable at once), then the tree
// combines, then one final compare step.
function buildSteps(st) {
  const s = scene(st), steps = [];
  let revLane = 0;
  s.spl.partials.forEach((p, li) => p.trace.forEach((tr) => {
    revLane++;
    steps.push({
      kind: 'lane', lane: li, idx: tr.i, revLane, revComb: 0,
      label: `lane ${li}: acc = acc + x[${tr.i}]  →  ${fmt(tr.s)}   (sequential after ${revLane}: ${fmt(s.seq.running[tr.i])})`,
    });
  }));
  const totalLane = revLane;
  s.spl.combines.forEach((c, k) => steps.push({
    kind: 'combine', lvl: c.lvl, out: c.out, revLane: totalLane, revComb: k + 1,
    label: `combine level ${c.lvl}: ${fmt(c.a)} + ${fmt(c.b)} → ${fmt(c.v)}   (rounded to float32)`,
  }));
  steps.push({
    kind: 'done', revLane: totalLane, revComb: s.spl.combines.length,
    label: s.diffBit < 0
      ? `both orders landed on the SAME float32 (${exact(s.seq.sum)}) — this shape has nothing to lose`
      : `sequential ${exact(s.seq.sum)}  vs  split ${exact(s.spl.sum)} — first difference at bit ${s.diffBit} of 32`,
  });
  return steps;
}

// ---- drawing --------------------------------------------------------------
const PAD = 16;
let valRect = null;          // the value-cell strip, for hover + drag
let batchRect = null;        // the batch-size strip, for drag
let nodeHits = [];           // [{x,y,w,h,tip}] tree nodes + sums, for hover
let costRows = [];           // [{y,h,batch}] cost-view rows, for click-to-select

function drawBitStrip(page, x, y, w, val, other, label) {
  const ctx = page.ctx, bits = bitStr(val), ob = other == null ? null : bitStr(other);
  const labW = 132, cw = Math.max(4, Math.min(19, (w - labW) / 32)), h = 20;
  const x0 = x + labW;
  ctx.save();
  ctx.font = '10.5px ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  ctx.fillStyle = T.n11; ctx.fillText(label, x, y + h / 2);
  for (let i = 0; i < 32; i++) {
    const field = i === 0 ? 'bad' : i <= 8 ? 'accent' : 'ok';
    const differs = ob ? ob[i] !== bits[i] : false;
    const cx = x0 + i * cw;
    ctx.fillStyle = differs ? alphaOf(T.warn, 0.92) : alphaOf(field, bits[i] === '1' ? 0.82 : 0.16);
    ctx.fillRect(cx, y, cw - 1, h);
    if (differs) { ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5; ctx.strokeRect(cx + 0.75, y + 0.75, cw - 2.5, h - 1.5); }
    if (cw >= 8) {
      ctx.fillStyle = differs ? inkOn(alphaOf(T.warn, 1)) : (bits[i] === '1' ? T.n0 : T.n11);
      ctx.textAlign = 'center'; ctx.font = (cw < 11 ? '8px' : '10px') + ' ui-monospace, monospace';
      ctx.fillText(bits[i], cx + (cw - 1) / 2, y + h / 2);
      ctx.textAlign = 'left';
    }
  }
  ctx.restore();
  return { x0, cw, h };
}

function drawReduce(page) {
  const ctx = page.ctx, r = page.renderer, st = page.state, W = page.W, H = page.H;
  const s = scene(st);
  const step = page.step();
  const revLane = step ? step.revLane : s.n;
  const revComb = step ? step.revComb : s.spl.combines.length;
  nodeHits = [];

  r.clear(T.n0);
  // Colour the row by LOG magnitude, not linear. The whole point of the mixed
  // preset is that it spans several decades; on a linear ramp everything except
  // the largest few elements collapses onto the page ground and the row reads
  // as empty -- which hides exactly the spread that makes the orders disagree.
  const mags = s.vals.map((v) => Math.abs(v)).filter((m) => m > 0);
  const loMag = mags.length ? Math.log10(Math.min(...mags)) : 0;
  const hiMag = mags.length ? Math.log10(Math.max(...mags)) : 0;
  const span = hiMag - loMag;
  const shade = (v) => {
    if (v === 0) return 0;
    // A row whose magnitudes are all alike (the ±1024 cancellation preset) has
    // no spread to normalise against; paint it at full saturation rather than
    // dividing by a near-zero span and washing every cell out.
    if (span < 0.35) return Math.sign(v);
    return Math.sign(v) * (0.2 + 0.8 * Math.min(1, Math.max(0, (Math.log10(Math.abs(v)) - loMag) / span)));
  };

  // --- header ---------------------------------------------------------------
  ctx.save(); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
  ctx.fillStyle = T.n14; ctx.font = 'bold 12px ui-monospace, monospace';
  ctx.fillText(`batch = ${st.batch}   →   ${s.lanes} reduction lanes for this row   ·   chunk = ${s.spl.chunk} element${s.spl.chunk === 1 ? '' : 's'} per lane`, PAD, 15);
  ctx.fillStyle = st.invariant ? T.okDeep : T.n11; ctx.font = '10.5px ui-monospace, monospace';
  ctx.fillText(ellipsize(ctx, st.invariant
    ? `batch-invariant kernel ON — the split is pinned at ${FIXED_LANES} lanes whatever the batch size`
    : `shape-adaptive kernel — ${POOL} lanes shared across ${st.batch} row${st.batch === 1 ? '' : 's'}, so the split moves with the batch`, W - 2 * PAD), PAD, 29);
  ctx.restore();

  // --- the batch strip (draggable) -----------------------------------------
  const bsY = 36, bsH = 12, bsW = W - 2 * PAD;
  batchRect = { x: PAD, y: bsY, w: bsW, h: bsH };
  ctx.save();
  ctx.fillStyle = rgbaToken('n14', 0.07); ctx.fillRect(PAD, bsY, bsW, bsH);
  const bFrac = st.batch / 32;
  ctx.fillStyle = alphaOf(T.accent, 0.55); ctx.fillRect(PAD, bsY, bsW * bFrac, bsH);
  ctx.fillStyle = T.accent; ctx.fillRect(PAD + bsW * bFrac - 1.5, bsY - 2, 3, bsH + 4);
  ctx.font = '9.5px ui-monospace, monospace'; ctx.textBaseline = 'middle';
  // Right-aligned, so the label never sits under the fill it describes.
  ctx.textAlign = 'right'; ctx.fillStyle = T.n11;
  ctx.fillText('◂ drag: batch size (how many requests are in flight) ▸', PAD + bsW - 6, bsY + bsH / 2);
  ctx.restore();

  // --- the value row (draggable cells) -------------------------------------
  const vY = 56, vH = 30, cw = (W - 2 * PAD) / s.n;
  valRect = { x: PAD, y: vY, w: W - 2 * PAD, h: vH };
  ctx.save(); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < s.n; i++) {
    const x = PAD + i * cw, fill = i < revLane ? signedColor(shade(s.vals[i])) : T.n2;
    ctx.fillStyle = fill; ctx.fillRect(x + 0.5, vY, cw - 1, vH);
    ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, vY, cw - 1, vH);
    if (step && step.kind === 'lane' && step.idx === i) { ctx.strokeStyle = T.n14; ctx.lineWidth = 2.5; ctx.strokeRect(x + 1.5, vY + 1, cw - 3, vH - 2); }
    if (cw >= 30 && i < revLane) { ctx.fillStyle = inkOn(fill); ctx.font = '8.5px ui-monospace, monospace'; ctx.fillText(fmt(s.vals[i]), x + cw / 2, vY + vH / 2); }
  }
  ctx.restore();

  // --- the sequential arm, advancing on the same element counter -----------
  const sqY = vY + vH + 6, sqH = 18;
  const seqNow = revLane > 0 ? s.seq.running[revLane - 1] : 0;
  ctx.save();
  ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(PAD, sqY, W - 2 * PAD, sqH);
  ctx.fillStyle = alphaOf(T.violet, 0.35); ctx.fillRect(PAD, sqY, (W - 2 * PAD) * (revLane / s.n), sqH);
  ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  ctx.fillStyle = T.n13; ctx.fillText(`sequential, one accumulator:  acc = ${fmt(seqNow)}   after ${revLane}/${s.n}`, PAD + 6, sqY + sqH / 2);
  ctx.restore();
  nodeHits.push({ x: PAD, y: sqY, w: W - 2 * PAD, h: sqH, tip: `sequential reduction — one accumulator, left to right\nacc after ${revLane} adds = ${exact(seqNow)}\nfinal = ${exact(s.seq.sum)}\nbits ${bitStr(s.seq.sum)}` });

  // --- the reduction tree ---------------------------------------------------
  const bitsTop = H - 96;
  const treeTop = sqY + sqH + 14, treeBot = bitsTop - 40;
  const nlv = s.spl.levels.length;
  const lvH = Math.max(20, Math.min(40, (treeBot - treeTop) / Math.max(1, nlv)));
  // Centre the tree in its band: a 4-level tree in a band sized for 6 otherwise
  // sits pinned to the top with a dead strip under it.
  const treeY = treeTop + Math.max(0, (treeBot - treeTop - nlv * lvH) / 2);
  const cxOf = (lo, hi) => PAD + ((lo + hi) / 2) * cw;
  ctx.save(); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let lv = 0; lv < nlv; lv++) {
    const row = s.spl.levels[lv], y = treeY + lv * lvH;
    for (let j = 0; j < row.length; j++) {
      const node = row[j];
      // Is this node computed yet?
      let ready;
      if (lv === 0) ready = revLane >= s.spl.partials[j].hi;
      else {
        const idx = s.spl.combines.findIndex((c) => c.lvl === lv - 1 && c.out === j);
        ready = idx < 0 ? true : revComb > idx;      // an odd node rides up for free
      }
      const nw = Math.max(20, Math.min(112, (node.hi - node.lo) * cw - 4)), nh = Math.min(22, lvH - 6);
      const x = cxOf(node.lo, node.hi) - nw / 2;
      // edges down to the children
      if (lv > 0) {
        ctx.strokeStyle = ready ? alphaOf(T.accent, 0.55) : rgbaToken('n9', 0.4); ctx.lineWidth = 1;
        for (const ch of s.spl.levels[lv - 1]) {
          if (ch.lo < node.lo || ch.hi > node.hi) continue;
          ctx.beginPath(); ctx.moveTo(cxOf(node.lo, node.hi), y); ctx.lineTo(cxOf(ch.lo, ch.hi), y - lvH + nh); ctx.stroke();
        }
      }
      const fill = ready ? alphaOf(lv === 0 ? T.teal : T.accent, 0.82) : T.n2;
      ctx.fillStyle = fill; ctx.fillRect(x, y, nw, nh);
      ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(x, y, nw, nh);
      if (step && step.kind === 'combine' && step.lvl === lv - 1 && step.out === j) { ctx.strokeStyle = T.n14; ctx.lineWidth = 2.5; ctx.strokeRect(x + 1, y + 1, nw - 2, nh - 2); }
      if (ready && nw >= 30) { ctx.fillStyle = inkOn(fill); ctx.font = (nw < 52 ? '8.5px' : '10px') + ' ui-monospace, monospace'; ctx.fillText(fmt(node.v), x + nw / 2, y + nh / 2); }
      const kids = lv === 0
        ? `elements ${node.lo}..${node.hi - 1}, summed left to right in this lane`
        : `children: ${s.spl.levels[lv - 1].filter((c) => c.lo >= node.lo && c.hi <= node.hi).map((c) => fmt(c.v)).join('  +  ')}`;
      nodeHits.push({ x, y, w: nw, h: nh, tip: `${lv === 0 ? `lane ${j}` : `combine, level ${lv}`}\n${kids}\npartial = ${exact(node.v)}\nbits ${bitStr(node.v)}` });
    }
  }
  ctx.restore();

  // --- the compare line -----------------------------------------------------
  const cmpY = bitsTop - 24;
  ctx.save(); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.font = '11px ui-monospace, monospace';
  const cmpTxt = s.diffBit < 0
    ? `both orders → ${exact(s.spl.sum)}   ·   identical bit pattern   ·   nothing was lost at this shape`
    : `same values, two orders → ${Math.abs(s.ulps)} representable step${Math.abs(s.ulps) === 1 ? '' : 's'} apart   ·   first differing bit: ${s.diffBit} of 32   ·   absolute gap ${fmt(Math.abs(s.spl.sum - s.seq.sum))}`;
  ctx.fillStyle = s.diffBit < 0 ? T.okDeep : T.warnDeep;
  ctx.fillText(ellipsize(ctx, cmpTxt, W - 2 * PAD), PAD, cmpY);
  ctx.restore();

  // --- the bits -------------------------------------------------------------
  const g = drawBitStrip(page, PAD, bitsTop, W - 2 * PAD, s.seq.sum, s.spl.sum, 'sequential');
  drawBitStrip(page, PAD, bitsTop + 24, W - 2 * PAD, s.spl.sum, s.seq.sum, `split (${s.lanes} lanes)`);
  ctx.save();
  ctx.font = '9px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'center';
  ctx.fillStyle = T.n10;
  ctx.fillText('s', g.x0 + g.cw / 2, bitsTop + 50);
  ctx.fillText('exponent (8)', g.x0 + g.cw * 5, bitsTop + 50);
  ctx.fillText('mantissa (23)', g.x0 + g.cw * 20.5, bitsTop + 50);
  if (s.diffBit >= 0) {
    ctx.fillStyle = T.warnDeep; ctx.font = 'bold 9px ui-monospace, monospace';
    ctx.fillText('▲ first difference', g.x0 + g.cw * (s.diffBit + 0.5), bitsTop + 62);
  }
  ctx.restore();

  // --- hover ---------------------------------------------------------------
  const p = page.pointer;
  if (p.over) {
    const hitV = valRect && cellAt(valRect, 1, s.n, p.x, p.y);
    if (hitV) {
      const i = hitV.c, lane = Math.min(s.spl.partials.length - 1, Math.floor(i / s.spl.chunk));
      page.setTip(`x[${i}] = ${exact(s.vals[i])}\nbits ${bitStr(s.vals[i])}\nin lane ${lane} (elements ${s.spl.partials[lane].lo}..${s.spl.partials[lane].hi - 1})\ndrag up/down to scale it — the split re-rounds around it`);
    } else {
      const hit = nodeHits.find((h) => p.x >= h.x && p.x < h.x + h.w && p.y >= h.y && p.y < h.y + h.h);
      if (hit) page.setTip(hit.tip);
      else if (batchRect && p.y >= batchRect.y && p.y < batchRect.y + batchRect.h) page.setTip(`batch size ${st.batch}\n${POOL} reduction lanes ÷ ${st.batch} row${st.batch === 1 ? '' : 's'} → ${lanesFor(st.batch, s.n)} lanes each\ndrag to change it and watch the tree re-shape`);
    }
  }

  page.probe = { view: 'reduce', ulps: Math.abs(s.ulps), diffBit: s.diffBit, lanes: s.lanes, invariant: !!st.invariant };
  let out = `n = ${s.n}   batch = ${st.batch}   lanes = ${s.lanes}   preset ${st.preset}${st.invariant ? '   [batch-invariant kernel]' : ''}    tier:${r.name}\n`;
  out += `sequential sum = ${exact(s.seq.sum)}   bits ${bitStr(s.seq.sum)}\n`;
  out += `split sum      = ${exact(s.spl.sum)}   bits ${bitStr(s.spl.sum)}\n`;
  out += s.diffBit < 0
    ? 'identical: 0 representable steps apart, no differing bit.'
    : `${Math.abs(s.ulps)} representable steps apart; first differing bit ${s.diffBit} of 32 (0 = sign, 1-8 = exponent, 9-31 = mantissa).`;
  out += step ? `\n${step.label}` : '\n(press ▶ or scrub · drag the batch strip · drag a value cell · hover any node)';
  page.setReadout(out);
}

// ---- argmax view ----------------------------------------------------------
function drawArgmax(page) {
  const ctx = page.ctx, r = page.renderer, st = page.state, W = page.W, H = page.H;
  const s = scene(st);
  nodeHits = [];
  r.clear(T.n0);
  const ref = s.seq.sum;
  const la = logitsFor(s.seq.sum, ref, st.rival), lb = logitsFor(s.spl.sum, ref, st.rival);
  const pa = softmax(la), pb = softmax(lb);
  const ia = argmax(la), ib = argmax(lb);

  ctx.save(); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = T.n14; ctx.font = 'bold 12px ui-monospace, monospace';
  ctx.fillText('greedy decoding: take the largest logit', PAD, 15);
  ctx.fillStyle = T.n11; ctx.font = '10.5px ui-monospace, monospace';
  ctx.fillText(ellipsize(ctx, `logit ⟨alpha⟩ is the reduction from the other view. The runner-up sits ${st.rival} representable step${Math.abs(st.rival) === 1 ? '' : 's'} from the sequential sum — inside the gap the reduction order moves.`, W - 2 * PAD), PAD, 29);
  ctx.restore();

  const top = 62, rowH = Math.min(64, Math.max(34, (H - top - 116) / 4));
  const colW = (W - 2 * PAD - 150) / 2;
  ctx.save(); ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
  ctx.fillStyle = T.n11;
  ctx.fillText('sequential order', PAD + 150, top - 10);
  ctx.fillText(`split order, ${s.lanes} lanes (batch ${st.batch})`, PAD + 150 + colW + 10, top - 10);
  for (let k = 0; k < 4; k++) {
    const y = top + k * rowH;
    ctx.fillStyle = T.n13; ctx.font = (k === ia || k === ib ? 'bold ' : '') + '11px ui-monospace, monospace';
    ctx.fillText(`⟨${TOKENS[k]}⟩`, PAD, y + rowH / 2);
    const draw = (x, logit, prob, win, other) => {
      const bw = Math.max(2, colW * prob);
      ctx.fillStyle = alphaOf(win ? T.accent : T.n9, win ? 0.85 : 0.45);
      ctx.fillRect(x, y + 6, bw, rowH - 20);
      ctx.fillStyle = T.n12; ctx.font = '9.5px ui-monospace, monospace';
      ctx.fillText(`p = ${prob.toFixed(9)}`, x + 4, y + rowH / 2);
      ctx.fillStyle = win ? T.accent : T.n10; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(`logit ${exact(logit)}${win ? '   ← argmax' : ''}`, x, y + rowH - 6);
      nodeHits.push({ x, y, w: colW, h: rowH, tip: `⟨${TOKENS[k]}⟩\nlogit = ${exact(logit)}\nbits ${bitStr(logit)}\np = ${prob}\n${other == null ? '' : `distance to the other arm: ${ulpDist(other, logit)} representable steps`}` });
    };
    draw(PAD + 150, la[k], pa[k], k === ia, lb[k]);
    draw(PAD + 150 + colW + 10, lb[k], pb[k], k === ib, la[k]);
  }
  ctx.restore();

  // verdict
  const vy = top + 4 * rowH + 18;
  ctx.save(); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  const flipped = ia !== ib;
  ctx.fillStyle = flipped ? alphaOf(T.warn, 0.16) : alphaOf(T.ok, 0.14);
  ctx.fillRect(PAD, vy - 16, W - 2 * PAD, 34);
  ctx.fillStyle = flipped ? T.warnDeep : T.okDeep; ctx.font = 'bold 12px ui-monospace, monospace';
  ctx.fillText(flipped
    ? `TOKEN FLIPPED:  alone → ⟨${TOKENS[ia]}⟩,  batched with ${st.batch - 1} other${st.batch === 2 ? '' : 's'} → ⟨${TOKENS[ib]}⟩`
    : `same token both ways: ⟨${TOKENS[ia]}⟩ — at this batch size the reduction gap does not cross the runner-up`, PAD + 8, vy);
  ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('same prompt · same weights · same seed · temperature 0 — the only thing that changed is who else was in the batch', PAD + 8, vy + 22);
  ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = T.n10;
  ctx.fillText(ellipsize(ctx, `Drag the batch: split sits ${ulpDist(ref, s.spl.sum)} steps from sequential at ${s.lanes} lanes; the runner-up is at +${st.rival}.`, W - 2 * PAD - 8), PAD + 8, vy + 46);
  ctx.fillText(ellipsize(ctx, 'The token turns over wherever those two cross. Greedy decoding is a threshold, and this one reads the moved bits.', W - 2 * PAD - 8), PAD + 8, vy + 62);
  ctx.restore();

  const p = page.pointer;
  if (p.over) { const hit = nodeHits.find((h) => p.x >= h.x && p.x < h.x + h.w && p.y >= h.y && p.y < h.y + h.h); if (hit) page.setTip(hit.tip); }

  page.probe = { view: 'argmax', flipped, ia, ib, ulps: Math.abs(s.ulps) };
  let out = `sequential logit(alpha) = ${exact(s.seq.sum)}    split logit(alpha) = ${exact(s.spl.sum)}    runner-up ⟨beta⟩ = ${exact(la[1])} (+${st.rival} steps)\n`;
  out += `argmax sequential = ⟨${TOKENS[ia]}⟩   ·   argmax split (${s.lanes} lanes, batch ${st.batch}) = ⟨${TOKENS[ib]}⟩   ·   ${flipped ? 'FLIPPED' : 'same token'}    tier:${r.name}\n`;
  out += `top-2 probabilities differ by ${Math.abs(pa[0] - pa[1]).toExponential(3)} (sequential) and ${Math.abs(pb[0] - pb[1]).toExponential(3)} (split) — a threshold on numbers this close is decided by the last bits.`;
  page.setReadout(out);
}

// ---- cost view ------------------------------------------------------------
const SWEEP = [1, 2, 4, 8, 16, 32];
function drawCost(page) {
  const ctx = page.ctx, r = page.renderer, st = page.state, W = page.W, H = page.H;
  const vals = valuesFor(st), n = vals.length, ref = seqTrace(vals).sum;
  nodeHits = []; costRows = [];
  r.clear(T.n0);

  ctx.save(); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = T.n14; ctx.font = 'bold 12px ui-monospace, monospace';
  ctx.fillText('what a fixed split costs: lanes left idle, and an answer that stops moving', PAD, 15);
  ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(ellipsize(ctx, `${POOL} reduction lanes in the pool. The adaptive kernel spends them all; the batch-invariant kernel always takes ${FIXED_LANES} per row, whatever the batch.`, W - 2 * PAD), PAD, 29);
  ctx.restore();

  const top = 56, rowH = Math.min(52, Math.max(24, (H - top - 96) / SWEEP.length));
  // Proportional columns: at a narrow canvas a fixed pixel grid ran the "steps"
  // header straight into the bar header.
  const cName = PAD, cLanes = PAD + W * 0.07, cSum = PAD + W * 0.20, cUlp = PAD + W * 0.44;
  const barX = PAD + W * 0.60, barW = Math.max(50, W - PAD - barX - 62);
  ctx.save(); ctx.textBaseline = 'middle'; ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = T.n11;
  ctx.fillText('batch', cName, top - 12);
  ctx.fillText('lanes/row', cLanes, top - 12);
  ctx.fillText('float32 sum', cSum, top - 12);
  ctx.fillText('Δ steps', cUlp, top - 12);
  ctx.fillText('lane pool (adaptive ▸ / fixed ▸)', barX, top - 12);

  for (let k = 0; k < SWEEP.length; k++) {
    const B = SWEEP[k], y = top + k * rowH, cur = B === (st.batch | 0);
    const laneA = lanesFor(B, n), laneI = Math.min(n, FIXED_LANES);
    const lane = st.invariant ? laneI : laneA;
    const sum = splitTrace(vals, lane).sum;
    const d = ulpDist(ref, sum);
    costRows.push({ y, h: rowH, batch: B });
    if (cur) { ctx.fillStyle = alphaOf(T.accent, 0.12); ctx.fillRect(PAD - 6, y, W - 2 * PAD + 12, rowH - 4); }
    ctx.fillStyle = cur ? T.n14 : T.n12; ctx.font = (cur ? 'bold ' : '') + '11px ui-monospace, monospace';
    ctx.fillText(String(B), cName, y + rowH / 2);
    ctx.font = '10.5px ui-monospace, monospace';
    ctx.fillStyle = st.invariant ? T.okDeep : T.n12;
    ctx.fillText(st.invariant ? `${laneI} (pinned)` : String(laneA), cLanes, y + rowH / 2);
    ctx.fillStyle = T.n12; ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(exact(sum), cSum, y + rowH / 2);
    ctx.fillStyle = d === 0 ? T.okDeep : T.warnDeep;
    ctx.fillText(d === 0 ? '0 (identical)' : `${d > 0 ? '+' : ''}${d}`, cUlp, y + rowH / 2);
    // lane-pool occupancy: lanes/row x rows, capped at the pool
    const useA = Math.min(1, (laneA * B) / POOL), useI = Math.min(1, (laneI * B) / POOL);
    ctx.fillStyle = alphaOf(T.accent, 0.55); ctx.fillRect(barX, y + 6, barW * useA, (rowH - 14) / 2 - 1);
    ctx.fillStyle = alphaOf(T.ok, 0.6); ctx.fillRect(barX, y + 6 + (rowH - 14) / 2, barW * useI, (rowH - 14) / 2 - 1);
    ctx.fillStyle = T.n11; ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(`${Math.round(useA * 100)}% / ${Math.round(useI * 100)}%`, barX + barW + 6, y + rowH / 2);
    nodeHits.push({ x: PAD - 6, y, w: W - 2 * PAD + 12, h: rowH - 4, tip: `batch ${B}\nadaptive split: ${laneA} lanes/row → ${Math.min(POOL, laneA * B)}/${POOL} of the pool busy\nfixed split: ${laneI} lanes/row → ${Math.min(POOL, laneI * B)}/${POOL} busy\nfloat32 sum ${exact(sum)}\n${d === 0 ? 'identical to the sequential sum' : `${d} representable steps from the sequential sum`}\nclick to set the batch size` });
  }
  ctx.restore();

  const ny = top + SWEEP.length * rowH + 14;
  ctx.save(); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = T.n11;
  const lines = [
    'Δ steps = how far this kernel\'s float32 sum sits from the sequential one, counted in representable float32 steps.',
    'The idle lanes are arithmetic from the lane counts actually reduced with: a fixed split cannot widen when the batch is small.',
    'The real throughput cost of batch-invariant kernels is NOT measured here, and is not measurable from a web page. It is',
    'reported as a real but workable slowdown from batch-invariant attention and matmul kernels by the source below.',
    'What invariance buys is the Δ column being CONSTANT: the answer stops depending on who else was in the batch.',
    'It does not become the sequential answer — it becomes ONE answer. Source: Thinking Machines, "Defeating',
    'Nondeterminism in LLM Inference" (2025-09), thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/',
  ];
  lines.forEach((t, i) => ctx.fillText(ellipsize(ctx, t, W - 2 * PAD), PAD, ny + i * 13));
  ctx.restore();

  const p = page.pointer;
  if (p.over) { const hit = nodeHits.find((h) => p.x >= h.x && p.x < h.x + h.w && p.y >= h.y && p.y < h.y + h.h); if (hit) page.setTip(hit.tip); }

  const spread = SWEEP.map((B) => ulpDist(ref, splitTrace(vals, st.invariant ? Math.min(n, FIXED_LANES) : lanesFor(B, n)).sum));
  const distinct = new Set(spread).size;
  page.probe = { view: 'cost', distinct, invariant: !!st.invariant };
  page.setReadout(
    `sequential reference = ${exact(ref)}    tier:${r.name}\n`
    + `across batch ${SWEEP.join('/')} the ${st.invariant ? 'batch-invariant' : 'shape-adaptive'} kernel produced ${distinct} distinct answer${distinct === 1 ? '' : 's'} (steps from sequential: ${spread.join(', ')}).\n`
    + 'Throughput cost is cited, never measured here — see the note on the canvas.',
  );
}

// ---- mount ----------------------------------------------------------------
mount({
  mount: 'body',
  title: 'batch-invariance — your reply depends on who else was batched with you',
  blurb: 'Floating-point addition is not associative: (a+b)+c and a+(b+c) round differently, so the ORDER of a sum changes its last bits. A GPU reduction kernel picks its order from the shape of the launch — how many parallel lanes each row gets — and that shape depends on how many requests are in flight. This page sums one real float32 row two ways, in the page: left-to-right with one accumulator, and split across lanes then combined in a tree, with the split chosen from the batch size. Nothing here is a stored "difference" — the two sums, their bit patterns, the first differing bit and the token flip are all computed live. Drag the batch strip to re-shape the tree, drag a value to make one huge and one tiny, hover any node for its exact arithmetic, and switch on the batch-invariant kernel to pin the split and watch the answer stop moving.',
  prefer: 'canvas2d',
  aspect: '8 / 5',
  autoplay: true,
  compare: { key: 'batch', a: 8, b: 16, rebuild: true, labelA: 'batch 8 — 8 lanes per row', labelB: 'batch 16 — 4 lanes per row' },
  challenges: [
    {
      goal: 'Find a batch size where the split reduction lands on EXACTLY the sequential answer (0 representable steps apart).',
      hint: 'drag the batch strip; some splits happen to re-round back onto the same float32. The "all ones" preset does it at every batch size — because there is no rounding to lose.',
      check: (api) => ({ solved: api.probe.view === 'reduce' && api.probe.ulps === 0, detail: api.probe.view === 'reduce' ? `${api.probe.ulps} steps apart` : 'switch to the reduction view' }),
    },
    {
      goal: 'Make the greedy token flip: same values, same runner-up, different batch size.',
      hint: 'set view = greedy argmax, then drag the batch strip. The flip happens when the reduction gap crosses the runner-up — try batch 8 against batch 16.',
      check: (api) => ({ solved: api.probe.view === 'argmax' && !!api.probe.flipped, detail: api.probe.view === 'argmax' ? (api.probe.flipped ? 'flipped' : 'same token — keep dragging') : 'switch to the argmax view' }),
    },
    {
      goal: 'Make the answer stop depending on the batch: one distinct result across the whole batch sweep.',
      hint: 'set view = the cost of invariance and switch on the batch-invariant kernel — a pinned split gives one answer at every batch size (a different answer from the sequential one, but the SAME one every time).',
      check: (api) => ({ solved: api.probe.view === 'cost' && api.probe.distinct === 1, detail: api.probe.view === 'cost' ? `${api.probe.distinct} distinct answers across the sweep` : 'switch to the cost view' }),
    },
  ],
  controls: (c, page) => {
    c.select('view', { label: 'view', value: 'reduce', rebuild: true, options: [
      { value: 'reduce', label: 'the reduction (bits)' },
      { value: 'argmax', label: 'greedy argmax (the flip)' },
      { value: 'cost', label: 'the cost of invariance' },
    ] });
    c.slider('batch', { label: 'batch size (requests in flight)', min: 1, max: 32, step: 1, value: 8, rebuild: true, format: (v) => String(v | 0) });
    c.toggle('invariant', { label: 'batch-invariant kernel (fixed split)', value: false, rebuild: true });
    c.select('preset', { label: 'row of values', value: 'mixed', rebuild: true, options: PRESETS, onInput: () => page.controls.set('edit', '', { rebuild: true, silent: true }) });
    c.stepper('n', { label: 'row length', min: 4, max: 64, step: 4, value: 32, rebuild: true });
    c.slider('rival', { label: 'runner-up logit, in float32 steps', min: -8, max: 48, step: 1, value: 10, format: (v) => String(v | 0) });
    c.text('edit', { label: 'edited values  i:v,…', value: '', placeholder: 'e.g. 3:900,7:0.0001', rebuild: true });
    c.transport({ compute: () => buildSteps(page.state), speed: 7, loop: true });
  },
  // Direct manipulation: drag the batch strip horizontally, drag a value cell
  // vertically (exponentially, so one drag can span decades — which is exactly
  // when cancellation bites).
  onPointer: (page, ev) => {
    if (page.state.view === 'cost') {
      if (ev.type !== 'down') return;
      const row = costRows.find((c) => ev.y >= c.y && ev.y < c.y + c.h);
      if (row) page.controls.set('batch', row.batch, { rebuild: true });
      return;
    }
    if (page.state.view !== 'reduce') return;
    if (ev.type === 'down' || (ev.type === 'move' && page.pointer.down)) {
      if (batchRect && ev.y >= batchRect.y - 6 && ev.y < batchRect.y + batchRect.h + 6) {
        const t = (ev.x - batchRect.x) / batchRect.w;
        const b = Math.max(1, Math.min(32, Math.round(t * 32)));
        if (b !== (page.state.batch | 0)) page.controls.set('batch', b, { rebuild: true });
        return;
      }
    }
    if (ev.type === 'move' && page.pointer.down && valRect) {
      const hit = cellAt(valRect, 1, Math.max(2, page.state.n | 0), ev.x, ev.y);
      if (!hit) return;
      const vals = valuesFor(page.state), i = hit.c;
      const mag = Math.max(1e-9, Math.abs(vals[i])) * Math.exp(-ev.dy * 0.03);
      const nv = fr((vals[i] < 0 ? -1 : 1) * Math.min(1e9, mag));
      const ed = parseEdits(page.state.edit); ed[i] = nv;
      page.controls.set('edit', editStr(ed), { rebuild: true });
    }
  },
  draw: (page) => {
    const v = page.state.view;
    if (v === 'argmax') { drawArgmax(page); return; }
    if (v === 'cost') { drawCost(page); return; }
    drawReduce(page);
  },
}).then((page) => {
  window.__batchInvariancePage = page;
  // The numeric core, exposed read-only so an independent implementation (numpy
  // float32, or C) can be pointed at the same row and check that the two sums
  // and the differing bit agree. Nothing in the page reads back through it.
  window.__batchInvarianceNum = { valuesFor, seqTrace, splitTrace, lanesFor, lanesUsed, bitsOf, bitStr, firstDiffBit, ulpDist, addUlps, logitsFor, softmax, argmax, POOL, FIXED_LANES };

  const q = Q;
  const t = page.controls._transport;
  // Deep links / headless hooks: every control is restorable, so a copied link
  // and a --screenshot run reproduce the same frame.
  if (['reduce', 'argmax', 'cost'].includes(q.get('view'))) page.controls.set('view', q.get('view'), { rebuild: true });
  if (q.has('preset')) page.controls.set('preset', q.get('preset'), { rebuild: true });
  if (q.has('n')) page.controls.set('n', Math.max(4, Math.min(64, +q.get('n') || 32)), { rebuild: true });
  if (q.has('batch')) page.controls.set('batch', Math.max(1, Math.min(32, +q.get('batch') || 8)), { rebuild: true });
  if (q.has('invariant')) page.controls.set('invariant', q.get('invariant') === '1' || q.get('invariant') === 'true', { rebuild: true });
  if (q.has('rival')) page.controls.set('rival', Math.max(-8, Math.min(48, +q.get('rival') || 0)), { rebuild: true });
  // ?edit=i:v,i:v -- the headless stand-in for dragging a value cell.
  if (q.has('edit')) page.controls.set('edit', q.get('edit'), { rebuild: true });
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  // Any capture hook pauses, so the frame is deterministic.
  if (t && (q.has('step') || q.has('hover') || q.get('view') === 'argmax' || q.get('view') === 'cost')) t.pause();
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
