// dtype-bits concept page -- fp/int bit layouts + bit-reveal reconstruction,
// plus a hardware dtype-support matrix (which vendors/architectures accept each
// dtype as a native matrix-engine operand).
// Uses the verified framework: layout.mount() + controls + a step Transport
// over a page-built reveal sequence. IEEE-style decode is page-specific, so it
// lives here (not tensor.js); the transport just consumes the step array.
//
// Interactive per the shared render framework's contract: CLICK any bit to
// flip it and watch the reconstructed floating value rebuild bit by bit (the
// dtype "aha"); HOVER a bit to see its field (sign/exponent/mantissa) and place
// value / contribution, or hover a field label to decode the whole value =
// (-1)^s · 2^(e-bias) · 1.m; scrub (or let it play) to reveal bits left→right.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';
import { T, effectiveTheme, alphaOf } from '../framework/theme.js';


const FIELD = { sign: [214, 39, 40], exp: [31, 119, 180], mant: [44, 160, 44], mag: [44, 160, 44] };
// Thin alias over the shared helper, kept only because the call sites read
// better as rgb(categoricalColour): alphaOf() now takes an [r,g,b] triple.
const rgb = (c, a = 1) => alphaOf(c, a);
// FIELD (above) and LEVELS (below) are [r,g,b] ARRAYS, so the hex codemod could
// not see them and they are still the light-mode tab10 values. They read fine on
// white, but on the dark ground they are too dark for the dark-on-fill glyph
// (`T.n0` inverts), so dark mode maps each one to its semantic token instead.
// Light is untouched -- it still paints the exact arrays it always did.
const FIELD_DARK = { sign: 'bad', exp: 'accent', mant: 'ok', mag: 'ok' };
// Adds over a bare theme.js `alphaOf`: the light arm keeps the EXACT tab10
// [r,g,b] array, so only dark mode is re-pointed at a semantic token.
const fieldFill = (name, a = 1) => (effectiveTheme() === 'dark' ? alphaOf(FIELD_DARK[name], a) : rgb(FIELD[name], a));
const levelFill = (lv, a = 1) => (effectiveTheme() === 'dark' ? alphaOf(lv.dark, a) : rgb(lv.c, a));

const DTYPES = [
  { key: 'fp32',    label: 'fp32 (E8M23)', kind: 'float', E: 8,  M: 23, bias: 127, bits: 32 },
  { key: 'fp16',    label: 'fp16 (E5M10)', kind: 'float', E: 5,  M: 10, bias: 15,  bits: 16 },
  { key: 'bf16',    label: 'bf16 (E8M7)',  kind: 'float', E: 8,  M: 7,  bias: 127, bits: 16 },
  { key: 'fp8e4m3', label: 'fp8 e4m3',     kind: 'float', E: 4,  M: 3,  bias: 7,   bits: 8  },
  { key: 'fp8e5m2', label: 'fp8 e5m2',     kind: 'float', E: 5,  M: 2,  bias: 15,  bits: 8  },
  { key: 'int8',    label: 'int8 (×scale)', kind: 'int',  bits: 8, range: 8 },
  { key: 'int4',    label: 'int4 (×scale)', kind: 'int',  bits: 4, range: 8 },
];

// ---- hardware dtype-support matrix ----------------------------------------
// The vendor/architecture capability catalogue lives in ../data/dtype-support.json
// (schema in ../data/README.md) and is fetched at runtime -- the page never
// hard-codes a capability. If the fetch fails (opened as file://, no server) the
// hardware view SAYS SO rather than inventing rows: an unknown renders as
// unknown, never as a guess. Same rule inside the data: a cell the vendor does
// not publish is `unknown`, which is a distinct colour from `none`.
async function loadJSON(path) {
  try { const r = await fetch(path, { cache: 'no-cache' }); if (!r.ok) return null; return await r.json(); } catch (_) { return null; }
}
const HW = await loadJSON('../data/dtype-support.json');

// Support levels, in the order the legend prints them.
const LEVELS = {
  native:   { c: [ 44, 160,  44], dark: 'ok',     glyph: '●', short: 'native matrix-engine operand' },
  vector:   { c: [ 31, 119, 180], dark: 'accent', glyph: '◐', short: 'vector / scalar ALU only (no matrix engine)' },
  emulated: { c: [230, 145,  30], dark: 'warn',   glyph: '○', short: 'emulated — upconverted to a wider native type' },
  none:     { c: [176, 184, 192], dark: 'n8',     glyph: '·', short: 'not supported' },
  unknown:  { c: [148, 103, 189], dark: 'violet', glyph: '?', short: 'not publicly documented' },
};
const LEVEL_ORDER = ['native', 'vector', 'emulated', 'none', 'unknown'];
const ENGINE_CLASSES = [
  { value: 'all', label: 'all engines' },
  { value: 'gpu-matrix', label: 'GPU matrix engines' },
  { value: 'gpu-alu', label: 'GPU vector ALUs' },
  { value: 'npu', label: 'NPUs' },
  { value: 'dsp', label: 'audio DSPs' },
  { value: 'cpu-simd', label: 'CPU SIMD' },
];

const hwCell = (a, key) => (a && a.dtypes && a.dtypes[key]) || { level: 'unknown' };
function hwArchs(state) {
  if (!HW || !Array.isArray(HW.architectures)) return [];
  const k = state.engines || 'all';
  return HW.architectures.filter((a) => k === 'all' || a.class === k);
}
const nativeList = (a) => DTYPES.filter((d) => hwCell(a, d.key).level === 'native').map((d) => d.key).join(' ');
// One reveal step per architecture row -- the transport walks the matrix
// top→bottom exactly the way it walks the bit row in the bit-layout view.
function buildHwSteps(state) {
  return hwArchs(state).map((a, i) => ({
    rev: i + 1, partial: 0,
    label: `${a.vendor} ${a.arch} — ${a.engine}: native ${nativeList(a) || '(none of the taught dtypes)'}`,
  }));
}

function fmt(x) {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e4 || a < 1e-3) return x.toExponential(2);
  return String(Number(x.toPrecision(4)));
}

// ---- encode / decode -------------------------------------------------------
function decodeFloat(sign, ef, mant, M, bias) {
  const v = ef === 0 ? Math.pow(2, 1 - bias) * (mant / (1 << M)) : Math.pow(2, ef - bias) * (1 + mant / (1 << M));
  return (sign ? -1 : 1) * v;
}
function encodeFloat(v, E, M, bias) {
  const sign = (v < 0 || Object.is(v, -0)) ? 1 : 0;
  const a = Math.abs(v), maxEf = (1 << E) - 1;
  let ef = 0, mant = 0;
  if (a !== 0) {
    let eu = Math.floor(Math.log2(a));
    ef = eu + bias;
    if (ef <= 0) {                                   // subnormal
      mant = Math.round((a / Math.pow(2, 1 - bias)) * (1 << M)); ef = 0;
      if (mant >= (1 << M)) { mant -= (1 << M); ef = 1; }
    } else if (ef >= maxEf) {                         // clamp to max normal (no inf)
      ef = maxEf - 1; mant = (1 << M) - 1;
    } else {
      mant = Math.round((a / Math.pow(2, eu) - 1) * (1 << M));
      if (mant >= (1 << M)) { mant = 0; ef++; if (ef >= maxEf) { ef = maxEf - 1; mant = (1 << M) - 1; } }
    }
  }
  return { kind: 'float', sign, ef, mant };
}
function encodeInt(v, N, range) {
  const qmax = (1 << (N - 1)) - 1, qmin = -(1 << (N - 1)), scale = range / qmax;
  let code = Math.max(qmin, Math.min(qmax, Math.round(v / scale)));
  return { kind: 'int', code, scale };
}
const encode = (dt, v) => (dt.kind === 'float' ? encodeFloat(v, dt.E, dt.M, dt.bias) : encodeInt(v, dt.bits, dt.range));

// Build the per-bit cell list { v, field } from a raw {sign,ef,mant} / {code}.
function bitcellsOf(dt, raw) {
  if (dt.kind === 'float') {
    const cells = [{ v: raw.sign, field: 'sign' }];
    for (let g = 0; g < dt.E; g++) cells.push({ v: (raw.ef >> (dt.E - 1 - g)) & 1, field: 'exp' });
    for (let g = 0; g < dt.M; g++) cells.push({ v: (raw.mant >> (dt.M - 1 - g)) & 1, field: 'mant' });
    return cells;
  }
  const N = dt.bits, pattern = raw.code & ((1 << N) - 1), cells = [];
  for (let j = 0; j < N; j++) cells.push({ v: (pattern >> (N - 1 - j)) & 1, field: j === 0 ? 'sign' : 'mag' });
  return cells;
}

// Decode a {sign,ef,mant}/{code} raw back to a floating value.
function decodeRaw(dt, raw) {
  if (dt.kind === 'float') return decodeFloat(raw.sign, raw.ef, raw.mant, dt.M, dt.bias);
  return raw.code * raw.scale;
}

// ---- live editable bit state ----------------------------------------------
// cur holds the CURRENT bit pattern, decoupled from the value slider once the
// user starts flipping bits, so a flip survives redraws and rebuilds the value.
// rebuild (dtype/value change) re-encodes the slider value into cur; a flip
// toggles one bit in cur and recomputes the decoded value live.
let cur = { dt: null, raw: null };
let bitRowRect = null;   // tight {x,y,w,h} around the N bit cells, captured in draw for hit-testing
let groupRects = [];     // [{name,x,y,w,h}] field-label bands above the cells, for label hover

function syncCur(dt, value) {
  cur = { dt, raw: encode(dt, value) };
}

// Toggle bit index i (0 = sign, then exponent MSB→LSB, then mantissa MSB→LSB;
// for ints: sign then magnitude MSB→LSB) and recompute cur.raw in place.
function flipBit(i) {
  const dt = cur.dt, raw = cur.raw; if (!dt) return;
  if (dt.kind === 'float') {
    if (i === 0) raw.sign ^= 1;
    else if (i <= dt.E) raw.ef ^= (1 << (dt.E - i));               // exp bit (i-1) from MSB
    else raw.mant ^= (1 << (dt.M - (i - dt.E)));                   // mant bit from MSB
  } else {
    const N = dt.bits;
    let pattern = raw.code & ((1 << N) - 1);
    pattern ^= (1 << (N - 1 - i));
    // re-interpret as two's-complement signed code
    raw.code = (pattern & (1 << (N - 1))) ? pattern - (1 << N) : pattern;
  }
}

// ---- step-reveal sequence (left -> right, value rebuilds) ------------------
// Built from cur.raw so the reveal reflects any flipped bits, not just the
// slider value.
function buildSteps() {
  const dt = cur.dt, raw = cur.raw, decoded = decodeRaw(dt, raw), steps = [];
  if (dt.kind === 'float') {
    const sgn = raw.sign ? -1 : 1, sub = raw.ef === 0;
    const scale = sub ? Math.pow(2, 1 - dt.bias) : Math.pow(2, raw.ef - dt.bias), implicit = sub ? 0 : 1;
    steps.push({ rev: 1, partial: 0, label: `sign bit = ${raw.sign} → ${raw.sign ? '−' : '+'}` });
    let partial = sgn * scale * implicit;
    steps.push({ rev: 1 + dt.E, partial, label: sub
      ? `exponent = 0 (subnormal) → scale 2^(1−${dt.bias}) = ${fmt(scale)}, no implicit 1 → ${fmt(partial)}`
      : `exponent = ${raw.ef} → 2^(${raw.ef}−${dt.bias}) = ${fmt(scale)}, ×(1 + mantissa) → ${fmt(partial)}` });
    for (let i = 1; i <= dt.M; i++) {
      const bit = (raw.mant >> (dt.M - i)) & 1;
      if (bit) partial += sgn * scale * Math.pow(2, -i);
      steps.push({ rev: 1 + dt.E + i, partial, label: `mantissa bit ${i} = ${bit}${bit ? ` → +${fmt(sgn * scale * Math.pow(2, -i))}` : ''} → ${fmt(partial)}` });
    }
  } else {
    const N = dt.bits, pattern = raw.code & ((1 << N) - 1);
    let partial = 0;
    for (let j = 0; j < N; j++) {
      const bit = (pattern >> (N - 1 - j)) & 1, place = j === 0 ? -(1 << (N - 1)) : (1 << (N - 1 - j));
      partial += bit * place * raw.scale;
      steps.push({ rev: j + 1, partial, label: j === 0
        ? `sign bit = ${bit} → ${bit ? `−${1 << (N - 1)}` : '0'} ×${fmt(raw.scale)} → ${fmt(partial)}`
        : `bit ${j} = ${bit}${bit ? ` → +${1 << (N - 1 - j)}×${fmt(raw.scale)}` : ''} → ${fmt(partial)}` });
    }
  }
  return steps.map((s) => ({ ...s, decoded }));
}

// Rebuild the transport's step list from cur (after a flip) so the scrub axis
// + reveal labels track the edited bits without regenerating from the slider.
function resyncTransport(page) {
  const t = page.controls._transport;
  if (!t) return;
  t.steps = buildSteps();
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  // A pending {rebuild:true} from a control set() would re-run compute() on the
  // next redraw, and compute() re-encodes the SLIDER value into cur -- silently
  // undoing the flips we just applied. The steps are current as of this call,
  // so clear the flag rather than let it clobber them (`?dtype=fp16&flip=0`
  // read back POSITIVE before this, while `?flip=0` alone worked).
  t._dirty = false;
  t._sync();
}

// ---- per-bit field metadata (for hover-to-inspect) -------------------------
// Returns {field, line} describing what bit index i contributes.
function bitInfo(dt, i) {
  if (dt.kind === 'float') {
    if (i === 0) return { field: 'sign', line: `sign bit → (−1)^s` };
    if (i <= dt.E) { const w = dt.E - i; return { field: 'exponent', line: `exponent bit ${i - 1} (weight 2^${w} = ${1 << w})` }; }
    const mi = i - dt.E;                                    // mantissa bit 1..M
    const pv = Math.pow(2, -mi);
    return { field: 'mantissa', line: `mantissa bit ${mi} → 2^-${mi} = ${fmt(pv)}` };
  }
  const N = dt.bits;
  if (i === 0) return { field: 'sign', line: `sign bit → −2^${N - 1} = −${1 << (N - 1)} (×scale)` };
  const w = N - 1 - i;
  return { field: 'magnitude', line: `bit ${i} → weight 2^${w} = ${1 << w} (×scale)` };
}

// ---- drawing ---------------------------------------------------------------
function drawBitRow(page, dt, bitcells, rev, activeIdx, rect) {
  const ctx = page.ctx, N = bitcells.length;
  const cw = Math.max(12, Math.min(54, rect.w / N)), ch = 44;
  const x0 = rect.x + (rect.w - N * cw) / 2, y0 = rect.y;
  // Capture the tight bit-row rect for pointer hit-testing (cellAt: 1 row × N cols).
  bitRowRect = { x: x0, y: y0, w: N * cw, h: ch };
  // group brackets / labels
  const groups = dt.kind === 'float'
    ? [['sign', 0, 1], ['exponent', 1, 1 + dt.E], ['mantissa', 1 + dt.E, N]]
    : [['sign', 0, 1], ['magnitude (×scale)', 1, N]];
  groupRects = [];
  ctx.save();
  ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
  for (const [name, lo, hi] of groups) {
    const xa = x0 + lo * cw, xb = x0 + hi * cw, xm = (xa + xb) / 2;
    ctx.fillStyle = T.n11; ctx.fillText(name, xm, y0 - 8);
    ctx.strokeStyle = T.n7; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xa + 2, y0 - 4); ctx.lineTo(xb - 2, y0 - 4); ctx.stroke();
    // label band sits above the cells; hover here decodes the whole field.
    groupRects.push({ name, x: xa, y: y0 - 20, w: xb - xa, h: 18 });
  }
  ctx.textBaseline = 'middle';
  for (let i = 0; i < N; i++) {
    const c = bitcells[i], shown = i < rev, x = x0 + i * cw;
    ctx.fillStyle = shown ? fieldFill(c.field, c.v ? 0.92 : 0.20) : T.n3;
    ctx.fillRect(x, y0, cw - 1.5, ch);
    ctx.fillStyle = shown ? (c.v ? T.n0 : T.n12) : T.n8;
    ctx.font = (cw < 18 ? '10px' : '13px') + ' ui-monospace, monospace';
    ctx.fillText(shown ? String(c.v) : '·', x + (cw - 1.5) / 2, y0 + ch / 2);
    if (i === activeIdx) { ctx.strokeStyle = T.n14; ctx.lineWidth = 2.5; ctx.strokeRect(x + 1, y0 + 1, cw - 3.5, ch - 2); }
  }
  ctx.restore();
}

function drawTable(page, value, focusKey, rect) {
  const r = page.renderer, ctx = page.ctx;
  const rows = DTYPES.map((dt) => { const e = encode(dt, value); const d = decodeRaw(dt, e); return { dt, decoded: d, err: Math.abs(value - d) }; });
  const maxErr = Math.max(1e-12, ...rows.map((e) => e.err));
  const colN = rect.x, colB = rect.x + 110, colD = rect.x + 175, colBar = rect.x + 250, barMax = Math.min(150, rect.w - 320), colE = colBar + barMax + 8;
  r.label('dtype', colN, rect.y, { color: T.n11, font: '11px ui-monospace, monospace' });
  r.label('bits', colB, rect.y, { color: T.n11, font: '11px ui-monospace, monospace' });
  r.label('decoded', colD, rect.y, { color: T.n11, font: '11px ui-monospace, monospace' });
  r.label('|error|  (bar ∝ across dtypes)', colBar, rect.y, { color: T.n11, font: '11px ui-monospace, monospace' });
  let y = rect.y + 20;
  for (const e of rows) {
    const foc = e.dt.key === focusKey;
    r.label(e.dt.label, colN, y, { color: foc ? T.n14 : T.n12, font: (foc ? 'bold ' : '') + '12px ui-monospace, monospace' });
    r.label(String(e.dt.bits), colB, y, { color: T.n12, font: '12px ui-monospace, monospace' });
    r.label(fmt(e.decoded), colD, y, { color: T.n12, font: '12px ui-monospace, monospace' });
    const bw = (e.err / maxErr) * barMax;
    ctx.fillStyle = fieldFill('sign', foc ? 0.85 : 0.5); ctx.fillRect(colBar, y - 9, Math.max(1, bw), 12);
    r.label(fmt(e.err), colE, y, { color: T.n12, font: '11px ui-monospace, monospace' });
    if (foc) { ctx.strokeStyle = alphaOf(T.accent, 0.5); ctx.lineWidth = 1; ctx.strokeRect(colN - 6, y - 15, rect.w - 6, 21); }
    y += 22;
  }
}

// Trim a string to fit `maxW` px in the context's CURRENT font, adding an
// ellipsis when it does not. Measured, not character-counted -- the row labels
// mix long vendor names with long engine names and must never overlap.
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

// ---- hardware matrix drawing ----------------------------------------------
// Layout state captured in draw so the pointer handler can hit-test the same
// rects: the scrollable row viewport, the clickable column headers, and the
// visible row list.
let hwScroll = 0;                 // px, vertical scroll of the architecture list
let hwView = null;                // {x,y,w,h} clipped row viewport
let hwHead = null;                // {x,y,w,h} dtype column-header band
let hwGeom = null;                // {x0,colW,rowH,archs}

function hwClampScroll() {
  if (!hwGeom || !hwView) return;
  const max = Math.max(0, hwGeom.archs.length * hwGeom.rowH - hwView.h);
  hwScroll = Math.max(0, Math.min(max, hwScroll));
}

function drawHardware(page) {
  const r = page.renderer, ctx = page.ctx, st = page.state, pad = 18;
  r.clear(T.n0);
  const archs = hwArchs(st);
  if (!archs.length) {
    const msg = HW
      ? 'no architectures match this engine-class filter'
      : 'dtype-support.json could not be loaded — serve the page over http (fetch + ES modules are blocked on file://)';
    r.label(msg, pad, 46, { color: T.bad, font: '12px ui-monospace, monospace' });
    page.probe = { view: 'hardware', focusNative: null };
    page.setReadout('hardware support matrix unavailable — no capability is guessed when the catalogue is missing.');
    return;
  }
  const s = page.step();
  const rev = s ? s.rev : archs.length;
  const nameW = Math.min(320, Math.max(200, page.W * 0.36));
  const x0 = pad + nameW, colW = (page.W - pad - x0) / DTYPES.length;
  const rowH = 21, headY = 30, top = headY + 26, legendH = 34;
  const viewH = Math.max(rowH, page.H - top - legendH - 4);
  hwHead = { x: x0, y: headY - 14, w: colW * DTYPES.length, h: 20 };
  hwView = { x: pad, y: top, w: page.W - 2 * pad, h: viewH };
  hwGeom = { x0, colW, rowH, archs };
  hwClampScroll();
  // Keep the transport's active row on screen while it plays.
  if (s) {
    const ry = (rev - 1) * rowH;
    if (ry < hwScroll) hwScroll = ry;
    else if (ry + rowH > hwScroll + viewH) hwScroll = ry + rowH - viewH;
    hwClampScroll();
  }

  // Column headers (click one to focus that dtype in the bit-layout view).
  ctx.save();
  ctx.textAlign = 'center';
  for (let j = 0; j < DTYPES.length; j++) {
    const d = DTYPES[j], foc = d.key === st.dtype, cx = x0 + j * colW + colW / 2;
    if (foc) { ctx.fillStyle = alphaOf(T.accent, 0.07); ctx.fillRect(x0 + j * colW, headY - 16, colW, top - headY + 16 + viewH); }
    ctx.fillStyle = foc ? T.n14 : T.n11;
    ctx.font = (foc ? 'bold ' : '') + '11px ui-monospace, monospace';
    ctx.fillText(d.key, cx, headY);
    ctx.fillStyle = T.n10; ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(`${d.bits}b`, cx, headY + 11);
  }
  ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, top - 4); ctx.lineTo(page.W - pad, top - 4); ctx.stroke();
  ctx.restore();

  // Scrollable rows (drag anywhere in the matrix to scroll).
  ctx.save();
  ctx.beginPath(); ctx.rect(hwView.x, hwView.y, hwView.w, hwView.h); ctx.clip();
  ctx.textBaseline = 'middle';
  for (let i = 0; i < archs.length; i++) {
    const a = archs[i], y = top + i * rowH - hwScroll;
    if (y + rowH < top - rowH || y > top + viewH + rowH) continue;
    const shown = i < rev, alpha = shown ? 1 : 0.18;
    if (i % 2 === 1) { ctx.fillStyle = T.n1; ctx.fillRect(pad, y, page.W - 2 * pad, rowH); }
    if (s && i === rev - 1) { ctx.strokeStyle = alphaOf(T.accent, 0.55); ctx.lineWidth = 1.5; ctx.strokeRect(pad + 1, y + 1, page.W - 2 * pad - 2, rowH - 2); }
    // Name column: the architecture, then its engine in the space that is left.
    // Both are ellipsized against a measured budget -- a long arch name must
    // never run under the engine label or bleed into the fp32 column.
    ctx.textAlign = 'left';
    const nameX = pad + 4, nameBudget = x0 - nameX - 8;
    ctx.font = '11px ui-monospace, monospace';
    const nameTxt = ellipsize(ctx, `${a.vendor} ${a.arch}`, nameBudget * 0.62);
    ctx.fillStyle = alphaOf(T.n14, alpha);
    ctx.fillText(nameTxt, nameX, y + rowH / 2 - 0.5);
    const usedW = ctx.measureText(nameTxt).width;
    ctx.font = '9.5px ui-monospace, monospace';
    const engX = nameX + usedW + 8, engBudget = x0 - 8 - engX;
    if (engBudget > 20) {
      ctx.fillStyle = alphaOf(T.n11, alpha);
      ctx.fillText(ellipsize(ctx, a.engine, engBudget), engX, y + rowH / 2 - 0.5);
    }
    ctx.textAlign = 'center';
    for (let j = 0; j < DTYPES.length; j++) {
      const lv = LEVELS[hwCell(a, DTYPES[j].key).level] || LEVELS.unknown;
      ctx.fillStyle = levelFill(lv, alpha * (lv === LEVELS.none ? 0.30 : 0.88));
      ctx.fillRect(x0 + j * colW + 2, y + 3, colW - 4, rowH - 6);
      ctx.fillStyle = alphaOf(T.n0, alpha); ctx.font = '11px ui-monospace, monospace';
      ctx.fillText(lv.glyph, x0 + j * colW + colW / 2, y + rowH / 2 - 0.5);
    }
  }
  ctx.restore();

  // Legend.
  ctx.save();
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.font = '10px ui-monospace, monospace';
  let lx = pad, ly = page.H - legendH / 2 - 2;
  for (const k of LEVEL_ORDER) {
    const lv = LEVELS[k];
    ctx.fillStyle = levelFill(lv, k === 'none' ? 0.30 : 0.88); ctx.fillRect(lx, ly - 6, 12, 12);
    ctx.fillStyle = T.n0; ctx.textAlign = 'center'; ctx.fillText(lv.glyph, lx + 6, ly);
    ctx.textAlign = 'left'; ctx.fillStyle = T.n12; ctx.fillText(k, lx + 16, ly);
    lx += 18 + ctx.measureText(k).width + 14;
  }
  ctx.restore();

  // Hover-to-inspect: a matrix cell -> level + why, with the public source.
  const p = page.pointer;
  if (p.over && p.x >= hwView.x && p.x < hwView.x + hwView.w && p.y >= hwView.y && p.y < hwView.y + hwView.h) {
    const i = Math.floor((p.y - top + hwScroll) / rowH);
    const j = Math.floor((p.x - x0) / colW);
    if (i >= 0 && i < archs.length) {
      const a = archs[i];
      if (j >= 0 && j < DTYPES.length) {
        const d = DTYPES[j], cell = hwCell(a, d.key), lv = LEVELS[cell.level] || LEVELS.unknown;
        let tip = `${a.vendor} ${a.arch} · ${a.engine}\n${d.label} → ${cell.level}: ${lv.short}`;
        if (cell.note) tip += `\n${cell.note}`;
        if (a.source) tip += `\nsource: ${a.source}`;
        page.setTip(tip);
      } else {
        page.setTip(`${a.vendor} ${a.arch} · ${a.engine}\nnative: ${nativeList(a) || '(none of the taught dtypes)'}\nsource: ${a.source || 'unspecified'}`);
      }
    }
  } else if (p.over && hwHead && p.y >= hwHead.y && p.y < hwHead.y + hwHead.h && p.x >= hwHead.x) {
    const j = Math.floor((p.x - x0) / colW);
    if (j >= 0 && j < DTYPES.length) page.setTip(`${DTYPES[j].label} — ${DTYPES[j].bits} bits\nclick to focus this dtype`);
  }

  // Readout: the focused dtype's tally across the visible engine classes.
  const key = st.dtype, tally = {};
  for (const k of LEVEL_ORDER) tally[k] = 0;
  for (const a of archs) { const lv = hwCell(a, key).level; if (tally[lv] == null) tally[lv] = 0; tally[lv]++; }
  page.probe = { view: 'hardware', focusNative: tally.native, focusDtype: key };
  const focusLabel = (DTYPES.find((d) => d.key === key) || {}).label || key;
  let out = `hardware support · ${archs.length} architectures · focus ${focusLabel}    tier:${r.name}\n`;
  out += LEVEL_ORDER.map((k) => `${k} ${tally[k]}`).join('   ');
  out += s ? `\n${s.label}` : '\n(drag the matrix to scroll · hover a cell for the level + public source · click a column header to focus that dtype)';
  page.setReadout(out);
}

mount({
  mount: 'body',
  title: 'dtype-bits — how a number is stored',
  blurb: 'Bit layout of fp32 / fp16 / bf16 / fp8 / int8 / int4 — and which silicon can actually multiply each one. BIT LAYOUT view: CLICK any bit to flip it and watch the reconstructed value rebuild; hover a bit for its field + place value (e.g. mantissa bit 3 → 2^-3 = 0.125), or hover a field label to decode it; scrub to reveal bits left→right. HARDWARE SUPPORT view: a vendor × dtype matrix from public vendor specs — GPU matrix engines, NPUs, audio DSPs and CPU SIMD paths accept very different dtype sets. Drag to scroll, hover a cell for the level and its source.',
  prefer: 'canvas2d',
  aspect: '8 / 5',
  autoplay: true,
  compare: { key: 'dtype', a: 'fp16', b: 'int4', rebuild: true, labelA: 'fp16 — 16-bit float', labelB: 'int4 — 4-bit integer' },
  challenges: [
    { goal: 'Zero the number — clear every bit so it reconstructs to 0.', hint: 'click each lit bit to flip it to 0 (all bits 0 = the value 0).', check: (api) => ({ solved: Math.abs(api.probe.dec ?? 9) < 1e-6, detail: `reconstructed value = ${(api.probe.dec ?? 0).toFixed(4)}` }) },
    { goal: 'Make the stored number NEGATIVE.', hint: 'flip the leftmost bit — the sign bit (works for the float dtypes).', check: (api) => ({ solved: (api.probe.dec ?? 0) < 0, detail: `value = ${(api.probe.dec ?? 0).toFixed(3)}` }) },
    { goal: 'Switch to the hardware view and focus a dtype that AT MOST 6 of the listed architectures take as a native matrix-engine operand.', hint: 'set view = hardware, then click column headers — fp8 is a recent arrival and fp32 is an accumulator almost nowhere multiplies in; int8 is the one nearly everybody has.', check: (api) => ({ solved: api.probe.view === 'hardware' && typeof api.probe.focusNative === 'number' && api.probe.focusNative <= 6, detail: api.probe.view === 'hardware' ? `${api.probe.focusDtype}: ${api.probe.focusNative} native` : 'switch view to "hardware support"' }) },
  ],
  controls: (c, page) => {
    c.select('view', { label: 'view', value: 'bits', rebuild: true, options: [
      { value: 'bits', label: 'bit layout' }, { value: 'hardware', label: 'hardware support' },
    ] });
    c.select('dtype', { label: 'focus dtype', value: 'fp16', rebuild: true, options: DTYPES.map((d) => ({ value: d.key, label: d.label })) });
    c.select('engines', { label: 'engine class (hardware view)', value: 'all', rebuild: true, options: ENGINE_CLASSES });
    c.slider('value', { label: 'value', min: -4, max: 4, step: 0.01, value: 1.3, rebuild: true, format: (v) => (+v).toFixed(2) });
    // rebuild (dtype/value change) re-encodes the slider value into cur, then
    // builds the reveal steps from cur -- so a flip-edited pattern resets to the
    // slider value only when the slider/dtype actually moves.
    c.transport({ compute: () => {
      if (page.state.view === 'hardware') return buildHwSteps(page.state);
      syncCur(DTYPES.find((d) => d.key === page.state.dtype), page.state.value); return buildSteps();
    }, speed: 5, loop: true });
  },
  // Direct manipulation. BIT LAYOUT: CLICK a bit cell to flip it; the
  // reconstructed value updates live (hit-test against the tight bit-row rect
  // captured in draw). HARDWARE: DRAG the matrix to scroll the architecture
  // list, CLICK a dtype column header to focus that dtype.
  onPointer: (page, ev) => {
    if (page.state.view === 'hardware') {
      if (ev.type === 'move' && page.pointer.down && hwGeom) { hwScroll -= ev.dy; hwClampScroll(); }
      else if (ev.type === 'down' && hwHead && hwGeom && ev.y >= hwHead.y && ev.y < hwHead.y + hwHead.h && ev.x >= hwHead.x) {
        const j = Math.floor((ev.x - hwGeom.x0) / hwGeom.colW);
        if (j >= 0 && j < DTYPES.length) page.controls.set('dtype', DTYPES[j].key, { rebuild: true });
      }
      return;
    }
    if (ev.type !== 'down' || !cur.dt) return;
    const N = bitcellsOf(cur.dt, cur.raw).length;
    const hit = bitRowRect && cellAt(bitRowRect, 1, N, ev.x, ev.y);
    if (hit) { flipBit(hit.c); resyncTransport(page); }
  },
  draw: (page) => {
    const r = page.renderer, st = page.state, pad = 18;
    if (st.view === 'hardware') { drawHardware(page); return; }
    r.clear(T.n0);
    const dt = cur.dt || DTYPES.find((d) => d.key === st.dtype);
    if (!cur.dt) syncCur(dt, st.value);
    const bitcells = bitcellsOf(dt, cur.raw), decoded = decodeRaw(dt, cur.raw);
    page.probe = { dec: decoded };
    const s = page.step();
    const rev = s ? s.rev : bitcells.length, partial = s ? s.partial : decoded, activeIdx = s ? s.rev - 1 : -1;

    drawBitRow(page, dt, bitcells, rev, activeIdx, { x: pad, y: 44, w: page.W - 2 * pad, h: 44 });
    drawTable(page, decoded, dt.key, { x: pad, y: 150, w: page.W - 2 * pad });

    // Hover-to-inspect: a bit cell -> field + place value / contribution; a
    // field label band -> decode the whole field / reconstructed value.
    if (page.pointer.over) {
      const p = page.pointer;
      const N = bitcells.length;
      const bh = bitRowRect && cellAt(bitRowRect, 1, N, p.x, p.y);
      let tip = null;
      if (bh) {
        const info = bitInfo(dt, bh.c), set = bitcells[bh.c].v;
        tip = `${info.line} (${set ? 'set' : 'clear'})\nfield: ${info.field} · click to flip`;
      } else {
        const lab = groupRects.find((g) => p.x >= g.x && p.x < g.x + g.w && p.y >= g.y && p.y < g.y + g.h);
        if (lab) {
          if (dt.kind === 'float') {
            if (lab.name === 'sign') tip = `sign s = ${cur.raw.sign} → ${cur.raw.sign ? '−' : '+'}`;
            else if (lab.name === 'exponent') tip = `exponent e = ${cur.raw.ef}, bias ${dt.bias} → 2^(e−bias) = 2^${cur.raw.ef - dt.bias} = ${fmt(Math.pow(2, cur.raw.ef - dt.bias))}`;
            else tip = `mantissa m = ${cur.raw.mant}/${1 << dt.M}\nvalue = (−1)^s · 2^(e−bias) · 1.m = ${fmt(decoded)}`;
          } else {
            if (lab.name === 'sign') tip = `sign bit → two's-complement code ${cur.raw.code}`;
            else tip = `magnitude × scale ${fmt(cur.raw.scale)} → ${fmt(decoded)}`;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    const err = st.value - decoded;
    const ulp = dt.kind === 'int' ? cur.raw.scale : Math.abs(decoded) * Math.pow(2, -dt.M) || Math.pow(2, 1 - dt.bias - dt.M);
    let out = `value = ${fmt(st.value)}    ${dt.label}    decoded = ${fmt(decoded)}    error = ${fmt(err)}    ulp ≈ ${fmt(ulp)}    tier:${r.name}\n`;
    out += s ? `${s.label}\nreconstructed so far = ${fmt(partial)}` : '(click a bit to flip it · press ▶ or scrub to reveal bits and rebuild the value)';
    page.setReadout(out);
  },
}).then((page) => {
  window.__dtypePage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Deep-link / headless restore for the two view knobs, so a copied link (and
  // a --screenshot run) reproduces the hardware matrix, not just the bit row.
  if (q.get('view') === 'hardware' || q.get('view') === 'bits') page.controls.set('view', q.get('view'), { rebuild: true });
  if (q.has('engines')) page.controls.set('engines', q.get('engines'), { rebuild: true });
  if (q.has('dtype')) page.controls.set('dtype', q.get('dtype'), { rebuild: true });
  // ?hwscroll=N scrolls the architecture list (headless stand-in for a drag).
  if (q.has('hwscroll')) { hwScroll = Math.max(0, +q.get('hwscroll') || 0); }
  // ?flip=i  (or comma-separated, e.g. ?flip=0,5,9) toggles bit index i --
  // headless stand-in for clicking bits, since --screenshot has no pointer.
  if (q.has('flip')) {
    if (!cur.dt) syncCur(DTYPES.find((d) => d.key === page.state.dtype), page.state.value);
    for (const tok of q.get('flip').split(',')) { const i = parseInt(tok, 10); if (Number.isFinite(i)) flipBit(i); }
    resyncTransport(page);
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover,
  // since --screenshot has no pointer) so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('flip') || q.has('hover') || q.has('hwscroll') || q.get('view') === 'hardware') { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
