// dtype-bits concept page -- the numeric zoo an accelerator actually advertises.
// Three views:
//   1. BIT LAYOUT  -- sign/exponent/mantissa (or sign/magnitude) cells for every
//      SCALAR type, fp64 down to fp4, int32 down to int1, with a bit-reveal
//      reconstruction and a rounding-error comparison table.
//   2. BLOCK       -- the block-scaled family (MXFP8/6/4, NVFP4, BFP8/BFP4): a
//      GROUP of elements sharing one scale. Shows the shared scale, the N
//      elements, the amortised bits/element, and value[i] = scale x element[i].
//   3. HARDWARE    -- a vendor x dtype support matrix from public vendor specs.
// Uses the verified framework: layout.mount() + controls + a step Transport over
// a page-built reveal sequence. IEEE-style decode is page-specific, so it lives
// here (not tensor.js); the transport just consumes the step array.
//
// NUMERIC NOTE -- why there are no `1 << M` shifts here any more: JS bitwise
// operators coerce to int32, so `1 << 52` is `1 << 20` and fp64 decoded to
// garbage. Every field is extracted with Math.pow(2, k) arithmetic instead,
// which is EXACT for the integers involved (a 52-bit mantissa is < 2^53, so it
// is representable in a double with no rounding) and works unchanged for int32's
// full-width sign bit.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';
import { T, effectiveTheme, alphaOf, signedColor, inkOn } from '../framework/theme.js';


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
// Fields added with the block-scaled family are token-only in BOTH themes -- a
// new [r,g,b] array would be a fresh hard-coded literal, which is exactly what
// the theme rule forbids.
const FIELD_TOKEN = { scale: 'violet', elem: 'teal', ghost: 'n5' };
// Adds over a bare theme.js `alphaOf`: the light arm keeps the EXACT tab10
// [r,g,b] array, so only dark mode is re-pointed at a semantic token.
const fieldFill = (name, a = 1) => (FIELD_TOKEN[name]
  ? alphaOf(FIELD_TOKEN[name], a)
  : (effectiveTheme() === 'dark' ? alphaOf(FIELD_DARK[name], a) : rgb(FIELD[name], a)));
const levelFill = (lv, a = 1) => (effectiveTheme() === 'dark' ? alphaOf(lv.dark, a) : rgb(lv.c, a));

// ---- exact power-of-two bit arithmetic (no int32 shifts) -------------------
const P2 = (k) => Math.pow(2, k);
const bitOf = (x, k) => Math.floor(x / P2(k)) % 2;
const flipAt = (x, k) => x + (bitOf(x, k) ? -P2(k) : P2(k));
// Round-to-nearest, TIES TO EVEN -- the rounding mode every FP unit and every
// fp32->fp8/fp6/fp4 converter actually implements. `Math.round` is ties-AWAY
// (half-up), which disagrees on exactly the values that sit midway between two
// codes: 2.5 -> fp4 e2m1 is 2.0 under RN-even and 3.0 under Math.round, and a
// low-mantissa format is midway between codes constantly. Cross-checked against
// ml_dtypes / Python struct, which are RN-even.
function rne(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// ---- block-scaled format catalogue ----------------------------------------
// A block format is NOT "an n-bit number": it is n bits of element PLUS a scale
// shared across `n` elements, so the honest cost is amortised.
//   MX (OCP Microscaling v1.0): 32 elements share one E8M0 (power-of-two) scale.
//   NVFP4: 16 elements share one fp8 e4m3 scale (a second, per-tensor fp32
//          scale exists above it; this page shows the per-block level).
//   BFP8/BFP4 (Tenstorrent block float): 16 elements share one 8-bit exponent
//          and keep sign + magnitude per element -- there is no per-element
//          exponent at all.
const BLOCKS = {
  mxfp8: { n: 32, elem: 'fp8e4m3', elemBits: 8, scaleKind: 'e8m0',    scaleBits: 8, spec: 'OCP Microscaling (MX) v1.0' },
  mxfp6: { n: 32, elem: 'fp6e3m2', elemBits: 6, scaleKind: 'e8m0',    scaleBits: 8, spec: 'OCP Microscaling (MX) v1.0' },
  mxfp4: { n: 32, elem: 'fp4e2m1', elemBits: 4, scaleKind: 'e8m0',    scaleBits: 8, spec: 'OCP Microscaling (MX) v1.0' },
  nvfp4: { n: 16, elem: 'fp4e2m1', elemBits: 4, scaleKind: 'fp8e4m3', scaleBits: 8, spec: 'NVIDIA NVFP4 (Blackwell)' },
  bfp8:  { n: 16, mantBits: 7,     elemBits: 8, scaleKind: 'e8m0',    scaleBits: 8, spec: 'Tenstorrent block float (BFP8_B)' },
  bfp4:  { n: 16, mantBits: 3,     elemBits: 4, scaleKind: 'e8m0',    scaleBits: 8, spec: 'Tenstorrent block float (BFP4_B)' },
};
const effBits = (bk) => bk.elemBits + bk.scaleBits / bk.n;

// ---- the dtype table -------------------------------------------------------
// `family` drives the column-family filter (floats / block-scaled / integers).
// `nan` says which codes the format reserves at the top exponent:
//   'ieee' -- all-ones exponent is Inf/NaN (fp64/fp32/tf32/fp16/bf16/fp8e5m2)
//   'e4m3' -- OCP fp8 e4m3: only mantissa-all-ones at max exponent is NaN, so
//             the largest finite is 448, not 240
//   'none' -- OCP fp6/fp4: NO Inf and NO NaN, every code is a number
// `container` (tf32) says the format is not a MEMORY width: 19 meaningful bits
// ride in a 32-bit register, and `ghost` is how many of those are unused.
const DTYPES = [
  { key: 'fp64',    label: 'fp64 (E11M52)',  kind: 'float', family: 'float', E: 11, M: 52, bias: 1023, bits: 64, nan: 'ieee' },
  { key: 'fp32',    label: 'fp32 (E8M23)',   kind: 'float', family: 'float', E: 8,  M: 23, bias: 127,  bits: 32, nan: 'ieee' },
  { key: 'tf32',    label: 'tf32 (E8M10)',   kind: 'float', family: 'float', E: 8,  M: 10, bias: 127,  bits: 19, nan: 'ieee', container: 32, ghost: 13 },
  { key: 'fp16',    label: 'fp16 (E5M10)',   kind: 'float', family: 'float', E: 5,  M: 10, bias: 15,   bits: 16, nan: 'ieee' },
  { key: 'bf16',    label: 'bf16 (E8M7)',    kind: 'float', family: 'float', E: 8,  M: 7,  bias: 127,  bits: 16, nan: 'ieee' },
  { key: 'fp8e4m3', label: 'fp8 e4m3',       kind: 'float', family: 'float', E: 4,  M: 3,  bias: 7,    bits: 8,  nan: 'e4m3' },
  { key: 'fp8e5m2', label: 'fp8 e5m2',       kind: 'float', family: 'float', E: 5,  M: 2,  bias: 15,   bits: 8,  nan: 'ieee' },
  { key: 'fp6e3m2', label: 'fp6 e3m2',       kind: 'float', family: 'float', E: 3,  M: 2,  bias: 3,    bits: 6,  nan: 'none' },
  { key: 'fp6e2m3', label: 'fp6 e2m3',       kind: 'float', family: 'float', E: 2,  M: 3,  bias: 1,    bits: 6,  nan: 'none' },
  { key: 'fp4e2m1', label: 'fp4 e2m1',       kind: 'float', family: 'float', E: 2,  M: 1,  bias: 1,    bits: 4,  nan: 'none' },

  { key: 'mxfp8',   label: 'MXFP8 (e4m3 ×32)', kind: 'block', family: 'block', block: 'mxfp8' },
  { key: 'mxfp6',   label: 'MXFP6 (e3m2 ×32)', kind: 'block', family: 'block', block: 'mxfp6' },
  { key: 'mxfp4',   label: 'MXFP4 (e2m1 ×32)', kind: 'block', family: 'block', block: 'mxfp4' },
  { key: 'nvfp4',   label: 'NVFP4 (e2m1 ×16)', kind: 'block', family: 'block', block: 'nvfp4' },
  { key: 'bfp8',    label: 'BFP8 (s+7m ×16)',  kind: 'block', family: 'block', block: 'bfp8' },
  { key: 'bfp4',    label: 'BFP4 (s+3m ×16)',  kind: 'block', family: 'block', block: 'bfp4' },

  { key: 'int32',   label: 'int32 (×scale)',  kind: 'int', family: 'int', bits: 32, signed: true,  range: 8 },
  { key: 'int16',   label: 'int16 (×scale)',  kind: 'int', family: 'int', bits: 16, signed: true,  range: 8 },
  { key: 'int8',    label: 'int8 (×scale)',   kind: 'int', family: 'int', bits: 8,  signed: true,  range: 8 },
  { key: 'uint8',   label: 'uint8 (zp+scale)', kind: 'int', family: 'int', bits: 8, signed: false, zp: 128, range: 8 },
  { key: 'int4',    label: 'int4 (×scale)',   kind: 'int', family: 'int', bits: 4,  signed: true,  range: 8 },
  { key: 'int2',    label: 'int2 (×scale)',   kind: 'int', family: 'int', bits: 2,  signed: true,  range: 8 },
  { key: 'int1',    label: 'int1 (±scale)',   kind: 'int', family: 'int', bits: 1,  binary: true,  range: 8 },
];
// Fill in the block rows' amortised width from BLOCKS, so the number on screen
// and the number in the accounting bar can never disagree.
for (const d of DTYPES) if (d.kind === 'block') d.bits = effBits(BLOCKS[d.block]);
const DT = Object.fromEntries(DTYPES.map((d) => [d.key, d]));

const COL_FAMILIES = [
  { value: 'all', label: `all ${DTYPES.length}` },
  { value: 'float', label: 'scalar floats' },
  { value: 'block', label: 'block-scaled' },
  { value: 'int', label: 'integers' },
];
const visDtypes = (st) => ((st.cols || 'all') === 'all' ? DTYPES : DTYPES.filter((d) => d.family === st.cols));

// ---- hardware dtype-support matrix ----------------------------------------
// The vendor/architecture capability catalogue lives in ../data/dtype-support.json
// (schema in ../data/README.md) and is fetched at runtime -- the page never
// hard-codes a capability. If the fetch fails (opened as file://, no server) the
// hardware view SAYS SO rather than inventing rows: an unknown renders as
// unknown, never as a guess. Same rule inside the data: a cell the vendor does
// not publish is `unknown`, which is a distinct colour from `none`. A dtype key
// the catalogue does not carry AT ALL is likewise `unknown` -- the column set is
// owned by this page, the capability facts by the JSON, and neither one being
// ahead of the other may crash or silently blank a cell.
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
  { value: 'cpu-simd', label: 'CPU SIMD' },
];

const hwCell = (a, key) => (a && a.dtypes && a.dtypes[key]) || { level: 'unknown' };
function hwArchs(state) {
  if (!HW || !Array.isArray(HW.architectures)) return [];
  const k = state.engines || 'all';
  return HW.architectures.filter((a) => k === 'all' || a.class === k);
}
function nativeList(a) {
  const ks = DTYPES.filter((d) => hwCell(a, d.key).level === 'native').map((d) => d.key);
  return ks.length > 6 ? `${ks.slice(0, 6).join(' ')} +${ks.length - 6} more` : ks.join(' ');
}
// One reveal step per architecture row -- the transport walks the matrix
// top→bottom exactly the way it walks the bit row in the bit-layout view.
function buildHwSteps(state) {
  return hwArchs(state).map((a, i) => ({
    rev: i + 1, partial: 0,
    label: `${a.vendor} ${a.arch} — ${a.engine}: native ${nativeList(a) || '(none of the taught dtypes)'}`,
  }));
}

function fmt(x) {
  if (x == null) return '—';
  if (Number.isNaN(x)) return 'NaN';
  if (!Number.isFinite(x)) return x > 0 ? '+Inf' : '−Inf';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e4 || a < 1e-3) return x.toExponential(2);
  return String(Number(x.toPrecision(4)));
}
const fmtBits = (b) => (Number.isInteger(b) ? String(b) : b.toFixed(2));

// ---- encode / decode -------------------------------------------------------
// The highest FINITE code of a float format: which (ef, mant) pair the top of
// the range actually is depends on what the format reserves for Inf/NaN.
function topCode(dt) {
  const maxEf = P2(dt.E) - 1;
  if (dt.nan === 'none') return { ef: maxEf, mant: P2(dt.M) - 1 };
  if (dt.nan === 'e4m3') return { ef: maxEf, mant: P2(dt.M) - 2 };
  return { ef: maxEf - 1, mant: P2(dt.M) - 1 };
}
function decodeFloat(raw, dt) {
  const maxEf = P2(dt.E) - 1, sgn = raw.sign ? -1 : 1;
  if (dt.nan === 'ieee' && raw.ef === maxEf) return raw.mant === 0 ? sgn * Infinity : NaN;
  if (dt.nan === 'e4m3' && raw.ef === maxEf && raw.mant === P2(dt.M) - 1) return NaN;
  const v = raw.ef === 0
    ? P2(1 - dt.bias) * (raw.mant / P2(dt.M))
    : P2(raw.ef - dt.bias) * (1 + raw.mant / P2(dt.M));
  return sgn * v;
}
function encodeFloat(v, dt) {
  const sign = (v < 0 || Object.is(v, -0)) ? 1 : 0;
  const top = topCode(dt), maxEf = P2(dt.E) - 1;
  if (Number.isNaN(v)) return { kind: 'float', sign: 0, ef: maxEf, mant: dt.nan === 'none' ? top.mant : Math.max(1, P2(dt.M) - 1) };
  const a = Math.abs(v);
  if (!Number.isFinite(a)) return { kind: 'float', sign, ef: dt.nan === 'ieee' ? maxEf : top.ef, mant: dt.nan === 'ieee' ? 0 : top.mant };
  let ef = 0, mant = 0;
  if (a !== 0) {
    const eu = Math.floor(Math.log2(a));
    ef = eu + dt.bias;
    if (ef <= 0) {                                        // subnormal
      mant = rne((a / P2(1 - dt.bias)) * P2(dt.M)); ef = 0;
      if (mant >= P2(dt.M)) { mant -= P2(dt.M); ef = 1; }
    } else if (ef > top.ef) {                             // clamp to max finite
      ef = top.ef; mant = top.mant;
    } else {
      mant = rne((a / P2(eu) - 1) * P2(dt.M));
      if (mant >= P2(dt.M)) { mant = 0; ef++; }
      if (ef > top.ef || (ef === top.ef && mant > top.mant)) { ef = top.ef; mant = top.mant; }
    }
  }
  return { kind: 'float', sign, ef, mant };
}
// Largest finite magnitude / its binade -- both needed by the block scalers.
const maxFinite = (dt) => decodeFloat({ sign: 0, ...topCode(dt) }, dt);
const emaxOf = (dt) => Math.floor(Math.log2(maxFinite(dt)));

// Integers carry a per-tensor (or per-group) scale; `range` is the absmax the
// scale is fitted to, so every int row on the page represents the same span and
// the comparison is about CODE COUNT, not about who got a friendlier scale.
function intSpan(dt) {
  if (dt.binary) return { qmin: 0, qmax: 1 };
  if (dt.signed === false) return { qmin: 0, qmax: P2(dt.bits) - 1 };
  return { qmin: -P2(dt.bits - 1), qmax: P2(dt.bits - 1) - 1 };
}
function intScale(dt) {
  if (dt.binary) return dt.range;                       // one code, ± the scale
  if (dt.signed === false) return dt.range / (dt.zp || 1);
  const { qmax } = intSpan(dt);
  return dt.range / Math.max(1, qmax);
}
function encodeInt(v, dt) {
  const scale = intScale(dt), { qmin, qmax } = intSpan(dt);
  if (dt.binary) return { kind: 'int', code: v < 0 ? 1 : 0, scale };
  const zp = dt.zp || 0;
  const code = Math.max(qmin, Math.min(qmax, rne(v / scale) + zp));
  return { kind: 'int', code, scale };
}
function decodeInt(raw, dt) {
  if (dt.binary) return (raw.code ? -1 : 1) * raw.scale;
  return (raw.code - (dt.zp || 0)) * raw.scale;
}
const encode = (dt, v) => (dt.kind === 'float' ? encodeFloat(v, dt) : encodeInt(v, dt));

// Two's-complement bit pattern <-> signed code, at any width (int32 included).
const patternOf = (dt, code) => (code < 0 ? code + P2(dt.bits) : code);
function codeOf(dt, pattern) {
  if (dt.binary || dt.signed === false) return pattern;
  return pattern >= P2(dt.bits - 1) ? pattern - P2(dt.bits) : pattern;
}

// Build the per-bit cell list { v, field } from a raw {sign,ef,mant} / {code}.
function bitcellsOf(dt, raw) {
  if (dt.kind === 'float') {
    const cells = [{ v: raw.sign, field: 'sign' }];
    for (let g = 0; g < dt.E; g++) cells.push({ v: bitOf(raw.ef, dt.E - 1 - g), field: 'exp' });
    for (let g = 0; g < dt.M; g++) cells.push({ v: bitOf(raw.mant, dt.M - 1 - g), field: 'mant' });
    return cells;
  }
  const N = dt.bits, pattern = patternOf(dt, raw.code), cells = [];
  const signed = !!dt.signed && !dt.binary;
  for (let j = 0; j < N; j++) cells.push({ v: bitOf(pattern, N - 1 - j), field: (signed && j === 0) || dt.binary ? 'sign' : 'mag' });
  return cells;
}

// Decode a {sign,ef,mant}/{code} raw back to a floating value.
function decodeRaw(dt, raw) {
  return dt.kind === 'float' ? decodeFloat(raw, dt) : decodeInt(raw, dt);
}

// ---- block-scaled encode ---------------------------------------------------
// One synthetic block of weights, deterministic so a screenshot is reproducible.
// Element 0 is REPLACED by the slider value, so the number the user is holding
// actually rides through the block and pays the block's rounding.
const synthBlock = (n, v) => Array.from({ length: n }, (_, i) => (i === 0 ? v : +(Math.sin(i * 1.7) * 2.6 + Math.cos(i * 0.41) * 1.3).toFixed(4)));

// Quantize a block. Returns the shared scale (value + its 8 stored bits), the
// per-element stored code + reconstructed value, and the reconstruction rule.
function quantBlock(bk, vals) {
  const amax = Math.max(0, ...vals.map((v) => Math.abs(v)));
  // --- Tenstorrent-style block float: ONE shared exponent, sign+magnitude
  //     elements, no per-element exponent at all.
  if (bk.mantBits != null) {
    const sharedExp = amax > 0 ? Math.floor(Math.log2(amax)) : 0;
    const q = P2(sharedExp - bk.mantBits + 1), mmax = P2(bk.mantBits) - 1;
    const els = vals.map((v) => {
      const s = v < 0 ? 1 : 0, m = Math.min(mmax, rne(Math.abs(v) / q));
      return { s, m, code: `${s ? '1' : '0'}·${m}`, bin: (s ? '1' : '0') + m.toString(2).padStart(bk.mantBits, '0'), val: (s ? -1 : 1) * m * q, elemVal: (s ? -1 : 1) * m };
    });
    return {
      kind: 'bfp', scale: q, scaleBin: (sharedExp + 127).toString(2).padStart(8, '0'),
      scaleLabel: `shared exponent 2^${sharedExp} (stored E8M0, biased ${sharedExp + 127})`,
      rule: `value[i] = (−1)^s · m[i] · 2^(${sharedExp} − ${bk.mantBits - 1})   ·   quantum ${fmt(q)}`,
      els, amax,
    };
  }
  // --- MX / NVFP4: a real element FORMAT, times a shared scale.
  const ed = DT[bk.elem];
  const emax = emaxOf(ed), elemMax = maxFinite(ed);
  let scale, scaleBin, scaleLabel;
  if (bk.scaleKind === 'e8m0') {
    // OCP MX v1.0: X = 2^(floor(log2(amax)) - emax_elem), a POWER OF TWO only.
    const se = amax > 0 ? Math.floor(Math.log2(amax)) - emax : 0;
    scale = P2(se);
    scaleBin = Math.max(0, Math.min(254, se + 127)).toString(2).padStart(8, '0');
    scaleLabel = `E8M0 scale = 2^${se} (exponent-only: 8 bits, no mantissa, biased ${se + 127})`;
  } else {
    // NVFP4: the scale is itself an fp8 e4m3 number, so it is NOT restricted to
    // powers of two -- it can land on amax/elemMax much more tightly.
    const sd = DT.fp8e4m3, want = amax > 0 ? amax / elemMax : 1;
    const sraw = encodeFloat(want, sd);
    scale = decodeFloat(sraw, sd) || P2(-127);
    scaleBin = bitcellsOf(sd, sraw).map((c) => c.v).join('');
    scaleLabel = `fp8 e4m3 scale = ${fmt(scale)} (a real float, not a power of two)`;
  }
  const els = vals.map((v) => {
    const raw = encodeFloat(v / scale, ed);
    const ev = decodeFloat(raw, ed);
    return { raw, bin: bitcellsOf(ed, raw).map((c) => c.v).join(''), code: fmt(ev), elemVal: ev, val: ev * scale };
  });
  return { kind: 'mx', scale, scaleBin, scaleLabel, rule: `value[i] = scale × element[i]   ·   scale = ${fmt(scale)}`, els, amax, elemMax };
}
const blockOf = (st) => {
  const d = DT[st.dtype];
  return d && d.kind === 'block' ? d.block : 'mxfp4';
};
// A block dtype's decoded value for the comparison table: put the slider value
// in slot 0 of a real block and read back what the block reconstructs.
function blockDecode(key, value) {
  const bk = BLOCKS[key], q = quantBlock(bk, synthBlock(bk.n, value));
  return q.els[0].val;
}

// ---- live editable bit state ----------------------------------------------
// cur holds the CURRENT bit pattern, decoupled from the value slider once the
// user starts flipping bits, so a flip survives redraws and rebuilds the value.
let cur = { dt: null, raw: null };
let bitRowRect = null;   // tight {x,y,w,h} around the N bit cells, captured in draw for hit-testing
let groupRects = [];     // [{name,x,y,w,h}] field-label bands above the cells, for label hover

// The bit-layout view teaches a SCALAR layout. A block dtype has no single bit
// strip -- its element format does -- so focusing one shows the element's strip
// with a banner pointing at the block view.
const scalarOf = (key) => {
  const d = DT[key] || DT.fp16;
  return d.kind === 'block' ? (BLOCKS[d.block].elem ? DT[BLOCKS[d.block].elem] : DT.fp8e4m3) : d;
};

// True once the user hand-edits the bit pattern, so the comparison table knows
// whether it is describing the CHOSEN value or an edited one (see drawTable's
// call site).
let patternEdited = false;

function syncCur(dt, value) {
  cur = { dt, raw: encode(dt, value) };
  patternEdited = false;
}

// Toggle bit index i (0 = sign, then exponent MSB→LSB, then mantissa MSB→LSB;
// for ints: sign then magnitude MSB→LSB) and recompute cur.raw in place.
function flipBit(i) {
  const dt = cur.dt, raw = cur.raw; if (!dt) return;
  patternEdited = true;
  if (dt.kind === 'float') {
    if (i === 0) raw.sign ^= 1;
    else if (i <= dt.E) raw.ef = flipAt(raw.ef, dt.E - i);           // exp bit (i-1) from MSB
    else raw.mant = flipAt(raw.mant, dt.M - (i - dt.E));             // mant bit from MSB
  } else {
    const pattern = flipAt(patternOf(dt, raw.code), dt.bits - 1 - i);
    raw.code = codeOf(dt, pattern);
  }
}

// ---- step-reveal sequence (left -> right, value rebuilds) ------------------
function buildSteps() {
  const dt = cur.dt, raw = cur.raw, decoded = decodeRaw(dt, raw), steps = [];
  if (dt.kind === 'float') {
    const sgn = raw.sign ? -1 : 1, sub = raw.ef === 0;
    const scale = sub ? P2(1 - dt.bias) : P2(raw.ef - dt.bias), implicit = sub ? 0 : 1;
    steps.push({ rev: 1, partial: 0, label: `sign bit = ${raw.sign} → ${raw.sign ? '−' : '+'}` });
    let partial = sgn * scale * implicit;
    steps.push({ rev: 1 + dt.E, partial, label: sub
      ? `exponent = 0 (subnormal) → scale 2^(1−${dt.bias}) = ${fmt(scale)}, no implicit 1 → ${fmt(partial)}`
      : `exponent = ${raw.ef} → 2^(${raw.ef}−${dt.bias}) = ${fmt(scale)}, ×(1 + mantissa) → ${fmt(partial)}` });
    for (let i = 1; i <= dt.M; i++) {
      const bit = bitOf(raw.mant, dt.M - i);
      if (bit) partial += sgn * scale * P2(-i);
      steps.push({ rev: 1 + dt.E + i, partial, label: `mantissa bit ${i} = ${bit}${bit ? ` → +${fmt(sgn * scale * P2(-i))}` : ''} → ${fmt(partial)}` });
    }
  } else {
    const N = dt.bits, pattern = patternOf(dt, raw.code), signed = !!dt.signed && !dt.binary;
    let partial = 0;
    for (let j = 0; j < N; j++) {
      const bit = bitOf(pattern, N - 1 - j);
      if (dt.binary) {
        partial = (bit ? -1 : 1) * raw.scale;
        steps.push({ rev: j + 1, partial, label: `the only bit = ${bit} → ${bit ? '−' : '+'}scale = ${fmt(partial)}` });
        continue;
      }
      const place = signed && j === 0 ? -P2(N - 1) : P2(N - 1 - j);
      partial += bit * place * raw.scale;
      const shown = partial - (dt.zp || 0) * raw.scale;
      steps.push({ rev: j + 1, partial: shown, label: signed && j === 0
        ? `sign bit = ${bit} → ${bit ? `−${fmt(P2(N - 1))}` : '0'} ×${fmt(raw.scale)} → ${fmt(shown)}`
        : `bit ${j} = ${bit}${bit ? ` → +${fmt(P2(N - 1 - j))}×${fmt(raw.scale)}` : ''}${dt.zp ? ` (−zp ${dt.zp})` : ''} → ${fmt(shown)}` });
    }
  }
  return steps.map((s) => ({ ...s, decoded }));
}

// Block view: reveal the shared scale, then the elements one at a time.
function buildBlockSteps(state) {
  const key = blockOf(state), bk = BLOCKS[key];
  const q = quantBlock(bk, synthBlock(bk.n, state.value));
  const steps = [{ rev: 1, partial: 0, label: `${key.toUpperCase()} — ${q.scaleLabel}` }];
  for (let i = 0; i < bk.n; i++) {
    const e = q.els[i];
    steps.push({ rev: i + 2, partial: e.val, label: `element ${i}: stored ${e.bin} → ${q.kind === 'bfp' ? `(−1)^s·m = ${fmt(e.elemVal)}` : `${fmt(e.elemVal)}`} × scale ${fmt(q.scale)} → ${fmt(e.val)}` });
  }
  return steps;
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
  // undoing the flips we just applied.
  t._dirty = false;
  t._sync();
}

// ---- per-bit field metadata (for hover-to-inspect) -------------------------
function bitInfo(dt, i) {
  if (dt.kind === 'float') {
    if (i === 0) return { field: 'sign', line: 'sign bit → (−1)^s' };
    if (i <= dt.E) { const w = dt.E - i; return { field: 'exponent', line: `exponent bit ${i - 1} (weight 2^${w} = ${fmt(P2(w))})` }; }
    const mi = i - dt.E;
    return { field: 'mantissa', line: `mantissa bit ${mi} → 2^-${mi} = ${fmt(P2(-mi))}` };
  }
  const N = dt.bits;
  if (dt.binary) return { field: 'sign', line: 'the single bit → + or − the scale' };
  if (i === 0 && dt.signed) return { field: 'sign', line: `sign bit → −2^${N - 1} = −${fmt(P2(N - 1))} (×scale)` };
  const w = N - 1 - i;
  return { field: 'magnitude', line: `bit ${i} → weight 2^${w} = ${fmt(P2(w))} (×scale)` };
}

// ---- drawing ---------------------------------------------------------------
function drawBitRow(page, dt, bitcells, rev, activeIdx, rect) {
  const ctx = page.ctx, N = bitcells.length, ghost = dt.ghost || 0, total = N + ghost;
  const cw = Math.max(6, Math.min(54, rect.w / total)), ch = 44;
  const x0 = rect.x + (rect.w - total * cw) / 2, y0 = rect.y;
  // Capture the tight bit-row rect for pointer hit-testing (cellAt: 1 row x N
  // cols) -- the ghost container bits are deliberately OUTSIDE it: they are not
  // part of the number and must not be clickable.
  bitRowRect = { x: x0, y: y0, w: N * cw, h: ch };
  const groups = dt.kind === 'float'
    ? [['sign', 0, 1], ['exponent', 1, 1 + dt.E], ['mantissa', 1 + dt.E, N]]
    : (dt.binary ? [['sign only (±scale)', 0, N]]
      : (dt.signed === false ? [['magnitude (− zero-point, ×scale)', 0, N]]
        : [['sign', 0, 1], ['magnitude (×scale)', 1, N]]));
  if (ghost) groups.push([`unused container bits (${ghost})`, N, total]);
  groupRects = [];
  ctx.save();
  ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
  for (const [name, lo, hi] of groups) {
    const xa = x0 + lo * cw, xb = x0 + hi * cw, xm = (xa + xb) / 2;
    ctx.fillStyle = T.n11; ctx.fillText(name, xm, y0 - 8);
    ctx.strokeStyle = T.n7; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xa + 2, y0 - 4); ctx.lineTo(xb - 2, y0 - 4); ctx.stroke();
    groupRects.push({ name, x: xa, y: y0 - 20, w: xb - xa, h: 18 });
  }
  ctx.textBaseline = 'middle';
  for (let i = 0; i < total; i++) {
    const x = x0 + i * cw;
    if (i >= N) {                                   // tf32's 13 unused container bits
      ctx.fillStyle = fieldFill('ghost', 0.45);
      ctx.fillRect(x, y0, cw - 1.5, ch);
      ctx.fillStyle = T.n9; ctx.font = (cw < 18 ? '9px' : '12px') + ' ui-monospace, monospace';
      if (cw >= 7) ctx.fillText('—', x + (cw - 1.5) / 2, y0 + ch / 2);
      continue;
    }
    const c = bitcells[i], shown = i < rev;
    ctx.fillStyle = shown ? fieldFill(c.field, c.v ? 0.92 : 0.20) : T.n3;
    ctx.fillRect(x, y0, cw - 1.5, ch);
    ctx.fillStyle = shown ? (c.v ? T.n0 : T.n12) : T.n8;
    ctx.font = (cw < 12 ? '8px' : cw < 18 ? '10px' : '13px') + ' ui-monospace, monospace';
    if (cw >= 7) ctx.fillText(shown ? String(c.v) : '·', x + (cw - 1.5) / 2, y0 + ch / 2);
    if (i === activeIdx) { ctx.strokeStyle = T.n14; ctx.lineWidth = 2.5; ctx.strokeRect(x + 1, y0 + 1, cw - 3.5, ch - 2); }
  }
  ctx.restore();
}

// The comparison table: encode ONE value in every visible dtype and show the
// rounding error. The bar is LOG-scaled -- with int1 and fp64 in the same table
// a linear bar renders every useful row as one pixel.
function drawTable(page, value, focusKey, rect) {
  const r = page.renderer, ctx = page.ctx, st = page.state;
  const list = visDtypes(st);
  const rows = list.map((dt) => {
    let d;
    if (dt.kind === 'block') d = blockDecode(dt.block, value);
    else d = decodeRaw(dt, encode(dt, value));
    return { dt, decoded: d, err: Number.isFinite(d) ? Math.abs(value - d) : Infinity };
  });
  const pos = rows.map((e) => e.err).filter((e) => e > 0 && Number.isFinite(e));
  const lo = pos.length ? Math.min(...pos) : 1e-18, hi = pos.length ? Math.max(...pos) : 1;
  const frac = (e) => {
    if (!(e > 0)) return 0;
    if (!Number.isFinite(e)) return 1;
    return hi <= lo ? 1 : Math.max(0.02, (Math.log10(e) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)));
  };
  const rowH = Math.max(13, Math.min(22, (page.H - rect.y - 26) / Math.max(1, rows.length)));
  const fs = rowH < 16 ? 10 : 12;
  // The label column has to clear 'MXFP8 (e4m3 x32)' / 'uint8 (zp+scale)'.
  const colN = rect.x, colB = rect.x + 154, colD = rect.x + 212, colBar = rect.x + 288;
  const barMax = Math.max(40, Math.min(150, rect.w - 366)), colE = colBar + barMax + 8;
  const hf = { color: T.n11, font: '11px ui-monospace, monospace' };
  r.label('dtype', colN, rect.y, hf);
  r.label('bits', colB, rect.y, hf);
  r.label('decoded', colD, rect.y, hf);
  r.label('|error|  (log-scaled bar)', colBar, rect.y, hf);
  let y = rect.y + 18;
  for (const e of rows) {
    const foc = e.dt.key === focusKey;
    r.label(e.dt.label, colN, y, { color: foc ? T.n14 : T.n12, font: (foc ? 'bold ' : '') + fs + 'px ui-monospace, monospace' });
    r.label(fmtBits(e.dt.bits) + (e.dt.container ? `/${e.dt.container}` : ''), colB, y, { color: e.dt.kind === 'block' ? T.violet : T.n12, font: fs + 'px ui-monospace, monospace' });
    r.label(fmt(e.decoded), colD, y, { color: T.n12, font: fs + 'px ui-monospace, monospace' });
    ctx.fillStyle = fieldFill(e.dt.kind === 'block' ? 'scale' : 'sign', foc ? 0.9 : 0.5);
    ctx.fillRect(colBar, y - rowH / 2 + 1, Math.max(1, frac(e.err) * barMax), Math.max(6, rowH - 8));
    r.label(fmt(e.err), colE, y, { color: T.n12, font: (fs - 1) + 'px ui-monospace, monospace' });
    if (foc) { ctx.strokeStyle = alphaOf(T.accent, 0.5); ctx.lineWidth = 1; ctx.strokeRect(colN - 6, y - rowH + 3, rect.w - 6, rowH); }
    y += rowH;
  }
}

// Trim a string to fit `maxW` px in the context's CURRENT font, adding an
// ellipsis when it does not. Measured, not character-counted.
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

// ---- block-scaled view -----------------------------------------------------
let blkCells = [];   // [{i,x,y,w,h}] element cells, for hover
let blkScaleRect = null;

function drawBlock(page) {
  const r = page.renderer, ctx = page.ctx, st = page.state, pad = 18;
  r.clear(T.n0);
  const key = blockOf(st), bk = BLOCKS[key];
  const vals = synthBlock(bk.n, st.value);
  const q = quantBlock(bk, vals);
  const s = page.step();
  const rev = s ? s.rev : bk.n + 1;

  // Header.
  ctx.save();
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = T.n14; ctx.font = 'bold 13px ui-monospace, monospace';
  ctx.fillText(`${key.toUpperCase()} — ${bk.n} elements share one scale`, pad, 20);
  ctx.fillStyle = T.n11; ctx.font = '10.5px ui-monospace, monospace';
  const elemDesc = bk.mantBits != null
    ? `element = sign + ${bk.mantBits}-bit magnitude (${bk.elemBits} b), NO per-element exponent`
    : `element = ${DT[bk.elem].label} (${bk.elemBits} b)`;
  ctx.fillText(`${bk.spec} · ${elemDesc} · shared scale ${bk.scaleBits} b`, pad, 34);
  ctx.restore();

  // Bits accounting -- the whole point: a block format's cost is amortised.
  const totalBits = bk.elemBits * bk.n + bk.scaleBits;
  const accY = 46, accH = 16, accW = page.W - 2 * pad;
  const elemW = accW * (bk.elemBits * bk.n) / totalBits;
  ctx.save();
  ctx.fillStyle = fieldFill('elem', 0.8); ctx.fillRect(pad, accY, elemW, accH);
  ctx.fillStyle = fieldFill('scale', 0.9); ctx.fillRect(pad + elemW, accY, accW - elemW, accH);
  ctx.textBaseline = 'middle'; ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = inkOn(fieldFill('elem', 1)); ctx.textAlign = 'left';
  ctx.fillText(`${bk.n} × ${bk.elemBits} b elements = ${bk.elemBits * bk.n} b`, pad + 6, accY + accH / 2);
  ctx.fillStyle = T.n12; ctx.textAlign = 'right';
  ctx.fillText(`+ ${bk.scaleBits} b scale  =  ${totalBits} b / ${bk.n}  =  ${effBits(bk).toFixed(2)} bits per element`, pad + accW - 6, accY + accH + 14);
  ctx.restore();

  // Shared scale panel.
  const scY = 84, scH = 30, scW = Math.min(300, page.W * 0.34);
  blkScaleRect = { x: pad, y: scY, w: scW, h: scH };
  ctx.save();
  ctx.fillStyle = fieldFill('scale', rev >= 1 ? 0.9 : 0.2);
  ctx.fillRect(pad, scY, scW, scH);
  ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  ctx.fillStyle = inkOn(fieldFill('scale', 1)); ctx.font = 'bold 11px ui-monospace, monospace';
  ctx.fillText(`scale ${rev >= 1 ? fmt(q.scale) : '·'}`, pad + 8, scY + scH / 2 - 6);
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(rev >= 1 ? q.scaleBin : '········', pad + 8, scY + scH / 2 + 8);
  ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(ellipsize(ctx, q.scaleLabel, page.W - pad - (pad + scW + 10)), pad + scW + 10, scY + scH / 2 - 6);
  ctx.fillStyle = T.n12;
  ctx.fillText(ellipsize(ctx, q.rule, page.W - pad - (pad + scW + 10)), pad + scW + 10, scY + scH / 2 + 8);
  ctx.restore();

  // Element grid.
  const cols = bk.n > 16 ? 16 : bk.n, rows = Math.ceil(bk.n / cols);
  const gTop = 128, gx = pad, gw = page.W - 2 * pad;
  const avail = Math.max(40, page.H - gTop - 46);
  const cw = gw / cols, chh = Math.min(104, Math.max(34, avail / rows));
  // Centre the block in whatever vertical room is left -- a 16-element format
  // is one row and would otherwise sit pinned to the top of an empty canvas.
  const gy = gTop + Math.max(0, (avail - rows * chh) / 2);
  blkCells = [];
  ctx.save();
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < bk.n; i++) {
    const cxi = i % cols, cyi = Math.floor(i / cols);
    const x = gx + cxi * cw, y = gy + cyi * chh;
    blkCells.push({ i, x, y, w: cw, h: chh });
    const e = q.els[i], shown = i + 2 <= rev;
    const t = q.amax > 0 ? e.val / q.amax : 0;
    const fill = shown ? signedColor(t) : T.n2;
    ctx.fillStyle = fill;
    ctx.fillRect(x + 1, y + 1, cw - 2, chh - 2);
    ctx.strokeStyle = i === 0 ? T.n14 : T.n5; ctx.lineWidth = i === 0 ? 2 : 1;
    ctx.strokeRect(x + 1, y + 1, cw - 2, chh - 2);
    if (s && i + 2 === rev) { ctx.strokeStyle = alphaOf(T.accent, 0.9); ctx.lineWidth = 2; ctx.strokeRect(x + 2, y + 2, cw - 4, chh - 4); }
    if (!shown) continue;
    // signedColor() runs the full ramp from the page ground to a saturated end,
    // so a fixed ink is unreadable at one end in every theme -- ask which end of
    // the neutral ramp actually contrasts with THIS cell.
    const ink = inkOn(fill);
    const cy = y + chh / 2;
    ctx.fillStyle = ink;
    // The stored-bit string is the widest thing in the cell (8 chars for an
    // 8-bit element); shrink to fit rather than let it bleed into a neighbour.
    let fs = 9.5;
    do { ctx.font = fs + 'px ui-monospace, monospace'; fs -= 0.5; }
    while (fs >= 6.5 && ctx.measureText(e.bin).width > cw - 8);
    ctx.fillText(e.bin, x + cw / 2, cy - 10);
    ctx.font = (cw < 44 ? '9px' : '11.5px') + ' ui-monospace, monospace';
    ctx.fillText(fmt(e.val), x + cw / 2, cy + 4);
    if (chh > 40) {
      ctx.fillStyle = alphaOf(ink, 0.66); ctx.font = '8.5px ui-monospace, monospace';
      ctx.fillText(`(${fmt(vals[i])})`, x + cw / 2, cy + 17);
    }
  }
  ctx.restore();

  // Caption under the grid.
  ctx.save();
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('each cell: stored bits · reconstructed value · (original)   — element 0 (outlined) is the value slider',
    pad, gy + rows * chh + 12);
  ctx.restore();

  // Hover an element -> full reconstruction for that element.
  const p = page.pointer;
  if (p.over) {
    const hit = blkCells.find((c) => p.x >= c.x && p.x < c.x + c.w && p.y >= c.y && p.y < c.y + c.h);
    if (hit) {
      const e = q.els[hit.i], orig = vals[hit.i];
      let tip = `element ${hit.i}${hit.i === 0 ? ' (the value slider)' : ''}\nstored bits: ${e.bin}  (${bk.elemBits} b)`;
      tip += q.kind === 'bfp'
        ? `\nsign ${e.s} · magnitude ${e.m} → ${fmt(e.elemVal)} quanta`
        : `\nelement value = ${fmt(e.elemVal)}  (${DT[bk.elem].label})`;
      tip += `\n× shared scale ${fmt(q.scale)} → ${fmt(e.val)}`;
      tip += `\noriginal ${fmt(orig)} · |error| ${fmt(Math.abs(orig - e.val))}`;
      page.setTip(tip);
    } else if (blkScaleRect && p.x >= blkScaleRect.x && p.x < blkScaleRect.x + blkScaleRect.w && p.y >= blkScaleRect.y && p.y < blkScaleRect.y + blkScaleRect.h) {
      page.setTip(`${q.scaleLabel}\nstored bits ${q.scaleBin} (${bk.scaleBits} b)\nblock absmax ${fmt(q.amax)} over ${bk.n} elements\n${q.rule}`);
    }
  }

  const err0 = Math.abs(vals[0] - q.els[0].val);
  page.probe = { view: 'block', dec: q.els[0].val, effBits: effBits(bk), blockKey: key };
  let out = `${key.toUpperCase()} · ${bk.n} elements / block · ${bk.elemBits} b element + ${bk.scaleBits} b shared scale = ${effBits(bk).toFixed(2)} bits/element    tier:${r.name}\n`;
  out += `${q.rule}    block absmax = ${fmt(q.amax)}    element 0: ${fmt(vals[0])} → ${fmt(q.els[0].val)} (|error| ${fmt(err0)})`;
  out += s ? `\n${s.label}` : '\n(hover an element for its reconstruction · hover the scale for the block accounting · scrub to reveal the scale then each element)';
  if (DT[st.dtype] && DT[st.dtype].kind !== 'block') out += `\nfocus dtype "${st.dtype}" is not block-scaled — showing ${key}; pick a block-scaled dtype to change it.`;
  page.setReadout(out);
}

// ---- hardware matrix drawing ----------------------------------------------
let hwScroll = 0;                 // px, vertical scroll of the architecture list
let hwView = null;                // {x,y,w,h} clipped row viewport
let hwHead = null;                // {x,y,w,h} dtype column-header band
let hwGeom = null;                // {x0,colW,rowH,archs,cols}

function hwClampScroll() {
  if (!hwGeom || !hwView) return;
  const max = Math.max(0, hwGeom.archs.length * hwGeom.rowH - hwView.h);
  hwScroll = Math.max(0, Math.min(max, hwScroll));
}

function drawHardware(page) {
  const r = page.renderer, ctx = page.ctx, st = page.state, pad = 18;
  r.clear(T.n0);
  const archs = hwArchs(st);
  const cols = visDtypes(st);
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
  const nameW = Math.min(320, Math.max(180, page.W * 0.32));
  const x0 = pad + nameW, colW = (page.W - pad - x0) / Math.max(1, cols.length);
  // 23 columns do not fit as horizontal labels; below ~44 px the key is turned
  // on its side rather than ellipsized into uselessness.
  const rot = colW < 44;
  const headTop = 10, headH = rot ? 58 : 26;
  const rowH = 21, top = headTop + headH + 8, legendH = 34;
  const viewH = Math.max(rowH, page.H - top - legendH - 4);
  hwHead = { x: x0, y: headTop, w: colW * cols.length, h: headH };
  hwView = { x: pad, y: top, w: page.W - 2 * pad, h: viewH };
  hwGeom = { x0, colW, rowH, archs, cols };
  hwClampScroll();
  if (s) {
    const ry = (rev - 1) * rowH;
    if (ry < hwScroll) hwScroll = ry;
    else if (ry + rowH > hwScroll + viewH) hwScroll = ry + rowH - viewH;
    hwClampScroll();
  }

  // Column headers (click one to focus that dtype).
  ctx.save();
  for (let j = 0; j < cols.length; j++) {
    const d = cols[j], foc = d.key === st.dtype, cx = x0 + j * colW + colW / 2;
    if (foc) { ctx.fillStyle = alphaOf(T.accent, 0.07); ctx.fillRect(x0 + j * colW, headTop, colW, headH + viewH + 8); }
    ctx.fillStyle = foc ? T.n14 : (d.family === 'block' ? T.violet : T.n11);
    if (rot) {
      ctx.save();
      ctx.translate(cx, headTop + headH - 4); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.font = (foc ? 'bold ' : '') + '10px ui-monospace, monospace';
      ctx.fillText(ellipsize(ctx, d.key, headH - 6), 0, 0);
      ctx.restore();
    } else {
      ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.font = (foc ? 'bold ' : '') + '11px ui-monospace, monospace';
      ctx.fillText(ellipsize(ctx, d.key, colW - 4), cx, headTop + 14);
      ctx.fillStyle = T.n10; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(`${fmtBits(d.bits)}b`, cx, headTop + 24);
    }
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
    for (let j = 0; j < cols.length; j++) {
      const lv = LEVELS[hwCell(a, cols[j].key).level] || LEVELS.unknown;
      ctx.fillStyle = levelFill(lv, alpha * (lv === LEVELS.none ? 0.30 : 0.88));
      ctx.fillRect(x0 + j * colW + 1.5, y + 3, Math.max(2, colW - 3), rowH - 6);
      ctx.fillStyle = alphaOf(T.n0, alpha); ctx.font = (colW < 16 ? '9px' : '11px') + ' ui-monospace, monospace';
      ctx.fillText(lv.glyph, x0 + j * colW + colW / 2, y + rowH / 2 - 0.5);
    }
  }
  ctx.restore();

  // Legend.
  ctx.save();
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.font = '10px ui-monospace, monospace';
  let lx = pad; const ly = page.H - legendH / 2 - 2;
  for (const k of LEVEL_ORDER) {
    const lv = LEVELS[k];
    ctx.fillStyle = levelFill(lv, k === 'none' ? 0.30 : 0.88); ctx.fillRect(lx, ly - 6, 12, 12);
    ctx.fillStyle = T.n0; ctx.textAlign = 'center'; ctx.fillText(lv.glyph, lx + 6, ly);
    ctx.textAlign = 'left'; ctx.fillStyle = T.n12; ctx.fillText(k, lx + 16, ly);
    lx += 18 + ctx.measureText(k).width + 14;
  }
  ctx.fillStyle = T.n10;
  ctx.fillText(ellipsize(ctx, `· ${cols.length}/${DTYPES.length} dtype columns shown`, page.W - pad - lx), lx + 6, ly);
  ctx.restore();

  // Hover-to-inspect.
  const p = page.pointer;
  if (p.over && p.x >= hwView.x && p.x < hwView.x + hwView.w && p.y >= hwView.y && p.y < hwView.y + hwView.h) {
    const i = Math.floor((p.y - top + hwScroll) / rowH);
    const j = Math.floor((p.x - x0) / colW);
    if (i >= 0 && i < archs.length) {
      const a = archs[i];
      if (j >= 0 && j < cols.length) {
        const d = cols[j], cell = hwCell(a, d.key), lv = LEVELS[cell.level] || LEVELS.unknown;
        let tip = `${a.vendor} ${a.arch} · ${a.engine}\n${d.label} (${fmtBits(d.bits)} b) → ${cell.level}: ${lv.short}`;
        if (cell.note) tip += `\n${cell.note}`;
        if (a.source) tip += `\nsource: ${a.source}`;
        page.setTip(tip);
      } else {
        page.setTip(`${a.vendor} ${a.arch} · ${a.engine}\nnative: ${nativeList(a) || '(none of the taught dtypes)'}\nsource: ${a.source || 'unspecified'}`);
      }
    }
  } else if (p.over && hwHead && p.y >= hwHead.y && p.y < hwHead.y + hwHead.h && p.x >= hwHead.x) {
    const j = Math.floor((p.x - x0) / colW);
    if (j >= 0 && j < cols.length) {
      const d = cols[j];
      page.setTip(`${d.label} — ${fmtBits(d.bits)} bits${d.container ? ` in a ${d.container}-bit container` : ''}${d.kind === 'block' ? ' amortised (element + shared scale)' : ''}\nclick to focus this dtype`);
    }
  }

  // Readout: the focused dtype's tally across the visible engine classes.
  const key = st.dtype, tally = {};
  for (const k of LEVEL_ORDER) tally[k] = 0;
  for (const a of archs) { const lv = hwCell(a, key).level; if (tally[lv] == null) tally[lv] = 0; tally[lv]++; }
  page.probe = { view: 'hardware', focusNative: tally.native, focusDtype: key };
  const focusLabel = (DT[key] || {}).label || key;
  let out = `hardware support · ${archs.length} architectures × ${cols.length} dtype columns · focus ${focusLabel}    tier:${r.name}\n`;
  out += LEVEL_ORDER.map((k) => `${k} ${tally[k]}`).join('   ');
  out += s ? `\n${s.label}` : '\n(drag the matrix to scroll · hover a cell for the level + public source · click a column header to focus that dtype · narrow the column family to read 23 columns)';
  page.setReadout(out);
}

mount({
  mount: 'body',
  title: 'dtype-bits — how a number is stored',
  blurb: 'The numeric zoo modern accelerators advertise: 10 scalar floats (fp64 → fp4), 6 block-scaled formats (MXFP8/6/4, NVFP4, BFP8/BFP4) and 7 integer widths (int32 → int1). BIT LAYOUT view: CLICK any bit to flip it and watch the reconstructed value rebuild; hover a bit for its field + place value; tf32 shows its 19 meaningful bits inside a 32-bit container. BLOCK view: a block-scaled value is not "4 bits" — it is 4 bits of element plus a scale shared across 32 of them, i.e. 4.25 bits amortised; the view shows the shared scale, the elements and value[i] = scale × element[i]. HARDWARE SUPPORT view: a vendor × dtype matrix from public vendor specs — filter by column family and engine class, drag to scroll, hover a cell for the level and its source.',
  prefer: 'canvas2d',
  aspect: '8 / 5',
  autoplay: true,
  compare: { key: 'dtype', a: 'fp16', b: 'int4', rebuild: true, labelA: 'fp16 — 16-bit float', labelB: 'int4 — 4-bit integer' },
  challenges: [
    { goal: 'Zero the number — clear every bit so it reconstructs to 0.', hint: 'click each lit bit to flip it to 0 (all bits 0 = the value 0).', check: (api) => ({ solved: Math.abs(api.probe.dec ?? 9) < 1e-6, detail: `reconstructed value = ${(api.probe.dec ?? 0).toFixed(4)}` }) },
    { goal: 'Make the stored number NEGATIVE.', hint: 'flip the leftmost bit — the sign bit (works for the float dtypes).', check: (api) => ({ solved: (api.probe.dec ?? 0) < 0, detail: `value = ${(api.probe.dec ?? 0).toFixed(3)}` }) },
    { goal: 'Switch to the hardware view and focus a dtype that AT MOST 6 of the listed architectures take as a native matrix-engine operand.', hint: 'set view = hardware, then click column headers — fp8 is a recent arrival and fp32 is an accumulator almost nowhere multiplies in; int8 is the one nearly everybody has.', check: (api) => ({ solved: api.probe.view === 'hardware' && typeof api.probe.focusNative === 'number' && api.probe.focusNative <= 6, detail: api.probe.view === 'hardware' ? `${api.probe.focusDtype}: ${api.probe.focusNative} native` : 'switch view to "hardware support"' }) },
    { goal: 'Switch to the block view and land on a format that costs UNDER 5 bits per element.', hint: 'set view = block-scaled, then focus mxfp4 / nvfp4 / bfp4 — 4 bits of element plus a scale shared across the block.', check: (api) => ({ solved: api.probe.view === 'block' && typeof api.probe.effBits === 'number' && api.probe.effBits < 5, detail: api.probe.view === 'block' ? `${api.probe.blockKey}: ${api.probe.effBits.toFixed(2)} bits/element` : 'switch view to "block-scaled"' }) },
  ],
  controls: (c, page) => {
    c.select('view', { label: 'view', value: 'bits', rebuild: true, options: [
      { value: 'bits', label: 'bit layout' }, { value: 'block', label: 'block-scaled' }, { value: 'hardware', label: 'hardware support' },
    ] });
    c.select('dtype', { label: 'focus dtype', value: 'fp16', rebuild: true, options: DTYPES.map((d) => ({ value: d.key, label: d.label })) });
    c.select('cols', { label: 'column family', value: 'all', rebuild: true, options: COL_FAMILIES });
    c.select('engines', { label: 'engine class (hardware view)', value: 'all', rebuild: true, options: ENGINE_CLASSES });
    c.slider('value', { label: 'value', min: -4, max: 4, step: 0.01, value: 1.3, rebuild: true, format: (v) => (+v).toFixed(2) });
    c.transport({ compute: () => {
      if (page.state.view === 'hardware') return buildHwSteps(page.state);
      if (page.state.view === 'block') return buildBlockSteps(page.state);
      syncCur(scalarOf(page.state.dtype), page.state.value); return buildSteps();
    }, speed: 5, loop: true });
  },
  // Direct manipulation. BIT LAYOUT: CLICK a bit cell to flip it. HARDWARE:
  // DRAG the matrix to scroll, CLICK a dtype column header to focus that dtype.
  onPointer: (page, ev) => {
    if (page.state.view === 'hardware') {
      if (ev.type === 'move' && page.pointer.down && hwGeom) { hwScroll -= ev.dy; hwClampScroll(); }
      else if (ev.type === 'down' && hwHead && hwGeom && ev.y >= hwHead.y && ev.y < hwHead.y + hwHead.h && ev.x >= hwHead.x) {
        const j = Math.floor((ev.x - hwGeom.x0) / hwGeom.colW);
        if (j >= 0 && j < hwGeom.cols.length) page.controls.set('dtype', hwGeom.cols[j].key, { rebuild: true });
      }
      return;
    }
    if (page.state.view === 'block') return;
    if (ev.type !== 'down' || !cur.dt) return;
    const N = bitcellsOf(cur.dt, cur.raw).length;
    const hit = bitRowRect && cellAt(bitRowRect, 1, N, ev.x, ev.y);
    if (hit) { flipBit(hit.c); resyncTransport(page); }
  },
  draw: (page) => {
    const r = page.renderer, st = page.state, pad = 18;
    if (st.view === 'hardware') { drawHardware(page); return; }
    if (st.view === 'block') { drawBlock(page); return; }
    r.clear(T.n0);
    const dt = cur.dt || scalarOf(st.dtype);
    if (!cur.dt) syncCur(dt, st.value);
    const bitcells = bitcellsOf(dt, cur.raw), decoded = decodeRaw(dt, cur.raw);
    page.probe = { view: 'bits', dec: decoded };
    const s = page.step();
    const rev = s ? s.rev : bitcells.length, partial = s ? s.partial : decoded, activeIdx = s ? s.rev - 1 : -1;

    // Banner lines: the two formats whose bit strip alone would MISLEAD.
    let bannerY = 22;
    const banner = (txt, col) => {
      page.ctx.save();
      page.ctx.font = '10.5px ui-monospace, monospace';
      const fit = ellipsize(page.ctx, txt, page.W - 2 * pad);
      page.ctx.restore();
      page.renderer.label(fit, pad, bannerY, { color: col, font: '10.5px ui-monospace, monospace' });
      bannerY += 13;
    };
    const focus = DT[st.dtype];
    if (focus && focus.kind === 'block') {
      banner(`${st.dtype} is BLOCK-SCALED — this strip is only its ELEMENT (${dt.label}). Real cost ${fmtBits(focus.bits)} b/element; see the "block-scaled" view.`, T.violet);
    }
    if (dt.container) {
      banner(`tf32 is NOT a memory format: ${dt.bits} meaningful bits ride in a ${dt.container}-bit register — it is what the tensor core INGESTS, not what anything stores.`, T.warn);
    }

    drawBitRow(page, dt, bitcells, rev, activeIdx, { x: pad, y: bannerY + 30, w: page.W - 2 * pad, h: 44 });
    // The table's whole point is "the SAME number, stored every way", so it must
    // encode the CHOSEN value -- not the focused dtype's already-quantised
    // decode. Feeding it `decoded` made every wider row read error 0 whenever a
    // coarse type held focus: with fp4 focused, fp64 claimed to store 1.5
    // exactly. Harmless-looking while int4 was the coarsest option, plainly
    // wrong once fp4/MXFP4 could be focused. After a hand-edited bit pattern
    // there is no "chosen value" to compare against, so the edit wins.
    drawTable(page, patternEdited ? decoded : st.value, st.dtype, { x: pad, y: bannerY + 110, w: page.W - 2 * pad });

    // Hover-to-inspect.
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
          if (lab.name.startsWith('unused')) tip = `${dt.key}: ${dt.bits} meaningful bits stored in a ${dt.container}-bit container — these ${dt.ghost} bits carry nothing.`;
          else if (dt.kind === 'float') {
            if (lab.name === 'sign') tip = `sign s = ${cur.raw.sign} → ${cur.raw.sign ? '−' : '+'}`;
            else if (lab.name === 'exponent') tip = `exponent e = ${cur.raw.ef}, bias ${dt.bias} → 2^(e−bias) = 2^${cur.raw.ef - dt.bias} = ${fmt(P2(cur.raw.ef - dt.bias))}\nmax finite ${fmt(maxFinite(dt))} · ${dt.nan === 'none' ? 'no Inf, no NaN (every code is a number)' : dt.nan === 'e4m3' ? 'only mantissa-all-ones at max exponent is NaN — no Inf' : 'all-ones exponent = Inf / NaN'}`;
            else tip = `mantissa m = ${cur.raw.mant}/${fmt(P2(dt.M))}\nvalue = (−1)^s · 2^(e−bias) · 1.m = ${fmt(decoded)}`;
          } else if (dt.binary) tip = `one bit: 0 → +scale, 1 → −scale (scale ${fmt(cur.raw.scale)})`;
          else if (lab.name === 'sign') tip = `sign bit → two's-complement code ${cur.raw.code}`;
          else tip = `(code ${cur.raw.code}${dt.zp ? ` − zero-point ${dt.zp}` : ''}) × scale ${fmt(cur.raw.scale)} → ${fmt(decoded)}`;
        }
      }
      if (tip) page.setTip(tip);
    }

    const err = st.value - decoded;
    const ulp = dt.kind === 'int' ? cur.raw.scale : Math.abs(decoded) * P2(-dt.M) || P2(1 - dt.bias - dt.M);
    let out = `value = ${fmt(st.value)}    ${dt.label}    ${dt.bits} bits${dt.container ? ` in a ${dt.container}-bit container` : ''}    decoded = ${fmt(decoded)}    error = ${fmt(err)}    ulp ≈ ${fmt(ulp)}    tier:${r.name}\n`;
    out += s ? `${s.label}\nreconstructed so far = ${fmt(partial)}` : '(click a bit to flip it · press ▶ or scrub to reveal bits and rebuild the value)';
    page.setReadout(out);
  },
}).then((page) => {
  window.__dtypePage = page;
  // Numeric core, exposed for the headless cross-check (framework/theme-verify
  // proves the page DRAWS; this is what lets an independent implementation --
  // Python `struct` / ml_dtypes -- prove it DECODES the same bits). Read-only:
  // nothing in the page reads back through this handle.
  window.__dtypeNum = { DTYPES, BLOCKS, encode, decodeRaw, bitcellsOf, quantBlock, synthBlock, maxFinite, effBits };
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Deep-link / headless restore for the view knobs, so a copied link (and a
  // --screenshot run) reproduces the matrix or the block view, not just the bits.
  if (['bits', 'block', 'hardware'].includes(q.get('view'))) page.controls.set('view', q.get('view'), { rebuild: true });
  if (q.has('engines')) page.controls.set('engines', q.get('engines'), { rebuild: true });
  if (q.has('cols')) page.controls.set('cols', q.get('cols'), { rebuild: true });
  if (q.has('dtype')) page.controls.set('dtype', q.get('dtype'), { rebuild: true });
  if (q.has('hwscroll')) { hwScroll = Math.max(0, +q.get('hwscroll') || 0); }
  // ?flip=i  (or comma-separated) toggles bit index i -- headless stand-in for
  // clicking bits, since --screenshot has no pointer.
  if (q.has('flip')) {
    // Re-sync when the focus dtype does not match the pattern we are holding:
    // `?dtype=fp4e2m1&flip=0` used to flip a bit of whatever the page mounted
    // with (fp16), because the guard only fired when there was NO pattern at
    // all. The flip then reported fp16 precision under an fp4 heading.
    const want = scalarOf(page.state.dtype);
    if (!cur.dt || (want && cur.dt.key !== want.key)) syncCur(want, page.state.value);
    for (const tok of q.get('flip').split(',')) { const i = parseInt(tok, 10); if (Number.isFinite(i)) flipBit(i); }
    resyncTransport(page);
  }
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('flip') || q.has('hover') || q.has('hwscroll') || q.get('view') === 'hardware' || q.get('view') === 'block') { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
