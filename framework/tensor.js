// tensor.js -- real in-page math for viz concept pages.
//
// The "show the math" guarantee: every number a page displays is computed
// HERE, deterministically, and is inspectable op-by-op. Design: plan/framework.md.
//
// Two forms per op:
//   - eager:  matmul(A,B) -> result                (fast path, hot render loop)
//   - stepped: matmulSteps(A,B) -> generator        (yields renderable snapshots
//              for the "step one operation" button / scrub axis)
// A step record is { op, label, i?, j?, k?, value?, phase?, partial }, where
// `partial` is the evolving output (a copy, safe for the page to hold) and the
// indices say which cells to highlight via render.js cell()/grid().
//
// Conventions:
//   matrix = { data: Float32Array (row-major), rows, cols }   (also accepts
//             number[][] or {data,rows,cols} on input; see asMat)
//   vector = Float32Array | number[]
//
// NOTE: quantize() here is NUMERIC dtype quantization (int4/int8, the quant
// page). It is unrelated to render.js's scalar->colormap-index quantize().
//
// No build step: ES module import, or <script> + window.VizTensor global.

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------
export function mat(rows, cols, fill = 0) {
  const data = new Float32Array(rows * cols);
  if (fill) data.fill(fill);
  return { data, rows, cols };
}
export function fromRows(rows2d) {
  const rows = rows2d.length, cols = rows2d[0].length, data = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) data[r * cols + c] = rows2d[r][c];
  return { data, rows, cols };
}
export function toRows(M) {
  M = asMat(M);
  const out = [];
  for (let r = 0; r < M.rows; r++) { const row = []; for (let c = 0; c < M.cols; c++) row.push(M.data[r * M.cols + c]); out.push(row); }
  return out;
}
// Accept {data,rows,cols} | number[][] -> normalized matrix (Float32 data).
export function asMat(M) {
  if (M && M.data && M.rows != null) return M.data instanceof Float32Array ? M : { data: Float32Array.from(M.data), rows: M.rows, cols: M.cols };
  if (Array.isArray(M) && Array.isArray(M[0])) return fromRows(M);
  throw new Error('asMat: expected {data,rows,cols} or number[][]');
}
export function asVec(v) {
  if (v && v.data && v.rows != null) return v.data instanceof Float32Array ? v.data : Float32Array.from(v.data);
  return v instanceof Float32Array ? v : Float32Array.from(v);
}
const at = (M, i, j) => M.data[i * M.cols + j];
const cloneMat = (M) => ({ data: Float32Array.from(M.data), rows: M.rows, cols: M.cols });

export function transpose(M) {
  M = asMat(M);
  const out = mat(M.cols, M.rows);
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) out.data[c * M.rows + r] = at(M, r, c);
  return out;
}
export function scale(M, s) {
  M = asMat(M);
  const out = cloneMat(M);
  for (let i = 0; i < out.data.length; i++) out.data[i] *= s;
  return out;
}

// ---------------------------------------------------------------------------
// dot / matmul
// ---------------------------------------------------------------------------
export function dot(a, b) {
  a = asVec(a); b = asVec(b);
  let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
}
// Yields per-term accumulation: {k, term, acc, label}.
export function* dotSteps(a, b) {
  a = asVec(a); b = asVec(b);
  let acc = 0;
  for (let k = 0; k < a.length; k++) {
    const term = a[k] * b[k]; acc += term;
    yield { op: 'dot', k, term, value: a[k] * b[k], acc, partial: acc, label: `+ a[${k}]·b[${k}] = ${a[k].toFixed(3)}·${b[k].toFixed(3)} = ${term.toFixed(3)}  (sum ${acc.toFixed(3)})` };
  }
}

// C[i,j] = sum_k A[i,k]·B[k,j].  A:(m×k) B:(k×n) -> C:(m×n).
export function matmul(A, B) {
  A = asMat(A); B = asMat(B);
  if (A.cols !== B.rows) throw new Error(`matmul: inner dims ${A.cols} != ${B.rows}`);
  const m = A.rows, n = B.cols, K = A.cols, C = mat(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    let s = 0; for (let k = 0; k < K; k++) s += at(A, i, k) * B.data[k * n + j];
    C.data[i * n + j] = s;
  }
  return C;
}
// Stepped matmul. Default: one yield per output cell (watch each cell light up).
// opts.perK: also yield each accumulation term inside a cell.
export function* matmulSteps(A, B, opts = {}) {
  A = asMat(A); B = asMat(B);
  const m = A.rows, n = B.cols, K = A.cols, C = mat(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < K; k++) {
      const term = at(A, i, k) * B.data[k * n + j]; s += term;
      if (opts.perK) yield { op: 'matmul', phase: 'term', i, j, k, term, acc: s, partial: cloneMat(C), label: `C[${i},${j}] += A[${i},${k}]·B[${k},${j}] = ${term.toFixed(3)} (${s.toFixed(3)})` };
    }
    C.data[i * n + j] = s;
    yield { op: 'matmul', phase: 'cell', i, j, k: K, value: s, partial: cloneMat(C), label: `C[${i},${j}] = row ${i} · col ${j} = ${s.toFixed(3)}` };
  }
}

// ---------------------------------------------------------------------------
// softmax (max-subtract stable). Vector form + row-wise matrix form.
// ---------------------------------------------------------------------------
export function softmax(v, opts = {}) {
  v = asVec(v); const t = opts.temp == null ? 1 : opts.temp;
  let mx = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] / t > mx) mx = v[i] / t;
  const e = new Float32Array(v.length); let sum = 0;
  for (let i = 0; i < v.length; i++) { e[i] = Math.exp(v[i] / t - mx); sum += e[i]; }
  for (let i = 0; i < e.length; i++) e[i] /= sum;
  return e;
}
// Phases: max -> per-elem exp -> sum -> per-elem normalize. partial is the
// evolving probability/exp vector.
export function* softmaxSteps(v, opts = {}) {
  v = asVec(v); const t = opts.temp == null ? 1 : opts.temp;
  let mx = -Infinity; for (let i = 0; i < v.length; i++) if (v[i] / t > mx) mx = v[i] / t;
  yield { op: 'softmax', phase: 'max', value: mx, partial: Float32Array.from(v), label: `max(logits${t !== 1 ? '/T' : ''}) = ${mx.toFixed(3)} (subtract for stability)` };
  const e = new Float32Array(v.length); let sum = 0;
  for (let i = 0; i < v.length; i++) {
    e[i] = Math.exp(v[i] / t - mx); sum += e[i];
    yield { op: 'softmax', phase: 'exp', i, value: e[i], acc: sum, partial: Float32Array.from(e), label: `exp(${(v[i] / t).toFixed(3)} - ${mx.toFixed(3)}) = ${e[i].toFixed(3)}` };
  }
  yield { op: 'softmax', phase: 'sum', value: sum, partial: Float32Array.from(e), label: `Σexp = ${sum.toFixed(3)}` };
  const p = new Float32Array(v.length);
  for (let i = 0; i < e.length; i++) {
    p[i] = e[i] / sum;
    yield { op: 'softmax', phase: 'norm', i, value: p[i], partial: Float32Array.from(p), label: `p[${i}] = ${e[i].toFixed(3)} / ${sum.toFixed(3)} = ${p[i].toFixed(3)}` };
  }
}
// Row-wise softmax of a matrix (attention weights: one softmax per query row).
export function softmaxRows(M, opts = {}) {
  M = asMat(M); const out = mat(M.rows, M.cols);
  for (let r = 0; r < M.rows; r++) {
    const row = softmax(M.data.subarray(r * M.cols, r * M.cols + M.cols), opts);
    out.data.set(row, r * M.cols);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------
export function layernorm(v, opts = {}) {
  v = asVec(v); const n = v.length, eps = opts.eps == null ? 1e-5 : opts.eps;
  let mean = 0; for (let i = 0; i < n; i++) mean += v[i]; mean /= n;
  let varr = 0; for (let i = 0; i < n; i++) { const d = v[i] - mean; varr += d * d; } varr /= n;
  const inv = 1 / Math.sqrt(varr + eps), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let y = (v[i] - mean) * inv;
    if (opts.gamma) y *= asVec(opts.gamma)[i];
    if (opts.beta) y += asVec(opts.beta)[i];
    out[i] = y;
  }
  return out;
}
export function* layernormSteps(v, opts = {}) {
  v = asVec(v); const n = v.length, eps = opts.eps == null ? 1e-5 : opts.eps;
  let mean = 0; for (let i = 0; i < n; i++) mean += v[i]; mean /= n;
  yield { op: 'layernorm', phase: 'mean', value: mean, partial: Float32Array.from(v), label: `μ = ${mean.toFixed(3)}` };
  let varr = 0; for (let i = 0; i < n; i++) { const d = v[i] - mean; varr += d * d; } varr /= n;
  yield { op: 'layernorm', phase: 'var', value: varr, partial: Float32Array.from(v), label: `σ² = ${varr.toFixed(3)}, σ = ${Math.sqrt(varr).toFixed(3)}` };
  const inv = 1 / Math.sqrt(varr + eps), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let y = (v[i] - mean) * inv;
    if (opts.gamma) y *= asVec(opts.gamma)[i];
    if (opts.beta) y += asVec(opts.beta)[i];
    out[i] = y;
    yield { op: 'layernorm', phase: 'norm', i, value: y, partial: Float32Array.from(out), label: `y[${i}] = (${v[i].toFixed(3)} - ${mean.toFixed(3)}) / ${Math.sqrt(varr + eps).toFixed(3)} = ${y.toFixed(3)}` };
  }
}
// RMSNorm: no mean-subtract; divide by RMS, optional per-dim weight.
export function rmsnorm(v, opts = {}) {
  v = asVec(v); const n = v.length, eps = opts.eps == null ? 1e-6 : opts.eps;
  let ms = 0; for (let i = 0; i < n; i++) ms += v[i] * v[i]; ms /= n;
  const inv = 1 / Math.sqrt(ms + eps), out = new Float32Array(n), w = opts.weight ? asVec(opts.weight) : null;
  for (let i = 0; i < n; i++) out[i] = v[i] * inv * (w ? w[i] : 1);
  return out;
}
export function* rmsnormSteps(v, opts = {}) {
  v = asVec(v); const n = v.length, eps = opts.eps == null ? 1e-6 : opts.eps;
  let ms = 0; for (let i = 0; i < n; i++) ms += v[i] * v[i]; ms /= n;
  const rms = Math.sqrt(ms + eps);
  yield { op: 'rmsnorm', phase: 'rms', value: rms, partial: Float32Array.from(v), label: `RMS = √(mean(x²)+ε) = ${rms.toFixed(3)}` };
  const inv = 1 / rms, out = new Float32Array(n), w = opts.weight ? asVec(opts.weight) : null;
  for (let i = 0; i < n; i++) {
    out[i] = v[i] * inv * (w ? w[i] : 1);
    yield { op: 'rmsnorm', phase: 'norm', i, value: out[i], partial: Float32Array.from(out), label: `y[${i}] = ${v[i].toFixed(3)} / ${rms.toFixed(3)}${w ? `·${w[i].toFixed(3)}` : ''} = ${out[i].toFixed(3)}` };
  }
}

// ---------------------------------------------------------------------------
// Activations (elementwise; accept scalar or vector)
// ---------------------------------------------------------------------------
export function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function _erf(x) { // Abramowitz & Stegun 7.1.26
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const _map = (v, f) => (typeof v === 'number' ? f(v) : Float32Array.from(asVec(v), f));
export function silu(v) { return _map(v, (x) => x * sigmoid(x)); }
export function gelu(v, opts = {}) {
  if (opts.approx === 'erf') return _map(v, (x) => 0.5 * x * (1 + _erf(x / Math.SQRT2)));
  const c = Math.sqrt(2 / Math.PI);                                   // tanh approx (default; gelu_pytorch_tanh)
  return _map(v, (x) => 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x))));
}
// SwiGLU gate: silu(gate) ⊙ up (elementwise).
export function swiglu(gate, up) {
  gate = asVec(gate); up = asVec(up);
  const out = new Float32Array(gate.length);
  for (let i = 0; i < out.length; i++) out[i] = gate[i] * sigmoid(gate[i]) * up[i];
  return out;
}

// ---------------------------------------------------------------------------
// RoPE -- pairwise 2D rotation by position·frequency.
// ---------------------------------------------------------------------------
// For dim d (even), pair (2i, 2i+1) rotates by θ = pos · base^(-2i/d).
export function ropeAngles(dim, pos, opts = {}) {
  const base = opts.theta == null ? 10000 : opts.theta, half = dim >> 1, out = new Float32Array(half);
  for (let i = 0; i < half; i++) out[i] = pos * Math.pow(base, -(2 * i) / dim);
  return out;
}
export function rope(vec, pos, opts = {}) {
  const v = asVec(vec), d = v.length, ang = ropeAngles(d, pos, opts), out = new Float32Array(d);
  for (let i = 0; i < (d >> 1); i++) {
    const a = v[2 * i], b = v[2 * i + 1], c = Math.cos(ang[i]), s = Math.sin(ang[i]);
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  if (d & 1) out[d - 1] = v[d - 1];                                   // odd tail passes through
  return out;
}
export function* ropeSteps(vec, pos, opts = {}) {
  const v = asVec(vec), d = v.length, ang = ropeAngles(d, pos, opts), out = Float32Array.from(v);
  for (let i = 0; i < (d >> 1); i++) {
    const a = v[2 * i], b = v[2 * i + 1], c = Math.cos(ang[i]), s = Math.sin(ang[i]);
    out[2 * i] = a * c - b * s; out[2 * i + 1] = a * s + b * c;
    yield { op: 'rope', phase: 'pair', i, k: i, angle: ang[i], before: [a, b], after: [out[2 * i], out[2 * i + 1]], partial: Float32Array.from(out), label: `pair ${i}: rotate (${a.toFixed(2)},${b.toFixed(2)}) by ${(ang[i]).toFixed(3)} rad` };
  }
}

// ---------------------------------------------------------------------------
// Quantization (numeric dtype: int4/int8, per-group affine or symmetric).
// ---------------------------------------------------------------------------
// Returns {q:Int32Array codes, scale:Float32Array(per group), zero:Int32Array,
//          dequant:Float32Array, error:Float32Array (x - dequant), mse, groups}.
export function quantize(v, opts = {}) {
  v = asVec(v);
  const bits = opts.bits == null ? 4 : opts.bits;
  const gs = opts.groupSize == null ? v.length : Math.min(opts.groupSize, v.length);
  const sym = !!opts.symmetric;
  const n = v.length, groups = Math.ceil(n / gs);
  const q = new Int32Array(n), dequant = new Float32Array(n), error = new Float32Array(n);
  const scale = new Float32Array(groups), zero = new Int32Array(groups);
  const qmax = (1 << bits) - 1, qmaxSym = (1 << (bits - 1)) - 1;
  let se = 0;
  for (let g = 0; g < groups; g++) {
    const s0 = g * gs, s1 = Math.min(s0 + gs, n);
    let lo = Infinity, hi = -Infinity, amax = 0;
    for (let i = s0; i < s1; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; if (Math.abs(v[i]) > amax) amax = Math.abs(v[i]); }
    let sc, zp;
    if (sym) { sc = (amax || 1) / qmaxSym; zp = 0; }
    else { sc = ((hi - lo) || 1) / qmax; zp = Math.round(-lo / sc); }
    scale[g] = sc; zero[g] = zp;
    for (let i = s0; i < s1; i++) {
      let code = Math.round(v[i] / sc) + zp;
      code = sym ? Math.max(-qmaxSym - 1, Math.min(qmaxSym, code)) : Math.max(0, Math.min(qmax, code));
      q[i] = code;
      dequant[i] = (code - zp) * sc;
      error[i] = v[i] - dequant[i];
      se += error[i] * error[i];
    }
  }
  return { q, scale, zero, dequant, error, mse: se / n, groups, bits, groupSize: gs, symmetric: sym };
}
export function dequantize(res) {
  const { q, scale, zero, groupSize } = res, out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) { const g = (i / groupSize) | 0; out[i] = (q[i] - zero[g]) * scale[g]; }
  return out;
}
// Per-group steps for the quant page.
export function* quantizeSteps(v, opts = {}) {
  v = asVec(v); const res = quantize(v, opts), gs = res.groupSize;
  for (let g = 0; g < res.groups; g++) {
    const s0 = g * gs, s1 = Math.min(s0 + gs, v.length);
    yield { op: 'quantize', phase: 'group', i: g, k: g, value: res.scale[g], range: [s0, s1], scale: res.scale[g], zero: res.zero[g], partial: Float32Array.from(res.dequant), label: `group ${g}: scale=${res.scale[g].toExponential(2)} zero=${res.zero[g]} (${res.bits}-bit)` };
  }
  yield { op: 'quantize', phase: 'done', value: res.mse, partial: Float32Array.from(res.error), label: `MSE = ${res.mse.toExponential(3)}` };
}

// ---------------------------------------------------------------------------
// Deterministic RNG -- same seed => same picture (portability guarantee).
// ---------------------------------------------------------------------------
function _hashSeed(seed) {
  if (typeof seed === 'number') return seed >>> 0;
  let h = 2166136261 >>> 0;                                           // FNV-1a over the string
  const s = String(seed);
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
// mulberry32: tiny, fast, well-distributed PRNG -> uniform [0,1).
export function rng(seed) {
  let a = _hashSeed(seed) || 1;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function _shape(shape) { // number -> [n] (vector); [r,c] -> matrix
  if (typeof shape === 'number') return { vec: true, n: shape };
  return { vec: false, rows: shape[0], cols: shape[1], n: shape[0] * shape[1] };
}
// Deterministic standard-normal (Box-Muller). shape: n (vector) or [rows,cols].
export function seededRandn(seed, shape, opts = {}) {
  const mean = opts.mean == null ? 0 : opts.mean, std = opts.std == null ? 1 : opts.std;
  const sp = _shape(shape), next = rng(seed), data = new Float32Array(sp.n);
  for (let i = 0; i < sp.n; i += 2) {
    const u1 = Math.max(next(), 1e-12), u2 = next(), r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    data[i] = mean + std * r * Math.cos(th);
    if (i + 1 < sp.n) data[i + 1] = mean + std * r * Math.sin(th);
  }
  return sp.vec ? data : { data, rows: sp.rows, cols: sp.cols };
}
// Deterministic uniform [lo,hi).
export function seededRand(seed, shape, opts = {}) {
  const lo = opts.lo == null ? 0 : opts.lo, hi = opts.hi == null ? 1 : opts.hi;
  const sp = _shape(shape), next = rng(seed), data = new Float32Array(sp.n);
  for (let i = 0; i < sp.n; i++) data[i] = lo + (hi - lo) * next();
  return sp.vec ? data : { data, rows: sp.rows, cols: sp.cols };
}

// ---------------------------------------------------------------------------
// Step-mode driver -- consume a *Steps generator at a cursor position.
// ---------------------------------------------------------------------------
// Collect every step up front (steps are cheap at viz scale) so a scrub axis
// can index any position O(1). Returns the step array.
export function collectSteps(gen) { return Array.from(gen); }
// State of a stepped computation AFTER consuming `cursor` steps: the last
// step's record (with its `partial`) and how many remain.
export function stepAt(steps, cursor) {
  const i = Math.max(-1, Math.min(steps.length - 1, cursor | 0));
  return { index: i, total: steps.length, step: i < 0 ? null : steps[i], done: i >= steps.length - 1 };
}

// <script>-tag global (non-module pages).
if (typeof window !== 'undefined') {
  window.VizTensor = {
    mat, fromRows, toRows, asMat, asVec, transpose, scale,
    dot, dotSteps, matmul, matmulSteps,
    softmax, softmaxSteps, softmaxRows,
    layernorm, layernormSteps, rmsnorm, rmsnormSteps,
    sigmoid, silu, gelu, swiglu,
    ropeAngles, rope, ropeSteps,
    quantize, dequantize, quantizeSteps,
    rng, seededRandn, seededRand,
    collectSteps, stepAt,
  };
}
