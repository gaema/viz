// Correctness gate for the real-attention page: the vanilla-JS GPT-2 forward in
// gpt2.js must reproduce PyTorch GPT-2's attention (output_attentions, eager).
// The fixture (gpt2-groundtruth.fixture.json) holds real PyTorch attention
// matrices + the literature head locations for a fixed sentence; this test runs
// the JS forward on the same token ids and compares.
//
// Needs the gpt2 safetensors (548 MB) — not committed. Point GPT2_WEIGHTS at it:
//   GPT2_WEIGHTS=~/.cache/huggingface/hub/models--gpt2/snapshots/*/model.safetensors \
//     node real-attention/gpt2.test.mjs
// Skips (exit 0) with a note if the weights aren't present.
import { readFileSync } from 'node:fs';
import { parseSafetensors, GPT2, GPT2_CONFIG, loadGPT2 } from './gpt2.js';

const fix = JSON.parse(readFileSync(new URL('./gpt2-groundtruth.fixture.json', import.meta.url)));
const wpath = process.env.GPT2_WEIGHTS;
if (!wpath) { console.log('SKIP: set GPT2_WEIGHTS=<model.safetensors | http://…> to run (548 MB, not committed)'); process.exit(0); }

// An http(s) URL exercises loadGPT2's full browser path (fetch → stream → parse);
// a file path tests parse + forward directly. node 20 has fetch + streaming.
let model;
if (/^https?:/.test(wpath)) {
  model = await loadGPT2(wpath, GPT2_CONFIG.gpt2, () => {});
  console.log('  (loaded via loadGPT2 streaming fetch — the in-browser code path)');
} else {
  const buf = readFileSync(wpath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  model = new GPT2(parseSafetensors(ab), GPT2_CONFIG.gpt2);
}
const ids = fix.ids, n = ids.length;
const out = model.forward(ids);

let pass = 0, fail = 0, worst = 0;
const eq = (c, msg) => { if (c) pass++; else { fail++; console.error('FAIL ' + msg); } };

// 1) attention matrices match PyTorch (tol 5e-3: fixture rounded to 4dp + fp order)
for (const m of fix.matrices) {
  const A = out.attentions[m.l][m.h]; let mx = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) mx = Math.max(mx, Math.abs(A[i * n + j] - m.A[i][j]));
  worst = Math.max(worst, mx);
  eq(mx < 5e-3, `attn (l=${m.l},h=${m.h}) max|Δ|=${mx.toExponential(2)} (tol 5e-3)`);
  if (mx < 5e-3) console.log(`  ok  attn (l=${m.l},h=${m.h}) max|Δ|=${mx.toExponential(2)}`);
}

// 2) JS detectors pick the same top head as the PyTorch detectors
const prev = (A) => { let s = 0; for (let i = 1; i < n; i++) s += A[i * n + i - 1]; return s / (n - 1); };
const sink = (A) => { let s = 0; for (let i = 0; i < n; i++) s += A[i * n]; return s / n; };
const induction = (A) => { let s = 0, c = 0; for (let i = 0; i < n; i++) { let j = -1; for (let k = i - 1; k >= 0; k--) if (ids[k] === ids[i]) { j = k; break; } if (j >= 0 && j + 1 < i) { s += A[i * n + j + 1]; c++; } } return c ? s / c : 0; };
const argmax = (fn) => { let best = -1, bv = -Infinity, L = out.nLayer, H = out.nHead; for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) { const v = fn(out.attentions[l][h]); if (v > bv) { bv = v; best = [l, h, v]; } } return best; };

for (const [name, fn, top] of [['prev', prev, fix.top_prev], ['sink', sink, fix.top_sink], ['induction', induction, fix.top_induction]]) {
  const [l, h, v] = argmax(fn), [tl, th] = top[0];
  eq(l === tl && h === th, `top ${name} head: JS=(${l},${h}) PyTorch=(${tl},${th})`);
  if (l === tl && h === th) console.log(`  ok  top ${name} head (${l},${h}) score=${v.toFixed(3)} matches PyTorch`);
}

// 3) next-token logits (GPT2.logits) match PyTorch: argmax + top-token logit values
if (fix.logits) {
  const { logits, V } = model.logits(ids);
  eq(V === fix.logits.V, `vocab size V=${V} (expect ${fix.logits.V})`);
  let am = 0; for (let v = 1; v < V; v++) if (logits[v] > logits[am]) am = v;
  eq(am === fix.logits.argmax_id, `argmax next-token id JS=${am} PyTorch=${fix.logits.argmax_id}`);
  let lw = 0; for (const t of fix.logits.top) lw = Math.max(lw, Math.abs(logits[t.id] - t.logit));
  eq(lw < 2e-2, `top-token logits max|Δ|=${lw.toExponential(2)} (tol 2e-2)`);
  if (am === fix.logits.argmax_id && lw < 2e-2) console.log(`  ok  next-token logits: argmax id ${am}, top-token logit max|Δ|=${lw.toExponential(2)}`);
}

// 4) logit lens (GPT2.lens): per-layer top-1 token id at every depth matches PyTorch
if (fix.lens) {
  const { perLayer, nLayer } = model.lens(ids);
  eq(perLayer.length === fix.lens.n_hidden, `lens depths = ${perLayer.length} (expect ${fix.lens.n_hidden} = nLayer+1)`);
  const jsTop = perLayer.map((lg) => { let a = 0; for (let v = 1; v < lg.length; v++) if (lg[v] > lg[a]) a = v; return a; });
  const match = jsTop.every((id, i) => id === fix.lens.lens_top_ids[i]);
  eq(match, `lens per-layer top ids: JS=[${jsTop}] PyTorch=[${fix.lens.lens_top_ids}]`);
  if (match) console.log(`  ok  logit lens: ${perLayer.length} depths, per-layer top-1 ids all match PyTorch (final id ${jsTop[nLayer]})`);
}

// 5) greedy generation (the real-generate loop): argmax(logits) → append → repeat
//    must reproduce HF model.generate(do_sample=False)
if (fix.greedy) {
  const seq = ids.slice(), got = [];
  for (let k = 0; k < fix.greedy.n_new; k++) {
    const { logits, V } = model.logits(seq);
    let a = 0; for (let v = 1; v < V; v++) if (logits[v] > logits[a]) a = v;
    seq.push(a); got.push(a);
  }
  const gmatch = got.every((id, i) => id === fix.greedy.greedy_ids[i]);
  eq(gmatch, `greedy gen ${fix.greedy.n_new} tokens: JS=[${got}] PyTorch=[${fix.greedy.greedy_ids}]`);
  if (gmatch) console.log(`  ok  greedy generation: ${got.length} tokens match HF generate(do_sample=False)`);
}

console.log(`\n${pass} passed, ${fail} failed (worst attn |Δ| = ${worst.toExponential(2)})`);
process.exit(fail ? 1 : 0);
