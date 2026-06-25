// Verifies the real-quant path: fetchTensor() (Range-read of one safetensors
// tensor) + quant.js groupQuant/stats reproduce numpy on the real GPT-2 weight
// (quant-groundtruth.fixture.json). Node 20 has fetch + Range.
//
// Needs the gpt2 safetensors served over http (Range-capable, e.g. python
// http.server) — point WEIGHTS_URL at it:
//   WEIGHTS_URL=http://127.0.0.1:8011/model.safetensors node real-quant/quant.test.mjs
// Skips (exit 0) if WEIGHTS_URL is unset.
import { readFileSync } from 'node:fs';
import { fetchTensor } from '../real-attention/gpt2.js';
import { groupQuant, stats } from './quant.js';

const fix = JSON.parse(readFileSync(new URL('./quant-groundtruth.fixture.json', import.meta.url)));
const url = process.env.WEIGHTS_URL;
if (!url) { console.log('SKIP: set WEIGHTS_URL=http://…/model.safetensors (Range-capable) to run'); process.exit(0); }

let pass = 0, fail = 0;
const eq = (c, msg) => { if (c) pass++; else { fail++; console.error('FAIL ' + msg); } };

const t = await fetchTensor(url, fix.name);
const w = t.data, n = w.length;

eq(JSON.stringify(t.shape) === JSON.stringify(fix.shape) && n === fix.n, `shape ${JSON.stringify(t.shape)} n=${n} (expect ${JSON.stringify(fix.shape)} ${fix.n})`);
let fmax = 0; for (let i = 0; i < 8; i++) fmax = Math.max(fmax, Math.abs(w[i] - fix.first8[i]));
eq(fmax < 1e-5, `first-8 values max|Δ|=${fmax.toExponential(2)} (Range-read correct)`);
const st = stats(w, n);
const sd = Math.max(...['min', 'max', 'mean', 'std'].map((k) => Math.abs(st[k] - fix.stats[k])));
eq(sd < 1e-4, `stats max|Δ|=${sd.toExponential(2)} (min/max/mean/std vs numpy)`);
let qd = 0;
for (const [bits, group] of [[4, 64], [4, 128], [8, 64], [8, 128]]) {
  const r = groupQuant(w, n, bits, group), ref = fix.rmse[`b${bits}_g${group}`];
  qd = Math.max(qd, Math.abs(r.rmse - ref));
}
eq(qd < 1e-4, `group-quant RMSE max|Δ|=${qd.toExponential(2)} (int4/int8 × g64/g128 vs numpy)`);
if (!fail) console.log(`  ok  real weight ${fix.name} ${JSON.stringify(t.shape)}: Range-read + stats + group-quant RMSE all match numpy`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
