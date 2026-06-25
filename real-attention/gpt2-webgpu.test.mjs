// Correctness gate for the WebGPU backend: GPT2WebGPU must reproduce the CPU
// GPT2 forward (which gpt2.test.mjs already pins to PyTorch). We compare on a
// TINY random model — no weights download, no real GPU needed beyond a WebGPU
// adapter — so it runs anywhere WebGPU is exposed.
//
//   deno run --allow-read gpt2-webgpu.test.mjs        (Deno ships WebGPU)
//   node  --experimental-... (with a WebGPU polyfill / Dawn-node)
//
// Plain Node has no navigator.gpu, so it SKIPS (exit 0). The page also runs this
// exact check live (GPU vs CPU on the real weights) and falls back to CPU on any
// mismatch — see analyze() in page.js — so the GPU path is self-verifying in the
// browser regardless of CI.
import { GPT2, GPT2_CONFIG } from './gpt2.js';
import { getWebGPU, GPT2WebGPU } from './gpt2-webgpu.js';

const wg = await getWebGPU();
if (!wg) { console.log('SKIP: no WebGPU adapter (navigator.gpu absent) — run under Deno or a WebGPU-capable runtime'); process.exit(0); }

// deterministic tiny random model
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry32(12345);
const rand = (len) => { const a = new Float32Array(len); for (let i = 0; i < len; i++) a[i] = rng() * 2 - 1; return a; };

const cfg = { nLayer: 2, nHead: 2, nEmbd: 16 }, D = cfg.nEmbd, V = 20, POS = 16;
const W = new Map();
const put = (name, shape) => W.set(name, { shape, data: rand(shape.reduce((a, b) => a * b, 1)) });
put('wte.weight', [V, D]); put('wpe.weight', [POS, D]);
for (let l = 0; l < cfg.nLayer; l++) {
  const p = `h.${l}.`;
  put(p + 'ln_1.weight', [D]); put(p + 'ln_1.bias', [D]);
  put(p + 'attn.c_attn.weight', [D, 3 * D]); put(p + 'attn.c_attn.bias', [3 * D]);
  put(p + 'attn.c_proj.weight', [D, D]); put(p + 'attn.c_proj.bias', [D]);
  put(p + 'ln_2.weight', [D]); put(p + 'ln_2.bias', [D]);
  put(p + 'mlp.c_fc.weight', [D, 4 * D]); put(p + 'mlp.c_fc.bias', [4 * D]);
  put(p + 'mlp.c_proj.weight', [4 * D, D]); put(p + 'mlp.c_proj.bias', [D]);
}
put('ln_f.weight', [D]); put('ln_f.bias', [D]);   // final norm, for the lm_head path

const ids = [3, 7, 1, 7, 2], n = ids.length;
const cpuModel = new GPT2(W, cfg);
const cpu = cpuModel.forward(ids);
const gpuModel = new GPT2WebGPU(W, cfg, wg.device); await gpuModel.init();
const gpu = await gpuModel.forward(ids);

let pass = true;
// 1) attention forward matches
let worst = 0;
for (let l = 0; l < cfg.nLayer; l++) for (let h = 0; h < cfg.nHead; h++) {
  const a = cpu.attentions[l][h], b = gpu.attentions[l][h];
  for (let k = 0; k < n * n; k++) worst = Math.max(worst, Math.abs(a[k] - b[k]));
}
const okA = worst < 1e-3; pass = pass && okA;
console.log(`${okA ? 'PASS' : 'FAIL'}: WebGPU vs CPU attention forward, ${cfg.nLayer}L×${cfg.nHead}H, n=${n}, worst |Δ| = ${worst.toExponential(2)} (tol 1e-3)`);

// 2) lm_head logits match (ln_f + tied lm_head on the GPU vs CPU)
const cl = cpuModel.logits(ids), gl = await gpuModel.logits(ids);
let worstL = 0; for (let v = 0; v < cl.V; v++) worstL = Math.max(worstL, Math.abs(cl.logits[v] - gl.logits[v]));
let cam = 0, gam = 0; for (let v = 1; v < cl.V; v++) { if (cl.logits[v] > cl.logits[cam]) cam = v; if (gl.logits[v] > gl.logits[gam]) gam = v; }
const okL = worstL < 1e-3 && cam === gam; pass = pass && okL;
console.log(`${okL ? 'PASS' : 'FAIL'}: WebGPU vs CPU logits, V=${cl.V}, argmax GPU=${gam}/CPU=${cam}, worst |Δ| = ${worstL.toExponential(2)} (tol 1e-3)`);

process.exit(pass ? 0 : 1);
