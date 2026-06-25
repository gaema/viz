// real-quant concept page — Phase 9 (real-model grounding).
//
// The synthetic `quantization` page shows int4 group-quant on seeded weights;
// this one shows it on REAL GPT-2 weights. It range-fetches a single weight
// tensor (~9 MB, not the whole 548 MB — fetchTensor() uses HTTP Range), draws its
// distribution, and applies group-wise symmetric int-N quantization to show the
// REAL error: real weights are a tight Gaussian (σ≈0.14) with a few outliers, so
// per-group scales + more bits drop the RMSE — exactly why low-bit quant works.
//
// Math in ./quant.js (verified vs numpy by quant.test.mjs). Breadcrumbs to A-attr
// quantization. Plan: ../plan/phase9.md. Reuses ../real-attention/gpt2.js fetchTensor.
//
// Offline: a synthetic Gaussian+outliers tensor (labelled), so the bits/group
// controls still teach. ?real=0 forces it.
import { mount } from '../framework/layout.js';
import { fetchTensor } from '../real-attention/gpt2.js';
import { groupQuant, stats, histogram } from './quant.js';

const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const GREEN = '#0a7227', AMBER = '#9a6700', BLUE = '#1f6feb', INK = '#24292e';
const TENSORS = ['h.0.mlp.c_fc.weight', 'h.0.attn.c_attn.weight', 'h.6.mlp.c_fc.weight', 'h.11.mlp.c_fc.weight'];
const GROUPS = { '32': 32, '64': 64, '128': 128, 'whole row (3072)': 3072 };

let M = { status: 'init', source: 'synthetic', name: TENSORS[0], shape: null, w: null, n: 0, st: null, fetching: false };
let loadStarted = false;

// deterministic synthetic stand-in: Gaussian σ0.14 + a few outliers
const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
function synthTensor() {
  const n = 768 * 3072, w = new Float32Array(n), rng = mulberry32(0x5eed);
  for (let i = 0; i < n; i++) { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); w[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 0.14; }
  for (let k = 0; k < 200; k++) w[(rng() * n) | 0] *= 1 + rng() * 25;   // outliers
  return { shape: [768, 3072], data: w };
}

function setTensor(name, t, source) { M = { ...M, name, shape: t.shape, w: t.data, n: t.data.length, st: stats(t.data, t.data.length), source }; }

async function ensureReal(page) {
  if (loadStarted) return; loadStarted = true;
  await fetchFor(page, page.state.tensor || M.name);
}
async function fetchFor(page, name) {
  if (M.fetching) return; M.fetching = true; M.status = 'loading'; page.redraw();
  try { const t = await fetchTensor(page.state.weights || WEIGHTS_URL, name); setTensor(name, t, 'real'); M.status = 'ready'; }
  catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); }
  M.fetching = false; page.redraw();
}

mount({
  mount: 'body',
  slug: 'real-quant',
  title: 'real quant — int4 on actual GPT-2 weights',
  blurb: 'Phase 9 (real-model grounding). The synthetic quantization page shows int4 group-quant on seeded weights; this one runs it on REAL GPT-2 weights — it range-fetches a single weight tensor (~9 MB, not the whole 548 MB) and shows its distribution + the real quantization error. Real weights are a tight Gaussian (σ≈0.14) with a few big outliers, so per-group scales and more bits drop the error fast — exactly why int4/int8 work. Pick a tensor, drag bits and group size.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  controls: (c, page) => {
    c.select('tensor', { label: 'weight tensor', options: TENSORS, value: TENSORS[0] });
    c.slider('bits', { label: 'bits', min: 2, max: 8, step: 1, value: 4 });
    c.select('group', { label: 'group size', options: Object.keys(GROUPS), value: '64' });
    c.button('load real GPT-2 weights', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear('#ffffff');
    // tensor switch → re-fetch (real mode only)
    if (M.source === 'real' && !M.fetching && st.tensor && st.tensor !== M.name) { fetchFor(page, st.tensor); }
    if (!M.w) {                                              // first paint: synthetic stand-in
      const t = synthTensor(); setTensor(M.name, t, M.source === 'real' ? 'real' : 'synthetic');
    }
    const bits = st.bits | 0, group = GROUPS[st.group] || 64, w = M.w, n = M.n, S = M.st;
    const q = groupQuant(w, n, bits, group);
    page.probe = { source: M.source, bits, group, rmse: q.rmse, std: S.std, name: M.name };

    const ban = (() => {
      if (M.status === 'loading') return { t: `↓ range-fetching ${M.name}…`, c: AMBER };
      if (M.source === 'real') return { t: `● real GPT-2 weight — ${M.name} ${JSON.stringify(M.shape)} (Range-fetched, ~${(n * 4 / 1e6).toFixed(1)} MB)`, c: GREEN };
      if (M.status === 'offline') return { t: '○ offline — synthetic Gaussian+outliers stand-in (click “load real GPT-2 weights”)', c: AMBER };
      return { t: '○ synthetic stand-in — click “load real GPT-2 weights” for actual weights', c: '#586069' };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();

    const pad = 16, topY = 40;
    // ---- left: weight histogram (clamped to ±5σ; outliers fall in the edge bins) ----
    const lo = S.mean - 5 * S.std, hi = S.mean + 5 * S.std, BINS = 90;
    const h = histogram(w, n, BINS, lo, hi);
    let hmax = 1; for (let i = 0; i < BINS; i++) if (h[i] > hmax) hmax = h[i];
    const hw = page.W * 0.55, hx = pad + 8, hy = topY + 18, hH = page.H - hy - 46;
    r.label(`weight distribution — ${n.toLocaleString()} values, σ=${S.std.toFixed(3)}, range [${S.min.toFixed(2)}, ${S.max.toFixed(2)}]`, hx, topY, { color: '#586069', font: '11px ui-monospace, monospace' });
    ctx.save();
    const bw = (hw - 20) / BINS;
    for (let i = 0; i < BINS; i++) { const bh = (h[i] / hmax) * hH; ctx.fillStyle = '#9ec1ef'; ctx.fillRect(hx + i * bw, hy + hH - bh, Math.max(1, bw - 0.5), bh); }
    // quant levels for group 0 (representative): ticks at q·scale
    const qmax = (1 << (bits - 1)) - 1, scale0 = q.scales[0];
    ctx.strokeStyle = 'rgba(207,34,46,0.55)'; ctx.lineWidth = 1;
    for (let qi = -qmax; qi <= qmax; qi++) { const v = qi * scale0; if (v < lo || v > hi) continue; const x = hx + ((v - lo) / (hi - lo)) * (hw - 20); ctx.beginPath(); ctx.moveTo(x, hy); ctx.lineTo(x, hy + hH); ctx.stroke(); }
    ctx.fillStyle = '#cf222e'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
    ctx.fillText(`${(2 ** bits)} levels (int${bits}, group 0 scale)`, hx + 4, hy + 11);
    ctx.restore();

    // ---- right: RMSE vs bits (current group), current highlighted ----
    const rx = hx + hw + 24, rw = page.W - rx - pad, ry = topY + 18, rH = hH;
    r.label(`quantization error (RMSE) vs bits — group ${group}`, rx, topY, { color: '#586069', font: '11px ui-monospace, monospace' });
    const bitsList = [2, 3, 4, 5, 6, 8], rmses = bitsList.map((b) => groupQuant(w, n, b, group).rmse);
    const rmax = Math.max(...rmses) || 1, cbw = (rw - 10) / bitsList.length;
    ctx.save(); ctx.textBaseline = 'alphabetic';
    for (let i = 0; i < bitsList.length; i++) {
      const b = bitsList[i], bh = (rmses[i] / rmax) * (rH - 24), x = rx + i * cbw, y = ry + rH - bh;
      ctx.fillStyle = b === bits ? BLUE : '#cdd3da'; ctx.fillRect(x + 4, y, cbw - 10, bh);
      ctx.fillStyle = b === bits ? INK : '#9aa4ad'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText(`int${b}`, x + cbw / 2, ry + rH + 12);
      ctx.fillText(rmses[i] < 0.01 ? rmses[i].toExponential(1) : rmses[i].toFixed(3), x + cbw / 2, y - 3);
    }
    ctx.restore();

    const o = `${M.source === 'real' ? 'REAL GPT-2' : 'synthetic'} · ${M.name} · group-wise symmetric int${bits} · group=${group} → RMSE = ${q.rmse < 0.01 ? q.rmse.toExponential(2) : q.rmse.toFixed(4)}    tier:${r.name}\n` +
      `real weights are ~Gaussian (σ=${S.std.toFixed(3)}) with outliers to ${Math.max(Math.abs(S.min), Math.abs(S.max)).toFixed(1)}; per-group scales + more bits drop the error` +
      (M.source === 'real' ? '.' : ' (load real weights to see GPT-2’s actual distribution).');
    page.setReadout(o);
  },
  challenges: [
    { goal: 'Ground it in REAL weights — range-fetch an actual GPT-2 weight tensor (needs network; “load real GPT-2 weights”).',
      hint: 'It fetches ~9 MB (one tensor) via HTTP Range, not the whole 548 MB. Banner turns green.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Quantize to under 0.005 RMSE — int4 is ~0.017; raise bits (and/or shrink the group) until the error drops below 0.005.',
      hint: 'int8 is far finer than int4; smaller groups give each block its own scale. Watch the RMSE bars.',
      check: (api) => ({ solved: api.probe.rmse < 0.005, detail: `RMSE = ${(+api.probe.rmse).toExponential(2)} (need < 0.005)` }) },
  ],
}).then((page) => {
  window.__realQuantPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('tensor') && TENSORS.includes(q.get('tensor'))) page.controls.set('tensor', q.get('tensor'));
  if (q.has('bits')) page.controls.set('bits', Math.max(2, Math.min(8, +q.get('bits'))));
  if (q.has('group') && GROUPS[q.get('group')]) page.controls.set('group', q.get('group'));
  page.redraw();
  if (q.get('real') !== '0') ensureReal(page);
});
