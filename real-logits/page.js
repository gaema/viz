// real-logits concept page.
//
// The synthetic lm-head + sampling pages show the mechanism (hidden → vocab
// logits → softmax → pick a token) on seeded numbers. This page shows the REAL
// numbers: it runs the verified GPT-2 forward (gpt2.js) + ln_f + the tied lm_head
// to get the actual next-token logits for a prefix you type, then draws the
// probability distribution and lets you reshape it with temperature / top-k /
// top-p exactly as a sampler would. "the cat ran" → up / away / off / out, real.
//
// Breadcrumbs to ../design/architectures.md A5 (lm_head) + the sampling concept.
// Reuses the real-attention GPT-2 loader (gpt2.js).
//
// Offline / no network: an idealized synthetic distribution (clearly labelled)
// over plausible continuations, so the sampling controls still teach; the real
// model swaps in once the ~548 MB weights download. ?real=0 forces synthetic.
import { mount } from '../framework/layout.js';
import { loadGPT2, GPT2_CONFIG } from '../real-attention/gpt2.js';
import { getWebGPU, GPT2WebGPU } from '../real-attention/gpt2-webgpu.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2;
const GREEN = '#0a7227', AMBER = '#9a6700', BLUE = '#1f6feb', GREY = '#c8ccd1';

const DEFAULT_PROMPT = 'the cat sat on the mat . the cat ran';
// idealized synthetic stand-in: plausible continuations + descending logits
const SYN = [[' away', 3.1], [' up', 3.0], [' to', 2.8], [' off', 2.6], [' out', 2.4], [' down', 2.2], [' into', 2.0], [' across', 1.8], [' back', 1.7], [' around', 1.5], [' through', 1.3], [' for', 1.1], [' over', 1.0], [' and', 0.9], [' in', 0.7], [' past', 0.6], [' toward', 0.4], [' home', 0.3], [' fast', 0.1], [' .', 0.0], [' ,', -0.2], [' the', -0.4]];

let M = { status: 'init', progress: 0, source: 'synthetic', prompt: DEFAULT_PROMPT, dist: [], top1: null, n: 0, backend: 'cpu', gpuAvail: false, gpuCheck: null, times: {} };
let tokenizer = null, gpt2 = null, gpu = null, webgpu = null, loadStarted = false;
let barRects = [];

const td = (s) => (s == null ? '' : String(s)).replace(/^ /, '·').replace(/\n/g, '⏎') || '∅';

// Build the display distribution from a logit source: full softmax(@temperature)
// → top-N by prob → mark which survive top-k / top-p (the sampler's keep-set).
function buildDist(getLogit, V, label, { temp, topk, topp }, decode, N = 22) {
  const t = Math.max(0.02, temp);
  let mx = -Infinity; for (let v = 0; v < V; v++) { const z = getLogit(v); if (z > mx) mx = z; }
  let Z = 0; for (let v = 0; v < V; v++) Z += Math.exp((getLogit(v) - mx) / t);
  // top-N by probability in one pass (insertion into a small sorted array)
  const top = [];
  for (let v = 0; v < V; v++) {
    const p = Math.exp((getLogit(v) - mx) / t) / Z;
    if (top.length < N) { top.push({ v, p }); if (top.length === N) top.sort((a, b) => b.p - a.p); }
    else if (p > top[N - 1].p) { top[N - 1] = { v, p }; let i = N - 1; while (i > 0 && top[i].p > top[i - 1].p) { const s = top[i]; top[i] = top[i - 1]; top[i - 1] = s; i--; } }
  }
  top.sort((a, b) => b.p - a.p);
  // sampler keep-set: top-k (k>0) then top-p (cumulative ≥ p over the sorted list)
  let cum = 0;
  for (let i = 0; i < top.length; i++) {
    const overK = topk > 0 && i >= topk;
    const overP = topp < 1 && cum >= topp + 1e-9;
    top[i].kept = !overK && !overP;
    cum += top[i].p;
  }
  return top.map((e) => ({ tok: decode(e.v), prob: e.p, kept: e.kept }));
}

function setDist(dist, source, status) { M = { ...M, dist, top1: dist[0] || null, source, status }; }

function compute(page) {
  const st = page.state, opts = { temp: +st.temp, topk: st.topk | 0, topp: +st.topp };
  if (M.source === 'real' && gpt2 && M._logits) {
    const { logits, V } = M._logits;
    setDist(buildDist((v) => logits[v], V, 'real', opts, (v) => tokenizer.decode([v])), 'real', 'ready');
  } else {
    setDist(buildDist((v) => SYN[v][1], SYN.length, 'syn', opts, (v) => SYN[v][0]), 'synthetic', M.status);
  }
}

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function ensureReal(page) {
  if (gpt2 || loadStarted) return;
  loadStarted = true; M.status = 'loading-tok'; page.redraw();
  try {
    const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js');
    env.allowLocalModels = false; tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL);
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(page.state.weights || WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    // bring up the WebGPU backend if available (shares the parsed weights Map)
    try { const wg = await getWebGPU(); if (wg) { const g = new GPT2WebGPU(gpt2.w, CFG, wg.device); await g.init(); gpu = g; webgpu = wg; M.gpuAvail = true; } } catch (ge) { gpu = null; M.gpuErr = String(ge && ge.message || ge); }
    M.status = 'running'; page.redraw();
    await runReal(page, page.state.prompt || DEFAULT_PROMPT);
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

async function runReal(page, prompt) {
  if (!gpt2 || !tokenizer) return;
  M.status = 'running'; page.redraw();
  try {
    const enc = await tokenizer(prompt || DEFAULT_PROMPT, { add_special_tokens: false });
    const ids = Array.from(enc.input_ids.data, Number);
    if (!ids.length) return;
    M.prompt = prompt; M.n = ids.length;
    // pick backend; lm_head + ln_f run on the GPU when WebGPU is on
    const useGpu = !!(gpu && page.state.gpu);
    let res, backend = 'cpu';
    if (useGpu) {
      try { const t0 = performance.now(); res = await gpu.logits(ids); M.times.gpu = performance.now() - t0; backend = 'gpu'; }
      catch (ge) { gpu = null; M.gpuErr = String(ge && ge.message || ge); }   // adapter died → CPU
    }
    if (backend !== 'gpu') { const c0 = performance.now(); res = gpt2.logits(ids); M.times.cpu = performance.now() - c0; }
    else if (!M.gpuCheck) {                                                   // self-verify GPU vs CPU once
      const c0 = performance.now(); const cr = gpt2.logits(ids); M.times.cpu = performance.now() - c0;
      let mx = 0; for (let v = 0; v < res.V; v++) { const dd = Math.abs(res.logits[v] - cr.logits[v]); if (dd > mx) mx = dd; }
      M.gpuCheck = { maxDiff: mx, ok: mx < 1e-2 };
      if (!M.gpuCheck.ok) { res = cr; backend = 'cpu'; page.controls.set('gpu', false); }   // mismatch → trust CPU
    }
    M._logits = res; M.source = 'real'; M.backend = backend;
    compute(page); page.redraw();
  } catch (e) { M.status = 'offline'; M.err = String(e); page.redraw(); }
}

mount({
  mount: 'body',
  slug: 'real-logits',
  title: 'real logits — what GPT-2 actually predicts next',
  blurb: 'The synthetic lm-head + sampling pages show the mechanism on seeded numbers; this page runs a REAL GPT-2 in your browser — the verified forward + ln_f + the tied lm_head — to get the actual next-token logits for a prefix you type, then draws the probability distribution. Reshape it with temperature, top-k, and top-p exactly as a sampler would, and watch which tokens survive. Offline it shows an idealized stand-in; the real model needs a ~548 MB one-time download.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c, page) => {
    c.text('prompt', { label: 'prompt', value: DEFAULT_PROMPT, placeholder: 'type a prefix…', rebuild: false });
    c.button('re-run', () => { if (gpt2) runReal(page, page.state.prompt); else page.redraw(); });
    c.slider('temp', { label: 'temperature', min: 0.1, max: 2, step: 0.05, value: 1 });
    c.stepper('topk', { label: 'top-k (0=off)', min: 0, max: 40, value: 0 });
    c.slider('topp', { label: 'top-p', min: 0.1, max: 1, step: 0.05, value: 1 });
    c.toggle('gpu', { label: 'WebGPU compute', value: true });
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    r.clear('#ffffff');
    compute(page);                                   // recompute on any control change
    const d = M.dist, kept = d.filter((e) => e.kept);
    page.probe = { source: M.source, temp: +page.state.temp, topk: page.state.topk | 0, topp: +page.state.topp, topProb: M.top1 ? M.top1.prob : 0, nKept: kept.length };

    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: AMBER };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100 | 0)}% (~548 MB, one time)`, c: AMBER };
      if (M.status === 'running') return { t: '⟳ running GPT-2…', c: AMBER };
      if (M.source === 'real') {
        const be = M.backend === 'gpu' ? 'WebGPU' : 'CPU', mism = M.gpuCheck && !M.gpuCheck.ok;
        const chk = M.gpuCheck ? `  ${mism ? '⚠ GPU≠CPU → CPU' : '✓ GPU=CPU'} (max|Δ|=${M.gpuCheck.maxDiff.toExponential(1)})` : (M.gpuAvail ? '' : '  (WebGPU n/a → CPU)');
        const tm = (M.times.gpu && M.times.cpu) ? `  ·  GPU ${M.times.gpu | 0}ms vs CPU ${M.times.cpu | 0}ms` : '';
        return { t: `● real GPT-2 (124M) — ${be} lm_head${chk}${tm}`, c: mism ? AMBER : GREEN };
      }
      if (M.status === 'offline') return { t: '○ offline — idealized synthetic stand-in (click “load real GPT-2”)', c: AMBER };
      return { t: '○ idealized synthetic stand-in — click “load real GPT-2” for real predictions', c: '#586069' };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();
    if (!d.length) { page.setReadout('…'); return; }

    // prompt + predicted next token
    const pad = 16, promptY = 34;
    ctx.save(); ctx.font = '13px ui-monospace, monospace'; ctx.textBaseline = 'middle';
    ctx.fillStyle = '#3a4047'; ctx.fillText(M.source === 'real' ? `“${(M.prompt || page.state.prompt).slice(0, 64)}”` : '“…”', pad, promptY);
    const pw = ctx.measureText(M.source === 'real' ? `“${(M.prompt || page.state.prompt).slice(0, 64)}”` : '“…”').width;
    ctx.fillStyle = BLUE; ctx.font = '13px ui-monospace, monospace';
    ctx.fillText(`→ next: "${(M.top1.tok || '').trim() || M.top1.tok}"  (${(M.top1.prob * 100).toFixed(1)}%)`, pad + pw + 10, promptY); ctx.restore();

    // distribution bars
    const top = promptY + 24, labW = 92, rowH = Math.max(13, Math.min(24, (page.H - top - 40) / d.length));
    const barX = pad + labW, barW = page.W - barX - 70, pmax = d[0].prob || 1;
    barRects = [];
    ctx.save(); ctx.textBaseline = 'middle'; ctx.font = `${Math.min(12, rowH - 3)}px ui-monospace, monospace`;
    for (let i = 0; i < d.length; i++) {
      const e = d[i], y = top + i * rowH, w = Math.max(1, (e.prob / pmax) * barW);
      ctx.fillStyle = e.kept ? (i === 0 ? GREEN : BLUE) : GREY;
      ctx.fillRect(barX, y + 1, w, rowH - 3);
      ctx.fillStyle = e.kept ? '#24292e' : '#9aa4ad'; ctx.textAlign = 'right';
      ctx.fillText(td(e.tok).slice(0, 11), barX - 5, y + rowH / 2);
      ctx.textAlign = 'left'; ctx.fillStyle = e.kept ? '#444' : '#b6bcc2';
      ctx.fillText(`${(e.prob * 100).toFixed(1)}%`, barX + w + 5, y + rowH / 2);
    }
    ctx.restore();

    const cutN = d.length - kept.length;
    const o = `${M.source === 'real' ? `REAL GPT-2 (124M, ${M.backend === 'gpu' ? 'WebGPU' : 'CPU'})` : 'synthetic (idealized)'} · next-token distribution · temp=${(+page.state.temp).toFixed(2)} top-k=${page.state.topk | 0 || 'off'} top-p=${(+page.state.topp).toFixed(2)}    tier:${r.name}\n` +
      `greedy: "${(M.top1.tok || '').trim() || M.top1.tok}" (${(M.top1.prob * 100).toFixed(1)}%) · ${kept.length} tokens kept by the sampler, ${cutN}+ cut (grey)` +
      (M.source === 'real' ? '  — real probabilities. Lower temperature to sharpen, raise to flatten.' : '  — idealized stand-in; load real GPT-2 for true predictions.');
    page.setReadout(o);
  },
  challenges: [
    { goal: 'Ground it in the REAL model — compute true next-token logits in-browser (needs network; “load real GPT-2”).',
      hint: 'The banner turns green “● real GPT-2” after the ~548 MB one-time download.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Use nucleus (top-p) sampling — lower top-p until the sampler keeps 5 or fewer tokens (the grey ones are cut).',
      hint: 'Top-p keeps the smallest set whose probabilities sum to p. Drag top-p down from 1.0.',
      check: (api) => ({ solved: api.probe.topp < 1 && api.probe.nKept <= 5, detail: `top-p=${api.probe.topp.toFixed(2)} keeps ${api.probe.nKept} tokens (need ≤ 5, top-p < 1)` }) },
  ],
}).then((page) => {
  window.__realLogitsPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('prompt')) page.controls.set('prompt', q.get('prompt'));
  if (q.has('temp')) page.controls.set('temp', +q.get('temp'));
  if (q.has('topk')) page.controls.set('topk', +q.get('topk'));
  if (q.has('topp')) page.controls.set('topp', +q.get('topp'));
  compute(page); page.redraw();
  if (q.get('real') !== '0') ensureReal(page);
});
