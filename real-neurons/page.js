// real-neurons concept page — real-model grounding.
//
// Each GPT-2 block's MLP has 3072 "neurons" — the post-gelu units between the two
// projections. This page runs a REAL GPT-2 in the browser (GPT2.mlp(), verified
// vs PyTorch) and shows which neurons fire for which tokens: the heatmap of the
// most-active neurons across the sentence, with the strongest (token, neuron)
// firing marked. Different neurons respond to different tokens — the raw material
// of mechanistic interpretability.
//
// Breadcrumbs to the MLP / SwiGLU concept (../mlp-gated).
// Reuses ../real-attention/gpt2.js (mlp()).
//
// Offline: a synthetic gelu-ish activation stand-in (labelled). ?real=0 forces it.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { T, rgbaToken } from '../framework/theme.js';
import { loadGPT2, GPT2_CONFIG } from '../real-attention/gpt2.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2, DFF = 4 * CFG.nEmbd;

const DEFAULT_TEXT = 'the cat sat on the mat . the cat ran';
const DEFAULT_IDS = [1169, 3797, 3332, 319, 262, 2603, 764, 262, 3797, 4966];
const DEFAULT_TOKS = ['the', ' cat', ' sat', ' on', ' the', ' mat', ' .', ' the', ' cat', ' ran'];

let M = { status: 'init', progress: 0, source: 'synthetic', ids: DEFAULT_IDS, tokens: DEFAULT_TOKS, n: 0, mlp: null };
let tokenizer = null, gpt2 = null, loadStarted = false, injected = null;
let hmRect = null, neuronIds = [];
const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const td = (s) => (s == null ? '' : String(s)).replace(/^ /, '·').replace(/\n/g, '⏎') || '∅';

// synthetic gelu-ish activations: mostly small, a few strong, per layer [n×DFF]
function synthMLP(n) {
  const layers = [];
  for (let l = 0; l < CFG.nLayer; l++) {
    const a = new Float32Array(n * DFF), rng = mulberry32(0x51 ^ (l * 131 + n));
    for (let i = 0; i < n * DFF; i++) { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); a[i] = g > 0 ? g * 0.5 : g * 0.1; }
    for (let k = 0; k < n; k++) a[k * DFF + ((rng() * DFF) | 0)] = 0.8 + rng() * (0.6 + l * 0.35);   // a few strong firers, deeper = harder (≈ real)
    layers.push(a);
  }
  return { layers, n, dFF: DFF };
}

function setMLP(ids, tokens, mlp, source, status) { M = { ...M, ids, tokens, n: ids.length, mlp, source, status }; }

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function ensureReal(page) {
  if (gpt2 || loadStarted) return; loadStarted = true;
  try {
    if (!injected) { M.status = 'loading-tok'; page.redraw(); const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js'); env.allowLocalModels = false; tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL); }
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(page.state.weights || WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    M.status = 'running'; M.source = 'real'; page.redraw();
    await analyze(page, page.state.text || DEFAULT_TEXT);
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

async function analyze(page, text) {
  if (!gpt2) return; M.status = 'running'; page.redraw();
  try {
    let ids, tokens;
    if (injected) { ids = injected; tokens = ids.map((x) => '·' + x); }
    else { const enc = await tokenizer(text || DEFAULT_TEXT, { add_special_tokens: false }); ids = Array.from(enc.input_ids.data, Number).slice(0, 24); tokens = ids.map((x) => tokenizer.decode([x])); }
    if (!ids.length) return;
    setMLP(ids, tokens, gpt2.mlp(ids), 'real', 'ready'); page.redraw();
  } catch (e) { M.status = 'offline'; M.err = String(e); page.redraw(); }
}

mount({
  mount: 'body',
  slug: 'real-neurons',
  title: 'real neurons — which MLP units fire',
  blurb: 'Each transformer block’s MLP has 3072 “neurons” — the post-gelu units that hold most of a model’s learned features. This page runs a REAL GPT-2 in your browser (verified vs PyTorch) and shows which neurons fire for which tokens: the most-active neurons across your sentence, with the strongest firing marked. Hover a cell for the activation; pick a layer. Different neurons light up for different tokens — the raw material of interpretability.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  controls: (c, page) => {
    c.text('text', { label: 'sentence', value: DEFAULT_TEXT, placeholder: 'type a sentence…', rebuild: false });
    c.button('re-run', () => { if (gpt2) analyze(page, page.state.text); else page.redraw(); });
    c.stepper('layer', { label: 'layer', min: 0, max: CFG.nLayer - 1, value: 5 });
    c.stepper('topn', { label: 'top neurons', min: 12, max: 48, value: 32 });
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    if (!M.mlp) setMLP(M.ids, M.tokens, synthMLP(M.ids.length), M.source, M.status);
    const L = Math.min(CFG.nLayer - 1, st.layer | 0), n = M.n, dFF = M.mlp.dFF, act = M.mlp.layers[L];
    const N = Math.min(st.topn | 0, dFF);
    // top-N neurons by max-over-tokens activation
    const maxByNeuron = new Float32Array(dFF);
    for (let j = 0; j < dFF; j++) { let m = -Infinity; for (let i = 0; i < n; i++) { const v = act[i * dFF + j]; if (v > m) m = v; } maxByNeuron[j] = m; }
    neuronIds = Array.from(maxByNeuron.keys()).sort((a, b) => maxByNeuron[b] - maxByNeuron[a]).slice(0, N);
    // sub-matrix [n × N] + global peak
    const sub = new Float32Array(n * N); let peak = 1e-6, pk = [0, 0];
    for (let i = 0; i < n; i++) for (let c = 0; c < N; c++) { const v = act[i * dFF + neuronIds[c]]; sub[i * N + c] = v; if (v > peak) { peak = v; pk = [i, c]; } }
    page.probe = { source: M.source, layer: L, n, maxAct: peak, topNeuron: neuronIds[0], nShown: N };

    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: T.goldDeep };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100 | 0)}% (~548 MB, one time)`, c: T.goldDeep };
      if (M.status === 'running') return { t: '⟳ running GPT-2…', c: T.goldDeep };
      if (M.source === 'real') return { t: '● real GPT-2 (124M) — MLP activations from the actual model', c: T.okDeep };
      if (M.status === 'offline') return { t: '○ offline — synthetic activation stand-in (click “load real GPT-2”)', c: T.goldDeep };
      return { t: '○ synthetic activation stand-in — click “load real GPT-2” for real neurons', c: T.n11 };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();

    const pad = 14, topY = 52, lblW = 56;
    r.label(`layer ${L} · ${N} most-active neurons (of ${dFF}) × ${n} tokens — diverging: red fires, blue suppresses`, pad + lblW, topY - 8, { color: T.n11, font: '11px ui-monospace, monospace' });
    const gx = pad + lblW, gy = topY + 10;
    const cell = Math.max(7, Math.min(26, Math.min((page.W - gx - pad) / N, (page.H - gy - 50) / Math.max(n, 1))));
    hmRect = { x: gx, y: gy, w: N * cell, h: n * cell };
    r.heatmap(sub, { rows: n, cols: N, rect: hmRect, ramp: ramps.diverging, domain: [-peak, peak] });
    r.grid({ stroke: rgbaToken('n14', 0.10) });
    // token labels (rows)
    ctx.save(); ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = T.n11; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) ctx.fillText(td(M.tokens[i]).slice(0, 8), gx - 4, gy + i * cell + cell / 2);
    // ring the global peak (the strongest-firing token×neuron)
    ctx.strokeStyle = T.n14; ctx.lineWidth = 2; ctx.strokeRect(gx + pk[1] * cell, gy + pk[0] * cell, cell, cell);
    // neuron-id ticks for a few columns
    ctx.fillStyle = T.n9; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.font = '8px ui-monospace, monospace';
    for (let c = 0; c < N; c += Math.ceil(N / 8)) ctx.fillText('#' + neuronIds[c], gx + c * cell + cell / 2, gy - 1);
    ctx.restore();

    if (page.pointer.over && hmRect) { const h = cellAt(hmRect, n, N, page.pointer.x, page.pointer.y); if (h) page.setTip(`neuron #${neuronIds[h.c]} (layer ${L})\ntoken "${(M.tokens[h.r] || '').trim() || M.tokens[h.r]}"\nactivation = ${sub[h.r * N + h.c].toFixed(3)}`); }

    page.setReadout(`${M.source === 'real' ? 'REAL GPT-2 (124M)' : 'synthetic'} MLP activations · layer ${L} · ${dFF} neurons · peak ${peak.toFixed(2)} at neuron #${neuronIds[pk[1]]} on "${(M.tokens[pk[0]] || '').trim() || M.tokens[pk[0]]}"    tier:${r.name}\n` +
      `the ${N} columns are the most-active neurons for this sentence; different neurons fire for different tokens` + (M.source === 'real' ? ' — these are GPT-2’s actual learned features.' : ' (load real GPT-2 for actual neurons).'));
  },
  challenges: [
    { goal: 'Ground it in the REAL model — load GPT-2 so the neurons are the actual MLP units (needs network).',
      hint: 'The banner turns green “● real GPT-2” after the ~548 MB one-time download.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Find a strongly-firing neuron — get the selected layer’s peak activation above 4.0 (deeper layers fire harder).',
      hint: 'Step the layer up; later layers have larger activations. Watch the peak in the readout.',
      check: (api) => ({ solved: api.probe.maxAct >= 4, detail: `peak activation = ${(+api.probe.maxAct).toFixed(2)} (need ≥ 4.0)` }) },
  ],
}).then((page) => {
  window.__realNeuronsPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('text')) page.controls.set('text', q.get('text'));
  if (q.has('ids')) { injected = q.get('ids').split(',').map((x) => parseInt(x, 10)).filter((x) => Number.isFinite(x)); M.ids = injected; M.tokens = injected.map((x) => '·' + x); }
  ['layer', 'topn'].forEach((k) => { if (q.has(k)) page.controls.set(k, +q.get(k)); });
  setMLP(M.ids, M.tokens, synthMLP(M.ids.length), 'synthetic', 'init'); page.redraw();
  if (q.get('real') === '1' || q.get('autoload') === '1') ensureReal(page);   // large download: opt-in only
});
