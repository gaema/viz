// real-kv-cache concept page — Phase 9 (real-model grounding).
//
// The synthetic kv-cache page shows the mechanism; this one shows the REAL cache.
// As GPT-2 generates, each layer caches K and V for every past position
// (GPT2.cache() — verified vs PyTorch past_key_values). The cache grows one
// position per token: the PROMPT fills it in parallel (prefill), then each
// generated token appends one column (decode). The page draws the actual cached
// K values for a chosen layer/head + the real memory (GPT-2's true dims) climbing
// live — which is why long context costs so much memory.
//
// Breadcrumbs to the KV-cache concept. Reuses ../real-attention/gpt2.js
// (cache()).
//
// Offline: a synthetic K cache (seeded) + the real memory math (dims are real),
// so the cost story still teaches. ?real=0 forces it.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { T } from '../framework/theme.js';
import { loadGPT2, GPT2_CONFIG } from '../real-attention/gpt2.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2, DH = CFG.nEmbd / CFG.nHead;

const DTYPES = { 'fp16 (2 B)': 2, 'fp32 (4 B)': 4, 'int8 (1 B)': 1 };
const DEFAULT_PROMPT = 'the cat sat on the mat . the cat ran';
const DEFAULT_IDS = [1169, 3797, 3332, 319, 262, 2603, 764, 262, 3797, 4966];

let M = { status: 'init', progress: 0, source: 'synthetic', ids: DEFAULT_IDS, nPrompt: DEFAULT_IDS.length, cache: null, n: 0, generating: false };
let tokenizer = null, gpt2 = null, loadStarted = false, injected = null;
let hmRect = null;
const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const perPosBytes = (dtB) => 2 * CFG.nLayer * CFG.nEmbd * dtB;   // K+V, all layers/heads
const fmtBytes = (b) => b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : (b / 1e3).toFixed(1) + ' KB';

// synthetic K cache (seeded) for the offline stand-in: per-layer [n×D]
function synthCache(ids) {
  const n = ids.length, D = CFG.nEmbd, rng = mulberry32(0x5eed ^ n), layers = [];
  for (let l = 0; l < CFG.nLayer; l++) { const K = new Float32Array(n * D), V = new Float32Array(n * D); for (let i = 0; i < n * D; i++) { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); K[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); V[i] = rng() * 2 - 1; } layers.push({ K, V }); }
  return { layers, n };
}

function setCache(ids, nPrompt, cache, source, status) { M = { ...M, ids, nPrompt, cache, n: ids.length, source, status }; }

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function ensureReal(page) {
  if (gpt2 || loadStarted) return; loadStarted = true;
  try {
    if (!injected) { M.status = 'loading-tok'; page.redraw(); const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js'); env.allowLocalModels = false; tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL); }
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(page.state.weights || WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    M.status = 'ready'; M.source = 'real';
    await prefill(page, page.state.prompt || DEFAULT_PROMPT);
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

async function prefill(page, prompt) {     // the prompt fills the cache in parallel
  let ids;
  if (injected) ids = injected;
  else if (tokenizer) { const enc = await tokenizer(prompt || DEFAULT_PROMPT, { add_special_tokens: false }); ids = Array.from(enc.input_ids.data, Number).slice(0, 40); }
  else ids = DEFAULT_IDS;
  const c = (M.source === 'real' && gpt2) ? gpt2.cache(ids) : synthCache(ids);
  setCache(ids, ids.length, c, M.source, 'ready'); page.redraw();
}

// decode: each step samples the next (greedy) token and appends ONE position
async function generate(page) {
  if (M.generating) { M.generating = false; return; }
  M.generating = true; page.redraw();
  const N = page.state.ntok | 0, fast = !!M._fast;
  let ids = M.ids.slice();
  for (let k = 0; k < N; k++) {
    if (!M.generating) break;
    let next;
    if (M.source === 'real' && gpt2) { const c = gpt2.cache(ids); let a = 0; for (let v = 1; v < c.V; v++) if (c.logits[v] > c.logits[a]) a = v; next = a; }
    else next = DEFAULT_IDS[k % DEFAULT_IDS.length];
    ids = ids.concat(next);
    const c = (M.source === 'real' && gpt2) ? gpt2.cache(ids) : synthCache(ids);
    setCache(ids, M.nPrompt, c, M.source, 'ready'); page.redraw();
    if (!fast) await sleep(70);
  }
  M.generating = false; page.redraw();
}

mount({
  mount: 'body',
  slug: 'real-kv-cache',
  title: 'real KV cache — what fills up as GPT-2 generates',
  blurb: 'Phase 9 (real-model grounding). As GPT-2 generates, every layer caches the Keys and Values for all past positions so each new token is cheap — the KV cache. This page shows the REAL cache (verified vs PyTorch past_key_values): the actual cached K values for a layer/head, and the real memory (GPT-2’s true 12×12×64 dims) growing one position per token. The prompt fills it in parallel (prefill); each generated token appends one column (decode). Watch the cache — and the megabytes — climb.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  controls: (c, page) => {
    c.text('prompt', { label: 'prompt', value: DEFAULT_PROMPT, placeholder: 'type a prompt…', rebuild: false });
    c.button('prefill', () => { if (gpt2 || M.source !== 'real') prefill(page, page.state.prompt); });
    c.stepper('layer', { label: 'layer', min: 0, max: CFG.nLayer - 1, value: 5 });
    c.stepper('head', { label: 'head', min: 0, max: CFG.nHead - 1, value: 5 });
    c.select('dtype', { label: 'KV dtype', options: Object.keys(DTYPES), value: 'fp16 (2 B)' });
    c.stepper('ntok', { label: 'gen tokens', min: 1, max: 32, value: 12 });
    c.button('▶ generate / ■ stop', () => generate(page));
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    if (!M.cache) setCache(M.ids, M.nPrompt, synthCache(M.ids), M.source, M.status);
    const sl = Math.min(CFG.nLayer - 1, st.layer | 0), sh = Math.min(CFG.nHead - 1, st.head | 0);
    const n = M.cache.n, dtB = DTYPES[st.dtype] || 2, bytes = perPosBytes(dtB) * n;
    page.probe = { source: M.source, seq: n, nPrompt: M.nPrompt, dtype: st.dtype, bytesMB: bytes / 1e6, layer: sl, head: sh };

    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: T.goldDeep };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100 | 0)}% (~548 MB, one time)`, c: T.goldDeep };
      if (M.generating) return { t: '✍ decoding — appending one cache column per token…', c: T.goldDeep };
      if (M.source === 'real') return { t: '● real GPT-2 (124M) — KV cache from the actual model', c: T.okDeep };
      if (M.status === 'offline') return { t: '○ offline — synthetic K cache (real dims/memory; click “load real GPT-2”)', c: T.goldDeep };
      return { t: '○ synthetic K cache (real dims/memory) — click “load real GPT-2” for the real cache', c: T.n11 };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();

    const pad = 16, topY = 40;
    // ---- left: K-cache heatmap for the selected layer/head, [n positions × dh] ----
    const D = CFG.nEmbd, K = M.cache.layers[sl].K, kh = new Float32Array(n * DH);
    let mx = 1e-6; for (let i = 0; i < n; i++) for (let cc = 0; cc < DH; cc++) { const v = K[i * D + sh * DH + cc]; kh[i * DH + cc] = v; if (Math.abs(v) > mx) mx = Math.abs(v); }
    const lblW = 30, gx = pad + lblW, gy = topY + 16;
    const cell = Math.max(4, Math.min(16, Math.min((page.W * 0.5 - gx) / DH, (page.H - gy - 40) / Math.max(n, 1))));
    hmRect = { x: gx, y: gy, w: DH * cell, h: n * cell };
    r.label(`cached K — layer ${sl}, head ${sh}  (${n} positions × ${DH} dims)`, gx, topY, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.heatmap(kh, { rows: n, cols: DH, rect: hmRect, ramp: ramps.diverging, domain: [-mx, mx] });
    // mark the prefill/decode boundary
    if (M.nPrompt < n) { const yb = gy + M.nPrompt * cell; ctx.save(); ctx.strokeStyle = T.okDeep; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(gx - 4, yb); ctx.lineTo(gx + hmRect.w, yb); ctx.stroke(); ctx.restore(); }
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = T.n9; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('pos 0', gx - 4, gy + cell / 2); ctx.fillText('pos ' + (n - 1), gx - 4, gy + (n - 0.5) * cell);
    if (M.nPrompt < n) { ctx.fillStyle = T.okDeep; ctx.textAlign = 'left'; ctx.fillText('↑prefill ' + M.nPrompt + ' · decode↓', gx + hmRect.w + 6, gy + M.nPrompt * cell); }
    ctx.restore();

    // ---- right: memory ----
    const rx = pad + page.W * 0.52, rw = page.W - rx - pad;
    ctx.save(); ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = T.n13; ctx.font = '13px ui-monospace, monospace';
    ctx.fillText(`KV cache now: ${fmtBytes(bytes)}`, rx, topY + 14);
    ctx.fillStyle = T.n11; ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`${n} positions × 2(K+V) × ${CFG.nLayer} layers × ${CFG.nEmbd} dim × ${dtB} B = ${fmtBytes(bytes)}`, rx, topY + 32);
    ctx.fillText(`= ${fmtBytes(perPosBytes(dtB))} per token (${st.dtype})`, rx, topY + 48);
    // memory-vs-context curve (why long context is expensive)
    const ctxs = [128, 512, 2048, 8192, 32768], by = ctxs.map((s) => perPosBytes(dtB) * s);
    const cy = topY + 80, ch = page.H - cy - 40, cw = rw, bmax = by[by.length - 1];
    ctx.fillStyle = T.n11; ctx.fillText('KV cache vs context length:', rx, cy - 6);
    const bw = cw / ctxs.length;
    for (let i = 0; i < ctxs.length; i++) { const h = (by[i] / bmax) * (ch - 18), x = rx + i * bw, y = cy + ch - h; ctx.fillStyle = T.accent; ctx.fillRect(x + 5, y, bw - 12, h); ctx.fillStyle = T.n9; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(ctxs[i] >= 1024 ? (ctxs[i] / 1024) + 'K' : ctxs[i], x + bw / 2, cy + ch + 11); ctx.fillText(fmtBytes(by[i]), x + bw / 2, y - 3); ctx.textAlign = 'left'; }
    ctx.restore();

    // hover on the K heatmap
    if (page.pointer.over && hmRect) { const h = cellAt(hmRect, n, DH, page.pointer.x, page.pointer.y); if (h) page.setTip(`K[layer ${sl}, head ${sh}]\npos ${h.r}, dim ${h.c} = ${kh[h.r * DH + h.c].toFixed(3)}${h.r < M.nPrompt ? '\n(prefill)' : '\n(decoded)'}`); }

    page.setReadout(`${M.source === 'real' ? 'REAL GPT-2 (124M)' : 'synthetic'} KV cache · ${n} positions (${M.nPrompt} prompt + ${n - M.nPrompt} generated) · ${st.dtype} → ${fmtBytes(bytes)}    tier:${r.name}\n` +
      `each token caches 2·${CFG.nLayer}·${CFG.nEmbd} = ${(2 * CFG.nLayer * CFG.nEmbd).toLocaleString()} values (K+V across all layers); that’s why a long context costs memory — and why the cache makes each new token cheap.`);
  },
  challenges: [
    { goal: 'Ground it in the REAL model — load GPT-2 so the cache is the actual K/V (needs network; “load real GPT-2”).',
      hint: 'The banner turns green “● real GPT-2” after the ~548 MB one-time download.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Watch the cache grow — generate so the sequence reaches at least the prompt length + 8 (the decode phase appends one column per token).',
      hint: 'Set gen tokens ≥ 8 and press generate; each token adds one position to the cache.',
      check: (api) => ({ solved: api.probe.seq >= api.probe.nPrompt + 8, detail: `${api.probe.seq} positions (prompt ${api.probe.nPrompt}, need +8)` }) },
  ],
}).then((page) => {
  window.__realKVPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('prompt')) page.controls.set('prompt', q.get('prompt'));
  if (q.has('ids')) { injected = q.get('ids').split(',').map((x) => parseInt(x, 10)).filter((x) => Number.isFinite(x)); M.ids = injected; M.nPrompt = injected.length; }
  ['layer', 'head', 'ntok'].forEach((k) => { if (q.has(k)) page.controls.set(k, +q.get(k)); });
  if (q.has('dtype') && DTYPES[q.get('dtype')]) page.controls.set('dtype', q.get('dtype'));
  setCache(M.ids, M.nPrompt, synthCache(M.ids), 'synthetic', 'init'); page.redraw();
  if (q.has('gen')) { M._fast = true; page.controls.set('ntok', Math.max(1, +q.get('gen'))); }
  if (q.get('real') === '1' || q.get('autoload') === '1') ensureReal(page); else if (q.has('gen')) generate(page);   // large download: opt-in only
});
