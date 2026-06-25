// real-attention concept page — Phase 9 (real-model grounding).
//
// REAL GPT-2 attention. transformers.js can't emit attentions (the ONNX export
// drops them — Optimum #325), so gpt2.js fetches the raw safetensors and runs a
// verified vanilla-JS GPT-2 forward, capturing A[layer][head] = softmax(QKᵀ/√d +
// causal mask) at every layer. (gpt2.test.mjs checks the forward reproduces
// PyTorch within ~5e-5.) Type any sentence; the 12×12 head-map colours each head
// by what it does, so you can find the famous heads:
//   - previous-token head (sub-diagonal): each token attends to the one before.
//   - induction head: on repeated text, attends to what followed the token last
//     time (the in-context-learning mechanism).
//   - attention-sink head: dumps attention on token 0 as a no-op.
//
// The synthetic `attention-patterns` page shows IDEALIZED versions of these
// shapes; this page shows the real model's. Breadcrumbs to ../design/
// architectures.md A2 (attention). Plan: ../plan/phase9.md.
//
// Offline / no network: renders a clearly-labelled IDEALIZED synthetic stand-in
// (the same three head shapes, hand-built) on the default sentence, and swaps in
// the real model once the ~548 MB weights download. ?real=0 forces synthetic.
import { mount } from '../framework/layout.js';
import { loadGPT2, GPT2_CONFIG } from './gpt2.js';
import { getWebGPU, GPT2WebGPU } from './gpt2-webgpu.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2;

// default sentence + its real GPT-2 token ids (so the offline stand-in works with
// no tokenizer download). "the cat" repeats → an induction head fires.
const DEFAULT_TEXT = 'the cat sat on the mat . the cat ran';
const DEFAULT_IDS = [1169, 3797, 3332, 319, 262, 2603, 764, 262, 3797, 4966];
const DEFAULT_TOKENS = ['the', ' cat', ' sat', ' on', ' the', ' mat', ' .', ' the', ' cat', ' ran'];
// literature head locations (used to seed the synthetic idealized stand-in).
const PREV = [4, 11], SINK = [7, 2], IND = [5, 5];
const CAT = { prev: { c: '#1f6feb', label: 'previous-token' }, sink: { c: '#9a6700', label: 'attention-sink' }, induction: { c: '#0a7227', label: 'induction' }, none: { c: '#c8ccd1', label: '—' } };
const GREEN = '#0a7227', AMBER = '#9a6700';

let M = { status: 'init', progress: 0, source: 'synthetic', ids: DEFAULT_IDS, tokens: DEFAULT_TOKENS, att: null, scores: null, sel: [0, 0], model: 'gpt2', err: '', backend: 'cpu', gpuAvail: false, gpuCheck: null, times: {} };
let tokenizer = null, gpt2 = null, gpu = null, webgpu = null, loadStarted = false, injected = null;
let hmRect = null, heatRect = null;   // head-map + heatmap rects (hit-testing)

// ---- detectors (must match gpt2.test.mjs / the PyTorch ground truth) --------
const prevScore = (A, n) => { let s = 0; for (let i = 1; i < n; i++) s += A[i * n + i - 1]; return n > 1 ? s / (n - 1) : 0; };
const sinkScore = (A, n) => { let s = 0; for (let i = 0; i < n; i++) s += A[i * n]; return s / n; };
const inductionScore = (A, n, ids) => { let s = 0, c = 0; for (let i = 0; i < n; i++) { let j = -1; for (let k = i - 1; k >= 0; k--) if (ids[k] === ids[i]) { j = k; break; } if (j >= 0 && j + 1 < i) { s += A[i * n + j + 1]; c++; } } return c ? s / c : 0; };

function computeScores(att, n, ids) {
  const out = [];
  for (let l = 0; l < att.length; l++) {
    out.push([]);
    for (let h = 0; h < att[l].length; h++) {
      const A = att[l][h], p = prevScore(A, n), s = sinkScore(A, n), ind = inductionScore(A, n, ids);
      // dominant category, only if it clears a floor (most heads are "none")
      let cat = 'none', best = 0.45;
      if (p > best) { best = p; cat = 'prev'; }
      if (s > 0.6 && s > best) { best = s; cat = 'sink'; }
      if (ind > 0.30 && ind >= best) { cat = 'induction'; }
      out[l].push({ prev: p, sink: s, induction: ind, cat });
    }
  }
  return out;
}

// ---- synthetic idealized stand-in: build A[l][h] for all heads --------------
function synthAtt(ids) {
  const n = ids.length, L = CFG.nLayer, H = CFG.nHead, att = [];
  const norm = (A) => { for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j <= i; j++) s += A[i * n + j]; if (s) for (let j = 0; j <= i; j++) A[i * n + j] /= s; } return A; };
  for (let l = 0; l < L; l++) { att.push([]); for (let h = 0; h < H; h++) {
    const A = new Float32Array(n * n);
    if (l === PREV[0] && h === PREV[1]) { for (let i = 0; i < n; i++) A[i * n + Math.max(0, i - 1)] = 1; }
    else if (l === SINK[0] && h === SINK[1]) { for (let i = 0; i < n; i++) { A[i * n] += 0.9; A[i * n + i] += 0.1; } }
    else if (l === IND[0] && h === IND[1]) { for (let i = 0; i < n; i++) { let j = -1; for (let k = i - 1; k >= 0; k--) if (ids[k] === ids[i]) { j = k; break; } if (j >= 0 && j + 1 < i) A[i * n + j + 1] += 0.85, A[i * n + i] += 0.15; else A[i * n + i] += 1; } }
    else { for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) A[i * n + j] = Math.exp(-(i - j) * 0.6); }   // generic decaying-causal
    att[l].push(norm(A));
  } }
  return att;
}

function setAnalysis(ids, tokens, att, source, status) {
  M = { ...M, ids, tokens, att, source, status, scores: computeScores(att, ids.length, ids) };
}
function runSynthetic(ids, tokens, status) { setAnalysis(ids, tokens, synthAtt(ids), 'synthetic', status || M.status); }

// Run the real forward on the selected backend. On the FIRST GPU run, also run
// the CPU forward and self-verify (max|Δ| over all attentions); if they disagree
// (a WGSL bug, a flaky adapter) fall back to the verified CPU path. Records both
// timings so the speedup is visible.
async function analyze(page, ids, tokens) {
  if (!gpt2) return;
  M.status = 'running'; page.redraw();
  try {
    const useGpu = !!(gpu && page.state.gpu);
    let att, backend = 'cpu';
    if (useGpu) {
      try { const t0 = performance.now(); const o = await gpu.forward(ids); M.times.gpu = performance.now() - t0; att = o.attentions; backend = 'gpu'; }
      catch (ge) { gpu = null; M.gpuErr = String(ge && ge.message || ge); }   // adapter died → CPU
    }
    if (backend !== 'gpu') { const c0 = performance.now(); const o = gpt2.forward(ids); M.times.cpu = performance.now() - c0; att = o.attentions; }
    else if (!M.gpuCheck) {                                                   // self-verify GPU vs CPU once
      const c0 = performance.now(); const co = gpt2.forward(ids); M.times.cpu = performance.now() - c0;
      let mx = 0; for (let l = 0; l < att.length; l++) for (let h = 0; h < att[l].length; h++) { const a = att[l][h], b = co.attentions[l][h]; for (let k = 0; k < a.length; k++) { const dd = Math.abs(a[k] - b[k]); if (dd > mx) mx = dd; } }
      M.gpuCheck = { maxDiff: mx, ok: mx < 1e-2 };
      if (!M.gpuCheck.ok) { att = co.attentions; backend = 'cpu'; page.controls.set('gpu', false); }   // mismatch → trust CPU
    }
    setAnalysis(ids, tokens, att, 'real', 'ready'); M.backend = backend;
    if (M.sel[1] >= CFG.nHead) M.sel = [0, 0];
    page.redraw();
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function tokenize(text) {
  const enc = await tokenizer(text, { add_special_tokens: false });
  const ids = Array.from(enc.input_ids.data, Number).slice(0, 48);     // cap for a legible heatmap
  const tokens = ids.map((id) => tokenizer.decode([id]));
  return { ids, tokens };
}

// load tokenizer (small) + GPT-2 weights (~548 MB). Synthetic stays until ready.
async function ensureReal(page) {
  if (gpt2 || loadStarted) return;
  loadStarted = true;
  try {
    // ?ids injects exact token ids (verification path) → skip the tokenizer (and
    // its CDN download) entirely, so the real forward can be checked fully offline
    // against local file:// weights.
    if (!injected) {
      M.status = 'loading-tok'; page.redraw();
      const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js');
      env.allowLocalModels = false;
      tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL);
    }
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(page.state.weights || WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    // bring up the WebGPU backend if available (shares the parsed weights Map)
    try { const wg = await getWebGPU(); if (wg) { const g = new GPT2WebGPU(gpt2.w, CFG, wg.device); await g.init(); gpu = g; webgpu = wg; M.gpuAvail = true; } } catch (ge) { gpu = null; M.gpuErr = String(ge && ge.message || ge); }
    M.status = 'running'; page.redraw();
    const { ids, tokens } = injected || await tokenize(page.state.text || DEFAULT_TEXT);
    await analyze(page, ids, tokens);
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

// re-tokenize + re-analyze typed text (real if the model is ready, else nothing —
// the idealized stand-in only knows the default sentence's ids offline).
async function reanalyze(page, text) {
  if (!tokenizer || !gpt2) return;
  M.status = 'running'; page.redraw();
  try { const { ids, tokens } = await tokenize(text || DEFAULT_TEXT); if (ids.length) await analyze(page, ids, tokens); page.redraw(); }
  catch (e) { M.status = 'offline'; M.err = String(e); page.redraw(); }
}

const td = (s) => (s || '').replace(/^ /, '·').replace(/\n/g, '⏎').slice(0, 7) || '∅';
const lerpCol = (a, t) => { const r = Math.round(255 + (31 - 255) * t), g = Math.round(255 + (111 - 255) * t), b = Math.round(255 + (235 - 255) * t); return `rgb(${r},${g},${b})`; };  // white→blue

mount({
  mount: 'body',
  slug: 'real-attention',
  title: 'real attention — finding GPT-2’s heads',
  blurb: 'Phase 9 (real-model grounding). The synthetic attention pages show idealized head shapes on seeded numbers; this page runs a REAL GPT-2 in your browser — we fetch the raw weights and run a verified forward in vanilla JS (transformers.js can’t emit attentions), capturing softmax(QKᵀ/√d) at every layer/head. Type a sentence; the 12×12 head-map colours each head by what it does. Hunt for the previous-token head (sub-diagonal), the induction head (attends to what followed a repeated token last time — the in-context-learning trick), and the attention-sink head (dumps on token 0). Offline it shows a labelled idealized stand-in; the real model needs a ~548 MB one-time download.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  controls: (c, page) => {
    c.text('text', { label: 'sentence', value: DEFAULT_TEXT, placeholder: 'type any sentence…', rebuild: false });
    c.button('re-run', () => reanalyze(page, page.state.text));
    c.stepper('layer', { label: 'layer', min: 0, max: CFG.nLayer - 1, value: 0 });
    c.stepper('head', { label: 'head', min: 0, max: CFG.nHead - 1, value: 0 });
    c.button('jump to induction head', () => { page.controls.set('layer', IND[0]); page.controls.set('head', IND[1]); M.sel = [IND[0], IND[1]]; page.redraw(); });
    c.toggle('gpu', { label: 'WebGPU compute', value: true });
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: (page, ev) => {
    if (ev.type !== 'down' || !hmRect || !M.scores) return;
    const { x, y } = ev, L = CFG.nLayer, H = CFG.nHead;
    if (x >= hmRect.x && x < hmRect.x + hmRect.w && y >= hmRect.y && y < hmRect.y + hmRect.h) {
      const h = Math.min(H - 1, Math.floor((x - hmRect.x) / (hmRect.w / H)));
      const l = Math.min(L - 1, Math.floor((y - hmRect.y) / (hmRect.h / L)));
      M.sel = [l, h]; page.controls.set('layer', l); page.controls.set('head', h); page.redraw();
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear('#ffffff');
    // sync selection from steppers (so keyboard/url changes move it too)
    if (st.layer != null && st.head != null && (st.layer !== M.sel[0] || st.head !== M.sel[1])) M.sel = [st.layer | 0, Math.min(CFG.nHead - 1, st.head | 0)];
    const [sl, sh] = M.sel, n = M.ids.length;
    const sc = M.scores && M.scores[sl] && M.scores[sl][sh];
    page.probe = { source: M.source, selLayer: sl, selHead: sh, n, selPrev: sc ? sc.prev : 0, selSink: sc ? sc.sink : 0, selInduction: sc ? sc.induction : 0 };

    // banner
    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: AMBER };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100 | 0)}% (~548 MB, one time)`, c: AMBER };
      if (M.status === 'running') return { t: '⟳ running GPT-2…', c: AMBER };
      if (M.source === 'real') {
        const be = M.backend === 'gpu' ? 'WebGPU compute' : 'CPU compute';
        const mism = M.gpuCheck && !M.gpuCheck.ok;
        const chk = M.gpuCheck ? `  ${mism ? '⚠ GPU≠CPU → using CPU' : '✓ GPU=CPU'} (max|Δ|=${M.gpuCheck.maxDiff.toExponential(1)})` : (M.gpuAvail ? '' : '  (WebGPU n/a → CPU)');
        const tm = (M.times.gpu && M.times.cpu) ? `  ·  GPU ${M.times.gpu | 0}ms vs CPU ${M.times.cpu | 0}ms` : '';
        return { t: `● real GPT-2 (124M) — ${be}${chk}${tm}`, c: mism ? AMBER : GREEN };
      }
      if (M.status === 'offline') return { t: '○ offline — idealized synthetic stand-in (click “load real GPT-2”)', c: AMBER };
      return { t: '○ idealized synthetic stand-in — click “load real GPT-2” for true attention', c: '#586069' };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();
    if (!M.att) { page.setReadout('loading…'); return; }

    const pad = 14, topY = 34;
    // ---- left: attention heatmap for the selected head ----
    const A = M.att[sl][sh];
    const lblW = 52, gridTop = topY + 22;
    const avail = Math.min(page.W * 0.52 - pad - lblW, page.H - gridTop - 40);
    const cell = Math.max(8, avail / n);
    heatRect = { x: pad + lblW, y: gridTop, w: cell * n, h: cell * n };
    const catSel = sc ? sc.cat : 'none';
    r.label(`layer ${sl} · head ${sh}`, heatRect.x, topY + 4, { color: '#24292e', font: '12px ui-monospace, monospace' });
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = CAT[catSel].c; ctx.fillText(catSel !== 'none' ? `  ← ${CAT[catSel].label} head` : '', heatRect.x + 96, topY + 4); ctx.restore();
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = heatRect.x + j * cell, y = heatRect.y + i * cell;
      if (j > i) { ctx.fillStyle = '#f3f4f6'; ctx.fillRect(x, y, cell - 1, cell - 1); continue; }   // causal mask
      ctx.fillStyle = lerpCol(0, Math.min(1, A[i * n + j])); ctx.fillRect(x, y, cell - 1, cell - 1);
    }
    // token labels (query rows on the left, key cols on top)
    ctx.save(); ctx.font = `${Math.min(11, cell - 1)}px ui-monospace, monospace`; ctx.fillStyle = '#586069';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) ctx.fillText(td(M.tokens[i]), heatRect.x - 4, heatRect.y + i * cell + cell / 2);
    ctx.textAlign = 'left'; ctx.translate(0, 0);
    for (let j = 0; j < n; j++) { ctx.save(); ctx.translate(heatRect.x + j * cell + cell / 2, heatRect.y - 4); ctx.rotate(-Math.PI / 2); ctx.fillText(td(M.tokens[j]), 0, 0); ctx.restore(); }
    ctx.restore();

    // ---- right: 12×12 head-map (rows=layer, cols=head), colored by category ----
    const L = CFG.nLayer, H = CFG.nHead;
    const hx = heatRect.x + heatRect.w + 70;
    const hmCell = Math.max(10, Math.min(22, Math.min((page.W - pad - hx) / H, (page.H - gridTop - 70) / L)));
    hmRect = { x: hx, y: gridTop, w: hmCell * H, h: hmCell * L };
    r.label('head-map — every head, colored by role (click to select)', hx, topY + 4, { color: '#586069', font: '11px ui-monospace, monospace' });
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
      const s = M.scores[l][h], x = hx + h * hmCell, y = gridTop + l * hmCell;
      const inten = s.cat === 'none' ? 0.12 : Math.min(1, s.cat === 'prev' ? s.prev : s.cat === 'sink' ? s.sink : s.induction);
      ctx.fillStyle = s.cat === 'none' ? '#eef0f2' : CAT[s.cat].c; ctx.globalAlpha = s.cat === 'none' ? 1 : 0.25 + 0.75 * inten;
      ctx.fillRect(x, y, hmCell - 1.5, hmCell - 1.5); ctx.globalAlpha = 1;
    }
    // selected ring
    ctx.save(); ctx.strokeStyle = '#111'; ctx.lineWidth = 2; ctx.strokeRect(hx + sh * hmCell - 1, gridTop + sl * hmCell - 1, hmCell, hmCell); ctx.restore();
    // axes
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = '#9aa4ad'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let l = 0; l < L; l++) ctx.fillText('L' + l, hx - 3, gridTop + l * hmCell + hmCell / 2);
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText('heads 0→11 →', hx + hmRect.w / 2, gridTop - 16); ctx.restore();
    // legend
    let lx = hx, ly = gridTop + hmRect.h + 12; ctx.save(); ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
    for (const k of ['prev', 'induction', 'sink']) { ctx.fillStyle = CAT[k].c; ctx.fillRect(lx, ly - 5, 10, 10); ctx.fillStyle = '#586069'; ctx.textAlign = 'left'; ctx.fillText(' ' + CAT[k].label, lx + 10, ly); lx += 11 + CAT[k].label.length * 6.2 + 14; }
    ctx.restore();

    // hover on the attention heatmap
    if (page.pointer.over && heatRect) {
      const p = page.pointer, j = Math.floor((p.x - heatRect.x) / cell), i = Math.floor((p.y - heatRect.y) / cell);
      if (i >= 0 && i < n && j >= 0 && j <= i) page.setTip(`query "${(M.tokens[i] || '').trim() || M.tokens[i]}" (pos ${i})\n→ key "${(M.tokens[j] || '').trim() || M.tokens[j]}" (pos ${j})\nattention = ${A[i * n + j].toFixed(3)}`);
    }

    const o = `${M.source === 'real' ? `REAL GPT-2 (124M, ${M.backend === 'gpu' ? 'WebGPU' : 'CPU'})` : 'synthetic (idealized)'} · ${n} tokens · selected head L${sl}·H${sh}: ` +
      `prev=${(sc ? sc.prev : 0).toFixed(2)} induction=${(sc ? sc.induction : 0).toFixed(2)} sink=${(sc ? sc.sink : 0).toFixed(2)}    tier:${r.name}\n` +
      (M.source === 'real' ? 'These are the real model’s attention weights. Try the head-map: blue=prev-token, green=induction, amber=sink.' :
        'Idealized stand-in (load real GPT-2 for true weights). In real gpt2-small: prev-token=L4·H11, induction=L5·H5, sink=L7·H2.');
    page.setReadout(o);
  },
  challenges: [
    { goal: 'Ground it in the REAL model — download GPT-2 and compute attention in-browser (needs network; “load real GPT-2”).',
      hint: 'The banner turns green “● real GPT-2” after the ~548 MB one-time download. Works on the mesh / online.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Find an INDUCTION head — select a head whose induction score ≥ 0.30 (on the repeated text it attends to what followed the token last time).',
      hint: 'Use the head-map (green cells) or the “jump to induction head” button. In real gpt2-small it’s L5·H5.',
      check: (api) => ({ solved: api.probe.selInduction >= 0.30, detail: `selected head induction = ${(+api.probe.selInduction).toFixed(2)} (need ≥ 0.30)` }) },
  ],
}).then((page) => {
  window.__realAttnPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('text')) page.controls.set('text', q.get('text'));
  // ?ids=1169,3797,... injects exact token ids (headless determinism; skips the
  // tokenizer). ?layer / ?head select a head. ?weights= overrides the URL.
  let ids = DEFAULT_IDS, tokens = DEFAULT_TOKENS;
  if (q.has('ids')) { ids = q.get('ids').split(',').map((x) => parseInt(x, 10)).filter((x) => Number.isFinite(x)); tokens = ids.map((x) => '·' + x); injected = { ids, tokens }; }
  if (q.has('layer')) page.controls.set('layer', Math.max(0, Math.min(CFG.nLayer - 1, +q.get('layer'))));
  if (q.has('head')) page.controls.set('head', Math.max(0, Math.min(CFG.nHead - 1, +q.get('head'))));
  M.sel = [page.state.layer | 0, page.state.head | 0];
  runSynthetic(ids, tokens, 'init');
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
  if (q.get('real') !== '0') ensureReal(page);
});
