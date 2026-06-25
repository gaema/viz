// real-generate concept page — Phase 9 (real-model grounding). The capstone.
//
// Watch a REAL GPT-2 write text. Autoregressive generation is just the verified
// next-token step (gpt2.js logits()) in a loop: logits → sample a token → append
// it → repeat. Greedy generation (argmax) is deterministic and is verified
// token-for-token against HF model.generate() in gpt2.test.mjs. Temperature /
// top-k / top-p (with a seed) turn the same machinery into creative sampling.
//
// "the cat sat on the mat . the cat ran" → greedy → " up and down the hall and
// down the hall …" (GPT-2's famous greedy loop).
//
// Breadcrumbs to ../design/architectures.md (A5 lm_head) + the sampling +
// prefill-vs-decode concepts. Plan: ../plan/phase9.md. Reuses gpt2.js.
//
// Offline: streams a clearly-labelled synthetic continuation so the animation
// still teaches; the real model writes real text once the weights download.
import { mount } from '../framework/layout.js';
import { loadGPT2, GPT2_CONFIG } from '../real-attention/gpt2.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2;
const GREEN = '#0a7227', AMBER = '#9a6700', BLUE = '#1f6feb', INK = '#24292e';

const DEFAULT_PROMPT = 'the cat sat on the mat . the cat ran';
const SYN_CONT = [' quickly', ' across', ' the', ' room', ' and', ' then', ' it', ' stopped', ' to', ' rest', ' for', ' a', ' while', ' .', ' the', ' dog', ' watched', ' it', ' go', ' by'];

let M = { status: 'init', progress: 0, source: 'synthetic', prompt: DEFAULT_PROMPT, promptIds: [], gen: [], generating: false, lastDist: null, n: 0 };
let tokenizer = null, gpt2 = null, loadStarted = false, injected = null;

const td = (s) => (s == null ? '' : String(s)).replace(/\n/g, '⏎');
const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// sample the next token id from logits: greedy (argmax) or temperature + top-k +
// top-p over the top-200 candidates (the tail is negligible for sampling).
function sampleNext(logits, V, opts, rng) {
  if (opts.greedy || opts.temp <= 0.05) { let a = 0; for (let v = 1; v < V; v++) if (logits[v] > logits[a]) a = v; return a; }
  const t = opts.temp; let mx = -Infinity; for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
  const N = 200, top = [];
  for (let v = 0; v < V; v++) { const l = logits[v];
    if (top.length < N) { top.push({ v, l }); if (top.length === N) top.sort((a, b) => b.l - a.l); }
    else if (l > top[N - 1].l) { top[N - 1] = { v, l }; let i = N - 1; while (i > 0 && top[i].l > top[i - 1].l) { const s = top[i]; top[i] = top[i - 1]; top[i - 1] = s; i--; } }
  }
  top.sort((a, b) => b.l - a.l);
  let Z = 0; for (const e of top) { e.p = Math.exp((e.l - mx) / t); Z += e.p; } for (const e of top) e.p /= Z;
  let keep = opts.topk > 0 ? top.slice(0, opts.topk) : top;
  if (opts.topp < 1) { let cum = 0, cut = keep.length; for (let i = 0; i < keep.length; i++) { cum += keep[i].p; if (cum >= opts.topp) { cut = i + 1; break; } } keep = keep.slice(0, cut); }
  let s = 0; for (const e of keep) s += e.p; let r = rng() * s, acc = 0;
  for (const e of keep) { acc += e.p; if (r <= acc) return e.v; }
  return keep[keep.length - 1].v;
}

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function ensureReal(page) {
  if (gpt2 || loadStarted) return;
  loadStarted = true;
  try {
    if (!injected) { M.status = 'loading-tok'; page.redraw(); const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js'); env.allowLocalModels = false; tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL); }
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(page.state.weights || WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    M.status = 'ready'; M.source = 'real';
    await setPrompt(page, page.state.prompt || DEFAULT_PROMPT);
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

async function setPrompt(page, prompt) {
  M.prompt = prompt; M.gen = [];
  if (injected) M.promptIds = injected;
  else if (tokenizer) { const enc = await tokenizer(prompt || DEFAULT_PROMPT, { add_special_tokens: false }); M.promptIds = Array.from(enc.input_ids.data, Number).slice(0, 48); }
  M.n = M.promptIds.length; page.redraw();
}

// the generation loop: sample → append → repeat, streaming each token
async function generate(page) {
  if (M.generating) { M.generating = false; return; }   // toggle = stop
  M.generating = true; page.redraw();
  const st = page.state, opts = { temp: +st.temp, topk: st.topk | 0, topp: +st.topp, greedy: !!st.greedy };
  const rng = mulberry32(((st.seed | 0) >>> 0) || 12345);
  const decode = injected ? ((id) => '·' + id) : ((id) => tokenizer ? tokenizer.decode([id]) : '·' + id);
  const real = M.source === 'real' && gpt2;
  let seq = M.promptIds.slice(); M.gen = [];
  const N = st.ntok | 0, fast = !!M._fast;
  for (let k = 0; k < N; k++) {
    if (!M.generating) break;
    let id;
    if (real) { const { logits, V } = gpt2.logits(seq); id = sampleNext(logits, V, opts, rng); }
    else { id = -1 - (k % SYN_CONT.length); }            // synthetic: canned continuation
    const tok = real ? decode(id) : SYN_CONT[k % SYN_CONT.length];
    seq.push(real ? id : 0); M.gen.push({ id, tok });
    page.redraw();
    if (!fast) await sleep(55);
  }
  M.generating = false; page.redraw();
}

mount({
  mount: 'body',
  slug: 'real-generate',
  title: 'real generate — watch GPT-2 write',
  blurb: 'Phase 9 (real-model grounding). The capstone: a REAL GPT-2 writing text in your browser. Generation is just the verified next-token step in a loop — logits → sample → append → repeat. Greedy (argmax) is deterministic and verified token-for-token against the reference; temperature / top-k / top-p (with a seed) make it creative. Type a prompt, hit generate, and watch it write one token at a time. Offline it streams a labelled synthetic continuation.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  controls: (c, page) => {
    c.text('prompt', { label: 'prompt', value: DEFAULT_PROMPT, placeholder: 'type a prompt…', rebuild: false });
    c.stepper('ntok', { label: 'tokens', min: 4, max: 48, value: 16 });
    c.toggle('greedy', { label: 'greedy (deterministic)', value: true });
    c.slider('temp', { label: 'temperature', min: 0.1, max: 1.5, step: 0.05, value: 0.8 });
    c.stepper('topk', { label: 'top-k (0=off)', min: 0, max: 40, value: 0 });
    c.slider('topp', { label: 'top-p', min: 0.1, max: 1, step: 0.05, value: 0.95 });
    c.slider('seed', { label: 'seed', min: 1, max: 99, step: 1, value: 42 });
    c.button('▶ generate / ■ stop', () => { if (M.promptIds.length) generate(page); else page.redraw(); });
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear('#ffffff');
    page.probe = { source: M.source, genCount: M.gen.length, greedy: !!st.greedy, temp: +st.temp, genText: M.gen.map((g) => g.tok).join('') };

    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: AMBER };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100 | 0)}% (~548 MB, one time)`, c: AMBER };
      if (M.generating) return { t: '✍ generating…', c: AMBER };
      if (M.source === 'real') return { t: '● real GPT-2 (124M) — generating in-browser', c: GREEN };
      if (M.status === 'offline') return { t: '○ offline — synthetic continuation (click “load real GPT-2”)', c: AMBER };
      return { t: '○ synthetic stand-in — click “load real GPT-2” to write real text', c: '#586069' };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();

    // the running text: prompt (grey) + generated tokens (ink, newest highlighted)
    const pad = 18, top = 44, lh = 26, maxW = page.W - 2 * pad;
    ctx.save(); ctx.textBaseline = 'alphabetic';
    const words = [];
    words.push({ s: (M.source === 'real' ? M.prompt : '“the cat sat on the mat . the cat ran”') + '', c: '#8a9099', bg: null });
    for (let i = 0; i < M.gen.length; i++) words.push({ s: td(M.gen[i].tok), c: INK, bg: i === M.gen.length - 1 && M.generating ? '#fff3c4' : null });
    let x = pad, y = top + 18; ctx.font = '17px Georgia, "Times New Roman", serif';
    for (const w of words) {
      // naive wrap on spaces
      const parts = w.s.split(/(\s+)/);
      for (const part of parts) {
        if (!part) continue;
        const pw = ctx.measureText(part).width;
        if (x + pw > pad + maxW && /\S/.test(part)) { x = pad; y += lh; }
        if (w.bg) { ctx.fillStyle = w.bg; ctx.fillRect(x - 1, y - 15, pw + 2, 21); }
        ctx.fillStyle = w.c; ctx.fillText(part, x, y); x += pw;
      }
    }
    ctx.restore();

    const o = `${M.source === 'real' ? 'REAL GPT-2 (124M)' : 'synthetic'} generation · ${st.greedy ? 'greedy (deterministic — verified vs HF generate)' : `sampling temp=${(+st.temp).toFixed(2)} top-k=${st.topk | 0 || 'off'} top-p=${(+st.topp).toFixed(2)} seed=${st.seed | 0}`} · ${M.gen.length}/${st.ntok | 0} tokens    tier:${r.name}\n` +
      (M.source === 'real' ? 'Each token is the verified next-token step fed back in. Toggle greedy off + change the seed for different continuations.' : 'Synthetic stand-in — load real GPT-2 to generate real text.');
    page.setReadout(o);
  },
  challenges: [
    { goal: 'Ground it in the REAL model — write text with the actual GPT-2 (needs network; “load real GPT-2”).',
      hint: 'The banner turns green “● real GPT-2” after the ~548 MB one-time download.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Generate at least 12 tokens of text — hit ▶ generate.',
      hint: 'Set tokens ≥ 12 and press generate; greedy is deterministic, or turn it off to sample.',
      check: (api) => ({ solved: api.probe.genCount >= 12, detail: `generated ${api.probe.genCount} tokens (need ≥ 12)` }) },
  ],
}).then((page) => {
  window.__realGenPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('prompt')) page.controls.set('prompt', q.get('prompt'));
  if (q.has('ids')) injected = q.get('ids').split(',').map((x) => parseInt(x, 10)).filter((x) => Number.isFinite(x));
  ['ntok', 'topk', 'seed'].forEach((k) => { if (q.has(k)) page.controls.set(k, +q.get(k)); });
  ['temp', 'topp'].forEach((k) => { if (q.has(k)) page.controls.set(k, +q.get(k)); });
  if (q.has('greedy')) page.controls.set('greedy', q.get('greedy') !== '0');
  // synthetic prompt ids for the offline render (so the stand-in has a length)
  if (injected) M.promptIds = injected; else M.promptIds = [1169, 3797, 3332, 319, 262, 2603, 764, 262, 3797, 4966];
  M.n = M.promptIds.length;
  // ?gen=N runs generation synchronously (headless capture; no streaming delay)
  if (q.has('gen')) { M._fast = true; page.controls.set('ntok', Math.max(1, +q.get('gen'))); }
  page.redraw();
  if (q.get('real') !== '0') ensureReal(page); else if (q.has('gen')) generate(page);
});
