// logit-lens concept page — Phase 9 (real-model grounding).
//
// The "logit lens" (nostalgebraist): apply the model's FINAL ln_f + tied lm_head
// to the residual stream at EVERY layer, not just the last, to watch the
// next-token prediction form across depth. Early layers echo the input or guess
// generically; the answer crystallizes in the upper layers. Runs the verified
// GPT-2 forward (gpt2.js: GPT2.lens()) in-browser on a prefix you type.
//
// "the cat sat on the mat . the cat ran" →
//   ran → running → through → away → … → up   (the real model's trajectory)
//
// Breadcrumbs to ../design/architectures.md (residual stream + A5 lm_head). Plan:
// ../plan/phase9.md. Reuses the real-attention GPT-2 loader (gpt2.js).
//
// Offline: an idealized synthetic trajectory (clearly labelled) that converges to
// a target token, so the depth-by-depth view still teaches. ?real=0 forces it.
import { mount } from '../framework/layout.js';
import { loadGPT2, GPT2_CONFIG } from '../real-attention/gpt2.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2;
const GREEN = '#0a7227', AMBER = '#9a6700', BLUE = '#1f6feb', INK = '#24292e';

const DEFAULT_PROMPT = 'the cat sat on the mat . the cat ran';
// idealized synthetic trajectory (per depth: top token + final-token prob), the
// real default's shape, hand-built so the offline view still teaches.
const SYN_TOPS = ['ran', 'running', 'running', 'through', 'through', 'through', 'through', 'away', 'through', 'away', 'away', 'away', 'up'];
const SYN_FINAL = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.04, 0.05, 0.06, 0.08, 0.12, 0.22];

let M = { status: 'init', progress: 0, source: 'synthetic', prompt: DEFAULT_PROMPT, depths: [], finalTok: '', n: 0 };
let tokenizer = null, gpt2 = null, loadStarted = false, injected = null;

const td = (s) => (s == null ? '' : String(s)).replace(/^ /, '·').replace(/\n/g, '⏎') || '∅';

// derive per-depth { topId, topTok, topProb, finalProb } from lens logits
function fromLens(perLayer, decode) {
  const last = perLayer[perLayer.length - 1];
  let finalId = 0; for (let v = 1; v < last.length; v++) if (last[v] > last[finalId]) finalId = v;
  const depths = perLayer.map((lg) => {
    let mx = -Infinity, topId = 0; for (let v = 0; v < lg.length; v++) { if (lg[v] > mx) mx = lg[v]; if (lg[v] > lg[topId]) topId = v; }
    let Z = 0; for (let v = 0; v < lg.length; v++) Z += Math.exp(lg[v] - mx);
    return { topId, topTok: decode(topId), topProb: Math.exp(lg[topId] - mx) / Z, finalProb: Math.exp(lg[finalId] - mx) / Z };
  });
  return { depths, finalId, finalTok: decode(finalId) };
}

function setLens(depths, finalTok, source, status) { M = { ...M, depths, finalTok, source, status }; }
function runSynthetic(status) {
  const depths = SYN_TOPS.map((t, i) => ({ topId: -1, topTok: ' ' + t, topProb: Math.max(SYN_FINAL[i], 0.05), finalProb: SYN_FINAL[i] }));
  setLens(depths, ' up', 'synthetic', status || M.status);
}

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function ensureReal(page) {
  if (gpt2 || loadStarted) return;
  loadStarted = true;
  try {
    if (!injected) { M.status = 'loading-tok'; page.redraw(); const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js'); env.allowLocalModels = false; tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL); }
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(page.state.weights || WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    M.status = 'running'; page.redraw();
    await runReal(page, page.state.prompt || DEFAULT_PROMPT);
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

async function runReal(page, prompt) {
  if (!gpt2) return;
  M.status = 'running'; page.redraw();
  try {
    let ids;
    if (injected) ids = injected;
    else { const enc = await tokenizer(prompt || DEFAULT_PROMPT, { add_special_tokens: false }); ids = Array.from(enc.input_ids.data, Number).slice(0, 48); }
    if (!ids.length) return;
    M.prompt = prompt; M.n = ids.length;
    const lens = gpt2.lens(ids);
    const dec = injected ? ((id) => '·' + id) : ((id) => tokenizer.decode([id]));
    const { depths, finalTok } = fromLens(lens.perLayer, dec);
    setLens(depths, finalTok, 'real', 'ready'); page.redraw();
  } catch (e) { M.status = 'offline'; M.err = String(e); page.redraw(); }
}

mount({
  mount: 'body',
  slug: 'logit-lens',
  title: 'logit lens — watching a prediction form across layers',
  blurb: 'Phase 9 (real-model grounding). The logit lens applies GPT-2’s final ln_f + tied lm_head to the residual stream at EVERY layer — not just the last — so you can watch the next-token prediction crystallize with depth. Early layers echo the input or guess generically; the answer locks in near the top. Runs a REAL GPT-2 in your browser. Type a prefix and read the trajectory bottom-up (embedding → layer 12). Offline it shows an idealized stand-in.',
  prefer: 'canvas2d',
  aspect: '4 / 3',
  controls: (c, page) => {
    c.text('prompt', { label: 'prompt', value: DEFAULT_PROMPT, placeholder: 'type a prefix…', rebuild: false });
    c.button('re-run', () => { if (gpt2) runReal(page, page.state.prompt); else page.redraw(); });
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    r.clear('#ffffff');
    const d = M.depths, nD = d.length;
    // lock-in depth: first depth (from the top) whose top-1 == the final token
    let lockIn = -1; for (let i = 0; i < nD; i++) if (d[i] && M.finalTok && d[i].topTok === M.finalTok) { lockIn = i; break; }
    page.probe = { source: M.source, nDepths: nD, lockInDepth: lockIn, topProbLast: nD ? d[nD - 1].topProb : 0, finalProbLast: nD ? d[nD - 1].finalProb : 0 };

    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: AMBER };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100 | 0)}% (~548 MB, one time)`, c: AMBER };
      if (M.status === 'running') return { t: '⟳ running GPT-2…', c: AMBER };
      if (M.source === 'real') return { t: '● real GPT-2 (124M) — logit lens computed in-browser', c: GREEN };
      if (M.status === 'offline') return { t: '○ offline — idealized synthetic trajectory (click “load real GPT-2”)', c: AMBER };
      return { t: '○ idealized synthetic trajectory — click “load real GPT-2” for the real lens', c: '#586069' };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();
    if (!nD) { page.setReadout('…'); return; }

    // header: final prediction
    const pad = 16, headY = 34;
    ctx.save(); ctx.textBaseline = 'middle'; ctx.font = '13px ui-monospace, monospace';
    ctx.fillStyle = '#3a4047'; ctx.fillText(M.source === 'real' ? `“${(M.prompt || '').slice(0, 56)}”` : '“…”', pad, headY);
    ctx.fillStyle = BLUE; ctx.fillText(`→ ${M.finalTok ? `"${M.finalTok.trim() || M.finalTok}"` : ''}`, pad + 360, headY); ctx.restore();

    // rows: deepest (final) at TOP, embedding at BOTTOM — read bottom-up = shallow→deep
    const top = headY + 22, labW = 64, rowH = Math.max(13, Math.min(26, (page.H - top - 30) / nD));
    const barX = pad + labW, barW = page.W - barX - 64;
    ctx.save(); ctx.textBaseline = 'middle'; ctx.font = `${Math.min(12, rowH - 3)}px ui-monospace, monospace`;
    for (let i = 0; i < nD; i++) {
      const e = d[nD - 1 - i], depth = nD - 1 - i;        // i=0 row → deepest
      const y = top + i * rowH, locked = M.finalTok && e.topTok === M.finalTok;
      ctx.fillStyle = '#9aa4ad'; ctx.textAlign = 'right';
      ctx.fillText(depth === 0 ? 'embed' : 'L' + depth, barX - 6, y + rowH / 2);
      // bar = probability the FINAL token has at this depth (the convergence curve)
      const w = Math.max(1, Math.min(1, e.finalProb / Math.max(0.02, d[nD - 1].finalProb)) * barW * 0.6);
      ctx.fillStyle = locked ? GREEN : '#cdd3da'; ctx.fillRect(barX, y + 1, w, rowH - 3);
      // the depth's own top-1 token
      ctx.fillStyle = locked ? GREEN : INK; ctx.textAlign = 'left';
      ctx.fillText(`${td(e.topTok).slice(0, 12)}`, barX + w + 6, y + rowH / 2);
      ctx.fillStyle = '#9aa4ad'; ctx.fillText(`  ${(e.finalProb * 100).toFixed(1)}%`, barX + w + 6 + 96, y + rowH / 2);
    }
    ctx.restore();

    const finalShort = (M.finalTok || '').trim() || M.finalTok;
    const lockTxt = lockIn < 0 ? 'the answer never tops the lens' :
      lockIn >= nD - 1 ? `"${finalShort}" only becomes the top-1 token at the final layer (a late/flat prediction)` :
        `"${finalShort}" first becomes the top-1 token at ${lockIn === 0 ? 'the embedding' : 'layer ' + lockIn} and holds`;
    const o = `${M.source === 'real' ? 'REAL GPT-2 (124M)' : 'synthetic (idealized)'} logit lens · ${nD} depths (embed + ${nD - 1} layers) · final = "${finalShort}"    tier:${r.name}\n` +
      lockTxt + (M.source === 'real' ? ' — bars show the final token’s probability climbing with depth.' : ' — idealized; load real GPT-2 for the true trajectory.');
    page.setReadout(o);
  },
  challenges: [
    { goal: 'Ground it in the REAL model — compute the lens from GPT-2’s actual layers (needs network; “load real GPT-2”).',
      hint: 'The banner turns green “● real GPT-2” after the ~548 MB one-time download.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Find a CONFIDENT prediction — type a prefix whose final-layer top token exceeds 50% probability.',
      hint: 'Open-ended prompts ("the cat ran …") are flat. Try something the model is sure of, e.g. "The Eiffel Tower is in the city of".',
      check: (api) => ({ solved: api.probe.topProbLast >= 0.5, detail: `final-layer top-token prob = ${(api.probe.topProbLast * 100).toFixed(1)}% (need ≥ 50%)` }) },
  ],
}).then((page) => {
  window.__logitLensPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('prompt')) page.controls.set('prompt', q.get('prompt'));
  if (q.has('ids')) injected = q.get('ids').split(',').map((x) => parseInt(x, 10)).filter((x) => Number.isFinite(x));
  runSynthetic('init'); page.redraw();
  if (q.get('real') !== '0') ensureReal(page);
});
