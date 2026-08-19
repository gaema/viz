// real-vision concept page — real-model grounding, the vision entry.
//
// Extends real grounding to the image family. Runs a real Vision Transformer
// (Xenova/vit-base-patch16-224) in-browser via transformers.js on a real photo,
// shows the image cut into 16×16 patches (the ViT tokenization — the vision
// analog of BPE tokens, grounds the patch-embedding page), and the real top-5
// class predictions. The vision analog of real-logits (real model output);
// transformers.js is the trusted engine here (no hand-written vision forward),
// and the top class is verified against PyTorch ViT.
//
// Breadcrumbs to ../patch-embedding (ViT image→patches).
//
// Offline: a canvas-drawn placeholder + labelled synthetic predictions, with the
// real patch grid (geometry is real). ?real=0 forces it.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const MODEL = 'Xenova/vit-base-patch16-224';

const GRID = 14, PATCH = 16;                                  // 224/16 = 14×14 = 196 patches
const IMAGES = {
  cats: 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg',
  tiger: 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg',
};
// synthetic stand-in predictions (clearly fake), so the layout teaches offline
const SYN = [['(synthetic) tabby cat', 0.41], ['(synthetic) tiger cat', 0.22], ['(synthetic) lynx', 0.13], ['(synthetic) Egyptian cat', 0.09], ['(synthetic) carton', 0.04]];

let M = { status: 'init', progress: 0, source: 'synthetic', image: 'cats', preds: SYN, img: null };
let clf = null, loadStarted = false, imgCache = {};

function loadImg(name) {                                      // for canvas display
  if (imgCache[name]) { M.img = imgCache[name]; return; }
  const im = new Image(); im.crossOrigin = 'anonymous';
  im.onload = () => { imgCache[name] = im; if (M.image === name) { M.img = im; if (window.__realVisionPage) window.__realVisionPage.redraw(); } };
  im.src = IMAGES[name];
}

const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);

async function ensureReal(page) {
  if (clf || loadStarted) return; loadStarted = true; M.status = 'loading-model'; M.progress = 0; page.redraw();
  try {
    const { pipeline, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js');
    env.allowLocalModels = false;
    clf = await pipeline('image-classification', MODEL, { progress_callback: (p) => { if (p && p.status === 'progress' && p.total) { M.progress = Math.max(M.progress, p.loaded / p.total); page.redraw(); } } });
    M.status = 'ready'; M.source = 'real';
    await classify(page, page.state.image || 'cats');
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

async function classify(page, name) {
  if (!clf) return; M.status = 'running'; M.image = name; loadImg(name); page.redraw();
  try { const out = await clf(IMAGES[name], { topk: 5 }); M.preds = out.map((o) => [o.label, o.score]); M.source = 'real'; M.status = 'ready'; page.redraw(); }
  catch (e) { M.status = 'offline'; M.err = String(e); page.redraw(); }
}

mount({
  mount: 'body',
  slug: 'real-vision',
  title: 'real vision — a ViT classifying a real image',
  blurb: 'The vision entry to the real-model pages. A real Vision Transformer (ViT-base) runs in your browser on a real photo: see the image cut into 16×16 patches — the ViT “tokens”, the image analog of BPE tokens — and the model’s real top-5 predictions. The vision counterpart of real-logits. transformers.js is the engine; the top class is verified against PyTorch ViT. Offline shows a labelled synthetic stand-in with the real patch grid.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  controls: (c, page) => {
    c.select('image', { label: 'image', options: Object.keys(IMAGES), value: 'cats' });
    c.button('classify', () => { if (clf) classify(page, page.state.image); else page.redraw(); });
    c.button('load real ViT (~340 MB)', () => ensureReal(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    if (st.image && st.image !== M.image && M.source === 'real' && clf && M.status !== 'running') classify(page, st.image);
    if (st.image && st.image !== M.image) { M.image = st.image; loadImg(st.image); }
    if (!M.img && imgCache[M.image]) M.img = imgCache[M.image];
    const top = M.preds[0] || ['—', 0];
    page.probe = { source: M.source, image: M.image, topLabel: top[0], topScore: top[1], nPatches: GRID * GRID };

    const ban = (() => {
      if (M.status === 'loading-model') return { t: `↓ downloading ViT… ${(M.progress * 100 | 0)}% (~340 MB, one time)`, c: T.goldDeep };
      if (M.status === 'running') return { t: '⟳ classifying…', c: T.goldDeep };
      if (M.source === 'real') return { t: '● real ViT-base (Xenova/vit-base-patch16-224) — in-browser', c: T.okDeep };
      if (M.status === 'offline') return { t: '○ offline — synthetic stand-in (click “load real ViT”)', c: T.goldDeep };
      return { t: '○ synthetic stand-in — click “load real ViT” to classify the real image', c: T.n11 };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ban.c; ctx.fillText(ban.t, 14, 9); ctx.restore();

    const pad = 16, topY = 40;
    // ---- left: the image (224 box) with the 14×14 patch grid ----
    const side = Math.min(page.W * 0.5 - pad, page.H - topY - 30), ix = pad, iy = topY + 6;
    ctx.save();
    if (M.img) { try { ctx.drawImage(M.img, ix, iy, side, side); } catch (e) { /* tainted */ } }
    else { const g = ctx.createLinearGradient(ix, iy, ix + side, iy + side); g.addColorStop(0, T.accentBg); g.addColorStop(1, T.warnLine); ctx.fillStyle = g; ctx.fillRect(ix, iy, side, side); ctx.fillStyle = T.n10; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(M.source === 'real' ? 'image' : '(synthetic image)', ix + side / 2, iy + side / 2); }
    // patch grid (the ViT tokenization)
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1;
    for (let k = 1; k < GRID; k++) { const o = (k / GRID) * side; ctx.beginPath(); ctx.moveTo(ix + o, iy); ctx.lineTo(ix + o, iy + side); ctx.stroke(); ctx.beginPath(); ctx.moveTo(ix, iy + o); ctx.lineTo(ix + side, iy + o); ctx.stroke(); }
    ctx.strokeStyle = T.n6; ctx.strokeRect(ix, iy, side, side);
    ctx.fillStyle = T.n11; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(`${GRID}×${GRID} = ${GRID * GRID} patches of ${PATCH}×${PATCH} px (the ViT “tokens”)`, ix, iy + side + 6);
    ctx.restore();

    // ---- right: top-5 predictions ----
    const rx = pad + page.W * 0.52, rw = page.W - rx - pad, ry = topY + 18;
    r.label(M.source === 'real' ? 'ViT top-5 predictions (real)' : 'predictions (synthetic stand-in)', rx, topY, { color: T.n11, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.textBaseline = 'middle'; ctx.font = '12px ui-monospace, monospace';
    const rowH = Math.min(40, (page.H - ry - 30) / 5), pmax = (M.preds[0] || [0, 1])[1] || 1;
    for (let i = 0; i < M.preds.length && i < 5; i++) {
      const [lab, sc] = M.preds[i], y = ry + i * rowH, bw = Math.max(2, (sc / pmax) * (rw - 4));
      ctx.fillStyle = i === 0 ? T.okDeep : T.accentLine; ctx.fillRect(rx, y + 4, bw, rowH - 12);
      // the top row's bar always spans the full width, so its text sits ON the
      // filled bar -- ink it with the page GROUND for contrast in both themes.
      ctx.fillStyle = i === 0 ? T.n0 : T.n13; ctx.textAlign = 'left'; ctx.fillText(`${(lab || '').slice(0, 28)}`, rx + 4, y + rowH / 2 - 6);
      ctx.fillStyle = i === 0 ? T.n0 : T.n11; ctx.fillText(`${(sc * 100).toFixed(1)}%`, rx + 4, y + rowH / 2 + 8);
    }
    ctx.restore();

    page.setReadout(`${M.source === 'real' ? 'REAL ViT-base' : 'synthetic'} · image "${M.image}" → "${(top[0] || '').slice(0, 32)}" (${(top[1] * 100).toFixed(1)}%) · ${GRID * GRID} patches    tier:${r.name}\n` +
      `a ViT splits the image into ${PATCH}×${PATCH} patches, linearly embeds each into a token, then runs a transformer — same machinery as a language model` + (M.source === 'real' ? '. These are the real model’s predictions.' : ' (load real ViT for actual predictions).'));
  },
  challenges: [
    { goal: 'Ground it in the REAL model — run the actual ViT on the image (needs network; “load real ViT”).',
      hint: 'The banner turns green “● real ViT-base” after the ~340 MB one-time download.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Classify the cats image correctly — the real ViT’s top class should be a cat (e.g. “Egyptian cat”).',
      hint: 'Load the real model with the “cats” image selected; ViT-base predicts “Egyptian cat” at ~94%.',
      check: (api) => ({ solved: api.probe.source === 'real' && /cat/i.test(api.probe.topLabel || ''), detail: `top = "${api.probe.topLabel}"` }) },
  ],
}).then((page) => {
  window.__realVisionPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('image') && IMAGES[q.get('image')]) page.controls.set('image', q.get('image'));
  M.image = page.state.image; loadImg(M.image);
  page.redraw();
  if (q.get('real') === '1' || q.get('autoload') === '1') ensureReal(page);   // large download: opt-in only
});
