// real-embeddings concept page — Phase 9 (real-model grounding).
//
// The synthetic `embedding/` page shows the *mechanism* (a token id is a row
// lookup in E[V×D]); the numbers there come from a seed. This page shows the
// numbers are *real*: it fetches a small trained sentence-embedding model
// (all-MiniLM-L6-v2, ≈23 MB) at runtime via transformers.js (a CDN ES module —
// no build, no server compute, same ethos as every other page) and runs it
// in-browser on words you type. The cosine-similarity heatmap and the 2D MDS
// scatter then show *real* semantic geometry — "king"≈"queen", "paris"≈"tokyo",
// not an artifact of the demo.
//
// Math: each word → a unit vector v (mean-pooled, L2-normalized). cosine(i,j) =
// vᵢ·vⱼ (a dot product, since the vectors are unit norm) — the exact same
// inner-product the attention pages use, here on trained vectors. The scatter
// is classical MDS / PCoA: top-2 eigenvectors of the double-centered Gram of
// the centered vectors (PCA scores), so nearby points = similar meaning.
//
// Breadcrumbs to ../design/architectures.md A1 (token embedding) — same
// attribute as the synthetic `embedding` sibling. Plan: ../plan/phase9.md.
//
// Offline / no-network (file:// with no fetch, CDN blocked): the page never
// goes blank — it renders a deterministic *synthetic stand-in* (clearly
// labelled) so it stays headless-verifiable, and swaps in the real vectors the
// moment the model finishes downloading.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const INK = '#111', BLUE = '#1f6feb', GREEN = '#0a7227', AMBER = '#9a6700';

// Default word list: four clear semantic clusters of three. The real model
// pulls each cluster together; the synthetic stand-in is hand-seeded to do the
// same so the page still teaches offline.
const DEFAULT = 'king, queen, prince, cat, dog, horse, paris, tokyo, berlin, bread, apple, cheese';
// Cluster index per default word (synthetic stand-in only — the real model
// needs no hints). Words not in this map get a per-word hashed pseudo-vector.
const CLUSTER = { king: 0, queen: 0, prince: 0, cat: 1, dog: 1, horse: 1, paris: 2, tokyo: 2, berlin: 2, bread: 3, apple: 3, cheese: 3 };
const CLUSTER_COL = ['#1f6feb', '#cf222e', '#0a7227', '#8250df', '#9a6700', '#1b7c83'];

// Shared module state. compute() (synthetic or real) fills `M`; draw() reads it.
let M = { status: 'init', progress: 0, source: 'synthetic', model: MODEL, words: [], vecs: [], sim: null, coords: null, span: 1 };
let extractor = null, loadStarted = false, runSeq = 0;
let simRect = null, scRect = null;   // heatmap + scatter rects (for hover)

const parseWords = (s) => Array.from(new Set((s || '').split(/[,\n]/).map((w) => w.trim().toLowerCase()).filter(Boolean))).slice(0, 16);

// ---- tiny deterministic RNG (synthetic stand-in only) --------------------
function strHash(s) { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const normal = (rng) => { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

// D-dim unit vector. If the word is a known default, place it near its cluster
// centroid (so same-cluster words are genuinely similar); else a hashed pseudo
// vector (honest: offline we can't know the meaning of an arbitrary typed word).
function synthVec(word, D) {
  const c = CLUSTER[word];
  const rng = mulberry32(strHash(word) ^ 0x9e3779b9);
  const v = new Float32Array(D);
  if (c != null) {
    const cr = mulberry32((c + 1) * 0x85ebca6b);              // shared cluster centroid
    for (let k = 0; k < D; k++) v[k] = 2.2 * normal(cr) + 0.9 * normal(rng);
  } else {
    for (let k = 0; k < D; k++) v[k] = normal(rng);
  }
  let n = 0; for (let k = 0; k < D; k++) n += v[k] * v[k]; n = Math.sqrt(n) || 1;
  for (let k = 0; k < D; k++) v[k] /= n;
  return v;
}

// ---- geometry: cosine matrix + 2D MDS (PCoA) -----------------------------
function cosineMatrix(vecs) {
  const n = vecs.length, S = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let d = 0; const a = vecs[i], b = vecs[j], D = a.length;
    for (let k = 0; k < D; k++) d += a[k] * b[k];
    S[i * n + j] = d;
  }
  return S;
}

// Classical MDS via top-2 eigenvectors of the double-centered Gram of the
// column-centered vectors (== PCA scores). n is small (≤16); power iteration
// with one deflation is plenty and avoids any matrix library.
function mds2d(vecs) {
  const n = vecs.length; if (n === 0) return { coords: [], span: 1 };
  const D = vecs[0].length;
  const mean = new Float32Array(D);
  for (const v of vecs) for (let k = 0; k < D; k++) mean[k] += v[k] / n;
  const Xc = vecs.map((v) => { const r = new Float32Array(D); for (let k = 0; k < D; k++) r[k] = v[k] - mean[k]; return r; });
  // Gram G[i][j] = Xc_i · Xc_j   (n×n)
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { let d = 0; for (let k = 0; k < D; k++) d += Xc[i][k] * Xc[j][k]; G[i * n + j] = G[j * n + i] = d; }
  const topEig = (Gm) => {
    let v = new Float64Array(n); for (let i = 0; i < n; i++) v[i] = ((i * 2654435761) % 1000) / 1000 - 0.5;
    let val = 0;
    for (let it = 0; it < 240; it++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += Gm[i * n + j] * v[j]; w[i] = s; }
      let nrm = 0; for (let i = 0; i < n; i++) nrm += w[i] * w[i]; nrm = Math.sqrt(nrm) || 1;
      for (let i = 0; i < n; i++) w[i] /= nrm; val = nrm; v = w;
    }
    return { val, vec: v };
  };
  const e1 = topEig(G);
  const G2 = G.slice();
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G2[i * n + j] -= e1.val * e1.vec[i] * e1.vec[j];   // deflate
  const e2 = topEig(G2);
  const s1 = Math.sqrt(Math.max(e1.val, 0)), s2 = Math.sqrt(Math.max(e2.val, 0));
  const coords = []; let span = 1e-6;
  for (let i = 0; i < n; i++) { const x = s1 * e1.vec[i], y = s2 * e2.vec[i]; coords.push([x, y]); span = Math.max(span, Math.abs(x), Math.abs(y)); }
  return { coords, span };
}

// probe extremes (for challenges): min/max off-diagonal cosine.
function simExtremes(S, n) {
  let mn = 2, mx = -2;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const s = S[i * n + j]; if (s < mn) mn = s; if (s > mx) mx = s; }
  return { minSim: n > 1 ? mn : 1, maxSim: n > 1 ? mx : 1 };
}

function buildFrom(words, vecs, source, status) {
  const sim = cosineMatrix(vecs), { coords, span } = mds2d(vecs);
  M = { ...M, words, vecs, sim, coords, span, source, status, model: MODEL };
}

// synthetic stand-in (instant, deterministic, offline-safe)
function runSynthetic(words, status) { const D = 48; buildFrom(words, words.map((w) => synthVec(w, D)), 'synthetic', status || M.status); }

// real model (transformers.js). Resolves the page from a blank → synthetic →
// real progression. Never throws into mount(); failures degrade to synthetic.
// Reject if a promise hasn't settled in `ms` — so a stuck CDN request (one that
// neither resolves nor rejects) still degrades to the synthetic stand-in instead
// of pinning the banner on "downloading…" forever. Only the *library* import is
// timed; the model download is progress-gated (visible progress = not stuck).
const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out (' + ms / 1000 + 's) — offline?')), ms))]);

async function ensureModel(page) {
  if (extractor || loadStarted) return;
  loadStarted = true; M.status = 'loading'; M.progress = 0; page.redraw();
  try {
    const { pipeline, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js load');
    env.allowLocalModels = false;                              // always fetch from the HF hub via CDN
    extractor = await pipeline('feature-extraction', MODEL, {
      progress_callback: (p) => { if (p && p.status === 'progress' && p.total) { M.progress = Math.max(M.progress, p.loaded / p.total); page.redraw(); } },
    });
    M.status = 'ready';
    await runReal(page, M.words.length ? M.words : parseWords(page.state.words));
  } catch (e) {
    M.status = 'offline'; M.err = String(e && e.message || e); page.redraw();
  }
}

async function runReal(page, words) {
  if (!extractor || !words.length) return;
  const seq = ++runSeq; M.status = 'running'; page.redraw();
  try {
    const out = await extractor(words, { pooling: 'mean', normalize: true });
    if (seq !== runSeq) return;                                // a newer run superseded us
    const arr = out.tolist();                                  // [n][384], unit-norm
    buildFrom(words, arr.map((a) => Float32Array.from(a)), 'real', 'ready');
    page.redraw();
  } catch (e) { M.status = 'offline'; M.err = String(e && e.message || e); page.redraw(); }
}

// re-embed the typed word list: synthetic immediately (instant feedback), real
// in the background if the model is ready.
function reembed(page, raw) {
  const words = parseWords(raw);
  if (!words.length) { runSynthetic([], M.status); page.redraw(); return; }
  if (extractor && M.status !== 'offline') runReal(page, words);
  else { runSynthetic(words, M.status); page.redraw(); }
}

mount({
  mount: 'body',
  slug: 'real-embeddings',
  title: 'real embeddings — a trained model’s semantic geometry',
  blurb: 'Phase 9 (real-model grounding). The synthetic embedding page shows the row-lookup mechanism on seeded numbers; this one runs a REAL trained model (all-MiniLM-L6-v2) in your browser — fetched at runtime via transformers.js, no build — and embeds the words you type. The cosine-similarity heatmap and the 2D map below are real semantic geometry: words that mean similar things (king·queen, paris·tokyo) sit close, with no demo trickery. Type your own words. Offline, it renders a clearly-labelled synthetic stand-in and swaps in the real vectors once the model downloads.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c, page) => {
    c.text('words', { label: 'words (comma-sep)', value: DEFAULT, placeholder: 'king, queen, paris, …', rebuild: false });
    c.button('re-embed', () => reembed(page, page.state.words));
    c.button('load real model', () => ensureModel(page));
  },
  onPointer: () => {},
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    r.clear('#ffffff');
    const n = M.words.length;
    // expose probe for challenges
    const ext = M.sim ? simExtremes(M.sim, n) : { minSim: 1, maxSim: 1 };
    page.probe = { source: M.source, n, minSim: ext.minSim, maxSim: ext.maxSim };

    // ---- load-state banner ----
    const banner = (() => {
      if (M.status === 'loading') return { t: `↓ downloading model… ${(M.progress * 100 | 0)}%`, c: AMBER };
      if (M.status === 'running') return { t: '⟳ running model…', c: AMBER };
      if (M.source === 'real') return { t: '● real model — ' + MODEL, c: GREEN };
      if (M.status === 'offline') return { t: '○ offline — synthetic stand-in (click “load real model” when online)', c: AMBER };
      return { t: '○ synthetic stand-in — click “load real model” for real vectors', c: '#586069' };
    })();
    ctx.save(); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = banner.c; ctx.fillText(banner.t, 14, 10); ctx.restore();

    if (!n || !M.sim) { ctx.save(); ctx.fillStyle = '#586069'; ctx.font = '13px ui-monospace, monospace'; ctx.fillText('type some words above', 14, 40); ctx.restore(); page.setReadout('no words'); return; }

    const pad = 14, topY = 36;
    const heatW = Math.min(page.H - topY - 70, (page.W - 3 * pad) * 0.48);
    const cell = heatW / n;
    simRect = { x: pad + 56, y: topY + 8, w: cell * n, h: cell * n };

    // ---- cosine-similarity heatmap (diverging, domain [-1,1]) ----
    r.label('cosine similarity  vᵢ·vⱼ  (unit vectors)', simRect.x, simRect.y - 8, { color: '#586069', font: '11px ui-monospace, monospace' });
    r.heatmap(M.sim, { rows: n, cols: n, rect: simRect, ramp: ramps.diverging, domain: [-1, 1] });
    r.grid({ stroke: 'rgba(0,0,0,0.06)' });
    ctx.save(); ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) { ctx.fillStyle = CLUSTER_COL[(CLUSTER[M.words[i]] ?? 5) % CLUSTER_COL.length] || '#444'; ctx.fillText(M.words[i].slice(0, 8), simRect.x - 4, simRect.y + i * cell + cell / 2); }
    ctx.restore();

    // ---- 2D MDS scatter (PCA scores) ----
    const scX = simRect.x + simRect.w + 60;
    scRect = { x: scX, y: topY + 8, w: Math.max(120, page.W - pad - scX), h: simRect.h };
    ctx.save(); ctx.strokeStyle = '#e1e4e8'; ctx.lineWidth = 1; ctx.strokeRect(scRect.x, scRect.y, scRect.w, scRect.h);
    ctx.fillStyle = '#586069'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('2D map (MDS — close = similar meaning)', scRect.x, scRect.y - 2);
    const sp = M.span * 1.15, cx = scRect.x + scRect.w / 2, cy = scRect.y + scRect.h / 2;
    const sx = (scRect.w * 0.46) / sp, sy = (scRect.h * 0.46) / sp;
    for (let i = 0; i < n; i++) {
      const [x, y] = M.coords[i], px = cx + x * sx, py = cy - y * sy;
      const col = CLUSTER_COL[(CLUSTER[M.words[i]] ?? 5) % CLUSTER_COL.length] || '#444';
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill();
      ctx.fillStyle = INK; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(' ' + M.words[i].slice(0, 9), px + 4, py);
    }
    ctx.restore();

    // ---- hover-to-inspect ----
    if (page.pointer.over) {
      const p = page.pointer, h = simRect && cellAt(simRect, n, n, p.x, p.y);
      if (h) page.setTip(`cosine("${M.words[h.r]}", "${M.words[h.c]}")\n= ${M.sim[h.r * n + h.c].toFixed(3)}\n${M.source === 'real' ? '(real ' + MODEL + ')' : '(synthetic stand-in)'}`);
    }

    const o = `${M.source === 'real' ? 'REAL' : 'synthetic'} embeddings · ${n} words · cosine = vᵢ·vⱼ (unit vectors) · 2D = MDS/PCA scores    tier:${r.name}\n` +
      `most-similar pair cosine ≈ ${ext.maxSim.toFixed(2)} · least-similar ≈ ${ext.minSim.toFixed(2)}` +
      (M.source === 'real' ? '  — these are an actual trained model’s numbers.' : '  — synthetic stand-in (clusters hand-seeded); load the real model to see real geometry.');
    page.setReadout(o);
  },
  challenges: [
    { goal: 'Ground the page in a REAL model — switch off the synthetic stand-in (needs network; click “load real model”).',
      hint: 'The banner turns green “● real model …” once ≈23 MB has downloaded. Works on the mesh / online.',
      check: (api) => ({ solved: api.probe.source === 'real', detail: `source = ${api.probe.source}` }) },
    { goal: 'Type a TIGHT cluster: replace the words so every word is closely related — minimum pairwise cosine ≥ 0.30.',
      hint: 'Try one theme, e.g. “king, queen, prince, monarch, royal, throne”. The default mixes four themes, so its minimum cosine is near 0.',
      check: (api) => ({ solved: api.probe.n >= 2 && api.probe.minSim >= 0.30, detail: `min pairwise cosine = ${(+api.probe.minSim).toFixed(2)} (need ≥ 0.30)` }) },
  ],
}).then((page) => {
  window.__realEmbPage = page;
  const q = new URLSearchParams(location.search);
  // ?words=a,b,c overrides the default list (also a headless hook).
  const initial = q.has('words') ? q.get('words') : DEFAULT;
  if (q.has('words')) page.controls.set('words', initial);
  runSynthetic(parseWords(initial), 'init');          // instant render
  // ?hover=x,y fakes the cursor (headless tooltip hook).
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
  // Kick off the real model unless explicitly suppressed (?real=0 keeps the
  // synthetic stand-in — used for fast, deterministic headless capture).
  if (q.get('real') !== '0') ensureModel(page);
});
