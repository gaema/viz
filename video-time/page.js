// video-time concept page -- time as a THIRD patching axis, and the arithmetic
// that follows from it.
//
// A video model does not invent a new mechanism for time. It extends the same
// patching the sibling `patch-embedding` page shows for a still image into a
// third dimension:
//
//   1. a CAUSAL 3-D AUTOENCODER compresses the clip in space AND in time
//      (a common recipe divides each spatial axis by 8 and the time axis by 4).
//      "Causal" here means the first frame is handled on its own, so a single
//      image is still representable: F original frames become
//      1 + floor((F-1)/t) latent frames.
//   2. the transformer PATCHIFIES that latent volume (a p x p patch on each
//      latent frame), giving a sequence of SPACE-TIME TOKENS.
//   3. self-attention runs over those tokens -- and its cost grows with the
//      SQUARE of how many there are. That square is the whole problem.
//
// Every number on this page is computed live from the controls; nothing is
// quoted from a particular product. The mitigations panel prices the three
// attention shapes against each other and states what each one gives up.
//
// Sources for the mechanism (public):
//   - "Video generation models as world simulators" (space-time patches):
//     https://openai.com/index/video-generation-models-as-world-simulators/
//   - CogVideoX, arXiv:2408.06072 (3-D causal VAE + expert transformer)
//
// Interactions, per the shared framework's contract:
//   - TRANSPORT through the pipeline (frames -> compress -> tokens -> attend),
//     autoplaying and looping.
//   - DIRECT MANIPULATION: drag any of the resolution / fps / duration /
//     compression controls and every figure moves together; drag the marker on
//     the cost curve to change the clip length; drag on the frame plate to move
//     the selected space-time patch; click a row of the mitigations table to
//     switch the attention shape.
//   - HOVER anything -- every ladder row, bar, curve, matrix and table row
//     carries the arithmetic behind its figure.
//   - URL hooks for all of it, plus ?step=N.
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';

// --- formatting --------------------------------------------------------------
const UNITS = [[1e18, 'E'], [1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
function fmtN(v) {
  if (!isFinite(v)) return '∞';
  for (const [m, s] of UNITS) if (v >= m) { const q = v / m; return (q < 10 ? q.toFixed(2) : q < 100 ? q.toFixed(1) : q.toFixed(0)) + ' ' + s; }
  return String(Math.round(v));
}
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
// A ratio is reported as a PERCENT OF a named baseline. Attention cost is a
// LOWER-is-better quantity, so >100% is WORSE -- the caller says so in words.
const pctOf = (v, b) => (b > 0 ? (v / b) * 100 : 0);
const fmtPct = (p) => (p >= 1e6 ? fmtN(p) : p >= 1000 ? p.toFixed(0) : p >= 10 ? p.toFixed(0) : p >= 1 ? p.toFixed(1) : p >= 0.01 ? p.toFixed(2) : p > 0 ? '<0.01' : '0') + '%';

// A long text context to hold the video figures against. 128 K tokens is a
// context length ordinary chat models are routinely asked for.
const TEXT_CTX = 128000;
const TEXT_COST = TEXT_CTX * TEXT_CTX;

const SECMIN = 0.5, SECMAX = 30;
const VIT_PATCH = 16;   // the 2-D-only baseline: one 16x16 patch per frame

// --- the arithmetic ----------------------------------------------------------
// Everything below is exact counting, not a measurement of any implementation.
function base(st) {
  const W0 = st.width | 0;
  const H0 = Math.round((W0 * 9 / 16) / 8) * 8;          // 16:9, kept a multiple of 8
  const sw = +st.sw, tc = +st.tc, lp = +st.lp;
  const latW = Math.floor(W0 / sw), latH = Math.floor(H0 / sw);
  const gw = Math.max(1, Math.floor(latW / lp)), gh = Math.max(1, Math.floor(latH / lp));
  const S = gw * gh;                                      // spatial tokens per latent frame
  const win = st.win | 0;
  return { W0, H0, sw, tc, lp, latW, latH, gw, gh, S, fps: st.fps | 0, win, wsp: win * win, wt: win };
}

function costsAt(b, sec) {
  const F = Math.max(1, Math.round(b.fps * sec));
  // Causal 3-D autoencoder: the first frame is its own latent frame, so a
  // single image (F = 1) is still representable.
  const Tt = 1 + Math.floor((F - 1) / b.tc);
  const N = b.S * Tt;
  const full = N * N;                                     // every space-time pair
  const fact = N * (b.S + Tt);                            // Tt*S^2 (spatial) + S*Tt^2 (temporal)
  const k = Math.min(b.S, b.wsp) * Math.min(Tt, b.wt);    // neighbours inside a local 3-D window
  const win = N * k;
  return { F, Tt, N, full, fact, win, k };
}

const SHAPES = ['full', 'factorised', 'windowed'];
const shapeCost = (c, s) => (s === 'full' ? c.full : s === 'factorised' ? c.fact : c.win);
const shapeColor = (s) => (s === 'full' ? T.bad : s === 'factorised' ? T.accent : T.ok);
const SHAPE_LABEL = {
  full: 'full 3-D attention',
  factorised: 'factorised: spatial, then temporal',
  windowed: 'windowed / local 3-D',
};
const SHAPE_GIVES_UP = {
  full: 'nothing — exact, every space-time pair scored. The cost is the square.',
  factorised: 'a single layer cannot relate two tokens that differ in BOTH space and time; that has to route through an intermediate position over two passes.',
  windowed: 'anything outside the window is invisible to that layer; long-range motion needs depth or extra global tokens.',
};

function calc(st) {
  const b = base(st);
  const c = costsAt(b, +st.seconds);
  const rawValues = b.W0 * b.H0 * c.F * 3;
  const raw2d = Math.floor(b.H0 / VIT_PATCH) * Math.floor(b.W0 / VIT_PATCH) * c.F;
  const shape = SHAPES.includes(st.attn) ? st.attn : 'full';
  return { b, c, rawValues, raw2d, shape, cost: shapeCost(c, shape) };
}

// --- transport ---------------------------------------------------------------
const STAGES = [
  { label: 'raw frames — a pixel volume with a time axis' },
  { label: '3-D causal autoencoder — compress in space AND in time' },
  { label: 'patchify the latent — space-time tokens' },
  { label: 'attend — cost grows with the SQUARE of the token count' },
];

// --- interaction state -------------------------------------------------------
let selR = 2, selC = 3;
let frameRect = null, curveRect = null, ladderRows = [], barRows = [], tableRows = [], matRect = null;
let dragPatch = false, dragSec = false;

function setSeconds(page, v) {
  const s = clamp(Math.round(v * 2) / 2, SECMIN, SECMAX);
  if (Math.abs(s - +page.state.seconds) < 1e-9) return;
  page.controls.set('seconds', s, { rebuild: true });
}

mount({
  mount: 'body',
  title: 'video-time — time is just another axis you tokenize',
  blurb: 'A video model does not invent a mechanism for time — it extends patching into a third dimension. A causal 3-D autoencoder compresses the clip spatially AND temporally (a common recipe divides each spatial axis by 8 and time by 4, with the first frame handled on its own so a single image is still representable), the transformer patchifies that latent volume, and self-attention then runs over the resulting SPACE-TIME tokens. Everything here is counted live from the controls: drag resolution, frame rate, duration or either compression factor and watch the raw pixels, the token count and — the punchline — the attention cost move together. Attention scores every pair of tokens, so its cost is the SQUARE of the count: a few seconds of ordinary video is already an enormous sequence before any compression, which is why the compression is not an optimisation but the thing that makes the model possible at all. The lower panel prices the honest mitigations — full 3-D is exact and quadratic, factorised spatial-then-temporal is far cheaper and cannot express some space-time interactions, windowed sits between — and switching between them changes the SHAPE of the cost curve, not just its height. Drag the marker on that curve, drag a patch on the frame plate to see which pixels across which frames it swallows, and hover anything for the arithmetic behind it.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  autoplay: true,
  animate: true,
  challenges: [
    {
      goal: 'Find a setting where one 4-second clip needs more attention work than a 128 K-token text context.',
      hint: 'raise the resolution, or weaken the spatial / temporal compression — both raise the token count, and cost is its square.',
      check: (api) => ({ solved: (api.probe.cost ?? 0) > TEXT_COST, detail: `attention cost ${fmtPct(pctOf(api.probe.cost ?? 0, TEXT_COST))} of the 128 K text context (lower is better; 100% = parity)` }),
    },
    {
      goal: 'Now bring that same clip back under the text context WITHOUT touching resolution, frame rate or duration.',
      hint: 'stronger compression, a bigger latent patch, or a cheaper attention shape — and read what the shape gives up.',
      check: (api) => ({ solved: (api.probe.cost ?? 1e99) <= TEXT_COST, detail: (api.probe.cost ?? 0) <= TEXT_COST ? `${fmtPct(pctOf(api.probe.cost ?? 0, TEXT_COST))} of the text context — under it` : 'still over the text context' }),
    },
  ],
  controls: (c, page) => {
    c.slider('width', { label: 'frame width (px, 16:9)', min: 192, max: 1920, step: 64, value: 768, rebuild: true });
    c.slider('fps', { label: 'frames per second', min: 4, max: 60, step: 2, value: 24, rebuild: true });
    c.slider('seconds', { label: 'clip length (s)', min: SECMIN, max: SECMAX, step: 0.5, value: 4, rebuild: true });
    c.select('sw', { label: 'spatial compression (÷ per axis)', options: ['4', '8', '16'], value: '8', rebuild: true });
    c.select('tc', { label: 'temporal compression (÷ time)', options: ['1', '2', '4', '8'], value: '4', rebuild: true });
    c.select('lp', { label: 'latent patch p (p×p)', options: ['1', '2'], value: '2', rebuild: true });
    c.select('attn', { label: 'attention shape', options: [{ value: 'full', label: 'full 3-D (exact)' }, { value: 'factorised', label: 'factorised space→time' }, { value: 'windowed', label: 'windowed / local' }], value: 'full' });
    c.slider('win', { label: 'window (tokens per axis)', min: 2, max: 24, step: 1, value: 6 });
    c.slider('lf', { label: 'selected latent frame', min: 0, max: 40, step: 1, value: 1 });
    c.transport({ compute: () => STAGES.map((s, i) => ({ ...s, idx: i })), speed: 0.9, loop: true });
  },

  // DIRECT MANIPULATION: the frame plate moves the space-time patch; the cost
  // curve's marker moves the clip length. Clicking a mitigations row switches
  // the attention shape.
  onPointer: (page, ev) => {
    const st = page.state, b = base(st);
    if (frameRect) {
      const fr = frameRect;
      const at = (x, y) => {
        if (x < fr.x || x > fr.x + fr.w || y < fr.y || y > fr.y + fr.h) return null;
        return { c: clamp(Math.floor((x - fr.x) / fr.w * b.gw), 0, b.gw - 1), r: clamp(Math.floor((y - fr.y) / fr.h * b.gh), 0, b.gh - 1) };
      };
      if (ev.type === 'down') { const p = at(ev.x, ev.y); dragPatch = !!p; if (p) { selR = p.r; selC = p.c; page.redraw(); } }
      else if (ev.type === 'move' && dragPatch && page.pointer.down) { const p = at(ev.x, ev.y); if (p) { selR = p.r; selC = p.c; page.redraw(); } }
    }
    if (curveRect) {
      const cv = curveRect;
      const inside = ev.x >= cv.x - 8 && ev.x <= cv.x + cv.w + 8 && ev.y >= cv.y - 10 && ev.y <= cv.y + cv.h + 12;
      const toSec = (x) => SECMIN * Math.pow(SECMAX / SECMIN, clamp((x - cv.x) / cv.w, 0, 1));
      if (ev.type === 'down' && !dragPatch) { dragSec = inside; if (dragSec) setSeconds(page, toSec(ev.x)); }
      else if (ev.type === 'move' && dragSec && page.pointer.down) setSeconds(page, toSec(ev.x));
    }
    if (ev.type === 'down' && !dragPatch && !dragSec) {
      for (const row of tableRows) {
        if (ev.x >= row.x && ev.x <= row.x + row.w && ev.y >= row.y && ev.y <= row.y + row.h) { page.controls.set('attn', row.shape); page.redraw(); break; }
      }
    }
    if (ev.type === 'up' || ev.type === 'leave') { dragPatch = false; dragSec = false; }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    const v = calc(st), b = v.b, c = v.c;
    page.probe = { cost: v.cost, tokens: c.N, raw2d: v.raw2d, shape: v.shape };
    const tr = page.controls._transport;
    const idx = tr ? tr.index : STAGES.length - 1;
    const rev = idx < 0 ? 0 : clamp(idx, 0, 3);
    selR = clamp(selR, 0, b.gh - 1); selC = clamp(selC, 0, b.gw - 1);
    const lf = clamp(st.lf | 0, 0, c.Tt - 1);

    const mono = (px, bold) => `${bold ? 'bold ' : ''}${px}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    // Wrap a sentence to `n` characters per line, at most `max` lines.
    const wrap = (s, n, max) => {
      const out = []; let line = '';
      for (const w of s.split(' ')) {
        if ((line + ' ' + w).trim().length > n) { out.push(line.trim()); line = w; if (out.length === max - 1 && s.length) { /* last line collects the rest */ } }
        else line = (line + ' ' + w).trim();
      }
      out.push(line.trim());
      if (out.length > max) { const tail = out.slice(max - 1).join(' '); out.length = max - 1; out.push(tail.length > n ? tail.slice(0, n - 1) + '…' : tail); }
      return out;
    };
    const pad = 14;
    const botY = Math.round(H * 0.52);

    // ===================== stage banner ======================================
    r.label(`${idx < 0 ? 1 : rev + 1} / 4 · ${STAGES[rev].label}`, pad, 18, { color: T.violet, font: mono(12, true) });

    // ===================== A. the frame plates ===============================
    // Original frames that collapse into ONE latent frame, with the transformer
    // patch grid on the front plate.
    const ax = pad, ay = 44;
    const plateW = Math.min(210, Math.round(W * 0.235)), plateH = Math.round(plateW * 9 / 16);
    const f0 = lf === 0 ? 0 : 1 + (lf - 1) * b.tc;
    const f1 = lf === 0 ? 0 : Math.min(c.F - 1, f0 + b.tc - 1);
    const nGroup = f1 - f0 + 1;
    const plates = Math.min(4, nGroup);
    const off = 7;
    r.label(`frames  ${b.W0}×${b.H0}×3`, ax, ay - 8, { color: T.n14, font: mono(11, true) });
    ctx.save();
    for (let i = plates - 1; i >= 0; i--) {
      const px = ax + i * off, py = ay + i * off;
      ctx.fillStyle = i === 0 ? T.n2 : T.n3;
      ctx.fillRect(px, py, plateW, plateH);
      // a moving disc, so the plates read as a clip rather than a still
      const ph = plates > 1 ? i / (plates - 1) : 0;
      ctx.fillStyle = alphaOf(T.violet, i === 0 ? 0.5 : 0.22);
      ctx.beginPath(); ctx.arc(ax + i * off + plateW * (0.24 + 0.44 * ph), ay + i * off + plateH * (0.62 - 0.28 * ph), plateH * 0.17, 0, 7); ctx.fill();
      ctx.strokeStyle = i === 0 ? T.n8 : T.n5; ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, py + 0.5, plateW, plateH);
    }
    ctx.restore();
    frameRect = { x: ax, y: ay, w: plateW, h: plateH };
    // patch grid on the front plate -- thinned so a 240-column grid stays legible
    ctx.save();
    ctx.strokeStyle = rgbaToken('n14', 0.16); ctx.lineWidth = 0.6;
    const stepC = Math.max(1, Math.ceil(b.gw / 40)), stepR = Math.max(1, Math.ceil(b.gh / 24));
    for (let i = 0; i <= b.gw; i += stepC) { const x = ax + i / b.gw * plateW; ctx.beginPath(); ctx.moveTo(x, ay); ctx.lineTo(x, ay + plateH); ctx.stroke(); }
    for (let j = 0; j <= b.gh; j += stepR) { const y = ay + j / b.gh * plateH; ctx.beginPath(); ctx.moveTo(ax, y); ctx.lineTo(ax + plateW, y); ctx.stroke(); }
    // the selected space-time patch, on every plate of the group
    const cw = plateW / b.gw, chh = plateH / b.gh;
    for (let i = plates - 1; i >= 0; i--) {
      ctx.strokeStyle = i === 0 ? T.warn : alphaOf(T.warn, 0.45); ctx.lineWidth = i === 0 ? 2 : 1.2;
      ctx.strokeRect(ax + i * off + selC * cw - 0.5, ay + i * off + selR * chh - 0.5, Math.max(3, cw) + 1, Math.max(3, chh) + 1);
    }
    ctx.restore();
    const aBot = ay + plateH + (plates - 1) * off;
    r.label(`${c.F} frames · grid ${b.gw}×${b.gh}${stepC > 1 || stepR > 1 ? ' (thinned)' : ''}`, ax, aBot + 14, { color: T.n10, font: mono(9) });
    const pxSpan = b.sw * b.lp;
    const perTok = pxSpan * pxSpan * nGroup * 3;
    r.label(`patch (${selR},${selC}) of latent frame ${lf}  ↔ drag`, ax, aBot + 30, { color: T.warn, font: mono(10, true) });
    r.label(`covers ${pxSpan}×${pxSpan} px`, ax, aBot + 44, { color: T.n12, font: mono(9.5) });
    r.label(`× frames ${f0}–${f1} × 3 channels`, ax, aBot + 56, { color: T.n12, font: mono(9.5) });
    r.label(`= ${fmtN(perTok)} raw values → 1 token`, ax, aBot + 68, { color: T.n13, font: mono(9.5, true) });

    // ===================== B. the arithmetic ladder ==========================
    const bx = ax + plateW + 34, bw = Math.max(210, Math.round(W * 0.33));
    const rows = [
      { minRev: 0, k: 'raw pixel values', val: v.rawValues, col: T.n11, f: `${b.W0}×${b.H0}×3 × ${c.F} frames`, tip: `Every sample the camera produced.\n${b.W0} × ${b.H0} × 3 channels × ${c.F} frames = ${fmtN(v.rawValues)}` },
      { minRev: 0, k: '2-D patching only', val: v.raw2d, col: T.n9, f: `${VIT_PATCH}×${VIT_PATCH} per frame, no compression`, tip: `Patch each frame on its own, as an image model would.\n⌊${b.H0}/${VIT_PATCH}⌋ × ⌊${b.W0}/${VIT_PATCH}⌋ × ${c.F} = ${fmtN(v.raw2d)} tokens.\nThis is the sequence a video model would face with no temporal compression at all.` },
      { minRev: 1, k: 'latent volume', val: b.latW * b.latH * c.Tt, col: T.teal, f: `${b.latW}×${b.latH}×${c.Tt}  (÷${b.sw} space, ÷${b.tc} time)`, tip: `Causal 3-D autoencoder.\nspace: ${b.W0}/${b.sw} × ${b.H0}/${b.sw} = ${b.latW}×${b.latH}\ntime:  1 + ⌊(${c.F}−1)/${b.tc}⌋ = ${c.Tt} latent frames\nThe leading 1 is the causal first frame — it is why a single image (F=1) is still representable.` },
      { minRev: 2, k: 'space-time tokens', val: c.N, col: T.violet, f: `${b.gw}×${b.gh}×${c.Tt}  (patch ${b.lp}×${b.lp})`, tip: `Patchify the latent: ${b.gw} × ${b.gh} = ${fmtN(b.S)} spatial tokens per latent frame, × ${c.Tt} latent frames = ${fmtN(c.N)} tokens.\nThat is ${fmtPct(pctOf(c.N, v.raw2d))} of the 2-D-patching-only count (lower is better here; 100% = no saving).` },
      { minRev: 3, k: `attention (${v.shape})`, val: v.cost, col: shapeColor(v.shape), f: v.shape === 'full' ? `N² = ${fmtN(c.N)}²` : v.shape === 'factorised' ? `N·(S+T) = ${fmtN(c.N)}·(${fmtN(b.S)}+${c.Tt})` : `N·k = ${fmtN(c.N)}·${fmtN(c.k)}`, tip: `Attention scores PAIRS of tokens, so its cost is counted in score entries, per head, per layer.\nfull 3-D:     N² = ${fmtN(c.full)}\nfactorised:   N·(S+T) = ${fmtN(c.fact)}\nwindowed:     N·k, k = ${fmtN(c.k)} → ${fmtN(c.win)}` },
    ];
    r.label('the arithmetic, counted live', bx, ay - 8, { color: T.n14, font: mono(11, true) });
    const rowH = 42;
    const logMax = Math.log10(Math.max(10, v.rawValues, c.full));
    ladderRows = [];
    rows.forEach((row, i) => {
      if (row.minRev > rev) return;
      const ry = ay - 2 + i * rowH;
      const fr = clamp(Math.log10(Math.max(1, row.val)) / logMax, 0.02, 1);
      ctx.save();
      ctx.fillStyle = alphaOf(row.col, 0.16); ctx.fillRect(bx, ry + 12, bw * fr, 11);
      ctx.strokeStyle = alphaOf(row.col, 0.5); ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, ry + 12.5, Math.max(1, bw * fr), 11);
      ctx.restore();
      r.label(row.k, bx, ry + 8, { color: row.col, font: mono(10.5, true) });
      r.label(fmtN(row.val), bx + bw - 66, ry + 8, { color: row.col, font: mono(11, true) });
      r.label(row.f, bx + 2, ry + 32, { color: T.n10, font: mono(9) });
      ladderRows.push({ x: bx, y: ry, w: bw, h: 26, tip: row.tip });
    });

    // ===================== C. against a long text context ====================
    const cx = bx + bw + 30, cwid = Math.max(150, W - cx - pad);
    barRows = [];
    if (rev >= 3 && cwid > 130) {
      r.label('vs a 128 K-token text context', cx, ay - 8, { color: T.n14, font: mono(11, true) });
      const pairs = [
        { k: 'tokens', a: c.N, bl: TEXT_CTX, col: T.violet, tip: `${fmtN(c.N)} space-time tokens vs a ${fmtN(TEXT_CTX)}-token text context = ${fmtPct(pctOf(c.N, TEXT_CTX))} of it.` },
        { k: 'attention score entries', a: v.cost, bl: TEXT_COST, col: shapeColor(v.shape), tip: `${fmtN(v.cost)} vs ${fmtN(TEXT_COST)} = ${fmtPct(pctOf(v.cost, TEXT_COST))} of the text context's full attention (lower is better; 100% = parity).` },
      ];
      let byy = ay - 2;
      for (const p of pairs) {
        const m = Math.max(p.a, p.bl);
        r.label(p.k, cx, byy + 8, { color: T.n12, font: mono(10, true) });
        ctx.save();
        ctx.fillStyle = alphaOf(p.col, 0.55); ctx.fillRect(cx, byy + 14, Math.max(1, cwid * p.a / m), 10);
        ctx.fillStyle = alphaOf(T.n9, 0.55); ctx.fillRect(cx, byy + 27, Math.max(1, cwid * p.bl / m), 10);
        ctx.restore();
        r.label(`video ${fmtN(p.a)}`, cx + 3, byy + 23, { color: inkOn(T.n0), font: mono(9) });
        r.label(`text  ${fmtN(p.bl)}`, cx + 3, byy + 36, { color: T.n11, font: mono(9) });
        r.label(`${fmtPct(pctOf(p.a, p.bl))} of the text context`, cx, byy + 50, { color: p.a > p.bl ? T.bad : T.ok, font: mono(9.5, true) });
        barRows.push({ x: cx, y: byy, w: cwid, h: 52, tip: p.tip });
        byy += 60;
      }

      // attention-shape schematic: which pairs actually get scored
      const mS = 6, mT = 4, mN = mS * mT, msz = Math.min(78, cwid - 4, Math.max(40, botY - byy - 42)), mcell = msz / mN;
      const mx = cx, my = byy + 14;
      matRect = { x: mx, y: my, w: msz, h: msz };
      r.label('which pairs get scored', mx, my - 6, { color: T.n12, font: mono(10, true) });
      const bandFrac = clamp(c.k / Math.max(1, c.N), 0, 1);
      const band = Math.max(1, Math.round(bandFrac * mN / 2));
      ctx.save();
      ctx.fillStyle = T.n2; ctx.fillRect(mx, my, msz, msz);
      ctx.fillStyle = alphaOf(shapeColor(v.shape), 0.75);
      for (let i = 0; i < mN; i++) for (let j = 0; j < mN; j++) {
        const ti = (i / mS) | 0, si = i % mS, tj = (j / mS) | 0, sj = j % mS;
        const on = v.shape === 'full' ? true : v.shape === 'factorised' ? (ti === tj || si === sj) : Math.abs(i - j) <= band;
        if (on) ctx.fillRect(mx + j * mcell, my + i * mcell, Math.ceil(mcell), Math.ceil(mcell));
      }
      ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(mx + 0.5, my + 0.5, msz, msz);
      ctx.restore();
      r.label(`schematic: ${mT} latent frames × ${mS} spatial tokens`, mx, my + msz + 12, { color: T.n10, font: mono(9) });
      barRows.push({ x: mx, y: my, w: msz, h: msz, tip: `Rows = query token, columns = key token, ordered (latent frame, spatial position).\n${SHAPE_LABEL[v.shape]}: ${v.shape === 'full' ? 'every cell is scored.' : v.shape === 'factorised' ? 'a spatial pass (same frame) plus a temporal pass (same position) — the off-diagonal blocks are never scored in one layer.' : 'only a band around the diagonal — a local 3-D neighbourhood.'}` });
    }

    // ===================== D. the cost curve =================================
    const dx = pad + 40, dw = Math.max(160, Math.round(W * 0.54));
    const dy = botY + 24, dh = Math.max(90, H - dy - 28);
    curveRect = { x: dx, y: dy, w: dw, h: dh };
    r.label('attention cost vs clip length — drag the marker ↔', pad, botY + 12, { color: T.n14, font: mono(11, true) });
    const SX = (s) => dx + Math.log(s / SECMIN) / Math.log(SECMAX / SECMIN) * dw;
    // log-y, spanning every shape across the whole clip-length axis
    let lo = Infinity, hi = 0;
    const samples = [];
    for (let i = 0; i <= 60; i++) {
      const s = SECMIN * Math.pow(SECMAX / SECMIN, i / 60);
      const cc = costsAt(b, s);
      samples.push({ s, cc });
      for (const sh of SHAPES) { const q = shapeCost(cc, sh); if (q > 0) { lo = Math.min(lo, q); hi = Math.max(hi, q); } }
    }
    if (!isFinite(lo) || lo <= 0) lo = 1;
    const lLo = Math.log10(lo), lHi = Math.log10(Math.max(hi, lo * 10));
    const KY = (q) => dy + dh - clamp((Math.log10(Math.max(1, q)) - lLo) / (lHi - lLo), 0, 1) * dh;
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1; ctx.strokeRect(dx + 0.5, dy + 0.5, dw, dh);
    // decade gridlines
    ctx.strokeStyle = rgbaToken('n14', 0.07);
    for (let e = Math.ceil(lLo); e <= lHi; e++) { const y = KY(Math.pow(10, e)); ctx.beginPath(); ctx.moveTo(dx, y); ctx.lineTo(dx + dw, y); ctx.stroke(); }
    // the text-context reference
    if (TEXT_COST >= lo && TEXT_COST <= hi) {
      ctx.strokeStyle = alphaOf(T.n9, 0.8); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(dx, KY(TEXT_COST)); ctx.lineTo(dx + dw, KY(TEXT_COST)); ctx.stroke(); ctx.setLineDash([]);
      r.label('128 K text context, full attention', dx + 4, KY(TEXT_COST) - 4, { color: T.n10, font: mono(8.5) });
    }
    for (const sh of SHAPES) {
      const on = sh === v.shape;
      ctx.strokeStyle = on ? shapeColor(sh) : alphaOf(shapeColor(sh), 0.3);
      ctx.lineWidth = on ? 2.4 : 1.1;
      ctx.beginPath();
      samples.forEach((p, i) => { const X = SX(p.s), Y = KY(shapeCost(p.cc, sh)); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
      ctx.stroke();
    }
    // marker on the selected curve
    const mkX = SX(clamp(+st.seconds, SECMIN, SECMAX)), mkY = KY(v.cost);
    ctx.strokeStyle = rgbaToken('n14', 0.35); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(mkX, dy); ctx.lineTo(mkX, dy + dh); ctx.stroke();
    ctx.fillStyle = shapeColor(v.shape); ctx.beginPath(); ctx.arc(mkX, mkY, 5, 0, 7); ctx.fill();
    ctx.fillStyle = T.n0; ctx.beginPath(); ctx.arc(mkX, mkY, 2, 0, 7); ctx.fill();
    ctx.restore();
    r.label('0.5 s', dx - 4, dy + dh + 13, { color: T.n10, font: mono(8.5) });
    r.label('30 s', dx + dw - 22, dy + dh + 13, { color: T.n10, font: mono(8.5) });
    ctx.save(); ctx.translate(pad + 6, dy + dh / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = T.n10; ctx.font = mono(8.5); ctx.textAlign = 'center'; ctx.fillText('score entries (log)', 0, 0); ctx.restore();

    // ===================== E. the mitigations, priced ========================
    const ex = dx + dw + 26, ewid = Math.max(140, W - ex - pad);
    r.label('mitigations — click a row to switch', ex, botY + 12, { color: T.n14, font: mono(11, true) });
    tableRows = [];
    const erh = Math.max(46, Math.min(74, dh / 3));
    const giveCols = Math.max(24, Math.floor((ewid - 22) / 5.3));
    SHAPES.forEach((sh, i) => {
      const q = shapeCost(c, sh), ry = dy + i * erh, on = sh === v.shape;
      ctx.save();
      if (on) { ctx.fillStyle = alphaOf(shapeColor(sh), 0.12); ctx.fillRect(ex, ry, ewid, erh - 6); ctx.strokeStyle = shapeColor(sh); ctx.lineWidth = 1.4; ctx.strokeRect(ex + 0.5, ry + 0.5, ewid, erh - 6); }
      ctx.fillStyle = shapeColor(sh); ctx.fillRect(ex + 5, ry + 7, 7, 7);
      ctx.restore();
      r.label(SHAPE_LABEL[sh], ex + 17, ry + 14, { color: on ? shapeColor(sh) : T.n12, font: mono(9.5, true) });
      r.label(`${fmtN(q)}  ·  ${fmtPct(pctOf(q, c.full))} of full 3-D`, ex + 17, ry + 26, { color: on ? T.n13 : T.n10, font: mono(9) });
      const give = SHAPE_GIVES_UP[sh];
      wrap('gives up: ' + give, giveCols, 3).forEach((ln, k) => r.label(ln, ex + 17, ry + 37 + k * 10, { color: T.n10, font: mono(8.5) }));
      tableRows.push({ x: ex, y: ry, w: ewid, h: erh - 6, shape: sh, tip: `${SHAPE_LABEL[sh]}\ncost ${fmtN(q)} score entries = ${fmtPct(pctOf(q, c.full))} of full 3-D (lower is better; 100% = parity)\ngives up: ${give}` });
    });

    // ===================== hover =============================================
    if (page.pointer.over && !dragPatch && !dragSec) {
      const p = page.pointer;
      const hit = (arr) => arr.find((q) => p.x >= q.x && p.x <= q.x + q.w && p.y >= q.y && p.y <= q.y + q.h);
      const h = hit(ladderRows) || hit(barRows) || hit(tableRows);
      if (h) page.setTip(h.tip);
      else if (frameRect && p.x >= frameRect.x && p.x <= frameRect.x + frameRect.w && p.y >= frameRect.y && p.y <= frameRect.y + frameRect.h) {
        const hc = clamp(Math.floor((p.x - frameRect.x) / frameRect.w * b.gw), 0, b.gw - 1);
        const hr = clamp(Math.floor((p.y - frameRect.y) / frameRect.h * b.gh), 0, b.gh - 1);
        page.setTip(`space-time patch (${hr},${hc})\n${pxSpan}×${pxSpan} px × ${nGroup} frame${nGroup === 1 ? '' : 's'} × 3 = ${fmtN(pxSpan * pxSpan * nGroup * 3)} raw values\n→ exactly 1 token  ·  drag to move`);
      } else if (curveRect && p.x >= curveRect.x && p.x <= curveRect.x + curveRect.w && p.y >= curveRect.y && p.y <= curveRect.y + curveRect.h) {
        const s = clamp(SECMIN * Math.pow(SECMAX / SECMIN, (p.x - curveRect.x) / curveRect.w), SECMIN, SECMAX);
        const cc = costsAt(b, s);
        page.setTip(`${s.toFixed(1)} s · ${cc.F} frames → ${cc.Tt} latent frames\n${fmtN(cc.N)} tokens\nfull ${fmtN(cc.full)} · factorised ${fmtN(cc.fact)} · windowed ${fmtN(cc.win)}\ndrag to set the clip length`);
      }
    }

    // ===================== readout ===========================================
    let o = `stage ${rev + 1}/4 — ${STAGES[rev].label}.   tier:${r.name}\n`;
    o += `${b.W0}×${b.H0} @ ${b.fps} fps for ${(+st.seconds).toFixed(1)} s = ${c.F} frames · 3-D causal autoencoder ÷${b.sw} space ÷${b.tc} time · latent patch ${b.lp}×${b.lp}.\n`;
    o += `raw pixel values ${fmtN(v.rawValues)} → 2-D patching alone would give ${fmtN(v.raw2d)} tokens → latent ${b.latW}×${b.latH}×${c.Tt} → ${fmtN(c.N)} space-time tokens (${b.gw}×${b.gh}×${c.Tt}) = ${fmtPct(pctOf(c.N, v.raw2d))} of the 2-D-only count.\n`;
    o += `${SHAPE_LABEL[v.shape]}: ${fmtN(v.cost)} attention score entries per head per layer = ${fmtPct(pctOf(v.cost, c.full))} of full 3-D and ${fmtPct(pctOf(v.cost, TEXT_COST))} of a 128 K-token text context (both lower-is-better; 100% = parity). Gives up: ${SHAPE_GIVES_UP[v.shape]}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__videoTimePage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k, lo, hi) => (q.has(k) ? clamp(parseFloat(q.get(k)) || lo, lo, hi) : null);
  const w = num('width', 192, 1920); if (w != null) page.controls.set('width', Math.round(w / 64) * 64, { rebuild: true, silent: true });
  const f = num('fps', 4, 60); if (f != null) page.controls.set('fps', Math.round(f / 2) * 2, { rebuild: true, silent: true });
  const s = num('seconds', SECMIN, SECMAX); if (s != null) page.controls.set('seconds', Math.round(s * 2) / 2, { rebuild: true, silent: true });
  const wn = num('win', 2, 24); if (wn != null) page.controls.set('win', Math.round(wn), { silent: true });
  const l = num('lf', 0, 40); if (l != null) page.controls.set('lf', Math.round(l), { silent: true });
  for (const k of ['sw', 'tc', 'lp']) if (q.has(k)) page.controls.set(k, q.get(k), { rebuild: true, silent: true });
  if (q.has('attn') && SHAPES.includes(q.get('attn'))) page.controls.set('attn', q.get('attn'), { silent: true });
  // ?patch=r,c is the headless stand-in for dragging a patch on the frame plate.
  if (q.has('patch')) { const [pr, pc] = q.get('patch').split(',').map(Number); if (isFinite(pr)) selR = Math.max(0, pr | 0); if (isFinite(pc)) selC = Math.max(0, pc | 0); }
  if (t) t.rebuild();
  // ?hover=x,y fakes the cursor (canvas-space px) so tooltips are capturable.
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  // Deterministic frame for capture: pause before seeking so autoplay cannot
  // advance off the requested step.
  if (q.has('step') || q.has('hover') || q.has('patch')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
