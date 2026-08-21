// mla concept page -- Multi-head Latent Attention: cache ONE small latent per
// token instead of a full K and V for every head.
//
// The sibling page `gqa-mqa` answers the same problem by SHARING key/value
// heads (fewer KV heads -> proportionally smaller cache). This page is the
// next answer: COMPRESS instead of share. The per-token cache stops being
// "K and V for every head" and becomes one low-rank latent plus a small
// decoupled RoPE key -- and the per-head K,V are rebuilt from that latent
// whenever attention needs them.
//
// The page is deliberately two-sided. The cache collapses; the projection
// arithmetic goes UP, because the latent has to be re-expanded (or, after the
// absorption trick, because the query projection now runs at latent width
// instead of head width). Both bars move under one hand when you drag the
// latent dimension -- that trade is the whole idea, and a page that showed
// only the win would be a sales pitch rather than an explanation.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: each step decodes one token and appends its cache entry to
//    both tapes -- a full-height block under MHA, a hairline under MLA.
//    Auto-plays and loops.
//  - DIRECT MANIPULATION: drag the latent-dimension rail (or drag sideways on
//    the MLA footprint square) and watch cached bytes shrink while the
//    absorbed projection matmul grows. Steppers resize heads / head_dim /
//    layers / context.
//  - HOVER any cached block, projection block or bar -> what it is, its shape,
//    and how its size was derived with the current numbers substituted.
//  - URL hooks for every handle plus ?step=N (see the loader at the bottom).
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';

// Bytes per cached element, by KV dtype.
const DT = { fp16: 2, fp8: 1 };

const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const fmtBytes = (b) => (b >= 1073741824 ? (b / 1073741824).toFixed(2) + ' GB'
  : b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB'
  : b >= 1024 ? (b / 1024).toFixed(1) + ' KB' : Math.round(b) + ' B');
const fmtMac = (m) => (m >= 1e9 ? (m / 1e9).toFixed(2) + 'G' : m >= 1e6 ? (m / 1e6).toFixed(1) + 'M' : fmtInt(m));

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// A labelled block. `col` is a live token value read at draw time.
function block(ctx, x, y, w, h, label, col, filled, font) {
  ctx.save();
  roundRect(ctx, x, y, w, h, 5);
  ctx.fillStyle = alphaOf(col, filled ? 0.85 : 0.16); ctx.fill();
  ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.stroke();
  ctx.fillStyle = filled ? inkOn(col) : col;
  ctx.font = font || '11px ui-monospace, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, y + h / 2 + 0.5);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// The math. Everything the page draws is derived here, so a hover tooltip can
// quote the same formula with the current numbers substituted.
// ---------------------------------------------------------------------------
function derive(st) {
  const heads = st.heads | 0, hdim = st.hdim | 0, dc = st.dc | 0, dR = st.dR | 0;
  const d = st.hidden | 0, L = st.layers | 0, B = DT[st.kvdtype] || 2;
  const ctx = (st.ctxk | 0) * 1024;

  // Cached elements per token per layer.
  const mhaElem = 2 * heads * hdim;   // K and V, for every head
  const mlaElem = dc + dR;            // one latent + the decoupled RoPE key

  const mhaTok = mhaElem * L * B;     // bytes / token, all layers
  const mlaTok = mlaElem * L * B;
  const mhaAll = mhaTok * ctx;        // bytes at the full context
  const mlaAll = mlaTok * ctx;

  // Attention-side projection arithmetic, MACs / token / layer.
  //   MHA: Q, K, V and the output projection, each d x (heads*hdim).
  //   MLA: the absorbed query and output projections run at LATENT width
  //        (heads*dc, not heads*hdim), plus the two small down-projections
  //        that produce the latent and the RoPE key.
  const macMHA = 4 * d * heads * hdim;
  const macMLA = 2 * d * heads * dc + d * (dc + dR);

  return {
    heads, hdim, dc, dR, d, L, B, ctx, mhaElem, mlaElem, mhaTok, mlaTok, mhaAll, mlaAll,
    macMHA, macMLA,
    shrink: mhaElem / mlaElem,                 // x smaller cache
    cachePct: 100 * mlaElem / mhaElem,         // MLA cache as % of MHA's
    macPct: 100 * macMLA / macMHA,             // MLA arithmetic as % of MHA's
  };
}

// One transport step = one decoded token appended to both caches.
function buildSteps(st) {
  const v = derive(st), n = st.ntok | 0, out = [];
  for (let t = 0; t < n; t++) {
    out.push({
      t,
      label: `decode token ${t + 1}: append c_KV(${v.dc}) + k_rope(${v.dR}) = ${v.mlaElem} elem/layer`
        + `  ·  MHA would append ${fmtInt(v.mhaElem)}`,
    });
  }
  return out;
}

// Hit-test rects captured each draw(), reused by onPointer() + hover.
let geom = null;
let dragging = null;   // 'rail' | 'square' while a latent-dim handle is held
let geomDcPerPx = 8;   // latent units per px of horizontal drag on the MLA square

const DC_MIN = 32, DC_MAX = 2048, DC_STEP = 32;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const snapDc = (v) => clamp(Math.round(v / DC_STEP) * DC_STEP, DC_MIN, DC_MAX);

function setDc(page, v) { page.controls.set('dc', snapDc(v), { rebuild: true }); }

mount({
  mount: 'body',
  title: 'mla — Multi-head Latent Attention (cache one latent, not K and V per head)',
  blurb: 'Standard attention caches K and V for every head: 2 · heads · head_dim elements per token per layer. MLA down-projects the token to ONE small latent, caches that (plus a small decoupled RoPE key), and re-expands it to per-head K,V at use. The two footprints are drawn to scale, so the collapse is visible rather than merely stated. It is a TRADE: drag the latent dimension and the cached bytes and the absorbed projection matmul move in opposite directions under one hand. KV is the term that grows with context — weights do not — so this is what makes long context affordable.',
  prefer: 'canvas2d',
  aspect: '4 / 3',
  autoplay: true,
  compare: { key: 'dc', a: 2048, b: 128, labelA: 'latent d_c = 2048 (barely compressed)', labelB: 'latent d_c = 128 (aggressive)', rebuild: true },
  challenges: [
    {
      goal: 'Shrink the MLA cache below 2% of the MHA cache.',
      hint: 'drag the latent-dimension rail left — the cache is (d_c + d_R) against 2 · heads · head_dim.',
      check: (api) => ({ solved: (api.probe.cachePct ?? 99) < 2, detail: `MLA cache = ${(api.probe.cachePct ?? 0).toFixed(2)}% of MHA (need < 2%)` }),
    },
    {
      goal: 'Now pay for it honestly: find a latent size where MLA also costs at most 150% of MHA’s projection arithmetic.',
      hint: 'the absorbed projections run at latent width, so the arithmetic scales with d_c — the two goals pull against each other.',
      check: (api) => ({ solved: (api.probe.macPct ?? 999) <= 150, detail: `MLA projection MACs = ${(api.probe.macPct ?? 0).toFixed(0)}% of MHA (lower is better; need ≤ 150%)` }),
    },
  ],
  controls: (c, page) => {
    c.slider('dc', { label: 'latent dim d_c  (drag me)', min: DC_MIN, max: DC_MAX, step: DC_STEP, value: 512, rebuild: true });
    c.stepper('dR', { label: 'decoupled RoPE dim d_R', min: 0, max: 256, value: 64 });
    c.stepper('heads', { label: 'attention heads n_h', min: 4, max: 128, step: 4, value: 128 });
    c.stepper('hdim', { label: 'head_dim d_h', min: 16, max: 256, step: 16, value: 128 });
    c.stepper('hidden', { label: 'hidden dim d', min: 512, max: 16384, step: 512, value: 7168 });
    c.stepper('layers', { label: 'layers', min: 1, max: 128, value: 61 });
    c.slider('ctxk', { label: 'context (K tokens)', min: 1, max: 128, step: 1, value: 8 });
    c.select('kvdtype', { label: 'cache dtype', value: 'fp16', options: [{ value: 'fp16', label: 'fp16 (2 B)' }, { value: 'fp8', label: 'fp8 (1 B)' }] });
    c.stepper('ntok', { label: 'decode steps (tape)', min: 4, max: 24, value: 12 });
    c.transport({ compute: () => buildSteps(page.state), speed: 2.5, loop: true });
  },

  // Direct manipulation: the latent-dimension rail, and a sideways drag on the
  // MLA footprint square itself (its area IS d_c + d_R drawn to scale).
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const onRail = (x, y) => x >= g.rail.x - 12 && x <= g.rail.x + g.rail.w + 12 && y >= g.rail.y - 12 && y <= g.rail.y + 20;
    const onSquare = (x, y) => x >= g.mlaSq.x - 10 && x <= g.mlaSq.x + Math.max(g.mlaSq.w, 26) + 10 && y >= g.mlaSq.y - 10 && y <= g.mlaSq.y + Math.max(g.mlaSq.h, 26) + 10;
    const fromRail = (x) => DC_MIN + ((x - g.rail.x) / g.rail.w) * (DC_MAX - DC_MIN);
    if (ev.type === 'down') {
      dragging = null;
      if (onRail(ev.x, ev.y)) { dragging = 'rail'; setDc(page, fromRail(ev.x)); }
      else if (onSquare(ev.x, ev.y)) { dragging = 'square'; }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragging = null;
    } else if (ev.type === 'move' && dragging && page.pointer.down) {
      if (dragging === 'rail') setDc(page, fromRail(ev.x));
      // A sideways drag on the square scales the latent by the drag distance:
      // area is proportional to d_c, so one px of travel is a fixed number of
      // latent elements at the current to-scale unit.
      else setDc(page, page.state.dc + ev.dx * g.dcPerPx);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const v = derive(st);
    r.clear(T.n0);
    const s = page.step();
    const tok = (s ? s.t : (st.ntok | 0) - 1) + 1;   // tokens decoded so far
    page.probe = { cachePct: v.cachePct, macPct: v.macPct, dc: v.dc, tok };

    const W = page.W, H = page.H, pad = 20;
    const hits = [];   // {x,y,w,h,tip} -- hover-to-inspect targets
    const hit = (x, y, w, h, tip) => hits.push({ x, y, w, h, tip });
    const lab = (t, x, y, col, font, align) => r.label(t, x, y, { color: col || T.n11, font: font || '11px ui-monospace, monospace', align: align || 'left' });

    // ---- header -----------------------------------------------------------
    lab(`n_h = ${v.heads}   d_h = ${v.hdim}   d_c = ${v.dc}   d_R = ${v.dR}   d = ${v.d}   layers = ${v.L}   cache dtype = ${st.kvdtype} (${v.B} B)`,
      pad, 26, T.n14, '12px ui-monospace, monospace');
    lab(`cached per token per layer:  MHA ${fmtInt(v.mhaElem)} elem  →  MLA ${fmtInt(v.mlaElem)} elem   (${v.cachePct.toFixed(2)}% of MHA, ${v.shrink.toFixed(1)}× smaller)`,
      pad, 44, T.accent, '12px ui-monospace, monospace');

    // ---- panel 1: per-token footprint, DRAWN TO SCALE ---------------------
    // Area is proportional to the element count, with ONE shared unit, so the
    // collapse is a picture and not a number.
    const p1x = pad, p1y = 60, p1w = Math.max(240, Math.min(400, W * 0.42)), p1h = 224;
    lab('per-token cached footprint (one layer, to scale)', p1x, p1y + 10, T.n12, '11.5px ui-monospace, monospace');
    const sqMax = Math.min(p1h - 58, p1w * 0.44);
    const unit = (sqMax * sqMax) / Math.max(1, v.mhaElem);            // px^2 per element
    const mhaSide = Math.sqrt(v.mhaElem * unit);
    const mlaSide = Math.sqrt(v.mlaElem * unit);
    const baseY = p1y + 34 + sqMax;                                   // shared baseline
    const mhaX = p1x + 6, mlaX = p1x + 6 + sqMax + 72;

    // MHA square: top half K, bottom half V, with per-head banding when legible.
    const mhaY = baseY - mhaSide;
    ctx.save();
    ctx.fillStyle = alphaOf(T.accent, 0.30); ctx.fillRect(mhaX, mhaY, mhaSide, mhaSide / 2);
    ctx.fillStyle = alphaOf(T.warn, 0.30); ctx.fillRect(mhaX, mhaY + mhaSide / 2, mhaSide, mhaSide / 2);
    const bandH = (mhaSide / 2) / v.heads;
    if (bandH >= 2.5) {
      ctx.strokeStyle = rgbaToken('n14', 0.10); ctx.lineWidth = 0.5;
      for (let i = 1; i < v.heads; i++) {
        const yy = mhaY + i * bandH; ctx.beginPath(); ctx.moveTo(mhaX, yy); ctx.lineTo(mhaX + mhaSide, yy);
        ctx.moveTo(mhaX, yy + mhaSide / 2); ctx.lineTo(mhaX + mhaSide, yy + mhaSide / 2); ctx.stroke();
      }
    }
    ctx.strokeStyle = T.n8; ctx.lineWidth = 1; ctx.strokeRect(mhaX, mhaY, mhaSide, mhaSide);
    ctx.restore();
    lab('MHA', mhaX, mhaY - 6, T.n13, '11px ui-monospace, monospace');
    lab(`${fmtInt(v.mhaElem)} elem`, mhaX, baseY + 14, T.n12, '10px ui-monospace, monospace');
    lab('K (all heads) / V (all heads)', mhaX, baseY + 27, T.n10, '9px ui-monospace, monospace');
    hit(mhaX, mhaY, mhaSide, mhaSide / 2,
      `MHA cached K — one d_h-vector per head, every token\nshape (n_h, d_h) = (${v.heads}, ${v.hdim}) = ${fmtInt(v.heads * v.hdim)} elem\ntogether with V: 2 · ${v.heads} · ${v.hdim} = ${fmtInt(v.mhaElem)} elem/token/layer`);
    hit(mhaX, mhaY + mhaSide / 2, mhaSide, mhaSide / 2,
      `MHA cached V — one d_h-vector per head, every token\nshape (n_h, d_h) = (${v.heads}, ${v.hdim}) = ${fmtInt(v.heads * v.hdim)} elem\nbytes/token = 2 · ${v.heads} · ${v.hdim} · ${v.L} layers · ${v.B} B = ${fmtBytes(v.mhaTok)}`);

    // MLA square: latent (accent) + RoPE key (violet), split BY AREA.
    const mlaY = baseY - mlaSide;
    const latFrac = v.mlaElem ? v.dc / v.mlaElem : 1;
    ctx.save();
    ctx.fillStyle = alphaOf(T.accent, 0.85); ctx.fillRect(mlaX, mlaY + mlaSide * (1 - latFrac), mlaSide, mlaSide * latFrac);
    ctx.fillStyle = alphaOf(T.violet, 0.85); ctx.fillRect(mlaX, mlaY, mlaSide, mlaSide * (1 - latFrac));
    ctx.strokeStyle = T.n8; ctx.lineWidth = 1; ctx.strokeRect(mlaX, mlaY, mlaSide, mlaSide);
    ctx.restore();
    lab('MLA', mlaX, mlaY - 6, T.n13, '11px ui-monospace, monospace');
    lab(`${fmtInt(v.mlaElem)} elem`, mlaX, baseY + 14, T.n12, '10px ui-monospace, monospace');
    lab(`c_KV ${v.dc} + k_rope ${v.dR}`, mlaX, baseY + 27, T.n10, '9px ui-monospace, monospace');
    lab('↔ drag', mlaX, baseY + 39, T.accent, '9px ui-monospace, monospace');
    hit(mlaX, mlaY + mlaSide * (1 - latFrac), mlaSide, mlaSide * latFrac,
      `c_KV — the ONLY per-head-free thing cached\nc_KV = x W_DKV, shape (d_c) = (${v.dc})\nshared by all ${v.heads} heads; per-head K,V are rebuilt from it\n↔ drag sideways to resize the latent`);
    hit(mlaX, mlaY, mlaSide, mlaSide * (1 - latFrac),
      `k_rope — decoupled RoPE key, cached alongside the latent\nk_rope = RoPE(x W_KR), shape (d_R) = (${v.dR})\nkept separate because RoPE is position-dependent, so it\ncannot be folded into a position-independent absorbed weight`);
    // px of horizontal drag per unit of d_c (area/unit -> side -> drag feel).
    geomDcPerPx = Math.max(1, (DC_MAX - DC_MIN) / 260);

    // Ratio caption between the squares.
    lab(`${v.shrink.toFixed(1)}×`, mlaX - 10, baseY - 6, T.accent, 'bold 13px ui-monospace, monospace', 'right');
    lab('smaller', mlaX - 10, baseY + 8, T.n11, '9px ui-monospace, monospace', 'right');

    // ---- panel 2: the flow -------------------------------------------------
    const p2x = p1x + p1w + 24, p2w = W - p2x - pad;
    lab('flow: token → latent (cached) → per-head K,V', p2x, p1y + 10, T.n12, '11.5px ui-monospace, monospace');
    const rowH = 23, gap = 7, cw = Math.min(272, p2w - 168);
    let fy = p1y + 26;
    const chain = [
      { t: `token x — d = ${v.d}`, col: T.n11, fill: false,
        tip: `the layer input for the token being decoded\nshape (d) = (${v.d})` },
      { t: `W_DKV  (d × d_c) = ${v.d}×${v.dc}`, col: T.teal, fill: false, op: true,
        tip: `down-projection to the latent\nweights (d × d_c) = (${v.d} × ${v.dc})\ncost ${fmtMac(v.d * v.dc)} MAC/token/layer — weights do NOT grow with context` },
      { t: `c_KV — d_c = ${v.dc}   ▣ CACHED`, col: T.accent, fill: true,
        tip: `the cache entry: one latent per token per layer\n${v.dc} elem · ${v.L} layers · ${v.B} B = ${fmtBytes(v.dc * v.L * v.B)}/token\nthis is what grows with context` },
      { t: `W_UK / W_UV  (d_c × n_h·d_h)`, col: T.teal, fill: false, op: true,
        tip: `re-expansion: rebuild per-head K and V from the latent\nweights (d_c × n_h·d_h) = (${v.dc} × ${fmtInt(v.heads * v.hdim)}), twice\nat decode this is usually ABSORBED into the query and\noutput projections, which then run at latent width d_c` },
      { t: `K,V for ${v.heads} heads — ${fmtInt(v.mhaElem)} elem`, col: T.warn, fill: false,
        tip: `the per-head K,V attention actually consumes\n2 · ${v.heads} · ${v.hdim} = ${fmtInt(v.mhaElem)} elem\nmaterialized per step and thrown away — never cached` },
      { t: 'attention → output', col: T.n11, fill: false,
        tip: 'softmax(QKᵀ/√d) V, then the output projection\n(absorbed with W_UV when the latent path is used)' },
    ];
    chain.forEach((row, i) => {
      block(ctx, p2x, fy, cw, rowH, row.t, row.col, !!row.fill);
      hit(p2x, fy, cw, rowH, row.tip);
      if (i < chain.length - 1) {
        ctx.save(); ctx.strokeStyle = T.n8; ctx.fillStyle = T.n8; ctx.lineWidth = 1.2;
        const ax = p2x + cw / 2, ay = fy + rowH;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax, ay + gap); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ax - 3.5, ay + gap - 3); ctx.lineTo(ax + 3.5, ay + gap - 3); ctx.lineTo(ax, ay + gap + 1); ctx.closePath(); ctx.fill();
        ctx.restore();
      }
      fy += rowH + gap;
    });
    // The decoupled RoPE branch, beside the latent row.
    const ropeY = p1y + 26 + 2 * (rowH + gap), ropeX = p2x + cw + 14, ropeW = Math.min(150, W - ropeX - pad);
    if (ropeW > 60) {
      block(ctx, ropeX, ropeY, ropeW, rowH, `k_rope ${v.dR} ▣ CACHED`, T.violet, true);
      hit(ropeX, ropeY, ropeW, rowH,
        `k_rope = RoPE(x W_KR), shape (d_R) = (${v.dR})\ncached beside the latent, ${v.dR} elem/token/layer\nseparate because RoPE is position-dependent and so\ncannot be absorbed into a fixed weight`);
      lab('decoupled RoPE key', ropeX, ropeY - 5, T.n10, '9px ui-monospace, monospace');
      lab(`cached = ${fmtInt(v.mlaElem)}/tok/layer`, ropeX, ropeY + rowH + 16, T.n11, '10px ui-monospace, monospace');
    }

    // ---- panel 3: decode tape ---------------------------------------------
    const p3y = Math.max(p1y + p1h + 18, fy + 6), tapeH = 22;
    lab(`decode tape — each step appends one token to both caches (${tok} / ${st.ntok} decoded)`, pad, p3y + 9, T.n12, '11.5px ui-monospace, monospace');
    const tx = pad + 96, tw = W - tx - pad - 130, n = st.ntok | 0, slot = tw / n;
    const rowMHA = p3y + 18, rowMLA = rowMHA + tapeH + 12;
    lab('MHA', pad, rowMHA + tapeH / 2 + 4, T.warn, '11px ui-monospace, monospace');
    lab('MLA', pad, rowMLA + tapeH / 2 + 4, T.accent, '11px ui-monospace, monospace');
    const mlaH = Math.max(1.5, tapeH * (v.mlaElem / v.mhaElem));
    ctx.save();
    for (let i = 0; i < n; i++) {
      const x = tx + i * slot, wdt = Math.max(2, slot - 2);
      ctx.strokeStyle = rgbaToken('n14', 0.10); ctx.lineWidth = 1;
      ctx.strokeRect(x, rowMHA, wdt, tapeH);
      ctx.strokeRect(x, rowMLA, wdt, tapeH);
      if (i < tok) {
        ctx.fillStyle = alphaOf(T.warn, i === tok - 1 ? 0.85 : 0.45); ctx.fillRect(x, rowMHA, wdt, tapeH);
        ctx.fillStyle = alphaOf(T.accent, i === tok - 1 ? 0.95 : 0.7); ctx.fillRect(x, rowMLA + tapeH - mlaH, wdt, mlaH);
      }
    }
    ctx.restore();
    hit(tx, rowMHA, tw, tapeH, `MHA tape — one full K,V set per token\nafter ${tok} token${tok > 1 ? 's' : ''}: ${fmtInt(tok * v.mhaElem * v.L)} elem = ${fmtBytes(tok * v.mhaTok)}`);
    hit(tx, rowMLA, tw, tapeH, `MLA tape — one latent (+ RoPE key) per token\nbar height is ${v.cachePct.toFixed(2)}% of the row above, drawn to scale\nafter ${tok} token${tok > 1 ? 's' : ''}: ${fmtInt(tok * v.mlaElem * v.L)} elem = ${fmtBytes(tok * v.mlaTok)}`);
    lab(fmtBytes(tok * v.mhaTok), tx + tw + 8, rowMHA + tapeH / 2 + 4, T.n12, '10px ui-monospace, monospace');
    lab(fmtBytes(tok * v.mlaTok), tx + tw + 8, rowMLA + tapeH / 2 + 4, T.n12, '10px ui-monospace, monospace');

    // ---- panel 4: the trade ------------------------------------------------
    const p4y = rowMLA + tapeH + 22, p4w = Math.max(250, Math.min(430, W * 0.46));
    lab('the trade — one hand, two directions', pad, p4y + 10, T.n12, '11.5px ui-monospace, monospace');
    // latent-dimension rail
    const railX = pad + 92, railY = p4y + 26, railW = p4w - 104;
    ctx.save();
    ctx.fillStyle = alphaOf(T.n8, 0.5); ctx.fillRect(railX, railY + 3, railW, 4);
    const hx = railX + railW * ((v.dc - DC_MIN) / (DC_MAX - DC_MIN));
    ctx.fillStyle = dragging === 'rail' ? T.accent : T.n12;
    roundRect(ctx, hx - 5, railY - 5, 10, 20, 3); ctx.fill();
    ctx.restore();
    lab('latent d_c', pad, railY + 12, T.n11, '11px ui-monospace, monospace');
    lab(`${v.dc}`, railX + railW + 8, railY + 12, T.accent, '11px ui-monospace, monospace');
    lab('↔ drag: cache shrinks, projection matmul grows', railX, railY + 30, T.accent, '9.5px ui-monospace, monospace');
    hit(railX - 12, railY - 12, railW + 24, 30,
      `latent dimension d_c = ${v.dc}\ncached elem/token/layer = d_c + d_R = ${v.dc} + ${v.dR} = ${fmtInt(v.mlaElem)}\nabsorbed projection MACs = 2 · d · n_h · d_c + d · (d_c + d_R)\n= 2 · ${v.d} · ${v.heads} · ${v.dc} + ${v.d} · ${fmtInt(v.mlaElem)} = ${fmtMac(v.macMLA)}\nd_c is a RANK BUDGET, not a free knob: it is how much of K and V the\nlatent can still represent, so a very small d_c buys cheap arithmetic\nby giving up capacity. The published production configuration is 512.`);

    // two opposed bars
    const bx = pad + 92, bw = p4w - 104;
    const bar = (y, label2, fracM, fracL, note) => {
      ctx.save();
      ctx.strokeStyle = T.n6; ctx.lineWidth = 1;
      ctx.strokeRect(bx, y, bw, 11); ctx.strokeRect(bx, y + 15, bw, 11);
      ctx.fillStyle = alphaOf(T.warn, 0.55); ctx.fillRect(bx, y, bw * fracM, 11);
      ctx.fillStyle = alphaOf(T.accent, 0.7); ctx.fillRect(bx, y + 15, bw * fracL, 11);
      ctx.restore();
      lab(label2, pad, y + 9, T.n11, '10px ui-monospace, monospace');
      lab(note, bx, y + 39, T.n11, '9px ui-monospace, monospace');
      lab('MHA', bx + bw + 6, y + 9, T.warn, '9px ui-monospace, monospace');
      lab('MLA', bx + bw + 6, y + 24, T.accent, '9px ui-monospace, monospace');
    };
    const cacheY = railY + 42;
    bar(cacheY, 'cached bytes', 1, v.mlaElem / v.mhaElem,
      `${fmtBytes(v.mlaTok)} vs ${fmtBytes(v.mhaTok)} /token — ${v.cachePct.toFixed(2)}% of MHA (lower=better)`);
    hit(bx, cacheY, bw, 26,
      `cached bytes / token (all ${v.L} layers)\nMHA: 2 · ${v.heads} · ${v.hdim} · ${v.L} · ${v.B} B = ${fmtBytes(v.mhaTok)}\nMLA: (${v.dc} + ${v.dR}) · ${v.L} · ${v.B} B = ${fmtBytes(v.mlaTok)}`);
    const macY = cacheY + 52;
    const macMax = Math.max(v.macMHA, v.macMLA);
    bar(macY, 'projection MACs', v.macMHA / macMax, v.macMLA / macMax,
      `${fmtMac(v.macMLA)} vs ${fmtMac(v.macMHA)} MAC/tok/layer — ${v.macPct.toFixed(0)}% of MHA (lower=better)`);
    lab('d_c is a rank budget — lowering it also gives up latent capacity', bx, macY + 52, T.n10, '9px ui-monospace, monospace');
    hit(bx, macY, bw, 26,
      `attention projection MACs / token / layer\nMHA: 4 · d · n_h · d_h = 4 · ${v.d} · ${v.heads} · ${v.hdim} = ${fmtMac(v.macMHA)}\n   (Q, K, V and the output projection)\nMLA: 2 · d · n_h · d_c + d · (d_c + d_R) = ${fmtMac(v.macMLA)}\n   (absorbed query + output run at LATENT width, plus the\n    two small down-projections) — this is the price of the cache win`);

    // ---- panel 5: growth with context -------------------------------------
    const p5x = pad + p4w + 28, p5w = W - p5x - pad, p5y = p4y;
    lab('total KV cache vs context length — the gap widens', p5x, p5y + 10, T.n12, '11.5px ui-monospace, monospace');
    const cx0 = p5x + 44, cy0 = p5y + 26, cw2 = Math.max(80, p5w - 60), ch2 = Math.max(60, H - cy0 - 40);
    const maxCtx = 131072;
    const yMax = v.mhaTok * maxCtx;
    ctx.save();
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy0 + ch2); ctx.lineTo(cx0 + cw2, cy0 + ch2); ctx.stroke();
    const lineTo = (bytesPerTok, col, wid) => {
      ctx.strokeStyle = col; ctx.lineWidth = wid;
      ctx.beginPath(); ctx.moveTo(cx0, cy0 + ch2);
      ctx.lineTo(cx0 + cw2, cy0 + ch2 - ch2 * (bytesPerTok * maxCtx) / Math.max(1, yMax));
      ctx.stroke();
    };
    lineTo(v.mhaTok, T.warn, 2);
    lineTo(v.mlaTok, T.accent, 2.5);
    // marker at the selected context
    const mxp = cx0 + cw2 * (v.ctx / maxCtx);
    ctx.strokeStyle = rgbaToken('n14', 0.35); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(mxp, cy0); ctx.lineTo(mxp, cy0 + ch2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    lab(fmtBytes(yMax), cx0 + 4, cy0 + 9, T.n10, '9px ui-monospace, monospace');
    lab('0', cx0, cy0 + ch2 + 14, T.n10, '9px ui-monospace, monospace');
    lab('128K tokens', cx0 + cw2, cy0 + ch2 + 14, T.n10, '9px ui-monospace, monospace', 'right');
    lab(`at ${fmtInt(v.ctx)} tokens:  MHA ${fmtBytes(v.mhaAll)}   MLA ${fmtBytes(v.mlaAll)}`, p5x, cy0 + ch2 + 28, T.n12, '10px ui-monospace, monospace');
    hit(cx0, cy0, cw2, ch2,
      `total KV cache = elem/token/layer · layers · bytes · context\nMHA: ${fmtInt(v.mhaElem)} · ${v.L} · ${v.B} · ${fmtInt(v.ctx)} = ${fmtBytes(v.mhaAll)}\nMLA: ${fmtInt(v.mlaElem)} · ${v.L} · ${v.B} · ${fmtInt(v.ctx)} = ${fmtBytes(v.mlaAll)}\nboth are LINEAR in context; only the slope differs, so the\nabsolute gap keeps growing. The weights do not grow at all.`);

    // ---- geometry for the pointer layer ------------------------------------
    geom = {
      rail: { x: railX, y: railY, w: railW },
      mlaSq: { x: mlaX, y: mlaY, w: mlaSide, h: mlaSide },
      dcPerPx: geomDcPerPx,
      hits,
    };

    // ---- hover-to-inspect ---------------------------------------------------
    if (page.pointer.over && !dragging) {
      const p = page.pointer;
      for (let i = hits.length - 1; i >= 0; i--) {
        const q = hits[i];
        if (p.x >= q.x && p.x <= q.x + q.w && p.y >= q.y && p.y <= q.y + q.h) { page.setTip(q.tip); break; }
      }
    }

    // ---- readout -------------------------------------------------------------
    let o = `cached elements/token/layer:  MHA 2·n_h·d_h = 2·${v.heads}·${v.hdim} = ${fmtInt(v.mhaElem)}   →   MLA d_c+d_R = ${v.dc}+${v.dR} = ${fmtInt(v.mlaElem)}   `
      + `(${v.cachePct.toFixed(2)}% of MHA, ${v.shrink.toFixed(1)}× smaller)    tier:${r.name}\n`;
    o += s ? `${s.label}\n` : '(plays on load; scrub or step to append tokens)\n';
    o += `at ${fmtInt(v.ctx)} tokens · ${v.L} layers · ${st.kvdtype}: MHA ${fmtBytes(v.mhaAll)} vs MLA ${fmtBytes(v.mlaAll)}. `
      + `The price: projection MACs/token/layer go ${fmtMac(v.macMHA)} → ${fmtMac(v.macMLA)} = ${v.macPct.toFixed(0)}% of MHA (lower is better; 100% = parity), `
      + 'because the absorbed query and output projections run at latent width d_c instead of head width d_h. '
      + 'KV is the only term that grows with context — weights do not — which is why shrinking it is what makes long context affordable.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__mlaPage = page;
  const q = new URLSearchParams(location.search);
  // Every control handle has a URL hook, so any state this page can be dragged
  // or stepped into is reproducible headlessly (--screenshot has no pointer).
  const NUM = ['dc', 'dR', 'heads', 'hdim', 'hidden', 'layers', 'ctxk', 'ntok'];
  for (const k of NUM) if (q.has(k)) page.controls.set(k, +q.get(k), { rebuild: true });
  if (q.has('kvdtype')) page.controls.set('kvdtype', q.get('kvdtype'));
  const t = page.controls._transport;
  // ?hover=x,y fakes the cursor (canvas-space px) so the tooltip path is
  // verifiable without a pointer.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause for any of these hooks so autoplay
  // does not advance off the requested step before the snapshot.
  if (q.has('step') || q.has('hover') || q.has('dc')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
