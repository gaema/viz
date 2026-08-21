// vlm-connector concept page -- how a vision encoder actually reaches a language
// model, and why IMAGE RESOLUTION is a TOKEN-BUDGET decision rather than a
// quality dial.
//
// A vision encoder turns an image into a grid of patch embeddings. Those cannot
// enter the language model as they are: wrong WIDTH (encoder dim, not the LLM's
// embedding dim) and wrong NUMBER (a grid that grows with the square of the
// resolution). A CONNECTOR fixes both. Three shipped shapes, three cost curves:
//
//   projector  -- a linear / MLP projection per patch (LLaVA). Every patch
//                 becomes one token, so the token count IS the patch count and
//                 scales with resolution squared. The most expensive shape.
//   merge      -- pixel-shuffle / patch-merge (InternVL, Qwen-VL): a k x k block
//                 of neighbouring patches is concatenated along the CHANNEL axis
//                 and projected once, dividing the token count by k^2 and giving
//                 up some spatial detail.
//   resampler  -- perceiver-style cross-attention (Flamingo, and Qwen-VL's early
//                 form): a FIXED number of learned queries attend to the patches,
//                 so the token count is constant at any resolution -- at the cost
//                 of a fixed bottleneck that cannot carry more detail no matter
//                 how far the resolution is pushed.
//
// The whole trade is arithmetic and is computed live here, never tabulated:
//     tokens = (H/patch) * (W/patch) / merge^2
// and the language model's attention work grows with the SQUARE of the total
// sequence length, so doubling the resolution quadruples the tokens and
// multiplies the attention work by about sixteen.
//
// Companion page: `multimodal-inject` shows THAT media tokens get spliced into
// the text sequence. This page is about HOW MANY, FROM WHERE, and AT WHAT COST.
//
// Papers: LLaVA https://arxiv.org/abs/2304.08485 ,
//         Flamingo https://arxiv.org/abs/2204.14198 ,
//         Qwen2-VL https://arxiv.org/abs/2409.12191 .
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, rgbaToken, signedColor, mixColor, inkOn } from '../framework/theme.js';

const S = 128;             // raster side of the stand-in image (px)
const D_ENC = 1024;        // vision-encoder width (illustrative, ViT-L-ish)
const D_LLM = 4096;        // language-model embedding width (illustrative)
const CAP = 48;            // max cells drawn per side; the COUNT is always exact
const RES_MIN = 224, RES_MAX = 1792, RES_STEP = 32;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const num = (n) => n.toLocaleString('en-US');
const sci = (n) => (n >= 1e6 ? n.toExponential(2) : num(Math.round(n)));

let im = null, imSeed = null;
let imgRect = null, gridRect = null, outRect = null, railRect = null, barRect = null;
let drag = null;

// ---- the stand-in image ----------------------------------------------------
// Generated in-page (no committed media, no download): soft colour blobs, so a
// patch has something to be an embedding OF and neighbouring patches genuinely
// correlate -- which is the whole reason patch-merge loses so little.
function buildImage(seed) {
  const cv = document.createElement('canvas'); cv.width = S; cv.height = S;
  const c = cv.getContext('2d'), img = c.createImageData(S, S), d = img.data;
  const rnd = seededRandn(seed, 20, { std: 1 });
  const blobs = [];
  for (let b = 0; b < 5; b++) {
    blobs.push({
      cx: (rnd[b * 3] * 0.45 + 0.5) * S, cy: (rnd[b * 3 + 1] * 0.45 + 0.5) * S,
      r: 16 + 16 * Math.abs(rnd[b * 3 + 2]),
      col: [[228, 84, 74], [64, 122, 226], [70, 194, 122], [238, 186, 62], [150, 96, 216]][b],
    });
  }
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let rr = 236, gg = 238, bb = 240;
    for (const bl of blobs) {
      const w = Math.exp(-((x - bl.cx) ** 2 + (y - bl.cy) ** 2) / (2 * bl.r * bl.r));
      rr = rr * (1 - w) + bl.col[0] * w; gg = gg * (1 - w) + bl.col[1] * w; bb = bb * (1 - w) + bl.col[2] * w;
    }
    const i = (y * S + x) * 4; d[i] = rr; d[i + 1] = gg; d[i + 2] = bb; d[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  return { cv, data: d };
}
// A signed stand-in "embedding channel" for the patch covering (u,v) in [0,1).
// One channel, read smoothly across the grid, so the encoder panel visibly
// still IS the picture -- which is what makes the merge step's loss legible.
function chan(u, v) {
  const x = clamp(Math.floor(u * S), 0, S - 1), y = clamp(Math.floor(v * S), 0, S - 1);
  const i = (y * S + x) * 4;
  const lum = (im.data[i] * 0.55 + im.data[i + 1] * 0.33 + im.data[i + 2] * 0.12) / 255;
  const tint = (im.data[i] - im.data[i + 2]) / 255;
  return clamp(tint * 1.5 + (lum - 0.68) * 1.4, -1, 1);
}

// ---- the arithmetic (computed, never tabulated) ----------------------------
function geom(st) {
  const patch = +st.patch, res = +st.res;
  const gridN = Math.max(1, Math.floor(res / patch));
  const N = gridN * gridN;
  const k = clamp(Math.round(+st.merge), 1, 8);
  const gm = Math.max(1, Math.floor(gridN / k));
  const tokens = st.conn === 'projector' ? N
    : st.conn === 'merge' ? gm * gm
      : Math.round(+st.queries);
  const outN = st.conn === 'resampler' ? 0 : (st.conn === 'merge' ? gm : gridN);
  return { patch, res, gridN, N, k, gm, tokens, outN };
}
// tokens/image for an arbitrary resolution -- the cost curve calls this per pixel
// column, so the three curves are the same function the readout uses.
function tokensAt(res, patch, conn, k, q) {
  const g = Math.max(1, Math.floor(res / patch));
  if (conn === 'projector') return g * g;
  if (conn === 'merge') { const gm = Math.max(1, Math.floor(g / k)); return gm * gm; }
  return q;
}

const CONN_LABEL = {
  projector: 'linear / MLP projector',
  merge: 'pixel-shuffle / patch-merge',
  resampler: 'resampler (cross-attention)',
};
const CONN_PAPER = {
  projector: 'LLaVA  arxiv.org/abs/2304.08485',
  merge: 'InternVL · Qwen2-VL  arxiv.org/abs/2409.12191',
  resampler: 'Flamingo  arxiv.org/abs/2204.14198',
};

const STEPS = [
  { stage: 0, label: 'patchify — the image is cut into a grid of patch×patch squares. The grid side is res/patch, so the patch COUNT is quadratic in the resolution.' },
  { stage: 1, label: 'encode — the vision transformer turns each patch into one D_enc-wide embedding. Still a grid, still the wrong width and the wrong number for the language model.' },
  { stage: 2, label: 'connect — the connector fixes both. Width is a projection either way; the interesting choice is how many tokens come OUT.' },
  { stage: 3, label: 'splice — those tokens sit in the language model’s context beside the text, and attention over the whole sequence costs seq².' },
];

mount({
  mount: 'body',
  title: 'vlm-connector — how the vision encoder reaches the language model',
  blurb: 'A vision encoder turns an image into a grid of patch embeddings, and those cannot enter a language model as they are: the wrong width (encoder dim, not the LLM embedding dim) and the wrong number (a grid that grows with the square of the resolution). A CONNECTOR fixes both — and the shape you pick decides how many tokens the picture costs. A linear/MLP projector (LLaVA) emits one token per patch, so the count scales with resolution squared. Pixel-shuffle / patch-merge (InternVL, Qwen-VL) concatenates a k×k block of neighbouring patches along the CHANNEL axis and projects it once, dividing the count by k² and giving up some spatial detail. A perceiver-style resampler (Flamingo, and Qwen-VL in its early form) lets a FIXED number of learned queries attend to the patches, so the count is constant at any resolution — at the cost of a bottleneck that cannot represent more detail however far you push the pixels. The whole trade is arithmetic: tokens = (H/patch)·(W/patch)/merge², and the language model’s attention grows with the SQUARE of the total sequence length. Drag the image’s corner handle to resize it, drag the rail inside the connector to change the merge factor (or the query count), and watch the token count and the context bar move under your hand.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  animate: true,
  autoplay: true,
  controls: (c) => {
    c.select('conn', { label: 'connector', options: ['projector', 'merge', 'resampler'], value: 'projector' });
    c.slider('res', { label: 'image resolution (px)', min: RES_MIN, max: RES_MAX, step: RES_STEP, value: 448 });
    c.select('patch', { label: 'patch size', options: ['14', '16', '28'], value: '14' });
    c.slider('merge', { label: 'merge factor k', min: 1, max: 4, step: 1, value: 2 });
    c.slider('queries', { label: 'resampler queries', min: 16, max: 256, step: 16, value: 64 });
    c.slider('imgs', { label: 'images in the chat', min: 1, max: 8, step: 1, value: 2 });
    c.slider('text', { label: 'text tokens', min: 16, max: 1024, step: 16, value: 256 });
    c.select('ctx', { label: 'context budget', options: ['4096', '8192', '32768', '131072'], value: '8192' });
    c.slider('seed', { label: 'image seed', min: 0, max: 99, step: 1, value: 6 });
    c.transport({ compute: () => STEPS, speed: 0.85, loop: true });
  },

  onPointer: (page, ev) => {
    const st = page.state;
    if (ev.type === 'up' || ev.type === 'leave') { drag = null; return; }
    if (ev.type === 'down') {
      drag = null;
      if (imgRect) {
        const hx = imgRect.x + imgRect.w - 11, hy = imgRect.y + imgRect.h - 11;
        if (ev.x >= hx - 3 && ev.x <= hx + 15 && ev.y >= hy - 3 && ev.y <= hy + 15) drag = 'res';
      }
      if (!drag && railRect && st.conn !== 'projector'
        && ev.x >= railRect.x - 8 && ev.x <= railRect.x + railRect.w + 8
        && ev.y >= railRect.y - 10 && ev.y <= railRect.y + 14) drag = 'rail';
    }
    if (!drag || !page.pointer.down) return;
    if (drag === 'res') {
      const step = (ev.dx + ev.dy) * 6;
      const v = clamp(Math.round((+st.res + step) / RES_STEP) * RES_STEP, RES_MIN, RES_MAX);
      if (v !== +st.res) page.controls.set('res', v);
    } else if (drag === 'rail' && railRect) {
      const f = clamp((ev.x - railRect.x) / railRect.w, 0, 1);
      if (st.conn === 'merge') {
        const v = clamp(Math.round(1 + f * 3), 1, 4);
        if (v !== Math.round(+st.merge)) page.controls.set('merge', v);
      } else {
        const v = clamp(Math.round((16 + f * 240) / 16) * 16, 16, 256);
        if (v !== Math.round(+st.queries)) page.controls.set('queries', v);
      }
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    if (imSeed !== (st.seed | 0)) { im = buildImage(st.seed | 0); imSeed = st.seed | 0; }
    const g = geom(st);
    const rec = page.step();
    const stage = rec ? rec.stage : 0;
    const ctxCap = +st.ctx, nImg = Math.round(+st.imgs), nText = Math.round(+st.text);
    const imgTok = g.tokens * nImg, seq = imgTok + nText;
    const pctCtx = (imgTok / ctxCap) * 100, pctSeq = (seq / ctxCap) * 100;
    const halfTok = tokensAt(Math.max(RES_MIN, +st.res / 2), g.patch, st.conn, g.k, Math.round(+st.queries));
    const seqHalf = halfTok * nImg + nText;
    const attnNow = seq * seq, attnFull = ctxCap * ctxCap;

    r.clear(T.n0);
    const mono = (px) => `${px}px ui-monospace, monospace`;
    const dim = (t2) => r.label(t2.text, t2.x, t2.y, { color: T.n10, font: mono(9) });
    const flow = (x0, y0, x1, y1, col, phase) => {
      ctx.save(); ctx.strokeStyle = alphaOf(col, 0.55); ctx.lineWidth = 1.1;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const f = (((page.t || 0) * 0.55 + (phase || 0)) % 1);
      ctx.fillStyle = col; ctx.beginPath();
      ctx.arc(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, 2.4, 0, 7); ctx.fill();
      ctx.restore();
    };

    // ===================== stage 0 — patchify ===============================
    const ix = 14, iy = 62, isz = 108;
    imgRect = { x: ix, y: iy, w: isz, h: isz };
    r.label('1 · patchify', ix, iy - 26, { color: T.n14, font: mono(11) });
    r.label(`${+st.res}×${+st.res} px`, ix, iy - 12, { color: T.warn, font: mono(10) });
    ctx.drawImage(im.cv, ix, iy, isz, isz);
    ctx.save();
    const cell = isz / g.gridN;
    if (cell >= 3) {
      ctx.strokeStyle = rgbaToken('n14', 0.28); ctx.lineWidth = 0.5;
      for (let i = 0; i <= g.gridN; i++) {
        ctx.beginPath(); ctx.moveTo(ix + i * cell, iy); ctx.lineTo(ix + i * cell, iy + isz); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ix, iy + i * cell); ctx.lineTo(ix + isz, iy + i * cell); ctx.stroke();
      }
    }
    // sweeping patch cursor -- the raster order the encoder walks
    const swp = Math.floor((page.t || 0) * 6) % g.N, swr = (swp / g.gridN) | 0, swc = swp % g.gridN;
    ctx.strokeStyle = T.gold; ctx.lineWidth = 1.6;
    ctx.strokeRect(ix + swc * cell, iy + swr * cell, Math.max(3, cell), Math.max(3, cell));
    ctx.strokeStyle = T.n7; ctx.lineWidth = 1; ctx.strokeRect(ix - 0.5, iy - 0.5, isz + 1, isz + 1);
    // resize handle
    const hx = ix + isz - 11, hy = iy + isz - 11;
    ctx.fillStyle = drag === 'res' ? T.warn : alphaOf(T.warn, 0.75);
    ctx.beginPath(); ctx.moveTo(hx + 12, hy); ctx.lineTo(hx + 12, hy + 12); ctx.lineTo(hx, hy + 12); ctx.closePath(); ctx.fill();
    ctx.restore();
    r.label(`${g.gridN}² = ${num(g.N)} patches`, ix, iy + isz + 14, { color: T.n12, font: mono(10) });
    dim({ text: `${g.patch}px each · drag ◢`, x: ix, y: iy + isz + 27 });

    // ===================== stage 1 — encode =================================
    const ex = 138, ebY = iy + isz / 2 - 27;
    const gx = 206, gsz = 108;
    gridRect = { x: gx, y: iy, w: gsz, h: gsz };
    if (stage >= 1) {
      ctx.save();
      ctx.fillStyle = alphaOf(T.violet, 0.10); ctx.fillRect(ex, ebY, 58, 54);
      ctx.strokeStyle = T.violet; ctx.lineWidth = 1.2; ctx.strokeRect(ex, ebY, 58, 54);
      ctx.fillStyle = T.violet; ctx.font = mono(9); ctx.textAlign = 'center';
      ctx.fillText('ViT', ex + 29, ebY + 21); ctx.fillText('encoder', ex + 29, ebY + 33);
      ctx.fillStyle = T.n11; ctx.fillText(`D_enc=${D_ENC}`, ex + 29, ebY + 46);
      ctx.restore();
      flow(ix + isz + 3, iy + isz / 2, ex - 2, iy + isz / 2, T.warn, 0);
      flow(ex + 58 + 2, iy + isz / 2, gx - 3, iy + isz / 2, T.violet, 0.4);

      r.label('2 · encode', gx, iy - 26, { color: T.n14, font: mono(11) });
      r.label(`${num(g.N)} patch embeddings`, gx, iy - 12, { color: T.violet, font: mono(10) });
      const dN = Math.min(g.gridN, CAP), dc = gsz / dN;
      ctx.save();
      for (let rr = 0; rr < dN; rr++) for (let cc = 0; cc < dN; cc++) {
        const u = (cc + 0.5) / dN, v = (rr + 0.5) / dN;
        ctx.fillStyle = signedColor(chan(u, v));
        ctx.fillRect(gx + cc * dc, iy + rr * dc, Math.ceil(dc), Math.ceil(dc));
      }
      ctx.strokeStyle = T.n7; ctx.lineWidth = 1; ctx.strokeRect(gx - 0.5, iy - 0.5, gsz + 1, gsz + 1);
      ctx.restore();
      r.label(`grid ${g.gridN}×${g.gridN}`, gx, iy + isz + 14, { color: T.n12, font: mono(10) });
      dim({ text: dN < g.gridN ? `drawn ${dN}², count exact` : 'one vector per patch', x: gx, y: iy + isz + 27 });
    }

    // ===================== stage 2 — connect ================================
    const cx0 = 352, cy0 = 42, cw = 210, chh = 186;
    if (stage >= 2) {
      ctx.save();
      ctx.fillStyle = alphaOf(T.accent, 0.07); ctx.fillRect(cx0, cy0, cw, chh);
      ctx.strokeStyle = T.accent; ctx.lineWidth = 1.3; ctx.strokeRect(cx0, cy0, cw, chh);
      ctx.restore();
      r.label('3 · connect', cx0, cy0 - 20, { color: T.n14, font: mono(11) });
      r.label(CONN_LABEL[st.conn], cx0, cy0 - 7, { color: T.accent, font: mono(10) });
      flow(gx + gsz + 3, iy + isz / 2, cx0 - 3, iy + isz / 2, T.violet, 0.2);

      const my = cy0 + 26;
      ctx.save(); ctx.textAlign = 'left'; ctx.font = mono(9); ctx.fillStyle = T.n12;
      if (st.conn === 'projector') {
        // 1 patch -> 1 token, drawn as parallel lanes
        for (let i = 0; i < 5; i++) {
          const yy = my + i * 15;
          ctx.fillStyle = signedColor(chan((i + 1) / 6, 0.4));
          ctx.fillRect(cx0 + 12, yy, 13, 11);
          ctx.strokeStyle = alphaOf(T.accent, 0.6); ctx.lineWidth = 0.9;
          ctx.beginPath(); ctx.moveTo(cx0 + 27, yy + 5.5); ctx.lineTo(cx0 + 120, yy + 5.5); ctx.stroke();
          ctx.fillStyle = T.ok; ctx.fillRect(cx0 + 122, yy, 13, 11);
        }
        ctx.fillStyle = T.n12; ctx.font = mono(9);
        ctx.fillText(`Linear ${D_ENC}→${D_LLM}`, cx0 + 12, my + 5 * 15 + 16);
        ctx.fillStyle = T.bad;
        ctx.fillText('1 patch → 1 token, nothing merged', cx0 + 12, my + 5 * 15 + 30);
        ctx.fillStyle = T.n11;
        ctx.fillText('cost is the resolution itself', cx0 + 12, my + 5 * 15 + 43);
      } else if (st.conn === 'merge') {
        // k x k block -> concatenate along channels -> one projection -> 1 token
        const bs = 13, bx = cx0 + 14, by = my + 2;
        for (let rr = 0; rr < g.k; rr++) for (let cc = 0; cc < g.k; cc++) {
          ctx.fillStyle = signedColor(chan((cc + 0.5) / g.k * 0.4 + 0.3, (rr + 0.5) / g.k * 0.4 + 0.3));
          ctx.fillRect(bx + cc * bs, by + rr * bs, bs - 1, bs - 1);
        }
        ctx.strokeStyle = T.warn; ctx.lineWidth = 1.4; ctx.strokeRect(bx - 1, by - 1, g.k * bs + 1, g.k * bs + 1);
        const barY = my + 58, barW = 168;
        ctx.fillStyle = T.n12; ctx.font = mono(9);
        ctx.fillText(`${g.k}×${g.k} block → concat on channels`, bx, barY - 8);
        for (let i = 0; i < g.k * g.k; i++) {
          ctx.fillStyle = signedColor(chan(0.3 + 0.4 * ((i % g.k) + 0.5) / g.k, 0.3 + 0.4 * (((i / g.k) | 0) + 0.5) / g.k));
          ctx.fillRect(bx + i * (barW / (g.k * g.k)), barY, Math.max(2, barW / (g.k * g.k) - 0.8), 11);
        }
        ctx.strokeStyle = T.warn; ctx.lineWidth = 1; ctx.strokeRect(bx - 0.5, barY - 0.5, barW + 1, 12);
        ctx.fillStyle = T.n12;
        ctx.fillText(`ℝ^${num(g.k * g.k * D_ENC)}  →  Linear  →  ℝ^${D_LLM}`, bx, barY + 24);
        ctx.fillStyle = T.ok; ctx.fillRect(bx + barW / 2 - 7, barY + 32, 14, 11);
        ctx.fillStyle = T.okDeep;
        ctx.fillText(`${g.k}² = ${g.k * g.k} patches → 1 token`, bx, barY + 54);
        ctx.fillStyle = T.n11;
        ctx.fillText('spatial detail is what pays', bx, barY + 66);
      } else {
        // Q learned queries cross-attend to every patch
        const qx = cx0 + 14, qy = my + 4, shown = Math.min(6, Math.round(+st.queries));
        ctx.fillStyle = T.n12; ctx.font = mono(9);
        ctx.fillText(`${Math.round(+st.queries)} learned queries`, qx, qy - 6);
        for (let i = 0; i < shown; i++) {
          ctx.fillStyle = T.teal; ctx.fillRect(qx, qy + i * 13, 12, 10);
          for (let j = 0; j < 4; j++) {
            ctx.strokeStyle = alphaOf(T.teal, 0.22); ctx.lineWidth = 0.6;
            ctx.beginPath(); ctx.moveTo(qx + 13, qy + i * 13 + 5); ctx.lineTo(qx + 74, qy + 6 + j * 20); ctx.stroke();
          }
        }
        ctx.fillStyle = T.n11;
        ctx.fillText('cross-', qx + 80, qy + 26);
        ctx.fillText('attention', qx + 80, qy + 38);
        ctx.fillStyle = T.ok; for (let i = 0; i < shown; i++) ctx.fillRect(qx + 156, qy + i * 13, 12, 10);
        ctx.fillStyle = T.n12;
        ctx.fillText(`over all ${num(g.N)} patches`, qx, my + 88);
        ctx.fillStyle = T.warnDeep;
        ctx.fillText('token count is CONSTANT in the', qx, my + 100);
        ctx.fillText('resolution — a fixed bottleneck:', qx, my + 112);
        ctx.fillText('more pixels have nowhere to go.', qx, my + 124);
      }
      ctx.restore();

      // the draggable rail: merge factor, or query count, or nothing to tune
      const rlx = cx0 + 14, rly = cy0 + chh - 16, rlw = cw - 28;
      railRect = { x: rlx, y: rly, w: rlw };
      ctx.save();
      ctx.strokeStyle = T.n6; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(rlx, rly); ctx.lineTo(rlx + rlw, rly); ctx.stroke();
      if (st.conn === 'projector') {
        ctx.restore();
        dim({ text: 'no knob — nothing to trade', x: rlx, y: rly - 7 });
      } else {
        const f = st.conn === 'merge' ? (g.k - 1) / 3 : (Math.round(+st.queries) - 16) / 240;
        ctx.strokeStyle = T.accent; ctx.beginPath(); ctx.moveTo(rlx, rly); ctx.lineTo(rlx + rlw * f, rly); ctx.stroke();
        ctx.fillStyle = drag === 'rail' ? T.accent : T.n0;
        ctx.strokeStyle = T.accent; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(rlx + rlw * f, rly, 6, 0, 7); ctx.fill(); ctx.stroke();
        ctx.restore();
        r.label(st.conn === 'merge' ? `drag ↔  merge k = ${g.k}` : `drag ↔  queries = ${Math.round(+st.queries)}`,
          rlx, rly - 7, { color: T.accent, font: mono(9) });
      }

      // ---- output tokens -------------------------------------------------
      const ox = 596, osz = 108;
      outRect = { x: ox, y: iy, w: osz, h: osz };
      r.label('out', ox, iy - 26, { color: T.n14, font: mono(11) });
      r.label(`${num(g.tokens)} tokens`, ox, iy - 12, { color: T.ok, font: mono(10) });
      flow(cx0 + cw + 3, iy + isz / 2, ox - 3, iy + isz / 2, T.ok, 0.6);
      ctx.save();
      if (g.outN > 0) {
        const dN = Math.min(g.outN, CAP), dc = osz / dN;
        for (let rr = 0; rr < dN; rr++) for (let cc = 0; cc < dN; cc++) {
          const u = (cc + 0.5) / dN, v = (rr + 0.5) / dN;
          ctx.fillStyle = mixColor(T.ok, signedColor(chan(u, v)), 0.55);
          ctx.fillRect(ox + cc * dc, iy + rr * dc, Math.ceil(dc), Math.ceil(dc));
        }
        r.label(g.outN === g.gridN ? `still ${g.outN}×${g.outN} — one per patch` : `${g.outN}×${g.outN} — ${g.k}× coarser per side`,
          ox, iy + isz + 14, { color: T.n12, font: mono(10) });
      } else {
        // a resampler's tokens have NO spatial layout -- draw them as a list
        const q = Math.round(+st.queries), rows = Math.min(q, 24), rh = osz / rows;
        for (let i = 0; i < rows; i++) {
          ctx.fillStyle = mixColor(T.ok, T.teal, i / Math.max(1, rows - 1) * 0.6);
          ctx.fillRect(ox, iy + i * rh, osz, Math.max(1.5, rh - 1));
        }
        r.label(`${q} queries — no spatial grid`, ox, iy + isz + 14, { color: T.n12, font: mono(10) });
      }
      ctx.strokeStyle = T.n7; ctx.lineWidth = 1; ctx.strokeRect(ox - 0.5, iy - 0.5, osz + 1, osz + 1);
      ctx.restore();
      dim({ text: CONN_PAPER[st.conn], x: ox, y: iy + isz + 27 });

      // ---- the cost curve (live, all three shapes) ------------------------
      const px0 = 28, py0 = 250, pw = 330, ph = 132;
      r.label('tokens per image vs resolution (computed live)', px0, py0 - 10, { color: T.n14, font: mono(11) });
      ctx.save();
      ctx.fillStyle = T.n1; ctx.fillRect(px0, py0, pw, ph);
      ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(px0 - 0.5, py0 - 0.5, pw + 1, ph + 1);
      const yLo = Math.log10(8), yHi = Math.log10(24000);
      const Y = (t2) => py0 + ph - (Math.log10(Math.max(8, t2)) - yLo) / (yHi - yLo) * ph;
      const X = (res) => px0 + (res - RES_MIN) / (RES_MAX - RES_MIN) * pw;
      ctx.font = mono(8); ctx.fillStyle = T.n10; ctx.textAlign = 'right';
      for (const tk of [16, 64, 256, 1024, 4096, 16384]) {
        const yy = Y(tk);
        ctx.strokeStyle = rgbaToken('n14', 0.08); ctx.beginPath(); ctx.moveTo(px0, yy); ctx.lineTo(px0 + pw, yy); ctx.stroke();
        ctx.fillText(tk >= 1024 ? `${tk / 1024}k` : String(tk), px0 - 3, yy + 3);
      }
      ctx.textAlign = 'center';
      for (const rs of [224, 672, 1120, 1568]) ctx.fillText(String(rs), X(rs), py0 + ph + 11);
      const CURVES = [['projector', T.bad], ['merge', T.accent], ['resampler', T.teal]];
      for (const [name, col] of CURVES) {
        ctx.strokeStyle = col; ctx.lineWidth = name === st.conn ? 2.2 : 1;
        ctx.globalAlpha = name === st.conn ? 1 : 0.45;
        ctx.beginPath();
        for (let sx = 0; sx <= pw; sx += 2) {
          const res = RES_MIN + (sx / pw) * (RES_MAX - RES_MIN);
          const t2 = tokensAt(res, g.patch, name, g.k, Math.round(+st.queries));
          const yy = Y(t2);
          if (sx === 0) ctx.moveTo(px0 + sx, yy); else ctx.lineTo(px0 + sx, yy);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
      }
      // current point + the vertical "you are here"
      const curX = X(+st.res), curY = Y(g.tokens);
      ctx.strokeStyle = rgbaToken('n14', 0.35); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(curX, py0); ctx.lineTo(curX, py0 + ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = T.ok; ctx.beginPath(); ctx.arc(curX, curY, 4, 0, 7); ctx.fill();
      ctx.strokeStyle = T.n0; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.restore();
      let lx = px0 + 6, lyy = py0 + 12;
      for (const [name, col] of CURVES) {
        ctx.save(); ctx.fillStyle = col; ctx.fillRect(lx, lyy - 7, 10, 3); ctx.restore();
        r.label(name, lx + 14, lyy, { color: name === st.conn ? T.n14 : T.n10, font: mono(9) });
        lx += 14 + name.length * 5.6 + 12;
      }
      r.label('log scale ↑ · double the resolution → 4× a projector’s tokens',
        px0, py0 + ph + 24, { color: T.n11, font: mono(9) });
    }

    // ===================== stage 3 — splice + budget =========================
    if (stage >= 3) {
      const bx = 388, by = 250, bw = W - bx - 14;
      r.label(`context budget — ${num(ctxCap)} tokens`, bx, by - 10, { color: T.n14, font: mono(11) });
      const barY = by + 6, barH = 28;
      barRect = { x: bx, y: barY, w: bw, h: barH };
      ctx.save();
      ctx.fillStyle = T.n2; ctx.fillRect(bx, barY, bw, barH);
      const wText = Math.min(bw, bw * nText / ctxCap);
      ctx.fillStyle = alphaOf(T.accent, 0.85); ctx.fillRect(bx, barY, wText, barH);
      let cur = nText;
      for (let i = 0; i < nImg; i++) {
        const x0 = bx + bw * clamp(cur / ctxCap, 0, 1);
        const x1 = bx + bw * clamp((cur + g.tokens) / ctxCap, 0, 1);
        ctx.fillStyle = alphaOf(T.warn, i % 2 ? 0.9 : 0.62);
        ctx.fillRect(x0, barY, Math.max(0, x1 - x0), barH);
        cur += g.tokens;
      }
      ctx.strokeStyle = T.n7; ctx.lineWidth = 1; ctx.strokeRect(bx - 0.5, barY - 0.5, bw + 1, barH + 1);
      if (seq > ctxCap) {
        ctx.fillStyle = T.bad; ctx.font = mono(10); ctx.textAlign = 'right';
        ctx.fillText(`OVER by ${num(seq - ctxCap)}`, bx + bw - 5, barY + 19);
      }
      ctx.restore();
      const sevCol = pctSeq >= 100 ? T.bad : pctSeq >= 60 ? T.warn : T.ok;
      r.label(`${num(nText)} text  +  ${nImg} × ${num(g.tokens)} image  =  ${num(seq)} tokens`,
        bx, barY + barH + 15, { color: T.n13, font: mono(10) });
      r.label(`${pctSeq.toFixed(1)}% of the context — the pictures alone are ${pctCtx.toFixed(1)}%`,
        bx, barY + barH + 29, { color: sevCol, font: mono(10) });

      // attention work: it is the SQUARE of the sequence length
      const ay = barY + barH + 52;
      r.label('attention work ∝ seq²  (the text model, over the whole sequence)', bx, ay, { color: T.n14, font: mono(11) });
      const bars = [
        ['text only', nText * nText, T.accent],
        ['half res', seqHalf * seqHalf, T.teal],
        ['now', attnNow, sevCol],
        ['full context', attnFull, T.n8],
      ];
      const mx = Math.max(...bars.map((b) => b[1]));
      ctx.save(); ctx.font = mono(9);
      bars.forEach((b, i) => {
        const yy = ay + 12 + i * 17, bwv = (bw - 170) * (b[1] / mx);
        ctx.fillStyle = b[2]; ctx.fillRect(bx + 74, yy, Math.max(1, bwv), 11);
        ctx.fillStyle = T.n12; ctx.textAlign = 'right'; ctx.fillText(b[0], bx + 70, yy + 9);
        ctx.textAlign = 'left'; ctx.fillStyle = T.n11;
        ctx.fillText(`${(b[1] / attnFull * 100).toFixed(1)}% of full`, bx + 78 + Math.max(1, bwv), yy + 9);
      });
      ctx.restore();
      r.label(st.conn === 'resampler'
        ? 'halve the resolution and nothing moves — the connector sets the bill'
        : `halve the resolution → ${num(halfTok)} tok/image → attention ${(seqHalf * seqHalf / attnNow * 100).toFixed(1)}% of now`,
      bx, ay + 12 + bars.length * 17 + 12, { color: T.n11, font: mono(9) });
    }

    // ===================== hover-to-inspect =================================
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      const inRect = (rc) => rc && p.x >= rc.x && p.x <= rc.x + rc.w && p.y >= rc.y && p.y <= rc.y + rc.h;
      const patchAt = (rc) => {
        const c = clamp(Math.floor((p.x - rc.x) / rc.w * g.gridN), 0, g.gridN - 1);
        const rw = clamp(Math.floor((p.y - rc.y) / rc.h * g.gridN), 0, g.gridN - 1);
        return { r: rw, c, i: rw * g.gridN + c };
      };
      const dest = (q) => {
        if (st.conn === 'projector') return `→ output token #${num(q.i)}\none patch, one token`;
        if (st.conn === 'merge') {
          if (q.r >= g.gm * g.k || q.c >= g.gm * g.k) return `→ DROPPED — outside the ${g.gm}×${g.gm} merged grid\n(${g.gridN} is not a multiple of k=${g.k})`;
          const mr = (q.r / g.k) | 0, mc = (q.c / g.k) | 0;
          return `→ merged into group (${mr},${mc}) = output token #${num(mr * g.gm + mc)}\nwith ${g.k * g.k - 1} neighbours, concatenated on the channel axis`;
        }
        return `→ no single token: all ${Math.round(+st.queries)} learned queries\ncross-attend to this patch. Position is not preserved.`;
      };
      if (inRect(imgRect)) {
        const q = patchAt(imgRect);
        page.setTip(`patch (${q.r},${q.c}) = #${num(q.i)} of ${num(g.N)}\n${g.patch}×${g.patch} px\n${dest(q)}`);
      } else if (stage >= 1 && inRect(gridRect)) {
        const q = patchAt(gridRect);
        page.setTip(`patch embedding #${num(q.i)} ∈ ℝ^${D_ENC}\n${dest(q)}`);
      } else if (stage >= 2 && inRect(outRect)) {
        if (g.outN > 0) {
          const c = clamp(Math.floor((p.x - outRect.x) / outRect.w * g.outN), 0, g.outN - 1);
          const rw = clamp(Math.floor((p.y - outRect.y) / outRect.h * g.outN), 0, g.outN - 1);
          page.setTip(`output token #${num(rw * g.outN + c)} ∈ ℝ^${D_LLM}\n${st.conn === 'merge' ? `carries the ${g.k}×${g.k} patch block at (${rw * g.k},${c * g.k})` : `carries patch (${rw},${c})`}`);
        } else {
          const q = Math.round(+st.queries), rows = Math.min(q, 24);
          const i = clamp(Math.floor((p.y - outRect.y) / outRect.h * rows), 0, rows - 1);
          page.setTip(`learned query #${i} ∈ ℝ^${D_LLM}\nsame ${q} queries at every resolution —\nthis is the bottleneck`);
        }
      } else if (stage >= 3 && inRect(barRect)) {
        const at = Math.floor((p.x - barRect.x) / barRect.w * ctxCap);
        page.setTip(at < nText ? `context position ${num(at)} — text`
          : at < seq ? `context position ${num(at)} — image ${Math.floor((at - nText) / g.tokens) + 1} of ${nImg}`
            : `context position ${num(at)} — free (${num(ctxCap - seq)} left)`);
      }
    }

    // ===================== readout ==========================================
    const eq = st.conn === 'resampler'
      ? `tokens = queries = ${num(g.tokens)} — independent of (${+st.res}/${g.patch})² = ${num(g.N)} patches`
      : st.conn === 'merge'
        ? `tokens = (${+st.res}/${g.patch})² / ${g.k}² = ${num(g.gridN)}² / ${g.k * g.k} = ${num(g.tokens)}`
        : `tokens = (${+st.res}/${g.patch})² = ${num(g.gridN)}² = ${num(g.tokens)}  (a projector folds nothing, so the k slider does not apply)`;
    let o = `${CONN_LABEL[st.conn]} · ${eq}   tier:${r.name}\n`;
    o += `${num(g.tokens)} tokens/image × ${nImg} image${nImg === 1 ? '' : 's'} = ${num(imgTok)} image tokens + ${num(nText)} text = ${num(seq)} of ${num(ctxCap)} context (${pctSeq.toFixed(1)}%; the pictures alone eat ${pctCtx.toFixed(1)}%). `;
    o += `Attention over the sequence costs seq² = ${sci(attnNow)} units = ${(attnNow / attnFull * 100).toFixed(2)}% of a full ${num(ctxCap)}-token pass. `;
    o += st.conn === 'resampler'
      ? `Halving the resolution changes nothing — still ${num(halfTok)} tokens/image, ${(seqHalf * seqHalf / attnNow * 100).toFixed(1)}% of the same work: the connector sets the bill, not the pixels.\n`
      : `Halving the resolution gives ${num(halfTok)} tokens/image and ${(seqHalf * seqHalf / attnNow * 100).toFixed(1)}% of that work — the cost moves with the SQUARE, twice over.\n`;
    o += st.conn === 'projector'
      ? `A projector spends one token per patch, so resolution IS the token budget: there is no knob to turn but the picture size.`
      : st.conn === 'merge'
        ? `Patch-merge divides by k² = ${g.k * g.k}, paying in spatial detail: ${g.k * g.k} neighbouring patches now share one token, and ${g.gridN % g.k ? `the ${g.gridN % g.k} leftover row(s)/column(s) fall outside the ${g.gm}×${g.gm} merged grid` : `the grid divides exactly`}.`
        : `A resampler flattens the curve to a horizontal line — and that is the whole cost: ${num(g.tokens)} queries is a FIXED bottleneck, so pushing the resolution past ${num(g.N)} patches adds detail the connector has no room to carry.`;
    o += rec ? `\nstep ${rec.stage + 1}/4 — ${rec.label}` : '';
    page.setReadout(o);
  },
}).then((page) => {
  window.__vlmConnectorPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  for (const k of ['conn', 'patch', 'ctx']) if (q.has(k)) page.controls.set(k, q.get(k), { silent: true });
  for (const k of ['res', 'merge', 'queries', 'imgs', 'text', 'seed']) {
    if (q.has(k)) page.controls.set(k, +q.get(k), { silent: true });
  }
  // ?hover=x,y is the headless stand-in for a real cursor (canvas-space px), the
  // same way ?res= / ?merge= stand in for the two drag handles.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause before seeking so autoplay cannot
  // advance off the requested step.
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
