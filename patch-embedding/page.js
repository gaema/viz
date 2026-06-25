// patch-embedding concept page -- the Vision Transformer (ViT) front end: how an
// image becomes a SEQUENCE OF TOKENS, the CNN->transformer bridge.
//   1. cut the [S×S×3] image into a grid of non-overlapping P×P patches
//   2. flatten each patch (P·P·3 numbers) and LINEARLY PROJECT it: token = W·flat
//      (W is D×(P²·3)) -- this is exactly a Conv2d with kernel=stride=P
//   3. prepend a learnable [CLS] token + add a positional embedding pos[i]
//   => a length-(N+1) sequence of D-dim tokens that a transformer consumes.
// Drag (or slider) to pick a patch; tune patch size P and embed dim D; the
// rasterize-to-tokens sweep animates.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df', RED = '#d1242f';
const S = 96;  // image side (px); divisible by every P option
const sign = (v, d) => { const t = Math.max(-1, Math.min(1, v / (d || 1))), m = Math.abs(t); return t >= 0 ? `rgb(255,${Math.round(255 - m * 150)},${Math.round(255 - m * 165)})` : `rgb(${Math.round(255 - m * 165)},${Math.round(255 - m * 120)},255)`; };

let cur = null, bsig = '';
let imgRect = null, dragging = false;

function buildImage(preset, seed) {
  const cv = document.createElement('canvas'); cv.width = S; cv.height = S;
  const c = cv.getContext('2d'), img = c.createImageData(S, S), d = img.data;
  const rnd = seededRandn(seed, 16, { std: 1 });
  const blobs = [];
  for (let b = 0; b < 4; b++) blobs.push({ cx: (rnd[b * 3] * 0.5 + 0.5) * S, cy: (rnd[b * 3 + 1] * 0.5 + 0.5) * S, r: 18 + 14 * Math.abs(rnd[b * 3 + 2]), col: [[230, 70, 70], [70, 120, 230], [70, 200, 120], [240, 190, 60]][b] });
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let rr, gg, bb;
    if (preset === 'gradient') { rr = 60 + 180 * x / S; gg = 60 + 180 * y / S; bb = 220 - 160 * (x + y) / (2 * S); }
    else if (preset === 'stripes') { const t = (Math.sin((x + y) * 0.35) * 0.5 + 0.5); rr = 60 + 180 * t; gg = 200 - 150 * t; bb = 120 + 100 * Math.sin(x * 0.2); }
    else { rr = gg = bb = 238; for (const bl of blobs) { const w = Math.exp(-((x - bl.cx) ** 2 + (y - bl.cy) ** 2) / (2 * bl.r * bl.r)); rr = rr * (1 - w) + bl.col[0] * w; gg = gg * (1 - w) + bl.col[1] * w; bb = bb * (1 - w) + bl.col[2] * w; } }
    const i = (y * S + x) * 4; d[i] = rr; d[i + 1] = gg; d[i + 2] = bb; d[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  return { cv, data: d };
}

function build(st) {
  const P = st.P | 0, D = st.D | 0, gN = S / P, N = gN * gN, dim = P * P * 3;
  const im = buildImage(st.preset, st.seed | 0);
  const W = seededRandn((st.seed | 0) + 1, [D, dim], { std: 1 / Math.sqrt(dim) }).data;
  const pos = seededRandn((st.seed | 0) + 2, [N, D], { std: 0.35 }).data;
  const cls = seededRandn((st.seed | 0) + 3, D, { std: 0.5 });
  // flatten + project each patch  (token i = W·flat_i + pos_i)
  const tokens = new Float32Array(N * D), flat = new Float32Array(dim);
  for (let pi = 0; pi < N; pi++) {
    const pr = (pi / gN | 0) * P, pc = (pi % gN) * P; let f = 0;
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const si = ((pr + y) * S + (pc + x)) * 4; flat[f++] = im.data[si] / 255 - 0.5; flat[f++] = im.data[si + 1] / 255 - 0.5; flat[f++] = im.data[si + 2] / 255 - 0.5; }
    for (let o = 0; o < D; o++) { let s = 0; for (let j = 0; j < dim; j++) s += W[o * dim + j] * flat[j]; tokens[pi * D + o] = s + pos[pi * D + o]; }
  }
  return { im, P, D, gN, N, dim, W, pos, cls, tokens };
}

mount({
  mount: 'body',
  title: 'patch-embedding — ViT: image → patches → tokens',
  blurb: 'A Vision Transformer has no convolutions in its trunk — so how does an image enter a transformer? It is cut into a grid of non-overlapping P×P patches; each patch (P·P·3 raw numbers) is flattened and LINEARLY PROJECTED by a shared matrix W (D×P²·3) into a D-dimensional token. That projection is mathematically identical to a single Conv2d with kernel size = stride = P — the CNN→transformer bridge. A learnable [CLS] token is prepended and a positional embedding pos[i] is added to each (the transformer is otherwise order-blind), giving a length-(N+1) sequence of D-dim tokens that self-attention then mixes. Drag on the image (or the slider) to pick a patch and watch it flatten → project → land as one token in the sequence; tune the patch size P (fewer, bigger patches → shorter sequence) and the embed dim D; the rasterize-to-tokens sweep animates.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  controls: (c, page) => {
    c.select('preset', { label: 'image', options: ['blobs', 'gradient', 'stripes'], value: 'blobs', rebuild: true });
    c.select('P', { label: 'patch size P', options: ['8', '12', '16', '24', '32'], value: '16', rebuild: true });
    c.select('D', { label: 'embed dim D', options: ['16', '24', '32'], value: '32', rebuild: true });
    c.slider('patch', { label: 'selected patch', min: 0, max: 143, step: 1, value: 7 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !imgRect) return;
    const cell = imgRect.w / cur.gN;
    const at = (x, y) => { const pc = Math.floor((x - imgRect.x) / cell), pr = Math.floor((y - imgRect.y) / cell); return (pr >= 0 && pr < cur.gN && pc >= 0 && pc < cur.gN && x >= imgRect.x && y >= imgRect.y) ? pr * cur.gN + pc : -1; };
    if (ev.type === 'down') { const i = at(ev.x, ev.y); dragging = i >= 0; if (i >= 0) { page.controls.set('patch', i); page.redraw(); } }
    else if (ev.type === 'up' || ev.type === 'leave') dragging = false;
    else if (ev.type === 'move' && dragging && page.pointer.down) { const i = at(ev.x, ev.y); if (i >= 0) { page.controls.set('patch', i); page.redraw(); } }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    const sig = `${st.preset}|${st.P}|${st.D}|${st.seed}`;
    if (sig !== bsig) { cur = build(st); bsig = sig; }
    r.clear('#ffffff');
    const { P, D, gN, N, dim, tokens } = cur, sel = Math.min(st.patch | 0, N - 1), sr = (sel / gN | 0), sc = sel % gN;
    const tdom = 1.2;

    // ===== image with patch grid (top-left) =====
    const ix = 20, iy = 52, isz = 138, cell = isz / gN;
    imgRect = { x: ix, y: iy, w: isz, h: isz };
    r.label('image  (S×S×3)', ix, iy - 8, { color: INK, font: '11px ui-monospace, monospace' });
    ctx.drawImage(cur.im.cv, ix, iy, isz, isz);
    ctx.save(); ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 0.6; for (let i = 0; i <= gN; i++) { ctx.beginPath(); ctx.moveTo(ix + i * cell, iy); ctx.lineTo(ix + i * cell, iy + isz); ctx.stroke(); ctx.beginPath(); ctx.moveTo(ix, iy + i * cell); ctx.lineTo(ix + isz, iy + i * cell); ctx.stroke(); }
    // animated rasterize sweep
    const swp = Math.floor((page.t || 0) / 0.25) % N, swr = (swp / gN | 0), swc = swp % gN;
    ctx.strokeStyle = 'rgba(255,193,7,0.9)'; ctx.lineWidth = 1.6; ctx.strokeRect(ix + swc * cell + 1, iy + swr * cell + 1, cell - 2, cell - 2);
    // selected patch
    ctx.strokeStyle = PURPLE; ctx.lineWidth = 2.4; ctx.strokeRect(ix + sc * cell, iy + sr * cell, cell, cell); ctx.restore();
    r.label(`${gN}×${gN} = ${N} patches of ${P}×${P}  ·  drag a patch`, ix, iy + isz + 14, { color: '#586069', font: '10px ui-monospace, monospace' });

    // ===== selected patch -> flatten -> project -> token (top-middle/right) =====
    const px = 210, py = 56, pe = Math.min(96, P * 7), pcell = pe / P;
    r.label('patch', px, py - 8, { color: PURPLE, font: '10px ui-monospace, monospace' });
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const si = ((sr * P + y) * S + (sc * P + x)) * 4; ctx.fillStyle = `rgb(${cur.im.data[si]},${cur.im.data[si + 1]},${cur.im.data[si + 2]})`; ctx.fillRect(px + x * pcell, py + y * pcell, Math.ceil(pcell), Math.ceil(pcell)); }
    ctx.save(); ctx.strokeStyle = PURPLE; ctx.lineWidth = 1.4; ctx.strokeRect(px, py, pe, pe); ctx.restore();
    // projection box
    const wx = px + pe + 28;
    ctx.save(); ctx.fillStyle = 'rgba(31,111,235,0.08)'; ctx.fillRect(wx, py + pe / 2 - 30, 116, 60); ctx.strokeStyle = BLUE; ctx.lineWidth = 1.3; ctx.strokeRect(wx, py + pe / 2 - 30, 116, 60);
    ctx.fillStyle = INK; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('flatten → ℝ^' + dim, wx + 58, py + pe / 2 - 14); ctx.fillText(`linear W: ${D}×${dim}`, wx + 58, py + pe / 2); ctx.fillStyle = BLUE; ctx.fillText('+ pos[' + sel + ']', wx + 58, py + pe / 2 + 16); ctx.restore();
    // token (D vertical bars)
    const tx = wx + 116 + 24, tw = 92, th = pe, bw = tw / D, zy = py + th / 2;
    r.label(`token ${sel}  (ℝ^${D})`, tx, py - 8, { color: GREEN, font: '10px ui-monospace, monospace' });
    ctx.save(); ctx.strokeStyle = '#e6e8ea'; ctx.strokeRect(tx, py, tw, th); ctx.strokeStyle = '#cfd4d9'; ctx.beginPath(); ctx.moveTo(tx, zy); ctx.lineTo(tx + tw, zy); ctx.stroke();
    for (let o = 0; o < D; o++) { const v = Math.max(-tdom, Math.min(tdom, tokens[sel * D + o])), bh = Math.abs(v) / tdom * (th / 2); ctx.fillStyle = GREEN; ctx.fillRect(tx + o * bw, v >= 0 ? zy - bh : zy, Math.max(1, bw - 0.5), bh); } ctx.restore();
    // arrows
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.3; const ay = py + pe / 2; ctx.beginPath(); ctx.moveTo(ix + isz + 2, ay - 6); ctx.lineTo(px - 4, ay - 6); ctx.moveTo(px + pe + 3, ay); ctx.lineTo(wx - 3, ay); ctx.moveTo(wx + 116 + 3, ay); ctx.lineTo(tx - 3, ay); ctx.stroke(); ctx.restore();

    // ===== token sequence (bottom) =====
    const seqY = 224, seqX = 20, cols = N + 1, colW = Math.min(9, (W - 40) / cols), seqH = Math.min(104, D * 4), ch = seqH / D;
    r.label('token sequence fed to the transformer:  [CLS]  +  patch tokens (raster order)  + positional embedding', seqX, seqY - 8, { color: INK, font: '11px ui-monospace, monospace' });
    // CLS
    ctx.save();
    for (let o = 0; o < D; o++) { ctx.fillStyle = sign(cur.cls[o], tdom); ctx.fillRect(seqX, seqY + o * ch, colW - 0.6, ch - 0.3); }
    ctx.strokeStyle = RED; ctx.lineWidth = 1.4; ctx.strokeRect(seqX - 0.5, seqY - 0.5, colW, seqH + 1); ctx.restore();
    r.label('CLS', seqX - 2, seqY + seqH + 12, { color: RED, font: '9px ui-monospace, monospace' });
    // patch tokens
    ctx.save();
    for (let pi = 0; pi < N; pi++) { const cx0 = seqX + (pi + 1) * colW; for (let o = 0; o < D; o++) { ctx.fillStyle = sign(tokens[pi * D + o], tdom); ctx.fillRect(cx0, seqY + o * ch, colW - 0.6, ch - 0.3); } }
    // selected + sweep highlight on the strip
    ctx.strokeStyle = PURPLE; ctx.lineWidth = 1.8; ctx.strokeRect(seqX + (sel + 1) * colW - 0.5, seqY - 0.5, colW, seqH + 1);
    ctx.strokeStyle = 'rgba(255,193,7,0.9)'; ctx.lineWidth = 1.4; ctx.strokeRect(seqX + (swp + 1) * colW - 0.5, seqY - 0.5, colW, seqH + 1);
    ctx.restore();
    r.label(`${N + 1} tokens × ${D} dims   (D rows ↓, sequence position →)`, seqX, seqY + seqH + 12, { color: '#586069', font: '9px ui-monospace, monospace' });
    r.label('patch-embedding ≡ Conv2d(kernel = stride = P): a single conv that tiles the image into tokens — the CNN→transformer bridge.', seqX, seqY + seqH + 30, { color: PURPLE, font: '10px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && !dragging) {
      const p = page.pointer;
      if (p.x >= ix && p.x <= ix + isz && p.y >= iy && p.y <= iy + isz) { const pc = Math.floor((p.x - ix) / cell), pr = Math.floor((p.y - iy) / cell); page.setTip(`patch (${pr},${pc}) = #${pr * gN + pc}\n${P}×${P}×3 = ${dim} numbers → token ${pr * gN + pc}\ndrag to select`); }
      else if (p.y >= seqY && p.y <= seqY + seqH && p.x >= seqX) { const ci = Math.floor((p.x - seqX) / colW); if (ci === 0) page.setTip('[CLS] token\nlearnable; its final state is the image summary'); else if (ci - 1 < N) page.setTip(`token ${ci - 1}  (patch ${(ci - 1) / gN | 0},${(ci - 1) % gN})\n= W·flatten(patch) + pos[${ci - 1}]`); }
    }

    let o = `ViT patch embedding: image → ${gN}×${gN}=${N} patches of ${P}×${P} → flatten (${dim}) → linear W (${D}×${dim}) → ${N} tokens (+[CLS]) + positional emb.   tier:${r.name}\n`;
    o += `selected patch #${sel} (${sr},${sc}) → token ${sel} ∈ ℝ^${D}. Sequence length ${N + 1} (= ${N} patches + 1 CLS). Smaller P → more, smaller patches → longer sequence (quadratic in 1/P). The projection is exactly Conv2d(kernel=stride=${P}).`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__pePage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['preset', 'P', 'D', 'seed']) if (q.has(key)) page.controls.set(key, q.get(key));
  if (q.has('patch')) page.controls.set('patch', +q.get('patch'));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
