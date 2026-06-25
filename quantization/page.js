// quantization concept page -- group-wise affine quantization of fp16 weights to
// low-bit integers (the int4/int3 weight-quant used by GPTQ/AWQ/GGUF k-quants).
// Split the weights into groups of G; per group find [min,max], pick a scale
// s=(max-min)/(2^bits-1) and an integer zero-point z, store each weight as a
// small code q=clamp(round((x-min)/s),0,2^bits-1). Dequantize x'=min+s·q. The
// gap x-x' is the quantization error: more bits -> finer levels -> less error;
// smaller groups adapt to the local range (less error) but cost more scale/zp
// overhead (less compression). Drag a weight into an outlier and watch its
// group's scale blow up, coarsening every other weight in that group.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df', RED = '#d1242f';
const GCOL = ['#1f6feb', '#d2691e', '#2ca02c', '#8250df', '#0a9396', '#bc6c25', '#d1242f', '#5a189a'];
const VMAX = 2.5;

let cur = null, bsig = '', barRects = null, dragI = -1;

function build(preset, seed, N) {
  const r = seededRandn(seed, N, { std: 1 }), x = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (preset === 'uniform') x[i] = r[i] * 0.9;
    else if (preset === 'outlier') { x[i] = r[i] * 0.45; if (i % 17 === 5) x[i] = (r[i] > 0 ? 1 : -1) * (1.8 + 0.4 * Math.abs(r[i])); }
    else x[i] = r[i] * 0.5;  // gaussian
    x[i] = Math.max(-VMAX, Math.min(VMAX, x[i]));
  }
  return { x };
}

// quantize one group's values -> {s, lo, z, q[], deq[]}
function qGroup(vals, bits) {
  let lo = Infinity, hi = -Infinity; for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const levels = (1 << bits) - 1, s = (hi - lo) / levels || 1e-9, z = Math.round(-lo / s);
  const q = vals.map((v) => Math.max(0, Math.min(levels, Math.round((v - lo) / s)))), deq = q.map((c) => lo + s * c);
  return { s, lo, hi, z, levels, q, deq };
}
function rmseAt(x, N, G, bits) {
  let se = 0; for (let g0 = 0; g0 < N; g0 += G) { const grp = Array.from(x.slice(g0, g0 + G)), { deq } = qGroup(grp, bits); for (let i = 0; i < grp.length; i++) se += (grp[i] - deq[i]) ** 2; }
  return Math.sqrt(se / N);
}

mount({
  mount: 'body',
  title: 'quantization — fp16 → int4, group-wise',
  blurb: 'Large-model weights ship as low-bit integers (int4, even int3/int2) instead of fp16 — that is how a 70B model fits in 24 GB. The scheme is group-wise affine quantization: chop the weights into groups of G, and per group store a scale s and an integer zero-point z so each weight becomes a tiny code q = round((x − min)/s) in [0, 2^bits−1]. To use a weight you DEQUANTIZE: x′ = min + s·q. The error is the gap x − x′. Two knobs trade off: more BITS = more levels = finer steps = less error (but bigger files); smaller GROUPS adapt s to the local range so outliers hurt fewer weights (less error) but the per-group scale/zp overhead grows, lowering the compression ratio. Drag a weight to an outlier and watch its whole group’s scale stretch and every neighbour get coarser. The error-vs-bits curve plots RMSE as you’d add bits; the level lines show each group snapping to its 2^bits reconstruction levels.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  compare: { key: 'bits', a: 8, b: 3, labelA: '8-bit (fine)', labelB: '3-bit (coarse)' },
  challenges: [
    { goal: 'Quantize accurately: get the reconstruction RMSE below 0.03.', hint: 'more bits = finer levels; a smaller group adapts to the local range.', check: (api) => ({ solved: (api.probe.rmse ?? 1) < 0.03, detail: `RMSE = ${(api.probe.rmse ?? 1).toFixed(4)} (need < 0.03)` }) },
    { goal: 'Compress hard: reach ≥ 4× smaller than fp16 while keeping RMSE < 0.1.', hint: 'fewer bits + larger groups shrink the size — watch the error climb.', check: (api) => ({ solved: (api.probe.ratio ?? 0) >= 4 && (api.probe.rmse ?? 1) < 0.1, detail: `${(api.probe.ratio ?? 0).toFixed(1)}×, RMSE ${(api.probe.rmse ?? 1).toFixed(3)}` }) },
  ],
  controls: (c, page) => {
    c.select('preset', { label: 'weights', options: ['gaussian', 'outlier', 'uniform'], value: 'gaussian', rebuild: true });
    c.slider('bits', { label: 'bits', min: 2, max: 8, step: 1, value: 4 });
    c.select('G', { label: 'group size', options: ['8', '16', '32', '64'], value: '16' });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !barRects) return;
    const N = cur.x.length, bw = barRects.w / N;
    const at = (x) => { const i = Math.floor((x - barRects.x) / bw); return (i >= 0 && i < N) ? i : -1; };
    const setv = (y) => Math.max(-VMAX, Math.min(VMAX, VMAX - (y - barRects.y) / barRects.h * (2 * VMAX)));
    if (ev.type === 'down') { dragI = (ev.y >= barRects.y - 6 && ev.y <= barRects.y + barRects.h + 6) ? at(ev.x) : -1; if (dragI >= 0) { cur.x[dragI] = setv(ev.y); page.redraw(); } }
    else if (ev.type === 'up' || ev.type === 'leave') dragI = -1;
    else if (ev.type === 'move' && dragI >= 0 && page.pointer.down) { cur.x[dragI] = setv(ev.y); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H, N = 64;
    if (`${st.preset}|${st.seed}` !== bsig) { cur = build(st.preset, st.seed | 0, N); bsig = `${st.preset}|${st.seed}`; }
    r.clear('#ffffff');
    const bits = st.bits | 0, G = st.G | 0, levels = (1 << bits) - 1, nG = N / G, x = cur.x;

    // ===== weights panel: original tick + dequant bar, per group =====
    const bx = 20, by = 52, bw = W - 40, bh = 132, barW = bw / N, zy = by + bh / 2, Y = (v) => zy - v / VMAX * (bh / 2);
    barRects = { x: bx, y: by, w: bw, h: bh };
    r.label('weights — bar = dequantized x′,  black tick = original fp16 x,  gap = error   (drag a weight ↕)', bx, by - 8, { color: INK, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.strokeStyle = '#e6e8ea'; ctx.strokeRect(bx, by, bw, bh); ctx.strokeStyle = '#cfd4d9'; ctx.beginPath(); ctx.moveTo(bx, zy); ctx.lineTo(bx + bw, zy); ctx.stroke(); ctx.restore();
    const swG = Math.floor((page.t || 0) / 0.6) % nG;  // animated group scan
    let totSE = 0, maxE = 0;
    for (let g = 0; g < nG; g++) {
      const g0 = g * G, grp = Array.from(x.slice(g0, g0 + G)), Q = qGroup(grp, bits), col = GCOL[g % GCOL.length];
      // group bg + scan highlight
      const gx0 = bx + g0 * barW, gxw = G * barW;
      if (g === swG) { ctx.save(); ctx.fillStyle = 'rgba(255,193,7,0.10)'; ctx.fillRect(gx0, by, gxw, bh); ctx.restore(); }
      // reconstruction level lines (only when few enough to read)
      if (levels <= 31) { ctx.save(); ctx.strokeStyle = 'rgba(130,80,223,0.18)'; ctx.lineWidth = 0.5; for (let k = 0; k <= levels; k++) { const yv = Y(Q.lo + Q.s * k); ctx.beginPath(); ctx.moveTo(gx0, yv); ctx.lineTo(gx0 + gxw, yv); ctx.stroke(); } ctx.restore(); }
      for (let i = 0; i < grp.length; i++) {
        const xi = bx + (g0 + i) * barW, e = Math.abs(grp[i] - Q.deq[i]); totSE += (grp[i] - Q.deq[i]) ** 2; if (e > maxE) maxE = e;
        ctx.fillStyle = (g0 + i) === dragI ? RED : col; ctx.globalAlpha = 0.85;
        const dq = Q.deq[i]; ctx.fillRect(xi + 1, dq >= 0 ? Y(dq) : zy, Math.max(1, barW - 1.5), Math.abs(Y(dq) - zy)); ctx.globalAlpha = 1;
        ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xi + 1, Y(grp[i])); ctx.lineTo(xi + barW - 0.5, Y(grp[i])); ctx.stroke();  // original tick
      }
      // group divider + s/z label
      ctx.save(); ctx.strokeStyle = '#b8bec4'; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.moveTo(gx0, by); ctx.lineTo(gx0, by + bh); ctx.stroke(); ctx.setLineDash([]); ctx.restore();
      r.label(`s=${Q.s.toFixed(3)} z=${Q.z}`, gx0 + 2, by + bh + 12, { color: col, font: '9px ui-monospace, monospace' });
    }
    const rmse = Math.sqrt(totSE / N);
    r.label(`${nG} groups of ${G}  ·  ${bits}-bit → ${levels + 1} levels per group  ·  codes q ∈ [0, ${levels}]`, bx, by + bh + 28, { color: '#586069', font: '10px ui-monospace, monospace' });

    // ===== metrics (bottom-left) =====
    const my = by + bh + 48;
    const effBits = bits + (16 + 8) / G, ratio = 16 / effBits;
    page.probe = { rmse, ratio };
    r.label(`reconstruction error   RMSE = ${rmse.toFixed(4)}   max|err| = ${maxE.toFixed(4)}`, bx, my, { color: INK, font: '11px ui-monospace, monospace' });
    r.label(`storage: ${bits} bits/weight + (16-bit scale + 8-bit zp)/${G} = ${effBits.toFixed(2)} eff. bits/weight`, bx, my + 16, { color: '#586069', font: '10px ui-monospace, monospace' });
    r.label(`compression vs fp16:  16 / ${effBits.toFixed(2)} = ${ratio.toFixed(2)}×   ${st.preset === 'outlier' ? '(note: outliers stretch s → coarser neighbours; smaller G limits the damage)' : ''}`, bx, my + 32, { color: GREEN, font: '10px ui-monospace, monospace' });

    // ===== error-vs-bits curve (bottom-right) =====
    const cwW = 250, chx = W - cwW - 24, chy = my - 10, chh = 84;
    r.label('RMSE vs bits (this G)', chx, chy - 6, { color: INK, font: '10px ui-monospace, monospace' });
    const eLo = -3, eHi = 0, EX = (b) => chx + (b - 2) / 6 * cwW, EY = (e) => chy + chh - (Math.max(eLo, Math.min(eHi, Math.log10(e + 1e-9)) ) - eLo) / (eHi - eLo) * chh;
    ctx.save(); ctx.strokeStyle = '#eceef0'; ctx.strokeRect(chx, chy, cwW, chh);
    ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.beginPath();
    for (let b = 2; b <= 8; b++) { const e = rmseAt(x, N, G, b), px = EX(b), py = EY(e); if (b === 2) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke();
    for (let b = 2; b <= 8; b++) { const e = rmseAt(x, N, G, b); ctx.fillStyle = b === bits ? RED : ORANGE; ctx.beginPath(); ctx.arc(EX(b), EY(e), b === bits ? 4 : 2.2, 0, 7); ctx.fill(); }
    ctx.restore();
    r.label('2', chx - 2, chy + chh + 11, { color: '#8a939b', font: '8px ui-monospace, monospace' }); r.label('8 bits', chx + cwW - 26, chy + chh + 11, { color: '#8a939b', font: '8px ui-monospace, monospace' });
    r.label(`now: ${bits}-bit → ${rmse.toFixed(4)}`, chx, chy + chh + 24, { color: RED, font: '10px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && dragI < 0) {
      const p = page.pointer;
      if (p.x >= bx && p.x <= bx + bw && p.y >= by && p.y <= by + bh) {
        const i = Math.floor((p.x - bx) / barW); if (i >= 0 && i < N) { const g = (i / G) | 0, Q = qGroup(Array.from(x.slice(g * G, g * G + G)), bits), li = i - g * G; page.setTip(`weight ${i} (group ${g})\nx = ${x[i].toFixed(3)} fp16\nq = ${Q.q[li]} (of 0..${levels})  →  x′ = ${Q.deq[li].toFixed(3)}\nerr = ${(x[i] - Q.deq[li]).toFixed(4)}   [s=${Q.s.toFixed(3)} z=${Q.z}]\ndrag ↕`); }
      }
    }

    let o = `group-wise affine quant: fp16 → ${bits}-bit, group size ${G}.  per group: s=(max−min)/${levels}, z=round(−min/s), q=clamp(round((x−min)/s),0,${levels}), x′=min+s·q.   tier:${r.name}\n`;
    o += `RMSE=${rmse.toFixed(4)}, max|err|=${maxE.toFixed(4)} at ${bits} bits / G=${G}. Effective ${effBits.toFixed(2)} bits/weight → ${ratio.toFixed(2)}× vs fp16. ${st.preset === 'outlier' ? 'Outlier preset: a few large weights stretch their group scale, coarsening the rest — exactly why small groups (or outlier-aware methods) help.' : 'More bits → finer levels → lower error; smaller G → lower error but more scale/zp overhead.'}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__qPage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['preset', 'G']) if (q.has(key)) page.controls.set(key, q.get(key));
  if (q.has('bits')) page.controls.set('bits', +q.get('bits'));
  if (q.has('seed')) page.controls.set('seed', +q.get('seed'));
  if (q.has('drag') && cur) for (const pr of q.get('drag').split(';')) { const [i, v] = pr.split(',').map(Number); if (i >= 0 && i < cur.x.length) cur.x[i] = v; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
