// convolution concept page -- a k×k kernel sliding over a zero-padded input,
// each output pixel a sum of products over the receptive field, with the
// geometry knobs stride / padding / dilation:
//   out[oy,ox] = Σᵢⱼ W[i,j] · Xpad[oy·s + i·dil, ox·s + j·dil]
// The current receptive field is outlined on the input with the kernel weights
// overlaid; the output fills as the kernel slides. Drag an input cell to change
// it; hover to inspect. cur is built ONLY by the transport compute (no draw-side
// rebuild -- that wipes drag edits).
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', GREY = '#9aa4ad';
const divcol = (v, dom) => { const t = Math.max(-1, Math.min(1, v / (dom || 1))), m = Math.abs(t); return t >= 0 ? `rgb(255,${Math.round(255 - m * 150)},${Math.round(255 - m * 165)})` : `rgb(${Math.round(255 - m * 165)},${Math.round(255 - m * 120)},255)`; };
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

let cur = null;
let inRect = null, kRect = null, outRect = null, ic = 24, oc = 24, geomP = 0;   // captured in draw
let grab = null;                                                                // {r,c} input cell (unpadded) while dragging

function buildData(st) {
  const n = st.n | 0, k = st.k | 0, seed = st.seed | 0;
  const X = seededRandn(seed, [n, n], { std: 1 }).data;
  const W = seededRandn(seed + 1, [k, k], { std: 0.8 }).data;
  cur = { X, W, n, k };
  const p = st.pad | 0, s = st.stride | 0, dil = st.dilation | 0;
  const Hout = Math.max(1, Math.floor((n + 2 * p - dil * (k - 1) - 1) / s) + 1);
  const steps = [];
  for (let oy = 0; oy < Hout; oy++) for (let ox = 0; ox < Hout; ox++) steps.push({ oy, ox, label: `output (${oy},${ox}) = Σ kernel·field` });
  return steps;
}

function conv(st) {
  const { X, W, n, k } = cur, p = st.pad | 0, s = st.stride | 0, dil = st.dilation | 0;
  const Hpad = n + 2 * p, Hout = Math.max(1, Math.floor((n + 2 * p - dil * (k - 1) - 1) / s) + 1);
  const xpad = (r, c) => (r >= p && r < p + n && c >= p && c < p + n) ? X[(r - p) * n + (c - p)] : 0;
  const out = new Float32Array(Hout * Hout);
  for (let oy = 0; oy < Hout; oy++) for (let ox = 0; ox < Hout; ox++) { let sum = 0; for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) sum += W[i * k + j] * xpad(oy * s + i * dil, ox * s + j * dil); out[oy * Hout + ox] = sum; }
  return { out, Hpad, Hout, xpad, p, s, dil, k, n };
}

mount({
  mount: 'body',
  title: 'convolution — kernel · receptive field, summed',
  blurb: 'A k×k kernel slides over the (zero-padded) input; each output pixel is a sum of products: out[oy,ox] = Σ W[i,j]·Xpad[oy·s + i·dil, ox·s + j·dil]. stride s sets how far the kernel jumps (bigger ⇒ smaller output); padding p adds a zero border (keeps the size up); dilation spreads the kernel taps apart (bigger receptive field, same weights). Output size = ⌊(n+2p − dil·(k−1) − 1)/s⌋ + 1. The current receptive field is outlined with the kernel weights overlaid; the output fills as the kernel slides. Drag an input cell to change it; hover to inspect.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  challenges: [
    { goal: '"Same" convolution — make the output the same size as the input.', hint: 'with stride 1 + dilation 1, set padding p = (k−1)/2.', check: (api) => ({ solved: (api.probe.Hout ?? -1) === (api.probe.n ?? 0), detail: `output ${api.probe.Hout}×${api.probe.Hout} vs input ${api.probe.n}×${api.probe.n}` }) },
    { goal: 'Downsample by 2 — make the output about half the input.', hint: 'set stride s = 2.', check: (api) => { const n = api.probe.n ?? 0, h = api.probe.Hout ?? 0; return { solved: h > 0 && (h === Math.ceil(n / 2) || h === Math.floor(n / 2)), detail: `output ${h} (≈ n/2 = ${Math.round(n / 2)})` }; } },
  ],
  controls: (c, page) => {
    c.stepper('n', { label: 'input size (n×n)', min: 4, max: 7, value: 5 });
    c.stepper('k', { label: 'kernel size (k)', min: 2, max: 4, value: 3 });
    c.stepper('stride', { label: 'stride s', min: 1, max: 3, value: 1 });
    c.stepper('pad', { label: 'padding p', min: 0, max: 2, value: 1 });
    c.stepper('dilation', { label: 'dilation', min: 1, max: 3, value: 1 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 2, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !inRect) return;
    const Hpad = cur.n + 2 * geomP;
    const cellOf = (x, y) => { const c = Math.floor((x - inRect.x) / ic), r = Math.floor((y - inRect.y) / ic); return (r >= geomP && r < geomP + cur.n && c >= geomP && c < geomP + cur.n) ? { r: r - geomP, c: c - geomP } : null; };
    if (ev.type === 'down') grab = cellOf(ev.x, ev.y);
    else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab && page.pointer.down) { const i = grab.r * cur.n + grab.c; cur.X[i] = Math.max(-3, Math.min(3, cur.X[i] - ev.dy * 0.02)); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { X, W, n, k } = cur;
    r.clear('#ffffff');
    const C = conv(st), { out, Hpad, Hout, p, s, dil } = C; geomP = p;
    page.probe = { Hout, n: cur.n };
    const sp = page.step(), oy = sp ? sp.oy : Hout - 1, ox = sp ? sp.ox : Hout - 1, ostep = sp ? (oy * Hout + ox) : Hout * Hout - 1;
    const xdom = Math.max(maxAbs(X), 0.5), wdom = Math.max(maxAbs(W), 0.3), odom = Math.max(maxAbs(out), 0.5);

    const pad = 16, topY = 64;
    ic = Math.max(16, Math.min(34, Math.min((page.W * 0.34) / Hpad, (page.H * 0.56) / Hpad)));
    inRect = { x: pad + 24, y: topY, w: Hpad * ic, h: Hpad * ic };
    r.label(`input X  (n=${n}, pad ${p})  — drag a cell ↕`, inRect.x, topY - 12, { color: INK, font: '11px ui-monospace, monospace' });

    // padded input grid
    ctx.save(); ctx.font = `${Math.max(8, ic * 0.34)}px ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let rr = 0; rr < Hpad; rr++) for (let cc = 0; cc < Hpad; cc++) {
      const x = inRect.x + cc * ic, y = inRect.y + rr * ic, isPad = !(rr >= p && rr < p + n && cc >= p && cc < p + n);
      const v = isPad ? 0 : X[(rr - p) * n + (cc - p)];
      ctx.fillStyle = isPad ? '#f4f5f6' : divcol(v, xdom); ctx.fillRect(x, y, ic - 1, ic - 1);
      if (isPad) { ctx.strokeStyle = '#e0e3e6'; ctx.strokeRect(x + 0.5, y + 0.5, ic - 2, ic - 2); ctx.fillStyle = '#c4ccd3'; ctx.fillText('0', x + ic / 2, y + ic / 2); }
      else { ctx.fillStyle = '#33383d'; ctx.fillText(v.toFixed(1), x + ic / 2, y + ic / 2); }
    }
    // receptive field for (oy,ox): the dilated kernel taps, with weights overlaid
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const rr = oy * s + i * dil, cc = ox * s + j * dil, x = inRect.x + cc * ic, y = inRect.y + rr * ic;
      ctx.strokeStyle = BLUE; ctx.lineWidth = 2.4; ctx.strokeRect(x + 1, y + 1, ic - 3, ic - 3);
      ctx.fillStyle = ORANGE; ctx.font = `${Math.max(7, ic * 0.26)}px ui-monospace, monospace`; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(W[i * k + j].toFixed(1), x + ic - 2, y + 1);
    }
    ctx.restore();

    // kernel legend
    const kc = Math.max(16, ic * 0.8), kx = inRect.x + Hpad * ic + 40, ky = topY + 6;
    kRect = { x: kx, y: ky, w: k * kc, h: k * kc };
    r.label('kernel W (orange)', kx, ky - 12, { color: ORANGE, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.font = `${Math.max(8, kc * 0.34)}px ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { const x = kx + j * kc, y = ky + i * kc; ctx.fillStyle = divcol(W[i * k + j], wdom); ctx.fillRect(x, y, kc - 1, kc - 1); ctx.strokeStyle = 'rgba(210,105,30,0.5)'; ctx.strokeRect(x + 0.5, y + 0.5, kc - 1, kc - 1); ctx.fillStyle = '#33383d'; ctx.fillText(W[i * k + j].toFixed(1), x + kc / 2, y + kc / 2); }
    ctx.restore();

    // output grid
    oc = Math.max(18, Math.min(40, Math.min((page.W * 0.20) / Hout, (page.H * 0.5) / Hout)));
    const outX = kx + k * kc + 60, outY = topY + 6;
    outRect = { x: outX, y: outY, w: Hout * oc, h: Hout * oc };
    r.label(`output  ${Hout}×${Hout}`, outX, outY - 12, { color: GREEN, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.font = `${Math.max(8, oc * 0.32)}px ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let yy = 0; yy < Hout; yy++) for (let xx = 0; xx < Hout; xx++) {
      const x = outX + xx * oc, y = outY + yy * oc, idx = yy * Hout + xx, done = idx <= ostep;
      ctx.fillStyle = done ? divcol(out[idx], odom) : '#f4f5f6'; ctx.fillRect(x, y, oc - 1, oc - 1);
      ctx.strokeStyle = '#e0e3e6'; ctx.strokeRect(x + 0.5, y + 0.5, oc - 1, oc - 1);
      if (done) { ctx.fillStyle = '#33383d'; ctx.fillText(out[idx].toFixed(1), x + oc / 2, y + oc / 2); }
      if (yy === oy && xx === ox) { ctx.strokeStyle = BLUE; ctx.lineWidth = 2.6; ctx.strokeRect(x + 1, y + 1, oc - 2, oc - 2); }
    }
    ctx.restore();
    // arrow kernel-field -> current output
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(kx + k * kc + 6, ky + k * kc / 2); ctx.lineTo(outX - 6, outY + oy * oc + oc / 2); ctx.stroke(); ctx.restore();

    // sum-of-products readout for the current pixel
    let terms = '', acc = 0; const xpad = C.xpad;
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { const w = W[i * k + j], xv = xpad(oy * s + i * dil, ox * s + j * dil); acc += w * xv; if (i * k + j < 4) terms += `${i + j === 0 ? '' : ' + '}${w.toFixed(1)}·${xv.toFixed(1)}`; }
    r.label(`out[${oy},${ox}] = Σ W⊙field = ${terms}${k * k > 4 ? ' + …' : ''} = ${acc.toFixed(2)}   (${k * k} products)`, inRect.x, inRect.y + Hpad * ic + 24, { color: INK, font: '12px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      const incc = Math.floor((pt.x - inRect.x) / ic), inrr = Math.floor((pt.y - inRect.y) / ic);
      if (inrr >= 0 && inrr < Hpad && incc >= 0 && incc < Hpad && pt.x >= inRect.x && pt.y >= inRect.y) {
        const isPad = !(inrr >= p && inrr < p + n && incc >= p && incc < p + n);
        page.setTip(isPad ? `padding cell = 0\n(zero border, p=${p})` : `X[${inrr - p},${incc - p}] = ${X[(inrr - p) * n + (incc - p)].toFixed(3)}\ndrag ↕ to change`);
      } else {
        const oxx = Math.floor((pt.x - outX) / oc), oyy = Math.floor((pt.y - outY) / oc);
        if (oyy >= 0 && oyy < Hout && oxx >= 0 && oxx < Hout && pt.x >= outX && pt.y >= outY) page.setTip(`out[${oyy},${oxx}] = ${out[oyy * Hout + oxx].toFixed(3)}\nΣ of ${k * k} kernel·field products`);
        else { const kj = Math.floor((pt.x - kx) / kc), ki = Math.floor((pt.y - ky) / kc); if (ki >= 0 && ki < k && kj >= 0 && kj < k && pt.x >= kx && pt.y >= ky) page.setTip(`W[${ki},${kj}] = ${W[ki * k + kj].toFixed(3)}\nkernel weight`); }
      }
    }

    let o = `2-D conv: out[oy,ox] = Σ W[i,j]·Xpad[oy·s+i·dil, ox·s+j·dil].   n=${n} k=${k} stride=${s} pad=${p} dil=${dil} → output ${Hout}×${Hout}.    tier:${r.name}\n`;
    o += sp ? `sliding: kernel at output (${oy},${ox}); receptive field outlined; sum of ${k * k} products = ${acc.toFixed(2)}.`
      : `output ${Hout}×${Hout} = ⌊(${n}+2·${p} − ${dil}·(${k}−1) − 1)/${s}⌋+1. Drag an input cell to change the affected outputs.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__convPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // geometry hooks (rebuild the transport so the output dims/step count update)
  let reb = false;
  for (const [qk, ck] of [['n', 'n'], ['k', 'k'], ['stride', 'stride'], ['pad', 'pad'], ['dil', 'dilation']]) if (q.has(qk)) { page.controls.set(ck, parseInt(q.get(qk), 10)); reb = true; }
  if (reb && t) t.rebuild();
  if (q.has('drag')) { const [rr, cc, v] = q.get('drag').split(',').map(Number); if (cur && rr * cur.n + cc < cur.X.length) cur.X[rr * cur.n + cc] = v; }   // after rebuild (cur is fresh)
  if (q.has('step') || q.has('drag') || q.has('hover') || reb) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
