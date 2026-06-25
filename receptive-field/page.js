// receptive-field concept page -- how stacking (and dilating) conv layers grows
// the patch of input a single output unit depends on. A 1-D stack of conv layers
// (input bottom, output top); each unit reads k units below, so its dependency
// CONE widens downward. RF = 1 + Σₗ (k−1)·dₗ (stride 1). Doubling dilation
// (WaveNet) makes the RF grow exponentially. Drag the layer slider to stack
// more; hover a unit to focus its cone. No transport -- the focus sweeps the
// top layer (animate); hover overrides.
import { mount } from '../framework/layout.js';

const INK = '#111', BLUE = '#1f6feb', RED = '#d1242f', GREY = '#9aa4ad';

mount({
  mount: 'body',
  title: 'receptive-field — stacked convs grow what each output sees',
  blurb: 'The receptive field is the patch of input a single output unit depends on. Each conv layer reads k units below it, so a unit\'s dependency cone widens downward; the receptive field is how wide that cone is at the input: RF = 1 + Σₗ (k−1)·dₗ. Stacking depth grows it (each 3×3 layer adds k−1=2, so a deep stack of cheap small kernels reaches a wide RF — two 3×3 ≈ one 5×5). Dilation grows it for the same k; doubling dilation (dₗ=2^(l−1), WaveNet/TCN) makes the RF grow exponentially. Drag "conv layers" to stack more and watch the cone + RF grow; hover a unit to focus its cone.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'input width', min: 11, max: 27, step: 2, value: 19 });
    c.slider('L', { label: 'conv layers', min: 1, max: 6, step: 1, value: 4 });
    c.stepper('k', { label: 'kernel k', min: 3, max: 5, step: 2, value: 3 });
    c.select('mode', { label: 'dilation', options: ['fixed', 'double'], value: 'fixed' });
    c.slider('dil', { label: 'dilation (fixed)', min: 1, max: 3, step: 1, value: 1 });
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear('#ffffff');
    const N = st.N | 0, L = st.L | 0, k = st.k | 0, half = (k - 1) / 2;
    const dilAt = (l) => st.mode === 'double' ? Math.pow(2, l - 1) : (st.dil | 0);
    let rf = 1; for (let l = 1; l <= L; l++) rf += (k - 1) * dilAt(l);

    const pad = 16, topY = 76, leftX = pad + 46, stackW = page.W - leftX - pad - 20;
    const dx = stackW / (N - 1), baseY = page.H - pad - 54, dy = (baseY - topY) / L;
    const nx = (pos, l) => ({ x: leftX + pos * dx, y: baseY - l * dy });
    const clamp = (v) => Math.max(0, Math.min(N - 1, v));

    // focus: hovered unit, else the top-layer sweep
    let fl = L, fp = Math.round((N - 1) / 2 + (N - 1) / 4 * Math.sin((page.t || 0) * 0.55));
    fp = clamp(fp);
    let hov = null;
    if (page.pointer.over) { const p = page.pointer; let best = Math.min(dx, dy) * 0.6 + 7; for (let l = 0; l <= L; l++) for (let pos = 0; pos < N; pos++) { const c = nx(pos, l), dd = Math.hypot(p.x - c.x, p.y - c.y); if (dd < best) { best = dd; hov = { l, pos }; } } }
    if (hov) { fl = hov.l; fp = hov.pos; }

    // receptive-field interval per layer (top-down)
    const lo = [], hi = []; lo[fl] = fp; hi[fl] = fp;
    for (let l = fl; l >= 1; l--) { const d = dilAt(l); lo[l - 1] = lo[l] - half * d; hi[l - 1] = hi[l] + half * d; }

    // all units (faint) + layer labels
    ctx.save();
    for (let l = 0; l <= L; l++) for (let pos = 0; pos < N; pos++) { const c = nx(pos, l); ctx.fillStyle = '#dfe3e6'; ctx.beginPath(); ctx.arc(c.x, c.y, 2.5, 0, 7); ctx.fill(); }
    ctx.fillStyle = '#586069'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let l = 0; l <= L; l++) ctx.fillText(l === 0 ? 'input' : `conv ${l}`, leftX - 8, nx(0, l).y);
    ctx.restore();

    // dependency cone (translucent) from focus down to the input RF interval
    ctx.save(); ctx.fillStyle = 'rgba(31,111,235,0.12)'; ctx.strokeStyle = 'rgba(31,111,235,0.55)'; ctx.lineWidth = 1.2; ctx.beginPath();
    const f = nx(fp, fl); ctx.moveTo(f.x, f.y);
    for (let l = fl - 1; l >= 0; l--) { const c = nx(clamp(hi[l]), l); ctx.lineTo(c.x, c.y); }
    for (let l = 0; l <= fl; l++) { const c = nx(clamp(lo[l]), l); ctx.lineTo(c.x, c.y); }
    ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();

    // highlight RF units per layer + the focus
    ctx.save();
    for (let l = 0; l <= fl; l++) for (let pos = clamp(lo[l]); pos <= clamp(hi[l]); pos++) { const c = nx(pos, l); ctx.fillStyle = l === 0 ? BLUE : 'rgba(31,111,235,0.72)'; ctx.beginPath(); ctx.arc(c.x, c.y, l === 0 ? 3.6 : 3, 0, 7); ctx.fill(); }
    ctx.fillStyle = RED; ctx.beginPath(); ctx.arc(f.x, f.y, 4.6, 0, 7); ctx.fill();
    ctx.restore();

    // input RF bracket
    const a = nx(clamp(lo[0]), 0), b = nx(clamp(hi[0]), 0), by = baseY + 16;
    ctx.save(); ctx.strokeStyle = BLUE; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(a.x - 3, by); ctx.lineTo(a.x - 3, by + 6); ctx.lineTo(b.x + 3, by + 6); ctx.lineTo(b.x + 3, by); ctx.stroke();
    ctx.fillStyle = BLUE; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`receptive field = ${rf} input unit${rf === 1 ? '' : 's'}${rf > N ? ' (wider than shown)' : ''}`, (a.x + b.x) / 2, by + 9); ctx.restore();

    r.label(`each "conv ${L}" unit sees ${rf} inputs    ·    RF = 1 + Σₗ (k−1)·dₗ`, leftX, topY - 48, { color: INK, font: '12px ui-monospace, monospace' });
    r.label(st.mode === 'double'
      ? `doubling dilation dₗ = 2^(l−1) → RF grows EXPONENTIALLY:  1 + (k−1)(2^L − 1) = ${rf}`
      : `fixed dilation d=${st.dil} → RF grows LINEARLY with depth:  1 + L·(k−1)·d = ${rf}`, leftX, topY - 32, { color: '#586069', font: '10px ui-monospace, monospace' });

    if (hov) page.setTip(`unit (conv ${hov.l}, pos ${hov.pos})\nreceptive field on input = ${rf} unit${rf === 1 ? '' : 's'}\n(cone widens k−1=${k - 1} per layer × dilation)`);

    let o = `receptive field: stacking convs grows what each output sees.  ${L} layers · kernel ${k} · ${st.mode === 'double' ? 'doubling' : 'fixed d=' + st.dil} dilation → RF = ${rf} input units.    tier:${r.name}\n`;
    o += st.mode === 'double'
      ? `doubling dilation makes the RF blow up exponentially (WaveNet/TCN): 1 + (k−1)(2^L−1). Drag "conv layers"; hover a unit to focus its cone.`
      : `each extra layer adds (k−1)·d = ${(k - 1) * st.dil} to the RF — a deep stack of small kernels reaches a wide RF cheaply. Drag "conv layers"; hover a unit.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__rfPage = page;
  const q = new URLSearchParams(location.search);
  for (const [qk, ck] of [['L', 'L'], ['k', 'k'], ['dil', 'dil'], ['N', 'N']]) if (q.has(qk)) page.controls.set(ck, parseInt(q.get(qk), 10));
  if (q.has('mode')) page.controls.set('mode', q.get('mode'));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
