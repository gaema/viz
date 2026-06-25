// ssm-scan concept page -- the selective state-space scan (Mamba / S6).
// A scalar sequence x[L] is scanned into an N-dim recurrent state:
//   Δₜ = softplus(bias + selectivity·xₜ)          (input-dependent timestep)
//   Āₜ,ₙ = exp(Δₜ·Aₙ),  B̄ₜ,ₙ = (Āₜ,ₙ−1)/Aₙ·Bₙ     (discretize)
//   hₜ,ₙ = Āₜ,ₙ·hₜ₋₁,ₙ + B̄ₜ,ₙ·xₜ                  (recurrence)
//   yₜ = Σₙ Cₙ·hₜ,ₙ
// Panels share a time axis: input x, the selective Δ, the [N×L] state
// trajectory, output y, + a per-step recurrence inset. Interactive: drag an
// input bar to re-scan; N (state size) + L (sequence length) steppers; hover.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', PURPLE = '#8957e5', ORANGE = '#d2691e', GREY = '#9aa4ad';
const softplus = (x) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

let cur = null;
let xBand = null, dBand = null, hRect = null, yBand = null, cw = 0;   // captured in draw
let grab = null;                                                      // input column t while dragging

function buildData(st) {
  const N = st.N | 0, L = st.L | 0, seed = st.seed | 0;
  const A = Float32Array.from({ length: N }, (_, n) => -(0.25 + 1.7 * n / Math.max(1, N - 1)));  // decay rates: slow (dim 0) -> fast
  const B = new Float32Array(N).fill(1);
  const C = seededRandn(seed + 7, N, { std: 1 });
  const x = seededRandn(seed, L, { std: 0.9 });
  cur = { x, A, B, C, N, L };
  return Array.from({ length: L }, (_, t) => ({ t, label: `step ${t}: read xₜ, update the state, emit yₜ` }));
}

// Full selective scan from the (editable) cur.x + cur.A/B/C.
function scan(st) {
  const { x, A, B, C, N, L } = cur, sel = st.sel, bias = st.dbias;
  const H = new Float32Array(N * L), D = new Float32Array(L), Y = new Float32Array(L), Ab = new Float32Array(N * L);
  const hp = new Float32Array(N);
  for (let t = 0; t < L; t++) {
    const dt = softplus(bias + sel * x[t]); D[t] = dt;
    let y = 0;
    for (let n = 0; n < N; n++) {
      const Abar = Math.exp(dt * A[n]), Bbar = (Abar - 1) / A[n] * B[n];
      const h = Abar * hp[n] + Bbar * x[t];
      hp[n] = h; H[n * L + t] = h; Ab[n * L + t] = Abar; y += C[n] * h;
    }
    Y[t] = y;
  }
  return { H, D, Y, Ab };
}

function sbar(ctx, cx, mid, half, v, vmax, color, faded) {            // signed bar from a mid baseline
  const h = (v / (vmax || 1)) * half;
  ctx.save(); ctx.globalAlpha = faded ? 0.22 : 1; ctx.fillStyle = color;
  ctx.fillRect(cx - cw * 0.32, h >= 0 ? mid - h : mid, cw * 0.64, Math.abs(h)); ctx.restore();
}

mount({
  mount: 'body',
  title: 'ssm-scan — the selective state-space scan',
  blurb: 'A selective SSM (Mamba) carries a recurrent state through the sequence. Each step: a content-dependent timestep Δₜ = softplus(bias + selectivity·xₜ) sets the per-dim decay Āₜ = exp(Δₜ·A) and input gain B̄ₜ; the state updates hₜ = Āₜ⊙hₜ₋₁ + B̄ₜ·xₜ and emits yₜ = C·hₜ. Big Δ on a salient token writes it into the state; small Δ holds the state (memory). Slow dims (small |A|) remember long, fast dims forget. Drag an input bar to re-scan from that point; set state size N and sequence length L; step the scan (auto-plays + loops).',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'state size (N)', min: 2, max: 8, value: 6 });
    c.stepper('L', { label: 'sequence length (L)', min: 4, max: 14, value: 10 });
    c.slider('sel', { label: 'selectivity (Δ←x)', min: 0, max: 2, step: 0.1, value: 1 });
    c.slider('dbias', { label: 'Δ bias', min: -0.5, max: 2, step: 0.1, value: 0.6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.3, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !xBand) return;
    const L = cur.L;
    if (ev.type === 'down') { const t = Math.floor((ev.x - xBand.x) / cw); grab = (t >= 0 && t < L && ev.y >= xBand.y && ev.y <= xBand.y + xBand.h) ? t : null; }
    else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab !== null && page.pointer.down) {
      cur.x[grab] = Math.max(-3, Math.min(3, cur.x[grab] - ev.dy * 0.02));
      page.redraw();
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;                       // built by the transport compute (rebuilds on N/L/seed)
    const { N, L } = cur;
    r.clear('#ffffff');
    const S = scan(st);
    const s = page.step(), tcur = s ? s.t : L - 1;                     // scan revealed up to here

    const pad = 16, lab = 38, insetW = 196;
    const barsX = pad + lab, barsW = page.W - barsX - insetW - pad;
    cw = barsW / L;
    const xmax = Math.max(maxAbs(cur.x), 0.5), dmax = Math.max(maxAbs(S.D), 0.5), ymax = Math.max(maxAbs(S.Y), 0.5), hmax = maxAbs(S.H);
    const ch = Math.max(11, Math.min(28, (page.H * 0.40) / N));
    let y = 52;
    xBand = { x: barsX, y, h: 32 }; const xMid = y + 16;
    dBand = { x: barsX, y: y + 50, h: 24 };
    hRect = { x: barsX, y: dBand.y + dBand.h + 22, w: L * cw, h: N * ch };
    yBand = { x: barsX, y: hRect.y + N * ch + 22, h: 32 }; const yMid = yBand.y + 16;

    // row labels
    r.label('xₜ', pad, xMid + 4, { color: BLUE, font: '12px ui-monospace, monospace' });
    r.label('Δₜ', pad, dBand.y + 16, { color: ORANGE, font: '12px ui-monospace, monospace' });
    r.label('state', pad - 2, hRect.y - 8, { color: INK, font: '11px ui-monospace, monospace' });
    r.label('hₙ', pad, hRect.y + N * ch / 2, { color: INK, font: '12px ui-monospace, monospace' });
    r.label('yₜ', pad, yMid + 4, { color: GREEN, font: '12px ui-monospace, monospace' });

    // current-step column highlight (down all bands)
    ctx.save(); ctx.fillStyle = 'rgba(31,111,235,0.07)'; ctx.fillRect(barsX + tcur * cw, xBand.y - 4, cw, (yBand.y + 32) - (xBand.y - 4)); ctx.restore();

    // input x bars (draggable) + zero lines
    ctx.save(); ctx.strokeStyle = '#e7e9ec'; ctx.beginPath(); ctx.moveTo(barsX, xMid); ctx.lineTo(barsX + L * cw, xMid); ctx.moveTo(barsX, yMid); ctx.lineTo(barsX + L * cw, yMid); ctx.stroke(); ctx.restore();
    for (let t = 0; t < L; t++) sbar(ctx, barsX + t * cw + cw / 2, xMid, 14, cur.x[t], xmax, BLUE, t > tcur && false);
    // Δ bars (positive, from band bottom), revealed up to tcur
    const dBase = dBand.y + dBand.h;
    for (let t = 0; t < L; t++) { const h = (S.D[t] / dmax) * dBand.h; ctx.save(); ctx.globalAlpha = t > tcur ? 0.2 : 1; ctx.fillStyle = ORANGE; ctx.fillRect(barsX + t * cw + cw * 0.18, dBase - h, cw * 0.64, h); ctx.restore(); }
    // state trajectory heatmap [N×L]
    r.heatmap({ data: S.H, rows: N, cols: L }, { rows: N, cols: L, rect: hRect, ramp: ramps.diverging, domain: [-hmax, hmax] });
    r.grid({ stroke: 'rgba(0,0,0,0.08)' });
    if (tcur < L - 1) { ctx.save(); ctx.fillStyle = 'rgba(255,255,255,0.66)'; ctx.fillRect(barsX + (tcur + 1) * cw, hRect.y, (L - 1 - tcur) * cw, N * ch); ctx.restore(); }   // veil future
    // y output bars
    for (let t = 0; t < L; t++) sbar(ctx, barsX + t * cw + cw / 2, yMid, 14, S.Y[t], ymax, GREEN, t > tcur);
    // time-axis ticks
    ctx.save(); ctx.fillStyle = GREY; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let t = 0; t < L; t++) ctx.fillText(String(t), barsX + t * cw + cw / 2, yBand.y + 34);
    ctx.fillStyle = GREEN; ctx.fillText('drag xₜ ↕', barsX + cw / 2, xBand.y - 16); ctx.restore();

    // ---- recurrence inset for the current step ----
    const ix = barsX + barsW + 20, iy = 50, iw = insetW - 8;
    const dt = S.D[tcur], salient = dt > softplus(st.dbias) * 1.15;
    r.label(`step ${tcur}:  hₜ = Ā⊙hₜ₋₁ + B̄·xₜ`, ix, iy, { color: INK, font: '11px ui-monospace, monospace' });
    r.label(`Δ=${dt.toFixed(2)}  →  ${salient ? 'WRITE (capture xₜ)' : 'hold (retain state)'}`, ix, iy + 16, { color: salient ? BLUE : GREY, font: '10px ui-monospace, monospace' });
    r.label('dim   Ā (retain)        hₜ', ix, iy + 36, { color: '#586069', font: '10px ui-monospace, monospace' });
    ctx.save(); ctx.font = '10px ui-monospace, monospace';
    const rowH = Math.min(20, (page.H - iy - 70) / N), aw = 70;
    for (let n = 0; n < N; n++) {
      const ry = iy + 50 + n * rowH, Abar = S.Ab[n * L + tcur], hv = S.H[n * L + tcur];
      ctx.fillStyle = '#3a4047'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(`h${n}`, ix, ry);
      ctx.strokeStyle = '#e3e6ea'; ctx.strokeRect(ix + 26, ry - 5, aw, 10);
      ctx.fillStyle = 'rgba(137,87,229,0.7)'; ctx.fillRect(ix + 26, ry - 5, aw * Abar, 10);   // retention Ā in (0,1)
      ctx.fillStyle = '#586069'; ctx.fillText(Abar.toFixed(2), ix + 26 + aw + 4, ry);
      ctx.fillStyle = hv >= 0 ? BLUE : ORANGE; ctx.textAlign = 'right'; ctx.fillText(hv.toFixed(2), ix + iw, ry);
    }
    ctx.restore();
    r.label(`yₜ = C·hₜ = ${S.Y[tcur].toFixed(3)}`, ix, iy + 56 + N * rowH, { color: GREEN, font: '11px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && grab === null) {
      const p = page.pointer;
      const hh = cellAt(hRect, N, L, p.x, p.y);
      if (hh) page.setTip(`h[dim ${hh.r}, t${hh.c}] = ${S.H[hh.r * L + hh.c].toFixed(3)}\nĀ=${S.Ab[hh.r * L + hh.c].toFixed(2)} (retain), A=${cur.A[hh.r].toFixed(2)}`);
      else { const t = Math.floor((p.x - barsX) / cw); if (t >= 0 && t < L) {
        if (p.y >= xBand.y && p.y <= xBand.y + xBand.h) page.setTip(`x[${t}] = ${cur.x[t].toFixed(3)}\nΔ=${S.D[t].toFixed(2)}  ·  drag ↕`);
        else if (p.y >= dBand.y && p.y <= dBase) page.setTip(`Δ[${t}] = softplus(${st.dbias.toFixed(1)} + ${st.sel.toFixed(1)}·${cur.x[t].toFixed(2)}) = ${S.D[t].toFixed(3)}`);
        else if (p.y >= yBand.y && p.y <= yBand.y + yBand.h) page.setTip(`y[${t}] = C·h_${t} = ${S.Y[t].toFixed(3)}`);
      } }
    }

    let o = `selective scan: Δₜ=softplus(bias+sel·xₜ) gates write-vs-hold; hₜ=Ā⊙hₜ₋₁+B̄·xₜ; yₜ=C·hₜ.   sel=${st.sel.toFixed(1)} ${st.sel < 0.1 ? '(non-selective linear SSM)' : '(selective)'}    tier:${r.name}\n`;
    o += s ? `step ${tcur}/${L - 1}: Δ=${dt.toFixed(2)} → ${salient ? 'writes xₜ into the state' : 'holds the state'}; yₜ=${S.Y[tcur].toFixed(3)}.`
      : `${L} steps scanned. Slow dims (small |A|) keep long memory; large Δ writes salient inputs, small Δ retains.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__ssmPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('sel')) page.controls.set('sel', parseFloat(q.get('sel')));
  if (q.has('drag')) { const [i, v] = q.get('drag').split(',').map(Number); if (cur && i < cur.x.length) cur.x[i] = v; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('drag') || q.has('hover') || q.has('sel')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
