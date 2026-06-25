// batchnorm concept page -- normalize a channel's activations, TRACK a running
// mean/variance during training (EMA over noisy batches), and at INFERENCE use
// that frozen running estimate instead of the batch's own stats:
//   y = γ·(x − μ)/√(σ²+ε) + β,   μ,σ = batch (train) vs running (inference)
// The training EMA smooths jittery batch stats toward the true channel mean; at
// inference a brighter-than-average input comes out brighter (not recentered).
// Paint the input; toggle train/inference. No transport -- the EMA is animated.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', RED = '#d1242f', GREY = '#9aa4ad';
const N = 12, EPS = 1e-3;
const mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
const variance = (a, m) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return s / a.length; };
const divcol = (v, lo, hi) => { const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))); return `rgb(${Math.round(255 - t * 30)},${Math.round(255 - t * 120)},${Math.round(255 - t * 200)})`; };  // light->blue
const signcol = (v, dom) => { const t = Math.max(-1, Math.min(1, v / (dom || 1))), m = Math.abs(t); return t >= 0 ? `rgb(255,${Math.round(255 - m * 150)},${Math.round(255 - m * 165)})` : `rgb(${Math.round(255 - m * 165)},${Math.round(255 - m * 120)},255)`; };

let cur = null, bsig = '';
let rmu = 0, rvar = 1, hist = [], lastT = 0, rng = 1, lastB = { bmu: 0, bvar: 1 };
let inRect = null, painting = false;

function nextRand() { rng = (rng + 0x6D2B79F5) | 0; let t = Math.imul(rng ^ (rng >>> 15), 1 | rng); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }

function buildInput(st) {
  const X = new Float32Array(N * N), c = (N - 1) / 2, rnd = seededRandn(st.seed | 0, [N, N], { std: 1 }).data;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const dx = x - c, dy = y - c, r = Math.hypot(dx, dy); let v;
    if (st.shape === 'bright') v = 0.68 + 0.18 * (rnd[y * N + x] > 0 ? 1 : 0.3);
    else if (st.shape === 'dark') v = 0.10 + 0.16 * (rnd[y * N + x] > 0 ? 1 : 0.3);
    else if (st.shape === 'gradient') v = 0.1 + 0.8 * (x / (N - 1));
    else if (st.shape === 'blob') v = Math.max(0.05, 0.95 - r * 0.12);
    else v = 0.2 + 0.6 * (rnd[y * N + x] * 0.5 + 0.5);
    X[y * N + x] = Math.max(0, Math.min(1, v));
  }
  return { X };
}

mount({
  mount: 'body',
  title: 'batchnorm — running stats, train vs inference',
  blurb: 'Batch Normalization normalizes each channel: y = γ·(x − μ)/√(σ²+ε) + β. The catch is which μ, σ it uses. In TRAINING it uses the current batch\'s mean/variance (each batch recentered to ~0) and updates a RUNNING estimate by EMA: running_μ ← (1−m)·running_μ + m·batch_μ — real batches are noisy, so this running average smooths the jitter toward the true channel mean. At INFERENCE there\'s no batch, so BN uses the frozen running stats it learned: a brighter-than-average input then comes out brighter (measured against the population), not recentered. Paint the input and watch the running average track it; toggle train ↔ inference on a bright input to see the output shift; tune momentum / γ / β.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  compare: { key: 'mode', a: 'train', b: 'inference', labelA: 'train — uses this batch\'s stats', labelB: 'inference — frozen running stats' },
  animate: true,
  controls: (c, page) => {
    c.select('shape', { label: 'input', options: ['bright', 'dark', 'gradient', 'blob', 'random'], value: 'bright', rebuild: true });
    c.select('mode', { label: 'mode', options: ['train', 'inference'], value: 'train' });
    c.slider('momentum', { label: 'momentum (EMA)', min: 0.02, max: 0.3, step: 0.02, value: 0.1 });
    c.slider('gamma', { label: 'γ scale', min: 0, max: 2, step: 0.1, value: 1 });
    c.slider('beta', { label: 'β shift', min: -1, max: 1, step: 0.1, value: 0 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !inRect) return;
    const px = inRect.w / N;
    const at = (x, y) => { const c = Math.floor((x - inRect.x) / px), r = Math.floor((y - inRect.y) / px); return (r >= 0 && r < N && c >= 0 && c < N && x >= inRect.x && y >= inRect.y) ? r * N + c : -1; };
    if (ev.type === 'down') { painting = at(ev.x, ev.y) >= 0; if (painting) { cur.X[at(ev.x, ev.y)] = 1; page.redraw(); } }
    else if (ev.type === 'up' || ev.type === 'leave') painting = false;
    else if (ev.type === 'move' && painting && page.pointer.down) { const i = at(ev.x, ev.y); if (i >= 0) { cur.X[i] = 1; page.redraw(); } }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const sig = `${st.shape}|${st.seed}`;
    if (sig !== bsig) { cur = buildInput(st); const pm0 = mean(cur.X); rmu = pm0 * 0.35; rvar = Math.max(0.05, variance(cur.X, pm0)); hist = [{ b: rmu, r: rmu }]; lastT = page.t || 0; rng = (st.seed | 0) + 1; lastB = { bmu: rmu, bvar: rvar }; bsig = sig; }
    r.clear('#ffffff');
    const X = cur.X, popMu = mean(X), popVar = variance(X, popMu), m = st.momentum, train = st.mode === 'train';

    // training step (EMA) -- only in train mode, gated ~6/sec
    if (train && (page.t || 0) - lastT > 0.16) {
      lastT = page.t;
      const bmu = popMu + (nextRand() * 2 - 1) * 0.32, bvar = Math.max(0.01, popVar * (1 + (nextRand() * 2 - 1) * 0.4));
      rmu = (1 - m) * rmu + m * bmu; rvar = (1 - m) * rvar + m * bvar; lastB = { bmu, bvar };
      hist.push({ b: bmu, r: rmu }); if (hist.length > 64) hist.shift();
    }

    // normalize: train uses the input's batch stats, inference uses frozen running
    const nMu = train ? popMu : rmu, nVar = train ? popVar : rvar;
    const out = Float32Array.from(X, (x) => st.gamma * (x - nMu) / Math.sqrt(nVar + EPS) + st.beta);
    const odom = Math.max(1, Math.abs(st.gamma) * 2);

    const pad = 16, topY = 50, cell = Math.max(8, Math.min(12, (page.H * 0.40) / N)), gw = N * cell;
    // input (paintable)
    const ix = pad + 8;
    inRect = { x: ix, y: topY, w: gw, h: gw };
    r.label('input x — paint ↕', ix, topY - 12, { color: INK, font: '11px ui-monospace, monospace' });
    ctx.save(); for (let i = 0; i < N * N; i++) { ctx.fillStyle = divcol(X[i], 0, 1); ctx.fillRect(ix + (i % N) * cell, topY + ((i / N) | 0) * cell, cell - 0.5, cell - 0.5); } ctx.restore();
    r.label(`batch: μ=${popMu.toFixed(2)}  σ=${Math.sqrt(popVar).toFixed(2)}`, ix, topY + gw + 14, { color: '#586069', font: '10px ui-monospace, monospace' });

    // BN box / formula
    const bx = ix + gw + 26;
    ctx.save(); ctx.fillStyle = train ? 'rgba(31,111,235,0.10)' : 'rgba(210,105,30,0.12)'; ctx.fillRect(bx, topY + gw / 2 - 34, 150, 68); ctx.strokeStyle = train ? BLUE : ORANGE; ctx.lineWidth = 1.6; ctx.strokeRect(bx, topY + gw / 2 - 34, 150, 68);
    ctx.fillStyle = INK; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('y = γ(x−μ)/√(σ²+ε)+β', bx + 75, topY + gw / 2 - 18);
    ctx.fillStyle = train ? BLUE : ORANGE; ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(train ? 'TRAIN' : 'INFERENCE', bx + 75, topY + gw / 2);
    ctx.fillStyle = '#3a4047'; ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(train ? `μ,σ = this batch (${popMu.toFixed(2)})` : `μ,σ = running (${rmu.toFixed(2)})`, bx + 75, topY + gw / 2 + 16);
    ctx.fillText(`γ=${st.gamma.toFixed(1)} β=${st.beta.toFixed(1)}`, bx + 75, topY + gw / 2 + 28);
    ctx.restore();

    // output
    const ox = bx + 150 + 26;
    r.label('normalized output y', ox, topY - 12, { color: GREEN, font: '11px ui-monospace, monospace' });
    ctx.save(); for (let i = 0; i < N * N; i++) { ctx.fillStyle = signcol(out[i], odom); ctx.fillRect(ox + (i % N) * cell, topY + ((i / N) | 0) * cell, cell - 0.5, cell - 0.5); } ctx.restore();
    r.label(`out: μ=${mean(out).toFixed(2)}  σ=${Math.sqrt(variance(out, mean(out))).toFixed(2)}`, ox, topY + gw + 14, { color: '#586069', font: '10px ui-monospace, monospace' });

    // arrows
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(ix + gw + 4, topY + gw / 2); ctx.lineTo(bx - 4, topY + gw / 2); ctx.moveTo(bx + 150 + 4, topY + gw / 2); ctx.lineTo(ox - 4, topY + gw / 2); ctx.stroke(); ctx.restore();

    // ---- running-stats tracking chart ----
    const cx = pad + 8, cyTop = topY + gw + 34, cw = page.W - cx - pad - 10, chH = Math.max(44, page.H - cyTop - 18);
    r.label('running μ tracks the noisy batch μ → converges to the channel mean (EMA)', cx, cyTop - 8, { color: INK, font: '11px ui-monospace, monospace' });
    const ymin = -0.2, ymax = 1.2, Y = (v) => cyTop + chH - (v - ymin) / (ymax - ymin) * chH;
    ctx.save();
    ctx.strokeStyle = '#eceef0'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, Y(0)); ctx.lineTo(cx + cw, Y(0)); ctx.stroke();
    // target (true channel mean)
    ctx.strokeStyle = 'rgba(44,160,44,0.7)'; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(cx, Y(popMu)); ctx.lineTo(cx + cw, Y(popMu)); ctx.stroke(); ctx.setLineDash([]);
    r.label(`channel mean ${popMu.toFixed(2)}`, cx + cw - 110, Y(popMu) - 6, { color: GREEN, font: '9px ui-monospace, monospace' });
    const n = hist.length, dx = cw / 64;
    // batch μ dots
    ctx.fillStyle = 'rgba(31,111,235,0.30)'; for (let i = 0; i < n; i++) { const x = cx + i * dx; ctx.beginPath(); ctx.arc(x, Y(hist[i].b), 1.8, 0, 7); ctx.fill(); }
    // running μ line
    ctx.strokeStyle = BLUE; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = cx + i * dx, y = Y(hist[i].r); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); } ctx.stroke();
    ctx.restore();
    r.label(`batch μ (jittery dots) vs running μ=${rmu.toFixed(3)}, running σ=${Math.sqrt(Math.max(0, rvar)).toFixed(3)}`, cx, cyTop + chH + 14, { color: '#586069', font: '10px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && !painting) {
      const p = page.pointer;
      const ic = Math.floor((p.x - ix) / cell), ir = Math.floor((p.y - topY) / cell);
      if (ir >= 0 && ir < N && ic >= 0 && ic < N && p.x >= ix && p.y >= topY && p.x < ix + gw) page.setTip(`x[${ir},${ic}] = ${X[ir * N + ic].toFixed(2)}\ndrag ↕ to paint`);
      else { const oc = Math.floor((p.x - ox) / cell), orr = Math.floor((p.y - topY) / cell); if (orr >= 0 && orr < N && oc >= 0 && oc < N && p.x >= ox && p.y >= topY && p.x < ox + gw) page.setTip(`y[${orr},${oc}] = γ(x−μ)/√(σ²+ε)+β = ${out[orr * N + oc].toFixed(2)}\nμ=${nMu.toFixed(2)} σ=${Math.sqrt(nVar).toFixed(2)} (${train ? 'batch' : 'running'})`); }
    }

    let o = `batchnorm: y=γ(x−μ)/√(σ²+ε)+β, per channel.  ${train ? 'TRAIN: μ,σ from this batch + update the running average' : 'INFERENCE: μ,σ = frozen running average (no batch)'}.    tier:${r.name}\n`;
    o += train
      ? `the running average smooths noisy per-batch means toward the true channel mean ${popMu.toFixed(2)} (running μ=${rmu.toFixed(2)}). Paint the input to shift it; switch to inference to use the frozen running stats.`
      : `using the frozen running μ=${rmu.toFixed(2)}: a brighter input (μ=${popMu.toFixed(2)}) is NOT recentered to 0 — its output mean is ${mean(out).toFixed(2)} (measured against the learned population). Paint brighter to see it.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__bnPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('shape')) page.controls.set('shape', q.get('shape'));
  if (q.has('mode')) page.controls.set('mode', q.get('mode'));
  if (q.has('paint') && cur) for (const pr of q.get('paint').split(';')) { const [rr, cc] = pr.split(',').map(Number); if (rr * N + cc < cur.X.length) cur.X[rr * N + cc] = 1; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
