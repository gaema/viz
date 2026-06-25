// backprop concept page -- the BACKWARD pass (autograd), the training counterpart
// to the forward pass. A tiny computation graph for one neuron with a squared
// loss runs forward (values flow →), then gradients flow back ← via the chain
// rule: each node multiplies the upstream gradient by its own local derivative.
//   m1=w1·x1  m2=w2·x2   z=m1+m2+b   a=tanh(z)   L=(a−y)²
// Backward: dL/da=2(a−y); dL/dz=dL/da·(1−a²); dL/dw1=dL/dz·x1; ... Each weight's
// gradient says which way to nudge it to reduce L. Turn on "descend" to take
// gradient steps (w ← w − lr·dL/dw) and watch the loss fall toward 0 -- that is
// training. Drag any leaf (inputs/weights/target) to change it; step the reveal.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', PURPLE = '#8250df', RED = '#d1242f';
const NODES = [
  { id: 'x1', label: 'x₁', x: 64, y: 86, leaf: 1, gl: 4 },
  { id: 'w1', label: 'w₁', x: 64, y: 132, leaf: 1, train: 1, gl: 4 },
  { id: 'm1', label: '×', x: 210, y: 109, gl: 3 },
  { id: 'x2', label: 'x₂', x: 64, y: 190, leaf: 1, gl: 4 },
  { id: 'w2', label: 'w₂', x: 64, y: 236, leaf: 1, train: 1, gl: 4 },
  { id: 'm2', label: '×', x: 210, y: 213, gl: 3 },
  { id: 'b', label: 'b', x: 64, y: 292, leaf: 1, train: 1, gl: 3 },
  { id: 'z', label: '+', x: 348, y: 170, gl: 2 },
  { id: 'a', label: 'tanh', x: 466, y: 170, gl: 1 },
  { id: 'yt', label: 'y', x: 64, y: 350, leaf: 1, gl: 1 },
  { id: 'L', label: '(·)²', x: 588, y: 220, gl: 1 },
];
const EDGES = [['x1', 'm1'], ['w1', 'm1'], ['x2', 'm2'], ['w2', 'm2'], ['m1', 'z'], ['m2', 'z'], ['b', 'z'], ['z', 'a'], ['a', 'L'], ['yt', 'L']];
const NMAP = Object.fromEntries(NODES.map((n) => [n.id, n]));
const STEPS = ['forward', '← dL/da, dL/dy', '← dL/dz (·tanh′)', '← dL/dm, dL/db', '← dL/dw, dL/dx'];

let g = null, bsig = '', hist = [], lastT = 0, dragId = '';

function reseed(seed) { const r = seededRandn(seed, 8, { std: 0.8 }); g = { x1: 0.8, x2: -0.5, w1: r[0], w2: r[1], b: r[2] * 0.5, yt: 0.6 }; hist = []; }
function fwd() { const m1 = g.w1 * g.x1, m2 = g.w2 * g.x2, z = m1 + m2 + g.b, a = Math.tanh(z), L = (a - g.yt) ** 2; return { x1: g.x1, x2: g.x2, w1: g.w1, w2: g.w2, b: g.b, yt: g.yt, m1, m2, z, a, L }; }
function bwd(v) { const ga = 2 * (v.a - v.yt), gyt = -2 * (v.a - v.yt), gz = ga * (1 - v.a * v.a), gm1 = gz, gm2 = gz, gb = gz; return { L: 1, a: ga, yt: gyt, z: gz, m1: gm1, m2: gm2, b: gb, w1: gm1 * v.x1, x1: gm1 * v.w1, w2: gm2 * v.x2, x2: gm2 * v.w2 }; }

mount({
  mount: 'body',
  title: 'backprop — the backward pass (how a model learns)',
  blurb: 'Every other page is the forward pass — turning inputs into an output. This is the BACKWARD pass: how a model learns from its mistakes. A tiny computation graph (one neuron, squared loss) runs forward to get a loss L, then gradients flow back through it by the chain rule — each node takes the gradient arriving from above and multiplies by its own local derivative, handing the result to its inputs. The gradient on each weight, dL/dw, is exactly the slope of the loss in that weight\'s direction: nudge the weight against its gradient and the loss goes down. Flip on "descend" and the page repeatedly does w ← w − lr·dL/dw and you watch L fall toward 0 on the loss curve — that loop, over billions of parameters, is training. Step the reveal to watch gradients propagate layer by layer; drag any leaf node (inputs x, weights w, bias b, target y) to change it and see every value and gradient update; tune the learning rate.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  animate: true,
  challenges: [
    { goal: 'Train the network: drive the loss L below 0.001.', hint: 'flip "descend" on (and raise the learning rate) — gradient descent does the rest.', check: (api) => ({ solved: (api.probe.L ?? 1) < 0.001, detail: `L = ${(api.probe.L ?? 1).toFixed(4)} (need < 0.0010)` }) },
    { goal: 'Make the loss WORSE: get L above 1.5 by hand.', hint: 'drag the target y far from the output a (or push a weight to a big value).', check: (api) => ({ solved: (api.probe.L ?? 0) > 1.5, detail: `L = ${(api.probe.L ?? 0).toFixed(3)} (need > 1.5)` }) },
  ],
  controls: (c, page) => {
    c.slider('lr', { label: 'learning rate', min: 0.02, max: 1.5, step: 0.02, value: 0.3 });
    c.toggle('train', { label: 'descend (train)', value: false });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => STEPS.map((s, i) => ({ stage: i, label: `${i} · ${s}` })), loop: false, speed: 1.1 });
  },
  onPointer: (page, ev) => {
    const hit = (x, y) => { for (const n of NODES) if (n.leaf && Math.abs(x - n.x) < 30 && Math.abs(y - n.y) < 18) return n.id; return ''; };
    if (ev.type === 'down') { dragId = hit(ev.x, ev.y); }
    else if (ev.type === 'up' || ev.type === 'leave') dragId = '';
    else if (ev.type === 'move' && dragId && page.pointer.down) { g[dragId] = Math.max(-2.5, Math.min(2.5, g[dragId] - ev.dy * 0.02)); hist = []; page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, Hh = page.H;
    if (`${st.seed}` !== bsig) { reseed(st.seed | 0); bsig = `${st.seed}`; }
    const v = fwd(), gr = bwd(v);
    page.probe = { L: v.L };
    // gradient descent step (when training), gated by the animate clock
    if (st.train && (page.t || 0) - lastT > 0.22 && dragId === '') { lastT = page.t; g.w1 -= st.lr * gr.w1; g.w2 -= st.lr * gr.w2; g.b -= st.lr * gr.b; hist.push(v.L); if (hist.length > 120) hist.shift(); }
    else if (!st.train && hist.length === 0) hist.push(v.L);
    r.clear('#ffffff');
    const cs = page.step(), k = cs ? cs.stage : 4;

    // ===== edges =====
    ctx.save(); ctx.lineWidth = 1.2;
    for (const [s, t] of EDGES) { const A = NMAP[s], B = NMAP[t]; ctx.strokeStyle = '#cdd5dd'; ctx.beginPath(); ctx.moveTo(A.x + 28, A.y); ctx.lineTo(B.x - 28, B.y); ctx.stroke(); }
    // backward-gradient flow arrows (revealed)
    for (const [s, t] of EDGES) { const A = NMAP[s], B = NMAP[t]; if (k >= A.gl) { ctx.strokeStyle = 'rgba(209,36,47,0.5)'; ctx.lineWidth = 1.4; const mx = (A.x + 28 + B.x - 28) / 2, my = (A.y + B.y) / 2; ctx.beginPath(); ctx.moveTo(B.x - 28, B.y); ctx.lineTo(A.x + 28, A.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(mx - 4, my - 4); ctx.lineTo(mx - 11, my); ctx.lineTo(mx - 4, my + 4); ctx.stroke(); } }
    ctx.restore();

    // ===== nodes =====
    for (const n of NODES) {
      const val = v[n.id], grad = gr[n.id], showG = k >= n.gl, w = 56, h = 30;
      ctx.save();
      ctx.fillStyle = n.id === dragId ? 'rgba(31,111,235,0.16)' : n.train ? 'rgba(44,160,44,0.10)' : n.leaf ? '#f4f6f8' : '#eef2f6';
      ctx.fillRect(n.x - w / 2, n.y - h / 2, w, h); ctx.strokeStyle = n.id === dragId ? BLUE : n.train ? GREEN : n.id === 'L' ? RED : '#cdd5dd'; ctx.lineWidth = (n.train || n.id === 'L') ? 1.5 : 1; ctx.strokeRect(n.x - w / 2, n.y - h / 2, w, h);
      ctx.fillStyle = INK; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(n.label, n.x, n.y - 9);
      ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = '#1a1d21'; ctx.fillText((val >= 0 ? ' ' : '') + val.toFixed(2), n.x, n.y + 3);
      if (showG) { ctx.fillStyle = RED; ctx.font = '9px ui-monospace, monospace'; ctx.fillText('g=' + grad.toFixed(2), n.x, n.y + 13); }
      if (n.leaf) { ctx.fillStyle = GREY; ctx.font = '7px ui-monospace, monospace'; ctx.fillText('drag', n.x, n.y - h / 2 - 3); }
      ctx.restore();
    }
    r.label('forward values flow →   ·   gradients (g=, red) flow ← by the chain rule   ·   green = trainable, red = loss', 20, 44, { color: INK, font: '11px ui-monospace, monospace' });
    r.label('step ' + k + '/4: ' + STEPS[k], 20, 58, { color: k === 0 ? BLUE : RED, font: '10px ui-monospace, monospace' });

    // ===== loss curve + descent state (bottom) =====
    const cx = 64, cy = 400, cw = W - cx - 230, chh = Hh - cy - 28;
    r.label('loss L over gradient steps' + (st.train ? '  (descending…)' : '  (toggle "descend" to train)'), cx, cy - 8, { color: INK, font: '11px ui-monospace, monospace' });
    const hmax = Math.max(0.05, ...hist), X = (i) => cx + (hist.length <= 1 ? 0 : i / (hist.length - 1) * cw), Y = (l) => cy + chh - l / hmax * chh;
    ctx.save(); ctx.strokeStyle = '#eceef0'; ctx.strokeRect(cx, cy, cw, chh);
    ctx.strokeStyle = GREEN; ctx.lineWidth = 1.8; ctx.beginPath(); hist.forEach((l, i) => { const px = X(i), py = Y(l); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }); ctx.stroke();
    ctx.fillStyle = RED; ctx.beginPath(); ctx.arc(X(hist.length - 1), Y(hist[hist.length - 1]), 3.5, 0, 7); ctx.fill(); ctx.restore();
    r.label(`0`, cx - 8, cy + chh, { color: '#8a939b', font: '8px ui-monospace, monospace' });

    // right panel: live numbers + the update rule
    const rx = cx + cw + 24, ry = cy - 4;
    r.label('current state', rx, ry, { color: INK, font: '11px ui-monospace, monospace' });
    r.label(`L = (a − y)² = ${v.L.toFixed(4)}`, rx, ry + 18, { color: RED, font: '11px ui-monospace, monospace' });
    r.label(`a = ${v.a.toFixed(3)}   target y = ${v.yt.toFixed(2)}`, rx, ry + 34, { color: '#3a4047', font: '10px ui-monospace, monospace' });
    r.label('weight gradients:', rx, ry + 54, { color: INK, font: '10px ui-monospace, monospace' });
    r.label(`dL/dw₁=${gr.w1.toFixed(2)}  dL/dw₂=${gr.w2.toFixed(2)}  dL/db=${gr.b.toFixed(2)}`, rx, ry + 70, { color: '#586069', font: '9px ui-monospace, monospace' });
    r.label(`update:  w ← w − ${st.lr.toFixed(2)}·dL/dw`, rx, ry + 90, { color: GREEN, font: '10px ui-monospace, monospace' });
    r.label(st.train ? `step ${hist.length}: descending` : 'paused — flip "descend" on', rx, ry + 106, { color: st.train ? GREEN : '#8a939b', font: '10px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && !dragId) { const p = page.pointer; for (const n of NODES) if (Math.abs(p.x - n.x) < 28 && Math.abs(p.y - n.y) < 15) { page.setTip(`${n.id} = ${v[n.id].toFixed(3)}\ngradient dL/d${n.id} = ${gr[n.id].toFixed(3)}${n.train ? '\n← a weight: nudged by −lr·grad' : n.leaf ? '\n(drag ↕ to change)' : ''}`); break; } }

    let o = `backprop · ${st.train ? 'TRAINING' : 'step ' + k + '/4: ' + STEPS[k]}.  forward: m=w·x, z=Σm+b, a=tanh(z), L=(a−y)².  backward (chain rule): dL/da=2(a−y), dL/dz=dL/da·(1−a²), dL/dw=dL/dz·x.   tier:${r.name}\n`;
    o += st.train ? `descending: w ← w − ${st.lr.toFixed(2)}·dL/dw each step → L = ${v.L.toFixed(4)} (step ${hist.length}). The loss curve falls toward 0 — this loop is training.` : `L = ${v.L.toFixed(4)}. Each weight's gradient is the slope of L in its direction; flip "descend" to step against the gradients and watch L drop. Drag a leaf node to change the inputs/weights/target.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__bpPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('lr')) page.controls.set('lr', +q.get('lr'));
  if (q.has('seed')) page.controls.set('seed', +q.get('seed'));
  if (q.has('train')) page.controls.set('train', q.get('train') === '1');
  if (q.has('step') && page.controls._transport) page.controls._transport.seek(+q.get('step'));
  page.redraw();
});
