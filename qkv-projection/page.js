// qkv-projection concept page -- one embedding x -> Q,K,V via three weight
// matrices. Uses the verified framework: layout.mount() + controls + a
// per-output-dim Transport.
//
// Interactive per the framework contract (plan/framework.md): drag an
// embedding cell vertically and watch ALL THREE projections (q, k, v)
// recompute live -- the one input fanning out to three roles is the qkv
// "aha". Hover an embedding cell for its value, a weight cell for its value,
// or a q/k/v output cell for its full derivation (q[o] = Σ_d x[d]·W[d,o]).
// The output-dim sweep auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e';
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) { const x = Math.abs(a[i]); if (x > m) m = x; } return m; };
const fx = (v) => (v >= 0 ? ' ' : '') + v.toFixed(2);

// Shared state between buildData() (builds x/W/q/k/v), draw() (renders +
// captures the rects), and onPointer() (hit-tests + edits). Drag edits mutate
// cur.x in place; recompute() re-projects q/k/v from the edited embedding,
// resync() rebuilds the per-output-dim transport axis.
let cur = null;
let rX = null;                 // embedding x rect [D x 1], captured in draw
let rW = [null, null, null];   // W_Q/W_K/W_V rects [D x D]
let rOut = [null, null, null]; // q/k/v output rects [1 x D]
let grab = null;               // {d} while dragging embedding row d

function project(cur) {
  const { x, Wq, Wk, Wv, q, k, v, D } = cur;
  q.fill(0); k.fill(0); v.fill(0);
  for (let o = 0; o < D; o++) for (let kk = 0; kk < D; kk++) { q[o] += x[kk] * Wq.data[kk * D + o]; k[o] += x[kk] * Wk.data[kk * D + o]; v[o] += x[kk] * Wv.data[kk * D + o]; }
}

function buildData(st) {
  const D = st.D, seed = st.seed | 0;
  const x = seededRandn(seed, D);
  const Wq = seededRandn(seed + 1, [D, D]), Wk = seededRandn(seed + 2, [D, D]), Wv = seededRandn(seed + 3, [D, D]);
  cur = { x, Wq, Wk, Wv, q: new Float32Array(D), k: new Float32Array(D), v: new Float32Array(D), D };
  project(cur);
  return Array.from({ length: D }, (_, o) => ({ o, label: `output dim ${o}: q[${o}], k[${o}], v[${o}] = Σₖ x[k]·W[k,${o}] (dot x with column ${o})` }));
}

// Re-project q/k/v from the (possibly edited) embedding x, then refresh the
// transport's step list -- the output dim count is unchanged so the axis is
// stable, but _sync() pushes the recomputed values into the live readout.
function resync(page) {
  project(cur);
  const t = page.controls._transport;
  if (t) t._sync();
}

mount({
  mount: 'body',
  title: 'qkv-projection — one embedding → Q, K, V',
  blurb: 'A token embedding x is projected by three learned weight matrices into query, key, value: q = x·W_Q, k = x·W_K, v = x·W_V. Same input, three roles. Drag any embedding cell ↕ and watch q, k, v ALL recompute (one input → three projections). Hover a q/k/v cell for its derivation. Scrub (or let it play) to fill q/k/v element by element.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('D', { label: 'dim (D)', min: 4, max: 10, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 6, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 2, loop: true });
  },
  // Direct manipulation: grab an embedding cell, drag vertically to change
  // x[d]; q, k, v all re-project live.
  onPointer: (page, ev) => {
    if (!cur) return;
    if (ev.type === 'down') {
      grab = null;
      const xh = rX && cellAt(rX, cur.D, 1, ev.x, ev.y);
      if (xh) grab = { d: xh.r };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      cur.x[grab.d] = Math.max(-3, Math.min(3, cur.x[grab.d] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { x, Wq, Wk, Wv, q, k, v, D } = cur;
    r.clear('#ffffff');
    const s = page.step();
    const o = s ? s.o : D - 1;

    const pad = 16, gap = 22, projY = 78;
    const cell = Math.max(12, Math.min(30, (page.W - 2 * pad - 30 - 4 * gap) / (1 + 3 * D)));
    const xX = pad + 24;
    const wx = [xX + cell + gap, 0, 0]; wx[1] = wx[0] + D * cell + gap; wx[2] = wx[1] + D * cell + gap;
    const outY = projY + D * cell + 26;
    const dOut = Math.max(maxAbs(q), maxAbs(k), maxAbs(v));

    // input embedding x (vertical [D x 1]) -- capture rect for hit-testing.
    rX = { x: xX, y: projY, w: cell, h: D * cell };
    r.label('x', xX + cell / 2 - 4, projY - 8, { color: INK, font: '12px ui-monospace, monospace' });
    r.heatmap(x, { rows: D, cols: 1, rect: rX, ramp: ramps.diverging, domain: [-maxAbs(x), maxAbs(x)] });
    r.grid({ stroke: 'rgba(0,0,0,0.12)' });
    ctx.save(); ctx.strokeStyle = BLUE; ctx.lineWidth = 2; ctx.strokeRect(xX - 1, projY - 1, cell, D * cell + 2); ctx.restore();
    // active input element on the current output dim's dot product
    if (o >= 0 && o < D) { ctx.save(); ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.strokeRect(xX + 1, projY + o * cell + 1, cell - 2, cell - 2); ctx.restore(); }

    const names = ['W_Q', 'W_K', 'W_V'], outNames = ['q', 'k', 'v'], roles = ['query — what I look for', 'key — what I offer', 'value — what I carry'];
    const mats = [Wq, Wk, Wv], outs = [q, k, v];
    for (let p = 0; p < 3; p++) {
      const X = wx[p], M = mats[p], out = outs[p];
      rW[p] = { x: X, y: projY, w: D * cell, h: D * cell };
      r.label(`${names[p]} [${D}×${D}]`, X, projY - 8, { color: '#586069', font: '11px ui-monospace, monospace' });
      r.heatmap(M, { rows: D, cols: D, rect: rW[p], ramp: ramps.diverging, domain: [-maxAbs(M.data), maxAbs(M.data)] });
      r.grid({ stroke: 'rgba(0,0,0,0.10)' });
      // highlight column o
      ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(X + o * cell + 1, projY + 1, cell - 2, D * cell - 2); ctx.restore();

      // output vector (horizontal [1 x D]), filled up to o -- capture rect.
      rOut[p] = { x: X, y: outY, w: D * cell, h: cell };
      r.heatmap(out, { rows: 1, cols: D, rect: rOut[p], ramp: ramps.diverging, domain: [-dOut, dOut] });
      ctx.save(); ctx.fillStyle = 'rgba(244,246,248,0.9)'; ctx.fillRect(X + (o + 1) * cell, outY, (D - o - 1) * cell, cell); ctx.restore();
      ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(X + o * cell + 1, outY + 1, cell - 2, cell - 2); ctx.restore();
      r.label(`${outNames[p]} = x·${names[p]}`, X, outY + cell + 14, { color: INK, font: '11px ui-monospace, monospace' });
      r.label(roles[p], X, outY + cell + 28, { color: '#586069', font: '10px ui-monospace, monospace' });
    }

    // Hover-to-inspect: embedding cell -> value (drag hint); weight cell ->
    // value; q/k/v output cell -> its full derivation Σ_d x[d]·W[d,o].
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      let tip = null;
      const xh = rX && cellAt(rX, D, 1, pt.x, pt.y);
      if (xh) tip = `x[${xh.r}] = ${x[xh.r].toFixed(3)}\ndrag ↕ to change — q,k,v all re-project`;
      if (!tip) for (let p = 0; p < 3 && !tip; p++) {
        const wh = rW[p] && cellAt(rW[p], D, D, pt.x, pt.y);
        if (wh) tip = `${names[p]}[${wh.r},${wh.c}] = ${mats[p].data[wh.r * D + wh.c].toFixed(3)}`;
      }
      if (!tip) for (let p = 0; p < 3 && !tip; p++) {
        const oh = rOut[p] && cellAt(rOut[p], 1, D, pt.x, pt.y);
        if (oh) {
          const oc = oh.c, M = mats[p];
          const terms = [];
          for (let d = 0; d < D; d++) terms.push(`${fx(x[d])}·${fx(M.data[d * D + oc])}`);
          const shown = terms.length <= 5 ? terms.join(' + ') : terms.slice(0, 4).join(' + ') + ' + … (' + D + ' terms)';
          tip = `${outNames[p]}[${oc}] = ${names[p]} col ${oc} · x = Σ_d x[d]·W[d,${oc}]\n= ${shown}\n= ${outs[p][oc].toFixed(3)}`;
        }
      }
      if (tip) page.setTip(tip);
    }

    let out = `q = x·W_Q   k = x·W_K   v = x·W_V    one embedding (dim ${D}) → three vectors    tier:${r.name}\n`;
    out += s ? `${s.label}\n` : '(drag an x cell to edit · press ▶ or scrub the output dim — q, k, v fill element by element)\n';
    out += `at o=${o}:  q[${o}]=${fx(q[o])}   k[${o}]=${fx(k[o])}   v[${o}]=${fx(v[o])}`;
    page.setReadout(out);
  },
}).then((page) => {
  window.__qkvPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=d,val sets embedding cell x[d] to a value (headless stand-in for a
  // vertical drag, since --screenshot has no pointer). e.g. ?drag=2,1.5 sets
  // x[2]=1.5 and re-projects q,k,v.
  if (q.has('drag')) {
    const [dd, vv] = q.get('drag').split(',').map(Number);
    if (cur && dd >= 0 && dd < cur.D) { cur.x[dd] = vv; resync(page); }
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover)
  // so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
