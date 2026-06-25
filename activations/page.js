// activations concept page -- GELU (erf/tanh), SiLU/Swish, Mish curves.
// Interactive per the framework contract (plan/framework.md): a draggable
// x-marker you slide horizontally to pick the evaluation point; hover anywhere
// over the plot to read all four activations at that x (vertical guide + a dot
// on each curve + a value tooltip); and an ambient sweep that animates the
// marker on load so the page is alive, frozen while you drag. A line plot
// (Canvas2D) rather than heatmaps.
import { mount } from '../framework/layout.js';
import { silu, gelu } from '../framework/tensor.js';

const INK = '#111';
const ACTS = [
  { key: 'gelu-erf', name: 'GELU (erf)', color: '#1f6feb', formula: '0.5·x·(1+erf(x/√2))', fn: (xs) => gelu(xs, { approx: 'erf' }) },
  { key: 'gelu-tanh', name: 'GELU (tanh)', color: '#17a2b8', formula: '0.5·x·(1+tanh(√(2/π)(x+0.0447x³)))', fn: (xs) => gelu(xs) },
  { key: 'silu', name: 'SiLU / Swish', color: '#d2691e', formula: 'x·σ(x)', fn: (xs) => silu(xs) },
  { key: 'mish', name: 'Mish', color: '#9467bd', formula: 'x·tanh(softplus(x))', fn: (xs) => Float32Array.from(xs, (x) => x * Math.tanh(Math.log1p(Math.exp(Math.min(x, 30))))) },
];
const val = (a, x) => a.fn(Float32Array.of(x))[0];

// Evaluation point. markX is the live x the marker + per-curve dots + readout
// track. It is driven by (priority order) a drag, then the ?x hook, else the
// ambient sweep (api.t). `frozen` stops the sweep once the user takes control
// (drag or ?x). plotRect / pxToX are captured in draw for pointer↔x mapping.
let markX = 1;
let frozen = false;          // true once a drag or ?x sets x -> sweep stops
let grabbing = false;        // true while dragging the x-marker
let plotRect = null;         // {XMIN,XMAX,x,y,w,h, px, xAt} captured each draw

// Map a canvas pixel x back to data x, clamped to the plot's x-range.
function xFromPx(px) {
  if (!plotRect) return markX;
  const { XMIN, XMAX, x, w } = plotRect;
  return Math.max(XMIN, Math.min(XMAX, XMIN + (px - x) / w * (XMAX - XMIN)));
}

function buildData() { return ACTS.map((a) => ({ ...a, label: `${a.name}:  f(x) = ${a.formula}` })); }

mount({
  mount: 'body',
  title: 'activations — GELU / SiLU / Mish',
  blurb: 'Smooth MLP activations, side by side. All are ≈x for large +x, ≈0 for large −x, with a small negative dip — nearly interchangeable. Drag the x-marker (or hover the plot) to compare all four values at a point you control; on load the marker sweeps on its own. Scrub to highlight each; zoom in to see where they differ.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  controls: (c, page) => {
    c.toggle('zoom', { label: 'zoom near 0', value: false });
    c.transport({ compute: buildData, speed: 1.2 });
  },
  // Direct manipulation: grab the x-marker (or anywhere in the plot) and drag
  // horizontally to move the evaluation point x. Freezes the ambient sweep.
  onPointer: (page, ev) => {
    if (ev.type === 'down') {
      if (plotRect && ev.x >= plotRect.x - 14 && ev.x <= plotRect.x + plotRect.w + 14 &&
          ev.y >= plotRect.y && ev.y <= plotRect.y + plotRect.h) {
        grabbing = true; frozen = true; markX = xFromPx(ev.x);
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grabbing = false;
    } else if (ev.type === 'move' && grabbing && page.pointer.down) {
      markX = xFromPx(ev.x);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear('#ffffff');
    const s = page.step(), ai = s ? ACTS.findIndex((a) => a.key === s.key) : -1;
    const XMIN = st.zoom ? -4 : -6, XMAX = st.zoom ? 3 : 6, M = 260;

    // Ambient sweep of the marker when the user isn't driving it.
    if (!frozen && !grabbing) markX = (XMAX + XMIN) / 2 + (XMAX - XMIN) / 2 * 0.92 * Math.sin(page.t * 0.6);

    const xs = new Float32Array(M); for (let i = 0; i < M; i++) xs[i] = XMIN + (XMAX - XMIN) * i / (M - 1);
    const curves = ACTS.map((a) => a.fn(xs));
    let ymin = Infinity, ymax = -Infinity; for (const c of curves) for (let i = 0; i < M; i++) { if (c[i] < ymin) ymin = c[i]; if (c[i] > ymax) ymax = c[i]; }
    ymin = Math.min(ymin, 0) - 0.3; ymax += 0.3;

    const pad = 16, plot = { x: pad + 40, y: 54, w: page.W * 0.6, h: page.H - 54 - 44 };
    const px = (x) => plot.x + (x - XMIN) / (XMAX - XMIN) * plot.w;
    const py = (y) => plot.y + plot.h - (y - ymin) / (ymax - ymin) * plot.h;
    plotRect = { XMIN, XMAX, x: plot.x, y: plot.y, w: plot.w, h: plot.h, px };

    // grid + axes
    ctx.save(); ctx.strokeStyle = '#eef0f2'; ctx.lineWidth = 1; ctx.fillStyle = '#9aa4ad'; ctx.font = '10px ui-monospace, monospace';
    for (let gx = Math.ceil(XMIN); gx <= XMAX; gx++) { ctx.beginPath(); ctx.moveTo(px(gx), plot.y); ctx.lineTo(px(gx), plot.y + plot.h); ctx.stroke(); if (gx !== 0) { ctx.textAlign = 'center'; ctx.fillText(String(gx), px(gx), py(0) + 13); } }
    for (let gy = Math.ceil(ymin); gy <= ymax; gy++) { ctx.beginPath(); ctx.moveTo(plot.x, py(gy)); ctx.lineTo(plot.x + plot.w, py(gy)); ctx.stroke(); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(gy), plot.x - 6, py(gy)); }
    ctx.strokeStyle = '#c4ccd3'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(plot.x, py(0)); ctx.lineTo(plot.x + plot.w, py(0)); ctx.stroke();           // x-axis
    if (XMIN <= 0 && XMAX >= 0) { ctx.beginPath(); ctx.moveTo(px(0), plot.y); ctx.lineTo(px(0), plot.y + plot.h); ctx.stroke(); }
    ctx.restore();

    // The evaluation x: marker / hover share one value. While hovering (and not
    // dragging) the cursor x takes over so the readout tracks the pointer.
    let xc = Math.max(XMIN, Math.min(XMAX, markX));
    const hovering = page.pointer.over && !grabbing &&
      page.pointer.x >= plot.x - 4 && page.pointer.x <= plot.x + plot.w + 4 &&
      page.pointer.y >= plot.y && page.pointer.y <= plot.y + plot.h;
    if (hovering) xc = xFromPx(page.pointer.x);

    // vertical guide line at the evaluation x + draggable marker handle
    ctx.save();
    ctx.strokeStyle = grabbing ? 'rgba(40,44,52,0.7)' : 'rgba(40,44,52,0.4)'; ctx.lineWidth = grabbing ? 1.6 : 1.2;
    ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(px(xc), plot.y); ctx.lineTo(px(xc), plot.y + plot.h); ctx.stroke();
    ctx.setLineDash([]);
    // marker handle at the x-axis (a draggable knob)
    ctx.fillStyle = grabbing ? '#111' : '#586069'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(px(xc), py(0), grabbing ? 6 : 5, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = INK; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`x = ${xc.toFixed(2)}`, px(xc), plot.y - 2);
    ctx.restore();

    // curves (highlighted one bold; others faint when one is selected) + dot at xc
    for (let i = 0; i < ACTS.length; i++) {
      const bold = ai === i, faint = ai >= 0 && !bold;
      ctx.save(); ctx.globalAlpha = faint ? 0.3 : 1; ctx.strokeStyle = ACTS[i].color; ctx.lineWidth = bold ? 3 : 1.8; ctx.beginPath();
      for (let k = 0; k < M; k++) { const X = px(xs[k]), Y = py(curves[i][k]); k === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y); }
      ctx.stroke();
      const v = val(ACTS[i], xc); ctx.globalAlpha = faint ? 0.4 : 1; ctx.fillStyle = ACTS[i].color;
      ctx.beginPath(); ctx.arc(px(xc), py(v), bold ? 4.5 : 3.5, 0, 7); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
      ctx.restore();
    }

    // legend + values at cursor
    let lx = plot.x + plot.w + 24, ly = plot.y + 8;
    r.label(`at x = ${xc.toFixed(2)}`, lx, ly, { color: INK, font: '12px ui-monospace, monospace' }); ly += 22;
    for (let i = 0; i < ACTS.length; i++) {
      const a = ACTS[i], bold = ai === i;
      ctx.save(); ctx.strokeStyle = a.color; ctx.lineWidth = bold ? 3 : 2; ctx.beginPath(); ctx.moveTo(lx, ly - 4); ctx.lineTo(lx + 22, ly - 4); ctx.stroke(); ctx.restore();
      r.label(`${a.name} = ${val(a, xc).toFixed(3)}`, lx + 30, ly, { color: bold ? INK : '#3a4047', font: (bold ? 'bold ' : '') + '11px ui-monospace, monospace' });
      ly += 20;
    }

    // Hover-to-inspect: a tooltip listing every activation's value at the cursor x.
    if (hovering) {
      let tip = `x = ${xc.toFixed(2)}`;
      for (const a of ACTS) tip += `\n${a.name} = ${val(a, xc).toFixed(3)}`;
      page.setTip(tip);
    }

    const sweepState = frozen ? (grabbing ? 'dragging' : 'frozen') : 'sweeping';
    let o = `Activations: f(x). All smooth, non-monotonic (small negative dip), ≈x for large +x, ≈0 for large −x.${st.zoom ? '  [zoomed]' : ''}    marker:${sweepState}    tier:${r.name}\n`;
    o += `at x = ${xc.toFixed(2)}:  ` + ACTS.map((a) => `${a.name.split(' ')[0]}=${val(a, xc).toFixed(2)}`).join('  ') + '\n';
    o += s ? `${s.label}` : '(drag the x-marker ↔ or hover the plot to compare values · press ▶ or scrub to highlight each)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__actPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('zoom')) { page.controls.set('zoom', q.get('zoom') !== '0'); }
  // ?x=VALUE sets the marker x and freezes the ambient sweep (headless
  // stand-in for a horizontal drag of the x-marker).
  if (q.has('x')) { markX = parseFloat(q.get('x')); frozen = true; }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover,
  // since --screenshot has no pointer) so the tooltip + guide path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
    frozen = true;   // hold a deterministic frame for capture
  }
  if (q.has('step') && page.controls._transport) page.controls._transport.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && page.controls._transport) page.controls._transport.play();
  page.redraw();
});
