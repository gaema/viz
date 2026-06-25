// dot-product concept page -- a·b term by term + geometry (projection, cosine).
// Interactive per the framework contract (plan/framework.md): drag any a/b cell
// to change its value and watch a·b + the geometry recompute live; hover any
// cell for its value + derivation; the running sum auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { dot, dotSteps, collectSteps, seededRandn } from '../framework/tensor.js';

const BLUE = '#1f6feb', ORANGE = '#d2691e', INK = '#111';
const norm = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };
const maxAbs = (v) => { let a = 0; for (let i = 0; i < v.length; i++) { const x = Math.abs(v[i]); if (x > a) a = x; } return a || 1; };

// Shared between compute(), draw(), and onPointer(). compute() runs (via
// transport rebuild) before the matching draw, so `cur` is fresh. Drag edits
// mutate cur.a/cur.b in place; resync() rebuilds the step sequence from them.
let cur = { a: null, b: null };
let aRect = null, bRect = null, prodRect = null;   // strip rects, captured in draw for hit-testing
let grab = null;                                   // {which:'a'|'b', i} while dragging a cell

function buildData(st) {
  const k = st.k, al = st.align;
  const a = seededRandn(st.seed | 0, k);
  const noise = seededRandn((st.seed | 0) + 101, k);
  const b = new Float32Array(k);              // b = align·a + (1-|align|)·noise
  for (let i = 0; i < k; i++) b[i] = al * a[i] + (1 - Math.abs(al)) * noise[i];
  cur = { a, b };
  return collectSteps(dotSteps(a, b));
}

// Recompute the transport's step list from the (possibly edited) cur.a/cur.b
// without regenerating from the seed -- so a drag edit survives + scrubs right.
function resync(page) {
  const t = page.controls._transport;
  if (!t) return;
  t.steps = collectSteps(dotSteps(cur.a, cur.b));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

// One 1×k strip heatmap + per-cell labels + active-cell outline.
function strip(r, vec, k, rect, dom, activeK, opts = {}) {
  r.heatmap(vec, { rows: 1, cols: k, rect, ramp: ramps.diverging, domain: [-dom, dom] });
  r.grid({ stroke: 'rgba(0,0,0,0.12)' });
  if (r.layout.cellW >= 22) {
    for (let i = 0; i < k; i++) {
      if (opts.upto != null && i > opts.upto) continue;
      const v = vec[i];
      r.cell(0, i, { stroke: false, label: v.toFixed(1), labelColor: Math.abs(v) > dom * 0.6 ? '#fff' : '#222', font: '10px ui-monospace, monospace' });
    }
  }
  if (activeK >= 0 && activeK < k) r.cell(0, activeK, { stroke: INK, width: 2.5 });
}

// Geometry: a along +x, b at the true angle θ, projection of b onto a shaded.
function geometry(page, a, b, rect) {
  const r = page.renderer, ctx = page.ctx;
  const na = norm(a), nb = norm(b), d = dot(a, b);
  const cos = na && nb ? d / (na * nb) : 0, th = Math.acos(Math.max(-1, Math.min(1, cos)));
  const cx = rect.x + rect.w * 0.20, cy = rect.y + rect.h * 0.60;
  const sc = Math.min(rect.w * 0.62, rect.h * 0.72) / (Math.max(na, nb) || 1);
  const aTip = { x: cx + na * sc, y: cy };
  const bTip = { x: cx + nb * cos * sc, y: cy - nb * Math.sin(th) * sc };
  const projLen = nb * cos, projPt = { x: cx + projLen * sc, y: cy };

  ctx.save();
  ctx.strokeStyle = '#e3e6ea'; ctx.lineWidth = 1;                 // baseline axis
  ctx.beginPath(); ctx.moveTo(rect.x + 8, cy); ctx.lineTo(rect.x + rect.w - 8, cy); ctx.stroke();
  ctx.strokeStyle = 'rgba(31,111,235,0.30)'; ctx.lineWidth = 7;   // projection of b onto a
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(projPt.x, projPt.y); ctx.stroke();
  ctx.setLineDash([4, 4]); ctx.strokeStyle = '#9aa4ad'; ctx.lineWidth = 1;  // drop line
  ctx.beginPath(); ctx.moveTo(bTip.x, bTip.y); ctx.lineTo(projPt.x, projPt.y); ctx.stroke();
  ctx.restore();

  r.arrow({ x: cx, y: cy }, aTip, { color: BLUE, width: 2.5 });
  r.arrow({ x: cx, y: cy }, bTip, { color: ORANGE, width: 2.5 });
  r.label('a', aTip.x + 6, aTip.y + 14, { color: BLUE, font: '13px ui-monospace, monospace' });
  r.label('b', bTip.x + 6, bTip.y, { color: ORANGE, font: '13px ui-monospace, monospace' });
  r.label(`θ = ${(th * 180 / Math.PI).toFixed(0)}°   cos θ = ${cos.toFixed(2)}`, rect.x + 8, rect.y + 16, { color: '#586069', font: '12px ui-monospace, monospace' });
  r.label(`projₐ b = |b|cos θ = ${projLen.toFixed(2)}`, rect.x + 8, rect.y + rect.h - 8, { color: '#586069', font: '12px ui-monospace, monospace' });
}

mount({
  mount: 'body',
  title: 'dot-product — a · b',
  blurb: 'The inner loop of every matmul, attention score, and projection. Drag any a or b cell to change its value and watch a·b and the geometry respond; hover a cell for its value + derivation. Scrub (or let it play) to add each term aₖ·bₖ into the running sum. Right: a·b = |a|·|b|·cos θ.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('k', { label: 'dimension (k)', min: 2, max: 12, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.slider('align', { label: 'b alignment to a', min: -1, max: 1, step: 0.05, value: 0.35, rebuild: true, format: (v) => (+v).toFixed(2) });
    c.transport({ compute: () => buildData(page.state), speed: 6, loop: true });
  },
  // Direct manipulation: grab an a/b cell, drag vertically to change its value.
  onPointer: (page, ev) => {
    const k = page.state.k;
    if (ev.type === 'down') {
      grab = null;
      for (const [which, rect] of [['a', aRect], ['b', bRect]]) {
        const hit = rect && cellAt(rect, 1, k, ev.x, ev.y);
        if (hit) { grab = { which, i: hit.c }; break; }
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down && cur[grab.which]) {
      const arr = cur[grab.which];
      arr[grab.i] = Math.max(-3, Math.min(3, arr[grab.i] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, st = page.state, { a, b } = cur;
    if (!a) return;
    r.clear('#ffffff');
    const k = st.k, pad = 16;
    const leftW = page.W * 0.56;
    const s = page.step();
    const ki = s ? s.k : -1, acc = s ? s.acc : 0;

    const na = norm(a), nb = norm(b), d = dot(a, b);
    const amax = maxAbs(a), bmax = maxAbs(b);
    const prodP = new Float32Array(k);          // products revealed up to ki
    let pmax = 1;
    for (let i = 0; i < k; i++) { const p = a[i] * b[i]; if (Math.abs(p) > pmax) pmax = Math.abs(p); prodP[i] = i <= ki ? p : 0; }

    const xs = pad + 46, cell = Math.max(10, Math.min(40, (leftW - xs - pad) / k)), sw = k * cell;
    let y = 42;
    const rowLabel = (t, col) => r.label(t, pad, y + cell * 0.62, { color: col, font: '13px ui-monospace, monospace' });

    rowLabel('a', BLUE); aRect = { x: xs, y, w: sw, h: cell }; strip(r, a, k, aRect, amax, ki); y += cell + 24;
    rowLabel('b', ORANGE); bRect = { x: xs, y, w: sw, h: cell }; strip(r, b, k, bRect, bmax, ki); y += cell + 24;
    rowLabel('aₖ·bₖ', '#586069'); prodRect = { x: xs, y, w: sw, h: cell }; strip(r, prodP, k, prodRect, pmax, ki, { upto: ki }); y += cell + 30;

    r.label(`running  Σ = ${acc.toFixed(3)}`, pad, y, { font: '13px ui-monospace, monospace', color: INK }); y += 22;
    r.label(`a · b = ${d.toFixed(3)}`, pad, y, { font: '13px ui-monospace, monospace', color: INK });

    geometry(page, a, b, { x: leftW + 8, y: 8, w: page.W - leftW - 16, h: page.H - 16 });

    // Hover-to-inspect: tooltip with the cell value + (for products) the derivation.
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const ah = aRect && cellAt(aRect, 1, k, p.x, p.y);
      const bh = bRect && cellAt(bRect, 1, k, p.x, p.y);
      const ph = prodRect && cellAt(prodRect, 1, k, p.x, p.y);
      let tip = null;
      if (ah) tip = `a[${ah.c}] = ${a[ah.c].toFixed(3)}\ndrag ↕ to change`;
      else if (bh) tip = `b[${bh.c}] = ${b[bh.c].toFixed(3)}\ndrag ↕ to change`;
      else if (ph && ph.c <= ki) tip = `a[${ph.c}]·b[${ph.c}]\n= ${a[ph.c].toFixed(2)} · ${b[ph.c].toFixed(2)} = ${(a[ph.c] * b[ph.c]).toFixed(3)}`;
      if (tip) page.setTip(tip);
    }

    const cos = na && nb ? d / (na * nb) : 0;
    let out = `a·b = Σ aₖ·bₖ    |a| = ${na.toFixed(3)}   |b| = ${nb.toFixed(3)}   cos θ = a·b/(|a||b|) = ${cos.toFixed(3)}    tier:${r.name}\n`;
    if (!s) out += '(drag a/b cells to edit · press ▶ or scrub to accumulate each term)';
    else {
      out += `${s.label}\n`;
      out += `a·b = ${d.toFixed(3)} = |a|·|b|·cos θ = ${na.toFixed(2)}·${nb.toFixed(2)}·${cos.toFixed(2)}`;
    }
    page.setReadout(out);
  },
}).then((page) => {
  window.__dotPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=which,i,val sets a cell to a value (headless stand-in for a drag).
  if (q.has('drag')) {
    const [w, i, v] = q.get('drag').split(',');
    if (cur[w]) { cur[w][+i] = +v; resync(page); }
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover,
  // since --screenshot has no pointer) so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }   // deterministic frame for capture
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
