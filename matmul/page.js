// matmul concept page -- C = A·B. First end-to-end exercise of the framework:
// layout.mount() chrome + controls (dims/seed/per-term) + the step Transport
// walking tensor.matmulSteps, drawn with render.heatmap/cell/label.
//
// Interactive per the framework contract (plan/framework.md): drag any A or B
// cell vertically to change its value and watch C (and the active dot product)
// recompute live; hover any cell for its value, or hover a C cell for its full
// derivation (row·col = a0·b0 + …); the row·col sweep auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn, matmul, matmulSteps, collectSteps } from '../framework/tensor.js';

const BLUE = '#1f6feb', ORANGE = '#d2691e', INK = '#111';

// Shared state between compute() (builds the steps), draw() (renders them), and
// onPointer() (hit-tests + edits). layout rebuilds the transport (=> compute())
// before the first/relevant draw, so `cur` is always fresh for the matching
// draw. Drag edits mutate cur.A/cur.B in place; resync() rebuilds the step
// sequence + the live product C from them.
let cur = { A: null, B: null, final: null };
let rA = null, rB = null, rC = null;   // matrix rects, captured in draw for hit-testing
let grab = null;                       // {which:'A'|'B', r, c} while dragging a cell

function buildData(st) {
  const A = seededRandn(st.seed | 0, [st.m, st.k]);
  const B = seededRandn((st.seed | 0) * 7 + 1, [st.k, st.n]);
  cur = { A, B, final: matmul(A, B) };
  return collectSteps(matmulSteps(A, B, { perK: !!st.perK }));
}

// Recompute C and the transport's step list from the (possibly edited)
// cur.A/cur.B without regenerating from the seed -- so a drag edit survives,
// C updates, and the scrub axis stays valid.
function resync(page) {
  cur.final = matmul(cur.A, cur.B);
  const t = page.controls._transport;
  if (!t) return;
  t.steps = collectSteps(matmulSteps(cur.A, cur.B, { perK: !!page.state.perK }));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

const maxAbs = (M) => { let a = 0; for (let i = 0; i < M.data.length; i++) { const v = Math.abs(M.data[i]); if (v > a) a = v; } return a || 1; };
const fx = (v) => (v >= 0 ? ' ' : '') + v.toFixed(2);

// Draw one matrix heatmap + (when cells are big enough) per-cell value labels.
// Leaves renderer.layout set to THIS matrix so the caller can highlight cells.
function drawMatrix(r, M, rect, m, title) {
  const dom = m; // symmetric domain (fixed) so colors don't rescale per step
  r.heatmap(M, { rect, ramp: ramps.diverging, domain: [-dom, dom] });
  r.grid({ stroke: 'rgba(0,0,0,0.10)' });
  const L = r.layout;
  r.label(title, rect.x, rect.y - 6, { color: '#586069', font: '12px ui-monospace, monospace' });
  if (L.cellW >= 26 && L.cellH >= 16 && M.rows * M.cols <= 64) {
    for (let i = 0; i < M.rows; i++) for (let j = 0; j < M.cols; j++) {
      const v = M.data[i * M.cols + j];
      r.cell(i, j, { stroke: false, label: v.toFixed(1), labelColor: Math.abs(v) > dom * 0.6 ? '#fff' : '#222', font: '10px ui-monospace, monospace' });
    }
  }
}

mount({
  mount: 'body',
  title: 'matmul — C = A · B',
  blurb: 'The GEMM atom every other mechanism reduces to. Drag any A or B cell to change its value and watch C respond; hover a C cell for its full derivation (row · col = Σ aₖ·bₖ). Scrub (or let it play) to watch each output cell fill as the dot product of a row of A and a column of B. C[i,j] = Σₖ A[i,k]·B[k,j].',
  prefer: 'webgl2',
  aspect: '8 / 5',
  autoplay: true,
  controls: (c, page) => {
    c.dimKnobs({
      m: { label: 'A rows (m)', min: 1, max: 8, value: 4 },
      k: { label: 'inner (k)', min: 1, max: 8, value: 5 },
      n: { label: 'B cols (n)', min: 1, max: 8, value: 4 },
    });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.toggle('perK', { label: 'step per multiply-add term', value: false, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 7, loop: true });
  },
  // Direct manipulation: grab an A/B cell, drag vertically to change its value.
  onPointer: (page, ev) => {
    const { A, B } = cur;
    if (!A) return;
    if (ev.type === 'down') {
      grab = null;
      const ah = rA && cellAt(rA, A.rows, A.cols, ev.x, ev.y);
      const bh = rB && cellAt(rB, B.rows, B.cols, ev.x, ev.y);
      if (ah) grab = { which: 'A', r: ah.r, c: ah.c };
      else if (bh) grab = { which: 'B', r: bh.r, c: bh.c };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const M = grab.which === 'A' ? cur.A : cur.B;
      const idx = grab.r * M.cols + grab.c;
      M.data[idx] = Math.max(-3, Math.min(3, M.data[idx] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, st = page.state;
    const { A, B, final } = cur;
    if (!A) return;
    r.clear('#ffffff');

    // Corner layout: A bottom-left (m×k), B top-right (k×n), C bottom-right (m×n).
    const m = st.m, k = st.k, n = st.n, gap = 18, padL = 16, padT = 26, padR = 12, padB = 12;
    const availW = page.W - padL - padR - gap, availH = page.H - padT - padB - gap;
    const cell = Math.max(4, Math.min(54, Math.floor(Math.min(availW / (k + n), availH / (k + m)))));
    const xA = padL, xBC = padL + k * cell + gap;
    const yB = padT, yAC = padT + k * cell + gap;
    rA = { x: xA, y: yAC, w: k * cell, h: m * cell };
    rB = { x: xBC, y: yB, w: n * cell, h: k * cell };
    rC = { x: xBC, y: yAC, w: n * cell, h: m * cell };

    const s = page.step();
    const i = s ? s.i : -1, j = s ? s.j : -1;
    const term = s && s.phase === 'term', kk = term ? s.k : -1;
    const partial = s ? s.partial : { data: new Float32Array(m * n), rows: m, cols: n };
    const dC = maxAbs(final);

    // A + highlight active row i (and active element on per-term).
    drawMatrix(r, A, rA, maxAbs(A), `A  (${m}×${k})`);
    if (i >= 0) for (let c = 0; c < k; c++) r.cell(i, c, { stroke: BLUE, width: 1.5 });
    if (term) r.cell(i, kk, { stroke: ORANGE, width: 2.5 });

    // B + highlight active col j.
    drawMatrix(r, B, rB, maxAbs(B), `B  (${k}×${n})`);
    if (j >= 0) for (let rr = 0; rr < k; rr++) r.cell(rr, j, { stroke: BLUE, width: 1.5 });
    if (term) r.cell(kk, j, { stroke: ORANGE, width: 2.5 });

    // C = partial (fixed domain so colors are stable as it fills).
    r.heatmap(partial, { rect: rC, ramp: ramps.diverging, domain: [-dC, dC] });
    r.grid({ stroke: 'rgba(0,0,0,0.10)' });
    r.label(`C = A·B  (${m}×${n})`, rC.x, rC.y - 6, { color: '#586069', font: '12px ui-monospace, monospace' });
    if (r.layout.cellW >= 26 && m * n <= 64) {
      for (let a = 0; a < m; a++) for (let b = 0; b < n; b++) {
        const done = s && (a < i || (a === i && b <= j) || s.done);
        if (!done && !(a === i && b === j)) continue;
        const v = partial.data[a * n + b];
        r.cell(a, b, { stroke: false, label: v.toFixed(1), labelColor: Math.abs(v) > dC * 0.6 ? '#fff' : '#222', font: '10px ui-monospace, monospace' });
      }
    }
    if (i >= 0) r.cell(i, j, { stroke: INK, width: 2.5 });

    // Hover-to-inspect: A/B cell -> value; C cell -> full row·col derivation.
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const ah = rA && cellAt(rA, m, k, p.x, p.y);
      const bh = rB && cellAt(rB, k, n, p.x, p.y);
      const ch = rC && cellAt(rC, m, n, p.x, p.y);
      let tip = null;
      if (ah) tip = `A[${ah.r},${ah.c}] = ${A.data[ah.r * k + ah.c].toFixed(3)}\ndrag ↕ to change`;
      else if (bh) tip = `B[${bh.r},${bh.c}] = ${B.data[bh.r * n + bh.c].toFixed(3)}\ndrag ↕ to change`;
      else if (ch) {
        const ci = ch.r, cj = ch.c;
        const terms = [];
        for (let t = 0; t < k; t++) terms.push(`${fx(A.data[ci * k + t])}·${fx(B.data[t * n + cj])}`);
        const shown = terms.length <= 5 ? terms.join(' + ') : terms.slice(0, 4).join(' + ') + ' + … (' + k + ' terms)';
        tip = `C[${ci},${cj}] = row ${ci} of A · col ${cj} of B\n= ${shown}\n= ${final.data[ci * n + cj].toFixed(3)}`;
      }
      if (tip) page.setTip(tip);
    }

    // Readout: the live computation for the active cell.
    let out = `C = A·B    A:${m}×${k}  B:${k}×${n}  C:${m}×${n}    tier:${r.name}\n`;
    if (!s) { out += '(drag A/B cells to edit · press ▶ or scrub to step through the multiply)'; }
    else {
      out += `${s.label}\n`;
      const terms = [];
      for (let t = 0; t < k; t++) terms.push(`${fx(A.data[i * k + t])}·${fx(B.data[t * n + j])}`);
      const shown = terms.length <= 6 ? terms.join(' + ') : terms.slice(0, 5).join(' + ') + ' + … (' + k + ' terms)';
      out += `C[${i},${j}] = ${shown} = ${fx(final.data[i * n + j])}`;
    }
    page.setReadout(out);
  },
}).then((page) => {
  window.__matmulPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=which,r,c,val sets matrix cell (A or B) to a value (headless stand-in
  // for a vertical drag, since --screenshot has no pointer). e.g. ?drag=A,0,0,3
  if (q.has('drag')) {
    const [w, rr, cc, v] = q.get('drag').split(',');
    const M = w === 'A' ? cur.A : w === 'B' ? cur.B : null;
    if (M) { M.data[(+rr) * M.cols + (+cc)] = +v; resync(page); }
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
