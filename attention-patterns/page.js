// attention-patterns concept page -- full / sliding-window / hybrid / sink.
// Uses the verified framework: layout.mount() + controls + per-pattern Transport.
//
// Interactive per the framework contract (plan/framework.md):
//   - HOVER any (query i, key j) cell -> whether it is attended and WHY for the
//     current pattern (within window / sink / blocked-future / outside-window).
//   - DRAG the current-query handle down the rows to move the highlighted query
//     whose attended set lights up; the band updates live for the pattern.
//   - DRAG the window edge (sliding / sink patterns) to widen / narrow w and
//     watch the band grow / shrink under your hand.
//   - The query position AUTO-SWEEPS down + loops (generation advancing through
//     the pattern); the pattern transport auto-plays + loops too.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';

const INK = '#111', BLUE = '#1f6feb', LIGHT = '#dbeafe', DARK = 'rgba(55,60,68,0.80)', ORANGE = '#d2691e';

const PATTERNS = [
  { key: 'full', name: 'Full (causal)', desc: 'every query attends to all past keys (j ≤ i) — O(i) per query; the KV cache grows unbounded with context' },
  { key: 'sliding', name: 'Sliding-window (SWA)', desc: 'query attends only to the last w keys (i−w < j ≤ i) — O(w) per query; bounded cache (Mistral)' },
  { key: 'hybrid', name: 'Hybrid by layer', desc: 'layers alternate local (SWA) and global (full): most layers cheap, a few see everything (Gemma 2/3: 5 local : 1 global)' },
  { key: 'sink', name: 'Attention sink', desc: 'sliding window PLUS the first s tokens always attended (the “sink”) — stabilizes long/streaming context (StreamingLLM)' },
];

// attended predicate for a (single-layer) pattern
function makeFn(kind, w, s) {
  if (kind === 'full') return (i, j) => j <= i;
  if (kind === 'sliding') return (i, j) => j <= i && j > i - w;
  if (kind === 'sink') return (i, j) => j <= i && (j > i - w || j < s);
  return (i, j) => j <= i;
}

// Per-(i,j) reason string for the hover tooltip, scoped to the active pattern.
// Returns 'attended (...)' or 'blocked (...)'. The wording mirrors the math
// each pattern enforces so the tooltip teaches the predicate, not just lights.
function reasonFor(kind, i, j, w, s) {
  if (j > i) return `q${i} → k${j}: blocked (future, j>i)`;
  if (kind === 'full') return `q${i} → k${j}: attended (causal, j≤i)`;
  if (kind === 'sliding') {
    return j > i - w
      ? `q${i} → k${j}: attended (within window w=${w}, i−w<j≤i)`
      : `q${i} → k${j}: blocked (outside window, j≤i−w=${i - w})`;
  }
  if (kind === 'sink') {
    if (j < s) return `q${i} → k${j}: attended (sink, j<s=${s})`;
    return j > i - w
      ? `q${i} → k${j}: attended (within window w=${w})`
      : `q${i} → k${j}: blocked (outside window+sink, j≤i−w and j≥s)`;
  }
  return `q${i} → k${j}`;
}

function buildData() { return PATTERNS.map((p) => ({ ...p, label: `${p.name} — ${p.desc}` })); }

// Shared between draw() and onPointer(): the hit-test geometry of the N×N grid
// captured at the end of draw(), plus the live current-query (a float so the
// auto-sweep is smooth) and the pause flag (drag / ?q / ?hover pin it).
let gridRect = null, gridN = 0, gridKind = 'full';
let curQ = 11;            // current query row (float); floor() = the lit row
let sweep = true;         // auto-sweep the query down + loop
let grab = null;          // 'query' | 'window' while dragging a handle
let lastSliderQ = null;   // detect manual qi-slider moves so they win over sweep
let lastT = 0;            // ambient-clock delta integrator for the sweep

// Push curQ into the qi control state (drag / hooks). controls.set() syncs the
// slider thumb too; keep lastSliderQ in lockstep so the draw() "manual slider
// moved?" check doesn't re-trigger on our own programmatic write. (silent: the
// drag's onPointer / the ambient sweep already redraw -- no extra onChange.)
function setQi(page) {
  const v = Math.round(curQ);
  page.controls.set('qi', v, { silent: true }); lastSliderQ = v;
}

function drawGrid(ctx, rect, N, fn, qi, sinkCols) {
  const cw = rect.w / N, ch = rect.h / N;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const att = fn(i, j);
    ctx.fillStyle = att ? (i === qi ? BLUE : LIGHT) : DARK;
    ctx.fillRect(rect.x + j * cw, rect.y + i * ch, cw - 1, ch - 1);
  }
  if (sinkCols > 0) { ctx.save(); ctx.fillStyle = ORANGE; for (let j = 0; j < sinkCols; j++) ctx.fillRect(rect.x + j * cw, rect.y - 3, cw - 1, 2.5); ctx.restore(); }
  if (qi >= 0 && qi < N) { ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(rect.x - 1, rect.y + qi * ch - 1, N * cw, ch); ctx.restore(); }
}

mount({
  mount: 'body',
  title: 'attention-patterns — who attends to whom',
  blurb: 'The mask decides which keys each query may read. Scrub: full (causal) → sliding-window → hybrid-by-layer → attention-sink. Lit = attended, dark = masked. Drag the current-query handle (◀ on the left) down the rows to move the highlighted query; on sliding-window / sink, drag the window edge to widen or narrow w. Hover any cell for whether it is attended and WHY. The query auto-sweeps down (generation) and the pattern transport loops. Fewer attended keys ⇒ cheaper decode + smaller KV cache.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  animate: true,   // ambient clock drives the query auto-sweep
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 8, max: 16, value: 12, rebuild: false });
    c.slider('w', { label: 'window w (drag the band edge too)', min: 2, max: 8, step: 1, value: 4, rebuild: false });
    c.slider('s', { label: 'sink size s', min: 0, max: 4, step: 1, value: 2, rebuild: false });
    c.slider('qi', { label: 'highlight query i (or drag ◀)', min: 0, max: 15, step: 1, value: 11, rebuild: false });
    c.transport({ compute: buildData, speed: 1.2, loop: true });
  },
  // Direct manipulation: grab the current-query handle and drag down the rows to
  // move the query; or grab the window edge (sliding/sink) to resize w live.
  onPointer: (page, ev) => {
    const st = page.state, N = st.N;
    const kind = (page.step() && page.step().key) || 'full';
    if (ev.type === 'down') {
      grab = null;
      if (!gridRect) return;
      const ch = gridRect.h / N;
      const qiRow = Math.floor(curQ);
      // Window-edge grab zone: the left edge of the lit band on the current row
      // (sliding / sink only). Otherwise default to grabbing the query handle.
      if ((kind === 'sliding' || kind === 'sink')) {
        const edgeJ = qiRow - st.w;                       // first masked col on the row
        const edgeX = gridRect.x + (edgeJ + 1) * (gridRect.w / N);
        const onRow = ev.y >= gridRect.y + qiRow * ch && ev.y < gridRect.y + (qiRow + 1) * ch;
        if (onRow && Math.abs(ev.x - edgeX) < (gridRect.w / N) * 0.6) { grab = 'window'; sweep = false; return; }
      }
      // Query handle: the left gutter (drag region) or anywhere on the grid.
      if (ev.x >= gridRect.x - 24 && ev.x <= gridRect.x + gridRect.w && ev.y >= gridRect.y && ev.y <= gridRect.y + gridRect.h) {
        grab = 'query'; sweep = false;
        curQ = Math.max(0, Math.min(N - 1, (ev.y - gridRect.y) / ch));
        setQi(page);
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down && gridRect) {
      const ch = gridRect.h / N, cw = gridRect.w / N;
      if (grab === 'query') {
        curQ = Math.max(0, Math.min(N - 1, (ev.y - gridRect.y) / ch));
        setQi(page);
      } else if (grab === 'window') {
        const qiRow = Math.floor(curQ);
        const j = (ev.x - gridRect.x) / cw;               // col under the cursor
        page.controls.set('w', Math.max(2, Math.min(8, Math.round(qiRow - j + 1))), { silent: true });
      }
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear('#ffffff');
    const N = st.N, w = st.w, s = st.s;
    const step = page.step(), kind = step ? step.key : 'full';
    const pad = 16, topY = 56;

    // --- query source-of-truth: slider OR drag OR auto-sweep -----------------
    // A manual qi-slider move wins (adopt it + stop sweeping). Otherwise the
    // ambient clock sweeps curQ down the rows and wraps (generation marching).
    if (lastSliderQ != null && st.qi !== lastSliderQ) { curQ = st.qi; sweep = false; }
    lastSliderQ = st.qi;
    if (sweep && !grab) {
      const dt = Math.max(0, Math.min(0.1, page.t - lastT));
      curQ = (curQ + dt * 2.2) % N;                       // ~2.2 rows/sec
    }
    lastT = page.t;
    const qi = Math.min(Math.floor(curQ), N - 1);

    if (kind !== 'hybrid') {
      const fn = makeFn(kind, w, s);
      const cell = Math.max(12, Math.min(34, Math.min((page.W * 0.46) / N, (page.H - topY - 70) / N)));
      const rect = { x: pad + 34, y: topY, w: N * cell, h: N * cell };
      gridRect = rect; gridN = N; gridKind = kind;        // capture for hit-testing
      r.label('keys →', rect.x, topY - 12, { color: '#586069', font: '11px ui-monospace, monospace' });
      r.label('queries ↓', pad, topY - 12, { color: '#586069', font: '11px ui-monospace, monospace' });
      drawGrid(ctx, rect, N, fn, qi, kind === 'sink' ? s : 0);

      // draggable current-query handle (◀) in the left gutter on the lit row
      ctx.save();
      ctx.fillStyle = grab === 'query' ? BLUE : '#3a4047';
      ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('◀', rect.x - 6, rect.y + qi * cell + cell / 2);
      ctx.restore();
      r.label('drag ◀ ↕', pad - 2, rect.y + N * cell + 16, { color: BLUE, font: '9px ui-monospace, monospace' });

      // legend + per-query count on the right
      let lx = rect.x + N * cell + 30, ly = topY + 6;
      const sw = (col, txt) => { ctx.save(); ctx.fillStyle = col; ctx.fillRect(lx, ly - 9, 13, 13); ctx.restore(); r.label(txt, lx + 19, ly + 2, { color: '#3a4047', font: '11px ui-monospace, monospace' }); ly += 22; };
      r.label(PATTERNS.find((p) => p.key === kind).name, lx, ly - 14, { color: INK, font: '13px ui-monospace, monospace' }); ly += 14;
      sw(LIGHT, 'attended  (j allowed)'); sw(BLUE, 'current query row i'); sw(DARK, 'masked');
      if (kind === 'sink') sw(ORANGE, `sink columns (first ${s})`);
      let cnt = 0; for (let j = 0; j < N; j++) if (fn(qi, j)) cnt++;
      ly += 8;
      r.label(`query i=${qi} attends to ${cnt} key${cnt !== 1 ? 's' : ''}`, lx, ly, { color: BLUE, font: '12px ui-monospace, monospace' }); ly += 18;
      const note = kind === 'full' ? `grows with i (here ${qi + 1}); unbounded` : kind === 'sliding' ? `capped at w=${w}, independent of context length` : `≈ w=${w} + sink s=${s}, bounded`;
      r.label(note, lx, ly, { color: '#586069', font: '11px ui-monospace, monospace' }); ly += 20;
      if (kind === 'sliding' || kind === 'sink') r.label('↔ drag the band edge to resize w', lx, ly, { color: ORANGE, font: '10px ui-monospace, monospace' });

      // --- hover-to-inspect: cell -> attended? + WHY for this pattern --------
      if (page.pointer.over && !grab) {
        const hit = cellAt(rect, N, N, page.pointer.x, page.pointer.y);
        if (hit) page.setTip(reasonFor(kind, hit.r, hit.c, w, s));
      }
    } else {
      // hybrid: a row of per-layer mini-grids, alternating global / local.
      // (The N×N hover/drag grid is hidden in this view -- clear its rect.)
      gridRect = null;
      const L = 8, gap = 12;
      const cell = Math.max(5, Math.min(14, Math.min((page.W - 2 * pad - (L - 1) * gap) / (L * N), (page.H - topY - 90) / N)));
      const gw = N * cell, totW = L * gw + (L - 1) * gap, x0 = Math.max(pad, (page.W - totW) / 2);
      r.label('hybrid-by-layer — layers alternate local (SWA) and global (full):', pad, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
      for (let l = 0; l < L; l++) {
        const local = l % 2 === 1, fn = makeFn(local ? 'sliding' : 'full', w, s);
        const rect = { x: x0 + l * (gw + gap), y: topY + 14, w: gw, h: N * cell };
        drawGrid(ctx, rect, N, fn, qi, 0);
        ctx.save(); ctx.fillStyle = local ? ORANGE : BLUE; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
        ctx.fillText('L' + l, rect.x + gw / 2, rect.y - 4);
        ctx.fillStyle = '#586069'; ctx.fillText(local ? 'local' : 'global', rect.x + gw / 2, rect.y + N * cell + 13);
        ctx.restore();
      }
    }

    let o = `attention pattern: ${PATTERNS.find((p) => p.key === kind).name}.  Lit = attended, dark = masked.  N=${N} w=${w} s=${s}  q=${qi}  tier:${r.name}\n`;
    o += step ? step.desc : '(press ▶ or scrub: full → sliding-window → hybrid → sink · drag ◀ to move the query · hover a cell for why)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__apPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?q=N sets the current (highlighted) query row and pauses the auto-sweep.
  if (q.has('q')) { curQ = Math.min(page.state.N - 1, Math.max(0, +q.get('q') | 0)); sweep = false; setQi(page); }
  // ?win=N sets the sliding-window size (and pauses the sweep so the band is stable).
  if (q.has('win')) { sweep = false; page.controls.set('w', Math.max(2, Math.min(8, +q.get('win') | 0))); }
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover) so the
  // attended-reason tooltip path is screenshot-verifiable; pauses the sweep.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; sweep = false;
  }
  // Any deterministic-frame hook pauses the pattern transport too.
  if (q.has('step') || q.has('hover') || q.has('q') || q.has('win')) { sweep = false; if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  // ?hover / ?q without an explicit ?step land on a single-layer pattern (so the
  // N×N hit-test grid is present, not the hybrid multi-grid view).
  else if ((q.has('hover') || q.has('q')) && t && t.current() && t.current().key === 'hybrid') t.seek(0);
  if (q.get('play') === '1' && t) { sweep = true; t.play(); }
  page.redraw();
});
