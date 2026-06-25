// causal-mask concept page -- lower-triangular masking; why decode only sees
// the past. Uses the verified framework: layout.mount() + controls + a
// per-query-position Transport.
//
// Interactive per the framework contract (plan/framework.md): the decode
// position AUTO-PLAYS + LOOPS, sweeping down the query rows (the live animation
// of generation advancing). Hover any mask cell for query i → key j: BLOCKED
// (j>i, future token) or allowed (j≤i). DRAG the current-query handle down the
// diagonal to move which row is "current"; the causal frontier (the visible key
// prefix) updates live, making the "only attend to the past" rule tactile.
import { mount } from '../framework/layout.js';
import { cellAt } from '../framework/render.js';

const INK = '#111', BLUE = '#1f6feb', RED = '#d6273a';
const SENT = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'by', 'noon'];

let cur = null;
// Hit-test rects captured in draw(): the N×N mask grid + its cell size, and the
// current-query drag handle (a tab on the diagonal of the active row).
let gridRect = null, gridCell = 0;
let handleRect = null;       // grabbable tab at the [qi,qi] diagonal of the row
let dragging = false;        // true while scrubbing the query position by drag

function buildData(st) {
  const N = st.N;
  cur = { N, toks: SENT.slice(0, N) };
  return Array.from({ length: N }, (_, i) => ({
    i,
    label: i < N - 1
      ? `query "${SENT[i]}" (pos ${i}) sees keys 0..${i}; it predicts "${SENT[i + 1]}" (pos ${i + 1}) — masked, so it can't read the answer`
      : `query "${SENT[i]}" (pos ${i}) sees all ${N} keys (last position)`,
  }));
}

// Map a pointer y within the grid's row span to a query position 0..N-1 -- the
// drag scrubs vertically down the rows (each row = one decode step).
function qAtY(rect, N, y) {
  const ch = rect.h / N;
  return Math.max(0, Math.min(N - 1, Math.floor((y - rect.y) / ch)));
}

mount({
  mount: 'body',
  title: 'causal-mask — attend only to the past',
  blurb: 'An autoregressive LM predicts the next token, so query position i must not see positions j > i (the future, including the token it is predicting). That is the lower-triangular causal mask. It auto-plays: the current decode position sweeps down the rows, the visible key prefix growing each step. Hover a cell to see whether query i → key j is allowed (j≤i) or BLOCKED (j>i, a future token); drag the ▸ handle down the diagonal to move which row is "current" and watch the causal frontier follow.',
  prefer: 'canvas2d',
  aspect: '8 / 5',
  autoplay: true,
  challenges: [
    { goal: 'Scrub to the LAST query position — the only row that attends to every token.', hint: 'drag the ▸ handle (or let it play) down to the bottom row.', check: (api) => ({ solved: api.probe.qi === (api.probe.N ?? 1) - 1, detail: `query at position ${api.probe.qi ?? '–'} / ${(api.probe.N ?? 1) - 1}` }) },
  ],
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 4, max: 8, value: 6 });
    c.transport({ compute: () => buildData(page.state), speed: 1.5, loop: true });
  },
  // Direct manipulation: drag the current-query handle (or anywhere on the grid)
  // vertically to scrub which query row is "current". The causal frontier — the
  // visible key prefix 0..qi — follows the hand, so "only attend to the past"
  // becomes a thing you move. Maps straight onto the step transport.
  onPointer: (page, ev) => {
    const t = page.controls._transport;
    if (!t || !cur) return;
    const { N } = cur;
    const onGrid = (x, y) => gridRect && x >= gridRect.x && x <= gridRect.x + gridRect.w && y >= gridRect.y && y <= gridRect.y + gridRect.h;
    const onHandle = (x, y) => handleRect && x >= handleRect.x && x <= handleRect.x + handleRect.w && y >= handleRect.y && y <= handleRect.y + handleRect.h;
    if (ev.type === 'down') {
      if (onHandle(ev.x, ev.y) || onGrid(ev.x, ev.y)) {
        dragging = true;
        t.pause();
        t.seek(qAtY(gridRect, N, ev.y));
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragging = false;
    } else if (ev.type === 'move' && dragging && page.pointer.down) {
      t.seek(qAtY(gridRect, N, ev.y));
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { N, toks } = cur;
    r.clear('#ffffff');
    const s = page.step();
    const qi = s ? s.i : N - 1;
    page.probe = { qi, N };

    const pad = 18, leftW = 56, headerH = 50, keyRowH = 22;
    const gridX = pad + leftW, gridY = headerH + keyRowH;
    const cell = Math.max(16, Math.min(48, Math.min((page.W - gridX - pad - 168) / N, (page.H - gridY - 60) / N)));
    gridRect = { x: gridX, y: gridY, w: N * cell, h: N * cell };  // capture for hit-testing
    gridCell = cell;

    r.label('keys → (positions a query may read)', gridX, headerH - 14, { color: '#586069', font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    for (let j = 0; j < N; j++) { ctx.fillStyle = j <= qi ? INK : '#c4ccd3'; ctx.fillText(toks[j], gridX + j * cell + cell / 2, gridY - 6); }
    ctx.restore();

    // left query token labels
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) { ctx.fillStyle = i === qi ? BLUE : '#586069'; ctx.font = (i === qi ? 'bold ' : '') + '11px ui-monospace, monospace'; ctx.fillText(toks[i], gridX - 8, gridY + i * cell + cell / 2); }
    ctx.restore();
    r.label('queries ↓', pad, gridY - 6, { color: '#586069', font: '11px ui-monospace, monospace' });

    // grid cells
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = gridX + j * cell, y = gridY + i * cell, allowed = j <= i;
      ctx.fillStyle = allowed ? (i === qi ? BLUE : '#dbeafe') : 'rgba(55,60,68,0.82)';
      ctx.fillRect(x, y, cell - 1.5, cell - 1.5);
      if (allowed && j === i) { ctx.fillStyle = i === qi ? '#fff' : '#7aa6e6'; ctx.beginPath(); ctx.arc(x + (cell - 1.5) / 2, y + (cell - 1.5) / 2, Math.max(2, cell * 0.09), 0, 7); ctx.fill(); }
    }
    // current row outline + the visible-prefix "frontier" emphasis (keys 0..qi)
    ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(gridX - 1, gridY + qi * cell - 1, N * cell, cell); ctx.restore();
    // the token it predicts (qi+1) is masked -> outline red
    if (qi < N - 1) { ctx.save(); ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.strokeRect(gridX + (qi + 1) * cell + 1, gridY + qi * cell + 1, cell - 3.5, cell - 3.5); ctx.restore(); }

    // --- draggable CURRENT-QUERY handle: a ▸ tab on the diagonal of the active
    // row. Drag it down the diagonal to move which query row is "current". ---
    const hx = gridX + qi * cell, hy = gridY + qi * cell;
    handleRect = { x: hx - 14, y: hy - 3, w: cell + 18, h: cell + 6 };
    ctx.save();
    ctx.strokeStyle = dragging ? RED : 'rgba(31,111,235,0.95)';
    ctx.lineWidth = dragging ? 3 : 2;
    ctx.strokeRect(hx + 0.5, hy + 0.5, cell - 2.5, cell - 2.5);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = (cell >= 22 ? '13px' : '10px') + ' ui-monospace, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('▸', hx - 4, hy + cell / 2);     // grab tab on the row's left
    ctx.restore();

    // legend (right of grid)
    const lx = gridX + N * cell + 22; let ly = gridY + 4;
    const sw = (col, txt) => { ctx.save(); ctx.fillStyle = col; ctx.fillRect(lx, ly - 9, 13, 13); ctx.restore(); r.label(txt, lx + 19, ly + 2, { color: '#3a4047', font: '11px ui-monospace, monospace' }); ly += 24; };
    sw('#dbeafe', 'visible  j ≤ i'); sw(BLUE, 'current query row'); sw('rgba(55,60,68,0.82)', 'masked  j > i → −∞');
    ctx.save(); ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.strokeRect(lx, ly - 9, 13, 13); ctx.restore();
    r.label('the next token (predicted, masked)', lx + 19, ly + 2, { color: '#3a4047', font: '11px ui-monospace, monospace' });
    ly += 30;
    r.label('▸ drag the handle down', lx, ly + 2, { color: BLUE, font: '10px ui-monospace, monospace' });
    r.label('the diagonal to move "now"', lx, ly + 16, { color: BLUE, font: '10px ui-monospace, monospace' });

    // --- hover-to-inspect: query i → key j, allowed vs BLOCKED (+ post-mask) ---
    if (page.pointer.over && !dragging) {
      const p = page.pointer;
      const mh = cellAt(gridRect, N, N, p.x, p.y);
      if (mh) {
        const i = mh.r, j = mh.c;
        const qtok = toks[i], ktok = toks[j];
        let tip;
        if (j <= i) {
          // visible cell: this key contributes to query i's attention.
          tip = `query ${i} "${qtok}" → key ${j} "${ktok}": allowed (j≤i)\nin query ${i}'s visible prefix (keys 0..${i}); score kept, softmax weight > 0`;
        } else {
          // masked cell: future token -> −∞ -> 0 weight after softmax.
          const why = (j === i + 1) ? ', the very token query ' + i + ' is predicting' : '';
          tip = `query ${i} "${qtok}" → key ${j} "${ktok}": BLOCKED (j>i, future token${why})\nscore set to −∞ → softmax weight = 0 (can't read the future)`;
        }
        page.setTip(tip);
      }
    }

    let o = `causal mask: query i may attend to keys j ≤ i only — the lower triangle. Upper triangle (future) → −∞ → 0 after softmax.    tier:${r.name}\n`;
    o += s ? s.label : '(plays + loops on load; hover a cell for allowed/BLOCKED · drag the ▸ handle down the diagonal to set the current query — the visible prefix follows)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__cmPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?q=N / ?pos=N set the current query position (which row is "now") -- the
  // headless stand-in for a vertical drag of the ▸ handle, since --screenshot
  // has no pointer. Both names accepted; N is 0-based (q 0 = first query row).
  const posKey = q.has('q') ? 'q' : (q.has('pos') ? 'pos' : null);
  if (posKey && t) t.seek(parseInt(q.get(posKey), 10));
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover)
  // so the allowed/BLOCKED tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('hover') || posKey) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
