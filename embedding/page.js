// embedding concept page -- token-id row lookup + tied/untied lm_head.
// Uses the verified framework: layout.mount() + controls + a per-token Transport.
//
// Interactive per the framework contract (plan/framework.md): hover any E (or
// untied lm_head) cell for E[token,dim] = value (+ the token label); hover the
// fetched-row cell for its value. Direct manipulation: click/drag a row of the
// embedding table (or drag along the token axis) to PIN which token id is looked
// up -- the highlighted row + the pulled embedding vector update live. The
// per-token sweep auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', PIN = '#0a7227';
const VOCAB = ['the', 'cat', 'sat', 'on', 'mat', 'dog'];   // V = 6
const SENT = ['the', 'cat', 'sat', 'on', 'the', 'mat'];    // input sequence
const V = VOCAB.length, N = SENT.length;
const IDS = SENT.map((w) => VOCAB.indexOf(w));
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

// Shared between compute(), draw(), and onPointer(). compute() runs (via the
// transport rebuild) before the matching draw, so `cur` is fresh. manualId pins
// the looked-up token id under direct manipulation (null = follow the transport).
let cur = null;
let Erect = null, Lrect = null, rowRect = null;   // table + fetched-row rects, captured in draw
let inputRects = [];                              // per-input-token box rects (token-axis drag)
let manualId = null;                              // pinned token id (drag/?tok) or null
let drag = null;                                  // 'E' | 'input' while dragging to set the id

function buildData(st) {
  const D = st.D, seed = st.seed | 0;
  const E = seededRandn(seed, [V, D]), L = seededRandn(seed + 7, [V, D]);
  cur = { E, L, D };
  return IDS.map((id, t) => ({ t, id, label: `token ${t} = "${SENT[t]}" → id ${id} → look up row ${id} of E (a ${D}-dim vector). No math, just indexing.` }));
}

// The token id actually being looked up: the pinned id if set, else the
// transport's current sequence position -> IDS[t].
function lookupId(page) {
  if (manualId != null) return manualId;
  const s = page.step();
  const t = s ? s.t : N - 1;
  return IDS[t];
}

// [V×D] table heatmap + vocab row labels + optional highlighted row
function table(r, M, rect, dom, hiRow, label) {
  const ctx = r.ctx, cell = rect.h / V;
  r.heatmap(M, { rows: V, cols: M.cols, rect, ramp: ramps.diverging, domain: [-dom, dom] });
  r.grid({ stroke: 'rgba(0,0,0,0.10)' });
  r.label(label, rect.x, rect.y - 7, { color: '#586069', font: '11px ui-monospace, monospace' });
  ctx.save(); ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i < V; i++) { ctx.fillStyle = i === hiRow ? BLUE : '#9aa4ad'; ctx.fillText(VOCAB[i], rect.x - 5, rect.y + i * cell + cell / 2); }
  ctx.restore();
  if (hiRow >= 0) {
    ctx.save();
    ctx.strokeStyle = manualId != null ? PIN : INK; ctx.lineWidth = 2.5;
    ctx.strokeRect(rect.x - 1, rect.y + hiRow * cell - 1, M.cols * cell + 2, cell + 2);
    ctx.restore();
  }
}

mount({
  mount: 'body',
  title: 'embedding — token-id → row lookup',
  blurb: 'A token is just an id (an index). Its embedding is a pure row lookup in the table E[V×D] — no arithmetic. Hover any cell for E[token,dim]; click or drag a row of E (or drag along the input-token strip) to pin which token id is looked up — the highlighted row + the pulled vector follow your hand. Let the sequence play (loops), or scrub it. The lm_head that turns hidden states back into vocab logits can reuse E (tied) or be a separate matrix (untied).',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('D', { label: 'dim (D)', min: 4, max: 8, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.toggle('tied', { label: 'tie lm_head to E', value: true });
    c.button('release pin', () => { manualId = null; page.redraw(); });
    c.transport({ compute: () => buildData(page.state), speed: 1.5, loop: true });
  },
  // Direct manipulation: click/drag a row of E (or along the input strip) to pin
  // the looked-up token id. Pause the sweep so the pinned row holds under the hand.
  onPointer: (page, ev) => {
    if (!cur) return;
    const t = page.controls._transport;
    const setFromE = (x, y) => { const h = Erect && cellAt(Erect, V, cur.D, x, y); if (h) { manualId = h.r; return true; } return false; };
    const setFromInput = (x, y) => {
      for (let i = 0; i < inputRects.length; i++) {
        const R = inputRects[i];
        if (R && x >= R.x && x <= R.x + R.w && y >= R.y && y <= R.y + R.h) { manualId = IDS[i]; return true; }
      }
      return false;
    };
    if (ev.type === 'down') {
      drag = null;
      if (setFromE(ev.x, ev.y)) { drag = 'E'; if (t) t.pause(); }
      else if (setFromInput(ev.x, ev.y)) { drag = 'input'; if (t) t.pause(); }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      if (drag === 'E') setFromE(ev.x, ev.y);
      else if (drag === 'input') setFromInput(ev.x, ev.y);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { E, L, D } = cur;
    r.clear('#ffffff');
    const s = page.step();
    const t = s ? s.t : N - 1, tied = st.tied;
    const id = lookupId(page);          // pinned id, else IDS[t]
    const pinned = manualId != null;

    const pad = 16, inputY = 46, tableY = 110;
    const cell = Math.max(14, Math.min(30, (page.W - 2 * pad - 130) / (2 * D)));
    const Ex = pad + 46;
    Erect = { x: Ex, y: tableY, w: D * cell, h: V * cell };
    const Lx = Ex + D * cell + 72;
    Lrect = { x: Lx, y: tableY, w: D * cell, h: V * cell };
    const dE = maxAbs(E.data), dL = tied ? dE : maxAbs(L.data);

    // input token sequence (top), arrow from the looked-up token to its row
    r.label('input tokens (ids)', pad, inputY - 14, { color: '#586069', font: '11px ui-monospace, monospace' });
    const bw = Math.min(64, (Erect.w + 60) / N - 6);
    inputRects = [];
    for (let i = 0; i < N; i++) {
      const bx = Ex + i * ((Erect.w + 60) / N);
      inputRects[i] = { x: bx, y: inputY, w: bw, h: 26 };
      // highlight the box whose id is currently looked up (sweep position, or all
      // matching ids when pinned to a token that appears more than once).
      const cur2 = pinned ? IDS[i] === id : i === t;
      ctx.save();
      ctx.fillStyle = cur2 ? 'rgba(31,111,235,0.16)' : '#f2f4f7'; ctx.strokeStyle = cur2 ? BLUE : '#d0d7de'; ctx.lineWidth = cur2 ? 2 : 1;
      ctx.fillRect(bx, inputY, bw, 26); ctx.strokeRect(bx, inputY, bw, 26);
      ctx.fillStyle = cur2 ? BLUE : '#3a4047'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`${SENT[i]}·${IDS[i]}`, bx + bw / 2, inputY + 13);
      if (cur2 && IDS[i] === id) { ctx.strokeStyle = pinned ? PIN : BLUE; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(bx + bw / 2, inputY + 26); ctx.lineTo(Erect.x - 2, Erect.y + id * cell + cell / 2); ctx.stroke(); }
      ctx.restore();
    }

    // embedding table E + highlighted row
    table(r, E, Erect, dE, id, `E (embedding table) [${V}×${D}]`);

    // pulled embedding vector (the highlighted row), below E
    const stripY = tableY + V * cell + 30;
    const row = E.data.subarray(id * D, id * D + D);
    rowRect = { x: Ex, y: stripY, w: D * cell, h: cell };
    r.label(`embedding["${VOCAB[id]}"] = E[${id}]  (one row)${pinned ? '   ⟵ pinned' : ''}`, Ex, stripY - 7, { color: pinned ? PIN : INK, font: '11px ui-monospace, monospace' });
    r.heatmap(row, { rows: 1, cols: D, rect: rowRect, ramp: ramps.diverging, domain: [-dE, dE] });
    r.grid({ stroke: 'rgba(0,0,0,0.12)' });

    // lm_head (right): tied = E, untied = separate L
    table(r, tied ? E : L, Lrect, dL, -1, tied ? `lm_head = Eᵀ  (tied → reuses E)` : `lm_head (untied, separate) [${V}×${D}]`);
    ctx.save(); ctx.fillStyle = tied ? '#0a7227' : '#b3261e'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
    const VD = V * D;
    ctx.fillText(tied ? `params = V·D = ${VD}` : `params = 2·V·D = ${2 * VD}`, Lrect.x + Lrect.w / 2, Lrect.y + Lrect.h + 20);
    ctx.restore();

    // Hover-to-inspect: E cell -> E[token,dim] + token label; untied L cell ->
    // lm_head[token,dim]; fetched-row cell -> its value.
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      const eh = Erect && cellAt(Erect, V, D, p.x, p.y);
      const lh = (!tied) && Lrect && cellAt(Lrect, V, D, p.x, p.y);
      const rh = rowRect && cellAt(rowRect, 1, D, p.x, p.y);
      let tip = null;
      if (eh) tip = `E["${VOCAB[eh.r]}" id ${eh.r}][dim ${eh.c}]\n= ${E.data[eh.r * D + eh.c].toFixed(3)}\nclick row to pin lookup`;
      else if (lh) tip = `lm_head["${VOCAB[lh.r]}" id ${lh.r}][dim ${lh.c}]\n= ${L.data[lh.r * D + lh.c].toFixed(3)}`;
      else if (rh) tip = `embedding["${VOCAB[id]}"][dim ${rh.c}]\n= E[${id}][${rh.c}] = ${E.data[id * D + rh.c].toFixed(3)}`;
      if (tip) page.setTip(tip);
    }

    let o = `embedding(id) = E[id] — a pure row lookup (no arithmetic).    tied lm_head = Eᵀ: ${VD} params  ·  untied: ${2 * VD} params (saves V·D by tying)    tier:${r.name}\n`;
    if (pinned) o += `pinned: looking up id ${id} = "${VOCAB[id]}" (drag a row of E or the input strip to change · "release pin" to resume the sweep)`;
    else o += s ? s.label : '(press ▶ or scrub the sequence — each token id indexes one row of E · click a row of E to pin)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__embPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('tied')) page.controls.set('tied', q.get('tied') !== '0');
  // ?tok=N pins the looked-up token id to row N of E (headless stand-in for a
  // row click/drag, since --screenshot has no pointer). Clamped to [0, V-1].
  if (q.has('tok')) { manualId = Math.max(0, Math.min(V - 1, parseInt(q.get('tok'), 10) | 0)); }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover) so
  // the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the sweep for any of these hooks.
  if (q.has('step') || q.has('hover') || q.has('tok')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
