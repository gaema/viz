// kv-cache concept page -- the KV cache during autoregressive decode.
// Uses the verified framework: layout.mount() + controls + a per-token
// attention Transport (built here from tensor.dot/softmax/seededRandn),
// drawn with render.heatmap/cell + ctx overlays.
//
// Interactive per the framework contract (plan/framework.md): the decode
// AUTO-PLAYS + LOOPS so the cache visibly fills column-by-column on load
// (the headline -- this page IS the live animation); hover any K / V / attn
// cell for which (pos, layer, dim) it is + its value, or an unfilled cell for
// when it gets written; and DRAG the fill handle horizontally to scrub the
// decode position by hand (how many tokens have been generated = how many
// cache columns are filled).
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { dot, softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e';
const colVec = (M, c) => { const v = new Float32Array(M.rows); for (let r = 0; r < M.rows; r++) v[r] = M.data[r * M.cols + c]; return v; };
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) { const x = Math.abs(a[i]); if (x > m) m = x; } return m; };
const fmtMB = (b) => (b >= 1073741824 ? (b / 1073741824).toFixed(2) + ' GB' : (b / 1048576).toFixed(1) + ' MB');

const DT_BYTES = { fp16: 2, int8: 1, int4: 0.5 };

// Shared between compute(), draw(), and onPointer(). The layer index is a UI
// control (the cache is per-layer; we visualize ONE layer's grids but the
// memory math counts all L layers) -- kept on `cur` so the hover tooltip can
// report K[layer ℓ][pos p][dim r].
let cur = { Q: null, K: null, V: null, d: 0, N: 0, layer: 0 };
let rK = null, rV = null, rW = null;   // band rects, captured in draw for hit-testing
let rHandle = null;                    // the drag-handle hot rect (right edge of the filled region)
let dragging = false;                  // true while scrubbing the decode position by drag

// Optional typed prompt: each word becomes one cached token. When ≥2 words are
// given they set the context length N (and label the columns); else the N stepper.
const promptWords = (s) => (s || '').toLowerCase().trim().split(/\s+/).filter(Boolean).slice(0, 16);
function buildData(st) {
  const d = st.d, pw = promptWords(st.prompt), N = pw.length >= 2 ? pw.length : st.N, seed = st.seed | 0, sq = Math.sqrt(d);
  // Offset the seed by the visualized layer so different layers show different
  // (deterministic) K/V content -- the tooltip's K[layer ℓ] is then truthful.
  const ls = seed + 1000 * ((st.layerSel | 0));
  const Q = seededRandn(ls, [d, N]), K = seededRandn(ls + 1, [d, N]), V = seededRandn(ls + 2, [d, N]);
  cur = { Q, K, V, d, N, words: pw.length >= 2 ? pw : null, layer: (st.layerSel | 0) };
  const steps = [];
  for (let t = 0; t < N; t++) {
    const qt = colVec(Q, t), scores = new Float32Array(t + 1);
    for (let i = 0; i <= t; i++) scores[i] = dot(qt, colVec(K, i)) / sq;
    const w = softmax(scores), out = new Float32Array(d);
    for (let i = 0; i <= t; i++) for (let r = 0; r < d; r++) out[r] += w[i] * V.data[r * N + i];
    let am = 0; for (let i = 1; i <= t; i++) if (w[i] > w[am]) am = i;
    steps.push({ t, weights: w, out, argmax: am, label: `decode token ${t}: q attends over ${t + 1} cached key${t ? 's' : ''}; strongest on pos ${am} (w=${w[am].toFixed(2)})` });
  }
  return steps;
}

// Heatmap one matrix band, outline the active column, veil future columns.
function band(r, M, rect, dom, activeCol, fromVeil, ramp) {
  r.heatmap(M, { rows: M.rows, cols: M.cols, rect, ramp: ramp || ramps.diverging, domain: [-dom, dom] });
  const ctx = r.ctx, cw = rect.w / M.cols;
  if (fromVeil < M.cols) { ctx.save(); ctx.fillStyle = 'rgba(244,246,248,0.88)'; ctx.fillRect(rect.x + fromVeil * cw, rect.y, rect.w - fromVeil * cw, rect.h); ctx.restore(); }
  if (activeCol >= 0 && activeCol < M.cols) { ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(rect.x + activeCol * cw + 1, rect.y + 1, cw - 2, rect.h - 2); ctx.restore(); }
}

function vec(r, v, rect, dom, ramp) {   // single-column vector heatmap
  const M = { data: v, rows: v.length, cols: 1 };
  r.heatmap(M, { rows: v.length, cols: 1, rect, ramp: ramp || ramps.diverging, domain: [-dom, dom] });
  r.grid({ stroke: 'rgba(0,0,0,0.12)' });
}

// Map a pointer x within a band's column span to a decode position 0..N-1.
function posAtX(rect, N, x) {
  const cw = rect.w / N;
  let p = Math.floor((x - rect.x) / cw);
  return Math.max(0, Math.min(N - 1, p));
}

mount({
  mount: 'body',
  title: 'kv-cache — keys & values during decode',
  blurb: 'Autoregressive decode caches past keys/values so each step is O(seq), not O(seq²). It auto-plays: each token appends a K and V column; its query attends over all cached keys (softmax); the output is the weighted sum of cached values. Hover any K/V/attn cell for its (pos, layer, dim) and value; drag the ◂▸ fill handle to scrub how many tokens have been generated. KV memory grows with context.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  compare: { key: 'kvdtype', a: 'fp16', b: 'int4', labelA: 'fp16 KV cache', labelB: 'int4 KV cache (¼ memory)' },
  challenges: [
    { goal: 'Fill the whole cache — scrub to the last decoded token.', hint: 'drag the ◂▸ fill handle to the right end (or let it play to the end).', check: (api) => ({ solved: (api.probe.t ?? -1) === (api.probe.N ?? 0) - 1, detail: `at token ${(api.probe.t ?? 0) + 1} / ${api.probe.N ?? 0}` }) },
    { goal: 'Shrink the KV cache below 0.5 MB.', hint: 'switch the KV dtype to int4 (¼ the bytes), and/or lower the layer count.', check: (api) => ({ solved: (api.probe.mb ?? 9) < 0.5, detail: `KV cache = ${(api.probe.mb ?? 0).toFixed(2)} MB (need < 0.5)` }) },
  ],
  controls: (c, page) => {
    c.text('prompt', { label: 'prompt → tokens (optional)', value: '', placeholder: 'type a sentence…', rebuild: true });
    c.stepper('d', { label: 'head_dim (vis)', min: 4, max: 14, value: 8 });
    c.stepper('N', { label: 'context (tokens, when no prompt)', min: 4, max: 16, value: 10 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.stepper('layerSel', { label: 'show layer ℓ', min: 0, max: 31, value: 1, rebuild: true });
    c.slider('layers', { label: 'layers (memory)', min: 1, max: 80, step: 1, value: 32 });
    c.select('kvdtype', { label: 'KV dtype (quant)', value: 'fp16', options: [{ value: 'fp16', label: 'fp16' }, { value: 'int8', label: 'int8' }, { value: 'int4', label: 'int4' }] });
    c.transport({ compute: () => buildData(page.state), speed: 4, loop: true });
  },
  // Direct manipulation: drag the fill handle (or anywhere on the K/V/weights
  // bands) horizontally to scrub the decode position -- how many cache columns
  // are filled is controlled by hand. Maps straight onto the step transport.
  onPointer: (page, ev) => {
    const t = page.controls._transport;
    if (!t || !cur.N) return;
    const N = cur.N;
    const inBands = (x, y) =>
      (rK && x >= rK.x && x <= rK.x + rK.w && y >= rK.y && y <= rV.y + rV.h);
    if (ev.type === 'down') {
      // Grab if the press is on the handle or anywhere over the band stack.
      const onHandle = rHandle && ev.x >= rHandle.x && ev.x <= rHandle.x + rHandle.w && ev.y >= rHandle.y && ev.y <= rHandle.y + rHandle.h;
      if (onHandle || inBands(ev.x, ev.y)) {
        dragging = true;
        t.pause();
        t.seek(posAtX(rK, N, ev.x));
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragging = false;
    } else if (ev.type === 'move' && dragging && page.pointer.down) {
      t.seek(posAtX(rK, N, ev.x));
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, { Q, K, V, d, N, layer } = cur;
    if (!K) return;
    r.clear('#ffffff');
    const s = page.step();
    const t = s ? s.t : N - 1;
    const w = s ? s.weights : (() => { const a = new Float32Array(N); a[N - 1] = 1; return a; })();
    const out = s ? s.out : colVec(V, N - 1);
    const seq = t + 1;

    // Geometry: K band / weights row / V band share N columns; q on the left, out on the right.
    const pad = 14, labelW = 60, sideW = 54, topY = 58;
    const gridW = page.W - pad - labelW - pad - 2 * sideW - 24;
    const cell = Math.max(9, Math.min(30, Math.min(gridW / N, (page.H - 150) / (2 * d + 1))));
    const x0 = pad + labelW + sideW + 12;
    const yK = topY, yW = yK + d * cell + 8, yV = yW + cell + 8;
    const gw = N * cell;
    const rK_ = { x: x0, y: yK, w: gw, h: d * cell }, rW_ = { x: x0, y: yW, w: gw, h: cell }, rV_ = { x: x0, y: yV, w: gw, h: d * cell };
    rK = rK_; rW = rW_; rV = rV_;
    const domK = maxAbs(K.data), domV = maxAbs(V.data);

    // labels
    const lab = (txt, y, col) => r.label(txt, pad, y, { color: col || '#586069', font: '11px ui-monospace, monospace' });
    lab(`K cache ℓ${layer}`, yK + d * cell / 2, BLUE);
    lab('attn w', yW + cell * 0.7);
    lab(`V cache ℓ${layer}`, yV + d * cell / 2, ORANGE);
    r.label(cur.words ? 'your tokens →' : 'position →', x0, topY - 10, { color: '#9aa4ad', font: '11px ui-monospace, monospace' });
    if (cur.words) { ctx.save(); ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillStyle = '#3a4047'; for (let j = 0; j < N; j++) { const wd = cur.words[j] || ''; ctx.fillText(wd.length > 7 ? wd.slice(0, 6) + '…' : wd, x0 + j * cell + cell / 2, topY - 22); } ctx.restore(); }

    // K / V bands (heatmap full, veil future cols, outline current col t)
    band(r, K, rK, domK, t, seq, ramps.diverging);
    band(r, V, rV, domV, t, seq, ramps.diverging);

    // attention weights row (0..t), veil future
    const wRow = new Float32Array(N); for (let i = 0; i <= t; i++) wRow[i] = w[i];
    const wMax = Math.max(1e-6, ...Array.from(w));
    r.heatmap(wRow, { rows: 1, cols: N, rect: rW, ramp: ramps.sequential, domain: [0, wMax] });
    ctx.save(); ctx.fillStyle = 'rgba(244,246,248,0.88)'; ctx.fillRect(rW.x + seq * cell, rW.y, rW.w - seq * cell, rW.h); ctx.restore();
    if (cell >= 18) for (let i = 0; i <= t; i++) { r.layout = { x: rW.x, y: rW.y, w: rW.w, h: rW.h, rows: 1, cols: N, cellW: cell, cellH: cell }; r.cell(0, i, { stroke: false, label: w[i].toFixed(2), labelColor: w[i] > wMax * 0.6 ? '#fff' : '#333', font: '9px ui-monospace, monospace' }); }
    // column-t outline across weights
    ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(rW.x + t * cell + 1, rW.y + 1, cell - 2, rW.h - 2); ctx.restore();

    // Drag handle: a grabbable bar at the right edge of the filled region,
    // spanning the K..V band stack. Drag it (or the bands) to scrub the
    // decode position by hand.
    const handleX = rK.x + seq * cell;
    rHandle = { x: handleX - 5, y: rK.y, w: 10, h: rV.y + rV.h - rK.y };
    ctx.save();
    ctx.strokeStyle = dragging ? ORANGE : 'rgba(31,111,235,0.85)';
    ctx.lineWidth = dragging ? 3 : 2;
    ctx.beginPath(); ctx.moveTo(handleX, rK.y - 4); ctx.lineTo(handleX, rV.y + rV.h + 4); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('◂▸', handleX, rK.y - 8);
    ctx.restore();

    // query q_t (left of K) and output (right of V)
    const qx = x0 - sideW - 12;
    r.label('q(t)', qx, yK - 8, { color: BLUE, font: '11px ui-monospace, monospace' });
    vec(r, colVec(Q, t), { x: qx, y: yK, w: cell, h: d * cell }, maxAbs(Q.data));
    const ox = x0 + gw + 12;
    r.label('out', ox, yV - 8, { color: ORANGE, font: '11px ui-monospace, monospace' });
    vec(r, out, { x: ox, y: yV, w: cell, h: d * cell }, maxAbs(out));

    // memory bar (grows with context) at the bottom
    const L = st.layers, Hkv = 8, dr = 128, db = DT_BYTES[st.kvdtype];
    const bytesNow = 2 * L * Hkv * dr * db * seq, bytesFull = 2 * L * Hkv * dr * db * N;
    page.probe = { t, N, mb: bytesFull / 1048576 };
    const barY = yV + d * cell + 22, barW = Math.min(360, page.W - 2 * pad - 120), barX = pad + 110;
    r.label('KV memory', pad, barY + 10, { color: '#586069', font: '11px ui-monospace, monospace' });
    ctx.save();
    ctx.strokeStyle = '#d0d7de'; ctx.lineWidth = 1; ctx.strokeRect(barX, barY, barW, 14);
    ctx.fillStyle = 'rgba(31,111,235,0.6)'; ctx.fillRect(barX, barY, barW * (seq / N), 14);
    ctx.restore();
    r.label(`${fmtMB(bytesNow)}  (seq ${seq}/${N})`, barX + barW + 8, barY + 11, { color: INK, font: '11px ui-monospace, monospace' });

    // Hover-to-inspect: which (pos, layer, dim) a K/V/attn cell is + its value;
    // an unfilled cell reports the decode step that writes it.
    if (page.pointer.over && !dragging) {
      const p = page.pointer;
      const kh = cellAt(rK, d, N, p.x, p.y);
      const vh = cellAt(rV, d, N, p.x, p.y);
      const wh = cellAt(rW, 1, N, p.x, p.y);
      let tip = null;
      if (kh) {
        tip = kh.c <= t
          ? `K[layer ${layer}][pos ${kh.c}][dim ${kh.r}] = ${K.data[kh.r * N + kh.c].toFixed(2)}`
          : `K[layer ${layer}][pos ${kh.c}][dim ${kh.r}]  (empty — filled at decode step ${kh.c})`;
      } else if (vh) {
        tip = vh.c <= t
          ? `V[layer ${layer}][pos ${vh.c}][dim ${vh.r}] = ${V.data[vh.r * N + vh.c].toFixed(2)}`
          : `V[layer ${layer}][pos ${vh.c}][dim ${vh.r}]  (empty — filled at decode step ${vh.c})`;
      } else if (wh) {
        tip = wh.c <= t
          ? `attn w[pos ${wh.c}] = ${w[wh.c].toFixed(3)}\n(q(t=${t}) · K[:,${wh.c}] / √d, softmax)`
          : `attn w[pos ${wh.c}]  (empty — pos ${wh.c} not yet decoded; available at step ${wh.c})`;
      }
      if (tip) page.setTip(tip);
    }

    let o = `decode token ${t} — q attends over ${seq} cached key${seq > 1 ? 's' : ''} (softmax) → out = Σ wᵢ·vᵢ    layer ℓ${layer}    tier:${r.name}\n`;
    o += s ? `${s.label}\n` : '(plays on load; scrub or DRAG the ◂▸ handle to set how many tokens are decoded; the cache fills column by column)\n';
    o += `KV cache  2·L=${L}·H_kv=${Hkv}·head_dim=${dr}·${st.kvdtype}(${db}B)·seq=${seq} = ${fmtMB(bytesNow)}  (full ctx ${N}: ${fmtMB(bytesFull)}) — grows linearly; without the cache each step recomputes K,V for all ${seq} tokens`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__kvPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('prompt')) page.controls.set('prompt', q.get('prompt'), { rebuild: true });
  if (q.has('kvdtype')) page.controls.set('kvdtype', q.get('kvdtype'));
  if (q.has('layers')) page.controls.set('layers', +q.get('layers'));
  const t = page.controls._transport;
  // ?pos=N / ?drag=N set the decode position (filled-column count) -- the
  // headless stand-in for a drag of the fill handle, since --screenshot has no
  // pointer. Both names accepted; N is 0-based (pos 0 = first token decoded).
  const posKey = q.has('pos') ? 'pos' : (q.has('drag') ? 'drag' : null);
  if (posKey && t) t.seek(parseInt(q.get(posKey), 10));
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover)
  // so the cell tooltip path is verifiable.
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
