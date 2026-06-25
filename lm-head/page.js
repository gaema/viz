// lm-head concept page -- the output projection: hidden state -> a logit per
// vocab token, plus tied weights, logit soft-cap, and decode-time slicing.
// Uses the verified framework: layout.mount() + controls + a per-vocab Transport.
//
// Interactive per the framework contract (plan/framework.md): drag any h-strip
// cell ↕ to steer the hidden state and watch every logit (= h · W_lm[v]) and the
// argmax (predicted token) recompute live; hover a logit bar for its full
// derivation (h · W_lm[v] = Σ_d h[d]·W[v,d]), an h cell or a W_lm cell for its
// value; the per-vocab transport fills each logit and auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', RED = '#d1242f', GREY = '#8a939b';
const VOCAB = ['the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast'];
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

// Shared between buildData() (builds H/W + the editable h), draw() (renders +
// captures rects for hit-testing), and onPointer() (drag an h cell). buildData
// runs via the transport rebuild before the matching draw, so `cur` is fresh.
// Drag edits mutate cur.h in place; logits are always recomputed from cur.h so
// the steer survives + every bar + the argmax update live.
let cur = null;
let Hrect = null, hStripRect = null, Wrect = null, barsRect = null;  // captured in draw
let grab = null;                                                     // {d} while dragging an h cell

function buildData(st) {
  const N = st.N | 0, D = st.D | 0, V = Math.min(st.V | 0, VOCAB.length), seed = st.seed | 0;
  const H = seededRandn(seed, [N, D]);          // hidden states [N x D]
  const E = seededRandn(seed + 10, [V, D]);     // tied: the embedding/unembedding table
  const Wsep = seededRandn(seed + 20, [V, D]);  // untied: a separate output matrix
  const last = N - 1;
  const h = Float32Array.from(H.data.subarray(last * D, last * D + D));  // editable hidden state (last row)
  cur = { H, E, Wsep, h, N, D, V };
  return Array.from({ length: V }, (_, v) => ({ v, label: `vocab ${v}: "${VOCAB[v]}" — logit = h · W_lm[${v}]` }));
}

// Recompute logits + argmax from the (possibly steered) cur.h. The transport
// step-list is fixed-length (one entry per vocab row) and independent of h's
// values, so a drag needs no transport rebuild -- just a redraw.
function logitsOf(h, Wlm, V, D, soft, cap) {
  const raw = new Float32Array(V), capped = new Float32Array(V);
  for (let v = 0; v < V; v++) {
    let acc = 0; for (let j = 0; j < D; j++) acc += h[j] * Wlm.data[v * D + j];
    raw[v] = acc; capped[v] = soft ? cap * Math.tanh(acc / cap) : acc;
  }
  let argmax = 0; for (let v = 1; v < V; v++) if (capped[v] > capped[argmax]) argmax = v;
  return { raw, capped, argmax };
}

mount({
  mount: 'body',
  title: 'lm-head — hidden state → vocab logits',
  blurb: 'The output projection: the last hidden state h is dotted with every row of W_lm to give one logit per token; argmax is the prediction. Drag any h cell ↕ to steer the hidden state and watch every logit and the predicted token recompute live; hover a logit bar for its full derivation (h · W_lm[v] = Σ h[d]·W[v,d]), an h or W_lm cell for its value. Tie weights (W_lm = E) to save V·D params, soft-cap to bound logits to ±cap, and in decode slice to the last token so you compute [1×V], not [N×V]. Scrub (or let it play) to fill each logit.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  // Direct manipulation: grab an h-strip cell, drag vertically to steer the
  // hidden state; every logit (= h · W_lm[v]) and the argmax recompute live.
  onPointer: (page, ev) => {
    if (!cur) return;
    const D = cur.D;
    if (ev.type === 'down') {
      grab = null;
      const hit = hStripRect && cellAt(hStripRect, 1, D, ev.x, ev.y);
      if (hit) grab = { d: hit.c };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down && cur.h) {
      cur.h[grab.d] = Math.max(-3, Math.min(3, cur.h[grab.d] - ev.dy * 0.02));  // drag up = larger
      page.redraw();
    }
  },
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 3, max: 6, value: 4 });
    c.stepper('D', { label: 'features (D)', min: 4, max: 8, value: 6 });
    c.stepper('V', { label: 'vocab (V)', min: 4, max: 8, value: 8 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.toggle('tie', { label: 'tie weights (W_lm = E)', value: true });
    c.toggle('softcap', { label: 'soft-cap logits', value: true });
    c.slider('cap', { label: 'cap', min: 1, max: 6, step: 0.5, value: 3 });
    c.toggle('slice', { label: 'slice last token (decode)', value: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.5, loop: true });
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { H, E, Wsep, h, N, D, V } = cur;
    r.clear('#ffffff');
    const Wlm = st.tie ? E : Wsep;
    const cap = st.cap, soft = st.softcap, slice = st.slice;
    const s = page.step(), upto = s ? s.v : V - 1;   // transport: fill logits up to current vocab row
    const last = N - 1;

    // logits recomputed from the live (possibly steered) h: logit[v] = h · W_lm[v]
    const { raw, capped, argmax } = logitsOf(h, Wlm, V, D, soft, cap);

    // ---- layout ----
    const pad = 16, topY = 80;
    const cell = Math.max(13, Math.min(26, Math.min((page.W * 0.30) / D, (page.H - topY - 150) / V)));
    const Hx = pad + 8;
    Hrect = { x: Hx, y: topY, w: D * cell, h: N * cell };

    // H [N x D] — hidden states, last row = the slice source
    r.label('H [N×D] hidden states', Hx, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    r.heatmap(H, { rows: N, cols: D, rect: Hrect, ramp: ramps.diverging, domain: [-maxAbs(H.data), maxAbs(H.data)] });
    r.grid({ stroke: 'rgba(0,0,0,0.10)' });
    ctx.save();
    if (slice) { ctx.fillStyle = 'rgba(255,255,255,0.62)'; ctx.fillRect(Hx, topY, D * cell, last * cell); }  // grey out non-sliced rows
    ctx.strokeStyle = GREEN; ctx.lineWidth = 2.5; ctx.strokeRect(Hx, topY + last * cell, D * cell, cell);    // box the last row
    ctx.fillStyle = GREEN; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(slice ? 'slice → h' : 'h = last row', Hx, topY + N * cell + 14);
    ctx.restore();

    // W_lm [V x D] — unembedding table (= E when tied), aligned with h above it
    const Wx = Hx + D * cell + 120, hY = topY - cell - 10;
    hStripRect = { x: Wx, y: hY, w: D * cell, h: cell };   // the editable h strip (drag target)
    r.label(`h [1×D]`, Wx, hY - 6, { color: GREEN, font: '11px ui-monospace, monospace' });
    r.heatmap({ data: h, rows: 1, cols: D }, { rows: 1, cols: D, rect: hStripRect, ramp: ramps.diverging, domain: [-maxAbs(H.data), maxAbs(H.data)] });
    Wrect = { x: Wx, y: topY, w: D * cell, h: V * cell };
    r.label(st.tie ? 'W_lm [V×D] = E' : 'W_lm [V×D]', Wx, topY - 14, { color: st.tie ? BLUE : INK, font: '12px ui-monospace, monospace' });
    r.heatmap(Wlm, { rows: V, cols: D, rect: Wrect, ramp: ramps.diverging, domain: [-maxAbs(Wlm.data), maxAbs(Wlm.data)] });
    r.grid({ stroke: 'rgba(0,0,0,0.10)' });
    ctx.save();
    for (let v = 0; v < V; v++) {
      const y = topY + v * cell;
      r.label(`"${VOCAB[v]}"`, Wx - 8, y + cell / 2 + 3, { color: v === argmax ? RED : '#586069', font: '10px ui-monospace, monospace', align: 'right' });
      if (s && v === upto) { ctx.strokeStyle = BLUE; ctx.lineWidth = 2; ctx.strokeRect(Wx, y, D * cell, cell); }  // current dot-product row
    }
    ctx.restore();

    // logit bars [V] — one per W_lm row, raw vs capped, ±cap lines, argmax highlighted
    const Bx = Wx + D * cell + 24, Bw = Math.max(120, page.W - (Wx + D * cell + 24) - pad - 10);
    barsRect = { x: Bx, y: topY, w: Bw, h: V * cell };   // bar area (one vocab row per cell of height, for hover hit-test)
    const zx = Bx + Bw * 0.5;                       // zero line (logits are signed)
    const scale = (Bw * 0.5 - 6) / Math.max(maxAbs(raw), soft ? cap : 1e-9);
    r.label('logit  (h · W_lm[v])', Bx, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    ctx.save();
    ctx.font = '9px ui-monospace, monospace';
    // ±cap guide lines when soft-capping
    if (soft) {
      ctx.strokeStyle = 'rgba(209,36,47,0.45)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      for (const sgn of [-1, 1]) { const x = zx + sgn * cap * scale; ctx.beginPath(); ctx.moveTo(x, topY - 2); ctx.lineTo(x, topY + V * cell + 2); ctx.stroke(); }
      ctx.setLineDash([]); ctx.fillStyle = RED; ctx.textAlign = 'center'; ctx.fillText(`±cap=${cap}`, zx + cap * scale, topY - 4);
    }
    ctx.strokeStyle = '#c4ccd3'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(zx, topY - 2); ctx.lineTo(zx, topY + V * cell + 2); ctx.stroke();  // zero line
    for (let v = 0; v < V; v++) {
      const y = topY + v * cell + 3, bh = cell - 6;
      const filled = !s || v <= upto;               // transport reveal
      if (!filled) continue;
      const val = capped[v];
      const isMax = v === argmax;
      // raw extent (faint) when capped differs
      if (soft && Math.abs(raw[v]) > Math.abs(val) + 1e-3) {
        ctx.fillStyle = 'rgba(31,111,235,0.14)';
        const rx = Math.min(zx, zx + raw[v] * scale), rw = Math.abs(raw[v] * scale);
        ctx.fillRect(rx, y, rw, bh);
      }
      ctx.fillStyle = isMax ? RED : 'rgba(31,111,235,0.62)';
      const bx = Math.min(zx, zx + val * scale), bw2 = Math.abs(val * scale);
      ctx.fillRect(bx, y, bw2, bh);
      ctx.fillStyle = isMax ? RED : '#3a4047'; ctx.textAlign = val >= 0 ? 'left' : 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(val.toFixed(2), zx + val * scale + (val >= 0 ? 4 : -4), y + bh / 2);
    }
    ctx.restore();

    // Hover-to-inspect: a logit bar -> its full derivation; an h cell or a W_lm
    // cell -> its value. Hit-test against the captured rects (drag suppresses it).
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      const bh = barsRect && cellAt(barsRect, V, 1, pt.x, pt.y);     // V rows, 1 col
      const hh = hStripRect && cellAt(hStripRect, 1, D, pt.x, pt.y); // 1 row, D cols
      const wh = Wrect && cellAt(Wrect, V, D, pt.x, pt.y);           // V rows, D cols
      let tip = null;
      if (bh) {
        const v = bh.r;
        const terms = [];
        for (let d = 0; d < D; d++) terms.push(`${(h[d] >= 0 ? ' ' : '') + h[d].toFixed(2)}·${(Wlm.data[v * D + d] >= 0 ? ' ' : '') + Wlm.data[v * D + d].toFixed(2)}`);
        const shown = terms.length <= 5 ? terms.join(' + ') : terms.slice(0, 4).join(' + ') + ' + … (' + D + ' terms)';
        tip = `"${VOCAB[v]}" : logit = h · W_lm[${v}] = Σ_d h[d]·W[${v},d]\n= ${shown}\n= ${raw[v].toFixed(3)}`;
        if (soft) tip += `\nsoft-cap: ${cap}·tanh(${raw[v].toFixed(2)}/${cap}) = ${capped[v].toFixed(3)}`;
      } else if (hh) {
        tip = `h[${hh.c}] = ${h[hh.c].toFixed(3)}\ndrag ↕ to steer the hidden state`;
      } else if (wh) {
        tip = `W_lm["${VOCAB[wh.r]}"][${wh.c}] = ${Wlm.data[wh.r * D + wh.c].toFixed(3)}`;
      }
      if (tip) page.setTip(tip);
    }

    // arrow: slice -> h (from H last row to the h strip)
    ctx.save(); ctx.strokeStyle = GREEN; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(Hx + D * cell, topY + last * cell + cell / 2); ctx.lineTo(Wx - 6, hY + cell / 2); ctx.stroke();
    ctx.fillStyle = GREEN; ctx.beginPath(); ctx.moveTo(Wx - 6, hY + cell / 2); ctx.lineTo(Wx - 13, hY + cell / 2 - 4); ctx.lineTo(Wx - 13, hY + cell / 2 + 4); ctx.fill(); ctx.restore();

    // readout
    const savedToy = V * D;
    let o = `predicted next token: "${VOCAB[argmax]}"  (logit ${capped[argmax].toFixed(2)}${soft ? `, raw ${raw[argmax].toFixed(2)}` : ''})    tier:${r.name}\n`;
    o += st.tie
      ? `tied: W_lm reuses the embedding table E → saves V·D = ${savedToy} params here (at real scale e.g. 256k·4096 ≈ 1.05B weights, and half the lm_head memory).`
      : `untied: W_lm is a separate [V×D] matrix → +V·D = ${savedToy} params over tying (full freedom for the output head).`;
    o += `\n` + (soft
      ? `soft-cap on: logit ← ${cap}·tanh(logit/${cap}), bounding every logit to ±${cap} (Gemma 2 caps final logits at 30); large raw logits squash toward the cap.`
      : `soft-cap off: raw logits passed straight to softmax (no bound).`);
    o += `\n` + (slice
      ? `slice on (decode): only the last hidden row → [1×V] logits; ${N}× fewer logit rows than prefilling all ${N} positions.`
      : `slice off (prefill): a logit row for every position → [${N}×V]; decode then needs only the last (the bars shown).`);
    o += `\n(drag any h cell ↕ to steer the hidden state · hover a logit bar / h / W_lm cell to inspect · press ▶ or scrub to fill each logit)`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__lmPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('tie')) { page.controls.set('tie', q.get('tie') !== '0'); }
  if (q.has('cap0')) { page.controls.set('softcap', false); }
  if (q.has('slice0')) { page.controls.set('slice', false); }
  if (q.has('cap')) { page.controls.set('cap', parseFloat(q.get('cap'))); }
  // ?drag=d,val sets h[d] to a value (headless stand-in for a vertical drag on
  // the h strip, since --screenshot has no pointer). e.g. ?drag=2,2.5 → h[2]=2.5.
  if (q.has('drag')) {
    const [d, val] = q.get('drag').split(',');
    if (cur && cur.h && +d >= 0 && +d < cur.h.length) cur.h[+d] = +val;
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover) so
  // the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  page.redraw();
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
});
