// multi-head concept page -- split D into H heads, attend in parallel, concat.
// Uses the verified framework: layout.mount() + controls + a per-head Transport.
//
// Interactive per the framework contract (plan/framework.md): the per-head
// attention matrices auto-play + loop (scrub walks head by head); hover any
// per-head attention cell for its derivation (head h: score[i,j] = qᵢ·kⱼ/√d_head),
// or hover a concat-output cell to see which head + dim it came from; drag a Q
// cell (in one head's slice) vertically to change that query component and watch
// THAT head's attention row and the concatenated output slice recompute live.
import { mount } from '../framework/layout.js';
import { ramps, categorical, cellAt } from '../framework/render.js';
import { dot, softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111';
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) { const x = Math.abs(a[i]); if (x > m) m = x; } return m; };

// Shared between buildData(), draw(), and onPointer(). recompute() rebuilds the
// per-head attention weights (raw score -> /√d_head -> causal mask -> softmax)
// and the concatenated output from cur.Q/K/V, so a drag edit of a Q cell flows
// all the way through its head's attention row and into the output slice.
let cur = null;
// Hit-test rects captured in draw(): the Q input strip [N×D], the H per-head
// attention matrices [N×N] (headRects[h]), and the concat output strip [N×D].
let qRect = null, headRects = [], outRect = null;
let grab = null;   // {i, c} (c is a GLOBAL column 0..D-1) while dragging a Q cell

// Recompute per-head attention (W[h]) and the concat output from cur.Q/K/V in
// place. Called by buildData() (fresh) and by resync() after a Q-cell drag.
function recompute() {
  const { Q, K, V, heads, output, N, H, hd, D } = cur, sq = Math.sqrt(hd);
  output.data.fill(0);
  for (let h = 0; h < H; h++) {
    const off = h * hd, W = heads[h].W, raw = heads[h].raw;
    for (let i = 0; i < N; i++) {
      const r = new Float32Array(N);
      for (let j = 0; j < N; j++) {
        const s = dot(Q.data.subarray(i * D + off, i * D + off + hd), K.data.subarray(j * D + off, j * D + off + hd));
        raw.data[i * N + j] = s;                       // un-scaled qᵢ·kⱼ (for the tooltip)
        r[j] = j > i ? -Infinity : s / sq;             // scaled + causal mask
      }
      const w = softmax(r);
      for (let j = 0; j < N; j++) W.data[i * N + j] = w[j];
    }
    for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
      const w = W.data[i * N + j];
      for (let c = 0; c < hd; c++) output.data[i * D + off + c] += w * V.data[j * D + off + c];
    }
  }
}

function buildData(st) {
  const N = st.N, H = st.H, hd = st.hd, D = H * hd, seed = st.seed | 0;
  const Q = seededRandn(seed, [N, D]), K = seededRandn(seed + 1, [N, D]), V = seededRandn(seed + 2, [N, D]);
  const heads = Array.from({ length: H }, () => ({
    W: { data: new Float32Array(N * N), rows: N, cols: N },
    raw: { data: new Float32Array(N * N), rows: N, cols: N },
  }));
  cur = { Q, K, V, heads, output: { data: new Float32Array(N * D), rows: N, cols: D }, N, H, hd, D };
  recompute();
  return Array.from({ length: H }, (_, h) => ({ h, label: `head ${h}: attends within head_dim slice [${h * hd}:${(h + 1) * hd}) — its own QₕKₕᵀ pattern` }));
}

// Recompute after a Q-cell drag, keeping the transport's per-head axis valid
// (the head list itself never changes -- only the numbers).
function resync(page) {
  recompute();
  const t = page.controls._transport;
  if (t) t._sync();
}

// [N×D] strip heatmap + head-slice dividers + labels + active-head highlight.
function dstrip(r, M, rect, hd, H, label, ah) {
  const ctx = r.ctx, N = M.rows, D = M.cols, cw = rect.w / D;
  r.heatmap(M, { rows: N, cols: D, rect, ramp: ramps.diverging, domain: [-maxAbs(M.data), maxAbs(M.data)] });
  r.label(label, rect.x - 26, rect.y + rect.h / 2, { color: '#586069', font: '11px ui-monospace, monospace' });
  ctx.save();
  for (let h = 0; h <= H; h++) {                      // slice dividers
    const x = rect.x + h * hd * cw;
    ctx.strokeStyle = h === 0 || h === H ? '#c4ccd3' : 'rgba(40,44,52,0.5)'; ctx.lineWidth = h === 0 || h === H ? 1 : 1.5;
    ctx.beginPath(); ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h); ctx.stroke();
  }
  ctx.textAlign = 'center'; ctx.font = '9px ui-monospace, monospace';
  for (let h = 0; h < H; h++) { ctx.fillStyle = h === ah ? rgb(categorical(h)) : '#9aa4ad'; ctx.fillText('h' + h, rect.x + (h + 0.5) * hd * cw, rect.y - 4); }
  if (ah >= 0) { ctx.strokeStyle = rgb(categorical(ah)); ctx.lineWidth = 2.5; ctx.strokeRect(rect.x + ah * hd * cw + 1, rect.y + 1, hd * cw - 2, rect.h - 2); }
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'multi-head — split, attend in parallel, concat',
  blurb: 'The model dim D is sliced into H heads of head_dim = D/H. Each head attends only within its slice, so the H attention matrices differ; the per-head outputs concatenate back to [N×D]. It auto-plays head by head; hover any per-head attention cell for its derivation (head h: score[i,j] = qᵢ·kⱼ/√d_head), or hover a concat-output cell to see which head + dim it came from; drag a Q cell (in one head’s slice) vertically to change that query component and watch that head’s attention row and the concat output slice recompute live.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 3, max: 7, value: 5 });
    c.stepper('H', { label: 'heads (H)', min: 2, max: 6, value: 4 });
    c.stepper('hd', { label: 'head_dim', min: 2, max: 5, value: 3 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 2, loop: true });
  },
  // Direct manipulation: grab a Q cell and drag vertically to change it. The Q
  // strip is [N×D]; the column maps to a head (head = floor(c / hd)), so a drag
  // re-steers exactly that head's attention row + the matching output slice.
  onPointer: (page, ev) => {
    if (!cur) return;
    const { N, D } = cur;
    if (ev.type === 'down') {
      grab = null;
      const qh = qRect && cellAt(qRect, N, D, ev.x, ev.y);
      if (qh) grab = { i: qh.r, c: qh.c };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const idx = grab.i * D + grab.c;
      cur.Q.data[idx] = Math.max(-3, Math.min(3, cur.Q.data[idx] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { Q, K, V, heads, output, N, H, hd, D } = cur;
    r.clear('#ffffff');
    const s = page.step();
    const ah = s ? s.h : -1;
    const sq = Math.sqrt(hd);

    const pad = 14, topY = 52;
    const cellS = Math.max(8, Math.min(18, (page.W - 2 * pad - 90) / D));
    qRect = { x: pad + 40, y: topY, w: D * cellS, h: N * cellS };
    r.label('split D into H heads (head_dim slices)', pad, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    dstrip(r, Q, qRect, hd, H, 'Q', ah);
    r.label('↕ drag Q', qRect.x + qRect.w + 8, qRect.y + qRect.h - 2, { color: '#1f6feb', font: '9px ui-monospace, monospace' });

    // per-head attention matrices [N×N] in a row
    const mTop = topY + N * cellS + 42, gapM = 14;
    const cellA = Math.max(10, Math.min(28, Math.min((page.W - 2 * pad - 36 - (H - 1) * gapM) / (H * N), (page.H - mTop - N * cellS - 80) / N)));
    const mw = N * cellA;
    const totW = H * mw + (H - 1) * gapM, mX0 = Math.max(pad + 30, (page.W - totW) / 2);
    r.label('H attention matrices — one per head, each a different pattern (causal)', pad, mTop - 10, { color: '#586069', font: '11px ui-monospace, monospace' });
    headRects = [];
    for (let h = 0; h < H; h++) {
      const rect = { x: mX0 + h * (mw + gapM), y: mTop, w: mw, h: N * cellA };
      headRects.push(rect);
      r.heatmap(heads[h].W, { rows: N, cols: N, rect, ramp: ramps.sequential, domain: [0, maxAbs(heads[h].W.data)] });
      ctx.save();
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { ctx.fillStyle = 'rgba(40,44,52,0.14)'; ctx.fillRect(rect.x + j * cellA, rect.y + i * cellA, cellA, cellA); }
      ctx.fillStyle = h === ah ? rgb(categorical(h)) : '#9aa4ad'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText('head ' + h, rect.x + mw / 2, rect.y - 4);
      if (h === ah) { ctx.strokeStyle = rgb(categorical(h)); ctx.lineWidth = 2.5; ctx.strokeRect(rect.x - 1, rect.y - 1, mw + 2, N * cellA + 2); }
      ctx.restore();
    }

    // concatenated output [N×D]
    const oTop = mTop + N * cellA + 40;
    r.label('concat head outputs → [N×D]', pad, oTop - 12, { color: INK, font: '12px ui-monospace, monospace' });
    outRect = { x: pad + 40, y: oTop, w: D * cellS, h: N * cellS };
    dstrip(r, output, outRect, hd, H, 'out', ah);

    // --- hover-to-inspect: per-head attention cell -> derivation; output cell ->
    // which head + dim it came from. (Skip while dragging a Q cell.)
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      let tip = null;
      // Q cell first (the draggable operand).
      const qh = qRect && cellAt(qRect, N, D, p.x, p.y);
      if (qh) {
        const h = Math.floor(qh.c / hd), c = qh.c % hd;
        tip = `Q[${qh.r},${qh.c}] = ${Q.data[qh.r * D + qh.c].toFixed(3)}\nhead ${h}, dim ${c} of its head_dim slice\ndrag ↕ to change the query`;
      } else {
        // per-head attention cell?
        for (let h = 0; h < H && !tip; h++) {
          const mh = headRects[h] && cellAt(headRects[h], N, N, p.x, p.y);
          if (mh && mh.c <= mh.r) {                       // visible (unmasked) cell
            const i = mh.r, j = mh.c, off = h * hd;
            const sc = heads[h].raw.data[i * N + j];
            tip = `head ${h}: score[${i},${j}] = q${i}·k${j}/√d_head\n= ${sc.toFixed(3)} / ${sq.toFixed(3)} = ${(sc / sq).toFixed(3)}\nw[${i},${j}] = ${heads[h].W.data[i * N + j].toFixed(3)} (softmax share, slice [${off}:${off + hd}))`;
          }
        }
        // concat output cell?
        if (!tip) {
          const oh = outRect && cellAt(outRect, N, D, p.x, p.y);
          if (oh) {
            const i = oh.r, gc = oh.c, h = Math.floor(gc / hd), c = gc % hd, off = h * hd;
            const terms = [];
            for (let j = 0; j <= i; j++) terms.push(`${heads[h].W.data[i * N + j].toFixed(2)}·${V.data[j * D + off + c].toFixed(2)}`);
            const shown = terms.length <= 3 ? terms.join(' + ') : terms.slice(0, 3).join(' + ') + ' + … (' + (i + 1) + ' terms)';
            tip = `out[${i},${gc}] ← head ${h}, dim ${c}\n= Σⱼ wₕ[${i},j]·v[j,${off + c}]\n= ${shown}\n= ${output.data[i * D + gc].toFixed(3)}`;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    let o = `multi-head: D=${D} = H=${H} × head_dim=${hd}.  Each head attends in its own slice; outputs concat to [N×${D}].  tier:${r.name}\n`;
    o += s ? `${s.label}` : '(drag a Q cell to edit · press ▶ or scrub to walk head by head — see each head’s distinct attention pattern)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__mhPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=head,i,c,val sets Q[i, head*hd + c] = val (headless stand-in for a
  // vertical Q-cell drag, since --screenshot has no pointer). `head` selects the
  // head slice, `c` is the dim WITHIN that head's head_dim (0..hd-1). e.g.
  // ?drag=1,0,2,2.5 -> head 1, query row 0, slice-dim 2 -> Q[0, 1*hd+2] = 2.5.
  if (q.has('drag') && cur) {
    const [h, i, c, v] = q.get('drag').split(',').map(Number);
    const gc = h * cur.hd + c;
    if (i >= 0 && i < cur.N && gc >= 0 && gc < cur.D) { cur.Q.data[i * cur.D + gc] = v; resync(page); }
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover) so
  // the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  // Park on a deterministic head so the captured frame is stable: ?drag lands on
  // the dragged head (so its recomputed attention + the highlighted slice show);
  // ?hover lands on head 0. An explicit ?step overrides both.
  if (!q.has('step') && t) {
    if (q.has('drag')) t.seek(Math.max(0, Math.min((cur ? cur.H : 1) - 1, parseInt(q.get('drag').split(',')[0], 10) || 0)));
    else if (q.has('hover')) t.seek(0);
  }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
