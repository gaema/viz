// prefill-vs-decode concept page -- the two autoregressive inference regimes.
// Prefill: all N prompt tokens in one parallel GEMM, full N×N causal attention
// (compute-bound). Decode: one 1×D query per step (a GEMV) that re-reads the
// whole KV cache to attend over the past, appends a row, repeats (memory-bound).
// The same causal softmax(QKᵀ/√d) is really computed; the difference is the
// SHAPE of the work. Interactive: drag the timeline to scrub steps, hover any
// cell, the prefill->decode loop auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', ORANGE = '#d2691e', GREY = '#8a939b';
const M = (r, c) => ({ data: new Float32Array(r * c), rows: r, cols: c });
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };
const dotRow = (A, i, B, j, D) => { let s = 0; for (let k = 0; k < D; k++) s += A.data[i * D + k] * B.data[j * D + k]; return s; };

let cur = null;
let cacheRect = null, prefRect = null, barRect = null, tlRect = null;  // captured in draw
let dragTL = false;

function buildData(st) {
  const N = st.N | 0, G = st.G | 0, D = st.D | 0, seed = st.seed | 0;
  const Q = seededRandn(seed, [N + G, D], { std: 1 });
  const K = seededRandn(seed + 1, [N + G, D], { std: 1 });
  cur = { Q, K, N, G, D };
  return [{ phase: 'prefill' }, ...Array.from({ length: G }, (_, t) => ({ phase: 'decode', t: t + 1 }))];
}

// Prefill: [N×N] causal attention, all rows at once.
function prefillAttn(Q, K, N, D) {
  const scale = 1 / Math.sqrt(D), A = M(N, N);
  for (let i = 0; i < N; i++) {
    const row = new Float32Array(N);
    for (let j = 0; j < N; j++) row[j] = j <= i ? dotRow(Q, i, K, j, D) * scale : -Infinity;
    A.data.set(softmax(row), i * N);
  }
  return A;
}
// Decode at position p: one query's softmax over keys 0..p -> [p+1].
function decodeScores(Q, K, p, D) {
  const scale = 1 / Math.sqrt(D), row = new Float32Array(p + 1);
  for (let j = 0; j <= p; j++) row[j] = dotRow(Q, p, K, j, D) * scale;
  return softmax(row);
}

mount({
  mount: 'body',
  title: 'prefill-vs-decode — one big GEMM vs a GEMV per token',
  blurb: 'Two regimes of autoregressive inference. PREFILL runs the whole N-token prompt in one parallel pass — a big matmul and the full N×N causal attention; lots of FLOPs at once ⇒ compute-bound. DECODE emits one token per step: a skinny 1×D query (a GEMV) that re-reads the entire KV cache, attends over every past token, appends a row, and repeats ⇒ memory-bound (the math units sit idle). Press play to run prefill then the decode loop; drag the timeline handle to scrub; hover any cell.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  challenges: [
    { goal: 'Step into the DECODE phase — emit the first generated token.', hint: 'play past the prefill pass, or scrub the timeline into the decode loop.', check: (api) => ({ solved: api.probe.phase === 'decode', detail: `current phase: ${api.probe.phase ?? '—'}` }) },
    { goal: 'Make the prompt at least as long as the generation (N ≥ G).', hint: 'raise prompt tokens N, or lower generated G.', check: (api) => ({ solved: (api.state.N | 0) >= (api.state.G | 0), detail: `N=${api.state.N}, G=${api.state.G}` }) },
  ],
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'prompt tokens (N)', min: 2, max: 6, value: 4 });
    c.stepper('G', { label: 'generated (G)', min: 2, max: 6, value: 5 });
    c.stepper('D', { label: 'features (D)', min: 4, max: 8, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.1, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !tlRect) return;
    const T = cur.N + cur.G, tp = page.controls._transport;
    const posToStage = (pos) => (pos < cur.N ? 0 : pos - cur.N + 1);
    if (ev.type === 'down') dragTL = ev.y >= tlRect.y - 10 && ev.y <= tlRect.y + tlRect.h + 10 && ev.x >= tlRect.x - 10 && ev.x <= tlRect.x + tlRect.w + 10;
    else if (ev.type === 'up' || ev.type === 'leave') dragTL = false;
    if (dragTL && (ev.type === 'down' || ev.type === 'move') && page.pointer.down && tp) {
      const pos = Math.max(0, Math.min(T - 1, Math.floor((ev.x - tlRect.x) / (tlRect.w / T))));
      tp.pause(); tp.seek(posToStage(pos));
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { Q, K, N, G, D } = cur;
    r.clear('#ffffff');
    const s = page.step(), stageIdx = s ? page.controls._transport.index : 0;
    page.probe = { phase: s ? s.phase : 'prefill' };
    const isPrefill = stageIdx === 0;
    const pos = isPrefill ? N - 1 : N + stageIdx - 1;   // current position (0-based)
    const filled = pos + 1;                              // cache rows filled
    const T = N + G;

    // shared diverging domain over the cache
    const Kdom = maxAbs(K.data);
    const pad = 16, topY = 70;
    const csC = Math.max(10, Math.min(20, Math.min((page.W * 0.13) / D, (page.H * 0.50) / T)));
    const cw = D * csC;

    // ---- left: the KV cache [(N+G)×D], rows filled up to `filled` ----
    cacheRect = { x: pad + 4, y: topY, w: cw, h: T * csC };
    r.label('KV cache  [(N+G)×D]', cacheRect.x, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    r.heatmap(K, { rows: T, cols: D, rect: cacheRect, ramp: ramps.diverging, domain: [-Kdom, Kdom] });
    r.grid({ stroke: 'rgba(0,0,0,0.10)' });
    ctx.save();
    // veil pending (not-yet-filled) rows
    if (filled < T) { ctx.fillStyle = 'rgba(255,255,255,0.66)'; ctx.fillRect(cacheRect.x, topY + filled * csC, cw, (T - filled) * csC); }
    // prefill bracket over rows 0..N-1
    ctx.strokeStyle = BLUE; ctx.lineWidth = isPrefill ? 3 : 1.5; ctx.strokeRect(cacheRect.x - 2, topY - 2, cw + 4, N * csC + 4);
    r.label('prompt (prefill: 1 pass)', cacheRect.x + cw + 8, topY + N * csC / 2, { color: BLUE, font: '10px ui-monospace, monospace' });
    // decode: outline current row + "reads rows 0..pos"
    if (!isPrefill) {
      ctx.strokeStyle = GREEN; ctx.lineWidth = 2.5; ctx.strokeRect(cacheRect.x - 2, topY + pos * csC, cw + 4, csC);
      r.label(`decode pos ${pos}: GEMV reads ${filled} rows`, cacheRect.x + cw + 8, topY + pos * csC + csC / 2, { color: GREEN, font: '10px ui-monospace, monospace' });
      ctx.strokeStyle = 'rgba(44,160,44,0.5)'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5; ctx.strokeRect(cacheRect.x - 1, topY - 1, cw + 2, filled * csC + 2); ctx.setLineDash([]);
    }
    ctx.restore();

    // ---- right: PREFILL panel (top) + DECODE panel (bottom) ----
    const rx = cacheRect.x + cw + 180, rw = page.W - rx - pad;
    const pHi = isPrefill, dHi = !isPrefill;

    // PREFILL: N×N causal attention triangle
    const csP = Math.max(10, Math.min(26, Math.min((rw * 0.45) / N, (page.H * 0.30) / N)));
    prefRect = { x: rx, y: topY, w: N * csP, h: N * csP };
    const Ap = prefillAttn(Q, K, N, D);
    ctx.save(); ctx.globalAlpha = pHi ? 1 : 0.4;
    r.label('PREFILL — N×N attention, computed in parallel', rx, topY - 14, { color: pHi ? BLUE : GREY, font: '12px ui-monospace, monospace' });
    r.heatmap(Ap, { rows: N, cols: N, rect: prefRect, ramp: ramps.sequential, domain: [0, 1] });
    r.grid({ stroke: 'rgba(0,0,0,0.12)' });
    ctx.restore();
    r.label(`${N}×${N} scores · big GEMM · COMPUTE-BOUND (GPU math saturated)`, rx, topY + N * csP + 16, { color: pHi ? INK : GREY, font: '10px ui-monospace, monospace' });

    // DECODE: 1×(pos+1) scores bar
    const dy = topY + N * csP + 44;
    barRect = { x: rx, y: dy, w: Math.min(rw, filled * 22), h: 22 };
    const sc = isPrefill ? decodeScores(Q, K, N - 1, D) : decodeScores(Q, K, pos, D);
    ctx.save(); ctx.globalAlpha = dHi ? 1 : 0.4;
    r.label('DECODE — 1×(N+t) attention, one growing row per step', rx, dy - 14, { color: dHi ? GREEN : GREY, font: '12px ui-monospace, monospace' });
    const bw = barRect.w / sc.length;
    for (let j = 0; j < sc.length; j++) {
      ctx.fillStyle = `rgba(44,160,44,${0.25 + 0.7 * sc[j]})`;
      ctx.fillRect(barRect.x + j * bw, dy, bw - 1, 22);
      ctx.strokeStyle = 'rgba(0,0,0,0.12)'; ctx.strokeRect(barRect.x + j * bw, dy, bw - 1, 22);
    }
    ctx.restore();
    r.label(`1×${sc.length} scores · skinny GEMV + reads ${filled} cache rows · MEMORY-BOUND · ×${G} steps`, rx, dy + 38, { color: dHi ? INK : GREY, font: '10px ui-monospace, monospace' });

    // ---- bottom: timeline of positions (draggable) ----
    const tly = page.H - pad - 26, tbw = Math.min(28, (page.W - 2 * pad) / T);
    tlRect = { x: pad + 4, y: tly, w: T * tbw, h: 20 };
    r.label('timeline:', pad + 4, tly - 8, { color: '#586069', font: '10px ui-monospace, monospace' });
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let p = 0; p < T; p++) {
      const bx = tlRect.x + p * tbw;
      const done = p <= pos, isPrompt = p < N;
      ctx.fillStyle = !done ? '#eef0f2' : isPrompt ? 'rgba(31,111,235,0.55)' : 'rgba(44,160,44,0.55)';
      ctx.fillRect(bx, tly, tbw - 2, 20);
      if (p === pos) { ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(bx - 1, tly - 1, tbw, 22); }
      ctx.fillStyle = done ? '#fff' : '#9aa4ad'; ctx.fillText(String(p), bx + (tbw - 2) / 2, tly + 10);
    }
    ctx.fillStyle = GREY; ctx.textAlign = 'left'; ctx.fillText('◀ prompt (prefill) ▏ generated (decode) ▶  — drag to scrub', tlRect.x, tly + 32);
    ctx.restore();

    // hover-to-inspect
    if (page.pointer.over && !dragTL) {
      const p = page.pointer;
      const hc = cacheRect && cellAt(cacheRect, T, D, p.x, p.y);
      const hp = prefRect && cellAt(prefRect, N, N, p.x, p.y);
      if (hc) { const fill = hc.r < N ? 'prefill (parallel)' : `decode step ${hc.r - N + 1}`; page.setTip(`cache[pos ${hc.r}, d${hc.c}] = ${K.data[hc.r * D + hc.c].toFixed(3)}\n${hc.r <= pos ? 'filled in ' + fill : 'not yet generated'}`); }
      else if (hp && hp.c <= hp.r) { page.setTip(`attn[q${hp.r}, k${hp.c}] = ${Ap.data[hp.r * N + hp.c].toFixed(3)}\nprefill: all queries at once`); }
      else if (barRect && p.x >= barRect.x && p.x <= barRect.x + barRect.w && p.y >= barRect.y && p.y <= barRect.y + barRect.h) {
        const j = Math.floor((p.x - barRect.x) / (barRect.w / sc.length));
        if (j >= 0 && j < sc.length) page.setTip(`decode w[k${j}] = ${sc[j].toFixed(3)}\nquery pos ${isPrefill ? N - 1 : pos} attends to key ${j}`);
      }
    }

    let o = `prefill: N tokens in 1 parallel GEMM (compute-bound).   decode: 1 token/step, GEMV re-reads the KV cache (memory-bound), ×G.    tier:${r.name}\n`;
    o += isPrefill
      ? `PREFILL — ${N} prompt tokens processed at once; full ${N}×${N} causal attention; cache rows 0..${N - 1} filled in one pass.`
      : `DECODE step ${stageIdx}/${G} — generating position ${pos}; 1×${filled} attention over the cache; appended cache row ${pos}.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__pdPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?pos=P selects the position (headless stand-in for a timeline drag): pos<N
  // stays in prefill, pos>=N is decode step pos-N+1.
  if (q.has('pos') && cur && t) {
    const P = Math.max(0, Math.min(cur.N + cur.G - 1, +q.get('pos') | 0));
    t.pause(); t.seek(P < cur.N ? 0 : P - cur.N + 1);
  }
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('pos') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
