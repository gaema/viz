// vram-budget concept page -- live VRAM math for LLM inference. Total GPU memory
// splits into four pieces; the page shows each to scale against a chosen GPU and
// flags OOM:
//   weights      = params · bytes_per_weight           (fixed by size + quant)
//   KV cache     = 2 · L · (d/H · n_kv) · seq · batch · kv_bytes   (grows w/ context!)
//   activations  ≈ batch · seq · d · bytes · c (prefill) / tiny (decode)
//   overhead     ≈ CUDA context + fragmentation (fixed)
// The KV-cache-vs-context curve shows KV growing LINEARLY with sequence length --
// at long context it overtakes the weights, the usual reason a model OOMs. GQA
// (n_kv << n_heads) and KV-quant shrink it; weight-quant shrinks the weights.
import { mount } from '../framework/layout.js';

const INK = '#111', GREY = '#9aa4ad', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', RED = '#d1242f', SLATE = '#7d8590';
const GB = 1e9;
const fmt = (b) => b >= GB ? (b / GB).toFixed(b < 10 * GB ? 2 : 1) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(0) + ' MB' : (b / 1e3).toFixed(0) + ' KB';
const MODELS = {
  'Llama-7B': { params: 6.7e9, L: 32, d: 4096, H: 32, KV: 32, V: 32000 },
  'Llama-13B': { params: 13e9, L: 40, d: 5120, H: 40, KV: 40, V: 32000 },
  'Llama-70B (GQA)': { params: 70e9, L: 80, d: 8192, H: 64, KV: 8, V: 32000 },
  'Qwen-1.8B': { params: 1.8e9, L: 24, d: 2048, H: 16, KV: 16, V: 151936 },
};
const QBYTES = { fp16: 2, int8: 1, int4: 0.5 };
const SEQMIN = 512, SEQMAX = 131072;

function calc(st) {
  const m = MODELS[st.model], wb = QBYTES[st.wq], kvb = QBYTES[st.kvq] || 2;
  const seq = st.context | 0, B = st.batch | 0;
  const weights = m.params * wb;
  const dHeadKV = (m.d / m.H) * m.KV;
  const kvPerTok = 2 * m.L * dHeadKV * kvb;            // bytes per (token·batch)
  const kv = kvPerTok * seq * B;
  const act = st.mode === 'prefill'
    ? B * seq * m.d * 2 * 3 + B * m.V * 2
    : B * m.d * 2 * 6 + B * m.V * 2;
  const overhead = 1.5 * GB;
  const total = weights + kv + act + overhead;
  return { m, weights, kv, act, overhead, total, kvPerTok, seq, B };
}

let dragSeq = false;

mount({
  mount: 'body',
  title: 'vram-budget — where the GPU memory goes',
  blurb: 'Running an LLM, your GPU memory splits four ways and you OOM when the sum exceeds the card. WEIGHTS = params × bytes/weight — fixed once you pick the size and the quant (fp16=2 B, int8=1, int4=0.5). KV CACHE = 2·layers·(head_dim·n_kv_heads)·seq·batch·kv_bytes — it caches every past token’s keys/values and so grows LINEARLY with context length and batch; at long context it overtakes the weights, which is the usual reason a long prompt OOMs. ACTIVATIONS are the transient forward-pass buffers — tiny during decode (one token), larger during a big prefill. OVERHEAD is the fixed CUDA context + fragmentation. The bar shows all four to scale against a chosen GPU (red = OOM); the curve plots KV cache vs context so you can watch it cross the weights line. Try: int4 weights to shrink the blue block, GQA (Llama-70B) to shrink the KV block, or drag the context marker out to 128k and watch KV explode.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  challenges: [
    { goal: 'Fit a 70B model on a 48 GB GPU (only int4 + GQA gets there).', hint: 'Llama-70B (GQA) + int4 weights + a 48 GB GPU + a short enough context.', check: (api) => { const s = api.state; return { solved: /70B/.test(s.model || '') && (+s.gpu) === 48 && (api.probe.total ?? 1e99) <= (api.probe.cap ?? 0), detail: `${/70B/.test(s.model || '') ? '' : 'pick the 70B model · '}${(+s.gpu) === 48 ? '' : 'set GPU = 48 · '}${(api.probe.total ?? 1e99) <= (api.probe.cap ?? 0) ? 'fits!' : 'still over'}` }; } },
    { goal: 'Cause an OOM — push the total past the GPU.', hint: 'crank the context length (KV cache) or the batch size up.', check: (api) => ({ solved: (api.probe.total ?? 0) > (api.probe.cap ?? 1e99), detail: (api.probe.total ?? 0) > (api.probe.cap ?? 1e99) ? 'over budget' : 'still fits' }) },
  ],
  controls: (c, page) => {
    c.select('model', { label: 'model', options: Object.keys(MODELS), value: 'Llama-7B' });
    c.select('wq', { label: 'weight quant', options: ['fp16', 'int8', 'int4'], value: 'fp16' });
    c.select('kvq', { label: 'KV cache dtype', options: ['fp16', 'int8'], value: 'fp16' });
    c.slider('context', { label: 'context length', min: SEQMIN, max: SEQMAX, step: 512, value: 4096 });
    c.slider('batch', { label: 'batch size', min: 1, max: 64, step: 1, value: 1 });
    c.select('mode', { label: 'phase', options: ['decode', 'prefill'], value: 'decode' });
    c.select('gpu', { label: 'GPU capacity', options: ['16', '24', '48', '80'], value: '24' });
  },
  onPointer: (page, ev) => {
    if (!page._curve) return;
    const cv = page._curve, inside = ev.x >= cv.x && ev.x <= cv.x + cv.w && ev.y >= cv.y - 10 && ev.y <= cv.y + cv.h + 10;
    const toSeq = (x) => { const f = Math.max(0, Math.min(1, (x - cv.x) / cv.w)); return Math.round(SEQMIN * Math.pow(SEQMAX / SEQMIN, f) / 512) * 512; };
    if (ev.type === 'down') { dragSeq = inside; if (dragSeq) { page.controls.set('context', toSeq(ev.x)); page.redraw(); } }
    else if (ev.type === 'up' || ev.type === 'leave') dragSeq = false;
    else if (ev.type === 'move' && dragSeq && page.pointer.down) { page.controls.set('context', toSeq(ev.x)); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear('#ffffff');
    const v = calc(st), cap = (+st.gpu) * GB, oom = v.total > cap;
    page.probe = { total: v.total, cap };
    const segs = [
      { k: 'weights', val: v.weights, col: BLUE, f: `params · ${QBYTES[st.wq]}B` },
      { k: 'KV cache', val: v.kv, col: ORANGE, f: `2·L·(d/H·n_kv)·seq·batch·${QBYTES[st.kvq] || 2}B` },
      { k: 'activations', val: v.act, col: GREEN, f: st.mode === 'prefill' ? 'batch·seq·d·2·3 + logits' : 'batch·d·… + logits (1 tok)' },
      { k: 'overhead', val: v.overhead, col: SLATE, f: 'CUDA ctx + fragmentation' },
    ];

    // ===== stacked VRAM bar vs capacity =====
    const bx = 20, by = 64, bw = W - 40, bh = 38, scaleMax = Math.max(v.total, cap) * 1.06, X = (b) => bx + b / scaleMax * bw;
    r.label(`total ${fmt(v.total)}  /  ${st.gpu} GB GPU`, bx, by - 10, { color: INK, font: '12px ui-monospace, monospace' });
    let acc = 0;
    ctx.save();
    for (const s of segs) { const x0 = X(acc), x1 = X(acc + s.val); ctx.fillStyle = s.col; ctx.fillRect(x0, by, Math.max(0, x1 - x0), bh); acc += s.val; }
    // capacity line
    const capX = X(cap); ctx.strokeStyle = oom ? RED : '#111'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(capX, by - 4); ctx.lineTo(capX, by + bh + 4); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = oom ? RED : '#111'; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(`${st.gpu} GB`, capX, by + bh + 14);
    // OOM border pulse
    if (oom) { const a = 0.4 + 0.4 * Math.sin((page.t || 0) * 4); ctx.strokeStyle = `rgba(209,36,47,${a})`; ctx.lineWidth = 2.5; ctx.strokeRect(bx - 1, by - 1, bw + 2, bh + 2); }
    ctx.restore();
    if (oom) r.label(`⚠ OOM by ${fmt(v.total - cap)}`, bx + bw - 150, by - 10, { color: RED, font: 'bold 12px ui-monospace, monospace' });

    // legend / numbers
    let lx = bx;
    for (const s of segs) { ctx.save(); ctx.fillStyle = s.col; ctx.fillRect(lx, by + bh + 22, 11, 11); ctx.restore(); r.label(`${s.k}  ${fmt(s.val)}`, lx + 16, by + bh + 31, { color: INK, font: '10px ui-monospace, monospace' }); lx += 190; }

    // ===== KV-vs-context curve =====
    const cx = 20, cyTop = by + bh + 58, cw = Math.min(470, W - 300), chh = H - cyTop - 30;
    page._curve = { x: cx, y: cyTop, w: cw, h: chh };
    r.label('KV cache vs context length  (drag the marker ↔)', cx, cyTop - 8, { color: INK, font: '11px ui-monospace, monospace' });
    const kvMax = v.kvPerTok * SEQMAX * v.B, yMax = Math.max(kvMax, v.weights) * 1.1;
    const SX = (seq) => cx + Math.log(seq / SEQMIN) / Math.log(SEQMAX / SEQMIN) * cw, KY = (b) => cyTop + chh - b / yMax * chh;
    ctx.save(); ctx.strokeStyle = '#eceef0'; ctx.strokeRect(cx, cyTop, cw, chh);
    // weights reference line
    ctx.strokeStyle = 'rgba(31,111,235,0.6)'; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(cx, KY(v.weights)); ctx.lineTo(cx + cw, KY(v.weights)); ctx.stroke(); ctx.setLineDash([]);
    r.label(`weights ${fmt(v.weights)}`, cx + cw - 96, KY(v.weights) - 5, { color: BLUE, font: '9px ui-monospace, monospace' });
    // KV curve (linear in seq -> curved on log-x)
    ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.beginPath();
    for (let s = SEQMIN; s <= SEQMAX; s *= 1.15) { const px = SX(s), py = KY(v.kvPerTok * s * v.B); if (s === SEQMIN) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.lineTo(SX(SEQMAX), KY(kvMax)); ctx.stroke();
    // crossover (KV == weights)
    const seqX = v.weights / (v.kvPerTok * v.B);
    if (seqX > SEQMIN && seqX < SEQMAX) { ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(SX(seqX), cyTop); ctx.lineTo(SX(seqX), cyTop + chh); ctx.stroke(); r.label(`KV=weights @ ${Math.round(seqX / 1024)}k`, Math.min(SX(seqX) + 3, cx + cw - 86), cyTop + 12, { color: '#555', font: '8px ui-monospace, monospace' }); }
    // animated ghost sweep
    const gf = ((page.t || 0) % 4) / 4, gs = SEQMIN * Math.pow(SEQMAX / SEQMIN, gf); ctx.fillStyle = 'rgba(210,105,30,0.35)'; ctx.beginPath(); ctx.arc(SX(gs), KY(v.kvPerTok * gs * v.B), 3, 0, 7); ctx.fill();
    // current marker
    ctx.fillStyle = RED; ctx.beginPath(); ctx.arc(SX(v.seq), KY(v.kv), 4.5, 0, 7); ctx.fill();
    ctx.restore();
    r.label('512', cx, cyTop + chh + 12, { color: '#8a939b', font: '8px ui-monospace, monospace' }); r.label('128k', cx + cw - 22, cyTop + chh + 12, { color: '#8a939b', font: '8px ui-monospace, monospace' });
    r.label(`@ ${v.seq >= 1024 ? (v.seq / 1024).toFixed(0) + 'k' : v.seq} ctx ×${v.B}: KV = ${fmt(v.kv)}`, cx, cyTop + chh + 26, { color: ORANGE, font: '10px ui-monospace, monospace' });

    // ===== breakdown panel (right) =====
    const px = cx + cw + 26, py = cyTop;
    r.label('breakdown', px, py - 8, { color: INK, font: '11px ui-monospace, monospace' });
    const m = v.m, rows = [
      `${st.model}: ${(m.params / 1e9).toFixed(1)}B params`,
      `${m.L} layers, d=${m.d}, ${m.H} heads`,
      m.KV < m.H ? `GQA: ${m.KV} kv-heads (${(m.H / m.KV).toFixed(0)}× less KV)` : `MHA: ${m.KV} kv-heads`,
      `weight quant ${st.wq} → ${fmt(v.weights)}`,
      `KV ${st.kvq}: ${fmt(v.kvPerTok)}/tok·batch`,
      `phase: ${st.mode}`,
      ``,
      v.kv > v.weights ? 'KV cache > weights — context-bound' : 'weights dominate — model-bound',
      oom ? `OOM: total ${fmt(v.total)} > ${st.gpu} GB` : `fits: ${fmt(cap - v.total)} free`,
    ];
    rows.forEach((t, i) => { if (t) r.label(t, px, py + 14 + i * 15, { color: i >= 7 ? (oom && i === 8 ? RED : (i === 7 ? (v.kv > v.weights ? ORANGE : BLUE) : GREEN)) : '#444', font: (i >= 7 ? 'bold ' : '') + '10px ui-monospace, monospace' }); });

    // hover on bar segments
    if (page.pointer.over && !dragSeq) {
      const p = page.pointer; let a = 0;
      if (p.y >= by && p.y <= by + bh) for (const s of segs) { if (p.x >= X(a) && p.x <= X(a + s.val)) { page.setTip(`${s.k}: ${fmt(s.val)}\n= ${s.f}\n${(s.val / v.total * 100).toFixed(0)}% of total`); break; } a += s.val; }
    }

    let o = `VRAM = weights + KV cache + activations + overhead.  ${st.model}, ${st.wq} weights, ${st.kvq} KV, ctx ${v.seq} ×${v.B}, ${st.mode}.   tier:${r.name}\n`;
    o += `weights ${fmt(v.weights)} + KV ${fmt(v.kv)} + act ${fmt(v.act)} + overhead ${fmt(v.overhead)} = ${fmt(v.total)} vs ${st.gpu} GB → ${oom ? 'OOM by ' + fmt(v.total - cap) : fmt(cap - v.total) + ' free'}. ${v.kv > v.weights ? 'KV cache now exceeds the weights — you are context-bound; GQA / KV-quant / shorter context help.' : 'Weights dominate; quantize them (int4) to fit a bigger model or longer context.'}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__vbPage = page;
  const q = new URLSearchParams(location.search);
  for (const key of ['model', 'wq', 'kvq', 'mode', 'gpu']) if (q.has(key)) page.controls.set(key, q.get(key));
  for (const key of ['context', 'batch']) if (q.has(key)) page.controls.set(key, +q.get(key));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
