// hybrid-by-layer concept page -- interleaved SSM layers + periodic full
// attention (Jamba / Zamba / Qwen3-Next style), showing WHERE the KV cache
// lives. Most layers are SSM (recurrent constant-size state, O(L), no KV);
// every P-th layer is full attention (O(L²), a KV cache [L×2d]). Steer the
// attention period to trade recall vs KV memory; click a layer to flip its
// type; hover to inspect. NOTE: no transport -- this is an architecture diagram;
// the forward-pass pulse is the only animation (api.t).
import { mount } from '../framework/layout.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', GREY = '#9aa4ad';
const fmt = (n) => n.toLocaleString('en-US');

let overrides = new Set(), baseSig = '';
let layerRects = [], curN = 0;

const isAttn = (i, P) => { const base = (i + 1) % P === 0; return overrides.has(i) ? !base : base; };

mount({
  mount: 'body',
  title: 'hybrid-by-layer — interleaved SSM + periodic attention',
  blurb: 'Hybrid models (Jamba, Zamba, Qwen3-Next) stack mostly cheap SSM / linear-attention layers (recurrent constant-size state, O(L) per token, NO KV cache) and insert a full-attention layer every P layers (O(L²), a KV cache [L×2d]). The interleave ratio is the whole knob: more attention = better exact long-range recall but more KV memory + O(L²) compute; fewer = cheaper, leaning on the SSM state to summarise the past. The KV cache lives ONLY in the attention layers, so total KV is a fraction (#attn / N) of a full-attention model. Drag the attention-period slider to steer the pattern + KV memory; click any layer to flip SSM↔attention; hover to inspect.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'layers (N)', min: 6, max: 20, value: 14 });
    c.slider('P', { label: 'attn period (1 attn / P)', min: 2, max: 8, step: 1, value: 5 });
    c.stepper('L', { label: 'context length (L)', min: 8, max: 64, step: 8, value: 32 });
    c.stepper('d', { label: 'head dim (d)', min: 16, max: 128, step: 16, value: 64 });
  },
  onPointer: (page, ev) => {
    if (ev.type !== 'down' || !layerRects.length) return;
    for (const lr of layerRects) if (ev.x >= lr.x && ev.x <= lr.x + lr.w && ev.y >= lr.y && ev.y <= lr.y + lr.h) {  // click a layer -> toggle type
      if (overrides.has(lr.i)) overrides.delete(lr.i); else overrides.add(lr.i); page.redraw(); break;
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const N = st.N | 0, P = st.P | 0, L = st.L | 0, d = st.d | 0;
    const sig = `${N}|${P}`; if (sig !== baseSig) { overrides.clear(); baseSig = sig; }   // pattern changed -> drop manual flips
    curN = N;
    r.clear('#ffffff');

    const types = Array.from({ length: N }, (_, i) => isAttn(i, P));
    const nAttn = types.filter(Boolean).length, nSSM = N - nAttn;
    const kvPer = L * 2 * d, kvTot = nAttn * kvPer, kvFull = N * kvPer;
    const saving = kvFull ? Math.round((1 - kvTot / kvFull) * 100) : 0;
    const fp = Math.floor((page.t || 0) * 2.5) % N;                                        // forward-pass pulse layer

    const pad = 16, topY = 56, stackW = page.W * 0.58;
    const rowH = Math.min(26, (page.H - topY - 42) / N);
    const labelW = 34, chipW = 46, kvBadgeW = 96;
    const seqX = pad + labelW + chipW + 10, seqW = stackW - labelW - chipW - kvBadgeW - 24;
    layerRects = [];

    r.label('layer stack (input ↓ bottom → output ↑ top)', pad, topY - 12, { color: INK, font: '11px ui-monospace, monospace' });

    let hoverLayer = -1;
    for (let i = 0; i < N; i++) {
      const attn = types[i], y = topY + (N - 1 - i) * rowH, col = attn ? ORANGE : BLUE;
      const rect = { x: pad, y: y + 1, w: stackW, h: rowH - 2, i };
      layerRects.push(rect);
      const hot = page.pointer.over && page.pointer.x >= pad && page.pointer.x <= pad + stackW && page.pointer.y >= rect.y && page.pointer.y <= rect.y + rect.h;
      if (hot) hoverLayer = i;
      ctx.save();
      if (i === fp) { ctx.fillStyle = 'rgba(44,160,44,0.10)'; ctx.fillRect(pad - 2, rect.y, stackW + 4, rect.h); }   // pulse
      if (hot) { ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.strokeRect(pad - 2, rect.y, stackW + 4, rect.h); }
      ctx.fillStyle = '#586069'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(`L${i}`, pad, y + rowH / 2);
      // type chip
      ctx.fillStyle = attn ? 'rgba(210,105,30,0.18)' : 'rgba(31,111,235,0.16)'; ctx.fillRect(pad + labelW, y + 2, chipW, rowH - 4);
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.strokeRect(pad + labelW, y + 2, chipW, rowH - 4);
      ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.fillText(attn ? 'ATTN' : 'SSM', pad + labelW + chipW / 2, y + rowH / 2);
      // sequence representation
      const cy = y + rowH / 2;
      if (attn) {                                                   // KV cache cells (the stored positions)
        const nc = Math.min(L, 24), cw = seqW / nc;
        for (let c = 0; c < nc; c++) { ctx.fillStyle = `rgba(210,105,30,${0.35 + 0.4 * (c / nc)})`; ctx.fillRect(seqX + c * cw, y + 3, cw - 1, rowH - 6); }
        ctx.strokeStyle = 'rgba(210,105,30,0.5)'; ctx.lineWidth = 1; ctx.strokeRect(seqX, y + 3, nc * cw, rowH - 6);
      } else {                                                      // recurrent state line + a flowing dot
        ctx.strokeStyle = 'rgba(31,111,235,0.55)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(seqX, cy); ctx.lineTo(seqX + seqW, cy); ctx.stroke();
        const dx = seqX + ((page.t || 0) * 60 + i * 7) % seqW; ctx.fillStyle = BLUE; ctx.beginPath(); ctx.arc(dx, cy, 2.6, 0, 7); ctx.fill();
      }
      // KV badge
      ctx.textAlign = 'left'; ctx.fillStyle = attn ? ORANGE : GREY; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(attn ? `KV [L×2d]` : 'no KV', seqX + seqW + 10, cy);
      ctx.restore();
    }

    // ---- rollup panel ----
    const px = stackW + pad + 18, py = topY + 4; let ly = py;
    const line = (k, v, col) => { r.label(k, px, ly, { color: '#586069', font: '11px ui-monospace, monospace' }); if (v != null) r.label(v, px + 6, ly + 13, { color: col || INK, font: '12px ui-monospace, monospace' }); ly += v != null ? 32 : 16; };
    r.label('KV cache budget', px, py - 14, { color: INK, font: '12px ui-monospace, monospace' });
    line(`pattern: 1 attn / ${P} layers`, `${nSSM} SSM  +  ${nAttn} ATTN  =  ${N}`, INK);
    line('KV cache total = #attn·L·2·d', `${fmt(kvTot)} elems`, ORANGE);
    line(`vs full attention (${N} layers)`, `${fmt(kvFull)}  →  ${saving}% less KV`, GREEN);
    line('compute', `${nSSM}×O(L) + ${nAttn}×O(L²)`, INK);
    ly += 6;
    const focus = hoverLayer >= 0 ? hoverLayer : fp, fAttn = types[focus];
    r.label(`layer L${focus}${hoverLayer >= 0 ? ' (hover)' : ' (forward pass)'}`, px, ly, { color: fAttn ? ORANGE : BLUE, font: '11px ui-monospace, monospace' }); ly += 16;
    r.label(fAttn ? 'full attention · O(L²) · holds a KV' : 'SSM · O(L) · recurrent state, no KV', px, ly, { color: '#3a4047', font: '10px ui-monospace, monospace' }); ly += 14;
    r.label(fAttn ? `cache = [L=${L} × 2·d=${2 * d}] = ${fmt(kvPer)} elems` : `state ≈ [d×d] constant (independent of L)`, px, ly, { color: '#3a4047', font: '10px ui-monospace, monospace' }); ly += 14;
    r.label('click a layer to flip SSM↔ATTN', px, ly, { color: GREY, font: '9px ui-monospace, monospace' });

    // hover tooltip
    if (page.pointer.over && hoverLayer >= 0) {
      const a = types[hoverLayer];
      page.setTip(`layer ${hoverLayer}: ${a ? 'full ATTENTION' : 'SSM'}\n${a ? `KV cache [L=${L} × 2·d=${2 * d}] = ${fmt(kvPer)} elems · O(L²)` : 'recurrent constant-size state · O(L) · no KV'}\nclick to flip`);
    }

    let o = `hybrid stack: ${nSSM} SSM (no KV, O(L)) + ${nAttn} attention (KV cache, O(L²)) of ${N} layers; 1 attn / ${P}.   KV = ${fmt(kvTot)} elems (${saving}% less than full attention).    tier:${r.name}\n`;
    o += `the KV cache lives only in the ${nAttn} attention layer${nAttn === 1 ? '' : 's'}; the SSM layers carry a constant-size state. Drag the period to trade recall (more attn) vs KV memory (fewer attn); click a layer to flip it.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__hblPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('p')) page.controls.set('P', parseInt(q.get('p'), 10));
  baseSig = `${page.state.N | 0}|${page.state.P | 0}`;   // pre-set so ?toggle survives the first draw's pattern-change clear
  if (q.has('toggle')) for (const s of q.get('toggle').split(',')) { const i = +s; if (i >= 0 && i < (page.state.N | 0)) overrides.add(i); }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
