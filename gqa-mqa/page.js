// gqa-mqa concept page -- KV-head sharing (MHA / GQA / MQA).
// Uses the verified framework: layout.mount() + controls + a per-scheme Transport.
//
// Interactive per the framework contract (plan/framework.md):
//  - HOVER a Q head -> which KV group it reads + its sibling Q heads in that
//    group; hover a KV head -> the set of Q heads that read it; either surfaces
//    the KV-cache saving (cache size ∝ #KV heads).
//  - DRAG the #KV-heads handle (on the KV-cache bar) left/right to MORPH
//    MHA (kv = n_q) -> GQA (kv = a divisor) -> MQA (kv = 1): the Q→KV mapping
//    arrows, the group coloring, and the memory bar all update live. This is
//    the GQA/MQA "aha" -- watch the cache shrink as sharing increases. The
//    handle snaps to the divisors of n_q (the only valid group ratios).
//  - AUTOPLAY + LOOP: the scheme transport auto-plays MHA→GQA→…→MQA and loops.
import { mount } from '../framework/layout.js';
import { categorical, cellAt } from '../framework/render.js';

const INK = '#111';
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
function box(ctx, x, y, w, h, label, col, filled) {
  ctx.save();
  roundRect(ctx, x, y, w, h, 6);
  ctx.fillStyle = rgb(col, filled ? 0.88 : 0.18); ctx.fill();
  ctx.strokeStyle = rgb(col); ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = filled ? '#fff' : rgb(col); ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, y + h / 2);
  ctx.restore();
}

// Divisors of n_q, descending (MHA=n_q first ... MQA=1 last) -- the valid group
// ratios (every group must be equal-sized, so n_kv must divide n_q).
function divisors(nq) { const d = []; for (let k = nq; k >= 1; k--) if (nq % k === 0) d.push(k); return d; }

function buildData(st) {
  return divisors(st.nq).map((nkv) => {
    const g = st.nq / nkv, scheme = nkv === st.nq ? 'MHA' : nkv === 1 ? 'MQA' : 'GQA';
    return { nkv, g, scheme, label: `${scheme} — ${nkv} KV head${nkv > 1 ? 's' : ''}, ${g} query head${g > 1 ? 's' : ''} per KV head; KV cache = ${nkv}/${st.nq} of MHA` };
  });
}

// Hit-test geometry captured each draw(), reused by onPointer() + the headless
// hooks: the n_q query-head boxes, the n_kv KV-head boxes, and the draggable
// KV-cache bar (whose handle position encodes n_kv).
let geom = null;
let dragging = false;   // true while the cache-bar handle is grabbed

// Seek the transport to the scheme whose n_kv is the divisor of n_q nearest to
// `target`. The transport's step list IS the divisor list, so "set #KV heads"
// is just selecting the matching step -- mapping arrows + memory all follow.
function setKV(page, target) {
  const t = page.controls._transport; if (!t || !t.steps.length) return;
  let bi = 0, bd = Infinity;
  for (let i = 0; i < t.steps.length; i++) { const dd = Math.abs(t.steps[i].nkv - target); if (dd < bd) { bd = dd; bi = i; } }
  t.seek(bi);
}

mount({
  mount: 'body',
  title: 'gqa-mqa — KV-head sharing (MHA / GQA / MQA)',
  blurb: 'Query heads can share key/value heads. DRAG the #KV-heads handle on the KV-cache bar to morph MHA → GQA → MQA: KV heads merge, query heads regroup onto them (colored by group), and the cache bar shrinks proportionally — the decode-time memory/bandwidth win. Hover a Q head to see its KV group + the sibling heads it shares with; hover a KV head to see which Q heads read it. It also auto-plays the schemes and loops.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  compare: { stepA: 'first', stepB: 'last', labelA: 'MHA — N KV heads (no sharing)', labelB: 'MQA — 1 shared KV head' },
  // Direct manipulation: grab the KV-cache bar handle and drag horizontally to
  // change n_kv. The bar's x-extent maps to [1 .. n_q] KV heads; the result
  // snaps to the nearest divisor (the only valid equal-group ratios).
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    if (ev.type === 'down') {
      dragging = false;
      // Grab if the press is on/near the bar (handle band is generous).
      if (ev.x >= g.barX - 14 && ev.x <= g.barX + g.barW + 14 && ev.y >= g.barY - 10 && ev.y <= g.barY + 26) {
        dragging = true;
        const frac = Math.max(0, Math.min(1, (ev.x - g.barX) / g.barW));
        setKV(page, Math.round(1 + frac * (g.nq - 1)));
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragging = false;
    } else if (ev.type === 'move' && dragging && page.pointer.down) {
      const frac = Math.max(0, Math.min(1, (ev.x - g.barX) / g.barW));
      setKV(page, Math.round(1 + frac * (g.nq - 1)));
    }
  },
  challenges: [
    { goal: 'Switch to MQA — a single shared KV head (the biggest cache saving).', hint: 'drag the group handle all the way down to 1 KV head.', check: (api) => ({ solved: api.probe.nkv === 1, detail: `${api.probe.nkv ?? '–'} KV head(s) — MQA needs exactly 1` }) },
    { goal: 'Use GQA to cut the KV cache to ¼ of MHA or less.', hint: 'pick a KV-head count ≤ query-heads ÷ 4.', check: (api) => ({ solved: (api.probe.nkv ?? 99) <= (api.probe.nq ?? 1) / 4, detail: `KV cache = ${api.probe.nkv}/${api.probe.nq} of MHA` }) },
  ],
  controls: (c, page) => {
    c.stepper('nq', { label: 'query heads', min: 4, max: 12, value: 8 });
    c.transport({ compute: () => buildData(page.state), speed: 1.5, loop: true });
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, nq = st.nq;
    r.clear('#ffffff');
    const s = page.step();
    const nkv = s ? s.nkv : nq, g = nq / nkv, scheme = s ? s.scheme : 'MHA';
    page.probe = { nkv, nq };

    const pad = 24, W = page.W, boxH = 34, qY = 78, kvY = qY + 148;
    const slot = (W - 2 * pad) / nq, boxW = Math.min(58, slot - 10);
    const qx = (i) => pad + i * slot + slot / 2;
    const kvx = (h) => { let s2 = 0; for (let i = h * g; i < (h + 1) * g; i++) s2 += qx(i); return s2 / g; };

    r.label(`scheme: ${scheme}    n_q = ${nq}    n_kv = ${nkv}    group = ${g} query head${g > 1 ? 's' : ''} per KV head`, pad, 40, { color: INK, font: '14px ui-monospace, monospace' });
    r.label('query heads', pad, qY - 12, { color: '#586069', font: '11px ui-monospace, monospace' });
    r.label('K/V heads', pad, kvY - 12, { color: '#586069', font: '11px ui-monospace, monospace' });

    // connections (behind boxes): each query head -> its KV head
    for (let i = 0; i < nq; i++) { const h = Math.floor(i / g); r.arrow({ x: qx(i), y: qY + boxH }, { x: kvx(h), y: kvY }, { color: rgb(categorical(h), 0.55), width: 1.5, head: 6, alpha: 0.8 }); }
    // query boxes (colored by group)
    for (let i = 0; i < nq; i++) box(ctx, qx(i) - boxW / 2, qY, boxW, boxH, 'q' + i, categorical(Math.floor(i / g)), false);
    // KV boxes (filled by group)
    for (let h = 0; h < nkv; h++) box(ctx, kvx(h) - boxW / 2, kvY, boxW, boxH, 'kv' + h, categorical(h), true);

    // KV-cache memory bar (∝ n_kv / n_q) -- also the drag handle for n_kv.
    const frac = nkv / nq, barY = kvY + 86, barX = pad + 96, barW = Math.min(380, W - barX - pad - 110);
    r.label('KV cache', pad, barY + 12, { color: '#586069', font: '12px ui-monospace, monospace' });
    ctx.save();
    ctx.strokeStyle = '#d0d7de'; ctx.lineWidth = 1; ctx.strokeRect(barX, barY, barW, 16);
    ctx.fillStyle = 'rgba(31,111,235,0.55)'; ctx.fillRect(barX, barY, barW * frac, 16);
    // draggable handle: a grip at the bar's fill edge (position encodes n_kv).
    const hx = barX + barW * ((nkv - 1) / Math.max(1, nq - 1));
    ctx.fillStyle = dragging ? '#1f6feb' : '#3a4047';
    roundRect(ctx, hx - 5, barY - 6, 10, 28, 3); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillRect(hx - 2, barY - 2, 1, 20); ctx.fillRect(hx + 1, barY - 2, 1, 20);
    ctx.restore();
    r.label(`${Math.round(frac * 100)}% of MHA  (${nkv}/${nq} KV heads)`, barX + barW + 8, barY + 12, { color: INK, font: '12px ui-monospace, monospace' });
    r.label('↔ drag to set #KV heads (MHA → GQA → MQA)', barX, barY + 38, { color: '#1f6feb', font: '10px ui-monospace, monospace' });

    // record hit-test geometry for onPointer + hover.
    geom = { nq, g, nkv, qY, boxH, kvY, boxW, slot, pad, qx, kvx, barX, barY, barW };

    // --- hover-to-inspect: Q head -> its KV group + sibling Q heads; KV head ->
    // the Q heads that read it. (Skip while dragging the cache handle.) ---
    if (page.pointer.over && !dragging) {
      const p = page.pointer; let tip = null;
      // which query box?
      for (let i = 0; i < nq && !tip; i++) {
        if (p.x >= qx(i) - boxW / 2 && p.x <= qx(i) + boxW / 2 && p.y >= qY && p.y <= qY + boxH) {
          const h = Math.floor(i / g);
          const sibs = []; for (let j = h * g; j < (h + 1) * g; j++) sibs.push('q' + j);
          tip = `Q head ${i} → reads KV group ${h} (kv${h})\nshared by Q heads {${sibs.join(',')}}  (${g} per group)\ncache holds ${nkv}/${nq} of MHA's K,V`;
        }
      }
      // which KV box?
      for (let h = 0; h < nkv && !tip; h++) {
        if (p.x >= kvx(h) - boxW / 2 && p.x <= kvx(h) + boxW / 2 && p.y >= kvY && p.y <= kvY + boxH) {
          const qs = []; for (let j = h * g; j < (h + 1) * g; j++) qs.push('q' + j);
          tip = `KV head ${h} ← Q heads {${qs.join(',')}}  (${g})\nthis K,V is stored ONCE, read by ${g} quer${g > 1 ? 'ies' : 'y'}\nfewer KV heads → smaller cache: ${nkv}/${nq} of MHA`;
        }
      }
      // hover the cache bar itself -> the memory saving.
      if (!tip && p.x >= barX - 14 && p.x <= barX + barW + 14 && p.y >= barY - 10 && p.y <= barY + 26) {
        tip = `KV cache = ${nkv}/${nq} = ${Math.round(frac * 100)}% of MHA\n↔ drag this handle to change #KV heads\ncache size ∝ #KV heads (decode memory/bandwidth)`;
      }
      if (tip) page.setTip(tip);
    }

    let o = `${scheme}: ${s ? s.label : `every query head has its own K,V head; KV cache = ${nq}/${nq} of MHA`}    tier:${r.name}\n`;
    o += 'MHA = own K,V per head · GQA = groups share a KV head (Llama-2/3-70B: 64 Q → 8 KV) · MQA = all share one (PaLM, Falcon). Fewer KV heads → proportionally smaller KV cache (the decode memory/bandwidth bottleneck). Drag the bar handle or scrub to morph.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__gqaPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?kv=N sets the number of KV heads (headless stand-in for a handle drag,
  // since --screenshot has no pointer). Snaps to the nearest divisor of n_q.
  // e.g. with n_q=8: ?kv=1 -> MQA, ?kv=2 -> GQA(2), ?kv=8 -> MHA.
  if (q.has('kv')) setKV(page, parseInt(q.get('kv'), 10) || 1);
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover) so
  // the sharing tooltip path is verifiable. Canvas-space px (x ~ 60..800).
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks
  // (so autoplay doesn't advance off the requested scheme before the snapshot).
  if (q.has('step') || q.has('hover') || q.has('kv')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
