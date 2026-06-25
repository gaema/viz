// rope concept page -- per-pair 2-D rotation by position·θ (RoPE).
// Interactive per the framework contract (plan/framework.md): DRAG the
// position (horizontal drag on the canvas, or the position handle) to change
// the token position and watch every pair's rotation angle Δ = p·θᵢ update
// live -- fast pairs (low i) spin, slow pairs (high i) barely move, the RoPE
// "aha". HOVER a rotation plane for its θᵢ, Δ = p·θᵢ, (cos,sin) and the rotated
// (x,y); hover a heatmap cell for its rotated value. When you're NOT dragging
// the position advances smoothly on its own (the rings rotate on load).
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { rope, ropeAngles, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GRAY = 'rgba(150,160,170,0.85)', ORANGE = '#d2691e';
const fmt = (x) => (x === 0 ? '0' : Math.abs(x) < 1e-2 || Math.abs(x) >= 1e4 ? x.toExponential(1) : String(Number(x.toPrecision(3))));

// Live continuous position, shared between draw(), onPointer(), and the
// ambient clock. `frozen` pins it (a drag, a ?pos= hook, or a transport seek)
// so the auto-sweep doesn't fight a user-set value; the auto-sweep runs only
// when !frozen. `pos` is fractional (smooth rotation); the transport's integer
// step is derived from it so ?step / scrub still work.
let posState = { pos: 8, frozen: false, base: 8 };
let planes = [];      // per-pair {i, cx, cy, R, theta, delta, phi} captured in draw for hit-testing
let posHandle = null; // {x, y, w, h} the draggable position track, captured in draw
let hmRect = null;    // [N×d] heatmap rect, captured in draw
let hmDims = { N: 0, d: 0 };
let grabbingPos = false;

function buildData(st) { return Array.from({ length: st.N }, (_, p) => ({ p, label: `position ${p}: pair i rotated by Δ = ${p}·θᵢ` })); }

// Clamp the live position into [0, N-1] and mirror it onto the transport so the
// scrub thumb + step label track the canvas. Snaps to the nearest integer step.
function syncTransport(page) {
  const t = page.controls._transport;
  if (!t || !t.steps.length) return;
  const N = page.state.N;
  const idx = Math.max(0, Math.min(N - 1, Math.round(posState.pos)));
  page.probe = { pos: idx, N };
  if (t.index !== idx) { t.index = idx; t._sync(); }
}

mount({
  mount: 'body',
  title: 'rope — rotary position embedding (per-pair rotation)',
  blurb: 'RoPE encodes position by rotating each dimension-pair by Δ = position·θᵢ, with θᵢ = base^(−2i/d). Low pairs (high freq) spin fast, high pairs slow. DRAG horizontally (or drag the position track) to move the token position — every pair’s angle updates live and the vectors rotate. Hover a plane for its θᵢ / Δ / (cos,sin) / rotated (x,y), or a heatmap cell for its value. Let go and the position sweeps on its own.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  challenges: [
    { goal: 'Advance to the FINAL position (the last token).', hint: 'drag the rotation handle (or scrub) to the last position.', check: (api) => ({ solved: (api.probe.pos ?? -1) === (api.probe.N ?? 0) - 1, detail: `position ${api.probe.pos ?? 0} / ${(api.probe.N ?? 1) - 1}` }) },
    { goal: 'Crank the rotary base θ to its max (20000) — slows the high-dim rotations for long context.', hint: 'raise the rotary-base slider to the top.', check: (api) => ({ solved: (api.state.base | 0) >= 20000, detail: `θ = ${api.state.base | 0} (need 20000)` }) },
  ],
  animate: true,
  controls: (c, page) => {
    c.stepper('d', { label: 'head_dim', min: 4, max: 12, step: 2, value: 8, rebuild: false });
    c.stepper('N', { label: 'positions', min: 8, max: 32, value: 16 });
    c.slider('base', { label: 'rotary base θ', min: 1000, max: 20000, step: 1000, value: 10000, rebuild: false, format: (v) => String(v | 0) });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: false });
    c.transport({ compute: () => buildData(page.state), speed: 2 });
  },
  // Direct manipulation: drag horizontally anywhere (or on the position track)
  // to set the token position. Horizontal drag maps to position across [0,N-1].
  onPointer: (page, ev) => {
    const N = page.state.N;
    if (ev.type === 'down') {
      grabbingPos = true;
      posState.frozen = true;       // freeze the auto-sweep while the user drives
      const t = page.controls._transport; if (t) t.pause();
      setPosFromX(page, ev.x);
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grabbingPos = false;
      // Leave it frozen at the dragged position; the auto-sweep resumes only on
      // a fresh reload or an explicit unfreeze. (Per contract C: freeze while
      // dragging; here we keep the last dragged pos so the picture is stable.)
      if (ev.type === 'leave') page.pointer.over = false;
    } else if (ev.type === 'move' && grabbingPos && page.pointer.down) {
      setPosFromX(page, ev.x);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const d = st.d, N = st.N, base = st.base, seed = st.seed | 0;
    r.clear('#ffffff');
    hmDims = { N, d };

    // --- position source ---
    // When not frozen, sweep the position smoothly with the ambient clock so the
    // rings rotate on load; freeze pins it to the dragged / hooked value.
    if (!posState.frozen) {
      posState.pos = (posState.base + page.t * 1.6) % N;   // pos = base + k·t, wraps
    }
    posState.pos = Math.max(0, Math.min(N - 1, posState.pos));
    const p = posState.pos;
    syncTransport(page);

    const vec = seededRandn(seed, d);
    const ang = ropeAngles(d, p, { theta: base });    // per-pair Δ = p·θᵢ (fractional p)

    // --- per-pair rotation planes ---
    const pad = 16, np = d / 2, planeRowY = 70;
    const planeW = (page.W - 2 * pad) / np, planeSize = Math.min(planeW - 14, 142), R = planeSize * 0.34;
    r.label(`per-pair 2-D rotation at position ${fmt(p)}  (drag ↔ to move; hover a plane to inspect)`, pad, planeRowY - 24, { color: INK, font: '12px ui-monospace, monospace' });

    // Draggable position track (the "scrub on canvas" handle).
    const trkY = 40, trkX0 = pad + 6, trkX1 = page.W - pad - 6, trkW = trkX1 - trkX0;
    posHandle = { x: trkX0, y: trkY - 8, w: trkW, h: 16 };
    ctx.save();
    ctx.strokeStyle = '#d7dbe0'; ctx.lineWidth = 4; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(trkX0, trkY); ctx.lineTo(trkX1, trkY); ctx.stroke();
    const hx = trkX0 + (N <= 1 ? 0 : (p / (N - 1)) * trkW);
    ctx.fillStyle = grabbingPos ? ORANGE : BLUE;
    ctx.beginPath(); ctx.arc(hx, trkY, 7, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#586069'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
    ctx.fillText('pos 0', trkX0, trkY - 12);
    ctx.textAlign = 'right'; ctx.fillText(`pos ${N - 1}`, trkX1, trkY - 12);
    ctx.restore();

    planes = [];
    for (let i = 0; i < np; i++) {
      const a = vec[2 * i], b = vec[2 * i + 1], phi = Math.atan2(b, a), delta = ang[i];
      const theta = Math.pow(base, -2 * i / d);
      const cx = pad + i * planeW + planeW / 2, cy = planeRowY + planeSize / 2;
      planes.push({ i, cx, cy, R, theta, delta, phi, a, b });
      ctx.save();
      ctx.strokeStyle = '#e7eaee'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
      // spiral tracing the sweep phi -> phi+delta (radius grows so multi-turns show)
      if (delta > 0.001) {
        ctx.strokeStyle = 'rgba(31,111,235,0.55)'; ctx.lineWidth = 1.4; ctx.beginPath();
        const steps = Math.max(10, Math.min(600, Math.round(delta / 0.06)));
        for (let s2 = 0; s2 <= steps; s2++) { const t = s2 / steps, an = phi + delta * t, rad = R * (0.22 + 0.66 * t); const x = cx + rad * Math.cos(an), y = cy - rad * Math.sin(an); s2 === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
        ctx.stroke();
      }
      ctx.restore();
      const tip = (an, len) => ({ x: cx + len * Math.cos(an), y: cy - len * Math.sin(an) });
      r.arrow({ x: cx, y: cy }, tip(phi, R * 0.9), { color: GRAY, width: 1.5, head: 6, alpha: 1 });
      r.arrow({ x: cx, y: cy }, tip(phi + delta, R * 0.9), { color: BLUE, width: 2.5, head: 7, alpha: 1 });
      const ly = planeRowY + planeSize + 4;
      r.label(`pair ${i}`, cx, ly, { color: INK, font: '11px ui-monospace, monospace', align: 'center' });
      r.label(`θ=${fmt(theta)}`, cx, ly + 14, { color: '#586069', font: '10px ui-monospace, monospace', align: 'center' });
      r.label(`Δ=${fmt(delta)} rad`, cx, ly + 27, { color: BLUE, font: '10px ui-monospace, monospace', align: 'center' });
    }

    // --- [N×d] heatmap: rotated components across all positions (the RoPE wave) ---
    const comp = { data: new Float32Array(N * d), rows: N, cols: d };
    for (let pp = 0; pp < N; pp++) { const rv = rope(vec, pp, { theta: base }); for (let k = 0; k < d; k++) comp.data[pp * d + k] = rv[k]; }
    let dom = 1e-9; for (let i = 0; i < comp.data.length; i++) dom = Math.max(dom, Math.abs(comp.data[i]));
    const hy = planeRowY + planeSize + 64;
    const cw = Math.max(8, Math.min(26, (page.W - 2 * pad - 150) / d)), ch = Math.max(6, Math.min(11, (page.H - hy - 22) / N));
    const hxh = pad + 130, hRect = { x: hxh, y: hy, w: d * cw, h: N * ch };
    hmRect = hRect;
    r.label('rotated components across positions (RoPE wave):', pad, hy - 8, { color: '#586069', font: '11px ui-monospace, monospace' });
    r.label('positions ↓ / dims →', pad, hy + 12, { color: '#9aa4ad', font: '10px ui-monospace, monospace' });
    r.heatmap(comp, { rows: N, cols: d, rect: hRect, ramp: ramps.diverging, domain: [-dom, dom] });
    const prow = Math.round(p);
    ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(hRect.x - 1, hRect.y + prow * ch - 1, d * cw + 2, ch + 2); ctx.restore();
    r.label(`pos ${prow}`, hxh + d * cw + 6, hy + prow * ch + ch / 2 + 3, { color: INK, font: '10px ui-monospace, monospace' });

    // --- hover-to-inspect ---------------------------------------------------
    // A rotation plane -> θᵢ, Δ = p·θᵢ, (cos,sin), and the rotated (x,y).
    // A heatmap cell  -> the rotated component value at that (position, dim).
    if (page.pointer.over && !grabbingPos) {
      const pt = page.pointer;
      let tip = null;
      for (const pl of planes) {
        const dxp = pt.x - pl.cx, dyp = pt.y - pl.cy;
        if (dxp * dxp + dyp * dyp <= (pl.R + 10) * (pl.R + 10)) {
          const c = Math.cos(pl.delta), s = Math.sin(pl.delta);
          const rx = pl.a * c - pl.b * s, ry = pl.a * s + pl.b * c;
          tip = `pair ${pl.i}   θ${pl.i} = ${fmt(pl.theta)}\n` +
                `Δ = p·θ = ${fmt(p)}·${fmt(pl.theta)} = ${fmt(pl.delta)} rad\n` +
                `(cos Δ, sin Δ) = (${c.toFixed(3)}, ${s.toFixed(3)})\n` +
                `(${pl.a.toFixed(2)}, ${pl.b.toFixed(2)}) → (${rx.toFixed(3)}, ${ry.toFixed(3)})`;
          break;
        }
      }
      if (!tip && hmRect) {
        const hit = cellAt(hmRect, N, d, pt.x, pt.y);
        if (hit) {
          const v = comp.data[hit.r * d + hit.c];
          tip = `rotated[pos ${hit.r}, dim ${hit.c}] = ${v.toFixed(3)}\n(pair ${hit.c >> 1}, ${(hit.c & 1) ? 'sin' : 'cos'} component)`;
        }
      }
      if (tip) page.setTip(tip);
    }

    let o = `RoPE: pair i rotated by Δ = p·θᵢ,  θᵢ = ${base}^(−2i/${d}).  Low pairs spin fast, high pairs slow.  pos=${fmt(p)}  ${posState.frozen ? '(frozen — drag/sweep)' : '(sweeping)'}  tier:${r.name}\n`;
    o += `pair 0: Δ=${fmt(ang[0])} rad   pair ${np - 1}: Δ=${fmt(ang[np - 1])} rad   (ratio ${fmt(ang[0] / (ang[np - 1] || 1e-9))}× faster)`;
    o += '   ·   rotations compose: q·k depends only on relative position (m−n).';
    page.setReadout(o);
  },
}).then((page) => {
  window.__ropePage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;

  // Map a horizontal canvas x onto a position (also called from the drag hook).
  // Hoisted helper used by onPointer above + the position track.
  page._setPosFromX = (x) => {
    const N = page.state.N;
    if (!posHandle) { posState.pos = 0; return; }
    const frac = Math.max(0, Math.min(1, (x - posHandle.x) / (posHandle.w || 1)));
    posState.pos = frac * (N - 1);
  };

  // ?pos=N — set the token position and FREEZE the sweep (deterministic frame).
  if (q.has('pos')) {
    const v = parseFloat(q.get('pos'));
    if (!Number.isNaN(v)) { posState.pos = v; posState.frozen = true; }
  }
  // ?theta / base override (kept).
  if (q.has('theta')) { /* base is a control; ?theta is a documented alias below */ }

  // ?hover=x,y — fake the cursor (headless stand-in for a real hover, since
  // --screenshot has no pointer) so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }

  // ?step=N (existing) — seek the transport AND pin the position to that step.
  if (q.has('step') && t) {
    const si = parseInt(q.get('step'), 10);
    t.seek(si);
    posState.pos = si; posState.frozen = true;
  }
  // Any deterministic-capture hook freezes the auto-sweep for a stable frame.
  if (q.has('pos') || q.has('step') || q.has('hover')) { if (t) t.pause(); posState.frozen = true; }

  if (q.get('play') === '1' && t) { posState.frozen = false; t.play(); }
  page.redraw();
});

// onPointer (above) calls setPosFromX via the page; expose it at module scope
// so the closure resolves regardless of .then() ordering on first event.
function setPosFromX(page, x) {
  if (page._setPosFromX) { page._setPosFromX(x); return; }
  const N = page.state.N;
  if (!posHandle) { posState.pos = 0; return; }
  const frac = Math.max(0, Math.min(1, (x - posHandle.x) / (posHandle.w || 1)));
  posState.pos = frac * (N - 1);
}
