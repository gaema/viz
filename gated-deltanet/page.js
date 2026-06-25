// gated-deltanet concept page -- the linear-attention delta-rule state update.
// A fixed [d×d] matrix state S (a key->value associative memory) is carried over
// the sequence and updated by the GatedDeltaNet rule:
//   Sₜ = αₜ·Sₜ₋₁·(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ ,   oₜ = Sₜ qₜ
// α = gate/decay (forget), β = write strength; (I − β k kᵀ) erases the value
// currently at key kₜ before writing vₜ there, so a repeated key overwrites
// instead of piling up (vs plain linear attention Sₜ = Sₜ₋₁ + vₜ kₜᵀ). The state
// matrix evolves as you step the sequence; drag a v component to change what is
// written. NOTE: cur is built ONLY by the transport compute (no draw-side
// rebuild -- that wipes drag edits; see the ssm-scan fix).
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', ORANGE = '#d2691e', PURPLE = '#8957e5', GREY = '#9aa4ad';
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

let cur = null, curStep = 0;
let sRect = null, wRect = null, kRect = null, vRect = null, qRect = null, oRect = null, cw = 22;
let grab = null;   // {i} dragging a v component at the current step

function buildData(st) {
  const L = st.L | 0, d = st.d | 0, seed = st.seed | 0;
  const K = seededRandn(seed, [L, d], { std: 1 }).data;
  for (let t = 0; t < L; t++) { let n = 0; for (let i = 0; i < d; i++) n += K[t * d + i] ** 2; n = Math.sqrt(n) || 1; for (let i = 0; i < d; i++) K[t * d + i] /= n; }  // unit keys
  const V = seededRandn(seed + 1, [L, d], { std: 1 }).data;
  const Q = seededRandn(seed + 2, [L, d], { std: 1 }).data;
  cur = { K, V, Q, L, d };
  return Array.from({ length: L }, (_, t) => ({ t, label: `step ${t}: gate ×α, erase at kₜ, write βvₜkₜᵀ, read oₜ=Sqₜ` }));
}

function scan(st) {
  const { K, V, Q, L, d } = cur, a = st.alpha, b = st.beta, delta = st.delta;
  const S = new Float32Array(d * d), states = [];
  for (let t = 0; t < L; t++) {
    const k = K.subarray(t * d, t * d + d), v = V.subarray(t * d, t * d + d), q = Q.subarray(t * d, t * d + d);
    if (delta) { const Sk = new Float32Array(d); for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += S[i * d + j] * k[j]; Sk[i] = s; }
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) S[i * d + j] = a * (S[i * d + j] - b * Sk[i] * k[j]) + b * v[i] * k[j];
    } else { for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) S[i * d + j] = a * S[i * d + j] + b * v[i] * k[j]; }
    const o = new Float32Array(d), W = new Float32Array(d * d);
    for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) { s += S[i * d + j] * q[j]; W[i * d + j] = b * v[i] * k[j]; } o[i] = s; }
    states.push({ S: Float32Array.from(S), o, W });
  }
  return states;
}

const strip = (r, data, d, rect, dom, labelEach) => {
  r.heatmap({ data, rows: 1, cols: d }, { rows: 1, cols: d, rect, ramp: ramps.diverging, domain: [-dom, dom] });
  r.grid({ stroke: 'rgba(0,0,0,0.12)' });
};

mount({
  mount: 'body',
  title: 'gated-deltanet — the linear-attention delta-rule state update',
  blurb: 'GatedDeltaNet (Qwen3-Next-style linear attention) keeps a fixed [d×d] matrix state S — a key→value associative memory — instead of a KV cache, and reads it with the query: o = S·q. The gated delta update: S ← α·S·(I − β·k·kᵀ) + β·v·kᵀ. α is a decay gate (forget); the delta rule (I − β·k·kᵀ) erases the value currently stored at key k before writing v there, so a repeated key overwrites instead of accumulating (the memory doesn’t saturate). Step the sequence to watch S evolve; drag a v component to change what is written; toggle the delta rule off to watch the memory saturate.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('L', { label: 'sequence length (L)', min: 4, max: 10, value: 6 });
    c.stepper('d', { label: 'key/value dim (d)', min: 3, max: 6, value: 4 });
    c.slider('alpha', { label: 'gate α (decay)', min: 0.3, max: 1, step: 0.05, value: 0.9 });
    c.slider('beta', { label: 'write β (delta)', min: 0, max: 1, step: 0.05, value: 0.85 });
    c.toggle('delta', { label: 'delta rule (erase old)', value: true });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.3, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !vRect) return;
    const d = cur.d;
    if (ev.type === 'down') { const h = cellAt(vRect, 1, d, ev.x, ev.y); grab = h ? { i: h.c } : null; }
    else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab && page.pointer.down) { const idx = curStep * d + grab.i; cur.V[idx] = Math.max(-3, Math.min(3, cur.V[idx] - ev.dy * 0.02)); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;                          // built by the transport compute
    const { K, V, Q, L, d } = cur;
    r.clear('#ffffff');
    const states = scan(st), s = page.step(), t = s ? s.t : L - 1;
    curStep = t;
    const cs = states[t], Sprev = t > 0 ? states[t - 1].S : new Float32Array(d * d);
    const sdom = Math.max(maxAbs(cs.S), 1e-3), wdom = Math.max(maxAbs(cs.W), 1e-3);
    const kvqdom = Math.max(maxAbs(K), maxAbs(V), maxAbs(Q), 0.5), odom = Math.max(maxAbs(cs.o), 0.5);

    r.label('Sₜ = αₜ · Sₜ₋₁ · (I − βₜ kₜ kₜᵀ)  +  βₜ vₜ kₜᵀ        oₜ = Sₜ qₜ', 18, 34, { color: INK, font: '13px ui-monospace, monospace' });

    const pad = 16, topY = 64;
    // left: the current token's k / v / q strips (v draggable) + α/β
    const lvX = pad + 28, sw = d * cw;
    const krow = topY + 6, vrow = krow + 34, qrow = vrow + 34;
    r.label(`token t=${t}`, pad, topY - 8, { color: INK, font: '11px ui-monospace, monospace' });
    kRect = { x: lvX, y: krow, w: sw, h: 22 }; strip(r, K.subarray(t * d, t * d + d), d, kRect, kvqdom);
    vRect = { x: lvX, y: vrow, w: sw, h: 22 }; strip(r, V.subarray(t * d, t * d + d), d, vRect, kvqdom);
    qRect = { x: lvX, y: qrow, w: sw, h: 22 }; strip(r, Q.subarray(t * d, t * d + d), d, qRect, kvqdom);
    r.label('kₜ (key, unit)', lvX, krow - 6, { color: BLUE, font: '10px ui-monospace, monospace' });
    r.label('vₜ (value) — drag ↕', lvX, vrow - 6, { color: ORANGE, font: '10px ui-monospace, monospace' });
    r.label('qₜ (query)', lvX, qrow - 6, { color: GREEN, font: '10px ui-monospace, monospace' });
    r.label(`α = ${st.alpha.toFixed(2)}  (decay/forget)`, pad, qrow + 44, { color: PURPLE, font: '11px ui-monospace, monospace' });
    r.label(`β = ${st.beta.toFixed(2)}  (write strength)`, pad, qrow + 62, { color: '#bc6c25', font: '11px ui-monospace, monospace' });
    r.label(st.delta ? 'delta rule ON: erase old value at kₜ, then write' : 'delta OFF: plain accumulate (memory saturates)', pad, qrow + 84, { color: st.delta ? INK : '#d1242f', font: '10px ui-monospace, monospace' });

    // center: the state matrix S [d×d]
    const Scell = Math.max(16, Math.min(34, (page.H * 0.42) / d));
    const Sx = lvX + sw + 70, Sy = topY + 8;
    sRect = { x: Sx, y: Sy, w: d * Scell, h: d * Scell };
    r.label('state  S [d×d]  (key→value memory)', Sx, Sy - 8, { color: INK, font: '12px ui-monospace, monospace' });
    r.heatmap({ data: cs.S, rows: d, cols: d }, { rows: d, cols: d, rect: sRect, ramp: ramps.diverging, domain: [-sdom, sdom] });
    r.grid({ stroke: 'rgba(0,0,0,0.12)' });
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = GREY; ctx.textAlign = 'center';
    ctx.fillText('key dim j →', Sx + d * Scell / 2, Sy + d * Scell + 12); ctx.save(); ctx.translate(Sx - 12, Sy + d * Scell / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('value dim i', 0, 0); ctx.restore(); ctx.restore();

    // written association + βvkᵀ
    const wcell = Math.max(10, Scell * 0.6), Wx = Sx + d * Scell + 54, Wy = Sy + 10;
    wRect = { x: Wx, y: Wy, w: d * wcell, h: d * wcell };
    r.label('+ βvₜkₜᵀ', Wx, Wy - 8, { color: ORANGE, font: '11px ui-monospace, monospace' });
    r.label('(write)', Wx, Wy + d * wcell + 12, { color: ORANGE, font: '9px ui-monospace, monospace' });
    r.heatmap({ data: cs.W, rows: d, cols: d }, { rows: d, cols: d, rect: wRect, ramp: ramps.diverging, domain: [-wdom, wdom] });
    r.grid({ stroke: 'rgba(0,0,0,0.10)' });
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(Sx + d * Scell + 6, Sy + d * Scell / 2); ctx.lineTo(Wx - 6, Wy + d * wcell / 2); ctx.stroke(); ctx.restore();

    // output o = S q
    const oX = Wx + d * wcell + 56, oc = Math.max(16, Math.min(30, Scell));
    oRect = { x: oX, y: Sy, w: oc, h: d * oc };
    r.label('o = Sq', oX - 4, Sy - 8, { color: GREEN, font: '11px ui-monospace, monospace' });
    ctx.save();
    for (let i = 0; i < d; i++) { const v = cs.o[i], h = (v / odom) * (oc * 0.46), y0 = Sy + i * oc + oc / 2; ctx.fillStyle = 'rgba(44,160,44,0.7)'; ctx.fillRect(oX, v >= 0 ? y0 - h : y0, oc, Math.abs(h)); ctx.strokeStyle = '#dfe3e6'; ctx.strokeRect(oX, Sy + i * oc, oc, oc); }
    ctx.restore();

    // time strip
    const tly = page.H - pad - 22, tbw = Math.min(26, (page.W - 2 * pad) / L);
    r.label('seq:', pad, tly - 6, { color: GREY, font: '10px ui-monospace, monospace' });
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let p = 0; p < L; p++) { const bx = pad + 28 + p * tbw; ctx.fillStyle = p < t ? 'rgba(31,111,235,0.35)' : p === t ? BLUE : '#eef0f2'; ctx.fillRect(bx, tly, tbw - 2, 18); ctx.fillStyle = p <= t ? '#fff' : '#9aa4ad'; ctx.fillText(String(p), bx + (tbw - 2) / 2, tly + 9); } ctx.restore();

    // hover
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const hs = cellAt(sRect, d, d, p.x, p.y);
      if (hs) page.setTip(`S[val ${hs.r}, key ${hs.c}] = ${cs.S[hs.r * d + hs.c].toFixed(3)}\nassociation: key dim ${hs.c} → value dim ${hs.r}`);
      else { for (const [nm, data, rect, col] of [['k', K.subarray(t * d, t * d + d), kRect, 'key'], ['v', V.subarray(t * d, t * d + d), vRect, 'value (drag ↕)'], ['q', Q.subarray(t * d, t * d + d), qRect, 'query']]) { const h = rect && cellAt(rect, 1, d, p.x, p.y); if (h) { page.setTip(`${nm}[${h.c}] = ${data[h.c].toFixed(3)}\n${col}`); break; } }
        if (oRect && p.x >= oRect.x && p.x <= oRect.x + oRect.w && p.y >= oRect.y && p.y <= oRect.y + d * (oRect.h / d)) { const i = Math.floor((p.y - oRect.y) / (oRect.h / d)); if (i >= 0 && i < d) page.setTip(`o[${i}] = (S·q)[${i}] = ${cs.o[i].toFixed(3)}\noutput = read the memory with q`); } }
    }

    let o = `linear attention: a [${d}×${d}] matrix memory S (constant size, no KV cache) updated by the gated delta rule; read with o=Sq.   α=${st.alpha.toFixed(2)} β=${st.beta.toFixed(2)} ${st.delta ? 'delta-ON' : 'delta-OFF'}    tier:${r.name}\n`;
    o += s ? `step ${t}/${L - 1}: ${st.delta ? 'erase the old value at key kₜ, then' : 'accumulate'} write βvₜkₜᵀ (decayed by α); o=Sqₜ.`
      : `${L} steps. ${st.delta ? 'Delta rule overwrites a repeated key — bounded memory.' : 'Delta OFF: associations just accumulate; the memory saturates.'}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__gdnPage = page;
  const q = new URLSearchParams(location.search);
  const tp = page.controls._transport;
  if (q.has('delta')) page.controls.set('delta', q.get('delta') !== '0');
  // ?drag=i,val sets the current step's v[i] (headless stand-in). Apply after seek.
  let pend = null; if (q.has('drag')) pend = q.get('drag').split(',').map(Number);
  if (q.has('step') || q.has('drag') || q.has('hover') || q.has('delta')) { if (tp) tp.pause(); }
  if (q.has('step') && tp) tp.seek(parseInt(q.get('step'), 10));
  if (pend && cur) { const tt = (tp && tp.index >= 0) ? tp.index : cur.L - 1; if (pend[0] < cur.d) cur.V[tt * cur.d + pend[0]] = pend[1]; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.get('play') === '1' && tp) tp.play();
  page.redraw();
});
