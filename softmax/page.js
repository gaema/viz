// softmax concept page -- logits -> probabilities, phase by phase.
// Interactive per the framework contract (plan/framework.md): drag any logit
// bar vertically to change z[i] and watch the probabilities redistribute live
// (softmax recomputed); hover a logit for its value, hover a probability for
// its full derivation; the max→exp→sum→normalize transport auto-plays + loops.
import { mount } from '../framework/layout.js';
import { softmaxSteps, collectSteps, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', ORANGE = '#d2691e', RED = '#d6273a';
const fmt = (x) => (Math.abs(x) >= 1e4 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(2) : String(Number(x.toPrecision(3))));

// Shared between buildData(), draw(), and onPointer(). buildData() runs (via
// transport rebuild) before the matching draw, so `cur` is fresh. Drag edits
// mutate cur.v in place; resync() rebuilds the step sequence from it.
let cur = { v: null };
let logitRect = null, probRect = null;   // band rects captured in draw for hit-testing
let grab = null;                          // {i} while dragging a logit bar

function buildData(st) {
  const v = seededRandn(st.seed | 0, st.k, { std: 2 });   // spread logits
  cur = { v };
  return collectSteps(softmaxSteps(v, { temp: st.temp }));
}

// Recompute the transport's step list from the (possibly edited) cur.v without
// regenerating from the seed -- so a drag edit survives + scrubs right.
function resync(page) {
  const t = page.controls._transport;
  if (!t) return;
  t.steps = collectSteps(softmaxSteps(cur.v, { temp: page.state.temp }));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

// Column index of a bar row under (x,y): x within the row's span, y within the
// band. Bars are laid out one column per i across rect.w; the whole column is
// the hit target (a thin bar is fiddly to land on, esp. headless).
function barAt(rect, k, x, y) {
  if (!rect || x < rect.x || x > rect.x + rect.w || y < rect.y || y > rect.y + rect.h) return -1;
  const i = Math.floor((x - rect.x) / (rect.w / k));
  return (i >= 0 && i < k) ? i : -1;
}

// One row of k bars. opts: {signed, color, domain, upto, activeI, maxIdx}.
function bars(page, vals, rect, o) {
  const ctx = page.ctx, k = vals.length, colW = rect.w / k, bw = Math.min(46, colW * 0.6), dom = o.domain || 1;
  const baseY = o.signed ? rect.y + rect.h * 0.52 : rect.y + rect.h - 2;
  const maxH = o.signed ? rect.h * 0.42 : rect.h - 16;
  ctx.save();
  if (o.signed) { ctx.strokeStyle = '#d0d7de'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(rect.x, baseY); ctx.lineTo(rect.x + rect.w, baseY); ctx.stroke(); }
  ctx.textAlign = 'center';
  for (let i = 0; i < k; i++) {
    if (o.upto != null && i > o.upto) continue;
    const cx = rect.x + i * colW + colW / 2, v = vals[i];
    const h = Math.min(maxH, (Math.abs(v) / dom) * maxH), top = o.signed ? (v >= 0 ? baseY - h : baseY) : baseY - h;
    ctx.globalAlpha = i === o.activeI ? 1 : 0.82;
    ctx.fillStyle = o.color; ctx.fillRect(cx - bw / 2, top, bw, h);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#3a4047'; ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(fmt(v), cx, (o.signed && v < 0) ? baseY + h + 10 : baseY - h - 4);
    ctx.fillStyle = '#9aa4ad'; ctx.fillText(String(i), cx, rect.y + rect.h + 1);
    if (i === o.maxIdx) { ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.strokeRect(cx - bw / 2 - 1.5, Math.min(top, baseY) - 1.5, bw + 3, h + 3); }
    if (i === o.activeI) { ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(cx - bw / 2 - 1.5, Math.min(top, baseY) - 1.5, bw + 3, h + 3); }
  }
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'softmax — logits → probabilities',
  blurb: 'Turn a vector of logits into a probability distribution. Drag any logit bar ↕ to change z[i] and watch the probabilities redistribute live; hover a logit for its value, a probability for its derivation. Scrub (or let it play): find the max (subtracted for stability), exponentiate each, sum, then normalize. Temperature flattens (high) or sharpens (low) the result.',
  prefer: 'canvas2d',
  aspect: '8 / 5',
  compare: { key: 'temp', a: 0.3, b: 3, labelA: 'low T — sharp (≈ greedy)', labelB: 'high T — flat (≈ uniform)' },
  autoplay: true,
  challenges: [
    { goal: 'Sharpen to near one-hot — top class ≥ 0.9.', hint: 'lower the temperature T toward 0.1 AND drag one logit bar far above the others.', check: (api) => ({ solved: (api.probe.maxP ?? 0) >= 0.9, detail: `top class p = ${((api.probe.maxP ?? 0) * 100).toFixed(0)}% (need ≥ 90%)` }) },
    { goal: 'Flatten to near-uniform — top class within 5% of 1/k.', hint: 'raise the temperature T toward the max.', check: (api) => { const uni = 1 / Math.max(1, api.probe.k ?? 1); return { solved: (api.probe.maxP ?? 1) <= uni + 0.05, detail: `top ${((api.probe.maxP ?? 1) * 100).toFixed(0)}% vs uniform ${(uni * 100).toFixed(0)}%` }; } },
  ],
  controls: (c, page) => {
    c.stepper('k', { label: 'classes (k)', min: 2, max: 10, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.slider('temp', { label: 'temperature T', min: 0.2, max: 4, step: 0.1, value: 1, rebuild: true, format: (v) => (+v).toFixed(1) });
    c.transport({ compute: () => buildData(page.state), speed: 6, loop: true });
  },
  // Direct manipulation: grab a logit bar, drag vertically to change z[i]; the
  // probabilities recompute live. This is the softmax "aha".
  onPointer: (page, ev) => {
    const k = page.state.k;
    if (ev.type === 'down') {
      grab = null;
      const i = barAt(logitRect, k, ev.x, ev.y);
      if (i >= 0) grab = { i };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down && cur.v) {
      cur.v[grab.i] = Math.max(-8, Math.min(8, cur.v[grab.i] - ev.dy * 0.04));  // drag up = larger logit
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, st = page.state, v = cur.v;
    if (!v) return;
    r.clear('#ffffff');
    const k = v.length, T = st.temp;
    const z = Float32Array.from(v, (x) => x / T);
    const mx = Math.max(...z), mi = z.indexOf(mx);
    const e = Float32Array.from(z, (x) => Math.exp(x - mx));
    const sum = e.reduce((a, b) => a + b, 0);
    const p = Float32Array.from(e, (x) => x / sum);
    const maxAbsV = Math.max(1e-9, ...Array.from(v, Math.abs)), maxP = Math.max(...p);
    page.probe = { maxP, k: p.length };

    // Reveal state from the current step phase.
    const s = page.step();
    let expUpto = k - 1, probUpto = k - 1, activeI = -1;
    if (s) {
      if (s.phase === 'max') { expUpto = -1; probUpto = -1; activeI = mi; }
      else if (s.phase === 'exp') { expUpto = s.i; probUpto = -1; activeI = s.i; }
      else if (s.phase === 'sum') { expUpto = k - 1; probUpto = -1; }
      else if (s.phase === 'norm') { expUpto = k - 1; probUpto = s.i; activeI = s.i; }
    }

    const pad = 18, lblW = 132, x = pad + lblW, w = page.W - x - pad, top = 40;
    const bandH = (page.H - top - 14) / 3 - 14;
    const band = (n) => ({ x, y: top + n * (bandH + 18), w, h: bandH });
    const rowLabel = (t, n, col) => r.label(t, pad, top + n * (bandH + 18) + bandH * 0.5, { color: col, font: '12px ui-monospace, monospace' });

    logitRect = band(0);
    probRect = band(2);

    rowLabel('logits z', 0, BLUE);
    bars(page, v, band(0), { signed: true, color: BLUE, domain: maxAbsV, maxIdx: mi, activeI: (s && s.phase !== 'norm') ? activeI : -1 });
    rowLabel('exp(zᵢ/T − max)', 1, GREEN);
    bars(page, e, band(1), { signed: false, color: GREEN, domain: 1, upto: expUpto, activeI: (s && s.phase === 'exp') ? activeI : -1 });
    rowLabel('softmax p', 2, ORANGE);
    bars(page, p, band(2), { signed: false, color: ORANGE, domain: maxP, upto: probUpto, activeI: (s && s.phase === 'norm') ? activeI : -1 });

    // Hover-to-inspect: tooltip with the logit value or the probability derivation.
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      const li = barAt(logitRect, k, pt.x, pt.y);
      const pi = barAt(probRect, k, pt.x, pt.y);
      let tip = null;
      if (li >= 0) tip = `z[${li}] = ${v[li].toFixed(3)}\ndrag ↕ to change`;
      else if (pi >= 0) {
        const zi = v[pi] / T;
        tip = `p[${pi}] = exp(z[${pi}]/T − max) / Σexp\n= e^(${(zi - mx).toFixed(2)}) / ${fmt(sum)}\n= ${e[pi].toFixed(3)} / ${fmt(sum)} = ${p[pi].toFixed(4)}`;
      }
      if (tip) page.setTip(tip);
    }

    const pSum = probUpto < 0 ? 0 : Array.from(p).slice(0, probUpto + 1).reduce((a, b) => a + b, 0);
    let out = `softmax(z)ᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)    T = ${T.toFixed(1)}    argmax = z[${mi}]    Σexp = ${fmt(sum)}    tier:${r.name}\n`;
    out += s ? `${s.label}\n` : '(drag logit bars ↕ to edit · press ▶ or scrub: max → exp → sum → normalize)\n';
    out += `Σ p (revealed) = ${fmt(pSum)}${probUpto === k - 1 ? '  →  1.0 ✓' : ''}`;
    page.setReadout(out);
  },
}).then((page) => {
  window.__softmaxPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=i,val sets logit i to a value (headless stand-in for a vertical drag).
  if (q.has('drag')) {
    const [i, val] = q.get('drag').split(',');
    if (cur.v && +i >= 0 && +i < cur.v.length) { cur.v[+i] = +val; resync(page); }
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover,
  // since --screenshot has no pointer) so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }   // deterministic frame for capture
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
