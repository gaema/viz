// guidance concept page -- classifier-free guidance (Ho & Salimans,
// arXiv:2207.12598): run the denoiser TWICE per step, once with the prompt and
// once without, and combine the two predictions as
//
//     eps~ = eps_u + w * (eps_c - eps_u)
//
// The point the page exists to make: that is an EXTRAPOLATION, not a filter.
// At w > 1 the result lies BEYOND the conditional prediction on the ray that
// leaves the unconditional one -- so nothing in the formula stops it at the
// plausible answer, which is why a large w over-saturates instead of merely
// sharpening. The difference vector and its amplified copy are drawn live.
//
// The model is a 2D toy, and the math is exact rather than mimed: the data is a
// mixture of 8 Gaussians (4 prompt classes x 2 sub-modes, so each class has its
// OWN internal variety, which is the diversity guidance spends). For a Gaussian
// mixture the denoiser's eps-prediction is available in closed form, so eps_u,
// eps_c and eps~ on screen are computed, not drawn by hand.
//
// The conditional model is deliberately IMPERFECT (the "conditioning leak" lambda):
// with lambda = 0 it is the exact conditional score and w > 1 buys no adherence at
// all, only stereotype; with lambda > 0 it is an under-fit conditional model and
// w > 1 measurably rescues adherence. That is the honest reason CFG helps in
// practice, and the slider lets a reader see both regimes.
//
// NOT this page: the training objective / how the noise is added (see the
// diffusion-noise page) and the integrator + step count (see the
// diffusion-sampler page). Steps are fixed at 24 deterministic DDIM steps here
// precisely so the step axis is not a variable this page competes over.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, mixColor, inkOn, onThemeChange, themeMode } from '../framework/theme.js';

// ---------------------------------------------------------------------------
// The toy distribution: 4 prompt classes, each a 2-component Gaussian mixture.
// ---------------------------------------------------------------------------
const CLS = ['A', 'B', 'C', 'D'];
const SUBW = [0.62, 0.38];        // the two sub-modes of a class are UNEQUAL, so
                                  // sharpening has a dominant one to collapse onto
const SD = 0.13;                  // per-sub-mode data std
const RAD = 1.15, OFF = 0.34;     // class radius, sub-mode offset
const TSTEP = 24;                 // fixed step count (see the header note)
const SIG_MAX = 1.2, SIG_MIN = 0.02;
const NS = 12;                    // samples per arm in the diversity strip
const VIEW = 2.0;                // math half-extent drawn in the map panel

const MU = [];
for (let k = 0; k < 4; k++) {
  const ang = -Math.PI / 2 + k * (Math.PI / 2);
  const bx = Math.cos(ang) * RAD, by = Math.sin(ang) * RAD;
  const ux = -Math.sin(ang), uy = Math.cos(ang);
  for (let j = 0; j < 2; j++) {
    const o = j === 0 ? OFF : -OFF;
    MU.push({ x: bx + ux * o, y: by + uy * o, cls: k, sub: j, w: SUBW[j] });
  }
}

// Variance-exploding schedule: x_t = x_0 + sigma_t * noise, sigma geometric.
const sigmaAt = (i) => SIG_MAX * Math.pow(SIG_MIN / SIG_MAX, i / TSTEP);

// eps-prediction of the exact denoiser for a Gaussian mixture with the given
// per-mode prior. Posterior responsibility r_m, then eps = sigma/(sd^2+sigma^2)
// * sum_m r_m (x - mu_m). Exact -- this is the whole model.
function epsMix(x, y, sig, prior) {
  const s2 = SD * SD + sig * sig;
  const lg = new Array(MU.length);
  let best = -Infinity;
  for (let m = 0; m < MU.length; m++) {
    const dx = x - MU[m].x, dy = y - MU[m].y;
    lg[m] = Math.log(Math.max(prior[m], 1e-12)) - (dx * dx + dy * dy) / (2 * s2);
    if (lg[m] > best) best = lg[m];
  }
  let sum = 0;
  for (let m = 0; m < MU.length; m++) { lg[m] = Math.exp(lg[m] - best); sum += lg[m]; }
  let ex = 0, ey = 0;
  for (let m = 0; m < MU.length; m++) { const q = lg[m] / sum; ex += q * (x - MU[m].x); ey += q * (y - MU[m].y); }
  const f = sig / s2;
  return [ex * f, ey * f];
}

// Priors. Unconditional = every mode. Conditional = the prompt's class carries
// (1 - leak), the other three share `leak` -- leak = 0 is a perfect conditional
// model, leak = 0.75 is indistinguishable from the unconditional one.
const priorU = () => MU.map((m) => m.w / 4);
const priorC = (c, leak) => MU.map((m) => m.w * (m.cls === c ? 1 - leak : leak / 3));

const nearestMode = (x, y) => {
  let bi = 0, bd = Infinity;
  for (let m = 0; m < MU.length; m++) { const dx = x - MU[m].x, dy = y - MU[m].y, d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = m; } }
  return bi;
};

// ---------------------------------------------------------------------------
// The two paired runs: arm A at w = 1 (the conditional prediction alone) and
// arm B at the chosen w, from the SAME 12 initial noise draws. Pairing the
// seeds is what makes the diversity difference attributable to w and not to
// luck -- it is the control, not a decoration.
// ---------------------------------------------------------------------------
function buildSteps(st) {
  const c = Math.max(0, CLS.indexOf(st.cls));
  const pu = priorU(), pc = priorC(c, st.leak);
  const gs = Math.min(st.gStart | 0, st.gEnd | 0), ge = Math.max(st.gStart | 0, st.gEnd | 0);
  const n0 = seededRandn((st.seed | 0) * 7 + 3, NS * 2, { std: SIG_MAX });
  const A = Float64Array.from(n0), B = Float64Array.from(n0);
  const recs = [{ i: 0, sig: sigmaAt(0), w: 1, guided: false, A: A.slice(), B: B.slice(), label: `step 0 / ${TSTEP} — pure noise, sigma = ${sigmaAt(0).toFixed(2)}` }];
  for (let i = 0; i < TSTEP; i++) {
    const sig = sigmaAt(i), sn = sigmaAt(i + 1), d = sig - sn;
    const guided = i >= gs && i < ge;
    const wi = guided ? st.w : 1;
    for (let p = 0; p < NS; p++) {
      const ax = A[p * 2], ay = A[p * 2 + 1];
      const ea = epsMix(ax, ay, sig, pc);              // w = 1 collapses to eps_c
      A[p * 2] = ax - d * ea[0]; A[p * 2 + 1] = ay - d * ea[1];
      const bx = B[p * 2], by = B[p * 2 + 1];
      const eu = epsMix(bx, by, sig, pu), ec = epsMix(bx, by, sig, pc);
      const gx = eu[0] + wi * (ec[0] - eu[0]), gy = eu[1] + wi * (ec[1] - eu[1]);
      B[p * 2] = bx - d * gx; B[p * 2 + 1] = by - d * gy;
    }
    recs.push({
      i: i + 1, sig: sn, w: wi, guided, A: A.slice(), B: B.slice(),
      label: `step ${i + 1} / ${TSTEP} — sigma ${sig.toFixed(2)} → ${sn.toFixed(2)}, w = ${wi.toFixed(1)}${guided ? ' (2 passes)' : ' (1 pass)'}`,
    });
  }
  return recs;
}

// Sample-set statistics, all measured off the record on screen.
function stats(arr, c) {
  let hit = 0, spread = 0, np = 0, typ = 0;
  const subs = new Set();
  for (let p = 0; p < NS; p++) {
    const x = arr[p * 2], y = arr[p * 2 + 1], m = nearestMode(x, y);
    if (MU[m].cls === c) { hit++; subs.add(m); }
    typ += Math.hypot(x - MU[m].x, y - MU[m].y) / SD;
    for (let q = p + 1; q < NS; q++) { spread += Math.hypot(x - arr[q * 2], y - arr[q * 2 + 1]); np++; }
  }
  return { adherence: hit / NS, modes: subs.size, spread: np ? spread / np : 0, typ: typ / NS };
}

// ---------------------------------------------------------------------------
// Page state that is not a control: the draggable probe point + hit geometry.
// ---------------------------------------------------------------------------
// Read the query ONCE, at module load: mount()'s deep-link sync rewrites
// location.search from the control state during mount (?compare=1 clicks the
// A/B toggle, whose seek() fires the sync), so a hook parsed after mount would
// find its own parameters already gone.
const Q = new URLSearchParams(typeof location !== 'undefined' ? location.search : '');
let probe = { x: 0.62, y: -0.62 };
let geom = null;
let grab = null;                  // 'probe' | 'gs' | 'ge' while dragging
const fmtV = (v) => `(${v[0] >= 0 ? ' ' : ''}${v[0].toFixed(2)}, ${v[1] >= 0 ? ' ' : ''}${v[1].toFixed(2)})`;
const mag = (v) => Math.hypot(v[0], v[1]);
const pct = (a, b) => (b > 1e-9 ? `${Math.round((a / b) * 100)}%` : 'n/a');

mount({
  mount: 'body',
  title: 'guidance — classifier-free guidance: two predictions, then extrapolate',
  blurb: 'A conditional denoiser is run TWICE per step — once with the prompt (ε_c) and once with it dropped (ε_u) — and the two are combined as ε~ = ε_u + w·(ε_c − ε_u). Read that as a vector: it starts at the unconditional prediction and travels w times the difference. At w = 1 you land exactly on the conditional prediction; at w > 1 you keep going PAST it, which is why large w does not merely sharpen, it over-shoots. The map shows the three vectors and the amplified difference at a draggable probe point; the strip runs 12 paired seeds at w = 1 and at your w so mode collapse is visible rather than asserted; the timeline restricts guidance to an interval of steps — the modern refinement (Kynkäänniemi et al., arXiv:2404.07724, who found a middle interval best for image models). In this toy the noisy early steps are the ones that decide which mode you land in, so shrinking the interval hands diversity back, and trimming the quiet tail costs nothing but saves passes. Guidance costs a second forward pass on every guided step, so the interval is also the compute dial.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  autoplay: true,
  compare: { key: 'w', a: 1, b: 12, labelA: 'w = 1 — the conditional prediction, nothing extrapolated', labelB: 'w = 12 — extrapolated far past it', rebuild: true },
  challenges: [
    { goal: 'Collapse the samples onto a single sub-mode — one stereotype for the whole prompt.', hint: 'raise w well above 1 with guidance on every step.', check: (api) => ({ solved: (api.probe.modesB ?? 2) <= 1 && (api.probe.adhB ?? 0) > 0.5, detail: `${api.probe.modesB ?? '–'} of 2 sub-modes populated` }) },
    { goal: 'Make guidance actually buy something: with an under-fit conditional model (leak ≥ 0.3), reach ≥ 90% prompt adherence.', hint: 'raise the conditioning leak, watch adherence fall, then raise w until it comes back.', check: (api) => ({ solved: (api.state.leak ?? 0) >= 0.3 && (api.probe.adhB ?? 0) >= 0.9, detail: `leak ${(api.state.leak ?? 0).toFixed(2)} · adherence ${Math.round((api.probe.adhB ?? 0) * 100)}%` }) },
    { goal: 'Buy the same adherence for less compute: ≥ 90% on-prompt with guidance on at most 12 of the 24 steps.', hint: 'the steps that decide WHICH mode you land in are the noisy ones — shrink "guide to" and watch the pass count fall while the strip does not move.', check: (api) => ({ solved: (api.probe.adhB ?? 0) >= 0.9 && (api.probe.guidedSteps ?? 99) <= 12 && (api.state.w ?? 1) >= 5, detail: `adherence ${Math.round((api.probe.adhB ?? 0) * 100)}% · guided on ${api.probe.guidedSteps ?? '–'} of ${24} steps · ${api.probe.passes ?? '–'} passes` }) },
  ],
  controls: (c, page) => {
    c.slider('w', { label: 'guidance scale w', min: 0, max: 20, step: 0.1, value: 3, rebuild: true });
    c.select('cls', { label: 'prompt (class)', options: CLS, value: 'A', rebuild: true });
    c.slider('leak', { label: 'conditioning leak λ (model imperfection)', min: 0, max: 0.75, step: 0.05, value: 0.25, rebuild: true });
    c.slider('gStart', { label: 'guide from step', min: 0, max: TSTEP, step: 1, value: 0, rebuild: true });
    c.slider('gEnd', { label: 'guide to step', min: 0, max: TSTEP, step: 1, value: TSTEP, rebuild: true });
    c.slider('seed', { label: 'seed (12 paired samples)', min: 0, max: 99, step: 1, value: 11, rebuild: true });
    c.transport({ compute: () => buildSteps(page.state), speed: 4, loop: true, onStep: (_rec, idx) => { c.state.step = idx; } });
  },

  // Direct manipulation: drag the probe point anywhere on the map (the three
  // vectors recompute under the hand), or drag either end of the guidance
  // interval on the timeline.
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (ev.type === 'down') {
      grab = null;
      if (ev.y >= g.tl.y - 12 && ev.y <= g.tl.y + g.tl.h + 12 && ev.x >= g.tl.x - 16 && ev.x <= g.tl.x + g.tl.w + 16) {
        const s = Math.round(((ev.x - g.tl.x) / g.tl.w) * TSTEP);
        grab = Math.abs(s - page.state.gStart) <= Math.abs(s - page.state.gEnd) ? 'gs' : 'ge';
      } else if (ev.x >= g.map.x && ev.x <= g.map.x + g.map.w && ev.y >= g.map.y && ev.y <= g.map.y + g.map.h) {
        grab = 'probe';
      }
    }
    if (!grab || !page.pointer.down) return;
    if (grab === 'probe') { probe = { x: g.ux(ev.x), y: g.uy(ev.y) }; page.redraw(); return; }
    const s = Math.max(0, Math.min(TSTEP, Math.round(((ev.x - g.tl.x) / g.tl.w) * TSTEP)));
    if (grab === 'gs') page.controls.set('gStart', Math.min(s, page.state.gEnd), { rebuild: true });
    else page.controls.set('gEnd', Math.max(s, page.state.gStart), { rebuild: true });
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const rec = page.step() || { i: 0, sig: sigmaAt(0), w: 1, guided: false, A: null, B: null };
    const c = Math.max(0, CLS.indexOf(st.cls));
    const pu = priorU(), pc = priorC(c, st.leak);
    const gs = Math.min(st.gStart | 0, st.gEnd | 0), ge = Math.max(st.gStart | 0, st.gEnd | 0);
    const guidedSteps = ge - gs, passes = TSTEP + guidedSteps;
    const steps = page.controls._transport ? page.controls._transport.steps : [];
    const W = page.W, H = page.H, pad = 12;

    // ---- panel geometry ---------------------------------------------------
    const tlH = 74;
    const mapW = Math.round(W * 0.5);
    const map = { x: pad, y: 26, w: mapW - pad, h: H - 26 - tlH - 14 };
    const strip = { x: mapW + 14, y: 26, w: W - mapW - 14 - pad, h: map.h };
    const tl = { x: pad + 96, y: H - tlH + 22, w: W - pad * 2 - 96 - 120, h: 22 };
    const sc = Math.min(map.w, map.h) / (2 * VIEW);
    const cx = map.x + map.w / 2, cy = map.y + map.h / 2;
    const PX = (u) => cx + u * sc, PY = (v) => cy - v * sc;
    geom = { map, strip, tl, PX, PY, ux: (px) => (px - cx) / sc, uy: (py) => (cy - py) / sc };

    // ---- map panel: modes, paired trajectories, the three vectors ----------
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
    ctx.strokeRect(map.x, map.y, map.w, map.h);
    ctx.restore();
    r.label(`the data: 4 prompt classes × 2 sub-modes   ·   σ_t = ${rec.sig.toFixed(3)}`, map.x, map.y - 8, { color: T.n11, font: '11px ui-monospace, monospace' });

    ctx.save();
    ctx.beginPath(); ctx.rect(map.x + 1, map.y + 1, map.w - 2, map.h - 2); ctx.clip();

    for (let m = 0; m < MU.length; m++) {
      const hue = categorical(MU[m].cls), isT = MU[m].cls === c;
      ctx.beginPath(); ctx.arc(PX(MU[m].x), PY(MU[m].y), SD * 2 * sc, 0, Math.PI * 2);
      ctx.fillStyle = alphaOf(hue, isT ? 0.16 : 0.06); ctx.fill();
      ctx.strokeStyle = alphaOf(hue, isT ? 0.85 : 0.3); ctx.lineWidth = isT ? 1.6 : 1; ctx.setLineDash(isT ? [] : [3, 3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = alphaOf(hue, isT ? 1 : 0.5); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`${CLS[MU[m].cls]}${MU[m].sub + 1}`, PX(MU[m].x), PY(MU[m].y));
    }

    // trajectories so far: arm A faint, arm B in the prompt's hue
    if (steps.length && rec.i > 0) {
      for (const [arm, col, alp, lw] of [['A', T.n9, 0.35, 1], ['B', categorical(c), 0.55, 1.6]]) {
        for (let p = 0; p < NS; p++) {
          ctx.beginPath();
          for (let s = 0; s <= rec.i && s < steps.length; s++) {
            const a = steps[s][arm]; const X = PX(a[p * 2]), Y = PY(a[p * 2 + 1]);
            if (s === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.strokeStyle = alphaOf(col, alp); ctx.lineWidth = lw; ctx.stroke();
        }
      }
      for (let p = 0; p < NS; p++) {
        const a = rec.A, b = rec.B;
        ctx.beginPath(); ctx.arc(PX(a[p * 2]), PY(a[p * 2 + 1]), 3, 0, Math.PI * 2);
        ctx.strokeStyle = T.n9; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.beginPath(); ctx.arc(PX(b[p * 2]), PY(b[p * 2 + 1]), 3.4, 0, Math.PI * 2);
        ctx.fillStyle = alphaOf(categorical(c), 0.9); ctx.fill();
      }
    }

    // --- the three predictions at the probe point, and the amplified delta ---
    // w for the step ABOUT to be taken from here (the marker's step), so the
    // probe vectors agree with the timeline bar under the marker.
    const si = Math.min(Math.max(rec.i, 0), TSTEP - 1);
    const wNow = si >= gs && si < ge ? st.w : 1;
    const eu = epsMix(probe.x, probe.y, rec.sig, pu);
    const ec = epsMix(probe.x, probe.y, rec.sig, pc);
    const dl = [ec[0] - eu[0], ec[1] - eu[1]];
    const eg = [eu[0] + wNow * dl[0], eu[1] + wNow * dl[1]];
    const P0 = { x: PX(probe.x), y: PY(probe.y) };
    // ONE pixels-per-eps-unit for all three arrows, pinned to the two MODEL
    // predictions so they stay readable at any w. eps~ is then drawn true to
    // that same scale and simply RUNS OFF the panel once w is large -- which is
    // the honest picture, and a better one than squashing every arrow to fit.
    const K = 72 / Math.max(mag(eu), mag(ec), 1e-6);
    const tip = (v) => ({ x: P0.x + v[0] * K, y: P0.y - v[1] * K });
    const tU = tip(eu), tC = tip(ec), tGfull = tip(eg);
    // Where the eps~ ray leaves the panel (if it does): the label goes there.
    const clipSeg = (a, b, m, inset) => {
      let t = 1;
      const dx = b.x - a.x, dy = b.y - a.y;
      const lim = (num, den) => { if (Math.abs(den) > 1e-9) { const tt = num / den; if (tt >= 0 && tt < t) t = tt; } };
      lim(m.x + inset - a.x, dx); lim(m.x + m.w - inset - a.x, dx);
      lim(m.y + inset - a.y, dy); lim(m.y + m.h - inset - a.y, dy);
      return { p: { x: a.x + dx * t, y: a.y + dy * t }, clipped: t < 1 };
    };
    const cg = clipSeg(P0, tGfull, map, 34);
    const tG = cg.clipped ? cg.p : tGfull;

    // delta, then its amplified copy: the SAME ray, continued past eps_c
    ctx.save();
    ctx.setLineDash([4, 3]); ctx.lineWidth = 1.3; ctx.strokeStyle = T.n10;
    ctx.beginPath(); ctx.moveTo(tU.x, tU.y); ctx.lineTo(tC.x, tC.y); ctx.stroke();
    ctx.strokeStyle = alphaOf(T.violet, 0.85); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(tU.x, tU.y); ctx.lineTo(tGfull.x, tGfull.y); ctx.stroke();
    ctx.setLineDash([]);
    // the w = 1 landmark: a tick you can watch the guided tip travel past
    const nx = -(tC.y - tU.y), ny = tC.x - tU.x, nl = Math.hypot(nx, ny) || 1;
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(tC.x + (nx / nl) * 5, tC.y + (ny / nl) * 5); ctx.lineTo(tC.x - (nx / nl) * 5, tC.y - (ny / nl) * 5); ctx.stroke();
    ctx.restore();

    r.arrow(P0, tU, { color: T.teal, width: 2.4, alpha: 0.95 });
    r.arrow(P0, tC, { color: T.accent, width: 2.4, alpha: 0.95 });
    if (cg.clipped) {
      ctx.save();
      ctx.strokeStyle = T.violet; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(P0.x, P0.y); ctx.lineTo(tG.x, tG.y); ctx.stroke();
      ctx.restore();
    } else r.arrow(P0, tG, { color: T.violet, width: 3, alpha: 1 });
    ctx.save();
    ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
    ctx.fillStyle = T.teal; ctx.fillText('ε_u', tU.x + 5, tU.y);
    ctx.fillStyle = T.accent; ctx.fillText('ε_c  (w=1)', tC.x + 7, tC.y);
    ctx.fillStyle = T.violet;
    const gLbl = cg.clipped ? `» ε~ w=${wNow.toFixed(1)} runs off the panel — ${pct(mag(eg), mag(ec))} of |ε_c|` : `ε~  w=${wNow.toFixed(1)}`;
    const gRight = tG.x > map.x + map.w * 0.5;
    ctx.textAlign = gRight ? 'right' : 'left';
    ctx.fillText(gLbl, tG.x + (gRight ? -6 : 6), tG.y + (tG.y < map.y + 20 ? 12 : -8));
    ctx.textAlign = 'left';
    ctx.fillStyle = T.n14; ctx.beginPath(); ctx.arc(P0.x, P0.y, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = T.n0; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.restore();
    r.label('drag the black dot · the tick on ε_c is where w = 1 stops', map.x + 6, map.y + map.h - 8, { color: T.n10, font: '10px ui-monospace, monospace' });
    ctx.restore();

    // ---- diversity strip ---------------------------------------------------
    const sA = rec.A ? stats(rec.A, c) : null, sB = rec.B ? stats(rec.B, c) : null;
    page.probe = { modesB: sB ? sB.modes : null, adhB: sB ? sB.adherence : null, spreadA: sA ? sA.spread : null, spreadB: sB ? sB.spread : null, guidedSteps, passes };
    r.label('diversity — the same 12 seeds, twice', strip.x, strip.y - 8, { color: T.n11, font: '11px ui-monospace, monospace' });
    const rowY = [strip.y + 22, strip.y + 108];
    const swW = Math.min(34, (strip.w - 11 * 4) / NS), swH = 30;
    const drawRow = (yy, arr, s, title, tint) => {
      r.label(title, strip.x, yy - 6, { color: tint, font: '11px ui-monospace, monospace' });
      if (!arr) {
        for (let p = 0; p < NS; p++) { const bx = strip.x + p * (swW + 4); ctx.save(); ctx.strokeStyle = T.n5; ctx.setLineDash([3, 3]); ctx.strokeRect(bx, yy, swW, swH); ctx.restore(); }
        r.label('not run yet — press ▶ or scrub', strip.x, yy + swH + 14, { color: T.n10, font: '10px ui-monospace, monospace' });
        return;
      }
      for (let p = 0; p < NS; p++) {
        const x = arr[p * 2], y = arr[p * 2 + 1];
        const m = nearestMode(x, y), hue = categorical(MU[m].cls);
        const tau = Math.hypot(x - MU[m].x, y - MU[m].y) / SD;
        // saturation IS the "colour fidelity" axis: a typical sample (tau ~ 1)
        // reads soft, the class stereotype (tau -> 0) reads blaring.
        const sat = Math.max(0.12, Math.min(1, 1.15 - 0.42 * tau));
        const bx = strip.x + p * (swW + 4);
        ctx.save();
        ctx.fillStyle = mixColor(T.n0, hue, sat); ctx.fillRect(bx, yy, swW, swH);
        ctx.strokeStyle = tau > 2.5 ? T.bad : alphaOf(hue, 0.7); ctx.lineWidth = tau > 2.5 ? 2 : 1; ctx.strokeRect(bx, yy, swW, swH);
        ctx.fillStyle = sat > 0.55 ? inkOn(mixColor(T.n0, hue, sat)) : T.n12;
        ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(`${CLS[MU[m].cls]}${MU[m].sub + 1}`, bx + swW / 2, yy + swH / 2);
        ctx.restore();
      }
      if (s) r.label(`sub-modes ${s.modes}/2 · on-prompt ${Math.round(s.adherence * 100)}% · spread ${s.spread.toFixed(3)} · typicality ${s.typ.toFixed(2)}σ`, strip.x, yy + swH + 14, { color: T.n11, font: '10px ui-monospace, monospace' });
    };
    drawRow(rowY[0], rec.A, sA, `w = 1  — conditional only (the control)`, T.n11);
    drawRow(rowY[1], rec.B, sB, `w = ${st.w.toFixed(1)}  — guided on steps ${gs}…${ge}`, categorical(c));
    if (sA && sB) {
      const dv = pct(sB.spread, sA.spread);
      r.label(`spread(guided) = ${dv} of the control's  (lower = less diverse)`, strip.x, rowY[1] + swH + 34, { color: sB.spread < sA.spread * 0.6 ? T.bad : T.n12, font: '11px ui-monospace, monospace' });
      r.label(`saturation = how stereotypical the sample is`, strip.x, rowY[1] + swH + 50, { color: T.n10, font: '10px ui-monospace, monospace' });
    }

    // ---- timeline: the guidance interval + the per-step compute cost -------
    r.label('guidance interval', pad, tl.y + 14, { color: T.n11, font: '11px ui-monospace, monospace' });
    const bw = tl.w / TSTEP;
    ctx.save();
    for (let i = 0; i < TSTEP; i++) {
      const on = i >= gs && i < ge;
      const wi = on ? st.w : 1;
      const h = Math.max(2, Math.min(1, wi / 20) * tl.h);
      ctx.fillStyle = on ? alphaOf(T.violet, 0.8) : T.n5;
      ctx.fillRect(tl.x + i * bw + 1, tl.y + tl.h - h, bw - 2, h);
    }
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(tl.x, tl.y, tl.w, tl.h);
    for (const [s, lab] of [[gs, '['], [ge, ']']]) {
      const hx = tl.x + s * bw;
      ctx.strokeStyle = T.violet; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(hx, tl.y - 6); ctx.lineTo(hx, tl.y + tl.h + 6); ctx.stroke();
      ctx.fillStyle = T.violet; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
      ctx.fillText(lab, hx, tl.y - 9);
    }
    const mx = tl.x + rec.i * bw;
    ctx.strokeStyle = T.n14; ctx.lineWidth = 1.4; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(mx, tl.y - 4); ctx.lineTo(mx, tl.y + tl.h + 4); ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
    r.label('↔ drag either end', tl.x, tl.y + tl.h + 18, { color: T.accent, font: '10px ui-monospace, monospace' });
    r.label(`${passes} forward passes`, tl.x + tl.w + 10, tl.y + 10, { color: T.n14, font: '11px ui-monospace, monospace' });
    r.label(`${pct(passes, TSTEP)} of unguided`, tl.x + tl.w + 10, tl.y + 24, { color: passes > TSTEP ? T.warn : T.n11, font: '11px ui-monospace, monospace' });

    // ---- hover-to-inspect --------------------------------------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer; let tipTxt = null;
      const near = (q, rr) => Math.hypot(p.x - q.x, p.y - q.y) < rr;
      if (near(tG, 16)) tipTxt = `ε~  = ε_u + w·(ε_c − ε_u)\n     = ${fmtV(eu)} + ${wNow.toFixed(1)}·${fmtV(dl)}\n     = ${fmtV(eg)}   |ε~| = ${mag(eg).toFixed(3)}\n|ε~| is ${pct(mag(eg), mag(ec))} of |ε_c| — past the conditional answer,\nnot a filtered version of it.`;
      else if (near(tC, 16)) tipTxt = `ε_c — the prediction WITH the prompt\n${fmtV(ec)}   |ε_c| = ${mag(ec).toFixed(3)}\nw = 1 lands exactly here. Everything beyond is extrapolation.`;
      else if (near(tU, 16)) tipTxt = `ε_u — the prediction with the prompt DROPPED\n${fmtV(eu)}   |ε_u| = ${mag(eu).toFixed(3)}\nThis is the second forward pass guidance pays for.`;
      else if (near(P0, 14)) tipTxt = `probe point (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)})\nσ_t = ${rec.sig.toFixed(3)}\nΔ = ε_c − ε_u = ${fmtV(dl)}, |Δ| = ${mag(dl).toFixed(3)}\ndrag me — Δ is tiny far from the data and grows as σ_t falls`;
      else if (p.y >= tl.y - 12 && p.y <= tl.y + tl.h + 12 && p.x >= tl.x && p.x <= tl.x + tl.w) {
        const i = Math.max(0, Math.min(TSTEP - 1, Math.floor((p.x - tl.x) / bw))), on = i >= gs && i < ge;
        tipTxt = `step ${i}  σ_t = ${sigmaAt(i).toFixed(3)}\nw = ${(on ? st.w : 1).toFixed(1)}  ·  ${on ? '2 forward passes (cond + uncond)' : '1 forward pass (cond only)'}\nearly steps choose the MODE; late steps only polish it`;
      } else {
        for (let row = 0; row < 2 && !tipTxt; row++) {
          const arr = row === 0 ? rec.A : rec.B, s = row === 0 ? sA : sB; if (!arr) continue;
          if (p.y >= rowY[row] && p.y <= rowY[row] + swH) {
            const q = Math.floor((p.x - strip.x) / (swW + 4));
            if (q >= 0 && q < NS && p.x >= strip.x) {
              const x = arr[q * 2], y = arr[q * 2 + 1], m = nearestMode(x, y);
              const tau = Math.hypot(x - MU[m].x, y - MU[m].y) / SD;
              const other = row === 0 ? rec.B : rec.A, om = nearestMode(other[q * 2], other[q * 2 + 1]);
              tipTxt = `seed ${q} · ${row === 0 ? 'w = 1 control' : `w = ${st.w.toFixed(1)}`}\nlanded on ${CLS[MU[m].cls]}${MU[m].sub + 1} at (${x.toFixed(2)}, ${y.toFixed(2)})\ntypicality ${tau.toFixed(2)}σ  (≈1 = a typical sample, →0 = the stereotype)\nsame seed in the other row: ${CLS[MU[om].cls]}${MU[om].sub + 1}`;
            }
          }
        }
      }
      if (tipTxt) page.setTip(tipTxt);
    }

    // ---- readout -----------------------------------------------------------
    // Say WHY w is 1 here when the slider reads otherwise: this step is outside
    // the guided interval, so the probe is honestly showing unguided vectors.
    // Without the note it reads as the slider having been ignored.
    const wWhy = (wNow !== st.w) ? ` (step ${si} is OUTSIDE the guided interval [${gs}, ${ge}) — slider w=${st.w.toFixed(1)} does not apply here)` : '';
    let o = `ε~ = ε_u + w·(ε_c − ε_u)   at probe (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)}), σ_t=${rec.sig.toFixed(3)}, w=${wNow.toFixed(1)}${wWhy}:  `;
    o += `ε_u ${fmtV(eu)} + ${wNow.toFixed(1)}·Δ ${fmtV(dl)} = ${fmtV(eg)}   |ε~| = ${mag(eg).toFixed(3)} = ${pct(mag(eg), mag(ec))} of |ε_c|.\n`;
    if (sA && sB) {
      o += `strip @ ${rec.i}/${TSTEP}: control w=1 → sub-modes ${sA.modes}/2, on-prompt ${Math.round(sA.adherence * 100)}%, spread ${sA.spread.toFixed(3)}, typicality ${sA.typ.toFixed(2)}σ  |  `;
      o += `guided w=${st.w.toFixed(1)} → sub-modes ${sB.modes}/2, on-prompt ${Math.round(sB.adherence * 100)}%, spread ${sB.spread.toFixed(3)} (${pct(sB.spread, sA.spread)} of control), typicality ${sB.typ.toFixed(2)}σ.\n`;
    } else o += `strip: press ▶ (or scrub) to run the 24 denoising steps.\n`;
    o += `cost: guided on steps ${gs}…${ge} (${guidedSteps} of ${TSTEP}) → ${passes} forward passes = ${pct(passes, TSTEP)} of the 1-pass-per-step baseline.   λ=${st.leak.toFixed(2)}   tier:${r.name}\n`;
    o += `λ is how imperfect the conditional model is. At λ=0 it is exact and w>1 buys NO adherence — only stereotype; a real model is imperfect, which is the whole reason w>1 helps at all. Noising is the diffusion-noise page; the integrator and step count are the diffusion-sampler page.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__guidancePage = page;
  const q = Q;
  const t = page.controls._transport;
  const num = (k, f) => { if (q.has(k)) page.controls.set(k, f(q.get(k)), { rebuild: true, silent: true }); };
  num('w', parseFloat); num('leak', parseFloat);
  num('gStart', (v) => parseInt(v, 10)); num('gEnd', (v) => parseInt(v, 10)); num('seed', (v) => parseInt(v, 10));
  if (q.has('cls')) page.controls.set('cls', q.get('cls'), { rebuild: true, silent: true });
  // ?px,?py place the probe (the headless stand-in for dragging it, since
  // --screenshot has no pointer); math coords, roughly -2.5 … 2.5.
  if (q.has('px')) probe.x = parseFloat(q.get('px'));
  if (q.has('py')) probe.y = parseFloat(q.get('py'));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  // A ?theme= pin is a statement about THIS view, so keep it in the query the
  // deep-link sync rewrites -- otherwise the first control change drops it and a
  // copied link comes back on whatever theme the next reader happens to prefer.
  if (q.has('theme')) {
    page.controls.state.theme = themeMode();
    onThemeChange(() => { page.controls.state.theme = themeMode(); });
  }
  // Deterministic frame for capture: any explicit hook pauses autoplay so the
  // requested step is what gets drawn.
  if (q.has('step') || q.has('hover') || q.has('px') || q.has('py')) { if (t) t.pause(); }
  if (t) t.rebuild();
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
