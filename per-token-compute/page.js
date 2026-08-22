// per-token-compute concept page -- not every token needs the same amount of
// work. A standard transformer spends identical compute on every token, however
// trivial: N tokens x L blocks, a perfect rectangle. Four shipped ways to break
// that rectangle are drawn here as modes of ONE widget -- Mixture-of-Depths,
// zero-computation experts, Mixture-of-Recursions, and early exit -- and all
// four produce the same picture: a RAGGED compute profile over the sequence.
//
// The honest half is the second counter. A ragged shape is a scheduling problem:
// the machine wants dense uniform work, so a batch whose tokens take different
// paths either pads up to a tile boundary or pays gather/scatter to compact
// itself. So the page reports the FLOP saving and the machine cost side by side,
// both as a percent of the same named baseline (the uniform-depth model), and
// there are settings where the ragged shape costs more than it saves.
//
// Which EXPERT a token picks is the moe-routing page. This one is only about how
// MUCH compute a token receives.
import { mount } from '../framework/layout.js';
import { seededRandn, seededRand } from '../framework/tensor.js';
import { T, alphaOf } from '../framework/theme.js';

// Router / exit-predictor overhead, charged on EVERY token at EVERY block even
// when the token is skipped: the decision itself is not free. Expressed as a
// fraction of one token-block of real work.
const ROUTER_COST = 0.02;

const MODES = {
  mod:  { label: 'Mixture-of-Depths',        hue: 'accent',   capName: 'block capacity (fraction of the sequence each block admits)' },
  zero: { label: 'zero-computation experts', hue: 'violet',   capName: 'real-expert share (how easily a token wins a computing expert)' },
  mor:  { label: 'Mixture-of-Recursions',    hue: 'tealDeep', capName: 'recursion budget (mean share of the loop count granted)' },
  exit: { label: 'early exit',               hue: 'goldDeep', capName: 'confidence threshold (higher = tokens must be surer to stop)' },
};

let cur = null;              // last computed model, captured in draw for hover
let diffOver = {};           // token index -> forced difficulty (canvas drag / ?dif=)
let rects = {};              // hit rects captured in draw
let grab = null;             // which draggable the pointer took

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// ---------------------------------------------------------------------------
// The model. difficulty -> per-(token, block) processed mask -> costs.
// ---------------------------------------------------------------------------

/** Per-token difficulty in (0,1]. `spread` 0 = every token identical, 1 = a few
 *  hard tokens and a long tail of trivial ones (punctuation, forced
 *  continuations). Seeded, so a reload shows the same sequence. */
function difficulty(st) {
  const N = st.N | 0, s = st.spread;
  const u = seededRandn((st.seed | 0) + 11, [N], { std: 1 });
  const d = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const p = 1 / (1 + Math.exp(-u[i]));            // (0,1)
    const skew = Math.pow(p, 1 + 2 * s);            // mass pushed toward "easy" as spread grows
    d[i] = clamp((1 - s) * 0.5 + s * skew, 0.02, 1);
    if (diffOver[i] != null) d[i] = clamp(diffOver[i], 0.02, 1);
  }
  return d;
}

/** Build the processed mask and every derived cost for the current settings. */
function compute(st) {
  const N = st.N | 0, L = st.L | 0, c = st.cap, mode = st.mode;
  const d = difficulty(st);
  const j = seededRand((st.seed | 0) + 29, [N * L], { lo: -0.12, hi: 0.12 });
  const proc = new Uint8Array(N * L);
  const conf = new Float32Array(N * L);            // early-exit confidence trace

  if (mode === 'mod') {
    // A router picks a CAPACITY-LIMITED subset per block; the rest skip the
    // block entirely via the residual. The count per block is fixed by
    // construction -- that is the whole scheduling virtue of MoD.
    const K = Math.max(1, Math.round(c * N));
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let l = 0; l < L; l++) {
      const rank = idx.slice().sort((a, b) => (d[b] + j[b * L + l]) - (d[a] + j[a * L + l])).slice(0, K);
      for (const i of rank) proc[i * L + l] = 1;
    }
  } else if (mode === 'zero') {
    // Some "experts" are the identity, so routing to one costs nothing. Each
    // token decides for itself -- nothing caps the per-block count, so the batch
    // width wanders block to block.
    for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) if (d[i] + j[i * L + l] > 1 - c) proc[i * L + l] = 1;
  } else if (mode === 'mor') {
    // Shared layers applied a variable number of times: a hard token loops more.
    // Contiguous from the bottom, so the profile is a clean skyline.
    for (let i = 0; i < N; i++) {
      const r = clamp(Math.round(L * c * (0.45 + 1.1 * d[i])), 1, L);
      for (let l = 0; l < r; l++) proc[i * L + l] = 1;
    }
  } else {
    // Early exit: a token stops once a confidence threshold is met.
    const tau = 0.30 + 0.65 * c;
    for (let i = 0; i < N; i++) {
      for (let l = 0; l < L; l++) {
        proc[i * L + l] = 1;
        const p = 1 - Math.exp(-(l + 1) * 0.55 * (1.25 - d[i]));
        conf[i * L + l] = p;
        if (p >= tau) break;
      }
    }
  }

  const active = new Int32Array(L), cnt = new Int32Array(N), exitD = new Int32Array(N).fill(-1);
  for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) if (proc[i * L + l]) { active[l]++; cnt[i]++; }
  for (let i = 0; i < N; i++) for (let l = L - 1; l >= 0; l--) if (proc[i * L + l]) { exitD[i] = l; break; }

  // ---- what the machine actually runs -------------------------------------
  const tile = Math.max(1, st.tile | 0), gs = st.gs, router = ROUTER_COST * N;
  const exec = new Float64Array(L), waste = new Float64Array(L);
  for (let l = 0; l < L; l++) {
    exec[l] = st.sched === 'pad'
      ? (active[l] ? Math.ceil(active[l] / tile) * tile : 0)     // pad up to a tile boundary
      : (active[l] ? active[l] + gs * N : 0);                    // compact, and pay to move the rows
    waste[l] = exec[l] - active[l];
  }
  let useful = 0, machine = 0;
  for (let l = 0; l < L; l++) { useful += active[l]; machine += exec[l] + router; }
  const baseline = N * L;                                        // uniform-depth model: dense, no router

  return {
    N, L, d, proc, conf, active, cnt, exitD, exec, waste, router, tile, gs,
    useful, machine, baseline,
    flopPct: (useful / baseline) * 100,
    machinePct: (machine / baseline) * 100,
    util: machine > 0 ? (useful / machine) * 100 : 100,
  };
}

// ---------------------------------------------------------------------------

mount({
  mount: 'body',
  title: 'per-token-compute — how much work a token gets',
  blurb: 'A standard transformer spends identical compute on every token, however trivial: N tokens × L blocks, one uniform rectangle. Four shipped ways to break that rectangle — Mixture-of-Depths (a router admits a capacity-limited subset per block, the rest skip it via the residual), zero-computation experts (some experts are the identity, so routing to one costs nothing), Mixture-of-Recursions (shared layers applied a variable number of times, so a hard token loops more), and early exit (a token stops once a confidence threshold is met). All four draw the same picture: a ragged compute profile, tall over hard tokens and short over trivial ones. The honest half is the second counter — a ragged shape is a scheduling problem, because the machine wants dense uniform work, so the batch either pads up to a tile boundary or pays gather/scatter to compact itself. Both counters are a percent of the same baseline: the uniform-depth model. Drag a token’s difficulty bar, drag the capacity and spread knobs, switch mechanism, hover any token for its path — and find a setting where the ragged shape costs MORE than it saves.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.select('mode', { label: 'mechanism', options: [
      { value: 'mod', label: 'Mixture-of-Depths' },
      { value: 'zero', label: 'zero-computation experts' },
      { value: 'mor', label: 'Mixture-of-Recursions' },
      { value: 'exit', label: 'early exit' },
    ], value: 'mod' });
    c.stepper('N', { label: 'tokens (N)', min: 6, max: 28, value: 16 });
    c.stepper('L', { label: 'blocks (L)', min: 4, max: 16, value: 10 });
    c.slider('cap', { label: 'capacity fraction', min: 0.05, max: 1, step: 0.05, value: 0.5 });
    c.slider('spread', { label: 'difficulty spread', min: 0, max: 1, step: 0.05, value: 0.7 });
    c.select('sched', { label: 'batch schedule', options: [
      { value: 'pad', label: 'pad to tile' },
      { value: 'gather', label: 'gather / scatter' },
    ], value: 'pad' });
    c.stepper('tile', { label: 'tile rows', min: 1, max: 16, value: 8, rebuild: false });
    c.slider('gs', { label: 'gather cost / token', min: 0, max: 0.4, step: 0.01, value: 0.12, rebuild: false });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.transport({ compute: () => Array.from({ length: page.state.L | 0 }, (_, l) => ({ l, label: `block ${l} of ${(page.state.L | 0) - 1}` })), speed: 1.7, loop: true });
  },

  onPointer: (page, ev) => {
    const st = page.state, N = st.N | 0;
    if (ev.type === 'down') {
      grab = null;
      const inRect = (r) => r && ev.x >= r.x - 6 && ev.x <= r.x + r.w + 6 && ev.y >= r.y - 8 && ev.y <= r.y + r.h + 8;
      if (inRect(rects.cap)) grab = 'cap';
      else if (inRect(rects.spread)) grab = 'spread';
      else if (inRect(rects.diff)) grab = 'diff';
    } else if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!grab || !page.pointer.down) return;
    if (grab === 'cap') {
      const r = rects.cap, f = clamp((ev.x - r.x) / r.w, 0, 1);
      page.controls.set('cap', +(0.05 + f * 0.95).toFixed(2));
    } else if (grab === 'spread') {
      const r = rects.spread, f = clamp((ev.x - r.x) / r.w, 0, 1);
      page.controls.set('spread', +f.toFixed(2));
    } else if (grab === 'diff') {
      const r = rects.diff, i = clamp(Math.floor((ev.x - r.x) / (r.w / N)), 0, N - 1);
      diffOver[i] = clamp((r.y + r.h - ev.y) / r.h, 0.02, 1);
      page.redraw();
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const N = st.N | 0, L = st.L | 0, mode = st.mode, M = MODES[mode] || MODES.mod;
    r.clear(T.n0);
    const C = compute(st);
    cur = C;
    const hue = T[M.hue];

    const s = page.step();
    const upto = s ? s.l : L - 1;            // depth reached by the transport

    // ---- geometry ---------------------------------------------------------
    const pad = 14, y0 = 52;
    const gx = pad + 34;
    const gw = Math.max(140, page.W * 0.50);
    const strip = 26;
    const gy = y0 + strip + 18;
    const bottom = Math.max(gy + 60, page.H - 68);
    const gh = bottom - gy - 14;
    const cw = gw / N, ch = gh / L;
    const rowY = (l) => gy + (L - 1 - l) * ch;
    rects.diff = { x: gx, y: y0, w: gw, h: strip };
    rects.grid = { x: gx, y: gy, w: gw, h: gh };
    rects.cap = { x: gx, y: bottom + 26, w: gw * 0.44, h: 12 };
    rects.spread = { x: gx + gw * 0.56, y: bottom + 26, w: gw * 0.44, h: 12 };

    const barsX = gx + gw + 48;
    const barsW = Math.max(60, page.W - pad - barsX);

    ctx.save();
    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'alphabetic';

    // ---- per-token difficulty strip (draggable) ---------------------------
    r.label('difficulty per token — drag a bar', gx, y0 - 8, { color: T.n11, font: '11px ui-monospace, monospace' });
    for (let i = 0; i < N; i++) {
      const h = Math.max(1, C.d[i] * strip), x = gx + i * cw;
      ctx.fillStyle = alphaOf(M.hue, 0.20 + 0.55 * C.d[i]);
      ctx.fillRect(x + 0.8, y0 + strip - h, Math.max(1, cw - 1.6), h);
      if (diffOver[i] != null) { ctx.strokeStyle = T.warnDeep; ctx.lineWidth = 1.2; ctx.strokeRect(x + 0.8, y0 + strip - h, Math.max(1, cw - 1.6), h); }
    }
    ctx.strokeStyle = alphaOf('n14', 0.20); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(gx, y0 + strip + 0.5); ctx.lineTo(gx + gw, y0 + strip + 0.5); ctx.stroke();

    // ---- the compute profile: tokens x depth ------------------------------
    r.label('depth', pad - 2, gy - 6, { color: T.n11, font: '11px ui-monospace, monospace' });
    for (let l = 0; l < L; l++) {
      const y = rowY(l);
      const reached = l <= upto;
      for (let i = 0; i < N; i++) {
        const x = gx + i * cw, on = C.proc[i * L + l];
        if (on && reached) {
          ctx.fillStyle = alphaOf(M.hue, 0.35 + 0.5 * C.d[i]);
          ctx.fillRect(x + 0.7, y + 0.7, Math.max(1, cw - 1.4), Math.max(1, ch - 1.4));
        } else if (on) {
          ctx.fillStyle = alphaOf('n14', 0.07);                       // not yet reached
          ctx.fillRect(x + 0.7, y + 0.7, Math.max(1, cw - 1.4), Math.max(1, ch - 1.4));
        } else {
          ctx.fillStyle = alphaOf('n14', 0.035);                      // skipped via the residual
          ctx.fillRect(x + 0.7, y + 0.7, Math.max(1, cw - 1.4), Math.max(1, ch - 1.4));
        }
      }
      // depth label, every row when there is room
      if (ch >= 12) r.label(String(l), gx - 6, y + ch / 2 + 3, { color: l === upto ? T.n14 : T.n9, align: 'right', font: '10px ui-monospace, monospace' });
    }
    ctx.strokeStyle = alphaOf('n14', 0.12); ctx.lineWidth = 1;
    ctx.strokeRect(gx + 0.5, gy + 0.5, gw, gh);

    // the ragged skyline: the top of each token's own compute path
    ctx.strokeStyle = hue; ctx.lineWidth = 1.6; ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const e = C.exitD[i], y = e < 0 ? gy + gh : rowY(e);
      const x0 = gx + i * cw, x1 = x0 + cw;
      if (i === 0) ctx.moveTo(x0, y); else ctx.lineTo(x0, y);
      ctx.lineTo(x1, y);
    }
    ctx.stroke();

    // tokens that have finished by the current transport depth drop out
    for (let i = 0; i < N; i++) {
      if (C.exitD[i] < upto) {
        const x = gx + i * cw;
        ctx.fillStyle = alphaOf('n14', 0.05);
        ctx.fillRect(x + 0.7, gy, Math.max(1, cw - 1.4), rowY(C.exitD[i]) - gy);
        if (cw >= 9) r.label('✓', x + cw / 2, rowY(C.exitD[i]) - 3, { color: T.ok, align: 'center', font: '9px ui-monospace, monospace' });
      }
    }
    // current block marker
    ctx.strokeStyle = T.n14; ctx.lineWidth = 1.4; ctx.setLineDash([3, 2]);
    ctx.strokeRect(gx - 1.5, rowY(upto) - 1.5, gw + 3, ch + 3);
    ctx.setLineDash([]);
    r.label('tokens →', gx, gy + gh + 12, { color: T.n11, font: '11px ui-monospace, monospace' });

    // ---- per-block machine width -----------------------------------------
    r.label('what the machine runs, per block', barsX, gy - 22, { color: T.n11, font: '11px ui-monospace, monospace' });
    let axis = N;
    for (let l = 0; l < L; l++) axis = Math.max(axis, C.exec[l] + C.router);
    const sc = barsW / Math.max(1, axis);
    for (let l = 0; l < L; l++) {
      const y = rowY(l), h = Math.max(2, ch - 2.4);
      const uW = C.active[l] * sc, wW = C.waste[l] * sc, rW = C.router * sc;
      ctx.globalAlpha = l <= upto ? 1 : 0.32;
      ctx.fillStyle = hue; ctx.fillRect(barsX, y + 1.2, uW, h);
      if (wW > 0) { ctx.fillStyle = alphaOf('bad', 0.55); ctx.fillRect(barsX + uW, y + 1.2, wW, h); }
      ctx.fillStyle = alphaOf('n14', 0.30); ctx.fillRect(barsX + uW + wW, y + 1.2, rW, h);
      ctx.globalAlpha = 1;
    }
    // the uniform-depth baseline width
    ctx.strokeStyle = T.n9; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(barsX + N * sc, gy - 2); ctx.lineTo(barsX + N * sc, gy + gh + 2); ctx.stroke();
    ctx.setLineDash([]);
    r.label('dense N', barsX + N * sc - 3, gy - 5, { color: T.n9, align: 'right', font: '10px ui-monospace, monospace' });

    // legend
    const lg = [[hue, 'useful'], [alphaOf('bad', 0.55), st.sched === 'pad' ? 'padding' : 'gather'], [alphaOf('n14', 0.30), 'router']];
    let lx = barsX;
    for (const [col, txt] of lg) {
      ctx.fillStyle = col; ctx.fillRect(lx, gy - 12, 8, 8);
      r.label(txt, lx + 11, gy - 5, { color: T.n11, font: '10px ui-monospace, monospace' });
      lx += 12 + ctx.measureText(txt).width + 14;
    }

    // ---- drag knobs: capacity + difficulty spread -------------------------
    const knob = (rect, frac, label, val) => {
      ctx.fillStyle = alphaOf('n14', 0.08); ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = alphaOf(M.hue, 0.45); ctx.fillRect(rect.x, rect.y, rect.w * frac, rect.h);
      const kx = rect.x + rect.w * frac;
      ctx.fillStyle = hue; ctx.fillRect(kx - 2.5, rect.y - 3, 5, rect.h + 6);
      r.label(`${label} ${val}  ⇔ drag`, rect.x, rect.y - 5, { color: T.n11, font: '10px ui-monospace, monospace' });
    };
    knob(rects.cap, (st.cap - 0.05) / 0.95, 'capacity', st.cap.toFixed(2));
    knob(rects.spread, st.spread, 'spread', (+st.spread).toFixed(2));

    // ---- the two counters, against the uniform-depth baseline -------------
    const cX = barsX, cW = barsW, cY = bottom + 4;
    const meter = (y, pct, col, label) => {
      const track = cW, full = track * 0.75;                 // 100% of baseline sits at 75% of the track
      ctx.fillStyle = alphaOf('n14', 0.08); ctx.fillRect(cX, y, track, 9);
      ctx.fillStyle = col; ctx.fillRect(cX, y, Math.min(track, full * pct / 100), 9);
      ctx.strokeStyle = T.n9; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cX + full, y - 2); ctx.lineTo(cX + full, y + 11); ctx.stroke();
      r.label(`${label} ${pct.toFixed(1)}%`, cX, y - 3, { color: T.n12, font: '10px ui-monospace, monospace' });
    };
    r.label('vs the uniform-depth model (tick = 100%)', cX, cY, { color: T.n11, font: '10px ui-monospace, monospace' });
    meter(cY + 15, C.flopPct, hue, 'FLOPs');
    meter(cY + 36, C.machinePct, C.machinePct > 100 ? T.bad : T.ok, 'machine');

    ctx.restore();

    // ---- hover ------------------------------------------------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const inG = p.x >= gx && p.x <= gx + gw && p.y >= y0 && p.y <= gy + gh;
      if (inG) {
        const i = clamp(Math.floor((p.x - gx) / cw), 0, N - 1);
        const path = [];
        for (let l = 0; l < L; l++) path.push(C.proc[i * L + l] ? '█' : '·');
        const share = C.useful > 0 ? (C.cnt[i] / C.useful) * 100 : 0;
        let extra = '';
        if (mode === 'exit' && C.exitD[i] >= 0) extra = `\nconfidence at exit ${C.conf[i * L + C.exitD[i]].toFixed(3)} (threshold ${(0.30 + 0.65 * st.cap).toFixed(3)})`;
        page.setTip(
          `token ${i}   difficulty ${C.d[i].toFixed(2)}${diffOver[i] != null ? ' (dragged)' : ''}\n` +
          `path depth 0→${L - 1}:  ${path.join('')}\n` +
          `blocks computed ${C.cnt[i]} / ${L}   exit depth ${C.exitD[i] < 0 ? '—' : C.exitD[i]}\n` +
          `${share.toFixed(1)}% of the batch's useful compute\ndrag its bar to re-score it`);
      } else if (p.x >= barsX && p.x <= barsX + barsW && p.y >= gy && p.y <= gy + gh) {
        const l = clamp(L - 1 - Math.floor((p.y - gy) / ch), 0, L - 1);
        page.setTip(
          `block ${l}\nactive ${C.active[l]} / ${N} tokens\n` +
          `machine runs ${(C.exec[l] + C.router).toFixed(2)} token-slots\n` +
          `${st.sched === 'pad' ? `padding ${C.waste[l].toFixed(2)} (tile ${C.tile})` : `gather/scatter ${C.waste[l].toFixed(2)}`} + router ${C.router.toFixed(2)}`);
      }
    }

    // ---- readout ----------------------------------------------------------
    const dir = C.machinePct > 100 ? 'NET LOSS — the ragged shape costs MORE than it saves'
      : C.machinePct > C.flopPct + 0.05 ? 'real saving, partly eaten by the schedule'
        : 'real saving';
    let o = `${M.label}: ${M.capName} = ${(+st.cap).toFixed(2)}; ${N} tokens x ${L} blocks; `;
    o += st.sched === 'pad' ? `padded to ${C.tile}-row tiles` : `gather/scatter at ${(+st.gs).toFixed(2)} token-slots per token per block`;
    o += `; router ${ROUTER_COST.toFixed(2)}/token/block on all ${N}.    tier:${r.name}\n`;
    o += `useful FLOPs ${C.useful.toFixed(0)} of ${C.baseline} token-blocks = ${C.flopPct.toFixed(1)}% of the uniform-depth model (lower is better; 100% = every token through every block).  `;
    o += `machine time ${C.machine.toFixed(1)} token-slots = ${C.machinePct.toFixed(1)}% of the same baseline (lower is better; >100% = worse than uniform).  `;
    o += `utilisation ${C.util.toFixed(1)}% (useful / machine).\n`;
    o += `${dir}.  block ${upto} of ${L - 1}: ${C.active[upto]} / ${N} tokens active, ${(C.exec[upto] + C.router).toFixed(2)} token-slots run.  `;
    const live = Array.from(C.exitD).filter((e) => e >= 0);
    const never = N - live.length;
    o += live.length ? `exit depths ${Math.min.apply(null, live)}–${Math.max.apply(null, live)}` : 'no token was computed anywhere';
    o += never ? `, ${never} token(s) never computed at all.` : '.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__ptcPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Deep-link restore. The framework MIRRORS control state into the query string
  // but does not read it back -- each page restores its own keys here, before
  // anything triggers the first sync (which would otherwise overwrite the link
  // with the defaults the controls were built with).
  const NUM = { N: 1, L: 1, cap: 1, spread: 1, tile: 1, gs: 1, seed: 1 };
  const REBUILD = { N: 1, L: 1, seed: 1 };
  for (const k of ['mode', 'sched', 'N', 'L', 'cap', 'spread', 'tile', 'gs', 'seed']) {
    if (!q.has(k)) continue;
    const raw = q.get(k);
    const v = NUM[k] ? parseFloat(raw) : raw;
    if (NUM[k] && !Number.isFinite(v)) continue;
    page.controls.set(k, v, { rebuild: !!REBUILD[k], silent: true });
  }
  if (t) t.rebuild();
  // ?dif=3:0.95,7:0.05 -- forced per-token difficulty, the headless stand-in for
  // dragging a token's bar (the same state a drag writes).
  if (q.has('dif')) {
    for (const part of q.get('dif').split(',')) {
      const [i, v] = part.split(':').map(Number);
      if (Number.isFinite(i) && Number.isFinite(v)) diffOver[i] = v;
    }
  }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('dif') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
