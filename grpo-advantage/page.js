// grpo-advantage -- the group IS the baseline, and a group that agrees teaches
// nothing.
//
// One prompt, G sampled rollouts, one verifiable reward each. The whole page is
// this arithmetic, computed here, never transcribed:
//
//   mu    = (1/G) * SUM r_i                       the group mean = the baseline
//   sigma = sqrt( (1/G) * SUM (r_i - mu)^2 )      spread within the group
//   A_i   = (r_i - mu)            [ / (sigma + 1e-4)  when normalisation is on ]
//
// There is no learned value network anywhere in that. The baseline is the other
// samples of the same prompt, which is the whole trick: it costs G forward
// passes and zero extra parameters.
//
// The clipped objective then bounds how far one step may move the policy. With
// rho_i the ratio between the new and old policy on rollout i, the per-rollout
// term is  min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i). When the clipped
// branch wins, that term is CONSTANT in rho_i -- so it has no gradient. That
// happens exactly when the step would push further in the direction the
// advantage already likes:
//
//   A_i > 0 and rho_i > 1+eps   ->   clamped, contributes nothing
//   A_i < 0 and rho_i < 1-eps   ->   clamped, contributes nothing
//
// THE FAILURE THIS PAGE IS BUILT AROUND: if every rollout in the group earns the
// SAME reward -- all correct, or all wrong -- then every r_i equals mu, so every
// A_i is EXACTLY zero and the group contributes no gradient at all. Dividing by
// sigma does not rescue it (0 / (0 + 1e-4) is still 0). Drag the rewards until
// the group agrees and the gradient signal reads 0.000. That is why dynamic
// sampling -- filtering out the all-correct and all-wrong groups and sampling
// replacements until the batch is full of groups that disagree -- exists.
//
// WHAT THIS PAGE DOES NOT SHOW, deliberately: no policy update, no weights, no
// training curve. A weight update over a transformer is not a thing a reader can
// watch, and animating one would be a fiction. What IS live math is the quantity
// the update is built from, so that is the whole scope.
//
// Sources for the mechanism (public):
//   Shao et al., "DeepSeekMath" (GRPO), arXiv:2402.03300
//   Liu et al., "Understanding R1-Zero-Like Training" (Dr. GRPO) --
//     the std-normalisation and response-length biases, arXiv:2503.20783
import { mount } from '../framework/layout.js';
import { seededRandn, rng } from '../framework/tensor.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

const MAXG = 16;
const STD_EPS = 1e-4;              // the +eps in A = (r - mu) / (sigma + eps)
const f2 = (x) => x.toFixed(2);
const f3 = (x) => x.toFixed(3);
const pct = (x) => (x * 100).toFixed(1) + '%';

// ------------------------------------------------------------------ the group
// A rollout is a sampled answer with a verifiable reward. Deterministic in the
// seed, so one URL replays one exact picture.
let base = { key: '', rew: [], rho: [], toks: [] };
let over = {};                 // index -> reward set by a drag / click / URL
let edits = 0;                 // bumped whenever `over` moves, so the memo misses
let geom = null;               // rects captured in draw(), read by onPointer
let grab = null;               // {row:i} while dragging a reward

function buildGroup(seed, G, pass, spread, drift) {
  const u = rng(seed * 6997 + 11);
  const z = Array.from(seededRandn(seed * 3121 + 7, MAXG, { std: 1 }));
  const rew = [], rho = [], toks = [];
  for (let i = 0; i < G; i++) {
    const solved = u() < pass;                 // pass = 1 -> every rollout correct
    const jitter = u();                        // partial credit / style penalty
    // spread = 0 makes the reward exactly 1 or 0, which is what a pure
    // pass/fail verifier gives and what makes an exact 0.000 reachable.
    rew.push(solved ? 1 - spread * jitter : spread * jitter);
    rho.push(Math.exp(drift * z[i]));          // policy ratio for this rollout
    toks.push(3 + Math.floor(u() * 5));        // a sketch of the answer's length
  }
  return { rew, rho, toks };
}

function ensure(st) {
  const key = `${st.seed | 0}|${st.g | 0}|${st.pass}|${st.spread}|${st.drift}`;
  if (base.key !== key) {
    base = { key, ...buildGroup(st.seed | 0, st.g | 0, st.pass, st.spread, st.drift) };
    over = {}; edits++;                        // a regenerated group drops the edits
  }
  return base;
}

/** Everything the page shows, from the rewards of the SCORED rollouts. */
function model(rew, rho, S, norm, eps) {
  const r = rew.slice(0, S);
  const mu = r.reduce((a, b) => a + b, 0) / (S || 1);
  const varc = r.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (S || 1);
  const sigma = Math.sqrt(varc);
  const scale = norm ? 1 / (sigma + STD_EPS) : 1;
  const rows = r.map((ri, i) => {
    const raw = ri - mu;
    const A = raw * scale;
    const p = rho[i];
    // The clipped branch wins exactly when the step pushes further in the
    // direction the advantage already likes -- and then the term is constant.
    const clamped = (A > 0 && p > 1 + eps) || (A < 0 && p < 1 - eps);
    return { i, r: ri, raw, A, rho: p, clamped };
  });
  const live = rows.filter((x) => !x.clamped);
  // "gradient signal": mean |A| over the rollouts still able to move the policy.
  const signal = live.length ? live.reduce((a, x) => a + Math.abs(x.A), 0) / live.length : 0;
  const clipFrac = S ? (S - live.length) / S : 0;
  return { mu, sigma, scale, rows, signal, clipFrac, S, nLive: live.length };
}

// ------------------------------------------------------------------- drawing
function hatch(ctx, x, y, w, h) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.strokeStyle = rgbaToken('n14', 0.28); ctx.lineWidth = 1;
  for (let d = -h; d < w + h; d += 5) {
    ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke();
  }
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'group-relative advantage — the group is the baseline',
  blurb: 'GRPO scores a whole GROUP of rollouts for one prompt with a verifiable reward, then sets each rollout\'s advantage to its reward minus the GROUP MEAN. No learned value network: the other samples are the baseline. Drag any reward bar and every advantage, the mean, the clipped fraction and the gradient signal recompute under your hand. Drag them until the group AGREES — all correct, or all wrong — and the signal reads exactly 0.000, because every reward then equals the mean. Mechanism: DeepSeekMath / GRPO, arXiv:2402.03300; the std-normalisation and length biases, Dr. GRPO, arXiv:2503.20783. No policy update, no weights and no training curve are shown — see the note under the chart.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  compare: {
    key: 'pass', a: 0.5, b: 1,
    labelA: 'a group that disagrees — real advantages, real gradient',
    labelB: 'every rollout correct — every advantage exactly 0, no gradient',
  },
  challenges: [
    {
      goal: 'Make the gradient signal exactly 0.000 — a group that agrees teaches nothing.',
      hint: 'click each ✓/✗ tag (or drag every reward bar) until all rollouts carry the same reward. Setting the solved fraction to 1 with partial-credit spread at 0 does it in two moves.',
      check: (api) => {
        const s = api.probe.signal ?? 1;
        return { solved: s === 0, detail: s === 0 ? 'signal = 0.000 — this group contributes nothing' : `signal = ${f3(s)} — the group still disagrees` };
      },
    },
    {
      goal: 'Clip at least half the group while the surviving rollouts still carry a signal above 0.15.',
      hint: 'raise the policy drift so the ratios scatter, then narrow the clip window. Clipping removes rollouts from the sum; it does not shrink the ones that remain.',
      check: (api) => {
        const c = api.probe.clipFrac ?? 0, s = api.probe.signal ?? 0;
        return { solved: c >= 0.5 && s > 0.15, detail: `${pct(c)} clipped · signal ${f3(s)}` };
      },
    },
  ],

  controls: (c) => {
    c.stepper('g', { label: 'group size G — rollouts sampled for this one prompt', min: 2, max: MAXG, value: 8 });
    c.slider('pass', { label: 'fraction of rollouts the verifier marks correct', min: 0, max: 1, step: 0.05, value: 0.5, format: (v) => f2(v) });
    c.slider('spread', { label: 'partial-credit spread inside each verdict (0 = pure pass/fail)', min: 0, max: 0.4, step: 0.01, value: 0.12, format: (v) => f2(v) });
    c.toggle('norm', { label: 'divide by the group std  (A = (r − μ) / (σ + 1e−4))', value: false });
    c.slider('eps', { label: 'clip window ε  — the trust region on the policy ratio', min: 0.02, max: 0.6, step: 0.01, value: 0.2, format: (v) => f2(v) });
    c.slider('drift', { label: 'policy drift — how far the ratios ρ scatter from 1', min: 0, max: 0.6, step: 0.01, value: 0.18, format: (v) => f2(v) });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 12 });
    // One step per rollout: the group is scored one sample at a time, and the
    // baseline moves as each arrives. Read G off the live control state, not off
    // a page global -- compute() runs on rebuild, before the next draw().
    c.transport({
      compute: () => {
        const G = Math.max(2, Math.min(MAXG, c.state.g | 0));
        return Array.from({ length: G }, (_, i) => ({ n: i + 1, label: `${i + 1} of ${G} rollouts scored` }));
      },
      speed: 1.7, loop: true,
    });
  },

  // Direct manipulation: drag a reward bar ↔, or click a rollout's ✓/✗ tag to
  // flip it between reward 1 and reward 0.
  onPointer: (page, ev) => {
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!geom) return;
    const { rows, rx, rw, labX, labW, rowH, S } = geom;
    const rowAt = (y) => {
      const j = Math.floor((y - rows.y) / rowH);
      return j >= 0 && j < S ? j : -1;
    };
    if (ev.type === 'down') {
      grab = null;
      const j = rowAt(ev.y);
      if (j < 0) return;
      if (ev.x >= labX && ev.x <= labX + labW) {
        const cur = over[j] != null ? over[j] : base.rew[j];
        over[j] = cur >= 0.5 ? 0 : 1;             // flip the verdict outright
        edits++; page.redraw(); return;
      }
      if (ev.x >= rx - 8 && ev.x <= rx + rw + 8) grab = { row: j };
    }
    if (grab && page.pointer.down) {
      over[grab.row] = Math.max(0, Math.min(1, (ev.x - rx) / rw));
      edits++; page.redraw();
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);

    const G = Math.max(2, Math.min(MAXG, st.g | 0));
    ensure(st);
    const rew = base.rew.slice(0, G).map((v, i) => (over[i] != null ? over[i] : v));
    const rho = base.rho.slice(0, G);

    const s = page.step();
    const S = Math.max(1, Math.min(G, s ? s.n : G));
    const m = model(rew, rho, S, !!st.norm, st.eps);
    page.probe = { signal: m.signal, clipFrac: m.clipFrac, sigma: m.sigma, mu: m.mu, S };

    // ------------------------------------------------------------- layout
    const pad = 16;
    const headY = 40;
    const bottomH = 118;
    const top = headY + 30;
    const areaH = Math.max(40, page.H - top - bottomH);
    const rowH = Math.min(30, areaH / G);
    const rowsRect = { y: top, h: rowH * G };

    // gapRV is wider than the others on purpose: each reward bar prints its own
    // value just past its right edge, and that number lives in this gap.
    const labX = pad, labW = 92, gapL = 12, gapRV = 46, gapAC = 14;
    const avail = Math.max(120, page.W - pad * 2 - labW - gapL - gapRV - gapAC);
    const rw = avail * 0.40, aw = avail * 0.34, cw = avail * 0.26;
    const rx = labX + labW + gapL;
    const ax = rx + rw + gapRV;
    const cx = ax + aw + gapAC;
    const acx = ax + aw / 2;                        // advantage zero axis
    geom = { rows: rowsRect, rx, rw, ax, aw, cx, cw, labX, labW, rowH, S, acx };

    // ------------------------------------------------------ column headings
    r.label('rollout', labX, headY, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label('verifiable reward  r', rx, headY, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label(st.norm ? 'advantage  A = (r−μ)/(σ+1e−4)' : 'advantage  A = r − μ',
      ax, headY, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label('ratio ρ vs clip window', cx, headY, { color: T.n11, font: '11px ui-monospace, monospace' });

    // reward axis ticks
    ctx.save();
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1;
    for (const t of [0, 0.5, 1]) {
      const x = rx + t * rw;
      ctx.beginPath(); ctx.moveTo(x, rowsRect.y - 4); ctx.lineTo(x, rowsRect.y + rowsRect.h + 4); ctx.stroke();
    }
    ctx.restore();
    for (const t of [0, 0.5, 1]) {
      r.label(f2(t), rx + t * rw, rowsRect.y + rowsRect.h + 17, { color: T.n9, font: '9px ui-monospace, monospace', align: 'center' });
    }

    // ------------------------------------------------------- the clip window
    const rhoDom = Math.max(0.32, st.eps * 1.7, ...rho.map((p) => Math.abs(p - 1) * 1.18));
    const cxFor = (p) => cx + ((p - 1) / (2 * rhoDom) + 0.5) * cw;
    ctx.save();
    ctx.fillStyle = alphaOf(T.ok, 0.13);
    ctx.fillRect(cxFor(1 - st.eps), rowsRect.y - 4, cxFor(1 + st.eps) - cxFor(1 - st.eps), rowsRect.h + 8);
    ctx.strokeStyle = alphaOf(T.ok, 0.55); ctx.lineWidth = 1;
    for (const e of [1 - st.eps, 1 + st.eps]) {
      ctx.beginPath(); ctx.moveTo(cxFor(e), rowsRect.y - 4); ctx.lineTo(cxFor(e), rowsRect.y + rowsRect.h + 4); ctx.stroke();
    }
    ctx.strokeStyle = alphaOf(T.n14, 0.35); ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(cxFor(1), rowsRect.y - 4); ctx.lineTo(cxFor(1), rowsRect.y + rowsRect.h + 4); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    // ONE centred label, not one per edge: at a small ε the two edge labels
    // collide into an unreadable run of digits.
    r.label(`clip window [${f2(1 - st.eps)}, ${f2(1 + st.eps)}]`, cx + cw / 2, rowsRect.y + rowsRect.h + 17,
      { color: T.ok, font: '9px ui-monospace, monospace', align: 'center' });

    // ----------------------------------------------------------- the rollouts
    const aDom = Math.max(1e-3, ...m.rows.map((x) => Math.abs(x.A)));
    for (let i = 0; i < G; i++) {
      const y = rowsRect.y + i * rowH, mid = y + rowH / 2, bh = Math.max(5, rowH * 0.44);
      const scored = i < S;
      const row = scored ? m.rows[i] : null;

      // -- label column: id, a length sketch, the verifier's verdict
      ctx.save();
      ctx.font = '10.5px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillStyle = scored ? T.n12 : T.n8;
      ctx.fillText(`o${i}`, labX, mid + 3.5);
      const tk = base.toks[i] || 4;
      for (let t = 0; t < tk; t++) {
        ctx.fillStyle = scored ? rgbaToken('n14', 0.16) : rgbaToken('n14', 0.07);
        ctx.fillRect(labX + 22 + t * 7, mid - 3, 5, 6);
      }
      if (scored) {
        const ok = rew[i] >= 0.5;
        ctx.fillStyle = ok ? T.ok : T.bad;
        ctx.font = '11px ui-monospace, monospace';
        ctx.fillText(ok ? '✓' : '✗', labX + labW - 12, mid + 3.5);
      }
      ctx.restore();

      // -- reward bar (draggable)
      ctx.save();
      if (!scored) {
        ctx.strokeStyle = rgbaToken('n14', 0.12); ctx.lineWidth = 1;
        ctx.strokeRect(rx, mid - bh / 2, rw, bh);
        ctx.fillStyle = T.n9; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left';
        ctx.fillText('not sampled yet', rx + 6, mid + 3);
      } else {
        ctx.fillStyle = rgbaToken('n14', 0.05); ctx.fillRect(rx, mid - bh / 2, rw, bh);
        ctx.fillStyle = alphaOf(T.accent, 0.8);
        ctx.fillRect(rx, mid - bh / 2, Math.max(1.5, rew[i] * rw), bh);
        // the drag handle
        ctx.fillStyle = T.accent;
        ctx.beginPath(); ctx.arc(rx + rew[i] * rw, mid, 4.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = T.n0;
        ctx.beginPath(); ctx.arc(rx + rew[i] * rw, mid, 1.6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = T.n11; ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'left';
        ctx.fillText(f3(rew[i]), rx + rw + 6, mid + 3);
      }
      ctx.restore();

      if (!scored) continue;

      // -- advantage bar, signed about the zero axis
      const half = aw / 2 - 26;
      const len = (row.A / aDom) * half;
      ctx.save();
      ctx.fillStyle = row.A >= 0 ? alphaOf(T.ok, row.clamped ? 0.3 : 0.85) : alphaOf(T.bad, row.clamped ? 0.3 : 0.85);
      const x0 = Math.min(acx, acx + len), wpx = Math.abs(len);
      ctx.fillRect(x0, mid - bh / 2, Math.max(0.8, wpx), bh);
      if (row.clamped) hatch(ctx, x0, mid - bh / 2, Math.max(0.8, wpx), bh);
      ctx.restore();
      if (Math.abs(row.A) < 1e-9) {
        r.label('A = 0 · no gradient', acx + 6, mid + 3.5,
          { color: T.bad, font: '9px ui-monospace, monospace' });
      } else {
        r.label((row.A >= 0 ? '+' : '') + f3(row.A), len >= 0 ? acx + wpx + 5 : acx - wpx - 5, mid + 3.5,
          { color: row.clamped ? T.n9 : (row.A >= 0 ? T.ok : T.bad), font: '9.5px ui-monospace, monospace', align: len >= 0 ? 'left' : 'right' });
      }

      // -- ratio marker
      ctx.save();
      const px = Math.max(cx + 2, Math.min(cx + cw - 2, cxFor(row.rho)));
      ctx.fillStyle = row.clamped ? T.warn : T.n11;
      ctx.beginPath(); ctx.arc(px, mid, row.clamped ? 4.6 : 3.2, 0, Math.PI * 2); ctx.fill();
      if (row.clamped) {
        ctx.strokeStyle = T.warn; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, mid - 6); ctx.lineTo(px, mid + 6); ctx.stroke();
        // Put the word on whichever side still has room inside the column.
        ctx.fillStyle = T.warn; ctx.font = '9px ui-monospace, monospace';
        const fits = px + 7 + 46 <= cx + cw;
        ctx.textAlign = fits ? 'left' : 'right';
        ctx.fillText('clamped', fits ? px + 7 : px - 7, mid + 3);
      }
      ctx.restore();
    }

    // ------------------------------------------- the mean line, drawn on top
    const mx = rx + m.mu * rw;
    ctx.save();
    ctx.strokeStyle = T.violet; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(mx, rowsRect.y - 8); ctx.lineTo(mx, rowsRect.y + rowsRect.h + 4); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    r.label(`μ = ${f3(m.mu)}  ← the baseline`, mx + 6, rowsRect.y - 11,
      { color: T.violet, font: 'bold 10.5px ui-monospace, monospace' });
    // the zero axis of the advantage column IS that same mean
    ctx.save();
    ctx.strokeStyle = alphaOf(T.violet, 0.55); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(acx, rowsRect.y - 4); ctx.lineTo(acx, rowsRect.y + rowsRect.h + 4); ctx.stroke();
    ctx.restore();
    r.label('A = 0', acx, rowsRect.y + rowsRect.h + 17, { color: T.violet, font: '9px ui-monospace, monospace', align: 'center' });

    // ------------------------------------------------------ the bottom strip
    const by = rowsRect.y + rowsRect.h + 34;
    const agree = m.sigma < 1e-12;
    const gx = pad, gw = Math.min(170, page.W * 0.22), gh = 16;

    r.label('gradient signal (unclipped mean |A|)',
      gx, by, { color: T.n11, font: '10.5px ui-monospace, monospace' });
    ctx.save();
    ctx.fillStyle = rgbaToken('n14', 0.07); ctx.fillRect(gx, by + 8, gw, gh);
    const sigDom = st.norm ? 1.4 : 0.55;
    ctx.fillStyle = agree ? T.bad : T.teal;
    ctx.fillRect(gx, by + 8, Math.max(agree ? 0 : 1.5, Math.min(1, m.signal / sigDom) * gw), gh);
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(gx, by + 8, gw, gh);
    ctx.font = 'bold 20px ui-monospace, monospace'; ctx.textAlign = 'left';
    ctx.fillStyle = agree ? T.bad : T.n13;
    ctx.fillText(f3(m.signal), gx + gw + 12, by + 25);
    ctx.restore();

    const col2 = gx + gw + 122;
    const lines = [
      `G = ${G} · scored ${S}/${G} · μ = ${f3(m.mu)} · σ = ${f3(m.sigma)}`,
      `baseline noise σ/√${S} = ${f3(m.sigma / Math.sqrt(S))}${S <= 3 ? '  — a group this small is a NOISY baseline' : ''}`,
      st.norm
        ? `normalised: every A ×${(m.scale).toFixed(1)}${m.sigma < 0.05 && !agree ? '  — a near-agreeing group amplified to full size' : ''}`
        : 'un-normalised: A stays in reward units',
      `clipped ${pct(m.clipFrac)} (${S - m.nLive} of ${S}) — a clamped term has no gradient`,
    ];
    for (let i = 0; i < lines.length; i++) {
      r.label(lines[i], col2, by + 2 + i * 14, { color: i === 3 && m.clipFrac > 0 ? T.warn : T.n11, font: '10.5px ui-monospace, monospace' });
    }
    r.label('shown: one group\'s advantage arithmetic. NOT shown: the policy update, the weights,',
      gx, by + 62, { color: T.n9, font: '9.5px ui-monospace, monospace' });
    r.label('or any training curve — a weight update is not something a reader can watch.',
      gx, by + 74, { color: T.n9, font: '9.5px ui-monospace, monospace' });

    // ---------------------------------------------------------- hover-inspect
    if (page.pointer.over && !grab) {
      const px = page.pointer.x, py = page.pointer.y;
      const j = Math.floor((py - rowsRect.y) / rowH);
      if (j >= 0 && j < S && py >= rowsRect.y && py <= rowsRect.y + rowsRect.h) {
        const row = m.rows[j];
        if (px >= rx - 8 && px <= rx + rw + 40) {
          page.setTip(
            `rollout o${j} — verifier says ${row.r >= 0.5 ? 'CORRECT' : 'wrong'}\n` +
            `reward r = ${f3(row.r)}\n` +
            `group mean μ = (${m.rows.map((x) => f3(x.r)).join(' + ')}) / ${S} = ${f3(m.mu)}\n` +
            `drag ↔ to change this reward · click the ✓/✗ tag to flip it`);
        } else if (px >= ax && px <= ax + aw) {
          page.setTip(
            `rollout o${j} advantage\n` +
            `r − μ = ${f3(row.r)} − ${f3(m.mu)} = ${(row.raw >= 0 ? '+' : '') + f3(row.raw)}\n` +
            (st.norm
              ? `÷ (σ + 1e−4) = ${f3(row.raw)} / ${(m.sigma + STD_EPS).toFixed(5)} = ${(row.A >= 0 ? '+' : '') + f3(row.A)}\n`
              : `A = ${(row.A >= 0 ? '+' : '') + f3(row.A)}  (no std normalisation)\n`) +
            (Math.abs(row.A) < 1e-9
              ? 'exactly zero — this rollout earned the group mean, so it moves nothing'
              : row.clamped ? 'clamped by the clip window: this term is constant in ρ, so its gradient is zero'
                : 'inside the clip window — this term carries gradient'));
        } else if (px >= cx - 6 && px <= cx + cw + 6) {
          page.setTip(
            `rollout o${j} policy ratio\n` +
            `ρ = ${f3(row.rho)}   window [${f2(1 - st.eps)}, ${f2(1 + st.eps)}]\n` +
            `term = min(ρ·A, clip(ρ)·A) = ${f3(Math.min(row.rho * row.A, Math.max(1 - st.eps, Math.min(1 + st.eps, row.rho)) * row.A))}\n` +
            (row.clamped
              ? `clamped: A ${row.A > 0 ? '> 0 and ρ > 1+ε' : '< 0 and ρ < 1−ε'}, so the clipped branch wins and the term stops depending on ρ`
              : 'unclamped: the ρ·A branch wins, so this rollout still pushes the policy'));
        }
      } else if (py >= by && py <= by + 34 && px >= gx && px <= gx + gw + 90) {
        page.setTip(
          `gradient signal = mean |A| over the ${m.nLive} unclipped rollout${m.nLive === 1 ? '' : 's'}\n` +
          (m.nLive
            ? `= (${m.rows.filter((x) => !x.clamped).map((x) => Math.abs(x.A).toFixed(3)).join(' + ')}) / ${m.nLive} = ${f3(m.signal)}`
            : 'every rollout is clamped, so the group moves nothing this step') +
          (agree ? '\nevery reward equals μ, so every A is exactly 0 — this group is filtered out by dynamic sampling' : ''));
      }
    }

    // ------------------------------------------------------------- readout
    let out = `${st.norm ? 'A_i = (r_i − μ) / (σ + 1e−4)' : 'A_i = r_i − μ'}    μ = (1/${S}) Σ r_i = ${f3(m.mu)}    σ = ${f3(m.sigma)}    no value network — the baseline IS the other ${S - 1} sample${S === 2 ? '' : 's'}    tier:${r.name}\n`;
    out += `scored ${S}/${G}:  A = [ ${m.rows.map((x) => (x.A >= 0 ? '+' : '') + f3(x.A)).join(', ')} ]   ·   clipped ${pct(m.clipFrac)}   ·   gradient signal = ${f3(m.signal)}\n`;
    if (agree) {
      out += `EVERY rollout scored ${f3(m.mu)} — the group AGREES, so every advantage is exactly 0.000 and this group contributes NO gradient at all. Dividing by σ does not rescue it: 0/(0+1e−4) is still 0. This is the case dynamic sampling exists to throw away — filter out the all-correct and all-wrong groups and sample replacements until the batch is full of groups that disagree.`;
    } else if (m.nLive === 0) {
      out += `every rollout is outside the clip window on the side the advantage pushes, so every term is constant in ρ — the advantages are non-zero but the step moves nothing. Widen ε or reduce the drift.`;
    } else if (m.sigma < 0.05) {
      out += `σ = ${f3(m.sigma)} — the group nearly agrees, so the advantages are tiny${st.norm ? `, and dividing by σ+1e−4 blows them back up ${m.scale.toFixed(1)}×. That is the std-normalisation bias: a prompt the model has almost mastered is weighted as heavily as one it is genuinely learning from.` : '. Turn on std normalisation to see them amplified back to full size — which is the bias Dr. GRPO objects to.'}`;
    } else {
      const nUp = m.rows.filter((x) => x.A > 0 && !x.clamped).length;
      const nDn = m.rows.filter((x) => x.A < 0 && !x.clamped).length;
      out += `the group disagrees (σ = ${f3(m.sigma)}), so ${m.nLive} of ${S} rollouts carry gradient: ${nUp} above the mean ${nUp === 1 ? 'is' : 'are'} pushed up, ${nDn} below ${nDn === 1 ? 'is' : 'are'} pushed down, and the baseline itself is never learned — it is just the other samples.`;
    }
    page.setReadout(out);
  },
}).then((page) => {
  window.__grpoPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  for (const k of ['pass', 'spread', 'eps', 'drift', 'seed']) if (q.has(k)) page.controls.set(k, parseFloat(q.get(k)));
  if (q.has('g')) page.controls.set('g', parseInt(q.get('g'), 10), { rebuild: true });
  if (q.has('norm')) page.controls.set('norm', q.get('norm') === '1' || q.get('norm') === 'true');
  // ?r=i,value sets ONE rollout's reward -- the headless stand-in for a bar drag.
  // ?rewards=a,b,c sets the whole group at once (…&rewards=1,1,1,1 is the
  // all-agree case, and it must read exactly 0.000).
  if (q.has('rewards')) {
    ensure(page.state);
    q.get('rewards').split(',').map(Number).forEach((v, i) => { if (Number.isFinite(v)) over[i] = Math.max(0, Math.min(1, v)); });
    edits++;
  }
  if (q.has('r')) {
    ensure(page.state);
    const [i, v] = q.get('r').split(',').map(Number);
    if (Number.isFinite(i) && Number.isFinite(v)) { over[i] = Math.max(0, Math.min(1, v)); edits++; }
  }
  // ?hover=x,y fakes the cursor (headless stand-in: --screenshot has no pointer).
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  // Rebuild BEFORE seeking: ?g= changed the step count, and a seek against the
  // stale list clamps to the old last index (a ?g=10&step=9 landed on 8 of 10).
  if (q.has('step') && t) { t.rebuildIfDirty(); t.seek(parseInt(q.get('step'), 10)); }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
