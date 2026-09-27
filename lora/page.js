// lora -- a low-rank adapter instead of a full weight update.
//
// A fine-tune that touches a layer normally updates all din*dout numbers in it.
// LoRA freezes that layer and trains two thin matrices instead, so the update is
// the PRODUCT of them: dW = B·A, with B dout×r and A r×din. That costs
// r*(din+dout) trained numbers instead of din*dout, and the update it can write
// has rank at most r -- an r-dimensional slice of a layer that has min(din,dout)
// of them available. So the rank slider trades one thing it can be measured on
// against the other: fewer parameters for a smaller say in the answer.
//
// The serving choice is the second, independent axis. Fold BA into the frozen
// layer and serve time is untouched -- the sum is just a wider W. Leave the
// adapter as a side path and every token pays one extra thin GEMM. Same trained
// numbers, different cost per token, which is why the toggle does not move the
// bars.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf } from '../framework/theme.js';
import { loraParams, fullParams, rankOf, loraProduct, extraMacs } from './math.js';

// One frozen square layer, so the only axes on screen are rank and the serving
// choice. 8x8 keeps the rank the page reports exactly the rank it draws.
const DIN = 8, DOUT = 8;
const TOKENS = 1;                                  // MACs reported per token
const RMAX = 8;
const BREAK_EVEN = (DIN * DOUT) / (DIN + DOUT);     // rank where the adapter stops being cheaper

// The adapter matrices, rebuilt for the current rank from fixed seeds so the
// picture is the same on every repaint. Only their SHAPES matter to what the page
// reports: A is r×din, B is dout×r, so whatever the numbers, B·A spans at most
// min(r, din, dout) dimensions.
function adapters(r) {
  const toRows = (m, rows, cols) => {
    const out = [];
    for (let i = 0; i < rows; i++) out.push(Array.from(m.data.subarray(i * cols, i * cols + cols)));
    return out;
  };
  return {
    A: toRows(seededRandn(11, [r, DIN], { std: 1 }), r, DIN),
    B: toRows(seededRandn(23, [DOUT, r], { std: 1 }), DOUT, r),
  };
}

let bars = null;   // [{x, y, w, h, kind}] for hover, captured during draw

mount({
  mount: 'body',
  title: 'lora — a low-rank update instead of a full one',
  blurb: 'A LoRA update is the product of two thin matrices, dW = B·A, so training it touches r·(din + dout) numbers instead of din · dout. The catch is that a rank-r product can only write an r-dimensional slice of the layer: shrink the rank and the two bars get further apart in cost while the update\'s say in the answer shrinks with them. Drag the rank and watch where the blue bar crosses the dashed break-even line — past rank 4 on an 8×8 layer the adapter costs MORE parameters than just training W, and only at rank 8 can it express every update a full matrix can. The merged toggle is the other, independent question: fold BA into W and serving is untouched, keep the adapter as a side path and every token pays one extra thin GEMM.',
  prefer: 'canvas2d',
  aspect: '2 / 1',

  controls: (c) => {
    c.slider('r', { label: `rank r — din = dout = ${DIN}`, min: 1, max: RMAX, step: 1, value: 2 });
    c.toggle('merged', { label: 'merge BA into W at serve time', value: true });
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);

    const rank = Math.max(1, Math.min(RMAX, st.r | 0));
    const merged = !!st.merged;
    const { A, B } = adapters(rank);
    const delta = loraProduct(B, A);                    // dout×din, what gets added to W
    const dRank = rankOf(delta);
    const lp = loraParams(DIN, DOUT, rank);
    const fp = fullParams(DIN, DOUT);
    const macMerged = extraMacs(DIN, DOUT, rank, TOKENS, true);
    const macOpen = extraMacs(DIN, DOUT, rank, TOKENS, false);
    const baseMac = DIN * DOUT * TOKENS;                // the frozen matmul every token pays anyway
    const pctOpen = baseMac ? Math.round((100 * macOpen) / baseMac) : 0;

    // ---- geometry: two bars filling most of the canvas ----------------------
    const pad = 16;
    const topY = page.H * 0.22, baseY = page.H * 0.80;
    const barW = Math.max(56, Math.min(page.W * 0.19, 150));
    const lx = page.W * 0.12, fx = page.W * 0.40;
    const scale = (baseY - topY) / Math.max(lp, fp);
    const lh = lp * scale, fh = fp * scale;
    bars = [];

    // ---- headings ----------------------------------------------------------
    r.label('trained numbers for one frozen 8×8 layer', pad, page.H * 0.12,
      { color: T.n14, font: '12px ui-monospace, monospace' });
    r.label('the update is a PRODUCT of two thin matrices, so rank is the only dial', pad, page.H * 0.12 + 15,
      { color: T.n10, font: '10px ui-monospace, monospace' });

    // ---- the two bars ------------------------------------------------------
    const rect = (x, h, fill, kind) => {
      const q = { x, y: baseY - h, w: barW, h, kind };
      ctx.save();
      ctx.fillStyle = fill; ctx.fillRect(q.x, q.y, q.w, q.h);
      ctx.strokeStyle = alphaOf(T.n12, 0.55); ctx.lineWidth = 1; ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1);
      ctx.restore();
      bars.push(q);
      return q;
    };
    const lr = rect(lx, lh, T.accent, 'lora');
    const fr = rect(fx, fh, T.teal, 'full');

    // The rank-r product is r rank-1 terms summed, and each one costs the same
    // din+dout numbers -- so the adapter bar divides into exactly r equal bands.
    ctx.save();
    ctx.strokeStyle = alphaOf(T.n0, 0.6); ctx.lineWidth = 1;
    for (let k = 1; k < rank; k++) {
      const y = baseY - (lh * k) / rank;
      ctx.beginPath(); ctx.moveTo(lx, y); ctx.lineTo(lx + barW, y); ctx.stroke();
    }
    ctx.restore();

    // Break-even: the level at which the adapter has as many numbers as W.
    ctx.save();
    ctx.strokeStyle = alphaOf(T.teal, 0.7); ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(lx - 10, baseY - fh); ctx.lineTo(fx + barW + 10, baseY - fh); ctx.stroke();
    ctx.restore();
    r.label(`break-even ${fp}`, fx - 6, baseY - fh - 5,
      { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'right' });

    for (const q of [lr, fr]) {
      r.label(String(q.kind === 'lora' ? lp : fp), q.x + q.w / 2, q.y - 6,
        { color: q.kind === 'lora' ? T.accent : T.tealDeep, font: '12px ui-monospace, monospace', align: 'center' });
    }
    r.label(`LoRA r = ${rank}`, lx + barW / 2, baseY + 15, { color: T.n12, font: '11px ui-monospace, monospace', align: 'center' });
    r.label('full ΔW', fx + barW / 2, baseY + 15, { color: T.n12, font: '11px ui-monospace, monospace', align: 'center' });
    r.label(`r·(din+dout) = ${rank}·${DIN + DOUT}`, lx + barW / 2, baseY + 29, { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'center' });
    r.label(`${DIN}·${DOUT}`, fx + barW / 2, baseY + 29, { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'center' });

    // ---- what the rank actually buys (right of the bars) --------------------
    const px = Math.max(fx + barW + 20, page.W * 0.62);
    let py = topY - 4;
    r.label('what rank r buys', px, py, { color: T.n14, font: '11px ui-monospace, monospace' });
    py += 16;
    r.label(`rank of B·A        = ${dRank}`, px, py, { color: T.n12, font: '10px ui-monospace, monospace' });
    py += 14;
    r.label(`directions used    = ${dRank} of ${Math.min(DIN, DOUT)}`, px, py, { color: T.n12, font: '10px ui-monospace, monospace' });
    py += 14;
    const cheaper = lp <= fp;
    const ratio = Math.round((100 * lp) / fp);
    r.label(`vs training W      = ${ratio}%`, px, py, { color: cheaper ? T.okDeep : T.warnDeep, font: '10px ui-monospace, monospace' });
    py += 20;

    // ---- the serving choice, which does not move the bars ------------------
    r.label('serve-time cost, per token', px, py, { color: T.n14, font: '11px ui-monospace, monospace' });
    py += 8;
    const rowH = 30, rowW = Math.max(120, page.W - px - pad);
    const rows = [
      { on: merged, tint: T.accent, head: 'merged — BA folded into W', mac: macMerged, note: 'one GEMM, nothing added' },
      { on: !merged, tint: T.teal, head: 'unmerged — side path', mac: macOpen, note: `+${pctOpen}% of the base ${baseMac}` },
    ];
    for (const row of rows) {
      const y0 = py;
      ctx.save();
      ctx.fillStyle = row.on ? alphaOf(row.tint, 0.16) : T.n1;
      ctx.strokeStyle = row.on ? row.tint : T.n5; ctx.lineWidth = row.on ? 1.8 : 1;
      ctx.fillRect(px, y0, rowW, rowH); ctx.strokeRect(px + 0.5, y0 + 0.5, rowW - 1, rowH - 1);
      ctx.restore();
      r.label(row.head, px + 7, y0 + 12, { color: row.on ? T.n13 : T.n10, font: '10px ui-monospace, monospace' });
      r.label(`${row.mac} extra MAC — ${row.note}`, px + 7, y0 + 24, { color: row.on ? T.n12 : T.n9, font: '9.5px ui-monospace, monospace' });
      py += rowH + 6;
    }
    r.label(`base x·W = ${baseMac} MAC either way`, px, py + 4, { color: T.n10, font: '9.5px ui-monospace, monospace' });

    // ---- hover -------------------------------------------------------------
    if (page.pointer.over) {
      const p = page.pointer;
      const hit = bars.find((q) => p.x >= q.x && p.x <= q.x + q.w && p.y >= q.y && p.y <= q.y + q.h);
      if (hit) {
        if (hit.kind === 'lora') {
          page.setTip(`LoRA adapter: ${lp} trained numbers = r·(din+dout) = ${rank}·${DIN + DOUT}\n` +
            `B is ${DOUT}×${rank}, A is ${rank}×${DIN}; B·A is a sum of ${rank} rank-1 term${rank > 1 ? 's' : ''}, the bands.\n` +
            `rank(B·A) = ${dRank}, so it writes ${dRank} of the ${Math.min(DIN, DOUT)} directions a full ΔW has.\n` +
            `every band is the same height because every one costs the same ${DIN + DOUT} numbers`);
        } else {
          page.setTip(`full update: ${fp} trained numbers = din·dout = ${DIN}·${DOUT}\n` +
            `rank ${Math.min(DIN, DOUT)} — every direction in the layer is editable.\n` +
            `the dashed line is this level: a blue bar above it is the costlier of the two`);
        }
      }
    }

    page.probe = { rank, dRank, lora: lp, full: fp, merged, macMerged, macOpen, ratio };

    const line = lp <= fp ? `the adapter is ${ratio}% of training W` : `the adapter is ${ratio}% of training W — MORE numbers than training W outright`;
    let o = `lora on a frozen ${DIN}×${DOUT} layer, rank r = ${rank}:  `;
    o += `${lp} trained numbers as an adapter (r·(din+dout) = ${rank}·${DIN + DOUT}) vs ${fp} for the full matrix (din·dout) — ${line}\n`;
    o += `rank(B·A) = ${dRank} = min(r, din, dout), so this adapter can write ${dRank} of the ${Math.min(DIN, DOUT)} directions in the layer`;
    if (dRank < Math.min(DIN, DOUT)) o += ' — the rest of the layer is untouched';
    o += `\nserve ${merged ? 'MERGED' : 'UNMERGED'}: extra MACs per token = ${merged ? macMerged : macOpen}`;
    o += merged ? ` (BA is inside W, so the serving GEMM is unchanged) | unmerged would cost ${macOpen} = +${pctOpen}% on the base ${baseMac}`
                : ` (${pctOpen}% on top of the base ${baseMac} MAC) | merging them in costs 0`;
    if (rank > BREAK_EVEN) o += `\nbreak-even rank is ${DIN * DOUT} / ${DIN + DOUT} = ${BREAK_EVEN}: above it the adapter has more numbers than W`;
    page.setReadout(o);
  },

  challenges: [
    {
      goal: 'Find the rank where the adapter stops being the cheap option: get the blue bar taller than the dashed line.',
      hint: 'r·(din+dout) passes din·dout once r is larger than 64/16 = 4.',
      check: (api) => ({
        solved: api.probe.lora != null && api.probe.lora > api.probe.full,
        detail: `adapter ${api.probe.lora == null ? '?' : api.probe.lora} vs full ${api.probe.full == null ? '?' : api.probe.full} (r = ${api.probe.rank})`,
      }),
    },
    {
      goal: 'Give the adapter the whole layer back: get rank(B·A) to 8.',
      hint: 'rank(B·A) is min(r, din, dout), and both sides are 8 here — so the rank has to reach the full 8.',
      check: (api) => ({
        solved: api.probe.dRank === Math.min(DIN, DOUT),
        detail: `rank(B·A) = ${api.probe.dRank == null ? '?' : api.probe.dRank}, at ${api.probe.ratio == null ? '?' : api.probe.ratio}% of training W`,
      }),
    },
    {
      goal: 'Serve it with zero extra arithmetic while keeping rank 4 or higher.',
      hint: 'Only folding BA into W leaves the serving GEMM alone — the toggle, not the rank.',
      check: (api) => ({
        solved: api.probe.macMerged === 0 && api.probe.rank >= 4 && api.probe.merged,
        detail: `extra = ${api.probe.macOpen == null ? '?' : api.probe.macOpen} unmerged / ${api.probe.macMerged == null ? '?' : api.probe.macMerged} merged, merged = ${api.probe.merged}, r = ${api.probe.rank}`,
      }),
    },
  ],
}).then((page) => {
  window.__loraPage = page;
});
