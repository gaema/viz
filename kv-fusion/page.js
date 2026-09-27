// A prefix cache reuses a chunk only when the request starts with it.
// Fusion keeps the stored KV of a chunk that sits later, recomputes a
// fraction of its tokens, and leaves a residual on the rest.
import { mount } from '../framework/layout.js';
import { T, rgbaToken } from '../framework/theme.js';
import { fusionBill } from './math.js';

function paintRun(ctx, x, y, w, h, n, paid, paidColor, freeColor) {
  if (n <= 0 || w <= 0) return;
  const cw = w / n;
  const whole = Math.floor(paid + 1e-9);
  const part = paid - whole;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = i < whole ? paidColor : freeColor;
    ctx.fillRect(x + i * cw + 1, y, Math.max(1, cw - 2), h);
  }
  if (part > 1e-9 && whole < n) {
    ctx.fillStyle = paidColor;
    ctx.fillRect(x + whole * cw + 1, y, Math.max(1, (cw - 2) * part), h);
  }
}

mount({
  mount: 'body',
  title: 'kv fusion — reuse a chunk that is not a prefix',
  blurb: 'A prefix cache pays for every token of a chunk that does not start the request. Fusion keeps that chunk and recomputes a fraction of it. The residual error is the unrecomputed share of the mismatch.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('header', { label: 'tokens before the cached chunk', min: 0, max: 12, step: 1, value: 4 });
    c.stepper('tokens', { label: 'cached chunk tokens', min: 4, max: 16, step: 4, value: 8 });
    c.slider('frac', { label: 'fraction recomputed', min: 0, max: 1, step: 0.25, value: 0.25 });
    c.slider('mismatch', { label: 'chunk mismatch', min: 0, max: 1, step: 0.25, value: 1 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const b = fusionBill(page.state.tokens, page.state.header, page.state.frac, page.state.mismatch);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);

    const pad = Math.max(12, W * 0.03);
    const labelW = Math.min(150, W * 0.22);
    const tapeX = pad + labelW;
    const tapeW = W - tapeX - pad;
    const rows = [
      { name: 'where it sits', headerPaid: b.header, chunkPaid: 0, free: T.n9 },
      { name: 'prefix cache', headerPaid: b.header, chunkPaid: b.prefixChunk, free: T.teal },
      { name: 'fusion', headerPaid: b.header, chunkPaid: b.fusionChunk, free: T.teal },
    ];
    const top = H * 0.06;
    const rowH = H * 0.16;
    const gap = H * 0.035;
    const n = b.header + b.tokens;
    const headerW = n === 0 ? 0 : tapeW * (b.header / n);
    const chunkW = tapeW - headerW;
    ctx.font = `${Math.max(12, H * 0.035)}px sans-serif`;
    ctx.textBaseline = 'middle';

    rows.forEach((row, i) => {
      const y = top + i * (rowH + gap);
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'left';
      ctx.fillText(row.name, pad, y + rowH / 2);
      if (b.header > 0) {
        paintRun(ctx, tapeX, y, headerW, rowH, b.header, row.headerPaid, T.accent, T.n6);
      }
      paintRun(ctx, tapeX + headerW, y, chunkW, rowH, b.tokens, row.chunkPaid, T.accent, row.free);
      if (row.name === 'fusion' && Number(b.error) > 0) {
        const kept = b.tokens - b.fusionChunk;
        if (kept > 0) {
          const cw = chunkW / b.tokens;
          ctx.fillStyle = rgbaToken('gold', Number(b.error));
          ctx.fillRect(tapeX + headerW + b.fusionChunk * cw, y, kept * cw, rowH);
        }
      }
    });

    const barTop = top + 3 * (rowH + gap) + H * 0.02;
    const barH = H - barTop - H * 0.08;
    const labels = [
      ['cold', b.coldText, T.n6],
      ['prefix cache', b.prefixText, T.accent],
      ['fusion', b.fusionText, T.teal],
    ];
    const bw = (tapeW - 16) / 3;
    labels.forEach((item, i) => {
      const paid = Number(item[1]);
      const h = b.cold === 0 ? 0 : barH * (paid / b.cold);
      const x = tapeX + i * (bw + 8);
      ctx.fillStyle = item[2];
      ctx.fillRect(x, barTop + (barH - h), bw, h);
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'center';
      ctx.fillText(`${item[0]} ${item[1]}`, x + bw / 2, barTop + barH + H * 0.04);
    });

    const where = b.header === 0
      ? 'The chunk starts the request, so the prefix cache pays less.'
      : 'The chunk is not a prefix, so fusion pays fewer of its tokens.';
    page.setReadout(
      `header ${b.header}, chunk ${b.tokens}, fraction ${b.frac}, mismatch ${b.mismatch}: `
      + `prefix cache pays ${b.prefixText} tokens, fusion pays ${b.fusionText} tokens, residual error ${b.error}. ${where}`,
    );
    page.probe = b;
  },
});
