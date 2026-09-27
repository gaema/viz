// A cached key was rotated for the position it was computed at. Moving the
// chunk means rotating that key by the position gap. The printed score is
// the cosine of the printed angle difference.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { ROPE_BASE, ROPE_DIM, moveErrorText, scoreFromPrinted, spliceError } from './math.js';

const CHUNK = 8;

mount({
  mount: 'body',
  title: 'RoPE splice — rotate a cached chunk to its new position',
  blurb: 'A stored key carries the position it was computed at. Moving the chunk changes the query position. Rotating the key by that gap makes the pairwise angle match a key computed at the destination.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('src', { label: 'cached position', min: 0, max: 32, step: 1, value: 0 });
    c.stepper('dest', { label: 'destination position', min: 0, max: 48, step: 1, value: 16 });
    c.stepper('pair', { label: 'RoPE pair', min: 0, max: 3, step: 1, value: 0 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const src = page.state.src | 0;
    const dest = page.state.dest | 0;
    const pair = page.state.pair | 0;
    const raw = scoreFromPrinted(dest, src, pair, ROPE_DIM, ROPE_BASE);
    const native = scoreFromPrinted(dest, dest, pair, ROPE_DIM, ROPE_BASE);
    const moved = spliceError(1, 0, src, dest, pair, ROPE_DIM, ROPE_BASE);
    const errText = moveErrorText(moved.err);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);

    const pad = W * 0.06;
    const axisMax = Math.max(src + CHUNK, dest + CHUNK, 1);
    const axisW = W - pad * 2;
    const xOf = (pos) => pad + (pos / axisMax) * axisW;
    const chip = Math.max(6, Math.min(18, (axisW / axisMax) * 0.8));
    ctx.font = `${Math.max(12, H * 0.034)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    const bands = [
      { y: H * 0.08, h: H * 0.16, name: 'cached at source', origin: src, color: T.n6 },
      { y: H * 0.30, h: H * 0.16, name: 'spliced at destination', origin: dest, color: T.teal },
    ];
    for (const band of bands) {
      ctx.fillStyle = T.n12;
      ctx.fillText(band.name, pad, band.y + band.h * 0.22);
      for (let i = 0; i < CHUNK; i++) {
        const x = xOf(band.origin + i);
        ctx.fillStyle = i === 0 ? T.accent : band.color;
        ctx.fillRect(x, band.y + band.h * 0.42, chip, band.h * 0.5);
      }
    }
    ctx.strokeStyle = T.gold;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xOf(src) + chip / 2, H * 0.08 + H * 0.16);
    ctx.lineTo(xOf(dest) + chip / 2, H * 0.30 + H * 0.16 * 0.42);
    ctx.stroke();

    const scores = [
      ['unrotated', raw, T.accent],
      ['spliced', native, T.teal],
    ];
    const rowY = H * 0.54;
    const rowH = H * 0.12;
    const nameW = W * 0.28;
    const barX = pad + nameW;
    const barW = W - barX - pad;
    const mid = barX + barW / 2;
    scores.forEach((item, i) => {
      const y = rowY + i * (rowH + H * 0.04);
      const score = Number(item[1].score);
      const reach = (barW / 2) * Math.min(1, Math.abs(score));
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'left';
      ctx.fillText(item[0], pad, y + rowH / 2);
      ctx.fillStyle = T.n3;
      ctx.fillRect(barX, y, barW, rowH);
      ctx.fillStyle = item[2];
      if (score >= 0) ctx.fillRect(mid, y, reach, rowH);
      else ctx.fillRect(mid - reach, y, reach, rowH);
      ctx.fillStyle = T.n12;
      ctx.fillRect(mid - 1, y, 2, rowH);
    });

    page.setReadout(
      `pair ${pair}, source ${src}, destination ${dest}: `
      + `unrotated cos(${raw.diff}) = ${raw.score}; `
      + `spliced cos(${native.diff}) = ${native.score}; `
      + `move error ${errText}.`,
    );
    page.probe = { raw, native, err: moved.err, errText };
  },
});
