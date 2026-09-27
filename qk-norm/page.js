// Each query and key is divided by its own RMS before the dot product.
// The mass bars are the softmax of the scores printed in the readout.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { massLabel } from './math.js';

const NAMES = ['off-axis', 'aligned'];

mount({
  mount: 'body',
  title: 'QK-norm — a long key stops winning by being long',
  blurb: 'Queries and keys are divided by their own RMS before the dot product. A long key then stops collecting the softmax just because it is long, and the aligned key can win.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.slider('scale', { label: 'off-axis key scale', min: 0.2, max: 6, step: 0.2, value: 3 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const raw = massLabel(page.state.scale, false);
    const fair = massLabel(page.state.scale, true);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const bands = [
      { name: 'raw softmax', row: raw, y: H * 0.08 },
      { name: 'QK-norm softmax', row: fair, y: H * 0.5 },
    ];
    const colors = [T.accent, T.teal];
    const pad = W * 0.06;
    const labelW = W * 0.22;
    const barX = pad + labelW;
    const barW = W - barX - pad;
    ctx.font = `${Math.max(12, H * 0.04)}px sans-serif`;
    ctx.textBaseline = 'middle';
    bands.forEach((band) => {
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'left';
      ctx.fillText(band.name, pad, band.y + H * 0.04);
      band.row.mass.forEach((m, i) => {
        const y = band.y + H * 0.12 + i * H * 0.14;
        const w = barW * Number(m);
        ctx.fillStyle = T.n3;
        ctx.fillRect(barX, y, barW, H * 0.1);
        ctx.fillStyle = colors[i];
        ctx.fillRect(barX, y, w, H * 0.1);
        ctx.fillStyle = T.n12;
        ctx.fillText(NAMES[i], pad, y + H * 0.05);
      });
    });
    const nameOf = (win) => (win === 'tie' ? 'tie' : NAMES[win]);
    const rawName = nameOf(raw.win);
    const fairName = nameOf(fair.win);
    page.setReadout(
      `raw scores ${raw.scores[0]} and ${raw.scores[1]}, mass ${raw.mass[0]} and ${raw.mass[1]}, winner ${rawName}. `
      + `QK-norm scores ${fair.scores[0]} and ${fair.scores[1]}, mass ${fair.mass[0]} and ${fair.mass[1]}, winner ${fairName}.`,
    );
    page.probe = { raw, fair };
  },
});
