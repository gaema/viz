// Each weight rounds to -1, 0, or +1. The drawn reconstruction is that code
// times the printed scale. The dots are the sums of the printed numbers.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { ternaryBill } from './math.js';

mount({
  mount: 'body',
  title: 'ternary weights — a multiply becomes an add',
  blurb: 'Each weight is rounded to −1, 0, or +1 and stored with one scale. Small weights become zero. Large ones clip to a sign. The reconstruction is the code times the scale.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.slider('scale', { label: 'scale', min: 0.1, max: 1, step: 0.1, value: 0.5 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const row = ternaryBill(page.state.scale);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const n = row.count;
    const pad = W * 0.06;
    const gap = 8;
    const cw = (W - pad * 2 - gap * (n - 1)) / n;
    const mid = H * 0.38;
    const room = H * 0.26;
    ctx.font = `${Math.max(12, H * 0.035)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = T.n9;
    ctx.fillRect(pad, mid, W - pad * 2, 1);
    row.weights.forEach((w, i) => {
      const x = pad + i * (cw + gap);
      const wv = Number(w);
      const rv = Number(row.recon[i]);
      const wh = room * Math.min(1, Math.abs(wv));
      const rh = room * Math.min(1, Math.abs(rv));
      ctx.fillStyle = T.n6;
      ctx.fillRect(x, mid - (wv >= 0 ? wh : 0), cw * 0.42, wh);
      ctx.fillStyle = row.codes[i] === 0 ? T.n3 : (row.codes[i] > 0 ? T.teal : T.accent);
      ctx.fillRect(x + cw * 0.5, mid - (rv >= 0 ? rh : 0), cw * 0.42, rh);
      const labelY = mid + room + Math.max(14, H * 0.05);
      ctx.fillStyle = T.n12;
      ctx.fillText(w, x + cw * 0.2, labelY);
      ctx.fillText(String(row.codes[i]), x + cw * 0.7, labelY + Math.max(14, H * 0.045));
    });
    ctx.fillStyle = T.n12;
    ctx.textAlign = 'left';
    ctx.fillText('weight', pad, H * 0.08);
    ctx.fillText('code × scale', pad + 90, H * 0.08);
    page.setReadout(
      `scale ${row.scale}: ${row.nonzero} of ${row.count} weights survive. `
      + `dot ${row.full} becomes ${row.tern}. absolute error ${row.abs}.`,
    );
    page.probe = row;
  },
});
