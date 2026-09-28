// Every share-th layer writes keys and values. The layers until the next
// writer reuse that cache. The readout is writers plus readers, and the cache.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { kvShare, shareSentence, writesAt } from './math.js';

mount({
  mount: 'body',
  title: 'cross-layer KV — later layers reuse an earlier cache',
  blurb: 'Every share-th layer writes keys and values, and the tail still gets a writer. The layers until the next writer reuse that cache and compute a new query. A share of 1 means every layer writes its own cache. The cache count is the number of writers.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('layers', { label: 'layers', min: 4, max: 16, step: 1, value: 8 });
    c.stepper('share', { label: 'layers per cache', min: 1, max: 4, step: 1, value: 2 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const row = kvShare(page.state.layers, page.state.share);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const pad = W * 0.06;
    const labelW = W * 0.16;
    const rowH = (H * 0.84) / row.layers;
    const y0 = H * 0.08;
    ctx.font = `${Math.max(11, Math.min(16, rowH * 0.45))}px sans-serif`;
    ctx.textBaseline = 'middle';
    let writer = 0;
    for (let i = 0; i < row.layers; i++) {
      const y = y0 + i * rowH;
      const write = writesAt(i, row.share);
      if (write) writer = i;
      ctx.fillStyle = T.n12;
      ctx.textAlign = 'left';
      ctx.fillText(write ? `L${i} writes` : `L${i} reads L${writer}`, pad, y + rowH * 0.5);
      const x = pad + labelW;
      const bw = W - x - pad;
      ctx.fillStyle = write ? T.teal : T.n6;
      ctx.fillRect(x, y + rowH * 0.15, write ? bw : bw * 0.55, rowH * 0.7);
      if (!write) {
        ctx.fillStyle = T.accent;
        ctx.fillRect(x, y + rowH * 0.4, bw * 0.55, Math.max(2, rowH * 0.15));
      }
    }
    page.setReadout(shareSentence(row) + '.');
    page.probe = row;
  },
});
