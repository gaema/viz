// KV crosses once, and grows with the context. Hidden states cross twice
// per layer, and grow with the batch. The printed totals are those products.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { shipCompare, shipLabel, shipSentence } from './math.js';

mount({
  mount: 'body',
  title: 'attention / expert split — a wire at every layer',
  blurb: 'Splitting attention from the experts ships the hidden state to the experts and back on every layer. Splitting prefill from decode ships the KV cache once. The larger one flips with context and batch.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('layers', { label: 'layers', min: 1, max: 8, step: 1, value: 4 });
    c.stepper('seq', { label: 'context length', min: 4, max: 64, step: 4, value: 32 });
    c.stepper('batch', { label: 'batch', min: 1, max: 16, step: 1, value: 4 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const row = shipLabel(page.state.layers, page.state.seq, page.state.batch);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const pad = W * 0.08;
    const gap = W * 0.08;
    const bw = (W - pad * 2 - gap) / 2;
    const base = H * 0.78;
    const room = H * 0.55;
    const max = Math.max(Number(row.pd), Number(row.afd), 1);
    const bars = [
      ['KV once', row.pd, T.accent],
      ['per step', row.afd, T.teal],
    ];
    ctx.font = `${Math.max(13, H * 0.04)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    bars.forEach((item, i) => {
      const x = pad + i * (bw + gap);
      const h = room * (Number(item[1]) / max);
      ctx.fillStyle = T.n3;
      ctx.fillRect(x, base - room, bw, room);
      ctx.fillStyle = item[2];
      ctx.fillRect(x, base - h, bw, h);
      ctx.fillStyle = T.n12;
      ctx.fillText(`${item[0]} ${item[1]}`, x + bw / 2, base + H * 0.06);
    });
    page.setReadout(
      `KV once is ${shipSentence(row).kv}. `
      + `Per step is ${shipSentence(row).hidden}. `
      + shipCompare(row),
    );
    page.probe = row;
  },
});
