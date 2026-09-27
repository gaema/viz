import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { fwht, l2, maxAbs, activationOnlyDelta } from './math.js';

mount({
  mount: 'body',
  title: 'rotation before quant — spread the spike',
  blurb: 'A Walsh–Hadamard rotation keeps the length of a vector and shrinks its largest entry, which is what makes a low-bit quantizer able to spend its range on typical values. Rotating only the activation, and not the weight, changes the product.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('spike', { label: 'spike magnitude', min: 1, max: 16, step: 1, value: 8 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const a = page.state.spike | 0;
    const x = [a, 0, 0, 0];
    const y = fwht(x);
    const W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const base = page.H * 0.8;
    const bw = page.W * 0.08;
    const scale = (page.H * 0.5) / a;
    x.forEach((v, i) => {
      ctx.fillStyle = T.teal;
      ctx.fillRect(page.W * 0.12 + i * (bw + 10), base - Math.abs(v) * scale, bw, Math.abs(v) * scale);
    });
    y.forEach((v, i) => {
      ctx.fillStyle = T.accent;
      ctx.fillRect(page.W * 0.55 + i * (bw + 10), base - Math.abs(v) * scale, bw, Math.abs(v) * scale);
    });
    page.setReadout(
      `L2 before ${l2(x).toFixed(4)}, after ${l2(y).toFixed(4)}. max ${maxAbs(x)} → ${maxAbs(y)}. `
      + `Rotating only the activation moves the product by ${activationOnlyDelta(W, x).toFixed(3)}.`,
    );
  },
});
