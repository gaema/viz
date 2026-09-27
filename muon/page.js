import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { appliesTo, newtonSchulz, singularValues } from './math.js';

const G = [[2, 0], [0, 0.5]];

mount({
  mount: 'body',
  title: 'muon — orthogonalise the matrix update',
  blurb: 'Muon runs a matrix gradient through Newton–Schulz so its singular values move toward 1. The iteration only applies to 2-D weights. Embeddings, the output head, and per-channel gains stay on a second optimizer.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('steps', { label: 'Newton–Schulz steps', min: 0, max: 8, step: 1, value: 6 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const steps = page.state.steps | 0;
    const before = singularValues(G);
    const after = singularValues(newtonSchulz(G, steps));
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const base = page.H * 0.78;
    const bw = page.W * 0.16;
    const scale = page.H * 0.28;
    before.forEach((v, i) => {
      ctx.fillStyle = T.teal;
      ctx.fillRect(page.W * 0.18 + i * (bw + 16), base - v * scale, bw, v * scale);
    });
    after.forEach((v, i) => {
      ctx.fillStyle = T.accent;
      ctx.fillRect(page.W * 0.55 + i * (bw + 16), base - Math.min(v, 2) * scale, bw, Math.min(v, 2) * scale);
    });
    page.setReadout(
      `singular values ${before.map((v) => v.toFixed(3)).join(', ')} → ${after.map((v) => v.toFixed(3)).join(', ')} after ${steps} steps. `
      + `embeddings take Muon: ${appliesTo('embedding')}. matrices take Muon: ${appliesTo('matrix')}.`,
    );
  },
});
