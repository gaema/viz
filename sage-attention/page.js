import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { attentionError } from './math.js';

const K = [[0.2, 0.5, -0.7], [1.1, -0.4, 0.2], [-0.6, 0.9, 0.3]];
const V = [[0.6, 0.1], [-0.3, 0.9], [0.4, -0.2]];

mount({
  mount: 'body',
  title: 'sage attention — low-precision scores',
  blurb: 'The score row is rounded to int8 against its own maximum, then softmax mixes the values. Exact attention uses the unrounded scores. The two outputs differ, and a row of zeros stays exact because it is already on the grid.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('amp', { label: 'query scale', min: 1, max: 8, step: 1, value: 2 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const amp = page.state.amp | 0;
    const Q = [[0.3 * amp, -1.7 * amp, 0.4 * amp]];
    const err = attentionError(Q, K, V);
    const zero = attentionError([[0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], V);
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const h = Math.min(page.H * 0.6, err * page.H * 4 + 40);
    ctx.fillStyle = T.accent;
    ctx.fillRect(page.W * 0.15, page.H * 0.75 - h, page.W * 0.28, h);
    ctx.fillStyle = T.teal;
    ctx.fillRect(page.W * 0.55, page.H * 0.75 - 24, page.W * 0.28, 24);
    page.setReadout(`int8 score error ${err.toExponential(3)} (greater than 0: ${err > 0}). all-zero keys error ${zero}.`);
  },
});
