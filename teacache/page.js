import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { drift, scheduleShare, skipped } from './math.js';

const RESIDUALS = [0.1, 0.4, 0.2, 0.8];

mount({
  mount: 'body',
  title: 'step cache — skip a small residual',
  blurb: 'A sampler step whose residual is under the threshold is skipped, and the drift is the residual that was not applied. A higher threshold skips more and the drift rises. One skipped step is a much larger share of a four-step schedule than of a forty-step one.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('threshold', { label: 'skip threshold', min: 0.05, max: 0.9, step: 0.05, value: 0.3 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const threshold = +page.state.threshold;
    const gone = drift(RESIDUALS, threshold);
    const n = skipped(RESIDUALS, threshold).length;
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const base = page.H * 0.8;
    const bw = page.W * 0.12;
    RESIDUALS.forEach((r, i) => {
      const skip = Math.abs(r) < threshold;
      ctx.fillStyle = skip ? T.teal : T.accent;
      ctx.fillRect(page.W * 0.12 + i * (bw + 18), base - r * page.H * 0.6, bw, r * page.H * 0.6);
    });
    const few = scheduleShare(1, 4);
    const many = scheduleShare(1, 40);
    page.setReadout(
      `threshold ${threshold.toFixed(2)}: skipped ${n}, drift ${gone.toFixed(3)}. `
      + `one skip is ${few} of a 4-step schedule and ${many} of a 40-step schedule.`,
    );
  },
});
