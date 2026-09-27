// Teacher forcing adds the one-step error again at every frame. Self-forcing
// stays at that one-step error. The tall bar is the frame number times the
// printed step.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { frameErrors } from './math.js';

mount({
  mount: 'body',
  title: 'self-forcing — a fast video model trained on its own frames',
  blurb: 'Teacher forcing adds one step of error at every frame, so a few-step video model drifts along the clip. Self-forcing trains on the model’s own frames, and every frame stays at the one-step error.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('frames', { label: 'frames', min: 1, max: 12, step: 1, value: 6 });
    c.slider('step', { label: 'one-step error', min: 0, max: 0.2, step: 0.01, value: 0.04 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const row = frameErrors(page.state.frames, page.state.step);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const bands = [
      { name: 'teacher forcing', values: row.teacher, color: T.accent, y: H * 0.1 },
      { name: 'self-forcing', values: row.self, color: T.teal, y: H * 0.52 },
    ];
    const pad = W * 0.05;
    const labelW = W * 0.24;
    const tapeW = W - pad - labelW - pad;
    const n = row.frames;
    const cw = tapeW / n;
    const top = Number(row.teacher[n - 1]);
    const room = H * 0.28;
    ctx.font = `${Math.max(12, H * 0.038)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    bands.forEach((band) => {
      ctx.fillStyle = T.n12;
      ctx.fillText(band.name, pad, band.y + room / 2);
      band.values.forEach((v, i) => {
        const h = top === 0 ? 0 : room * (Number(v) / top);
        const x = pad + labelW + i * cw;
        ctx.fillStyle = T.n3;
        ctx.fillRect(x + 2, band.y, Math.max(1, cw - 4), room);
        ctx.fillStyle = band.color;
        ctx.fillRect(x + 2, band.y + room - h, Math.max(1, cw - 4), h);
      });
    });
    const last = n;
    page.setReadout(
      `frame ${last}: teacher ${row.teacher[last - 1]} = ${last} × ${row.step}. self-forcing ${row.self[last - 1]}.`,
    );
    page.probe = row;
  },
});
