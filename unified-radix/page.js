// One matched prefix feeds three stores. Full-attention KV keeps every
// matched token. Sliding-window KV keeps the tail that fits the window.
// A recurrent layer keeps one checkpoint, and only when the match exists.
// Turning that layer off leaves the checkpoint at zero.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { unifiedReuse } from './math.js';

const TAPE = 24;

mount({
  mount: 'body',
  title: 'unified radix — one match, three kinds of state',
  blurb: 'A single prefix match reuses full-attention KV for every matched token and sliding-window KV for only the tail that fits the window. A recurrent layer adds one checkpoint when the match is non-empty, and turning that layer off leaves the checkpoint at zero.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('match', { label: 'matched prefix tokens', min: 0, max: 24, step: 1, value: 12 });
    c.stepper('window', { label: 'sliding window', min: 1, max: 16, step: 1, value: 4 });
    c.toggle('recurrent', { label: 'recurrent layer', value: true });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const u = unifiedReuse(page.state.match, page.state.window, !!page.state.recurrent);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);

    const pad = W * 0.04;
    const labelW = Math.min(168, W * 0.24);
    const tapeX = pad + labelW;
    const tapeW = W - tapeX - pad;
    const cw = tapeW / TAPE;
    const lanes = [
      {
        name: `full KV ${u.full}`,
        y: H * 0.08,
        fill: (i) => (i < u.full ? T.teal : T.n6),
      },
      {
        name: `window KV ${u.windowed}`,
        y: H * 0.36,
        fill: (i) => (i >= u.full - u.windowed && i < u.full ? T.accent : T.n6),
      },
    ];
    const laneH = H * 0.2;
    ctx.font = `${Math.max(12, H * 0.034)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (const lane of lanes) {
      ctx.fillStyle = T.n12;
      ctx.fillText(lane.name, pad, lane.y + laneH / 2);
      for (let i = 0; i < TAPE; i++) {
        ctx.fillStyle = lane.fill(i);
        ctx.fillRect(tapeX + i * cw + 1, lane.y, Math.max(1, cw - 2), laneH);
      }
    }

    const y = H * 0.68;
    ctx.fillStyle = T.n12;
    ctx.fillText(`checkpoint ${u.state}`, pad, y + laneH / 2);
    ctx.fillStyle = T.n6;
    ctx.fillRect(tapeX, y, tapeW, laneH);
    if (u.state === 1) {
      const at = Math.min(TAPE - 1, Math.max(0, u.full - 1));
      ctx.fillStyle = T.gold;
      ctx.fillRect(tapeX + at * cw + 1, y, Math.max(1, cw - 2), laneH);
    }

    page.setReadout(
      `match ${u.match}, window ${u.window}: full KV ${u.full} tokens, `
      + `window KV ${u.windowed} tokens, recurrent checkpoints ${u.state}.`,
    );
    page.probe = u;
  },
});
