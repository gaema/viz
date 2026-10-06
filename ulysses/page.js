// Each device is drawn with the heads it holds after the first exchange.
// A head count that does not divide by the device count is refused on the readout.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { ulyssesPlan, ulyssesSentence } from './math.js';

function pack(heads, seq, seed) {
  let x = seed | 0;
  const next = () => { x = (Math.imul(x, 1103515245) + 12345) & 0x7fffffff; return ((x % 1000) / 1000) - 0.5; };
  const col = () => Array.from({ length: seq }, next);
  const q = [], k = [], v = [];
  for (let h = 0; h < heads; h++) { q.push(col()); k.push(col()); v.push(col()); }
  return { q, k, v };
}

mount({
  mount: 'body',
  title: 'ulysses — all heads become a full sequence on fewer heads',
  blurb: 'One exchange gives each device the full sequence for an equal share of the heads. Attention on that share matches attention computed on one device for those same heads. A second exchange puts each device back on its sequence shard, now with every head. When the head count does not divide by the device count, the exchange is refused.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('heads', { label: 'heads', min: 1, max: 8, value: 4 });
    c.stepper('devices', { label: 'devices', min: 1, max: 4, value: 2 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const heads = page.state.heads | 0;
    const devices = page.state.devices | 0;
    const plan = ulyssesPlan(heads, devices, pack(Math.max(heads, 1), 4, 7));
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    ctx.font = `${Math.max(12, H * 0.04)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const n = Math.max(1, devices);
    const pad = W * 0.06;
    const gap = 10;
    const cw = (W - pad * 2 - gap * (n - 1)) / n;
    for (let d = 0; d < n; d++) {
      const x = pad + d * (cw + gap);
      ctx.fillStyle = plan.ok ? T.teal : T.n3;
      ctx.fillRect(x, H * 0.22, cw, H * 0.46);
      ctx.fillStyle = T.n12;
      ctx.fillText('device ' + d, x + cw / 2, H * 0.16);
      if (plan.ok) {
        const per = Number(plan.per);
        ctx.fillText(per + ' heads', x + cw / 2, H * 0.45);
        const [lo, hi] = plan.ranges[d];
        ctx.fillText('seq ' + lo + '-' + (hi - 1), x + cw / 2, H * 0.78);
      } else {
        ctx.fillText('refused', x + cw / 2, H * 0.45);
      }
    }
    page.setReadout(ulyssesSentence(plan) + (plan.ok ? '. attention on the gathered heads matches the single-device heads, and the second exchange restores the sequence shards.' : '. the head count does not divide by the device count.'));
    page.probe = plan;
  },
});
