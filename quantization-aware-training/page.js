// The forward bar is the high-precision weight after round-then-scale.
// The step under it moves that high-precision weight by the full printed step.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { qatForward, qatSentence, steStep, steSentence, steNextSentence } from './math.js';

mount({
  mount: 'body',
  title: 'quantization-aware training — round in the forward, step the full weight',
  blurb: 'The forward value is the high-precision weight rounded onto a grid and scaled back. The backward step moves that high-precision weight by the full step, as if the rounding had slope one. Inside a flat bin the forward code stays put until the weight crosses a boundary. A scale of zero sends the forward value to zero and the step still applies to the stored weight.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.slider('weight', { label: 'weight', min: -2, max: 2, step: 0.05, value: 1.2 });
    c.slider('scale', { label: 'scale', min: 0, max: 1, step: 0.1, value: 0.5 });
    c.stepper('bits', { label: 'bits', min: 2, max: 8, value: 8 });
    c.slider('grad', { label: 'gradient', min: 0, max: 2, step: 0.25, value: 1 });
    c.slider('lr', { label: 'step size', min: 0, max: 0.2, step: 0.01, value: 0.05 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const st = page.state;
    const fwd = qatForward(st.weight, st.scale, st.bits);
    const step = steStep(st.weight, st.grad, st.lr);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const mid = H * 0.42;
    const room = H * 0.28;
    const bars = [
      { label: 'weight', value: Number(fwd.weight), color: T.n6 },
      { label: 'forward', value: Number(fwd.forward), color: T.teal },
      { label: 'next weight', value: Number(step.next), color: T.accent },
    ];
    const pad = W * 0.08;
    const gap = 18;
    const cw = (W - pad * 2 - gap * (bars.length - 1)) / bars.length;
    ctx.font = `${Math.max(12, H * 0.04)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = T.n9;
    ctx.fillRect(pad, mid, W - pad * 2, 1);
    bars.forEach((b, i) => {
      const x = pad + i * (cw + gap);
      const h = room * Math.min(1, Math.abs(b.value) / 2);
      ctx.fillStyle = b.color;
      ctx.fillRect(x + cw * 0.25, mid - (b.value >= 0 ? h : 0), cw * 0.5, h);
      ctx.fillStyle = T.n12;
      ctx.fillText(b.label, x + cw * 0.5, mid + room + 16);
      ctx.fillText(b.value.toFixed(2), x + cw * 0.5, mid + room + 36);
    });
    ctx.textAlign = 'left';
    ctx.fillText('code ' + fwd.code, pad, H * 0.08);
    page.setReadout(`${qatSentence(fwd)}. ${steSentence(step)}. ${steNextSentence(step)}.`);
    page.probe = { fwd, step };
  },
});
