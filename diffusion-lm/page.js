// Masked cells are open. A step fills the highest-scoring masked cell from the whole row.
// The next step is drawn from the new mask alone.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { diffusionAt, maskSentence, DEMO_TOKENS } from './math.js';

mount({
  mount: 'body',
  title: 'diffusion language model — unmask the whole sequence, both directions',
  blurb: 'Each step scores every still-masked position from the whole sequence, including positions that come later, and unmasks the highest score. The next step receives the updated tokens and the updated mask, and it receives no causal cache. While any position stays masked, the masked count falls by one. An empty mask stays empty.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('steps', { label: 'steps already taken', min: 0, max: 8, value: 0 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const at = diffusionAt(page.state.steps);
    const pending = at.pending;
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const n = DEMO_TOKENS.length;
    const pad = W * 0.08;
    const gap = 12;
    const cw = (W - pad * 2 - gap * (n - 1)) / n;
    const y = H * 0.32;
    const h = H * 0.28;
    ctx.font = `${Math.max(12, H * 0.04)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      const x = pad + i * (cw + gap);
      const open = at.mask[i];
      ctx.fillStyle = open ? T.n3 : T.teal;
      ctx.fillRect(x, y, cw, h);
      ctx.fillStyle = T.n12;
      ctx.fillText(open ? 'masked' : String(at.tokens[i]), x + cw / 2, y + h / 2);
      ctx.fillText(String(i), x + cw / 2, y + h + 22);
    }
    ctx.textAlign = 'left';
    ctx.fillText('pending score at 0 includes every position', pad, H * 0.12);
    const score = pending.masked ? pending.scores[pending.mask.indexOf(true)] : 0;
    page.setReadout(`next step ${maskSentence(pending)}. position 0 score ${score}. no causal cache is passed on.`);
    page.probe = { pending, mask: at.mask };
  },
});
