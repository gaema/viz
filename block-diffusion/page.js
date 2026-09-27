import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { positionAccept, tailAccept } from './math.js';

mount({
  mount: 'body',
  title: 'block diffusion draft — a wider block, a weaker tail',
  blurb: 'A block drafter proposes several tokens in one parallel step. The first position is the one it is most sure of. Each later position is accepted less often, so the tail of a wider block is accepted less often than the tail of a short one.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('width', { label: 'draft block width', min: 1, max: 8, step: 1, value: 4 });
    c.slider('quality', { label: 'draft quality', min: 0.4, max: 0.95, step: 0.05, value: 0.8 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const width = page.state.width | 0;
    const quality = +page.state.quality;
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const bw = (page.W * 0.7) / width;
    const base = page.H * 0.8;
    for (let i = 0; i < width; i++) {
      const p = positionAccept(i, quality);
      ctx.fillStyle = i === width - 1 ? T.accent : T.teal;
      ctx.fillRect(page.W * 0.12 + i * (bw + 6), base - p * page.H * 0.55, bw, p * page.H * 0.55);
    }
    const tail = tailAccept(width, quality);
    const shorterW = Math.max(1, width - 1);
    const short = tailAccept(shorterW, quality);
    const widerLower = width > 1 && tail < short;
    page.setReadout(
      `width ${width}, quality ${quality.toFixed(2)}: tail acceptance ${tail.toFixed(4)}, which is quality^width. `
      + (width > 1
        ? `A block of width ${shorterW} has tail ${short.toFixed(4)}. This wider tail is lower: ${widerLower}.`
        : `Width 1 is only the first position, so there is no shorter block.`),
    );
  },
});
