import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { scaledFate, masterBytes, fp8Bytes } from './math.js';

const TINY = 1e-4;
const PARAMS = 1000;

mount({
  mount: 'body',
  title: 'training numerics — loss scale and master weights',
  blurb: 'FP8 E4M3 cannot hold a tiny gradient or a huge one. The loss scale lifts the gradient into range. Too small and it flushes. Too large and it overflows. The optimizer still keeps a fp32 master copy, which is the memory price.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('scale', { label: 'loss scale', min: 1, max: 2000, step: 1, value: 1 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const scale = page.state.scale | 0;
    const tiny = scaledFate(TINY, scale);
    const unit = scaledFate(1, scale);
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const tone = (fate) => (fate === 'kept' ? T.accent : T.teal);
    ctx.fillStyle = tone(tiny);
    ctx.fillRect(page.W * 0.12, page.H * 0.2, page.W * 0.32, page.H * 0.5);
    ctx.fillStyle = tone(unit);
    ctx.fillRect(page.W * 0.55, page.H * 0.2, page.W * 0.32, page.H * 0.5);
    page.setReadout(
      `scale ${scale}: gradient ${TINY} is ${tiny}, gradient 1 is ${unit}. `
      + `master weights ${masterBytes(PARAMS)} bytes versus fp8 weights ${fp8Bytes(PARAMS)} bytes.`,
    );
  },
});
