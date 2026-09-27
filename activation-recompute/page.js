import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { recomputeCost } from './math.js';

const LAYERS = 16;

mount({
  mount: 'body',
  title: 'activation recompute — memory against a second forward',
  blurb: 'Storing every activation uses the most memory and no extra forward. Storing only the input uses one slot and recomputes the stack. A segment length in between is cheaper than either end.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('segment', { label: 'segment length', min: 1, max: LAYERS, step: 1, value: 4 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const s = page.state.segment | 0;
    const cur = recomputeCost(LAYERS, s);
    const ends = [recomputeCost(LAYERS, 1), recomputeCost(LAYERS, 4), recomputeCost(LAYERS, LAYERS)];
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const max = Math.max(...ends.map((e) => e.objective), cur.objective);
    const bw = page.W * 0.18;
    const base = page.H * 0.8;
    ends.forEach((e, i) => {
      const h = (e.objective / max) * page.H * 0.55;
      ctx.fillStyle = e.segment === s ? T.accent : T.teal;
      ctx.fillRect(page.W * 0.12 + i * (bw + 28), base - h, bw, h);
    });
    const mid = ends[1].objective;
    page.setReadout(
      `segment ${cur.segment}: memory ${cur.memory} stored boundaries, extra forward per layer ${cur.extraPerLayer}, objective ${cur.objective} = memory + extra. `
      + `store-all (segment 1) objective ${ends[0].objective}, memory ${ends[0].memory}. `
      + `segment 4 objective ${mid}, memory ${ends[1].memory}. `
      + `store-none (segment ${LAYERS}) objective ${ends[2].objective}, memory ${ends[2].memory}. `
      + `Segment 4's objective is below both ends: ${mid < ends[0].objective && mid < ends[2].objective}.`,
    );
  },
});
