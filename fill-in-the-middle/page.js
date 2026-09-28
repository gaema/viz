// Prefix, then suffix, then the middle. The middle can see the suffix.
// A second example under the same causal mask can see the first.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { fimLabel, fimSentence } from './math.js';

mount({
  mount: 'body',
  title: 'fill-in-the-middle — the middle can see the ending',
  blurb: 'The example is ordered prefix, then suffix, then the missing middle, so a causal mask lets the middle read the ending. Two examples under that same mask let the second read the first. Isolate them and those pairs drop to zero, while each middle token still sees its own suffix.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.stepper('prefix', { label: 'prefix tokens', min: 1, max: 4, step: 1, value: 3 });
    c.stepper('suffix', { label: 'suffix tokens', min: 1, max: 3, step: 1, value: 2 });
    c.stepper('middle', { label: 'middle tokens', min: 1, max: 3, step: 1, value: 2 });
    c.toggle('isolate', { label: 'isolate examples', value: false });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const row = fimLabel(page.state.prefix, page.state.suffix, page.state.middle, page.state.isolate);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const n = row.one * 2;
    const pad = W * 0.08;
    const top = H * 0.12;
    const side = Math.min(W - pad * 2, H - top - H * 0.08);
    const cw = side / n;
    const kind = (i) => {
      const local = i % row.one;
      if (local < row.prefix) return 0;
      if (local < row.prefix + row.suffix) return 1;
      return 2;
    };
    for (let q = 0; q < n; q++) {
      for (let k = 0; k < n; k++) {
        const x = pad + k * cw;
        const y = top + q * cw;
        let color = T.n3;
        if (k <= q) {
          const same = Math.floor(q / row.one) === Math.floor(k / row.one);
          if (!same && row.blocked) color = T.n6;
          else if (!same) color = T.accent;
          else if (kind(q) === 2 && kind(k) === 1 && Math.floor(q / row.one) === 0) color = T.gold;
          else color = T.teal;
        }
        ctx.fillStyle = color;
        ctx.fillRect(x + 1, y + 1, Math.max(1, cw - 2), Math.max(1, cw - 2));
      }
    }
    page.setReadout(fimSentence(row));
    page.probe = row;
  },
});
