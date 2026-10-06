// Four contiguous weights. The two largest magnitudes stay; the rest are drawn as zero.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { pruneGroup, magnitudeSentence, dropSentence } from './math.js';

mount({
  mount: 'body',
  title: 'structured sparsity — two nonzeros in every four',
  blurb: 'Every group of four neighbouring weights keeps at most two nonzeros, the two largest magnitudes, and a tie keeps the earlier index. A group that already has two or fewer nonzeros keeps those nonzeros. The readout adds the two kept magnitudes and subtracts the kept count from the nonzero count.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c) => {
    c.slider('third', { label: 'third weight', min: -1, max: 1, step: 0.1, value: -0.8 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const row = pruneGroup([0.9, 0.2, page.state.third, 0.1]);
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const n = 4;
    const pad = W * 0.08;
    const gap = 12;
    const cw = (W - pad * 2 - gap * (n - 1)) / n;
    const mid = H * 0.42;
    const room = H * 0.28;
    ctx.font = `${Math.max(12, H * 0.04)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = T.n9;
    ctx.fillRect(pad, mid, W - pad * 2, 1);
    row.printed.forEach((w, i) => {
      const x = pad + i * (cw + gap);
      const kept = row.keep.includes(i);
      const v = Number(row.out[i]);
      const h = room * Math.min(1, Math.abs(Number(w)));
      const kh = room * Math.min(1, Math.abs(v));
      ctx.fillStyle = T.n6;
      ctx.fillRect(x, mid - (Number(w) >= 0 ? h : 0), cw * 0.4, h);
      ctx.fillStyle = kept && v !== 0 ? T.teal : T.n3;
      ctx.fillRect(x + cw * 0.5, mid - (v >= 0 ? kh : 0), cw * 0.4, kh);
      ctx.fillStyle = T.n12;
      ctx.fillText(w, x + cw * 0.2, mid + room + 18);
      ctx.fillText(row.out[i], x + cw * 0.7, mid + room + 18);
    });
    ctx.textAlign = 'left';
    ctx.fillText('weight', pad, H * 0.08);
    ctx.fillText('kept', pad + W * 0.28, H * 0.08);
    page.setReadout(`${magnitudeSentence(row)}. ${dropSentence(row)}.`);
    page.probe = row;
  },
});
