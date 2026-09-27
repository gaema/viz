// The draft reads a sink plus a recent window. Acceptance is 1 when the
// fact sits in that span, and the blind rate when it sits in the gap.
import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { acceptLabel } from './math.js';

mount({
  mount: 'body',
  title: 'windowed MTP — the draft does not read the whole context',
  blurb: 'Multi-token draft heads read a sink plus a recent window. The target still checks the full context, so a fact in the dropped middle is accepted only at the blind rate.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  controls: (c, page) => {
    const clampPos = (st) => {
      const max = Math.max(0, (st.length | 0) - 1);
      if ((st.pos | 0) > max) page.controls.set('pos', max);
    };
    c.stepper('length', {
      label: 'context length', min: 8, max: 48, step: 1, value: 24,
      onInput: (_v, st) => clampPos(st),
    });
    c.stepper('sink', { label: 'sink tokens', min: 0, max: 8, step: 1, value: 2 });
    c.stepper('window', { label: 'recent window', min: 1, max: 16, step: 1, value: 4 });
    c.stepper('pos', {
      label: 'where the answer looks', min: 0, max: 47, step: 1, value: 12,
      onInput: (_v, st) => clampPos(st),
    });
    c.slider('blind', { label: 'blind acceptance', min: 0, max: 1, step: 0.05, value: 0.25 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const a = acceptLabel(page.state.pos, page.state.length, page.state.sink, page.state.window, page.state.blind);
    if ((page.state.pos | 0) !== a.pos) page.controls.set('pos', a.pos, { silent: true });
    if ((page.state.sink | 0) !== a.sink) page.controls.set('sink', a.sink, { silent: true });
    if ((page.state.window | 0) !== a.window) page.controls.set('window', a.window, { silent: true });
    const W = page.W;
    const H = page.H;
    ctx.fillStyle = T.n0;
    ctx.fillRect(0, 0, W, H);
    const pad = W * 0.05;
    const tapeW = W - pad * 2;
    const cw = tapeW / a.length;
    const y = H * 0.16;
    const h = H * 0.28;
    for (let i = 0; i < a.length; i++) {
      const inSink = i < a.sink;
      const inWindow = i >= a.length - a.window;
      ctx.fillStyle = inSink || inWindow ? T.teal : T.n6;
      ctx.fillRect(pad + i * cw + 1, y, Math.max(1, cw - 2), h);
    }
    const mark = pad + a.pos * cw;
    ctx.fillStyle = T.gold;
    ctx.fillRect(mark + 1, y - H * 0.045, Math.max(2, cw - 2), H * 0.03);
    ctx.font = `${Math.max(12, H * 0.04)}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = T.n12;
    ctx.textAlign = 'left';
    ctx.fillText('draft sees the teal tokens', pad, H * 0.08);
    const bars = [
      ['full draft', a.length, T.accent],
      ['windowed draft', a.visible, T.teal],
    ];
    const max = a.length;
    const barY = H * 0.58;
    const barH = H * 0.12;
    const gap = H * 0.06;
    bars.forEach((item, i) => {
      const yy = barY + i * (barH + gap);
      ctx.fillStyle = T.n3;
      ctx.fillRect(pad, yy, tapeW, barH);
      ctx.fillStyle = item[2];
      ctx.fillRect(pad, yy, tapeW * (item[1] / max), barH);
      ctx.fillStyle = T.n12;
      ctx.fillText(`${item[0]} ${item[1]}`, pad + 8, yy + barH / 2);
    });
    page.setReadout(
      `length ${a.length}, sink ${a.sink}, window ${a.window}: draft reads ${a.visible} of ${a.length}. `
      + `fact at ${a.pos} is ${a.seen ? 'visible' : 'dropped'}. acceptance ${a.accept}.`,
    );
    page.probe = a;
  },
});
