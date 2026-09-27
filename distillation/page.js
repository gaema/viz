import { mount } from '../framework/layout.js';
import { T } from '../framework/theme.js';
import { copiesTeacherError, softMass } from './math.js';

const TEACHER = [4, 0];
const TRUTH = 1;

mount({
  mount: 'body',
  title: 'distillation — the student copies the teacher',
  blurb: 'A soft target is the teacher\'s distribution, warmed by a temperature. When the teacher\'s top class is wrong, that error is what the student is trained to match. Raising the temperature spreads the mass back toward the truth.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  controls: (c) => {
    c.slider('temp', { label: 'temperature', min: 1, max: 8, step: 1, value: 1 });
  },
  draw: (page) => {
    const ctx = page.ctx;
    const Tm = page.state.temp | 0;
    const wrong = softMass(TEACHER, 0, Tm);
    const truth = softMass(TEACHER, TRUTH, Tm);
    const copies = copiesTeacherError(TEACHER, TRUTH, Tm);
    ctx.fillStyle = T.n3;
    ctx.fillRect(0, 0, page.W, page.H);
    const base = page.H * 0.8;
    ctx.fillStyle = T.teal;
    ctx.fillRect(page.W * 0.18, base - wrong * page.H * 0.55, page.W * 0.22, wrong * page.H * 0.55);
    ctx.fillStyle = T.accent;
    ctx.fillRect(page.W * 0.55, base - truth * page.H * 0.55, page.W * 0.22, truth * page.H * 0.55);
    page.setReadout(
      `T=${Tm}: soft mass on the teacher's wrong class ${wrong.toFixed(3)}, on the truth ${truth.toFixed(3)}. `
      + `The student target copies the error: ${copies}.`,
    );
  },
});
