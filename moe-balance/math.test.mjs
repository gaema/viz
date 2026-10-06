import { readFileSync } from 'node:fs';
import { auxMix, biasRoute, moveSentence, compareCaptions } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const flipped = biasRoute([2, 0], [0, 3], 1, 0.5, 10);
ok(flipped.picked[0] === 1, 'bias flips top-1 onto the lower raw score');
ok(flipped.combine.length === 1 && Math.abs(flipped.combine[0].w - 1) < 1e-12, 'a single pick combines at 1');
ok(flipped.lam === 0, 'bias mode reports auxiliary weight 0');
const down = flipped.moves.find((m) => m.e === 1);
const up = flipped.moves.find((m) => m.e === 0);
ok(down && down.dir === 'down', 'the overloaded expert bias decreases');
ok(up && up.dir === 'up', 'the underloaded expert bias increases');
ok(down.after === (Number(down.before) - Number(down.step)).toFixed(2), 'down step identity');
ok(up.after === (Number(up.before) + Number(up.step)).toFixed(2), 'up step identity');
ok(moveSentence(down) === `${down.before} - ${down.step} = ${down.after}`, 'down sentence');
ok(moveSentence(up) === `${up.before} + ${up.step} = ${up.after}`, 'up sentence');

const both = biasRoute([2, 0], [0, 5], 2, 0.25, 4);
const z = 1 + Math.exp(-2);
ok(Math.abs(both.combine.find((c) => c.i === 0).w - 1 / z) < 1e-9, 'combine uses the raw score');
ok(Math.abs(both.combine.find((c) => c.i === 1).w - Math.exp(-2) / z) < 1e-9, 'the biased score is not the combine weight');

const skew = [0.7, 0.2, 0.1];
const loss0 = auxMix(skew, 0, null);
const loss1 = auxMix(skew, 1, null);
ok(loss0.lam === 0 && Math.abs(loss0.eff[0] / loss0.eff[1] - 0.7 / 0.2) < 1e-4, 'λ=0 keeps the skewed route');
ok(loss1.lam === 1 && Math.abs(loss1.eff[0] - 1 / 3) < 1e-5 && Math.abs(loss1.eff[2] - 1 / 3) < 1e-5, 'λ=1 is uniform');
ok(compareCaptions('loss').a.indexOf('λ=0') === 0, 'loss captions name λ');
ok(compareCaptions('bias').a.indexOf('bias route') === 0, 'bias captions do not claim a λ collapse');

const page = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(page.includes('auxMix('), 'page calls auxMix');
ok(page.includes('biasRun('), 'page calls biasRun');
ok(page.includes("key: 'lam'"), 'compare key stays λ');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS moe-balance');
