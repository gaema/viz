import { readFileSync } from 'node:fs';
import { seededRandn } from '../framework/tensor.js';
import { compressionLabel, qGroup, reconLevels, weightTip } from './quant.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const vals = [-1.25, -0.4, 0.1, 0.8, 2.2];
const Q = qGroup(vals, 4);
ok(Math.abs(-Q.lo / Q.s - Q.z) > 1e-6, '-min/s is not an integer, so the zero-point rounding matters');
let gap = 0;
for (let i = 0; i < vals.length; i++) gap = Math.max(gap, Math.abs(Q.deq[i] - (Q.q[i] - Q.z) * Q.s));
ok(gap === 0, 'displayed dequant matches (q - z) * s');
const old = Q.lo + Q.s * Q.q[0];
ok(Math.abs(old - Q.deq[0]) > 1e-6, 'min + s*q is a different reconstruction on this group');
const lines = reconLevels(Q);
ok(lines.length === Q.levels + 1, 'one level line per code');
ok(lines.every((lv, k) => Math.abs(lv - (Q.lo + Q.s * k)) > 1e-6), 'level lines differ from min+s*k on this group');
ok(lines[0] === (0 - Q.z) * Q.s, 'code 0 sits at (0-z)*s');
ok(Q.deq.every((d) => lines.includes(d)), 'every dequantized value sits on a level line');
for (const bits of [2, 3, 4, 5]) {
  const eff = bits + 24 / 64;
  const shown = eff.toFixed(2);
  const trueRounded = (16 / eff).toFixed(2);
  const fromShown = (16 / Number(shown)).toFixed(2);
  const line = compressionLabel(bits, 64);
  ok(line.text === `16 / ${shown} = ${fromShown}×`, 'compression label ' + line.text);
  if (trueRounded !== fromShown) ok(!line.text.includes(`16 / ${shown} = ${trueRounded}×`), 'old compression string at ' + bits + ' bits');
}

const VMAX = 2.5;
const x = seededRandn(0, 64, { std: 1 });
for (let i = 0; i < x.length; i++) x[i] = Math.max(-VMAX, Math.min(VMAX, x[i] * 0.5));
let oldTip = null;
const bits = 2, G = 8, levels = (1 << bits) - 1;
for (let g0 = 0; g0 < x.length && !oldTip; g0 += G) {
  const grp = Array.from(x.slice(g0, g0 + G));
  const Q = qGroup(grp, bits);
  for (let i = 0; i < grp.length; i++) {
    const tip = weightTip(grp[i], Q, i, levels);
    const fromPrinted = ((tip.q - tip.z) * Number(tip.s)).toFixed(3);
    const trueShown = Q.deq[i].toFixed(3);
    ok(tip.xp === fromPrinted && tip.err === (Number(tip.x) - Number(tip.xp)).toFixed(4), 'hover reconstruction divides the printed scale (' + tip.text.replace(/\n/g, ' | ') + ')');
    if (fromPrinted !== trueShown && !oldTip) oldTip = { tip, trueShown };
  }
}
ok(oldTip && oldTip.tip.xp !== oldTip.trueShown && !oldTip.tip.text.includes('= ' + oldTip.trueShown), 'rejects x′ = ' + (oldTip ? oldTip.trueShown : '?') + ' when the printed scale gives ' + (oldTip ? oldTip.tip.xp : '?'));
const pageSrc = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(pageSrc.includes('weightTip('), 'the hover calls weightTip');
ok(!pageSrc.includes('Q.deq[li].toFixed'), 'the hover does not round the true reconstruction on its own');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS quant');
