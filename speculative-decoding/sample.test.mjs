import { readFileSync } from 'node:fs';
import { acceptCaption, distributionAt, buildRun, netLabel, ratioLabel } from './sample.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const empty = distributionAt(1, 0, 1, []);
const kept = distributionAt(1, 0, 1, [2]);
ok(empty.some((v, i) => v !== kept[i]), 'the next distribution changes when a kept token changes');

const run = buildRun({ k: 4, rounds: 1, quality: 1, seed: 1 });
const round = run.rounds[0];
ok(round.slots.every((s) => s.acc), 'agreement 1 accepts the whole draft');
const expect = distributionAt(1, 0, 1, [round.slots[0].x]);
ok(round.slots[1].p.every((v, i) => v === expect[i]), 'slot 1 is conditioned on the token kept at slot 0');
const accepted = run.rounds.reduce((s, x) => s + x.nAcc, 0);
ok(run.tokens === accepted + run.R, 'each round adds one correction or bonus (' + run.tokens + ' = ' + accepted + ' + ' + run.R + ')');
const low = buildRun({ k: 4, rounds: 2, quality: 0, seed: 3 });
const keptLow = low.rounds.reduce((s, rd) => s + rd.nAcc, 0);
const tailOnly = low.rounds.reduce((s, rd) => s + (rd.firstRej < 0 ? 0 : low.K - rd.firstRej - 1), 0);
const rejected = low.rounds.filter((rd) => rd.firstRej >= 0).length;
ok(low.wasted === low.K * low.R - keptLow, 'discarded drafts are every proposal that was not accepted');
ok(low.wasted === tailOnly + rejected, 'wasted drafts are the rejected proposal plus the untested tail (' + low.wasted + ' = ' + tailOnly + ' + ' + rejected + ')');

const spec = buildRun({ k: 4, rounds: 6, quality: 0.7, seed: 7 });
let split2 = null;
let split3 = null;
for (const rd of spec.rounds) {
  for (const sl of rd.slots) {
    for (const digits of [2, 3]) {
      const shown = ratioLabel(sl.p[sl.x], sl.q[sl.x], digits);
      const fromPrinted = (Number(shown.p) / Number(shown.q)).toFixed(digits);
      const fromTrue = (sl.p[sl.x] / sl.q[sl.x]).toFixed(digits);
      ok(shown.ratio === fromPrinted && shown.text === 'p/q = ' + fromPrinted, 'printed p/q divides the printed probabilities (' + shown.p + '/' + shown.q + ' = ' + shown.ratio + ')');
      if (shown.ratio !== fromTrue) {
        ok(shown.text !== 'p/q = ' + fromTrue, 'rejects rounding the true ratio to p/q = ' + fromTrue);
        if (digits === 2 && !split2) split2 = { shown, fromTrue };
        if (digits === 3 && !split3) split3 = { shown, fromTrue, sl };
      }
    }
  }
}
ok(split2 && split2.shown.p === '0.31' && split2.shown.q === '0.34', 'default seed prints p 0.31 and q 0.34');
ok(split2 && split2.fromTrue === '0.90' && split2.shown.ratio === '0.91', 'rejects p/q = 0.90; 0.31/0.34 prints 0.91');
let flipped = null;
for (const rd of spec.rounds) {
  for (const sl of rd.slots) {
    const shown = ratioLabel(sl.p[sl.x], sl.q[sl.x], 2);
    const gate = Math.min(1, Number(shown.ratio));
    const printedLe = Number(sl.u.toFixed(2)) <= gate;
    const trueLe = sl.u <= Math.min(1, sl.ratio);
    if (printedLe !== trueLe && !flipped) flipped = { sl, shown, cap: acceptCaption(sl.u, sl.p[sl.x], sl.q[sl.x], sl.ratio, 2) };
  }
}
ok(flipped && flipped.shown.text === 'p/q = 0.67' && flipped.cap === 'draw 0.60 > 0.57', 'rejects draw 0.60 > min(1, p/q) beside p/q = 0.67 (' + (flipped ? flipped.cap : 'missing') + ')');
ok(flipped && !flipped.cap.includes('min(1, p/q)'), 'a flipped comparison does not call the printed p/q the threshold');
ok(split3 && split3.shown.text !== 'p/q = ' + split3.fromTrue, 'the 3-decimal tip rejects p/q = ' + (split3 ? split3.fromTrue : '?'));
const cap = acceptCaption(split3.sl.u, split3.sl.p[split3.sl.x], split3.sl.q[split3.sl.x], split3.sl.ratio, 3);
ok(cap.includes('min(1, p/q) = ' + split3.shown.ratio) && !cap.includes(split3.fromTrue), 'the tip threshold is the printed p/q (' + cap + '), not ' + split3.fromTrue);
const pageSrc = readFileSync(new URL('./page.js', import.meta.url), 'utf8');
ok(pageSrc.includes('ratioLabel(') && pageSrc.includes('acceptCaption('), 'the page calls the printed-ratio functions');
ok(!pageSrc.includes('.ratio.toFixed'), 'the page does not round the true ratio into a label');

let zeroQ = null;
for (let seed = 0; seed < 80 && !zeroQ; seed++) {
  const run = buildRun({ k: 8, rounds: 3, quality: 0.05, seed });
  for (const rd of run.rounds) for (const sl of rd.slots) {
    const shown = ratioLabel(sl.p[sl.x], sl.q[sl.x], 2);
    if (shown.q === '0.00') zeroQ = { shown, trueRatio: (sl.p[sl.x] / sl.q[sl.x]).toFixed(2) };
  }
}
ok(zeroQ && zeroQ.shown.ratio !== zeroQ.trueRatio && (zeroQ.shown.ratio === '∞' || zeroQ.shown.ratio === '—'), 'a printed q of 0.00 is not the true ratio rounded (' + (zeroQ ? zeroQ.shown.text : 'missing') + ')');

const econ = buildRun({ k: 1, rounds: 3, quality: 0, seed: 0 });
const priced = netLabel(econ.tokens / econ.R, econ.K, 0.02);
const denom = 1 + econ.K * Number(priced.cost);
const fromPrinted = (100 * Number(priced.rate) / denom).toFixed(0);
const fromTrue = (100 * (econ.tokens / econ.R) / (1 + econ.K * 0.02)).toFixed(0);
ok(priced.pct === fromPrinted && priced.gauge.includes(priced.pct + '%') && priced.charge.includes(priced.cost) && priced.charge.includes(priced.pct + '%'), 'the net percent divides the printed rate and cost (' + priced.rate + ' / (1+' + econ.K + '·' + priced.cost + ') = ' + priced.pct + '%)');
ok(fromPrinted !== fromTrue && !priced.charge.includes(fromTrue + '%') && !priced.gauge.includes(fromTrue + '%'), 'rejects ' + priced.rate + ' with cost ' + priced.cost + ' = ' + fromTrue + '%');
ok(pageSrc.includes('netLabel('), 'the gauge and the readout call netLabel');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS sample');
