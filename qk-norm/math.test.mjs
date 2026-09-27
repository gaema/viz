import { massLabel } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function identity(row) {
  const nums = row.scores.map(Number);
  const top = Math.max(...nums);
  const exps = nums.map((n) => Math.exp(n - top));
  const z = exps.reduce((a, b) => a + b, 0);
  row.mass.forEach((m, i) => {
    ok(m === (exps[i] / z).toFixed(3), 'mass ' + i + ' is the softmax of the printed scores');
  });
  let win = 0;
  for (let i = 1; i < row.mass.length; i++) {
    if (Number(row.mass[i]) > Number(row.mass[win])) win = i;
  }
  ok(row.win === win, 'the winner is the larger printed mass');
}

const loud = massLabel(3, false);
const fair = massLabel(3, true);
identity(loud);
identity(fair);
ok(loud.scores[0] === '3.000' && loud.scores[1] === '1.000', 'raw scores are the bare dots');
ok(loud.win === 0, 'without the norm the long key wins');
ok(fair.win === 1, 'with the norm the aligned key wins');
ok(loud.win !== fair.win, 'the same keys pick different winners');

const quiet = massLabel(0.5, false);
identity(quiet);
ok(quiet.win === 1, 'a short off-axis key does not win even before the norm');
ok(massLabel(0.5, true).win === 1, 'the norm keeps the aligned key ahead when the other key is short');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS qk-norm');
