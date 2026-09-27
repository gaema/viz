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
  const a = Number(row.mass[0]);
  const b = Number(row.mass[1]);
  const win = a === b ? 'tie' : (a > b ? 0 : 1);
  ok(row.win === win, 'the winner is the larger printed mass, or a tie when they match');
}

const loud = massLabel(3, false);
const fair = massLabel(3, true);
identity(loud);
identity(fair);
ok(loud.scores[0] === '3.000' && loud.scores[1] === '1.000', 'raw scores are the bare dots');
ok(loud.win === 0, 'without the norm the long key wins');
ok(fair.win === 1, 'with the norm the aligned key wins');
const tied = massLabel(1, false);
identity(tied);
ok(tied.scores[0] === tied.scores[1] && tied.mass[0] === tied.mass[1], 'scale 1 prints equal raw scores and equal mass');
ok(tied.win === 'tie', 'equal printed mass is a tie, not a win for the first key');
ok(loud.win !== fair.win, 'the same keys pick different winners');

const quiet = massLabel(0.5, false);
identity(quiet);
ok(quiet.win === 1, 'a short off-axis key does not win even before the norm');
ok(massLabel(0.5, true).win === 1, 'the norm keeps the aligned key ahead when the other key is short');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS qk-norm');
