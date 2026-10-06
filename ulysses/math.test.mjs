import { ulyssesPlan, ulyssesSentence } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function pack(heads, seq, seed) {
  let x = seed | 0;
  const next = () => { x = (Math.imul(x, 1103515245) + 12345) & 0x7fffffff; return (x % 1000) / 1000; };
  const col = () => Array.from({ length: seq }, next);
  const q = [], k = [], v = [];
  for (let h = 0; h < heads; h++) { q.push(col()); k.push(col()); v.push(col()); }
  return { q, k, v };
}

function identity(row) {
  if (!row.ok) {
    ok(row.rem === String(Number(row.heads) % Number(row.devices)), 'remainder');
    ok(ulyssesSentence(row) === `${row.heads} % ${row.devices} = ${row.rem}`, 'refuse sentence');
    return;
  }
  ok(row.per === String(Number(row.heads) / Number(row.devices)), 'per device');
  ok(ulyssesSentence(row) === `${row.heads} / ${row.devices} = ${row.per}`, 'share sentence');
}

const plan = ulyssesPlan(4, 2, pack(4, 4, 7));
identity(plan);
ok(plan.ok, '4 heads on 2 devices is accepted');
const per = Number(plan.per);
for (let d = 0; d < 2; d++) {
  for (let i = 0; i < per; i++) {
    const h = d * per + i;
    ok(plan.local[d][i].every((y, t) => y === plan.ref[h][t]), `device ${d} head ${h} matches the single-device head`);
  }
}
for (let d = 0; d < plan.ranges.length; d++) {
  const [lo, hi] = plan.ranges[d];
  ok(plan.restored[d].length === hi - lo, 'sequence shard length');
  for (let p = 0; p < hi - lo; p++) {
    ok(plan.restored[d][p].length === 4, 'the shard carries every head');
    for (let h = 0; h < 4; h++) ok(plan.restored[d][p][h] === plan.ref[h][lo + p], `restored d${d} p${p} h${h}`);
  }
}

const refused = ulyssesPlan(3, 2, pack(3, 4, 1));
identity(refused);
ok(!refused.ok && refused.rem === '1', '3 heads on 2 devices is refused');

identity(ulyssesPlan(8, 4, pack(8, 3, 3)));
identity(ulyssesPlan(5, 3, pack(5, 2, 2)));
identity(ulyssesPlan(1, 1, pack(1, 2, 9)));

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS ulysses');
